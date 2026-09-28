"""EXPERIMENT C — buchwald_hartwig 4.3, Step 3 measurements (signal confirmed real).

Runs the three measurements that never ran:
 (a) SIGNAL SURVIVAL   — best rep (cgr_morgan) contributed-features-only vs full width,
                          paired per fold; plus slice-alone vs full-hybrid.
 (b) TRUNCATION DEPTH  — realised quota q for cgr_morgan vs its saturation curve.
 (c) GLOBAL RANKING    — gain-based global top-B vs best quota hybrid at matched budget,
                          with origin counts.

FIXED: importance=lightgbm_gain, internal_evaluator=lightgbm, downstream scorer=lightgbm
(for internal consistency of paired deltas; the rf target anchor 14.37 is kept from step1).
Full 5-fold GroupKFold-by-additive. Importance always computed on TRAIN rows only.
Calls hybrid.create_hybrid / apply_feature_selection DIRECTLY; find_best_hybrid() never called.
Writes JSON checkpoints so partial progress survives.

The pool of representations is buchwald_hartwig's staged ten, read from
complexity_helpers.DATASET_REPS — the same list c30, c31 and c32 use on this
dataset (user decision 2026-09-11). It used to be ten names typed into this
file from before the 2026-08-20 pruning; see _staged_pool().
"""
import os
import sys
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
from kirby import hybrid              # noqa: E402  read-only
from scipy.stats import spearmanr  # noqa: E402

# Checkpoint next to the other complexity results, not in a scratch directory.
# C43_OUT_DIR (dirty local run 2026-09-28) redirects both the checkpoint and
# the signal file the anchor is read from; unset -> tests/results/complexity.
_OUT_DIR = (os.environ.get("C43_OUT_DIR")
            or os.path.join(_REPO, "tests", "results", "complexity"))
CKPT = os.path.join(_OUT_DIR, "c43_step3_ckpt.json")
os.makedirs(os.path.dirname(CKPT), exist_ok=True)

DATASET = "buchwald_hartwig"


def _staged_pool(ds=DATASET):
    """This dataset's post-pruning representation list, read from the one place
    that holds it.

    complexity_helpers.DATASET_REPS is the list c30, c31 and c32 use on this
    dataset. It is also what _shared.load_data auto-filters to when a caller
    passes no representation names, so H.load_full returns exactly these
    matrices and no others.

    2026-09-11, user decision: keep this file to the staged ten. Its pool used
    to be ten names typed into this file. They were typed before the 2026-08-20
    pruning that cut buchwald_hartwig from 19 names to 10, so six of them were
    not in the staged set: drfp_2048, diff_ecfp4 and reactant_ecfp4 were cut by
    that pruning for correlating 0.94 to 0.98 with a sibling that was kept, and
    product_ecfp4, component_descriptors and ligand_descriptors were never on
    the staged list at all. Four overlapped: cgr_morgan, reactiont5, rxnfp and
    diff_maccs. Reading the list here puts this file's numbers on the same
    footing as c30, c31 and c32 on this dataset.
    """
    entry = DATASET_REPS.get(ds) or {}
    names = list(entry.get("general") or [])
    if not names:
        raise RuntimeError(
            f"{ds} has no 'general' representation list in "
            f"complexity_helpers.DATASET_REPS, so the staged pool cannot be "
            f"resolved. Nothing is guessed here.")
    return names


# The staged list, cut the same way load_data cuts it. Under a local run that
# sets KIRBY_LOCAL_REPS, H.load_full returns only the kept representations, so
# the uncut list made the "did not load" check below raise every time. Unset,
# cut_pool returns the list unchanged.
from experiments._local_grid import cut_pool  # noqa: E402
POOL = cut_pool(DATASET, _staged_pool())
BEST = "cgr_morgan"
_ANCHOR = None

def _lightgbm_anchor():
    """cgr_morgan alone, lightgbm, 5-fold — written by c43_buchwald_signal.

    Was a typed-in literal, 14.3664, which was a RANDOM FOREST number. This
    file fits lightgbm, so subtracting it compared two different models.
    Returns None when the signal file has not been run; the delta is then
    omitted rather than computed against the wrong model.
    """
    path = os.path.join(_OUT_DIR, "c43_buchwald_signal_buchwald_hartwig.json")
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except Exception:
        return None
    return (blob.get("target_by_model", {}).get("lightgbm", {}) or {}).get("rmse")

BUDGETS = [100, 1000]
# Two of the five in find_best_hybrid.DEFAULT_ALLOCATION_METHODS. Was
# ["top_5", "performance_weighted"]; top_5 left the default set on 2026-09-01
# because it welds a representation count into the weighting slot. power_2 is
# the surviving concentrator -- it weights by the square of each
# representation's own score instead of zeroing all but the best five.
METHODS = ["power_2", "performance_weighted"]
SAT_KS = [10, 50, 100, 200, 500, 1000]   # + full width (2048)


def lgbm_rmse(Xtr, ytr, Xva, yva):
    from sklearn.preprocessing import StandardScaler
    from kirby.find_best_hybrid import get_model
    sc = StandardScaler()
    Xtr = sc.fit_transform(H._clean(Xtr)); Xva = sc.transform(H._clean(Xva))
    m = get_model("lightgbm", task_type="regression", n_samples=len(ytr))
    m.fit(Xtr, ytr); p = m.predict(Xva)
    return float(np.sqrt(np.mean((yva - p) ** 2))), p


def main():
    t_start = time.time()
    global _ANCHOR
    _ANCHOR = _lightgbm_anchor()
    if _ANCHOR is None:
        print("  [anchor] c43_buchwald_signal has not been run — the "
              "hybrid-minus-target delta will be omitted.", flush=True)
    else:
        print(f"  [anchor] cgr_morgan alone, lightgbm: {_ANCHOR:.4f} RMSE",
              flush=True)
    reps, y, groups, mname, mfn, lower, task = H.load_full(DATASET)
    y = np.asarray(y, float); groups = np.asarray(groups)
    missing = [r for r in POOL if r not in reps]
    if missing:
        raise RuntimeError(
            f"{DATASET}: the staged pool names {len(POOL)} representations but "
            f"{len(missing)} did not load: {missing}")
    if BEST not in POOL:
        raise RuntimeError(
            f"{BEST} is not in {DATASET}'s staged pool {POOL}; every "
            f"measurement in this file is about that representation.")
    print(f"  [pool] {len(POOL)} staged representations, read from "
          f"complexity_helpers.DATASET_REPS['{DATASET}']['general']: {POOL}",
          flush=True)
    # Dirty local run 2026-09-28: KIRBY_LOCAL_FOLDS sets the fold count; unset
    # -> 5. Must match c43_buchwald_signal, whose anchor is subtracted below.
    folds = H.make_folds(groups, y, task,
                         n=int(os.environ.get("KIRBY_LOCAL_FOLDS") or 5))
    R = {"pool": POOL, "best_rep": BEST, "budgets": BUDGETS, "methods": METHODS,
         "pool_source": (
             f"complexity_helpers.DATASET_REPS['{DATASET}']['general'] — the "
             "post-pruning staged ten, the same list c30, c31 and c32 use"),
         "downstream_model": "lightgbm", "importance": "lightgbm_gain",
         "note": "target anchor is cgr_morgan alone under lightgbm, read from c43_buchwald_signal's result file; both sides of the delta are lightgbm"}

    # ---- (a1) SATURATION CURVE for cgr_morgan (train-fold gain ranking) ------
    print("\n== (a1) cgr_morgan saturation curve ==", flush=True)
    W = reps[BEST].shape[1]
    sat = {}   # K -> per-fold rmse
    full_per_fold = []
    rank_cache = []  # per fold: (order desc, tr, va)
    for fi, (tr, va) in enumerate(folds):
        order = np.argsort(H.rank_columns(reps[BEST][tr], y[tr]))[::-1]
        rank_cache.append(order)
        r_full, _ = lgbm_rmse(reps[BEST][tr], y[tr], reps[BEST][va], y[va])
        full_per_fold.append(r_full)
        for K in SAT_KS:
            cols = order[:K]
            r_k, _ = lgbm_rmse(reps[BEST][tr][:, cols], y[tr], reps[BEST][va][:, cols], y[va])
            sat.setdefault(K, []).append(r_k)
        print(f"  fold{fi}: full={r_full:.3f}  " +
              " ".join(f"K{K}={sat[K][-1]:.3f}" for K in SAT_KS), flush=True)
    R["saturation"] = {
        "native_width": W,
        "full_rmse_mean": float(np.mean(full_per_fold)),
        "full_per_fold": [float(x) for x in full_per_fold],
        # delta = K-rmse - full-rmse (RMSE lower better, so positive = K WORSE than full)
        "K_rmse_mean": {K: float(np.mean(sat[K])) for K in SAT_KS},
        "K_delta_vs_full_mean": {K: float(np.mean(sat[K]) - np.mean(full_per_fold)) for K in SAT_KS},
        "K_delta_per_fold": {K: [float(sat[K][i] - full_per_fold[i]) for i in range(len(folds))] for K in SAT_KS},
    }
    json.dump(R, open(CKPT, "w"), indent=2)

    # ---- run quota hybrids: (a2) slice + hybrid + (b) quota + (c) matched ----
    R["cells"] = {}
    for method in METHODS:
        for B in BUDGETS:
            key = f"{method}_B{B}"
            print(f"\n== cell {key} ==", flush=True)
            slice_pf, hyb_pf, full_pf, q_pf = [], [], [], []
            for fi, (tr, va) in enumerate(folds):
                pool_tr = {r: H._clean(reps[r][tr]) for r in POOL}
                pool_va = {r: H._clean(reps[r][va]) for r in POOL}
                Xh_tr, info = hybrid.create_hybrid(
                    pool_tr, y[tr], allocation_method=method, budget=B, n_per_rep=B,
                    importance_method="lightgbm_gain", internal_evaluator="lightgbm",
                    groups=groups[tr], is_clf=False)
                Xh_va = hybrid.apply_feature_selection(pool_va, info)
                # hybrid score
                r_hyb, _ = lgbm_rmse(Xh_tr, y[tr], Xh_va, y[va])
                hyb_pf.append(r_hyb)
                # realised quota for best rep
                sel = info.get(BEST, {})
                q = len(sel["selected_indices"]) if isinstance(sel, dict) and "selected_indices" in sel else 0
                q_pf.append(q)
                # best-rep full width (reuse saturation full)
                full_pf.append(full_per_fold[fi])
                # slice = best rep's contributed columns ALONE
                if q > 0:
                    cols = sel["selected_indices"]
                    r_sl, _ = lgbm_rmse(reps[BEST][tr][:, cols], y[tr], reps[BEST][va][:, cols], y[va])
                    slice_pf.append(r_sl)
                else:
                    slice_pf.append(np.nan)
                print(f"  fold{fi}: q(cgr)={q:>4}  slice={slice_pf[-1] if not np.isnan(slice_pf[-1]) else 'NA'}  "
                      f"full={full_pf[fi]:.3f}  hybrid={r_hyb:.3f}", flush=True)
            valid = ~np.isnan(slice_pf)
            R["cells"][key] = {
                "quota_best_per_fold": [int(x) for x in q_pf],
                "quota_best_mean": float(np.mean(q_pf)),
                "slice_rmse_per_fold": [None if np.isnan(x) else float(x) for x in slice_pf],
                "full_rmse_per_fold": [float(x) for x in full_pf],
                "hybrid_rmse_per_fold": [float(x) for x in hyb_pf],
                "hybrid_rmse_mean": float(np.mean(hyb_pf)),
                # SIGNAL SURVIVAL: slice - full  (positive = slice WORSE, i.e. truncation hurt the rep alone)
                "slice_minus_full_mean": float(np.nanmean(np.array(slice_pf) - np.array(full_pf))),
                "slice_minus_full_per_fold": [None if np.isnan(x) else float(x)
                                              for x in (np.array(slice_pf) - np.array(full_pf))],
                "n_folds_best_got_features": int(valid.sum()),
                # anchor: cgr_morgan alone under lightgbm, from the signal
                # file. Both sides lightgbm, so the difference is readable.
                "hybrid_minus_target_mean": (
                    float(np.mean(hyb_pf) - _ANCHOR) if _ANCHOR is not None else None),
                "anchor_rmse": _ANCHOR,
                "anchor_model": "lightgbm",
            }
            json.dump(R, open(CKPT, "w"), indent=2)

    # ---- (c) GLOBAL RANKING counterfactual (gain-based, matched budget) ------
    print("\n== (c) global-ranking counterfactual ==", flush=True)
    R["global"] = {}
    for B in BUDGETS:
        g_pf, origin_share = [], []
        for fi, (tr, va) in enumerate(folds):
            # concat pool (train) -> one lightgbm gain over all columns
            parts_tr, parts_va, owners = [], [], []
            for r in POOL:
                Xtr = H._clean(reps[r][tr]); Xva = H._clean(reps[r][va])
                parts_tr.append(Xtr); parts_va.append(Xva)
                owners += [r] * Xtr.shape[1]
            Ctr = np.hstack(parts_tr); Cva = np.hstack(parts_va); owners = np.array(owners)
            imp = H.rank_columns(Ctr, y[tr])
            top = np.argsort(imp)[::-1][:B]
            r_g, _ = lgbm_rmse(Ctr[:, top], y[tr], Cva[:, top], y[va])
            g_pf.append(r_g)
            share = {r: int(np.sum(owners[top] == r)) for r in POOL}
            origin_share.append(share)
            print(f"  B{B} fold{fi}: global_rmse={r_g:.3f}  cgr_share={share[BEST]}/{B}", flush=True)
        R["global"][f"B{B}"] = {
            "global_rmse_mean": float(np.mean(g_pf)),
            "global_rmse_per_fold": [float(x) for x in g_pf],
            "cgr_share_per_fold": [s[BEST] for s in origin_share],
            "origin_share_fold0": origin_share[0],
        }
        json.dump(R, open(CKPT, "w"), indent=2)

    R["wall_s"] = round(time.time() - t_start, 1)
    json.dump(R, open(CKPT, "w"), indent=2)
    print("\n===== STEP3 JSON =====")
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
