"""EXPERIMENT C — buchwald_hartwig 4.3, SIGNAL-FIRST.

Step 1 RESOLVE THE SCALE + Step 2 VERDICT ON SIGNAL come first. Only if the
base has real signal do we run Step 3 (the three 4.3 measurements).

Read-only: imports the shared harness (which itself imports find_best_hybrid /
hybrid read-only). Never calls find_best_hybrid() the evaluator; calls
hybrid.create_hybrid / internals directly. No row subsampling. Full CV.
"""
import os
import sys
import json
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
from scipy.stats import spearmanr  # noqa: E402


def r2_from_oof(y, oof):
    m = ~np.isnan(oof)
    y, p = np.asarray(y)[m], oof[m]
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    DS = "buchwald_hartwig"
    print("=" * 70)
    print(f"STEP 1 — RESOLVE THE SCALE for {DS}")
    print("=" * 70)

    reps, y, groups, mname, mfn, lower, task = H.load_full(DS)
    y = np.asarray(y, dtype=float)
    n = len(y)
    print(f"\nn={n}  task={task}  metric={mname} lower_is_better={lower}")
    print(f"label stats: min={y.min():.3f} max={y.max():.3f} "
          f"mean={y.mean():.3f} std={y.std():.4f} median={np.median(y):.3f}")
    print(f"n_groups={len(np.unique(groups))} "
          f"group sizes min/median/max="
          f"{np.min(np.bincount(groups.astype(int))) if np.issubdtype(np.asarray(groups).dtype, np.integer) else 'n/a'}")
    # group size distribution robust to non-int groups
    _, counts = np.unique(groups, return_counts=True)
    print(f"group size: min={counts.min()} median={int(np.median(counts))} "
          f"max={counts.max()} n_groups={len(counts)}")

    # Dirty local run 2026-09-28: KIRBY_LOCAL_FOLDS sets the fold count; unset -> 5.
    folds = H.make_folds(groups, y, task,
                         n=int(os.environ.get("KIRBY_LOCAL_FOLDS") or 5))

    # --- TARGET: cgr_morgan, scored under BOTH downstream models -----------
    # c43_step3 fits lightgbm. This anchor used to be a random forest number
    # only, so subtracting it from a lightgbm hybrid mixed two models. Both are
    # measured here on the same folds; c43_step3 reads the lightgbm one.
    assert "cgr_morgan" in reps, "cgr_morgan not staged!"
    Xt = reps["cgr_morgan"]
    print(f"\ncgr_morgan shape: {Xt.shape}")

    per_model = {}
    for model_name in ("rf", "lightgbm"):
        m_mean, m_pf, m_oof = H.score_matrix(Xt, y, folds, model_name,
                                             mfn, lower, task_type=task)
        m_ok = ~np.isnan(m_oof)
        m_rho, _ = spearmanr(y[m_ok], m_oof[m_ok])
        m_r2 = r2_from_oof(y, m_oof)
        per_model[model_name] = {
            "rmse": float(m_mean),
            "rmse_per_fold": [float(x) for x in m_pf],
            "spearman": float(m_rho),
            "r2": float(m_r2),
            "rmse_over_std": float(m_mean / y.std()),
        }
        print(f"\n--- TARGET cgr_morgan x {model_name} "
              f"({len(folds)}-fold GroupKFold by additive) ---")
        print(f"  RMSE (mean of folds) = {m_mean:.4f}")
        print(f"  RMSE per fold        = {[round(x,3) for x in m_pf]}")
        print(f"  Spearman rho (OOF)   = {m_rho:.4f}")
        print(f"  R2 (OOF)             = {m_r2:.4f}")
        print(f"  RMSE / std(y)        = {m_mean / y.std():.4f}  "
              f"(fraction of label spread; 1.0 = predict-the-mean)")

    # The historic top-level fields stay on the random forest so anything that
    # already reads them keeps reading the same quantity.
    rmse_mean = per_model["rf"]["rmse"]
    rmse_pf = per_model["rf"]["rmse_per_fold"]
    rho = per_model["rf"]["spearman"]
    r2 = per_model["rf"]["r2"]
    print(f"\n  std(y)               = {y.std():.4f}")

    # naive baseline: predict train-fold mean
    naive_pf = []
    for tr, va in folds:
        pred = np.full(len(va), y[tr].mean())
        naive_pf.append(float(mfn(y[va], pred)))
    print(f"  predict-mean RMSE    = {np.mean(naive_pf):.4f} "
          f"(per fold {[round(x,2) for x in naive_pf]})")

    out = {
        "n": int(n),
        "std_y": float(y.std()),
        "y_min": float(y.min()), "y_max": float(y.max()),
        "target_rmse": float(rmse_mean),
        "target_rmse_per_fold": [float(x) for x in rmse_pf],
        "target_spearman": float(rho),
        "target_r2": float(r2),
        "rmse_over_std": float(rmse_mean / y.std()),
        "predict_mean_rmse": float(np.mean(naive_pf)),
        "n_groups": int(len(counts)),
        "group_size_min": int(counts.min()),
        "group_size_max": int(counts.max()),
        # Per downstream model, so a later delta can be taken against the same
        # model it was fitted with.
        "target_by_model": per_model,
        "anchor_model_for_step3": "lightgbm",
    }
    print("\n===== STEP1 JSON =====")
    print(json.dumps(out, indent=2))

    # Write it. Printing only meant the anchor was lost when the process
    # exited, and c43_step3 held it as a typed-in literal instead.
    # C43_OUT_DIR (dirty local run 2026-09-28) redirects the result file;
    # unset -> the repo's tests/results/complexity as before.
    out_dir = (os.environ.get("C43_OUT_DIR")
               or os.path.join(_REPO, "tests", "results", "complexity"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"c43_buchwald_signal_{DS}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    main()
