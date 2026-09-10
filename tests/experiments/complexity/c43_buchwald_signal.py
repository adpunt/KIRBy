"""EXPERIMENT C — buchwald_hartwig 4.3, SIGNAL-FIRST.

Step 1 RESOLVE THE SCALE + Step 2 VERDICT ON SIGNAL come first. Only if the
base has real signal do we run Step 3 (the three 4.3 measurements).

Read-only: imports the shared harness (which itself imports find_best_hybrid /
hybrid read-only). Never calls find_best_hybrid() the evaluator; calls
hybrid.create_hybrid / internals directly. No row subsampling. Full CV.
"""
import sys
import json
import numpy as np

HARNESS = "/private/tmp/claude-501/-Users-apunt-repos-KIRBy/45a3af00-2ccf-4e35-a9c6-538fb157f521/scratchpad"
sys.path.insert(0, HARNESS)

# lightgbm-first segfault guard lives inside the harness module.
import round2_harness as H  # noqa: E402
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

    folds = H.make_folds(groups, y, task, n=5)

    # --- TARGET: cgr_morgan x rf ------------------------------------------
    assert "cgr_morgan" in reps, "cgr_morgan not staged!"
    Xt = reps["cgr_morgan"]
    print(f"\ncgr_morgan shape: {Xt.shape}")
    rmse_mean, rmse_pf, oof = H.score_matrix(Xt, y, folds, "rf",
                                             mfn, lower, task_type=task)
    rho, _ = spearmanr(y[~np.isnan(oof)], oof[~np.isnan(oof)])
    r2 = r2_from_oof(y, oof)
    print(f"\n--- TARGET cgr_morgan x rf (5-fold GroupKFold by additive) ---")
    print(f"  RMSE (mean of folds) = {rmse_mean:.4f}")
    print(f"  RMSE per fold        = {[round(x,3) for x in rmse_pf]}")
    print(f"  Spearman rho (OOF)   = {rho:.4f}")
    print(f"  R2 (OOF)             = {r2:.4f}")
    print(f"  std(y)               = {y.std():.4f}")
    print(f"  RMSE / std(y)        = {rmse_mean / y.std():.4f}  "
          f"(fraction of label spread; 1.0 = predict-the-mean)")

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
    }
    print("\n===== STEP1 JSON =====")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
