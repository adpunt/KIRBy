"""f30_analyse — read the f30 result files and produce the three deliverables.

  1. PREDICTIVE PERFORMANCE, per dataset, per downstream model: the paired
     per-fold delta of the hybrid against the best single representation for that
     same model, in that dataset's own metric, direction-corrected.  NEVER pooled
     across datasets — rmse, spearman and f1_weighted are not commensurable and
     two of the four run in opposite directions.

  2. AGREEMENT — which arms are saying the same thing.  Four statistics, all of
     them already in this repository and already used in this project:
        Spearman            f05_importance_correlation.py:161
        Kendall tau_b       find_best_hybrid_molecular.py:1168
        top-k set overlap   find_best_hybrid_molecular.py:1179
        Nogueira phi-hat    c10_noise_robustness.py:42, c24_blueprint_transfer.py:50
     Each feature-rank statistic is reported TWICE: over all features, and over
     the union of features both arms scored non-zero.  The second is the honest
     one, because lightgbm_gain scores only ~200 of a 2048-feature representation
     (gain is 0 for any feature no tree split on) and np.argsort orders the tied
     zeros by index, so an all-features Spearman is dominated by how two arms'
     tied-zero blocks happen to line up.

     Prediction agreement is separate and uses the PREDICTION vectors, not the
     feature scores: Spearman for the regression datasets, Cohen's kappa for
     tcga_brca_pam50.

  3. COMPUTE TIME per arm per dataset, and relative to the cheapest arm there.

Decision rule: two arms that agree closely collapse to one and the cheaper is
kept.  Applied PER DATASET — a pair that agrees on caco2 need not on tcga.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, kendalltau, rankdata

OUTDIR = os.environ.get(
    "F30_OUT", os.path.join(os.environ.get("KIRBY_CACHE_DIR", "tests/results"),
                            "f30_importance_methods"))


def nogueira_stability(feature_sets, total_features):
    """Nogueira, Sechidis & Brown 2018, JMLR 19(53), Eq. 2 — chance-corrected, so
    unlike Jaccard it is not biased by how many features were selected.
    Reference: nogueirs/JMLR2018 `python/stability/__init__.py`; the repo's own
    copies are c10_noise_robustness.py:42 and c24_blueprint_transfer.py:50.
    """
    M = len(feature_sets)
    if M < 2:
        return float("nan")
    Z = np.zeros((M, total_features))
    for i, s in enumerate(feature_sets):
        Z[i, list(s)] = 1.0
    hatPF = Z.mean(axis=0)
    kbar = hatPF.sum()
    denom = (kbar / total_features) * (1.0 - kbar / total_features)
    if denom <= 0:
        return float("nan")
    return float(1.0 - (M / (M - 1.0)) * np.mean(hatPF * (1.0 - hatPF)) / denom)


def _pair_stats(a, b, topk):
    """Feature-rank agreement between two score vectors, both ways."""
    out = {}
    for tag, mask in (("all", np.ones(len(a), dtype=bool)),
                      ("both_scored", (a != 0) | (b != 0))):
        if mask.sum() < 3:
            out[f"spearman_{tag}"] = float("nan")
            out[f"kendall_{tag}"] = float("nan")
            continue
        x, yv = a[mask], b[mask]
        r = spearmanr(x, yv).statistic if hasattr(spearmanr(x, yv), "statistic") \
            else spearmanr(x, yv)[0]
        t = kendalltau(x, yv, variant="b")[0]
        out[f"spearman_{tag}"] = float(np.nan_to_num(r, nan=0.0))
        out[f"kendall_{tag}"] = float(np.nan_to_num(t, nan=0.0))
    k = int(min(topk, len(a)))
    sa = set(np.argsort(a)[-k:].tolist())
    sb = set(np.argsort(b)[-k:].tolist())
    out["topk_overlap"] = float(len(sa & sb) / k) if k else float("nan")
    out["topk"] = k
    return out


def load_dataset(ds, tag="full"):
    arms, npzs = {}, {}
    for p in sorted(glob.glob(os.path.join(OUTDIR, f"{ds}__*__{tag}.json"))):
        arm = os.path.basename(p)[len(ds) + 2:-(len(tag) + 7)]  # strip "<ds>__" and "__<tag>.json"
        with open(p) as f:
            arms[arm] = json.load(f)
        z = p[:-4] + "npz"
        if os.path.exists(z):
            npzs[arm] = z
    return arms, npzs


def report(ds, tag="full", topk=None):
    arms, npzs = load_dataset(ds, tag)
    if not arms:
        print(f"[{ds}] no {tag} result files in {OUTDIR}")
        return None
    any_arm = next(iter(arms.values()))
    metric, lower = any_arm["metric"], any_arm["lower_is_better"]
    widths = any_arm["rep_widths"]
    print(f"\n{'=' * 78}\n{ds}   metric={metric} "
          f"({'lower' if lower else 'higher'} is better)   arms={len(arms)}\n{'=' * 78}")

    # ---- 1. predictive ----------------------------------------------------
    print("\n1. PREDICTIVE — paired per-fold delta vs the best single representation")
    print("   positive = the hybrid beat the target, in this dataset's own metric\n")
    rows = []
    for arm, r in arms.items():
        for key, c in r["cells"].items():
            rows.append((arm, key, c["delta_mean"], c["folds_won"],
                         len(c["delta_per_fold"]), r["seconds_total"],
                         r["family"], r["redundancy_aware"]))
    rows.sort(key=lambda x: -x[2])
    print(f"   {'arm':26s} {'cell':38s} {'delta':>9s} {'won':>6s} {'family':18s} redun")
    for arm, key, dm, won, nf, _s, fam, red in rows:
        print(f"   {arm:26s} {key:38s} {dm:+9.4f} {won:>3d}/{nf:<2d} {fam:18s} "
              f"{'yes' if red else '-'}")

    best_by_arm = {}
    for arm, r in arms.items():
        best_by_arm[arm] = max(c["delta_mean"] for c in r["cells"].values())

    # ---- 2. agreement -----------------------------------------------------
    print("\n2. AGREEMENT — feature rankings, averaged over representations and folds")
    names = sorted(a for a in arms if a in npzs)
    pair = defaultdict(list)
    stab = {}
    if names:
        cache = {a: np.load(npzs[a]) for a in names}
        keys = [k for k in cache[names[0]].files if k.startswith("f")]
        for a in names:
            sets_by_rep = defaultdict(list)
            for k in keys:
                fi, rn = k[1:].split("__", 1)
                v = cache[a][k]
                kk = int(topk or max(1, round(100 / len(widths))))
                sets_by_rep[rn].append(set(np.argsort(v)[-kk:].tolist()))
            vals = [nogueira_stability(s, widths[rn])
                    for rn, s in sets_by_rep.items() if len(s) > 1]
            stab[a] = float(np.nanmean(vals)) if vals else float("nan")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for k in keys:
                    if k not in cache[b].files:
                        continue
                    rn = k[1:].split("__", 1)[1]
                    kk = int(topk or max(1, round(100 / len(widths))))
                    pair[(a, b)].append(_pair_stats(cache[a][k].astype(float),
                                                    cache[b][k].astype(float), kk))

        print(f"\n   {'arm A':24s} {'arm B':24s} {'rho_all':>8s} {'rho_scored':>11s} "
              f"{'tau_scored':>11s} {'top-k':>7s}")
        agg = []
        for (a, b), lst in pair.items():
            m = {k: float(np.nanmean([d[k] for d in lst])) for k in lst[0]}
            agg.append((a, b, m))
        agg.sort(key=lambda x: -x[2]["spearman_both_scored"])
        for a, b, m in agg:
            print(f"   {a:24s} {b:24s} {m['spearman_all']:8.3f} "
                  f"{m['spearman_both_scored']:11.3f} {m['kendall_both_scored']:11.3f} "
                  f"{m['topk_overlap']:7.3f}")

        print("\n   Nogueira phi-hat — does the arm pick the same features on "
              "different folds")
        for a in sorted(stab, key=lambda x: -(stab[x] if stab[x] == stab[x] else -9)):
            print(f"   {a:26s} {stab[a]:6.3f}")

        print("\n   COLLAPSE CANDIDATES (rho over both-scored features > 0.95) — "
              "keep the cheaper")
        secs = {a: arms[a]["seconds_total"] for a in names}
        found = False
        for a, b, m in agg:
            if m["spearman_both_scored"] > 0.95:
                found = True
                keep, drop = (a, b) if secs[a] <= secs[b] else (b, a)
                print(f"   {a} ~ {b}  rho={m['spearman_both_scored']:.3f}  "
                      f"-> keep {keep} ({secs[keep]:.0f}s), drop {drop} "
                      f"({secs[drop]:.0f}s)")
        if not found:
            print("   none — no pair agrees closely enough to collapse")

    # ---- 3. compute time --------------------------------------------------
    print("\n3. COMPUTE TIME")
    print("   `total` includes the downstream model fits, which are IDENTICAL for every")
    print("   arm, so it is a poor discriminator. Rank on the scoring column.\n")
    secs = {a: arms[a]["seconds_total"] for a in arms}
    per = {a: float(np.mean([t["seconds"] for t in arms[a]["timings"]])) for a in arms}
    scoring = {a: float(np.sum([t["seconds"] for t in arms[a]["timings"]])) for a in arms}
    cheapest = min(v for v in per.values() if v > 0) or 1.0
    print(f"   {'arm':26s} {'scoring':>10s} {'per rep/fold':>13s} {'x cheapest':>11s} "
          f"{'total':>10s}")
    for a in sorted(per, key=per.get):
        print(f"   {a:26s} {scoring[a]:9.1f}s {per[a]:12.3f}s "
              f"{per[a] / cheapest:10.1f}x {secs[a]:9.1f}s")

    return {"dataset": ds, "metric": metric, "lower_is_better": lower,
            "best_delta_by_arm": best_by_arm, "stability": stab,
            "seconds": secs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*",
                    default=["caco2", "buchwald_hartwig",
                             "flab_expression_koenig", "tcga_brca_pam50"])
    ap.add_argument("--tag", default="full", choices=["full", "smoke"])
    ap.add_argument("--topk", type=int, default=None)
    a = ap.parse_args()
    print(f"f30 analysis — reading {OUTDIR}")
    out = {}
    for ds in a.datasets:
        r = report(ds, tag=a.tag, topk=a.topk)
        if r:
            out[ds] = r
    if out:
        p = os.path.join(OUTDIR, f"f30_summary__{a.tag}.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nwrote {p}")
        print("\nNOTE: the four datasets are NEVER pooled. Each table above is in "
              "its own metric and its own direction.")


if __name__ == "__main__":
    main()
