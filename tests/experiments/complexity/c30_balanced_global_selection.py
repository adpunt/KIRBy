"""EXPERIMENT c30 — balanced global feature selection tied to the pool-size knob.

Question: does BALANCED global selection (per-rep top-N cap, THEN global
lightgbm_gain top-B over the balanced pool) beat the per-rep QUOTA, and does the
per-rep-top-N cap actually fix the density bias that sank NAIVE global (Finding 3)?

Three arms, per dataset, full-size grouped CV, never averaged across datasets:
  TARGET : best single staged rep x lightgbm (the number to beat).
  ARM 1  : per-rep quota  — equal fixed quota B/|pool| feats/rep, top by
           lightgbm_gain (== create_hybrid('fixed'), the 'original working
           hybrid'). budgets [100,500,1000] on pool5 and pool10.
  ARM 2  : balanced global — per-rep top-N (N in 50,100,200,300) concatenated
           over the pool, THEN global lightgbm_gain top-B (B in 100,500,1000).
  ARM 3  : naive global — glue the whole pool, no cap, global lightgbm_gain
           top-B  (Finding-3 control).

*** FIX 2026-09-04 (SPLIT_AUDIT c30:121 and exp_b:142): the TARGET rep and the
pool5 / pool10 membership were ranked on the same 5 held-out folds that then
graded all three arms.  Both are now ranked INSIDE each outer fold's TRAIN split
via an inner group-aware CV (N_INNER folds), so no fold is graded against a
choice made from its own held-out labels.  The target's score for a fold is that
fold's HELD-OUT score for the rep its own training rows picked. ***

*** PORT 2026-09-04: the /private/tmp scratchpad harness this file imported was
deleted, so it could not run at all.  It now uses the committed
tests/experiments/_alloc_harness (same call surface, lightgbm-first import
guard) and writes to tests/results/complexity, like c31 and c32. ***

HARD RULES honoured: never edits find_best_hybrid.py / hybrid.py (import-only);
never calls find_best_hybrid(); full rows always (row-count asserted in harness);
lightgbm_gain importance FIXED; lightgbm downstream model FIXED (apples-to-apples
paired delta); repo cv_split GroupKFold; per-fold train-only selection.

Run:  python c30_balanced_global_selection.py <dataset> [<dataset> ...]
Env:  C30_OUT   output json path (default tests/results/complexity/...)
"""
import os
import sys
from pathlib import Path

# Portable path bootstrap: locate the repo root from this file (no hardcoded
# absolute path) and put tests/ + src/ on sys.path so the imports resolve
# whether run locally or on the server.
_REPO = str(Path(__file__).resolve().parents[3])
for _p in (os.path.join(_REPO, "tests"), os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Shared allocation harness — a committed repo module (imports lightgbm FIRST
# for the macOS segfault guard).
from experiments import _alloc_harness as H  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from kirby.hybrid import create_hybrid  # noqa: E402
from kirby.find_best_hybrid import get_model  # noqa: E402

MODEL = "lightgbm"          # downstream model, FIXED
IMP = "lightgbm_gain"       # importance method, FIXED
BUDGETS = [100, 500, 1000]
NCAPS = [50, 100, 200, 300]
POOLS = [5, 10]
NFOLDS = 5
N_INNER = 3                 # inner group-aware CV folds for the train-only rank

DEFAULT_DATASETS = ["herg_ki", "logd", "flab_thermo_tresanco"]

CAVEATS = [
    "2026-09-04 (SPLIT_AUDIT c30:121, exp_b:142): the TARGET rep and the pool5 / "
    "pool10 membership are ranked INSIDE each outer fold's TRAIN split, via an "
    "inner group-aware CV over every staged rep.  Before this they came from one "
    "whole-data ranking computed on the same held-out folds that graded the arms. "
    "Results written before 2026-09-04 are NOT comparable to results after it.",
    "The target can be a DIFFERENT rep in different folds, and pool5 can hold "
    "different reps in different folds; target.best_rep_per_fold and "
    "pool_per_fold record which.  whole_data_ranking_top12 is descriptive only — "
    "nothing is selected from it.",
    "If an outer fold's TRAIN split has too few groups for the inner CV, that "
    "fold falls back to the whole-data ranking and is listed in "
    "rank_fallback_folds; that fold's target and pools carry the old leak.",
    "pool10 == every staged rep on these datasets (10 staged), so its membership "
    "is not a selection; only pool5 and the target are.",
]


def gsign(lower_is_better):
    """+1 if higher metric is better, -1 if lower is better."""
    return -1.0 if lower_is_better else 1.0


# ---------------------------------------------------------------------------
# fold-safe scoring for an arbitrary per-fold column selection
# ---------------------------------------------------------------------------
def score_selection(reps, y, folds, per_fold_sel, task_type, metric_fn):
    """per_fold_sel[i] = {rep_name: np.array(col_indices)} chosen on fold i's TRAIN.

    Fits lightgbm per fold on the selected columns (scaled), predicts held-out.
    Returns (per_fold_metric, oof_preds, mean_nfeat).
    """
    oof = np.full(len(y), np.nan)
    per_fold = []
    nfeat = []
    for i, (tr, va) in enumerate(folds):
        sel = per_fold_sel[i]
        parts_tr, parts_va = [], []
        for rn, idx in sel.items():
            idx = np.asarray(idx)
            if len(idx) == 0:
                continue
            X = reps[rn]
            parts_tr.append(H._clean(X[tr][:, idx]))
            parts_va.append(H._clean(X[va][:, idx]))
        if not parts_tr:
            p = np.full(len(va), float(np.mean(y[tr])))
            oof[va] = p
            nfeat.append(0)
            per_fold.append(float(metric_fn(y[va], p)))
            continue
        Xtr = np.hstack(parts_tr)
        Xva = np.hstack(parts_va)
        nfeat.append(Xtr.shape[1])
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xva = sc.transform(Xva)
        m = get_model(MODEL, task_type=task_type, n_samples=len(tr))
        if m is None:
            raise RuntimeError(f"get_model returned None for {MODEL} at n={len(tr)}")
        m.fit(Xtr, y[tr])
        p = m.predict(Xva)
        oof[va] = p
        per_fold.append(float(metric_fn(y[va], p)))
    return per_fold, oof, float(np.mean(nfeat))


def run_dataset(ds):
    t_start = time.time()
    out = {"dataset": ds, "model": MODEL, "importance": IMP, "caveats": CAVEATS}

    reps, labels, groups, mname, mfn, lower, task = H.load_full(ds)
    folds = H.make_folds(groups, labels, task, n=NFOLDS)
    y = np.asarray(labels, dtype=float)
    gs = gsign(lower)
    all_reps = sorted(reps)
    out.update({"n": int(len(y)), "metric": mname, "lower_is_better": bool(lower),
                "n_reps": len(reps), "n_folds": NFOLDS,
                "n_inner_rank_folds": N_INNER})

    # -- HELD-OUT standalone score for every staged rep.  These are the numbers
    #    the target is GRADED with; the whole-data ranking off them is kept only
    #    as a descriptive table and as the per-fold ranking's fallback. --------
    rep_scores = {}      # rep -> (score_mean, per_fold, oof)
    for rn in all_reps:
        s, pf, oof = H.score_matrix(reps[rn], y, folds, MODEL, mfn, lower,
                                    task_type=task)
        rep_scores[rn] = (s, pf, oof)
    ranked_whole = sorted(rep_scores, key=lambda r: gs * rep_scores[r][0],
                          reverse=True)
    out["whole_data_ranking_top12"] = [
        (r, round(rep_scores[r][0], 5)) for r in ranked_whole[:12]]

    # -- PER-FOLD TRAIN-ONLY ranking: inner group-aware CV on the outer fold's
    #    training rows, over every staged rep.  Picks that fold's target and its
    #    pool5 / pool10 membership. ------------------------------------------
    score_whole = {rn: gs * rep_scores[rn][0] for rn in all_reps}
    fold_train_score, rank_fallback = [], []
    for i, (tr, va) in enumerate(folds):
        sub_groups, sub_y = groups[tr], y[tr]
        n_inner = min(N_INNER, len(np.unique(sub_groups)))
        if n_inner < 2:
            fold_train_score.append(dict(score_whole))
            rank_fallback.append(i)
            continue
        try:
            inner = H.make_folds(sub_groups, sub_y, task, n=n_inner)
            sc = {}
            for rn in all_reps:
                m_native, _pf, _oof = H.score_matrix(
                    H._clean(reps[rn][tr]), sub_y, inner, MODEL, mfn, lower,
                    task_type=task)
                sc[rn] = m_native
            fold_train_score.append({rn: gs * sc[rn] for rn in all_reps})
        except Exception as e:
            print(f"  [rank-fallback] fold {i}: inner CV failed "
                  f"({type(e).__name__}: {e})")
            fold_train_score.append(dict(score_whole))
            rank_fallback.append(i)
    out["rank_fallback_folds"] = rank_fallback

    def _rank_key(i):
        def k(rn):
            v = fold_train_score[i][rn]
            return v if np.isfinite(v) else -np.inf
        return k

    fold_ranked = [sorted(all_reps, key=_rank_key(i), reverse=True)
                   for i in range(NFOLDS)]
    # pools[p][i] = the p reps fold i's OWN training rows put on top.
    pools = {p: [fold_ranked[i][:p] for i in range(NFOLDS)] for p in POOLS}
    out["pool_per_fold"] = {str(p): {str(i): pools[p][i] for i in range(NFOLDS)}
                            for p in POOLS}
    out["fold_ranking"] = {str(i): fold_ranked[i] for i in range(NFOLDS)}

    # -- TARGET: per fold, the train-picked rep's HELD-OUT score for that fold --
    fold_target = [fold_ranked[i][0] for i in range(NFOLDS)]
    target_pf = [rep_scores[fold_target[i]][1][i] for i in range(NFOLDS)]
    target_oof = np.full(len(y), np.nan)
    for i, (tr, va) in enumerate(folds):
        target_oof[va] = rep_scores[fold_target[i]][2][va]
    # reproducibility: re-score every rep the folds picked, on the SAME folds,
    # and assert the per-fold values are identical.
    reproduces = True
    for rn in sorted(set(fold_target)):
        _s2, pf2, _o2 = H.score_matrix(reps[rn], y, folds, MODEL, mfn, lower,
                                       task_type=task)
        reproduces &= bool(np.allclose(pf2, rep_scores[rn][1], atol=1e-9))
    assert reproduces, f"TARGET did not reproduce on {ds}"

    out["target"] = {
        "picked_per_fold": True,
        "best_rep_per_fold": fold_target,
        "per_fold": [round(x, 5) for x in target_pf],
        "score_mean": float(np.mean(target_pf)),
        "reproduces": bool(reproduces),
        "whole_data_best_rep": ranked_whole[0],
        "folds_agreeing_with_whole_data": int(
            sum(1 for r in fold_target if r == ranked_whole[0])),
    }

    # -- per-fold, per-rep TRAIN importances (reused by ARM1 quota + ARM2 caps).
    #    Over EVERY staged rep: pool membership is now a per-fold decision.
    # keyed [rep][fold] -> importance vector over that rep's columns
    rep_imp = {rn: [] for rn in all_reps}
    for i, (tr, va) in enumerate(folds):
        for rn in all_reps:
            rep_imp[rn].append(H.rank_columns(reps[rn][tr], y[tr], method=IMP))

    def topk_local(rn, fold_i, k):
        imp = rep_imp[rn][fold_i]
        k = int(min(k, len(imp)))
        if k <= 0:
            return np.array([], dtype=int)
        return np.argsort(imp)[-k:][::-1]

    # ======================================================================
    # ARM 1 — per-rep quota (equal fixed quota B/|pool| feats/rep)
    # ======================================================================
    arm1 = {}
    for p in POOLS:
        for B in BUDGETS:
            q = max(1, B // p)
            per_fold_sel = [{rn: topk_local(rn, i, q) for rn in pools[p][i]}
                            for i in range(NFOLDS)]
            pf, oof, nf = score_selection(reps, y, folds, per_fold_sel, task, mfn)
            arm1[f"pool{p}_B{B}"] = {
                "per_fold": pf, "oof": oof, "nfeat": nf, "quota_per_rep": q}

    # validate ARM1 selection matches engine create_hybrid('fixed') on one fold/config
    engine_check = _validate_against_engine(reps, y, folds, pools[min(POOLS)][0])

    # ======================================================================
    # ARM 2 — balanced global: per-rep top-N cap, THEN global gain top-B
    # ======================================================================
    arm2 = {}
    for p in POOLS:
        for N in NCAPS:
            # per fold: build balanced pool (top-N per rep by train imp), rank globally
            fold_global = []
            for i, (tr, va) in enumerate(folds):
                origins = []        # (rep, local_col) per balanced column
                mats = []
                for rn in pools[p][i]:
                    idx = topk_local(rn, i, N)
                    if len(idx) == 0:
                        continue
                    mats.append(H._clean(reps[rn][tr][:, idx]))
                    origins.extend([(rn, int(c)) for c in idx])
                if not mats:
                    fold_global.append(([], np.array([])))
                    continue
                balanced = np.hstack(mats)
                gimp = H.rank_columns(balanced, y[tr], method=IMP)
                fold_global.append((origins, gimp))
            for B in BUDGETS:
                per_fold_sel, origin_counts = _topB(fold_global, B)
                pf, oof, nf = score_selection(reps, y, folds, per_fold_sel, task, mfn)
                arm2[f"pool{p}_N{N}_B{B}"] = {
                    "per_fold": pf, "oof": oof, "nfeat": nf,
                    "origin_counts": origin_counts}

    # ======================================================================
    # ARM 3 — naive global: glue whole pool, no cap, global gain top-B
    # ======================================================================
    arm3 = {}
    for p in POOLS:
        fold_global = []
        for i, (tr, va) in enumerate(folds):
            origins = []
            mats = []
            for rn in pools[p][i]:
                W = reps[rn].shape[1]
                mats.append(H._clean(reps[rn][tr]))
                origins.extend([(rn, int(c)) for c in range(W)])
            glued = np.hstack(mats)
            gimp = H.rank_columns(glued, y[tr], method=IMP)
            fold_global.append((origins, gimp))
        for B in BUDGETS:
            per_fold_sel, origin_counts = _topB(fold_global, B)
            pf, oof, nf = score_selection(reps, y, folds, per_fold_sel, task, mfn)
            arm3[f"pool{p}_B{B}"] = {
                "per_fold": pf, "oof": oof, "nfeat": nf,
                "origin_counts": origin_counts}

    # ======================================================================
    # COMPARE — paired per-fold delta vs target (direction-corrected)
    # ======================================================================
    tpf = np.array(target_pf)

    def summarize(cfg):
        pf = np.array(cfg["per_fold"])
        deltas = gs * (pf - tpf)                     # +ve = beat target
        return {
            "score_mean": float(np.mean(pf)),
            "delta_mean": float(np.mean(deltas)),
            "delta_per_fold": [round(float(x), 5) for x in deltas],
            "folds_won": int(np.sum(deltas > 0)),
            "nfeat": round(cfg["nfeat"], 1),
            **({"origin_counts": cfg["origin_counts"]} if "origin_counts" in cfg else {}),
        }

    out["arm1_quota"] = {k: summarize(v) for k, v in arm1.items()}
    out["arm2_balanced"] = {k: summarize(v) for k, v in arm2.items()}
    out["arm3_naive"] = {k: summarize(v) for k, v in arm3.items()}
    out["engine_validation"] = engine_check

    # prediction correlation between arms (representative: pool10, B=500;
    # ARM2 uses its best N at that pool/budget) vs target OOF.
    def pearson(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < 3:
            return None
        return float(np.corrcoef(a[m], b[m])[0, 1])
    p_ref = 10 if 10 in POOLS else POOLS[-1]
    b_ref = 500 if 500 in BUDGETS else BUDGETS[len(BUDGETS) // 2]
    a1 = arm1[f"pool{p_ref}_B{b_ref}"]["oof"]
    a3 = arm3[f"pool{p_ref}_B{b_ref}"]["oof"]
    # ARM2's N here is the one with the largest delta on these same folds — a
    # choice of WHICH correlation to show, not a reported delta.
    best_a2_key = max(
        [k for k in arm2 if k.startswith(f"pool{p_ref}_") and k.endswith(f"_B{b_ref}")],
        key=lambda k: out["arm2_balanced"][k]["delta_mean"])
    a2 = arm2[best_a2_key]["oof"]
    out["pred_correlation_pool%d_B%d" % (p_ref, b_ref)] = {
        "arm2_key": best_a2_key,
        "arm2_key_note": "N chosen as the largest delta on the same folds",
        "target_vs_arm1": pearson(target_oof, a1),
        "target_vs_arm2": pearson(target_oof, a2),
        "target_vs_arm3": pearson(target_oof, a3),
        "arm1_vs_arm2": pearson(a1, a2),
        "arm2_vs_arm3": pearson(a2, a3),
        "arm1_vs_arm3": pearson(a1, a3),
    }

    out["wall_time_s"] = round(time.time() - t_start, 1)
    print(f"[{ds}] target/fold={fold_target} "
          f"mean={np.mean(target_pf):.4f}  ({out['wall_time_s']}s)")
    return out


def _topB(fold_global, B):
    """Take the global top-B per fold from (origins, importance) pairs."""
    per_fold_sel, origin_counts = [], {}
    for origins, gimp in fold_global:
        if len(gimp) == 0:
            per_fold_sel.append({})
            continue
        k = min(B, len(gimp))
        top = np.argsort(gimp)[-k:][::-1]
        sel = {}
        for t in top:
            rn, c = origins[t]
            sel.setdefault(rn, []).append(c)
            origin_counts[rn] = origin_counts.get(rn, 0) + 1
        per_fold_sel.append({rn: np.array(v) for rn, v in sel.items()})
    return per_fold_sel, origin_counts


def _validate_against_engine(reps, y, folds, pool5):
    """Prove ARM1 manual quota == create_hybrid('fixed') on fold 0, pool5, q=20.

    Runs create_hybrid(allocation_method='fixed', n_per_rep=20) on fold-0 TRAIN,
    applies to fold-0 VAL, and checks the selected column indices per rep match
    our manual top-20-by-lightgbm_gain selection.
    """
    tr, va = folds[0]
    train_reps = {rn: reps[rn][tr] for rn in pool5}
    try:
        _, finfo = create_hybrid(train_reps, y[tr], allocation_method='fixed',
                                 n_per_rep=20, importance_method=IMP)
    except Exception as e:
        return {"ran": False, "reason": f"{type(e).__name__}: {e}"}
    match = {}
    for rn in pool5:
        eng = set(int(x) for x in finfo[rn]['selected_indices'])
        imp = H.rank_columns(reps[rn][tr], y[tr], method=IMP)
        man = set(int(x) for x in np.argsort(imp)[-20:][::-1])
        match[rn] = (eng == man)
    return {"ran": True, "all_reps_match": bool(all(match.values())),
            "per_rep": match, "config": "fold0 pool5 fixed n_per_rep=20"}


def _strip_oof(obj):
    """Drop the bulky per-row prediction arrays before dumping."""
    if isinstance(obj, dict):
        return {k: _strip_oof(v) for k, v in obj.items() if k != "oof"}
    return obj


def main(datasets):
    t_run = time.time()
    _default_out = os.path.join(_REPO, "tests", "results", "complexity",
                                "c30_balanced_global.json")
    respath = Path(os.environ.get("C30_OUT", _default_out))
    respath.parent.mkdir(parents=True, exist_ok=True)

    all_out = None
    if respath.exists():
        try:
            with open(respath) as f:
                all_out = json.load(f)
            all_out.pop("complete", None)
        except Exception:
            all_out = None
    if all_out is None:
        all_out = {"datasets": {}}
    all_out.setdefault("datasets", {})

    # BLOCKING metric-direction gate (raises on failure).
    if not all_out.get("gate"):
        gate = H.assert_metric_direction(
            verbose=True, datasets=sorted(set(H.EXPECTED_DIRECTION) | set(datasets)))
        all_out["gate"] = {k: {"direction": v["direction"],
                               "monotone": v["monotone"], "metric": v["metric"]}
                           for k, v in gate.items()}

    all_out["config"] = {"model": MODEL, "importance": IMP, "budgets": BUDGETS,
                         "ncaps": NCAPS, "pools": POOLS, "n_folds": NFOLDS,
                         "n_inner_rank_folds": N_INNER, "per_fold_ranking": True}
    all_out["caveats"] = CAVEATS

    def checkpoint():
        all_out["wall_time_s"] = round(time.time() - t_run, 1)
        tmp = respath.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(all_out, f, indent=2, default=str)
        tmp.replace(respath)

    for ds in datasets:
        prior = all_out["datasets"].get(ds)
        if prior and "error" not in prior and prior.get("target"):
            print(f"[skip] {ds} already complete in checkpoint.")
            continue
        try:
            all_out["datasets"][ds] = _strip_oof(run_dataset(ds))
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_out["datasets"][ds] = {"error": f"{type(e).__name__}: {e}"}
        checkpoint()
        print(f"[CHECKPOINT] wrote {respath} after {ds}")

    all_out["complete"] = True
    checkpoint()
    print(f"\n[ALL DONE] -> {respath}  ({all_out['wall_time_s']}s)")
    return all_out


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DATASETS)
