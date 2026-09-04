"""EXPERIMENT B — balanced global feature selection vs per-rep quota, tied to
the pool-size knob (pool5 / pool10).

Reuses the round2 harness (load_full / make_folds / rank_columns / get_model /
metric registry) as the read-only substrate. NEVER calls find_best_hybrid();
selection is done with kirby.hybrid.compute_feature_importance (via the harness
rank_columns) — a hybrid internal — with per-fold train-only ranking.

Arms (per dataset, per pool in {top5, top10} reps):
  TARGET : best single rep x best model (model chosen from {lightgbm, rf}); same
           model reused for every arm (rebuild-roadmap "same-model" rule).
  ARM 1  : per-rep quota — split budget B equally across the pool's K reps, each
           rep contributes its own top-(B/K) lightgbm_gain features.
  ARM 2  : balanced global — top-N per rep (N in 50,100,200,300) concatenated
           over the pool, then ONE global lightgbm_gain ranking picks top-B.
  ARM 3  : naive global — glue ALL columns of the pool's reps (no per-rep cap),
           one global lightgbm_gain ranking picks top-B (Finding-3 control).
  budgets B in {100, 500, 1000}.

Fold safety: every importance ranking is computed on TRAIN rows only, then the
selected column indices are applied to both train and val rows. Splitting is the
repo's cv_split (GroupKFold, deterministic). Full dataset sizes, no subsampling.

*** FIX 2026-09-04 (SPLIT_AUDIT exp_b:142): best_rep, pool5, pool10 and the
comparator model were all ranked on the same 5 held-out folds that then graded
every arm, so the pool5-vs-pool10 comparison was decided partly by the scores
that graded it.  All four are now picked INSIDE each outer fold's TRAIN split,
via an inner group-aware CV (N_INNER folds).  The comparator model can differ
between folds; every arm in a fold uses that fold's model, so the pairing holds.
Results written before 2026-09-04 are NOT comparable to results after it. ***

*** PORT 2026-09-04: the /private/tmp scratchpad harness this file imported was
deleted, so it could not run at all.  It now uses the committed
tests/experiments/_alloc_harness, and the metric-direction gate is the harness's
own assert_metric_direction (the old _label_signal_ladder does not exist). ***

Direction-corrected delta vs target (positive = arm BEATS target):
  lower_is_better (RMSE): delta = target - arm
  higher_is_better (rho): delta = arm - target
"""
import sys
import os
import json
import time
import traceback
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[3])
for _p in (os.path.join(_REPO, "tests"), os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments import _alloc_harness as H  # noqa: E402  (lightgbm-first guard)
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BUDGETS = [100, 500, 1000]
NS = [50, 100, 200, 300]
MODEL_SWEEP = ["lightgbm", "rf"]
SELECT_MODEL_FALLBACK = "lightgbm"
N_INNER = 3                 # inner group-aware CV folds for the train-only rank

# ---- Speed-only monkeypatch: force a sane n_jobs on returned models. -----
# n_jobs=-1 causes catastrophic OpenMP thread-thrash on these small-n datasets
# (4.0s vs 0.34s per lightgbm fit) while producing BYTE-IDENTICAL predictions
# (verified: rmse 0.6025 for n_jobs in {-1,1,4}). This changes NOTHING about the
# model's output — it is a pure wall-clock knob. We patch H.get_model so BOTH the
# harness's score_matrix (which references the module-global get_model) and our
# own _fit_predict pick it up. We do NOT edit find_best_hybrid.py / hybrid.py.
_orig_get_model = H.get_model
def _fast_get_model(name, task_type="regression", n_samples=None):
    m = _orig_get_model(name, task_type=task_type, n_samples=n_samples)
    if m is not None:
        try:
            if "n_jobs" in m.get_params():
                m.set_params(n_jobs=4)
        except Exception:
            pass
    return m
H.get_model = _fast_get_model
OUTDIR = os.environ.get(
    "EXPB_OUT", os.path.join(_REPO, "tests", "results", "complexity", "exp_b"))
os.makedirs(OUTDIR, exist_ok=True)


def _fit_predict(Xtr, ytr, Xval, model_name, task_type):
    sc = StandardScaler()
    Xtr = sc.fit_transform(H._clean(Xtr))
    Xval = sc.transform(H._clean(Xval))
    m = H.get_model(model_name, task_type=task_type, n_samples=len(ytr))
    if m is None:
        raise RuntimeError(f"get_model None for {model_name} n={len(ytr)}")
    m.fit(Xtr, ytr)
    return m.predict(Xval)


def _topk_idx(imp, k):
    k = int(min(k, len(imp)))
    if k <= 0:
        return np.array([], dtype=int)
    return np.argsort(-imp)[:k]


def _direction_delta(target, arm, lower):
    """positive = arm better than target."""
    return (target - arm) if lower else (arm - target)


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def direction_gate(ds):
    """Blocking per-dataset metric-direction check. Uses the committed harness
    gate: it builds good/bad predictors from a label ladder and RAISES if the
    metric does not move the way the registry claims."""
    return H.assert_metric_direction(verbose=True, datasets=[ds])[ds]


def run_dataset(ds):
    out = {"dataset": ds, "caveats": []}
    reps, labels, groups, mname, mfn, lower, task = H.load_full(ds)
    labels = np.asarray(labels, float)
    folds = H.make_folds(groups, labels, task, n=5)
    out["n"] = int(len(labels))
    out["metric"] = mname
    out["lower_is_better"] = bool(lower)
    out["n_reps_staged"] = len(reps)

    # ---- metric-direction gate (blocking) -----------------------------------
    out["direction_gate"] = direction_gate(ds)

    rep_names = sorted(reps.keys())
    gs = -1.0 if lower else 1.0          # +1 if higher is better

    # ---- PASS A: HELD-OUT single-rep scores under every model in the sweep.
    # These are the numbers the target is GRADED with. The whole-data ranking
    # off them is descriptive only — nothing is selected from it (2026-09-04).
    base = {}
    base_oof = {}
    for r in rep_names:
        s, pf, oof = H.score_matrix(reps[r], labels, folds, "lightgbm", mfn,
                                    lower, task_type=task)
        base[r] = {"mean": s, "per_fold": pf}
        base_oof[r] = oof
    ranked_whole = sorted(rep_names, key=lambda r: gs * base[r]["mean"],
                          reverse=True)
    out["whole_data_ranking"] = [(r, round(base[r]["mean"], 4))
                                 for r in ranked_whole]

    # held-out per-fold scores for every (rep, model) in the sweep — the grade
    # side of the per-fold target, whichever rep and model a fold picks.
    rep_model_pf, rep_model_oof = {}, {}
    models_ok = []
    for mdl in MODEL_SWEEP:
        try:
            for r in rep_names:
                s, pf, oof = H.score_matrix(reps[r], labels, folds, mdl, mfn,
                                            lower, task_type=task)
                rep_model_pf[(r, mdl)] = pf
                rep_model_oof[(r, mdl)] = oof
            models_ok.append(mdl)
        except Exception as e:
            out["caveats"].append(f"model {mdl} failed on {ds}: {e}")
    if not models_ok:
        raise RuntimeError(f"no usable model in {MODEL_SWEEP} on {ds}")

    # ---- PER-FOLD TRAIN-ONLY ranking: inner group-aware CV on each outer
    # fold's TRAIN rows. It picks that fold's rep ranking (hence best_rep,
    # pool5, pool10) and its comparator model. -------------------------------
    score_whole = {r: gs * base[r]["mean"] for r in rep_names}
    fold_train_score, rank_fallback = [], []
    fold_model = []
    for fi, (tr, va) in enumerate(folds):
        sub_groups, sub_y = groups[tr], labels[tr]
        n_inner = min(N_INNER, len(np.unique(sub_groups)))
        if n_inner < 2:
            fold_train_score.append(dict(score_whole))
            fold_model.append(models_ok[0])
            rank_fallback.append(fi)
            continue
        try:
            inner = H.make_folds(sub_groups, sub_y, task, n=n_inner)
            sc = {}
            for r in rep_names:
                m_native, _pf, _oof = H.score_matrix(
                    reps[r][tr], sub_y, inner, "lightgbm", mfn, lower,
                    task_type=task)
                sc[r] = gs * m_native
            fold_train_score.append(sc)
            # comparator model: best of the sweep on THIS fold's train rows,
            # measured on the rep those same rows put on top.
            top_r = max(rep_names, key=lambda r: sc[r] if np.isfinite(sc[r])
                        else -np.inf)
            m_sc = {}
            for mdl in models_ok:
                try:
                    ms, _pf, _oof = H.score_matrix(
                        reps[top_r][tr], sub_y, inner, mdl, mfn, lower,
                        task_type=task)
                    m_sc[mdl] = gs * ms
                except Exception:
                    continue
            fold_model.append(max(m_sc, key=m_sc.get) if m_sc else models_ok[0])
        except Exception as e:
            print(f"  [rank-fallback] fold {fi}: inner CV failed "
                  f"({type(e).__name__}: {e})", flush=True)
            fold_train_score.append(dict(score_whole))
            fold_model.append(models_ok[0])
            rank_fallback.append(fi)
    out["rank_fallback_folds"] = rank_fallback

    def _rank_key(i):
        def k(r):
            v = fold_train_score[i][r]
            return v if np.isfinite(v) else -np.inf
        return k

    fold_ranked = [sorted(rep_names, key=_rank_key(i), reverse=True)
                   for i in range(len(folds))]
    fold_pools = {"pool5": [fold_ranked[i][:5] for i in range(len(folds))],
                  "pool10": [fold_ranked[i][:10] for i in range(len(folds))]}
    fold_best_rep = [fold_ranked[i][0] for i in range(len(folds))]
    out["fold_ranking"] = {str(i): fold_ranked[i] for i in range(len(folds))}
    out["pool_per_fold"] = {p: {str(i): v[i] for i in range(len(folds))}
                            for p, v in fold_pools.items()}
    out["best_rep_per_fold"] = fold_best_rep
    out["target_model_per_fold"] = fold_model
    out["models_usable"] = models_ok

    # ---- TARGET: per fold, the train-picked (rep, model)'s HELD-OUT score ----
    target_pf = np.array([rep_model_pf[(fold_best_rep[i], fold_model[i])][i]
                          for i in range(len(folds))], float)
    target_oof = np.full(len(labels), np.nan)
    for i, (tr, va) in enumerate(folds):
        target_oof[va] = rep_model_oof[(fold_best_rep[i], fold_model[i])][va]
    # reproducibility: re-score every (rep, model) pair the folds picked, on the
    # SAME folds, and assert the per-fold values are identical.
    repro = True
    for pair in sorted(set(zip(fold_best_rep, fold_model))):
        _s2, pf2, _o2 = H.score_matrix(reps[pair[0]], labels, folds, pair[1], mfn,
                                       lower, task_type=task)
        repro &= bool(np.allclose(np.array(pf2), np.array(rep_model_pf[pair]),
                                  atol=1e-9))
    out["target_reproduces"] = bool(repro)
    assert repro, f"TARGET did not reproduce on {ds}"

    out["target_per_fold"] = [round(x, 4) for x in target_pf.tolist()]
    out["target_mean"] = round(float(target_pf.mean()), 4)
    out["whole_data_best_rep"] = ranked_whole[0]
    out["folds_agreeing_with_whole_data"] = int(
        sum(1 for r in fold_best_rep if r == ranked_whole[0]))

    # =========================================================================
    # PASS B: fold loop, compute all arms.  Accumulate per-fold metric + oof.
    # =========================================================================
    # config key -> {'pf': [per fold metric], 'oof': array, 'origins': Counter,
    #                'eff_feats': [per fold], 'eff_B': ...}
    from collections import defaultdict, Counter
    cfg = defaultdict(lambda: {"pf": [], "oof": np.full(len(labels), np.nan),
                               "origins": Counter(), "eff_feats": []})

    for fi, (tr, va) in enumerate(folds):
        ytr = labels[tr]
        target_model = fold_model[fi]      # this fold's train-picked comparator
        # per-rep train-only importance for this fold's pool10 (the superset).
        imp = {}
        for r in fold_pools["pool10"][fi]:
            imp[r] = H.rank_columns(reps[r][tr], ytr, method="lightgbm_gain")

        for pname in ("pool5", "pool10"):
            pool = fold_pools[pname][fi]
            K = len(pool)

            # ---------- ARM 1: per-rep quota ----------
            for B in BUDGETS:
                q = B // K
                rem = B - q * K
                sel_tr, sel_va, origins = [], [], []
                for i, r in enumerate(pool):
                    quota = q + (1 if i < rem else 0)
                    idx = _topk_idx(imp[r], quota)
                    if len(idx) == 0:
                        continue
                    sel_tr.append(reps[r][tr][:, idx])
                    sel_va.append(reps[r][va][:, idx])
                    origins += [r] * len(idx)
                Xtr = np.hstack(sel_tr); Xva = np.hstack(sel_va)
                pred = _fit_predict(Xtr, ytr, Xva, target_model, task)
                m = float(mfn(labels[va], pred))
                key = f"ARM1_quota|{pname}|B{B}"
                cfg[key]["pf"].append(m)
                cfg[key]["oof"][va] = pred
                cfg[key]["origins"].update(origins)
                cfg[key]["eff_feats"].append(Xtr.shape[1])

            # ---------- ARM 2: balanced global (top-N per rep, then global) ----------
            for N in NS:
                # balanced candidate pool: top-N per rep
                cand_tr, cand_va, cand_origin = [], [], []
                for r in pool:
                    idx = _topk_idx(imp[r], N)
                    if len(idx) == 0:
                        continue
                    cand_tr.append(reps[r][tr][:, idx])
                    cand_va.append(reps[r][va][:, idx])
                    cand_origin += [r] * len(idx)
                Ctr = np.hstack(cand_tr); Cva = np.hstack(cand_va)
                cand_origin = np.array(cand_origin)
                g_imp = H.rank_columns(Ctr, ytr, method="lightgbm_gain")
                for B in BUDGETS:
                    gidx = _topk_idx(g_imp, B)
                    Xtr = Ctr[:, gidx]; Xva = Cva[:, gidx]
                    pred = _fit_predict(Xtr, ytr, Xva, target_model, task)
                    m = float(mfn(labels[va], pred))
                    key = f"ARM2_balanced|{pname}|N{N}|B{B}"
                    cfg[key]["pf"].append(m)
                    cfg[key]["oof"][va] = pred
                    cfg[key]["origins"].update(cand_origin[gidx].tolist())
                    cfg[key]["eff_feats"].append(Xtr.shape[1])

            # ---------- ARM 3: naive global (all columns, no cap) ----------
            allc_tr, allc_va, allc_origin = [], [], []
            for r in pool:
                allc_tr.append(reps[r][tr])
                allc_va.append(reps[r][va])
                allc_origin += [r] * reps[r].shape[1]
            Atr = np.hstack(allc_tr); Ava = np.hstack(allc_va)
            allc_origin = np.array(allc_origin)
            g_imp = H.rank_columns(Atr, ytr, method="lightgbm_gain")
            for B in BUDGETS:
                gidx = _topk_idx(g_imp, B)
                Xtr = Atr[:, gidx]; Xva = Ava[:, gidx]
                pred = _fit_predict(Xtr, ytr, Xva, target_model, task)
                m = float(mfn(labels[va], pred))
                key = f"ARM3_naive|{pname}|B{B}"
                cfg[key]["pf"].append(m)
                cfg[key]["oof"][va] = pred
                cfg[key]["origins"].update(allc_origin[gidx].tolist())
                cfg[key]["eff_feats"].append(Xtr.shape[1])

        print(f"  [{ds}] fold {fi+1}/5 done  ({time.strftime('%H:%M:%S')})",
              flush=True)

    # ---- summarise every config: paired delta vs target, folds won ----------
    results = {}
    for key, d in cfg.items():
        arm_pf = np.array(d["pf"], float)
        deltas = _direction_delta(target_pf, arm_pf, lower)
        results[key] = {
            "arm_per_fold": [round(x, 4) for x in arm_pf.tolist()],
            "arm_mean": round(float(arm_pf.mean()), 4),
            "delta_per_fold": [round(x, 4) for x in deltas.tolist()],
            "delta_mean_of_folds": round(float(deltas.mean()), 4),
            "folds_won": int((deltas > 0).sum()),
            "n_folds": int(len(deltas)),
            "mean_features": round(float(np.mean(d["eff_feats"])), 1),
            "origins": dict(d["origins"].most_common()),
            "pred_corr_vs_target": round(_pearson(d["oof"], target_oof), 3),
        }
    out["results"] = results

    # ---- arm-vs-arm prediction correlation (representative: pool5, B=500) ----
    def _oof(key):
        return cfg[key]["oof"] if key in cfg else None
    rep_corr = {}
    a1 = _oof("ARM1_quota|pool5|B500")
    a2 = _oof("ARM2_balanced|pool5|N100|B500")
    a3 = _oof("ARM3_naive|pool5|B500")
    pairs = {"ARM1_vs_ARM2": (a1, a2), "ARM1_vs_ARM3": (a1, a3),
             "ARM2_vs_ARM3": (a2, a3), "ARM1_vs_TARGET": (a1, target_oof),
             "ARM2_vs_TARGET": (a2, target_oof), "ARM3_vs_TARGET": (a3, target_oof)}
    for k, (x, y) in pairs.items():
        rep_corr[k] = round(_pearson(x, y), 3) if x is not None and y is not None else None
    out["pred_corr_pool5_B500"] = rep_corr

    return out


def main():
    order = ["herg_ki", "flab_thermo_tresanco", "logd", "buchwald_hartwig"]
    if len(sys.argv) > 1:
        order = sys.argv[1:]
    allres = {}
    for ds in order:
        t0 = time.time()
        print(f"\n===== {ds} =====  ({time.strftime('%H:%M:%S')})", flush=True)
        try:
            allres[ds] = run_dataset(ds)
            allres[ds]["wall_s"] = round(time.time() - t0, 1)
        except Exception as e:
            allres[ds] = {"dataset": ds, "FAILED": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()}
            traceback.print_exc()
        # incremental dump
        with open(os.path.join(OUTDIR, f"expB_{ds}.json"), "w") as f:
            json.dump(allres[ds], f, indent=2, default=str)
        print(f"  wrote expB_{ds}.json  ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUTDIR, "expB_all.json"), "w") as f:
        json.dump(allres, f, indent=2, default=str)
    print("\nDONE all datasets", flush=True)


if __name__ == "__main__":
    main()
