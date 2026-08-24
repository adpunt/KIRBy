"""EXPERIMENT C32 — ALLOCATION STRATEGY CROSS (weighting x mechanism x N).

Question the earlier data could NOT answer: a "strategy" is a full point across
THREE independent axes —

  Axis A  N  : how many of the top reps get any budget  {2, 3, 5, 10(all)}
  Axis B  W  : how the budget splits among them (weighting)
               {linear, power_2, softmax_T1, rank_exp_d0.3}
  Axis C  M  : the mechanism
               quota  = each selected rep gets its OWN weighted per-rep budget
                        (create_hybrid 'performance_weighted' semantics)
               global = each rep offers a weighted candidate slice, the pool is
                        glued, ONE global ranking picks the final B (weighted-
                        global / c31 semantics)

All prior runs (f01/f12, c31) only ever moved ONE axis: they varied N with the
weighting pinned to linear (top_2/top_3/top_5), OR varied the weighting at N=all,
OR tested global with only linear/power_2/top_2.  The (weighting x N) interior of
the grid — power_2 x N=3, softmax x N=2, ... in EACH mechanism — was never run.
C32 fills it.

Faithfulness: this is NOT a substitute allocator.  The per-rep weighting comes
from the ENGINE'S OWN `kirby.hybrid._compute_allocation_weights` /
`_sanitize_baseline_scores`, driven with the exact call sequence
`allocate_performance_weighted` / `allocate_weighted_global` use.  The N axis is
just the pool membership (top-N reps); the quota arm applies the engine weights as
a per-rep budget; the global arm applies them as pool-slot sizing then one global
rank — byte-for-byte the two engine mechanisms.  Built on the committed c31/c30
harness (`_alloc_harness`); never edits or calls find_best_hybrid.py / hybrid.py.

Per (dataset x model x budget) CELL the output holds every strategy
`W|M|N` with its paired-per-fold delta vs TARGET (best single rep x model), R2 +
native, folds-won, and a group-block bootstrap CI — i.e. the per-full-strategy
win-count material directly.

Size control (fits the medium wall like c31): global's two sub-knobs are PINNED —
pool_total_mult = 2 and the global-ranking importance = lightgbm_gain (the
knob-sensitivity data says importance ~= noise).  Only the three real axes are
crossed: 4 N x 4 weighting x 2 mechanism = 32 strategies/cell.

Run:  python c32_allocation_strategy_cross.py [ds1 ds2 ...]
Env:  C32_OUT    output json path (SLURM: one file per (ds,model) task)
      C32_MODELS comma-list to restrict the model loop (SLURM array dimension)
      C32_BOOT   bootstrap resamples for the noise floor (0 disables; default 1000)
"""
import os
import sys
from pathlib import Path

# lightgbm-first import guard (macOS OpenMP segfault) lives inside _alloc_harness.
_REPO = str(Path(__file__).resolve().parents[3])
for _p in (os.path.join(_REPO, "tests"), os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments import _alloc_harness as H  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

# ENGINE weighting functions — READ-ONLY, never edited.  Guarantees the per-rep
# weighting is byte-for-byte what the allocation strategies do.
from kirby.hybrid import (  # noqa: E402
    create_hybrid, _compute_allocation_weights, _sanitize_baseline_scores,
)
from kirby.find_best_hybrid import get_model  # noqa: E402

# --------------------------------------------------------------------------- #
# GRID — the three real axes.  Edit here.
# --------------------------------------------------------------------------- #
N_VALUES = [2, 3, 5, 10]              # Axis A: top-N reps (10 == whole pool)
SCHEMES = {                          # Axis B: DISTINCT weighting shapes
    "linear":       ("linear",    {}),                     # flat spread
    "power_2":      ("power_law", {"power": 2.0}),         # concentrate hard
    "softmax_T1":   ("softmax",   {"temperature": 1.0}),   # concentrate soft
    "rank_exp_d0.3": ("rank_exp", {"decay": 0.3}),         # rank-based (ignores magnitude)
}
MECHANISMS = ["quota", "global"]     # Axis C

POOL_MAX = 10                        # candidate pool = top-10 staged reps
BUDGETS = [500, 1000]                # B (part of the CELL, not a strategy axis)
POOL_TOTAL_MULT = 2                  # global sub-knob, PINNED (c31 default)
IMP = "lightgbm_gain"                # per-rep candidate + global-ranking importance, PINNED

NFOLDS = 5
N_INNER = 3                          # inner group-aware CV folds for train-only pool sizing
N_BOOT = int(os.environ.get("C32_BOOT", "1000"))   # group-block bootstrap; 0 disables

_ALL_MODELS = ["lightgbm", "rf", "catboost"]
_env_models = os.environ.get("C32_MODELS", "").strip()
MODELS = [m.strip() for m in _env_models.split(",") if m.strip()] or _ALL_MODELS

DEFAULT_DATASETS = ["herg_ki", "logd", "flab_thermo_tresanco"]

CAVEATS = [
    "STRATEGY = (weighting W x mechanism M x N).  32 per cell (4 N x 4 W x 2 M). "
    "The CELL is (dataset x model x budget); TARGET is the best single staged rep "
    "x that model at full native width, re-derived per model.  All deltas are "
    "paired per fold vs that target (higher R2 delta = beats it).",
    "Pool-sizing weights are computed INSIDE each outer fold's TRAIN split via an "
    "inner group-aware CV (N_INNER folds), no whole-data leak.  A train split too "
    "small for the inner CV falls back to whole-data sizing (listed in "
    "size_fallback_folds).  N-subsetting reuses the same per-fold hib scores.",
    "quota mechanism: per-rep budget = round(w_i/sum(w) * B), each rep offers its "
    "top-that-many columns by its own lightgbm_gain (train-fold).  No global "
    "re-rank.  == allocate_performance_weighted.",
    "global mechanism: per-rep candidate slots = round(w_i/sum(w) * 2B), glue the "
    "candidate columns, ONE global lightgbm_gain rank, take top-B.  == "
    "allocate_weighted_global at pool_total_mult=2.",
    "PINNED to control size: pool_total_mult=2, global importance=lightgbm_gain. "
    "Restore a sweep of either by editing POOL_TOTAL_MULT / IMP.",
    "Only NFOLDS=5 paired folds -> per-cell CIs are wide; the deliverable is "
    "per-cell WIN-COUNTS across the 60 cells, NOT per-cell significance.",
    "At small N the weightings converge (2 reps split 2 ways ~ identical across "
    "schemes); that redundancy is expected and informative, not a bug.",
]


def gsign(lower_is_better):
    """+1 if higher metric is better, -1 if lower is better."""
    return -1.0 if lower_is_better else 1.0


def score_selection(reps, y, folds, per_fold_sel, task_type, metric_fn, model):
    """per_fold_sel[i] = {rep_name: np.array(col_indices)} chosen on fold i TRAIN.
    Fits `model` per fold on the selected (scaled) columns, predicts held-out.
    Returns (native_per_fold, r2_per_fold, oof_preds, mean_nfeat)."""
    oof = np.full(len(y), np.nan)
    native_pf, r2_pf, nfeat = [], [], []
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
            native_pf.append(float(metric_fn(y[va], p)))
            r2_pf.append(float(r2_score(y[va], p)))
            continue
        Xtr = np.hstack(parts_tr)
        Xva = np.hstack(parts_va)
        nfeat.append(Xtr.shape[1])
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xva = sc.transform(Xva)
        m = get_model(model, task_type=task_type, n_samples=len(tr))
        if m is None:
            raise RuntimeError(f"get_model returned None for {model} at n={len(tr)}")
        m.fit(Xtr, y[tr])
        p = m.predict(Xva)
        oof[va] = p
        native_pf.append(float(metric_fn(y[va], p)))
        r2_pf.append(float(r2_score(y[va], p)))
    return native_pf, r2_pf, oof, float(np.mean(nfeat))


def _validate_against_engine(reps, y, folds, pool5):
    """Prove the quota arm's per-rep top-k == create_hybrid('fixed') on fold 0."""
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


def _run_model(ds, model, reps, y, groups, folds, mfn, lower, task, gs, boot_ci):
    """Full model-dependent pipeline for ONE downstream model: standalone ranking
    -> target -> the 3-axis strategy cross."""
    mout = {"model": model}

    # -- standalone score for EVERY staged rep (full native width) -> rank ------
    rep_native, rep_r2, rep_oof, rep_width = {}, {}, {}, {}
    for rn in sorted(reps):
        W = reps[rn].shape[1]
        rep_width[rn] = W
        sel = [{rn: np.arange(W)} for _ in range(NFOLDS)]
        npf, r2pf, oof, _ = score_selection(reps, y, folds, sel, task, mfn, model)
        rep_native[rn] = (float(np.mean(npf)), npf)
        rep_r2[rn] = (float(np.mean(r2pf)), r2pf)
        rep_oof[rn] = oof

    ranked = sorted(rep_native, key=lambda r: gs * rep_native[r][0], reverse=True)
    P_full = ranked[:POOL_MAX]
    best_rep = ranked[0]
    target_native_pf = rep_native[best_rep][1]
    target_r2_pf = rep_r2[best_rep][1]
    target_oof = rep_oof[best_rep]
    tR2 = np.array(target_r2_pf)
    tNat = np.array(target_native_pf)

    mout["target"] = {
        "best_rep": best_rep, "width": rep_width[best_rep],
        "native_score_mean": rep_native[best_rep][0],
        "r2_score_mean": rep_r2[best_rep][0],
        "rep_ranking_top12": [(r, round(rep_native[r][0], 5), round(rep_r2[r][0], 5),
                               rep_width[r]) for r in ranked[:12]],
    }
    mout["pool_full"] = P_full
    mout["pool_widths"] = {rn: rep_width[rn] for rn in P_full}

    # -- per-fold, per-rep TRAIN candidate importances (which cols a rep offers) -
    rep_imp = {rn: [H.rank_columns(reps[rn][tr], y[tr], method=IMP)
                    for tr, _ in folds] for rn in P_full}

    def topk_local(rn, fold_i, k):
        imp = rep_imp[rn][fold_i]
        k = min(int(k), len(imp))
        if k <= 0:
            return np.array([], dtype=int)
        return np.argsort(imp)[-k:][::-1]

    # -- per-fold TRAIN-only pool-sizing scores (hib), computed once over P_full -
    hib_whole = _sanitize_baseline_scores({rn: gs * rep_native[rn][0] for rn in P_full})
    fold_hib, fold_pool_train, size_fallback = [], [], []
    for i, (tr, va) in enumerate(folds):
        sub_groups, sub_y = groups[tr], y[tr]
        n_inner = min(N_INNER, len(np.unique(sub_groups)))
        train_mats = {rn: H._clean(reps[rn][tr]) for rn in P_full}
        fold_pool_train.append(train_mats)
        if n_inner < 2:
            fold_hib.append(dict(hib_whole))
            size_fallback.append(i)
            continue
        try:
            inner = H.make_folds(sub_groups, sub_y, task, n=n_inner)
            sc = {}
            for rn in P_full:
                m_native, _pf, _oof = H.score_matrix(
                    train_mats[rn], sub_y, inner, model, mfn, lower, task)
                sc[rn] = m_native
            fold_hib.append(_sanitize_baseline_scores({rn: gs * sc[rn] for rn in P_full}))
        except Exception as e:
            print(f"  [size-fallback] fold {i}: inner CV failed ({type(e).__name__}: {e})")
            fold_hib.append(dict(hib_whole))
            size_fallback.append(i)
    mout["size_fallback_folds"] = size_fallback

    def summarize(cfg):
        r2 = np.array(cfg["r2_pf"])
        nat = np.array(cfg["native_pf"])
        dR2 = r2 - tR2
        dNat = gs * (nat - tNat)
        ci95, p_boot = boot_ci(cfg.get("oof"), target_oof)
        return {
            "r2_mean": float(np.mean(r2)),
            "r2_delta_mean": float(np.mean(dR2)),
            "r2_delta_per_fold": [round(float(x), 5) for x in dR2],
            "r2_folds_won": int(np.sum(dR2 > 0)),
            "r2_delta_ci95": ci95,
            "r2_delta_p_boot": p_boot,
            "native_delta_mean": float(np.mean(dNat)),
            "native_folds_won": int(np.sum(dNat > 0)),
            "nfeat": round(cfg["nfeat"], 1),
            "scheme": cfg["scheme"], "mech": cfg["mech"], "N": cfg["N"], "B": cfg["B"],
        }

    def global_sel(pool, slots_by_fold, B):
        """Weighted-global: glue each rep's top-slots candidate cols, one global
        rank per fold, take top-B.  Returns (per_fold_sel, origin_counts)."""
        per_fold_sel, origin_counts = [], {}
        for i, (tr, va) in enumerate(folds):
            origins, mats = [], []
            for rn in pool:
                idx = topk_local(rn, i, slots_by_fold[i][rn])
                if len(idx) == 0:
                    continue
                mats.append(H._clean(reps[rn][tr][:, idx]))
                origins.extend([(rn, int(c)) for c in idx])
            if not mats:
                per_fold_sel.append({})
                continue
            gimp = H.rank_columns(np.hstack(mats), y[tr], method=IMP)
            k = min(B, len(gimp))
            top = np.argsort(gimp)[-k:][::-1]
            sel = {}
            for t in top:
                rn, c = origins[t]
                sel.setdefault(rn, []).append(c)
                origin_counts[rn] = origin_counts.get(rn, 0) + 1
            per_fold_sel.append({rn: np.array(v) for rn, v in sel.items()})
        return per_fold_sel, origin_counts

    # ======================================================================
    # THE CROSS: N x weighting x mechanism, keyed under budget (the cell).
    # ======================================================================
    strategies = {f"B{B}": {} for B in BUDGETS}
    n_used = sorted({min(N, len(P_full)) for N in N_VALUES})
    for N in n_used:
        P = ranked[:N]
        for label, (scheme, params) in SCHEMES.items():
            # per-fold weights over the N-subset (engine function)
            wl = []
            for i in range(NFOLDS):
                hib_i = {rn: fold_hib[i][rn] for rn in P}
                w = _compute_allocation_weights(
                    hib_i, scheme=scheme, params=params,
                    rep_dict_train={rn: fold_pool_train[i][rn] for rn in P})
                wl.append({rn: float(w.get(rn, 0.0)) for rn in P})

            for B in BUDGETS:
                # -- QUOTA: per-rep weighted budget --------------------------
                per_fold_sel = []
                for i in range(NFOLDS):
                    w = wl[i]
                    tot = sum(w.values())
                    sel = {}
                    for rn in P:
                        q = int(round((w[rn] / tot) * B)) if tot > 0 else 0
                        q = min(max(q, 0), rep_width[rn])
                        sel[rn] = topk_local(rn, i, q)
                    per_fold_sel.append(sel)
                npf, r2pf, oof, nf = score_selection(
                    reps, y, folds, per_fold_sel, task, mfn, model)
                strategies[f"B{B}"][f"{label}|quota|N{N}"] = summarize({
                    "native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                    "scheme": label, "mech": "quota", "N": N, "B": B})

                # -- GLOBAL: weighted pool sizing + one global rank ----------
                POOL_TOTAL = POOL_TOTAL_MULT * B
                slots_by_fold = {}
                for i in range(NFOLDS):
                    w = wl[i]
                    tot = sum(w.values())
                    slots = {}
                    for rn in P:
                        s = int(round((w[rn] / tot) * POOL_TOTAL)) if tot > 0 else 0
                        slots[rn] = min(max(s, 0), rep_width[rn])
                    slots_by_fold[i] = slots
                per_fold_sel_g, oc = global_sel(P, slots_by_fold, B)
                npf, r2pf, oof, nf = score_selection(
                    reps, y, folds, per_fold_sel_g, task, mfn, model)
                s = summarize({
                    "native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                    "scheme": label, "mech": "global", "N": N, "B": B})
                s["origin_counts"] = oc
                strategies[f"B{B}"][f"{label}|global|N{N}"] = s

    mout["strategies"] = strategies
    mout["engine_validation"] = _validate_against_engine(reps, y, folds, P_full[:5])
    print(f"  [{model}] target={best_rep} r2={rep_r2[best_rep][0]:.4f} "
          f"strategies/cell={len(SCHEMES) * len(MECHANISMS) * len(n_used)}")
    return mout


def run_dataset(ds):
    t0 = time.time()
    print("\n" + "=" * 70)
    print(f"DATASET: {ds}")
    print("=" * 70)
    out = {"dataset": ds}

    reps, labels, groups, mname, mfn, lower, task = H.load_full(ds)
    folds = H.make_folds(groups, labels, task, n=NFOLDS)
    y = np.asarray(labels, dtype=float)
    gs = gsign(lower)
    out.update({"n": int(len(y)), "metric": mname, "lower_is_better": bool(lower),
                "n_reps_staged": len(reps), "n_folds": NFOLDS})

    # -- group-block bootstrap machinery (shared across models) -----------------
    uniq_groups = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq_groups}
    ug_list = list(uniq_groups)

    def boot_ci(arm_oof, tgt_oof):
        if N_BOOT <= 0 or arm_oof is None or tgt_oof is None:
            return [None, None], None
        arm_oof = np.asarray(arm_oof, dtype=float)
        tgt_oof = np.asarray(tgt_oof, dtype=float)
        valid = ~np.isnan(arm_oof) & ~np.isnan(tgt_oof)
        rng = np.random.default_rng(12345)
        deltas = []
        n_g = len(ug_list)
        for _ in range(N_BOOT):
            pick = rng.integers(0, n_g, size=n_g)
            idx = np.concatenate([gidx[ug_list[j]] for j in pick])
            idx = idx[valid[idx]]
            if len(idx) < 3:
                continue
            yy = y[idx]
            if np.ptp(yy) == 0:
                continue
            try:
                d = r2_score(yy, arm_oof[idx]) - r2_score(yy, tgt_oof[idx])
            except Exception:
                continue
            deltas.append(d)
        if not deltas:
            return [None, None], None
        deltas = np.asarray(deltas)
        return ([round(float(np.percentile(deltas, 2.5)), 5),
                 round(float(np.percentile(deltas, 97.5)), 5)],
                round(float(np.mean(deltas <= 0)), 4))

    max_train_n = max(len(tr) for tr, _ in folds)
    out["by_model"] = {}
    for model in MODELS:
        print(f"\n----- MODEL: {model} -----")
        if get_model(model, task_type=task, n_samples=max_train_n) is None:
            print(f"  [skip] {model}: max_train_n={max_train_n} exceeds _max_samples guard")
            out["by_model"][model] = {"skipped": f"{model}_max_samples",
                                      "max_train_n": int(max_train_n)}
            continue
        out["by_model"][model] = _run_model(
            ds, model, reps, y, groups, folds, mfn, lower, task, gs, boot_ci)

    out["wall_time_s"] = round(time.time() - t0, 1)
    print(f"[{ds}] done in {out['wall_time_s']}s  models={list(out['by_model'])}")
    return out


def main(datasets):
    t_run = time.time()
    _default_out = os.path.join(_REPO, "tests", "results", "complexity",
                                "c32_allocation_strategy_cross.json")
    respath = Path(os.environ.get("C32_OUT", _default_out))
    respath.parent.mkdir(parents=True, exist_ok=True)

    all_out = None
    if respath.exists():
        try:
            with open(respath) as f:
                all_out = json.load(f)
            all_out.pop("complete", None)
        except Exception:
            all_out = None

    gate_ok = bool(all_out and all_out.get("gate"))
    if not gate_ok:
        print("=" * 70)
        print("METRIC-DIRECTION GATE (blocking)")
        print("=" * 70)
        _gate_dsets = sorted(set(H.EXPECTED_DIRECTION) | set(datasets))
        gate = H.assert_metric_direction(verbose=True, datasets=_gate_dsets)
        gate_rec = {k: {"direction": v["direction"], "monotone": v["monotone"],
                        "metric": v["metric"]} for k, v in gate.items()}
    else:
        print("[gate] already PASSED in existing checkpoint — skipping re-run.")
        gate_rec = all_out["gate"]

    if all_out is None:
        all_out = {"datasets": {}}
    all_out["gate"] = gate_rec
    all_out["config"] = {"models": MODELS, "importance": IMP, "pool_max": POOL_MAX,
                         "budgets": BUDGETS, "n_values": N_VALUES,
                         "schemes": list(SCHEMES), "mechanisms": MECHANISMS,
                         "pool_total_mult": POOL_TOTAL_MULT, "n_folds": NFOLDS,
                         "n_inner_sizing_folds": N_INNER, "n_boot": N_BOOT,
                         "weighting_source": "kirby.hybrid._compute_allocation_weights"}
    all_out["caveats"] = CAVEATS
    all_out.setdefault("datasets", {})

    def checkpoint():
        all_out["wall_time_s"] = round(time.time() - t_run, 1)
        tmp = respath.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(all_out, f, indent=2, default=str)
        tmp.replace(respath)

    for ds in datasets:
        prior = all_out["datasets"].get(ds)
        if prior and "error" not in prior and prior.get("by_model"):
            print(f"[skip] {ds} already complete in checkpoint.")
            continue
        try:
            all_out["datasets"][ds] = run_dataset(ds)
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
    dsets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DATASETS
    main(dsets)
