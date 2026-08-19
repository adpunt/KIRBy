"""EXPERIMENT C31 — PERFORMANCE-WEIGHTED global feature selection (FULL sweep).

Question: today's BALANCED global selection caps EACH rep at an EQUAL top-N,
concatenates across the pool's reps, globally ranks all pooled columns by
lightgbm_gain, and takes the top-B budget. Does sizing each rep's candidate
contribution by a PERFORMANCE WEIGHT (stronger reps offer MORE candidate
columns) beat the EQUAL balanced global — and WHICH weighting family sizes the
global pool best?

*** FIX 2026-08-11: the prior run only tested TWO weighting families
(performance_weighted / power_2).  It now sweeps the FULL weighting system the
allocation strategies expose, and — critically — it no longer reimplements the
weighting locally.  It imports the ENGINE'S OWN `_compute_allocation_weights`
and `_sanitize_baseline_scores` from src/kirby/hybrid.py and drives them with
the exact call sequence `allocate_performance_weighted` uses (sanitize the
higher-is-better per-rep scores, then `_compute_allocation_weights(scheme,
params)`).  The pool-sizing weighting is therefore GUARANTEED identical to the
allocation strategies. ***

Design (implemented exactly):
  1. Score each rep standalone (grouped CV, full native width) -> per-rep score.
     Put in higher-is-better orientation (gs * native), then sanitize exactly
     as the engine does (`_sanitize_baseline_scores`: shift to non-negative,
     non-finite -> 0), matching allocate_performance_weighted's precomputed_
     scores path.
  2. weights = _compute_allocation_weights(sanitized, scheme, params, rep_dict
     _train=pool_matrices) — the ENGINE function, for the full scheme set:
       linear (performance_weighted); power_law power=2, power=3;
       softmax T in {0.5,1,2}; rank_linear; rank_exp decay in {0.3,0.5};
       thresholded alpha=0.5; score_gap gap=0.5; entropy diversity_penalty in
       {1.0,2.0}; top_k k=3 (optional hard concentrator, disclosed).
  3. pool_slots_i = round(w_i/sum(w) * POOL_TOTAL), capped at rep width.
     POOL_TOTAL swept over {2*B, 4*B}.
  4. Each rep offers its top pool_slots_i columns by ITS OWN lightgbm_gain
     (train-fold only); concatenate -> weighted candidate pool.
  5. ONE global lightgbm_gain ranking over that weighted pool (train-fold only);
     take top-B; score with the downstream model.

ARMS compared, paired per-fold, at matched budget B (all fold-safe):
  TARGET : best single staged rep x downstream model (full native width).
  WEIGHTED-GLOBAL : the variant above (each scheme x POOL_TOTAL).
  ARM1 quota : per-rep equal fixed quota B/|pool| feats/rep (create_hybrid
               'fixed' behaviour), verified against the engine on one fold.
  ARM2 equal-global : EQUAL top-N per rep (N = POOL_TOTAL/|pool|), concat,
               global gain top-B — the equal-weight analogue at matched
               POOL_TOTAL (c30's balanced global).
  ARM3 naive : glue the whole pool, no cap, global gain top-B (density control).

PRIMARY metric everywhere = R-squared (higher better; no sign flip), OOF-style
per fold.  SECONDARY = the dataset's native metric (herg_ki/logd/buchwald RMSE
lower better; flab Spearman higher better) — direction-corrected.

HARD RULES honoured: never edits find_best_hybrid.py / hybrid.py (import-only);
never calls find_best_hybrid(); full rows (row-count asserted in harness);
lightgbm_gain importance FIXED; downstream model FIXED across ALL arms + target;
repo cv_split GroupKFold; per-fold train-only selection & ranking.

Run:  python c31_perf_weighted_global_selection.py [ds1 ds2 ...]
      (default: herg_ki logd flab_thermo_tresanco; buchwald_hartwig if listed)
"""
import os
import sys
from pathlib import Path

# Portable path bootstrap: locate the repo root from this file (no hardcoded
# absolute path) and put tests/ on sys.path so `experiments.*` imports resolve
# whether run locally or on the server (where setup.sh also sets PYTHONPATH).
_REPO = str(Path(__file__).resolve().parents[3])
for _p in (os.path.join(_REPO, "tests"), os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Shared allocation harness — now a committed repo module (imports lightgbm
# FIRST for the macOS segfault guard). Replaces the old /private/tmp scratchpad
# harness so this experiment runs unchanged on the server.
from experiments import _alloc_harness as H  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

# ENGINE weighting functions — imported READ-ONLY, never edited.  Using these
# guarantees the pool-sizing weighting is byte-for-byte what the allocation
# strategies do.
from kirby.hybrid import (  # noqa: E402
    create_hybrid, _compute_allocation_weights, _sanitize_baseline_scores,
)
from kirby.find_best_hybrid import get_model  # noqa: E402

MODEL = "lightgbm"                # downstream model, FIXED across all arms
IMP = "lightgbm_gain"            # importance method, FIXED
POOL = 10                        # candidate pool = top-10 staged reps
BUDGETS = [500, 1000]            # B
POOL_TOTAL_MULTS = [2, 4]        # POOL_TOTAL in {2B, 4B}
NFOLDS = 5
DEFAULT_DATASETS = ["herg_ki", "logd", "flab_thermo_tresanco"]

# FULL weighting-family set, mapped 1:1 to NAMED_ALLOCATIONS' (scheme, params).
# label -> (engine_scheme, params).  'linear' == performance_weighted.
SCHEMES = {
    "linear":        ("linear",      {}),
    "power_2":       ("power_law",   {"power": 2.0}),
    "power_3":       ("power_law",   {"power": 3.0}),
    "softmax_T0.5":  ("softmax",     {"temperature": 0.5}),
    "softmax_T1":    ("softmax",     {"temperature": 1.0}),
    "softmax_T2":    ("softmax",     {"temperature": 2.0}),
    "rank_linear":   ("rank_linear", {}),
    "rank_exp_d0.3": ("rank_exp",    {"decay": 0.3}),
    "rank_exp_d0.5": ("rank_exp",    {"decay": 0.5}),
    "thresh_0.5":    ("thresholded", {"alpha": 0.5}),
    "score_gap_0.5": ("score_gap",   {"gap": 0.5}),
    "entropy_1.0":   ("entropy",     {"diversity_penalty": 1.0}),
    "entropy_2.0":   ("entropy",     {"diversity_penalty": 2.0}),
    # optional HARD concentrator (produces zeros for all but top-k reps) — cheap,
    # included and disclosed:
    "top_3":         ("top_k",       {"k": 3}),
}

CAVEATS = [
    "Pool-sizing weights are computed ONCE per dataset from WHOLE-DATA standalone "
    "rep scores (not per-fold). This mirrors the existing arm and only sizes each "
    "rep's CANDIDATE budget; the FINAL top-B selection is per-fold train-only, so "
    "the target-vs-arm comparison stays fold-honest, but a rep's candidate slot "
    "count is not itself cross-validated.",
    "The 'entropy' scheme's redundancy penalty is measured on the WHOLE-DATA pool "
    "matrices (engine _compute_allocation_weights was handed the full pool reps), "
    "again a pool-sizing quantity only.",
    "'top_3' is a HARD concentrator (top_k weighting zeroes all but 3 reps). "
    "Included as a concentrator reference; not a smooth per-rep weighting.",
    "Downstream model is FIXED to lightgbm across target + all arms so the "
    "measured delta isolates the SELECTION method (project memory: model swing "
    "~0.27 dominates and would otherwise confound).",
    "R-squared is computed per held-out fold on that fold's predictions (an OOF-"
    "per-fold R2), then averaged/paired vs target — it is NOT a single pooled-OOF "
    "R2. Native secondary metric handled identically.",
]


def gsign(lower_is_better):
    """+1 if higher metric is better, -1 if lower is better."""
    return -1.0 if lower_is_better else 1.0


# ---------------------------------------------------------------------------
# fold-safe scoring for an arbitrary per-fold column selection.
# Returns BOTH R-squared (primary) and native-metric per-fold values.
# ---------------------------------------------------------------------------
def score_selection(reps, y, folds, per_fold_sel, task_type, metric_fn):
    """per_fold_sel[i] = {rep_name: np.array(col_indices)} chosen on fold i TRAIN.

    Fits the downstream model per fold on the selected columns (scaled),
    predicts the held-out fold.
    Returns (native_per_fold, r2_per_fold, oof_preds, mean_nfeat).
    """
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
            # degenerate selection (e.g. all slots zero) — predict train mean
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
        m = get_model(MODEL, task_type=task_type, n_samples=len(tr))
        if m is None:
            raise RuntimeError(f"get_model returned None for {MODEL} at n={len(tr)}")
        m.fit(Xtr, y[tr])
        p = m.predict(Xva)
        oof[va] = p
        native_pf.append(float(metric_fn(y[va], p)))
        r2_pf.append(float(r2_score(y[va], p)))
    return native_pf, r2_pf, oof, float(np.mean(nfeat))


def main(datasets):
    t_run = time.time()
    # Output path: env-overridable (C31_OUT) so a local run can append to an
    # existing checkpoint and a server run can point at its own results dir.
    # Default: repo results tree (created if absent).
    _default_out = os.path.join(_REPO, "tests", "results", "complexity",
                                "c31_perf_weighted_global.json")
    respath = Path(os.environ.get("C31_OUT", _default_out))
    respath.parent.mkdir(parents=True, exist_ok=True)

    # -- RESUME: merge with any existing checkpoint so per-dataset invocations
    # accumulate (a killed run keeps completed datasets). ----------------------
    all_out = None
    if respath.exists():
        try:
            with open(respath) as f:
                all_out = json.load(f)
            all_out.pop("complete", None)
        except Exception:
            all_out = None

    gate_ok = bool(all_out and all_out.get("gate"))

    # -- BLOCKING GATE: prove metric direction end-to-end (both directions).
    # Skip only if a prior run in this same JSON already recorded a PASS. ------
    if not gate_ok:
        print("=" * 70)
        print("METRIC-DIRECTION GATE (blocking)")
        print("=" * 70)
        # Gate the local anchors AND every dataset this invocation runs, so a
        # server task on a non-anchor dataset (qm9, adme_*, caco2, ...) still
        # verifies its own metric direction. Loads no rep matrices.
        _gate_dsets = sorted(set(H.EXPECTED_DIRECTION) | set(datasets))
        gate = H.assert_metric_direction(verbose=True, datasets=_gate_dsets)  # raises on failure
        gate_rec = {k: {"direction": v["direction"], "monotone": v["monotone"],
                        "metric": v["metric"]} for k, v in gate.items()}
    else:
        print("[gate] already PASSED in existing checkpoint — skipping re-run.")
        gate_rec = all_out["gate"]

    if all_out is None:
        all_out = {"datasets": {}}
    all_out["gate"] = gate_rec
    all_out["config"] = {"model": MODEL, "importance": IMP, "pool": POOL,
                         "budgets": BUDGETS, "pool_total_mults": POOL_TOTAL_MULTS,
                         "schemes": list(SCHEMES), "n_folds": NFOLDS,
                         "weighting_source": "kirby.hybrid._compute_allocation_weights"}
    all_out["caveats"] = CAVEATS
    all_out.setdefault("datasets", {})

    def checkpoint():
        all_out["wall_time_s"] = round(time.time() - t_run, 1)
        # atomic-ish: write to tmp then replace, so a kill never truncates it
        tmp = respath.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(all_out, f, indent=2, default=str)
        tmp.replace(respath)

    # per-dataset CHECKPOINT: persist after EVERY dataset so a kill mid-run
    # never discards completed datasets.
    for ds in datasets:
        prior = all_out["datasets"].get(ds)
        if prior and "error" not in prior and "weighted_global" in prior:
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

    # -- standalone score for EVERY staged rep (full native width) -> rank -----
    rep_native, rep_r2, rep_oof, rep_width = {}, {}, {}, {}
    for rn in sorted(reps):
        W = reps[rn].shape[1]
        rep_width[rn] = W
        sel = [{rn: np.arange(W)} for _ in range(NFOLDS)]
        npf, r2pf, oof, _ = score_selection(reps, y, folds, sel, task, mfn)
        rep_native[rn] = (float(np.mean(npf)), npf)
        rep_r2[rn] = (float(np.mean(r2pf)), r2pf)
        rep_oof[rn] = oof

    ranked = sorted(rep_native, key=lambda r: gs * rep_native[r][0], reverse=True)
    P = ranked[:POOL]
    best_rep = ranked[0]
    target_native_pf = rep_native[best_rep][1]
    target_r2_pf = rep_r2[best_rep][1]
    target_oof = rep_oof[best_rep]

    out["target"] = {
        "best_rep": best_rep, "width": rep_width[best_rep],
        "native_score_mean": rep_native[best_rep][0],
        "native_per_fold": [round(x, 5) for x in target_native_pf],
        "r2_score_mean": rep_r2[best_rep][0],
        "r2_per_fold": [round(x, 5) for x in target_r2_pf],
        "rep_ranking_top12": [(r, round(rep_native[r][0], 5), round(rep_r2[r][0], 5),
                               rep_width[r]) for r in ranked[:12]],
    }
    out["pool"] = P
    out["pool_widths"] = {rn: rep_width[rn] for rn in P}

    # -- higher-is-better, engine-sanitized standalone scores over the POOL -----
    # mirror allocate_performance_weighted's precomputed_scores path exactly.
    hib = {rn: gs * rep_native[rn][0] for rn in P}
    hib_shift = _sanitize_baseline_scores(hib)           # ENGINE function
    # entropy needs the raw pool matrices (whole-data, pool-sizing only)
    pool_matrices = {rn: H._clean(reps[rn]) for rn in P}
    out["pool_standalone"] = {rn: {"native": round(rep_native[rn][0], 5),
                                   "r2": round(rep_r2[rn][0], 5),
                                   "hib_shifted": round(hib_shift[rn], 5),
                                   "width": rep_width[rn]} for rn in P}

    # -- per-fold, per-rep TRAIN lightgbm_gain importances (reused everywhere) --
    rep_imp = {rn: [] for rn in P}
    for i, (tr, va) in enumerate(folds):
        for rn in P:
            rep_imp[rn].append(H.rank_columns(reps[rn][tr], y[tr], method=IMP))

    def topk_local(rn, fold_i, k):
        imp = rep_imp[rn][fold_i]
        k = min(int(k), len(imp))
        if k <= 0:
            return np.array([], dtype=int)
        return np.argsort(imp)[-k:][::-1]

    # build the per-fold GLOBAL ranking over a candidate pool (expensive) — this
    # is memoized on the slots signature so identical candidate pools (e.g. the
    # same POOL_TOTAL reached via B=500,mult=4 and B=1000,mult=2, or two schemes
    # that yield identical slots) are ranked once.
    _rank_cache = {}

    def build_fold_global(slots):
        sig = tuple(sorted((rn, int(slots[rn])) for rn in P))
        if sig in _rank_cache:
            return _rank_cache[sig]
        fold_global = []
        for i in range(NFOLDS):
            tr, va = folds[i]
            origins, mats = [], []
            for rn in P:
                idx = topk_local(rn, i, slots[rn])
                if len(idx) == 0:
                    continue
                mats.append(H._clean(reps[rn][tr][:, idx]))
                origins.extend([(rn, int(c)) for c in idx])
            if not mats:
                fold_global.append(([], np.array([])))
                continue
            glued = np.hstack(mats)
            gimp = H.rank_columns(glued, y[tr], method=IMP)
            fold_global.append((origins, gimp))
        _rank_cache[sig] = fold_global
        return fold_global

    def topB_from_global(fold_global, B):
        per_fold_sel, origin_counts = [], {}
        for i in range(NFOLDS):
            origins, gimp = fold_global[i]
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

    # ======================================================================
    # WEIGHTED-GLOBAL variant — FULL scheme sweep x POOL_TOTAL x budget
    # ======================================================================
    weighted = {}
    scheme_weights = {}   # label -> {rep: weight} (for reporting)
    for label, (scheme, params) in SCHEMES.items():
        # ENGINE weighting — identical to the allocation strategies.
        w = _compute_allocation_weights(hib_shift, scheme=scheme, params=params,
                                        rep_dict_train=pool_matrices)
        scheme_weights[label] = {rn: round(float(w.get(rn, 0.0)), 6) for rn in P}
        total_w = sum(w.values())
        for B in BUDGETS:
            for mult in POOL_TOTAL_MULTS:
                POOL_TOTAL = mult * B
                slots = {}
                for rn in P:
                    s = int(round((w[rn] / total_w) * POOL_TOTAL)) if total_w > 0 else 0
                    slots[rn] = min(max(s, 0), rep_width[rn])
                fold_global = build_fold_global(slots)
                per_fold_sel, oc = topB_from_global(fold_global, B)
                npf, r2pf, oof, nf = score_selection(reps, y, folds, per_fold_sel,
                                                     task, mfn)
                key = f"{label}_B{B}_PT{mult}xB"
                weighted[key] = {
                    "native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                    "pool_total": POOL_TOTAL, "pool_slots": slots,
                    "candidate_pool_size": int(sum(min(slots[rn], rep_width[rn])
                                                    for rn in P)),
                    "origin_counts": oc, "scheme": label, "B": B, "mult": mult}

    out["scheme_weights"] = scheme_weights

    # ======================================================================
    # ARM1 — per-rep equal quota (create_hybrid 'fixed' behaviour)
    # ======================================================================
    arm1 = {}
    for B in BUDGETS:
        q = max(1, B // POOL)
        per_fold_sel = [{rn: topk_local(rn, i, q) for rn in P} for i in range(NFOLDS)]
        npf, r2pf, oof, nf = score_selection(reps, y, folds, per_fold_sel, task, mfn)
        arm1[f"B{B}"] = {"native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                         "quota_per_rep": q}

    # ======================================================================
    # ARM2 — EQUAL balanced global (equal top-N per rep at matched POOL_TOTAL)
    # ======================================================================
    arm2 = {}
    for B in BUDGETS:
        for mult in POOL_TOTAL_MULTS:
            POOL_TOTAL = mult * B
            N = max(1, int(round(POOL_TOTAL / POOL)))     # equal cap per rep
            slots = {rn: min(N, rep_width[rn]) for rn in P}
            fold_global = build_fold_global(slots)        # reuses memo
            per_fold_sel, oc = topB_from_global(fold_global, B)
            npf, r2pf, oof, nf = score_selection(reps, y, folds, per_fold_sel, task, mfn)
            arm2[f"B{B}_PT{mult}xB"] = {
                "native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                "equal_cap_N": N, "pool_total": POOL_TOTAL, "origin_counts": oc}

    # ======================================================================
    # ARM3 — naive global (whole pool, no cap, global gain top-B) density ctrl
    # ======================================================================
    arm3 = {}
    slots_full = {rn: rep_width[rn] for rn in P}
    fold_global_full = build_fold_global(slots_full)
    for B in BUDGETS:
        per_fold_sel, oc = topB_from_global(fold_global_full, B)
        npf, r2pf, oof, nf = score_selection(reps, y, folds, per_fold_sel, task, mfn)
        arm3[f"B{B}"] = {"native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                         "origin_counts": oc,
                         "total_pool_width": int(sum(rep_width[rn] for rn in P))}

    # ======================================================================
    # ENGINE VALIDATION — ARM1 manual quota == create_hybrid('fixed')
    # ======================================================================
    out["engine_validation"] = _validate_against_engine(reps, y, folds, P[:5])

    # ======================================================================
    # SUMMARISE — paired per-fold deltas vs TARGET
    #   PRIMARY  : R-squared (higher better, no sign flip)
    #   SECONDARY: native metric (direction-corrected via gs)
    # ======================================================================
    tR2 = np.array(target_r2_pf)
    tNat = np.array(target_native_pf)

    def summarize(cfg):
        r2 = np.array(cfg["r2_pf"])
        nat = np.array(cfg["native_pf"])
        dR2 = r2 - tR2                       # +ve = beats target on R2
        dNat = gs * (nat - tNat)             # +ve = beats target on native
        s = {
            "r2_mean": float(np.mean(r2)),
            "r2_delta_mean": float(np.mean(dR2)),
            "r2_delta_per_fold": [round(float(x), 5) for x in dR2],
            "r2_folds_won": int(np.sum(dR2 > 0)),
            "native_mean": float(np.mean(nat)),
            "native_delta_mean": float(np.mean(dNat)),
            "native_folds_won": int(np.sum(dNat > 0)),
            "nfeat": round(cfg["nfeat"], 1),
        }
        for k in ("origin_counts", "pool_slots", "pool_total",
                  "candidate_pool_size", "equal_cap_N", "quota_per_rep",
                  "total_pool_width", "scheme", "B", "mult"):
            if k in cfg:
                s[k] = cfg[k]
        return s

    out["weighted_global"] = {k: summarize(v) for k, v in weighted.items()}
    out["arm1_quota"] = {k: summarize(v) for k, v in arm1.items()}
    out["arm2_equal_global"] = {k: summarize(v) for k, v in arm2.items()}
    out["arm3_naive"] = {k: summarize(v) for k, v in arm3.items()}

    # ---- BEST SCHEME per (POOL_TOTAL,budget) by R2 delta -------------------
    best_by_cell = {}
    for B in BUDGETS:
        for mult in POOL_TOTAL_MULTS:
            cands = {lbl: out["weighted_global"][f"{lbl}_B{B}_PT{mult}xB"]
                     for lbl in SCHEMES}
            best_lbl = max(cands, key=lambda l: cands[l]["r2_delta_mean"])
            best_by_cell[f"B{B}_PT{mult}xB"] = {
                "best_scheme": best_lbl,
                "r2_delta_mean": round(cands[best_lbl]["r2_delta_mean"], 5),
                "r2_folds_won": cands[best_lbl]["r2_folds_won"],
                "native_delta_mean": round(cands[best_lbl]["native_delta_mean"], 5),
            }
    out["best_scheme_by_cell"] = best_by_cell

    # ---- DENSITY-BIAS CHECK ------------------------------------------------
    widest = max(P, key=lambda rn: rep_width[rn])
    rank_of = {rn: i + 1 for i, rn in enumerate(P)}
    density = {"widest_rep": widest, "widest_width": rep_width[widest],
               "widest_rank_in_pool": rank_of[widest],
               "widest_standalone_r2": round(rep_r2[widest][0], 5)}
    for B in BUDGETS:
        n_oc = arm3[f"B{B}"]["origin_counts"]
        tot = sum(n_oc.values()) or 1
        density[f"naive_B{B}_widest_share"] = round(n_oc.get(widest, 0) / tot, 4)
    # widest-rep share + slots for the BEST scheme at each cell + all schemes at
    # the representative cell (B=1000, PT=2B).
    per_cell_best_density = {}
    for cell, rec in best_by_cell.items():
        key = f"{rec['best_scheme']}_{cell}"
        w_oc = weighted[key]["origin_counts"]
        tot = sum(w_oc.values()) or 1
        per_cell_best_density[cell] = {
            "best_scheme": rec["best_scheme"],
            "widest_share": round(w_oc.get(widest, 0) / tot, 4),
            "widest_slots": weighted[key]["pool_slots"].get(widest, 0),
        }
    density["best_scheme_widest_share_by_cell"] = per_cell_best_density
    rep_cell = f"B1000_PT2xB"
    density["all_schemes_widest_share_rep_cell"] = rep_cell
    density["all_schemes_widest_share"] = {
        lbl: round((lambda oc: oc.get(widest, 0) / (sum(oc.values()) or 1))(
            weighted[f"{lbl}_{rep_cell}"]["origin_counts"]), 4)
        for lbl in SCHEMES}
    out["density_check"] = density

    out["wall_time_s"] = round(time.time() - t0, 1)
    print(f"[{ds}] done in {out['wall_time_s']}s  target={best_rep} "
          f"native={rep_native[best_rep][0]:.4f} r2={rep_r2[best_rep][0]:.4f}")
    return out


def _validate_against_engine(reps, y, folds, pool5):
    """Prove ARM1 manual quota == create_hybrid('fixed', n_per_rep=20) on fold 0."""
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


if __name__ == "__main__":
    dsets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DATASETS
    main(dsets)
