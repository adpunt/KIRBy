"""EXPERIMENT C31 — PERFORMANCE-WEIGHTED global feature selection (FULL sweep).

Question: today's BALANCED global selection caps EACH rep at an EQUAL top-N,
concatenates across the pool's reps, globally ranks all pooled columns by
lightgbm_gain, and takes the top-B budget. Does sizing each rep's candidate
contribution by a PERFORMANCE WEIGHT (stronger reps offer MORE candidate
columns) beat the EQUAL balanced global — and WHICH weighting family sizes the
global pool best?

*** FIX 2026-08-11: the prior run only tested TWO weighting families
(performance_weighted / power_2).  It imports the ENGINE'S OWN
`_compute_allocation_weights` and `_sanitize_baseline_scores` from
src/kirby/hybrid.py and drives them with the exact call sequence
`allocate_performance_weighted` uses.  The pool-sizing weighting is therefore
GUARANTEED identical to the allocation strategies. ***

*** EXTENSION 2026-08-20 (items 1-4): the run was lightgbm-only, single global
importance, whole-data pool sizing, and had no significance control.  It now:
  ITEM 1 — loops the DOWNSTREAM MODEL over {lightgbm, rf, catboost, mlp}, and
           for EACH model recomputes the TARGET (best base rep x THAT model)
           and re-scores every arm under it.  Output is keyed by model.
  ITEM 2 — sweeps the IMPORTANCE method used for the GLOBAL ranking
           ({lightgbm_gain, xgboost_gain}); per-rep CANDIDATE importance stays
           lightgbm_gain (it decides which columns a rep OFFERS, not the pick).
  ITEM 3 — emits a per-cell NOISE FLOOR: a group-block bootstrap on the stored
           out-of-fold predictions (paired arm-minus-target pooled-OOF R2).
           r2_delta_ci95 excluding 0 => the win survives fold/sample noise.
  ITEM 4 — computes the pool-SIZING weights INSIDE each outer fold's TRAIN split
           only (inner group-aware CV), removing the whole-data leak.
This makes the deliverable per (dataset x model x budget), with a global
importance axis and a significance flag on every cell. ***

Design (implemented exactly), per downstream model:
  1. Score each rep standalone (grouped CV, full native width) -> per-rep score,
     rank, pool = top-POOL, target = best rep x THIS model.
  2. PER FOLD (item 4): inner group-aware CV over TRAIN indices only -> per-rep
     TRAIN score -> gs*score -> `_sanitize_baseline_scores` -> ENGINE
     `_compute_allocation_weights(scheme, params, rep_dict_train=train_pool)`.
  3. pool_slots_i = round(w_i/sum(w) * POOL_TOTAL), capped at rep width, PER FOLD.
     POOL_TOTAL swept over {2*B, 4*B}.
  4. Each rep offers its top pool_slots_i columns by ITS OWN lightgbm_gain
     (train-fold only); concatenate -> weighted candidate pool.
  5. ONE global ranking (item-2 importance sweep) over that weighted pool
     (train-fold only); take top-B; score with the downstream model.

ARMS compared, paired per-fold, at matched budget B (all fold-safe):
  TARGET : best single staged rep x downstream model (full native width).
  WEIGHTED-GLOBAL : the variant above (each scheme x POOL_TOTAL x importance).
  ARM1 quota : per-rep equal fixed quota B/|pool| feats/rep (create_hybrid
               'fixed' behaviour), verified against the engine on one fold.
               IMPORTANCE-INVARIANT (uses per-rep lightgbm_gain only).
  ARM2 equal-global : EQUAL top-N per rep, concat, global gain top-B (c30's
               balanced global) — per importance.
  ARM3 naive : glue the whole pool, no cap, global gain top-B — per importance.

PRIMARY metric everywhere = R-squared (higher better; no sign flip), OOF-style
per fold.  SECONDARY = the dataset's native metric (direction-corrected).

HARD RULES honoured: never edits find_best_hybrid.py / hybrid.py (import-only);
never calls find_best_hybrid(); full rows; per-rep candidate importance FIXED at
lightgbm_gain; per-fold train-only selection & ranking; NO random splits (inner
sizing CV is group-aware too).

Run:  python c31_perf_weighted_global_selection.py [ds1 ds2 ...]
Env knobs (all optional):
  C31_MODELS   comma-list to restrict the model loop (SLURM makes model an array
               dimension by passing ONE model per task).  Default all four.
  C31_IMPS     comma-list to restrict the global-importance sweep.
  C31_BOOT     bootstrap resamples for the noise floor (default 1000; 0 = off).
  C31_OUT      results path (per-task file on the server).
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

# --- ITEM 1: downstream model loop -----------------------------------------
# Recompute the TARGET (best base rep x THIS model) and re-score every arm under
# each model.  SLURM makes model an ARRAY dimension (one model per task) via
# C31_MODELS so per-task wall time stays bounded; locally the full loop runs.
_ALL_MODELS = ["lightgbm", "rf", "catboost"]   # trimmed (2026-08-20): mlp dropped
                                               # (auto-skips large-n -> ragged coverage)
_env_models = os.environ.get("C31_MODELS", "").strip()
MODELS = [m.strip() for m in _env_models.split(",") if m.strip()] or _ALL_MODELS

IMP = "lightgbm_gain"            # per-rep CANDIDATE importance (which cols a rep
                                 # OFFERS) — FIXED, independent of downstream model.
# --- ITEM 2: global-ranking importance sweep -------------------------------
_ALL_IMPS = ["lightgbm_gain", "treeshap"]   # 2 of the engine's main 3
                                            # (deeplift dropped: OOMs on wide reps)
_env_imps = os.environ.get("C31_IMPS", "").strip()
IMPS = [m.strip() for m in _env_imps.split(",") if m.strip()] or _ALL_IMPS

POOL = 10                        # candidate pool = top-10 staged reps
BUDGETS = [500, 1000]            # B
POOL_TOTAL_MULTS = [2, 4]        # POOL_TOTAL in {2B, 4B}
NFOLDS = 5
N_INNER = 3                      # ITEM 4: inner group-aware CV folds for sizing
N_BOOT = int(os.environ.get("C31_BOOT", "1000"))   # ITEM 3: 0 disables
DEFAULT_DATASETS = ["herg_ki", "logd", "flab_thermo_tresanco"]

# SCHEME CUT (2026-08-20): the round-3 study named the survivors — 'linear'
# (best single default under the fuller roster), 'power_2' (prior default), and
# a hard concentrator 'top_2' as the third reference.  The model x importance
# sweep (items 1-2) multiplies runtime ~x8, so the 15-scheme sweep is cut to
# these 3 to keep the SLURM task inside its wall.  Restore the fuller SCHEMES
# dict (git history) if a scheme-only re-sweep is wanted.
SCHEMES = {
    "linear":  ("linear",    {}),
    "power_2": ("power_law", {"power": 2.0}),
    "top_2":   ("top_k",     {"k": 2}),
}

CAVEATS = [
    "ITEM 4 (per-fold sizing): pool-sizing weights are computed INSIDE each "
    "outer fold's TRAIN split via an inner group-aware CV (N_INNER folds), "
    "removing the prior whole-data leak (the old hib read rep scores measured "
    "on held-out validation folds). If a fold's TRAIN split has too few groups "
    "to build the inner CV, that fold falls back to the whole-data sizing and "
    "the fold is listed in size_fallback_folds.",
    "The candidate POOL membership (top-POOL reps) and the TARGET (best rep) are "
    "still chosen from WHOLE-DATA standalone ranking, so pool/target selection "
    "carries a mild leak that item 4 does NOT remove; only the per-rep slot "
    "SIZES within the pool are now cross-validated.",
    "2026-09-04 (SPLIT_AUDIT c31:428): a SECOND baseline is now reported next to "
    "the leaky one. target_trainpick picks the best rep per fold from that "
    "fold's TRAIN rows only (the same inner group-aware CV item 4 uses) and "
    "reads its HELD-OUT score for that fold, so pick and grade share no row. "
    "Every arm carries r2_delta_mean_vs_trainpick / "
    "native_delta_mean_vs_trainpick alongside its delta vs the whole-data "
    "target. Because the whole-data target is the inflated side, the original "
    "r2_delta_mean UNDERSTATES the gap; the trainpick delta is the fair one. "
    "The ranking of allocation methods within a cell is identical under both "
    "(the same baseline is subtracted from all of them); only win counts and CI "
    "flags move.",
    "2026-09-04 (SPLIT_AUDIT c31:711): best_scheme_by_cell no longer reports a "
    "delta or a CI as if one method had been fixed in advance. Its headline "
    "field is r2_delta_max_over_schemes — the largest of the 3 weighting "
    "methods, chosen on the folds it is scored on, so it runs high. The CI was "
    "REMOVED from this record: it was built for one fixed method and does not "
    "cover the 3-way choice. scheme_fold_wins_vs_target (out of "
    "scheme_fold_pairs) does not depend on picking a winner.",
    "ITEM 1 (model loop): the candidate pool P and target are recomputed per "
    "model (fully honest 'best hybrid vs best base under model M'), so the "
    "importance/global-ranking machinery runs once per model. A cheaper "
    "'fix P by lightgbm once, re-score arms per model' variant was NOT taken — "
    "it would change what the comparison means; that is a scoping choice.",
    "ITEM 1: mlp (and any _max_samples-guarded model) is SKIPPED, not crashed, "
    "on datasets whose largest train fold exceeds its guard (get_model returns "
    "None). Skipped models appear as by_model[model]={'skipped': ...} — the "
    "model sweep is NOT uniformly 4-wide across datasets.",
    "ITEM 2 (importance sweep): only the GLOBAL ranking importance is swept "
    "(IMPS); the per-rep CANDIDATE importance stays lightgbm_gain. arm1_quota "
    "is importance-INVARIANT (per-rep lightgbm_gain) and is NOT nested under "
    "IMPS. If an importance method is unavailable at runtime (e.g. xgboost "
    "missing) it is probed once and dropped, recorded in skipped_importances.",
    "ITEM 3 (noise floor): the significance control is a GROUP-BLOCK BOOTSTRAP "
    "on the stored out-of-fold predictions — it resamples GROUPS (so correlated "
    "same-group rows move together) and reports r2_delta_ci95 / r2_delta_p_boot "
    "for arm-minus-target POOLED-OOF R2. This is a POOLED-OOF significance "
    "PROXY, NOT a CI on the headline mean-of-per-fold R2. The rigorous "
    "shuffled-label control (Option B) is NOT implemented here (it doubles the "
    "whole-pipeline cost); only the cheap bootstrap ships.",
    "Only NFOLDS=5 paired folds, so any per-fold sign/Wilcoxon test is "
    "underpowered — the bootstrap over stored predictions is used instead.",
    "R-squared is per held-out fold on that fold's predictions (OOF-per-fold "
    "R2), averaged/paired vs target — NOT a single pooled-OOF R2. Native "
    "secondary metric handled identically.",
    "SCHEME CUT: the 15-scheme weighting sweep is cut to linear + power_2 + "
    "top_2 (round-3 winners) to keep the model x importance sweep inside the "
    "SLURM wall. Restore SCHEMES from git history for a scheme-only re-sweep.",
]


def gsign(lower_is_better):
    """+1 if higher metric is better, -1 if lower is better."""
    return -1.0 if lower_is_better else 1.0


# ---------------------------------------------------------------------------
# fold-safe scoring for an arbitrary per-fold column selection.
# Returns BOTH R-squared (primary) and native-metric per-fold values.
# `model` (ITEM 1) is now an explicit argument, threaded from run_dataset.
# ---------------------------------------------------------------------------
def score_selection(reps, y, folds, per_fold_sel, task_type, metric_fn, model):
    """per_fold_sel[i] = {rep_name: np.array(col_indices)} chosen on fold i TRAIN.

    Fits the downstream `model` per fold on the selected columns (scaled),
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
        m = get_model(model, task_type=task_type, n_samples=len(tr))
        if m is None:
            raise RuntimeError(f"get_model returned None for {model} at n={len(tr)}")
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
    all_out["config"] = {"models": MODELS, "candidate_importance": IMP,
                         "importance_methods": IMPS, "pool": POOL,
                         "budgets": BUDGETS, "pool_total_mults": POOL_TOTAL_MULTS,
                         "schemes": list(SCHEMES), "n_folds": NFOLDS,
                         "n_inner_sizing_folds": N_INNER,
                         "per_fold_sizing": True,
                         "noise_floor": "group_block_bootstrap",
                         "n_boot": N_BOOT,
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
        # ITEM 1 schema change: completeness is now keyed by "by_model".
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

    # -- group-block bootstrap machinery (ITEM 3), shared across all models -----
    uniq_groups = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq_groups}
    ug_list = list(uniq_groups)

    def boot_ci(arm_oof, tgt_oof):
        """Group-block bootstrap on stored OOF preds: arm-minus-target pooled-OOF
        R2. Resamples GROUPS with replacement so correlated rows move together.
        Returns ([lo95, hi95], p_boot=frac(delta<=0)). ([None,None],None) if off
        or degenerate."""
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
            if np.ptp(yy) == 0:                    # r2 undefined on constant y
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

    # -- ITEM 2: probe the global-importance methods ONCE, drop any unavailable -
    imps_ok, skipped_imps = [], {}
    _probe_rep = sorted(reps)[0]
    _pn = min(64, len(y))
    for imp in IMPS:
        try:
            H.rank_columns(reps[_probe_rep][:_pn], y[:_pn], method=imp)
            imps_ok.append(imp)
        except Exception as e:
            skipped_imps[imp] = f"{type(e).__name__}: {e}"
    if not imps_ok:
        raise RuntimeError(f"no usable global-importance method in {IMPS}: {skipped_imps}")

    # -- ENGINE VALIDATION (model-independent: create_hybrid uses lightgbm_gain
    #    per-rep importance, not the downstream model) — computed ONCE. ---------
    #    Needs a 5-rep pool; use the whole-data ranking under lightgbm to pick it.
    out["by_model"] = {}
    out["skipped_importances"] = skipped_imps
    out["global_importances_used"] = imps_ok

    max_train_n = max(len(tr) for tr, _ in folds)

    for model in MODELS:
        print(f"\n----- MODEL: {model} -----")
        # ITEM 1: size guard — skip (not crash) models whose largest train fold
        # exceeds their _max_samples (mlp on big-n datasets).
        if get_model(model, task_type=task, n_samples=max_train_n) is None:
            print(f"  [skip] {model}: max_train_n={max_train_n} exceeds _max_samples guard")
            out["by_model"][model] = {"skipped": f"{model}_max_samples",
                                      "max_train_n": int(max_train_n)}
            continue
        out["by_model"][model] = _run_model(
            ds, model, reps, y, groups, folds, mfn, lower, task, gs,
            imps_ok, boot_ci, max_train_n)

    out["wall_time_s"] = round(time.time() - t0, 1)
    print(f"[{ds}] done in {out['wall_time_s']}s  models={list(out['by_model'])}")
    return out


def _run_model(ds, model, reps, y, groups, folds, mfn, lower, task, gs,
               imps_ok, boot_ci, max_train_n):
    """The full model-dependent pipeline for ONE downstream model.  Everything
    that depends on the model (target, pool, arms) recomputes here (ITEM 1)."""
    mout = {"model": model}

    # -- standalone score for EVERY staged rep (full native width) -> rank -----
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
    P = ranked[:POOL]
    best_rep = ranked[0]
    target_native_pf = rep_native[best_rep][1]
    target_r2_pf = rep_r2[best_rep][1]
    target_oof = rep_oof[best_rep]

    mout["target"] = {
        "best_rep": best_rep, "width": rep_width[best_rep],
        "native_score_mean": rep_native[best_rep][0],
        "native_per_fold": [round(x, 5) for x in target_native_pf],
        "r2_score_mean": rep_r2[best_rep][0],
        "r2_per_fold": [round(x, 5) for x in target_r2_pf],
        "rep_ranking_top12": [(r, round(rep_native[r][0], 5), round(rep_r2[r][0], 5),
                               rep_width[r]) for r in ranked[:12]],
    }
    mout["pool"] = P
    mout["pool_widths"] = {rn: rep_width[rn] for rn in P}

    # -- whole-data sanitized hib (FALLBACK only, item-4 uses per-fold) ---------
    hib_whole = _sanitize_baseline_scores({rn: gs * rep_native[rn][0] for rn in P})
    pool_matrices_whole = {rn: H._clean(reps[rn]) for rn in P}
    mout["pool_standalone"] = {rn: {"native": round(rep_native[rn][0], 5),
                                    "r2": round(rep_r2[rn][0], 5),
                                    "hib_shifted": round(hib_whole[rn], 5),
                                    "width": rep_width[rn]} for rn in P}

    # -- per-fold, per-rep TRAIN lightgbm_gain CANDIDATE importances ------------
    #    (which columns a rep OFFERS — FIXED at lightgbm_gain, item-2 invariant).
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

    # ======================================================================
    # ITEM 4 — per-fold TRAIN-only pool-sizing.  Compute the per-rep TRAIN
    # score ONCE per outer fold (scheme/importance/budget-independent), then
    # derive per-scheme weights cheaply.  Inner CV is GROUP-AWARE (no random).
    # ======================================================================
    fold_hib = []                 # [i] -> {rn: sanitized hib} (train-only)
    fold_train_score = []         # [i] -> {rn: raw direction-corrected train score}
    fold_pool_train = []          # [i] -> {rn: cleaned train matrix} (for entropy)
    size_fallback_folds = []
    score_whole = {rn: gs * rep_native[rn][0] for rn in P}
    for i, (tr, va) in enumerate(folds):
        sub_groups = groups[tr]
        sub_y = y[tr]
        n_uni = len(np.unique(sub_groups))
        n_inner = min(N_INNER, n_uni)
        train_mats = {rn: H._clean(reps[rn][tr]) for rn in P}
        fold_pool_train.append(train_mats)
        if n_inner < 2:
            fold_hib.append(dict(hib_whole))          # fallback: whole-data sizing
            fold_train_score.append(dict(score_whole))
            size_fallback_folds.append(i)
            continue
        try:
            inner = H.make_folds(sub_groups, sub_y, task, n=n_inner)
            sc = {}
            for rn in P:
                m_native, _pf, _oof = H.score_matrix(
                    train_mats[rn], sub_y, inner, model, mfn, lower, task)
                sc[rn] = m_native
            tsc = {rn: gs * sc[rn] for rn in P}
            fold_train_score.append(tsc)
            fold_hib.append(_sanitize_baseline_scores(tsc))
        except Exception as e:
            print(f"  [size-fallback] fold {i}: inner CV failed ({type(e).__name__}: {e})")
            fold_hib.append(dict(hib_whole))
            fold_train_score.append(dict(score_whole))
            size_fallback_folds.append(i)
    mout["size_fallback_folds"] = size_fallback_folds

    # ======================================================================
    # SECOND BASELINE (2026-09-04, SPLIT_AUDIT c31:428) — the target above is
    # the rep with the best score across the same 5 folds it is then scored on,
    # so its score is inflated and every delta against it reads low.  This one
    # picks the rep per fold from that fold's TRAIN rows only (fold_train_score,
    # the inner group-aware CV), then reads that rep's HELD-OUT score for the
    # fold.  Pick and grade share no row.  Reported alongside, never instead of:
    # every existing field is unchanged.
    # ======================================================================
    def _tp_key(i):
        def k(rn):
            v = fold_train_score[i][rn]
            return v if np.isfinite(v) else -np.inf
        return k

    tp_rep = [max(P, key=_tp_key(i)) for i in range(NFOLDS)]
    tp_native_pf = [rep_native[tp_rep[i]][1][i] for i in range(NFOLDS)]
    tp_r2_pf = [rep_r2[tp_rep[i]][1][i] for i in range(NFOLDS)]
    tpR2 = np.array(tp_r2_pf)
    tpNat = np.array(tp_native_pf)
    mout["target_trainpick"] = {
        "best_rep_per_fold": tp_rep,
        "native_per_fold": [round(x, 5) for x in tp_native_pf],
        "native_score_mean": float(np.mean(tp_native_pf)),
        "r2_per_fold": [round(x, 5) for x in tp_r2_pf],
        "r2_score_mean": float(np.mean(tp_r2_pf)),
        "folds_agreeing_with_target": int(sum(1 for r in tp_rep if r == best_rep)),
        "note": "picked on each fold's training rows only; scored on its held-out "
                "rows. Deltas against it are r2_delta_mean_vs_trainpick.",
    }

    # per-scheme, per-fold weight vectors (POOL_TOTAL-independent) -------------
    scheme_fold_weights = {}      # label -> [ {rn: w} per fold ]
    for label, (scheme, params) in SCHEMES.items():
        wl = []
        for i in range(NFOLDS):
            w = _compute_allocation_weights(fold_hib[i], scheme=scheme, params=params,
                                            rep_dict_train=fold_pool_train[i])
            wl.append({rn: float(w.get(rn, 0.0)) for rn in P})
        scheme_fold_weights[label] = wl

    def slots_for(label, POOL_TOTAL):
        """Per-fold slot dict for a scheme at a given POOL_TOTAL (item-4 sizing)."""
        sbf = {}
        for i in range(NFOLDS):
            w = scheme_fold_weights[label][i]
            tot = sum(w.values())
            slots = {}
            for rn in P:
                s = int(round((w[rn] / tot) * POOL_TOTAL)) if tot > 0 else 0
                slots[rn] = min(max(s, 0), rep_width[rn])
            sbf[i] = slots
        return sbf

    # scheme_weights reporting: mean + std over folds (density check reads mean).
    scheme_weights = {}
    for label in SCHEMES:
        wl = scheme_fold_weights[label]
        scheme_weights[label] = {
            rn: {"mean": round(float(np.mean([wl[i][rn] for i in range(NFOLDS)])), 6),
                 "std": round(float(np.std([wl[i][rn] for i in range(NFOLDS)])), 6)}
            for rn in P}
    mout["scheme_weights"] = scheme_weights

    # ======================================================================
    # per-fold GLOBAL ranking builder — memoized on (importance, per-fold slots).
    # Per-fold slots rarely collide across cells, so memo hit-rate is low; arm2/
    # arm3 (constant-per-fold slots) still benefit.
    # ======================================================================
    _rank_cache = {}

    def build_fold_global(slots_by_fold, gimp_method):
        sig = (gimp_method,) + tuple(
            (i,) + tuple(sorted((rn, int(slots_by_fold[i][rn])) for rn in P))
            for i in range(NFOLDS))
        if sig in _rank_cache:
            return _rank_cache[sig]
        fold_global = []
        for i in range(NFOLDS):
            tr, va = folds[i]
            origins, mats = [], []
            for rn in P:
                idx = topk_local(rn, i, slots_by_fold[i][rn])
                if len(idx) == 0:
                    continue
                mats.append(H._clean(reps[rn][tr][:, idx]))
                origins.extend([(rn, int(c)) for c in idx])
            if not mats:
                fold_global.append(([], np.array([])))
                continue
            glued = np.hstack(mats)
            gimp = H.rank_columns(glued, y[tr], method=gimp_method)
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

    def slot_summary(slots_by_fold):
        """mean/std slots per rep + mean candidate-pool size, for reporting."""
        mean = {rn: round(float(np.mean([slots_by_fold[i][rn] for i in range(NFOLDS)])), 2)
                for rn in P}
        std = {rn: round(float(np.std([slots_by_fold[i][rn] for i in range(NFOLDS)])), 2)
               for rn in P}
        cand = float(np.mean([sum(slots_by_fold[i][rn] for rn in P) for i in range(NFOLDS)]))
        return mean, std, round(cand, 1)

    # ======================================================================
    # WEIGHTED-GLOBAL — scheme x POOL_TOTAL x budget x IMPORTANCE (items 2,4)
    # ======================================================================
    weighted = {imp: {} for imp in imps_ok}
    for label, (scheme, params) in SCHEMES.items():
        for B in BUDGETS:
            for mult in POOL_TOTAL_MULTS:
                POOL_TOTAL = mult * B
                slots_by_fold = slots_for(label, POOL_TOTAL)
                smean, sstd, cand = slot_summary(slots_by_fold)
                for imp in imps_ok:
                    fold_global = build_fold_global(slots_by_fold, imp)
                    per_fold_sel, oc = topB_from_global(fold_global, B)
                    npf, r2pf, oof, nf = score_selection(
                        reps, y, folds, per_fold_sel, task, mfn, model)
                    key = f"{label}_B{B}_PT{mult}xB"
                    weighted[imp][key] = {
                        "native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                        "pool_total": POOL_TOTAL, "pool_slots_mean": smean,
                        "pool_slots_std": sstd, "candidate_pool_size": cand,
                        "origin_counts": oc, "scheme": label, "B": B, "mult": mult}

    # ======================================================================
    # ARM1 — per-rep equal quota (create_hybrid 'fixed'). IMPORTANCE-INVARIANT.
    # ======================================================================
    arm1 = {}
    for B in BUDGETS:
        q = max(1, B // POOL)
        per_fold_sel = [{rn: topk_local(rn, i, q) for rn in P} for i in range(NFOLDS)]
        npf, r2pf, oof, nf = score_selection(reps, y, folds, per_fold_sel, task, mfn, model)
        arm1[f"B{B}"] = {"native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                         "quota_per_rep": q}

    # ======================================================================
    # ARM2 — EQUAL balanced global (equal top-N per rep) — per importance
    # ======================================================================
    arm2 = {imp: {} for imp in imps_ok}
    for B in BUDGETS:
        for mult in POOL_TOTAL_MULTS:
            POOL_TOTAL = mult * B
            N = max(1, int(round(POOL_TOTAL / POOL)))     # equal cap per rep
            slots = {rn: min(N, rep_width[rn]) for rn in P}
            slots_by_fold = {i: slots for i in range(NFOLDS)}
            for imp in imps_ok:
                fold_global = build_fold_global(slots_by_fold, imp)
                per_fold_sel, oc = topB_from_global(fold_global, B)
                npf, r2pf, oof, nf = score_selection(
                    reps, y, folds, per_fold_sel, task, mfn, model)
                arm2[imp][f"B{B}_PT{mult}xB"] = {
                    "native_pf": npf, "r2_pf": r2pf, "oof": oof, "nfeat": nf,
                    "equal_cap_N": N, "pool_total": POOL_TOTAL, "origin_counts": oc}

    # ======================================================================
    # ARM3 — naive global (whole pool, no cap) density ctrl — per importance
    # ======================================================================
    arm3 = {imp: {} for imp in imps_ok}
    slots_full = {rn: rep_width[rn] for rn in P}
    slots_by_fold_full = {i: slots_full for i in range(NFOLDS)}
    for imp in imps_ok:
        fold_global_full = build_fold_global(slots_by_fold_full, imp)
        for B in BUDGETS:
            per_fold_sel, oc = topB_from_global(fold_global_full, B)
            npf, r2pf, oof, nf = score_selection(
                reps, y, folds, per_fold_sel, task, mfn, model)
            arm3[imp][f"B{B}"] = {"native_pf": npf, "r2_pf": r2pf, "oof": oof,
                                  "nfeat": nf, "origin_counts": oc,
                                  "total_pool_width": int(sum(rep_width[rn] for rn in P))}

    # ======================================================================
    # ENGINE VALIDATION — ARM1 manual quota == create_hybrid('fixed')
    # (per-rep lightgbm_gain; independent of downstream model, but cheap.)
    # ======================================================================
    mout["engine_validation"] = _validate_against_engine(reps, y, folds, P[:5])

    # ======================================================================
    # SUMMARISE — paired per-fold deltas vs TARGET + ITEM-3 bootstrap noise floor
    # ======================================================================
    tR2 = np.array(target_r2_pf)
    tNat = np.array(target_native_pf)

    def summarize(cfg):
        r2 = np.array(cfg["r2_pf"])
        nat = np.array(cfg["native_pf"])
        dR2 = r2 - tR2                       # +ve = beats target on R2
        dNat = gs * (nat - tNat)             # +ve = beats target on native
        dR2tp = r2 - tpR2                    # vs the per-fold train-picked baseline
        dNattp = gs * (nat - tpNat)
        ci95, p_boot = boot_ci(cfg.get("oof"), target_oof)   # ITEM 3
        s = {
            "r2_mean": float(np.mean(r2)),
            "r2_delta_mean": float(np.mean(dR2)),
            "r2_delta_per_fold": [round(float(x), 5) for x in dR2],
            "r2_folds_won": int(np.sum(dR2 > 0)),
            "r2_delta_ci95": ci95,           # pooled-OOF group-block bootstrap
            "r2_delta_p_boot": p_boot,       # frac(delta<=0); CI excludes 0 => win
            "r2_delta_mean_vs_trainpick": float(np.mean(dR2tp)),
            "r2_folds_won_vs_trainpick": int(np.sum(dR2tp > 0)),
            "native_mean": float(np.mean(nat)),
            "native_delta_mean": float(np.mean(dNat)),
            "native_folds_won": int(np.sum(dNat > 0)),
            "native_delta_mean_vs_trainpick": float(np.mean(dNattp)),
            "nfeat": round(cfg["nfeat"], 1),
        }
        for k in ("origin_counts", "pool_slots_mean", "pool_slots_std",
                  "pool_total", "candidate_pool_size", "equal_cap_N",
                  "quota_per_rep", "total_pool_width", "scheme", "B", "mult"):
            if k in cfg:
                s[k] = cfg[k]
        return s

    mout["weighted_global"] = {imp: {k: summarize(v) for k, v in weighted[imp].items()}
                               for imp in imps_ok}
    mout["arm1_quota"] = {k: summarize(v) for k, v in arm1.items()}
    mout["arm2_equal_global"] = {imp: {k: summarize(v) for k, v in arm2[imp].items()}
                                 for imp in imps_ok}
    mout["arm3_naive"] = {imp: {k: summarize(v) for k, v in arm3[imp].items()}
                          for imp in imps_ok}

    # ---- BEST SCHEME per (importance, POOL_TOTAL, budget) by R2 delta -------
    # 2026-09-04 (SPLIT_AUDIT c31:711): the winning weighting method is chosen on
    # the same 5 folds it is then reported on, so this delta is a MAXIMUM over
    # the 3 methods, not one method's delta — it runs high, and the per-method
    # bootstrap CI does not cover the choice, so it is NOT carried here.  The
    # honest per-method deltas, fold-win counts and CIs stay in
    # weighted_global[imp].  Added: how many of the 3 x 5 method-fold pairs beat
    # the target, which does not depend on picking a winner.
    best_by_cell = {imp: {} for imp in imps_ok}
    for imp in imps_ok:
        for B in BUDGETS:
            for mult in POOL_TOTAL_MULTS:
                cands = {lbl: mout["weighted_global"][imp][f"{lbl}_B{B}_PT{mult}xB"]
                         for lbl in SCHEMES}
                best_lbl = max(cands, key=lambda l: cands[l]["r2_delta_mean"])
                best_by_cell[imp][f"B{B}_PT{mult}xB"] = {
                    "best_scheme": best_lbl,
                    "r2_delta_max_over_schemes": round(
                        cands[best_lbl]["r2_delta_mean"], 5),
                    "r2_folds_won_of_max_scheme": cands[best_lbl]["r2_folds_won"],
                    "native_delta_max_over_schemes": round(
                        cands[best_lbl]["native_delta_mean"], 5),
                    "scheme_fold_wins_vs_target": int(
                        sum(c["r2_folds_won"] for c in cands.values())),
                    "scheme_fold_pairs": int(len(cands) * NFOLDS),
                    "r2_delta_mean_by_scheme": {
                        lbl: round(c["r2_delta_mean"], 5) for lbl, c in cands.items()},
                    "selection_note": "r2_delta_max_over_schemes is the LARGEST of "
                                      f"{len(cands)} weighting methods, picked on the "
                                      "same folds it is scored on. Not a measured "
                                      "delta for any one method; no CI is given "
                                      "because the per-method CI does not cover the "
                                      "choice. Use scheme_fold_wins_vs_target.",
                }
    mout["best_scheme_by_cell"] = best_by_cell

    # ---- DENSITY-BIAS CHECK (per importance) -------------------------------
    widest = max(P, key=lambda rn: rep_width[rn])
    rank_of = {rn: i + 1 for i, rn in enumerate(P)}
    density = {imp: {} for imp in imps_ok}
    for imp in imps_ok:
        d = {"widest_rep": widest, "widest_width": rep_width[widest],
             "widest_rank_in_pool": rank_of[widest],
             "widest_standalone_r2": round(rep_r2[widest][0], 5)}
        for B in BUDGETS:
            n_oc = arm3[imp][f"B{B}"]["origin_counts"]
            tot = sum(n_oc.values()) or 1
            d[f"naive_B{B}_widest_share"] = round(n_oc.get(widest, 0) / tot, 4)
        per_cell_best_density = {}
        for cell, rec in best_by_cell[imp].items():
            key = f"{rec['best_scheme']}_{cell}"
            w_oc = weighted[imp][key]["origin_counts"]
            tot = sum(w_oc.values()) or 1
            per_cell_best_density[cell] = {
                "best_scheme": rec["best_scheme"],
                "widest_share": round(w_oc.get(widest, 0) / tot, 4),
                "widest_slots_mean": weighted[imp][key]["pool_slots_mean"].get(widest, 0),
            }
        d["best_scheme_widest_share_by_cell"] = per_cell_best_density
        # representative cell = largest budget x first POOL_TOTAL mult, derived
        # from the live sweep so it always exists (was hardcoded B1000_PT2xB).
        rep_cell = f"B{BUDGETS[-1]}_PT{POOL_TOTAL_MULTS[0]}xB"
        d["all_schemes_widest_share_rep_cell"] = rep_cell
        d["all_schemes_widest_share"] = {
            lbl: round((lambda oc: oc.get(widest, 0) / (sum(oc.values()) or 1))(
                weighted[imp][f"{lbl}_{rep_cell}"]["origin_counts"]), 4)
            for lbl in SCHEMES}
        density[imp] = d
    mout["density_check"] = density

    print(f"  [{model}] target={best_rep} native={rep_native[best_rep][0]:.4f} "
          f"r2={rep_r2[best_rep][0]:.4f}  imps={imps_ok}")
    return mout


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
