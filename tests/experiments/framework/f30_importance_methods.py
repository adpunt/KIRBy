"""f30 — do new feature importance scoring methods build better hybrids than the
current ones, and which of them are saying the same thing?

Mimics the search rather than calling find_best_hybrid: it uses the engine's own
`_compute_allocation_weights` and `_sanitize_baseline_scores` (import-only) and
reproduces the quota mechanism's per-representation top-k, exactly as
c32_allocation_strategy_cross.py does.  Because find_best_hybrid is never called,
NO result_db and NO search/baseline cache is touched — there is nothing to delete
before or after this runs.

Varied axis: the importance scoring method (tests/experiments/framework/f30_scorers.py).
Held fixed: budget 100, 5 group-aware folds, the pruned 10 representations per
dataset, allocation in {rank_exp_d0.3, power_2} on the quota mechanism, internal
evaluator and downstream model each in {lightgbm, rf}.

Two stages, because the per-representation baseline scores depend only on
(dataset, fold, model) and would otherwise be recomputed once per arm:

  baselines : folds + metric gate + inner-CV per-representation scores -> one file
  arms      : one importance method, all 10 representations x 5 folds, both
              allocations, both downstream models -> one JSON + one npz

Metric direction is gated before any scoring and every result carries its metric
name and direction.  Datasets are never pooled: rmse, spearman and f1_weighted are
not commensurable and two of the four run in opposite directions.

A failing scorer RAISES — the silent drop at find_best_hybrid.py:4473 is not copied.
"""
import lightgbm as _lgb  # noqa: F401  — segfault guard, must precede torch/kirby

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = str(_HERE.parents[3])
for _p in (os.path.join(_REPO, "src"),
           os.path.join(_REPO, "tests"),
           os.path.join(_REPO, "tests", "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import _alloc_harness as H  # noqa: E402
from kirby.hybrid import (  # noqa: E402
    _compute_allocation_weights, _sanitize_baseline_scores, create_hybrid,
)
from kirby.registry import require  # noqa: E402
from complexity.complexity_helpers import DATASET_REPS  # noqa: E402
import f30_scorers as SC  # noqa: E402

DATASETS = ["caco2", "buchwald_hartwig", "flab_expression_koenig", "tcga_brca_pam50"]
# registered allocation names -> (scheme, params).  hybrid.py:350 and :361.
ALLOCATIONS = {
    "power_2":       ("power_law", {"power": 2.0}),
    "rank_exp_d0.3": ("rank_exp", {"decay": 0.3}),
}
# USER DECISION 2026-09-03, after the tcga cost breakdown.
#   internal evaluator (supplies the allocation weights): rf only
#   downstream model (scores the held-out fold, and defines the target): both
# NOTE the baselines stage still runs the inner CV for BOTH models, because the
# target is "best single representation for that same downstream model" and that
# choice has to be made with the downstream model. Cutting the internal evaluator
# halves the hybrid fits, not the baselines.
INTERNAL_EVALUATORS = ["rf"]
DOWNSTREAM_MODELS = ["lightgbm", "rf"]
MODELS = sorted(set(INTERNAL_EVALUATORS) | set(DOWNSTREAM_MODELS))  # inner CV union
BUDGET = 100
NFOLDS = 3          # was 5
N_INNER = 2         # was 3
SEED = 42
OUTDIR = os.environ.get(
    "F30_OUT", os.path.join(os.environ.get("KIRBY_CACHE_DIR", "tests/results"),
                            "f30_importance_methods"))


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate_metric_direction(ds):
    """Blocking. For regression datasets use the harness gate; for a
    classification metric build a class ladder instead of a value ladder, because
    the harness's synthetic predictor is continuous."""
    info = require(ds)
    from kirby.find_best_hybrid import BUILTIN_METRICS
    fn, lower = BUILTIN_METRICS[info.metric]
    if info.task_type == "regression":
        H.assert_metric_direction(verbose=True, datasets=[ds])
        return info.metric, bool(lower)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=400)
    good = y.copy()
    flip = rng.random(400) < 0.15
    good[flip] = rng.integers(0, 5, size=int(flip.sum()))
    bad = rng.integers(0, 5, size=400)
    s_good, s_bad = float(fn(y, good)), float(fn(y, bad))
    ok = (s_good < s_bad) if lower else (s_good > s_bad)
    if not ok:
        raise AssertionError(
            f"[gate] {ds}: metric {info.metric} not monotone in claimed direction "
            f"(good={s_good:.4f} bad={s_bad:.4f} lower={lower})")
    print(f"  [gate] {ds:22s} metric={info.metric:11s} "
          f"dir={'lower' if lower else 'higher':6s} good={s_good:.4f} "
          f"bad={s_bad:.4f}  PASS", flush=True)
    return info.metric, bool(lower)


def load_and_check(ds):
    """Load the pruned 10 and assert we got exactly them, by name.

    The antibody and reaction loaders drop a failed representation with a warning
    rather than an error (_shared.py:2925, :3096), so this has to be checked.
    """
    reps, y, groups, mname, mfn, lower, task = H.load_full(ds)
    want = list(DATASET_REPS[ds]["general"])
    got = sorted(reps)
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing or extra:
        raise AssertionError(
            f"{ds}: expected the pruned 10 {want}; missing={missing} extra={extra}")
    assert len(reps) == 10, f"{ds}: {len(reps)} representations loaded, expected 10"
    print(f"  [reps] {ds}: 10/10 present, widths="
          f"{ {k: reps[k].shape[1] for k in want} }", flush=True)
    return reps, y, groups, mname, mfn, lower, task, want


# Datasets whose folds are made with plain GroupKFold instead of the classification
# default. USER DECISION 2026-09-03, for tcga_brca_pam50 only.
#
# cv_split sends any classification dataset to StratifiedGroupKFold
# (_shared.py:1200-1205). tcga_brca_pam50 is grouped by TSS site and the 19 sites
# are very uneven (133 samples down to a site with 1), so stratifying produced
# validation folds of 3, 79, 403, 142 and 194 — a 3-sample fold scoring 1.0 and
# carrying a fifth of the mean, and a fold training on 418 of 821 samples.
#
# MEASURED both ways at k=3 and k=5: at least one PAM50 class is absent from at
# least one validation fold under EITHER splitter, so stratifying does not buy the
# class balance it costs the size balance for. GroupKFold gives 162/166/163/168/162.
#
# Scoped to this experiment on purpose — _shared.cv_split is NOT changed, so every
# other classification experiment in the repo keeps its current behaviour.
FORCE_GROUPKFOLD = {"tcga_brca_pam50"}


def make_folds_for(ds, groups, labels, task, n=NFOLDS):
    """Folds for one dataset, honouring FORCE_GROUPKFOLD."""
    if ds in FORCE_GROUPKFOLD:
        from sklearn.model_selection import GroupKFold
        X = np.zeros((len(labels), 1))
        folds = [(np.asarray(tr), np.asarray(va))
                 for tr, va in GroupKFold(n_splits=n).split(X, labels, groups=groups)]
        assert len(folds) == n
        cov = np.concatenate([va for _, va in folds])
        assert len(np.unique(cov)) == len(cov), "validation folds overlap"
        print(f"  [folds] {ds}: GroupKFold (forced, see FORCE_GROUPKFOLD) "
              f"val sizes={[len(va) for _, va in folds]}", flush=True)
        return folds
    folds = H.make_folds(groups, labels, task, n=n)
    print(f"  [folds] {ds}: repo default via cv_split, "
          f"val sizes={[len(va) for _, va in folds]}", flush=True)
    return folds


def is_binary_rep(X):
    u = np.unique(X[: min(len(X), 2000)])
    return bool(len(u) <= 3 and set(np.round(u, 6)).issubset({-1.0, 0.0, 1.0}))


# ---------------------------------------------------------------------------
# STAGE A — folds + per-representation baseline scores (shared by every arm)
# ---------------------------------------------------------------------------
def inner_cv_scores(ds, reps, y, groups, folds, model, mfn, lower, task, rep_order):
    """Per (fold, representation) score from an inner group-aware CV on the TRAIN
    split only.  Never touches the validation fold — selecting the target
    representation on validation scores is the leak that had to be corrected in
    SELECTOR_PREDICTION_RESULTS_2026_07_18.md.

    Returned already sign-corrected so higher is better, which is what
    _compute_allocation_weights expects (hybrid.py:919-926).
    """
    gs = -1.0 if lower else 1.0
    out = []
    for fi, (tr, _va) in enumerate(folds):
        sub_groups, sub_y = groups[tr], y[tr]
        n_inner = min(N_INNER, len(np.unique(sub_groups)))
        per_rep, fallback = {}, False
        if n_inner < 2:
            fallback = True
        else:
            # same splitter rule as the outer folds, so the representation that
            # becomes the target is not chosen on lopsided inner folds either
            if ds in FORCE_GROUPKFOLD:
                from sklearn.model_selection import GroupKFold
                inner = [(np.asarray(a), np.asarray(b)) for a, b in
                         GroupKFold(n_splits=n_inner).split(
                             np.zeros((len(sub_y), 1)), sub_y, groups=sub_groups)]
            else:
                inner = H.make_folds(sub_groups, sub_y, task, n=n_inner)
            for rn in rep_order:
                try:
                    mean, _pf, _oof = H.score_matrix(
                        reps[rn][tr], sub_y, inner, model, mfn, lower, task)
                    per_rep[rn] = gs * float(mean)
                except Exception as e:
                    raise RuntimeError(
                        f"inner CV failed: fold={fi} rep={rn} model={model}: "
                        f"{type(e).__name__}: {e}")
        if fallback:
            raise RuntimeError(
                f"fold {fi}: only {len(np.unique(sub_groups))} groups in train — "
                f"cannot run an inner CV; the split is degenerate, not the method")
        out.append(per_rep)
        print(f"    [inner] fold {fi} model={model} best="
              f"{max(per_rep, key=per_rep.get)} ({max(per_rep.values()):.4f})",
              flush=True)
    return out


def run_baselines(ds):
    t0 = time.time()
    mname, lower_gate = gate_metric_direction(ds)
    reps, y, groups, mname2, mfn, lower, task, rep_order = load_and_check(ds)
    assert mname == mname2 and lower_gate == lower
    folds = make_folds_for(ds, groups, y, task, n=NFOLDS)

    inner = {}
    for model in MODELS:
        inner[model] = inner_cv_scores(ds, reps, y, groups, folds, model, mfn,
                                       lower, task, rep_order)

    # target = best single representation for that downstream model, chosen on the
    # inner CV, then scored on the held-out fold.  "best rep x model = THE target".
    targets = {}
    for model in MODELS:
        per_fold, names = [], []
        for fi, (tr, va) in enumerate(folds):
            best = max(inner[model][fi], key=inner[model][fi].get)
            sc = StandardScaler()
            Xtr = sc.fit_transform(reps[best][tr])
            Xva = sc.transform(reps[best][va])
            m = H.get_model(model, task_type=task, n_samples=len(tr))
            m.fit(Xtr, y[tr])
            per_fold.append(float(mfn(y[va], m.predict(Xva))))
            names.append(best)
        targets[model] = {"per_fold": per_fold, "rep_per_fold": names}
        print(f"  [target] {model}: {names} -> {np.round(per_fold, 4).tolist()}",
              flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    payload = {
        "dataset": ds, "metric": mname, "lower_is_better": bool(lower),
        "task_type": task, "n_samples": int(len(y)), "n_folds": NFOLDS,
        "rep_order": rep_order,
        "rep_widths": {rn: int(reps[rn].shape[1]) for rn in rep_order},
        "rep_is_binary": {rn: is_binary_rep(reps[rn]) for rn in rep_order},
        "folds": [[np.asarray(tr).tolist(), np.asarray(va).tolist()]
                  for tr, va in folds],
        "inner_scores": {m: inner[m] for m in MODELS},
        "targets": targets,
        "budget": BUDGET, "seed": SEED,
        "wall_time_s": round(time.time() - t0, 1),
    }
    path = os.path.join(OUTDIR, f"{ds}__baselines.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    print(f"  [write] {path}  ({payload['wall_time_s']}s)", flush=True)
    return payload


# ---------------------------------------------------------------------------
# STAGE B — one importance method
# ---------------------------------------------------------------------------
def quota_from_weights(base_scores, alloc, rep_order, widths, budget):
    """The quota mechanism's per-representation slot count.

    Weights come from the ENGINE (_compute_allocation_weights, hybrid.py:924) on
    the engine-sanitised baseline scores (_sanitize_baseline_scores, hybrid.py:752),
    so this arm cannot drift from production.
    """
    scheme, params = ALLOCATIONS[alloc]
    hib = _sanitize_baseline_scores({rn: float(base_scores[rn]) for rn in rep_order})
    w = _compute_allocation_weights(hib, scheme=scheme, params=params)
    tot = float(sum(w.values())) or 1.0
    return {rn: int(np.clip(round((w[rn] / tot) * budget), 0, widths[rn]))
            for rn in rep_order}


def validate_against_engine(reps, y, folds, rep_order):
    """Prove this file's per-representation top-k equals create_hybrid('fixed').
    Same check c32 makes at c32_allocation_strategy_cross.py:167."""
    tr, _va = folds[0]
    pool = rep_order[:5]
    train_reps = {rn: reps[rn][tr] for rn in pool}
    try:
        _, finfo = create_hybrid(train_reps, y[tr], allocation_method="fixed",
                                 n_per_rep=20, importance_method="lightgbm_gain")
    except Exception as e:
        return {"ran": False, "reason": f"{type(e).__name__}: {e}"}
    match = {}
    for rn in pool:
        eng = set(int(x) for x in finfo[rn]["selected_indices"])
        imp = H.rank_columns(reps[rn][tr], y[tr], method="lightgbm_gain")
        match[rn] = (eng == set(int(x) for x in np.argsort(imp)[-20:][::-1]))
    return {"ran": True, "all_reps_match": bool(all(match.values())),
            "per_rep": match, "config": "fold0 pool5 fixed n_per_rep=20"}


def run_arm(ds, arm, base, reps, y, mfn, lower, task, folds, n_jobs=1,
            fold_subset=None, alloc_subset=None, model_subset=None):
    rep_order = base["rep_order"]
    widths = {rn: int(reps[rn].shape[1]) for rn in rep_order}
    gs = -1.0 if lower else 1.0
    allocs = alloc_subset or list(ALLOCATIONS)
    internals = model_subset or INTERNAL_EVALUATORS
    downstreams = model_subset or DOWNSTREAM_MODELS
    fidx = fold_subset if fold_subset is not None else list(range(len(folds)))

    timings, scores_store, fail = [], {}, None
    # --- score every representation on every fold's TRAIN rows ---------------
    imp = {}
    for fi in fidx:
        tr, _va = folds[fi]
        for rn in rep_order:
            ctx = {"task_type": task, "seed": SEED + fi, "n_jobs": n_jobs,
                   "budget": max(BUDGET, 50),
                   "is_binary_rep": bool(base["rep_is_binary"][rn])}
            v, secs = SC.score(arm, reps[rn][tr], y[tr], ctx)
            imp[(fi, rn)] = v
            timings.append({"fold": fi, "rep": rn, "n_features": widths[rn],
                            "seconds": round(secs, 3)})
            scores_store[f"f{fi}__{rn}"] = v.astype(np.float32)
        print(f"    [{arm}] fold {fi} scored {len(rep_order)} reps "
              f"({sum(t['seconds'] for t in timings if t['fold'] == fi):.1f}s)",
              flush=True)

    cells = evaluate_imp(imp, label=arm, base=base, reps=reps, y=y, mfn=mfn,
                         lower=lower, task=task, folds=folds, fidx=fidx,
                         allocs=allocs, internals=internals, downstreams=downstreams)
    return cells, timings, scores_store, fail


def evaluate_imp(imp, *, label, base, reps, y, mfn, lower, task, folds, fidx,
                 allocs, internals, downstreams):
    """Build and score the hybrid from a per-(fold, representation) score dict.

    Shared by the scoring arms and by the blends, so a blend is evaluated by
    exactly the same code path as any single method.
    """
    rep_order = base["rep_order"]
    widths = {rn: int(reps[rn].shape[1]) for rn in rep_order}
    gs = -1.0 if lower else 1.0
    arm = label
    cells = {}
    for internal in internals:              # supplies the allocation baseline scores
        for alloc in allocs:
            for downstream in downstreams:  # scores the held-out fold
                key = f"{alloc}|quota|int_{internal}|dn_{downstream}"
                per_fold, per_fold_target, nfeat, sel_counts = [], [], [], []
                for fi in fidx:
                    tr, va = folds[fi]
                    q = quota_from_weights(base["inner_scores"][internal][fi],
                                           alloc, rep_order, widths, BUDGET)
                    mats, chosen = [], {}
                    for rn in rep_order:
                        k = q[rn]
                        if k <= 0:
                            continue
                        top = np.argsort(imp[(fi, rn)])[-k:][::-1]
                        chosen[rn] = [int(t) for t in top]
                        mats.append(reps[rn][:, top])
                    if not mats:
                        raise RuntimeError(
                            f"{arm}/{key} fold {fi}: allocation gave every "
                            f"representation zero slots — quota={q}")
                    Xh = np.hstack(mats)
                    sc = StandardScaler()
                    m = H.get_model(downstream, task_type=task, n_samples=len(tr))
                    m.fit(sc.fit_transform(Xh[tr]), y[tr])
                    p = m.predict(sc.transform(Xh[va]))
                    per_fold.append(float(mfn(y[va], p)))
                    per_fold_target.append(
                        float(base["targets"][downstream]["per_fold"][fi]))
                    nfeat.append(int(Xh.shape[1]))
                    sel_counts.append({rn: len(v) for rn, v in chosen.items()})

                # direction-corrected paired delta: positive = hybrid better
                d = [gs * (h - t) for h, t in zip(per_fold, per_fold_target)]
                cells[key] = {
                    "hybrid_per_fold": per_fold,
                    "target_per_fold": per_fold_target,
                    "delta_per_fold": d,
                    "delta_mean": float(np.mean(d)),
                    "folds_won": int(sum(1 for x in d if x > 0)),
                    "n_features_per_fold": nfeat,
                    "slots_per_fold": sel_counts,
                }
    return cells


def run_blends(ds, n_jobs=1, top_k=4, max_rho=0.8):
    """Blend the best-performing scoring methods and score the blends.

    Costs almost nothing: every arm already wrote its per-(fold, representation)
    score vectors to <ds>__<arm>__full.npz, so a blend is a rank fusion of arrays
    already on disk plus the same hybrid evaluation every single method got.

    WHICH arms are blended is decided from the results, not chosen in advance:

      - `top{k}`      the k best arms by their best cell's delta.
      - `complement`  the best arm, plus the best arm that DISAGREES with it
                      (mean Spearman over both-scored features below `max_rho`),
                      plus the next such. Blending two methods that already rank
                      features the same way cannot do anything, which is why the
                      agreement matrix picks the members rather than the ranking.

    Controls (`random`, `variance`, `laplacian`) are never blended in.
    Fusion rules are the two published ones already in this repo — rank averaging
    / Borda (rankers.py:146 `combine_ranks`) and reciprocal rank fusion
    (aggregate.py:90). See f30_scorers.BLENDS.
    """
    import glob
    from scipy.stats import spearmanr

    base_path = os.path.join(OUTDIR, f"{ds}__baselines.json")
    if not os.path.exists(base_path):
        raise SystemExit(f"missing {base_path} — run the baselines stage first")
    with open(base_path) as f:
        base = json.load(f)

    # collect finished arms
    found = {}
    for jp in sorted(glob.glob(os.path.join(OUTDIR, f"{ds}__*__full.json"))):
        arm = os.path.basename(jp)[len(ds) + 2:-len("__full.json")]
        zp = jp[:-4] + "npz"
        if arm == "baselines" or not os.path.exists(zp):
            continue
        with open(jp) as f:
            r = json.load(f)
        if r.get("smoke"):
            continue
        found[arm] = (r, zp)
    ranked = [(a, max(c["delta_mean"] for c in r["cells"].values()))
              for a, (r, _z) in found.items()
              if SC.META[a][3] != "control"]
    ranked.sort(key=lambda t: -t[1])
    if len(ranked) < 2:
        raise SystemExit(
            f"{ds}: only {len(ranked)} non-control arms finished — need at least 2 "
            f"to blend. Run the arms stage first.")
    print(f"  [blend] {len(ranked)} non-control arms available, best first: "
          f"{[a for a, _ in ranked[:6]]}", flush=True)

    vecs = {a: np.load(found[a][1]) for a, _ in ranked}
    keys = [k for k in vecs[ranked[0][0]].files if k.startswith("f")]

    def mean_rho(a, b):
        rs = []
        for k in keys:
            if k not in vecs[b].files:
                continue
            x = vecs[a][k].astype(float); yv = vecs[b][k].astype(float)
            m = (x != 0) | (yv != 0)
            if m.sum() < 3:
                continue
            rs.append(np.nan_to_num(spearmanr(x[m], yv[m])[0], nan=0.0))
        return float(np.mean(rs)) if rs else 0.0

    best = ranked[0][0]
    # top-k, but never spend a slot on an arm that is a near-duplicate of one
    # already in the set (rho > 0.99). That is de-duplication, not selection:
    # lightgbm_gain and lightgbm_gain_nocap are the same numbers whenever no
    # representation exceeds the 2000-feature cap, and fusing a vector with
    # itself changes nothing while displacing a method that would.
    topk = []
    for a, _d in ranked:
        if all(mean_rho(a, m) <= 0.99 for m in topk):
            topk.append(a)
        if len(topk) == top_k:
            break
    members = {f"top{top_k}": topk}
    comp = [best]
    for a, _d in ranked[1:]:
        if all(mean_rho(a, m) < max_rho for m in comp):
            comp.append(a)
        if len(comp) == 3:
            break
    if len(comp) > 1:
        members["complement"] = comp
    print(f"  [blend] members: " + "; ".join(f"{k}={v}" for k, v in members.items()),
          flush=True)

    reps, y, groups, mname, mfn, lower, task, rep_order = load_and_check(ds)
    folds = [(np.asarray(tr), np.asarray(va)) for tr, va in base["folds"]]
    fidx = list(range(len(folds)))
    os.makedirs(OUTDIR, exist_ok=True)

    for set_name, mem in members.items():
        for rule_name, rule in SC.BLENDS.items():
            arm = f"blend_{rule_name}_{set_name}"
            t0 = time.time()
            imp, store = {}, {}
            for fi in fidx:
                for rn in rep_order:
                    k = f"f{fi}__{rn}"
                    stack = [vecs[a][k].astype(float) for a in mem if k in vecs[a].files]
                    if len(stack) != len(mem):
                        raise RuntimeError(
                            f"{arm}: {k} missing from one of {mem} — the arms were "
                            f"not all run on the same folds")
                    v = np.asarray(rule(stack), dtype=float)
                    imp[(fi, rn)] = v
                    store[k] = v.astype(np.float32)
            cells = evaluate_imp(imp, label=arm, base=base, reps=reps, y=y, mfn=mfn,
                                 lower=lower, task=task, folds=folds, fidx=fidx,
                                 allocs=list(ALLOCATIONS),
                                 internals=INTERNAL_EVALUATORS,
                                 downstreams=DOWNSTREAM_MODELS)
            out = {
                "experiment": "f30", "dataset": ds, "arm": arm,
                "family": "blend", "redundancy_aware": False,
                "score_conversion": "rank fusion", "origin": "blend",
                "blend_rule": rule_name, "blend_members": mem,
                "blend_member_set": set_name, "blend_max_rho": max_rho,
                "metric": mname, "lower_is_better": bool(lower), "task_type": task,
                "budget": BUDGET, "allocations": list(ALLOCATIONS),
                "internal_evaluators": INTERNAL_EVALUATORS,
                "downstream_models": DOWNSTREAM_MODELS,
                "n_folds": NFOLDS, "n_inner": N_INNER,
                "rep_widths": base["rep_widths"],
                "rep_is_binary": base["rep_is_binary"],
                # a blend does no scoring of its own; its cost is the sum of the
                # arms it fuses, which are recorded in their own files.
                "timings": [], "seconds_total": round(time.time() - t0, 1),
                "smoke": False, "cells": cells,
            }
            jp = os.path.join(OUTDIR, f"{ds}__{arm}__full.json")
            with open(jp + ".tmp", "w") as f:
                json.dump(out, f)
            os.replace(jp + ".tmp", jp)
            np.savez_compressed(os.path.join(OUTDIR, f"{ds}__{arm}__full.npz"), **store)
            b = max(cells.values(), key=lambda c: c["delta_mean"])
            print(f"  -> {arm}: best delta_mean={b['delta_mean']:+.4f} "
                  f"({b['folds_won']} folds won)  members={mem}", flush=True)


def run_arms(ds, arms, smoke=False, n_jobs=1, skip_engine_check=False):
    base_path = os.path.join(OUTDIR, f"{ds}__baselines.json")
    if not os.path.exists(base_path):
        raise SystemExit(
            f"missing {base_path} — run the baselines stage for {ds} first:\n"
            f"  python -u {__file__} {ds} --stage baselines")
    with open(base_path) as f:
        base = json.load(f)

    reps, y, groups, mname, mfn, lower, task, rep_order = load_and_check(ds)
    assert mname == base["metric"] and bool(lower) == base["lower_is_better"]
    assert rep_order == base["rep_order"], "representation order drifted since stage A"
    folds = [(np.asarray(tr), np.asarray(va)) for tr, va in base["folds"]]

    fold_subset = [0] if smoke else None
    alloc_subset = ["power_2"] if smoke else None
    model_subset = ["lightgbm"] if smoke else None

    # The engine check proves this file's per-representation top-k equals
    # create_hybrid('fixed'). It is a property of the CODE, not of the arm, so it
    # only needs running once per dataset — and it is expensive (a full
    # create_hybrid call, single-threaded, ~1h on caco2). Cached to disk so a run
    # split into several invocations pays for it once.
    ck = os.path.join(OUTDIR, f"{ds}__engine_check.json")
    if os.path.exists(ck):
        with open(ck) as f:
            eng = json.load(f)
        print(f"  [engine-check] cached: {eng}", flush=True)
    elif skip_engine_check:
        eng = {"ran": False, "reason": "skipped by --skip-engine-check"}
        print("  [engine-check] skipped", flush=True)
    else:
        eng = validate_against_engine(reps, y, folds, rep_order)
        with open(ck + ".tmp", "w") as f:
            json.dump(eng, f)
        os.replace(ck + ".tmp", ck)
        print(f"  [engine-check] {eng}  (cached to {ck})", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    summary = {}
    for arm in arms:
        t0 = time.time()
        print(f"  == {arm} ==", flush=True)
        try:
            cells, timings, store, _ = run_arm(
                ds, arm, base, reps, y, mfn, lower, task, folds, n_jobs=n_jobs,
                fold_subset=fold_subset, alloc_subset=alloc_subset,
                model_subset=model_subset)
        except Exception as e:
            # RAISES rather than silently dropping (cf. find_best_hybrid.py:4473),
            # but records what failed first so a grid run is diagnosable.
            rec = {"arm": arm, "dataset": ds, "failed": True,
                   "error": f"{type(e).__name__}: {e}"}
            with open(os.path.join(OUTDIR, f"{ds}__{arm}__FAILED.json"), "w") as f:
                json.dump(rec, f, indent=2)
            raise
        fam, redundancy_aware, rule, origin = SC.META[arm]
        out = {
            "experiment": "f30", "dataset": ds, "arm": arm,
            "family": fam, "redundancy_aware": redundancy_aware,
            "score_conversion": rule, "origin": origin,
            "metric": mname, "lower_is_better": bool(lower), "task_type": task,
            "budget": BUDGET, "allocations": alloc_subset or list(ALLOCATIONS),
            "internal_evaluators": model_subset or INTERNAL_EVALUATORS,
            "downstream_models": model_subset or DOWNSTREAM_MODELS,
            "n_folds": NFOLDS, "n_inner": N_INNER,
            "folds_run": fold_subset if fold_subset is not None else "all",
            "smoke": bool(smoke),
            "rep_widths": base["rep_widths"], "rep_is_binary": base["rep_is_binary"],
            "prescreen_n_cand": SC.N_CAND, "relief_m": SC.RELIEF_M,
            "n_bins": SC.N_BINS, "n_boot": SC.N_BOOT,
            "engine_validation": eng,
            "timings": timings,
            "seconds_total": round(time.time() - t0, 1),
            "cells": cells,
        }
        tag = "smoke" if smoke else "full"
        jp = os.path.join(OUTDIR, f"{ds}__{arm}__{tag}.json")
        with open(jp + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(jp + ".tmp", jp)
        np.savez_compressed(os.path.join(OUTDIR, f"{ds}__{arm}__{tag}.npz"), **store)
        best = max(cells.values(), key=lambda c: c["delta_mean"])
        summary[arm] = {"seconds": out["seconds_total"],
                        "best_delta_mean": best["delta_mean"],
                        "best_folds_won": best["folds_won"]}
        print(f"  -> {arm}: {out['seconds_total']}s  best delta_mean="
              f"{best['delta_mean']:+.4f} ({best['folds_won']} folds won)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=DATASETS)
    ap.add_argument("--stage", choices=["baselines", "arms", "blends"],
                    default="arms")
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm names; default = every pass-1 arm")
    ap.add_argument("--smoke", action="store_true",
                    help="one fold, one allocation, one downstream model")
    ap.add_argument("--check", action="store_true",
                    help="gates and shape checks only, loads no representation")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--skip-engine-check", action="store_true",
                    help="skip the create_hybrid equivalence check; it is cached "
                         "per dataset once it has run")
    a = ap.parse_args()

    if a.check:
        print("f30 pre-flight")
        for ds in DATASETS:
            gate_metric_direction(ds)
        missing = [m for m in SC.PASS1 if m not in SC.META]
        assert not missing, f"arms with no META entry: {missing}"
        print(f"  [arms] {len(SC.PASS1)} registered, all have metadata")
        for ds in DATASETS:
            want = DATASET_REPS[ds]["general"]
            assert len(want) == 10, f"{ds}: DATASET_REPS has {len(want)}, expected 10"
            print(f"  [reps] {ds:24s} 10 named in complexity_helpers")
        print("PRE-FLIGHT PASSED")
        return

    if not a.dataset:
        raise SystemExit("a dataset is required unless --check")
    print(f"f30 stage={a.stage} dataset={a.dataset} smoke={a.smoke}")
    print(f"  KIRBY_CACHE_DIR={os.environ.get('KIRBY_CACHE_DIR')}")
    print(f"  OUTDIR={OUTDIR}")
    if a.stage == "baselines":
        run_baselines(a.dataset)
    elif a.stage == "blends":
        run_blends(a.dataset, n_jobs=a.n_jobs)
    else:
        arms = a.arms.split(",") if a.arms else list(SC.PASS1)
        unknown = [m for m in arms if m not in SC.SCORERS]
        if unknown:
            raise SystemExit(f"unknown arms: {unknown}")
        run_arms(a.dataset, arms, smoke=a.smoke, n_jobs=a.n_jobs,
                 skip_engine_check=a.skip_engine_check)


if __name__ == "__main__":
    main()
