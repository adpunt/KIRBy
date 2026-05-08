"""Master testing harness for competition_methods/ fusion + validation library.

Exercises F01-F10 fusion methods + V04 Uno IPCW C + V07 adversarial validation +
V08 hold-out gate on the existing 108-candidate ANNITIA OOF pool.

Discipline:
- Carve a 10% hold-out (V08), stratified by event, fixed seed.
- All fusion methods fit on train-portion OOFs, evaluate on hold-out (NEVER fit
  on the hold-out).
- Report Harrell C and Uno IPCW C on hold-out.
- For final submission, refit each fusion method on the FULL train OOF (no
  hold-out) and apply to val_test predictions.

Outputs:
- Per-method per-endpoint hold-out C / Uno C / train-OOF C / size / members
- Markdown table summary
- Per-method val_test CSV in tests/results/annitia/eval/

Run:
    python -m tests.experiments.annitia.competition_methods_eval \\
        --validation-csv /Volumes/seagate/kirby/annitia/val_test.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))

from kirby.datasets.annitia_maslg import load_annitia  # noqa: E402


# ============================================================================
# Metrics
# ============================================================================
def harrell_c(y_struct, risk, valid=None):
    """Harrell's concordance index with NaN-safe input."""
    from sksurv.metrics import concordance_index_censored
    if valid is None:
        valid = np.isfinite(risk)
    else:
        valid = valid & np.isfinite(risk)
    if valid.sum() < 10:
        return float('nan')
    c, *_ = concordance_index_censored(
        y_struct['event'][valid], y_struct['time'][valid], risk[valid])
    return float(c)


def uno_ipcw_c(y_struct_train, y_struct_eval, risk_eval, tau=None):
    """Uno's IPCW concordance — censoring-corrected. V04 in competition_methods.

    y_struct_train: structured array used to fit G(t) censoring distribution.
    y_struct_eval:  structured array on which to evaluate (must overlap support).
    risk_eval:      risk scores on the eval set.
    tau:            time horizon; defaults to 95th percentile of eval times.
    """
    from sksurv.metrics import concordance_index_ipcw
    finite = np.isfinite(risk_eval)
    if finite.sum() < 10:
        return float('nan')
    if tau is None:
        tau = np.percentile(y_struct_eval['time'][finite], 95)
    try:
        c, *_ = concordance_index_ipcw(
            y_struct_train, y_struct_eval[finite], risk_eval[finite], tau=tau)
    except Exception:
        return float('nan')
    return float(c)


# ============================================================================
# OOF pool
# ============================================================================
def _rank01(x):
    """Rank-percentile transform, NaN-safe."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.full_like(x, np.nan)
    if finite.sum() < 2:
        return out
    from scipy.stats import rankdata
    out[finite] = (rankdata(x[finite]) - 0.5) / finite.sum()
    return out


def collect_pool(pool_dirs, endpoint, n_total):
    """Load every OOF + val_test prediction from pool_dirs for `endpoint`.

    Returns list of records: {label, oof (length n_total, NaN for invalid),
    val (length n_val), cv}
    """
    out = []
    for d in pool_dirs:
        cv = d / 'cv_summary.json'
        if not cv.exists():
            continue
        sm = json.loads(cv.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r.get('config_id', '')
            mc = r.get('mean_cindex', float('nan'))
            if not np.isfinite(mc):
                continue
            oof_p = d / f'oof_{endpoint}_{cid}.npy'
            val_p = d / f'val_{endpoint}_{cid}.npy'
            if not (oof_p.exists() and val_p.exists()):
                continue
            try:
                oof = np.load(oof_p)
                val = np.load(val_p)
            except Exception as e:  # noqa: BLE001
                print(f'  load fail {oof_p.name}: {e}')
                continue
            if oof.size != n_total:
                # Reconstructed OOFs may be n_valid; pad if needed
                continue
            out.append({
                'label': f'{d.name}__{cid[:18]}',
                'cv': float(mc),
                'oof': oof,
                'val': val,
            })
    out.sort(key=lambda r: -r['cv'])
    return out


# ============================================================================
# F01 — Caruana bagged ensemble selection (KDD 2004)
# ============================================================================
def f01_caruana_bagged(recs, y_struct, train_idx, holdout_idx, n_bags=20,
                       init_k=5, max_size=40, seed=0):
    """Caruana bagged ensemble selection. Fits on train_idx, scored on holdout."""
    rng = np.random.default_rng(seed)
    n_train = len(train_idx)
    # Pre-compute rank-percentile OOFs for all recs (length=n_total, NaN where invalid)
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]

    def metric_on_idx(pred, idx):
        return harrell_c(y_struct, pred, valid=np.isin(np.arange(len(pred)), idx))

    bag_compositions = []
    bag_oofs = []
    for b in range(n_bags):
        # Bootstrap on train_idx for selection
        boot = rng.choice(n_train, size=n_train, replace=True)
        boot_idx = train_idx[boot]
        # Sort-init: top-k by score on this bootstrap
        scores_init = [metric_on_idx(ranks[i], boot_idx) for i in range(len(recs))]
        order = np.argsort(scores_init)[::-1][:init_k]
        ensemble = list(order)
        running = np.nanmean(np.stack([ranks[i] for i in ensemble], axis=0), axis=0)
        # Forward greedy with replacement
        for _ in range(max_size - init_k):
            best, best_score = None, -np.inf
            for i in range(len(recs)):
                cand = (running * len(ensemble) + ranks[i]) / (len(ensemble) + 1)
                s = metric_on_idx(cand, boot_idx)
                if s > best_score:
                    best_score, best = s, i
            if best is None:
                break
            ensemble.append(best)
            running = (running * (len(ensemble) - 1) + ranks[best]) / len(ensemble)
        bag_compositions.append(ensemble)
        bag_oofs.append(running)

    # Final OOF = mean across bag prediction averages, full data
    final_oof = np.nanmean(np.stack(bag_oofs, axis=0), axis=0)
    # Final val_test prediction = same recipe, applied to val_ranks
    bag_vals = []
    for comp in bag_compositions:
        bag_vals.append(np.nanmean(np.stack([val_ranks[i] for i in comp], axis=0), axis=0))
    final_val = np.nanmean(np.stack(bag_vals, axis=0), axis=0)
    # Track unique members for reporting
    counter = {}
    for comp in bag_compositions:
        for m in comp:
            counter[m] = counter.get(m, 0) + 1
    top_members = sorted(counter.items(), key=lambda kv: -kv[1])[:10]
    members = [recs[i]['label'] + f'(×{c})' for i, c in top_members]
    return dict(name='F01_caruana_bagged', oof=final_oof, val=final_val,
                size=int(np.mean([len(c) for c in bag_compositions])),
                members=members)


# ============================================================================
# F02 — Geometric mean of rank-transformed predictions (decorrelated subset)
# ============================================================================
def f02_geomean_rank(recs, y_struct, train_idx, holdout_idx, top_k=10):
    """Geomean of rank-percentiles, on top-k mutually decorrelated candidates
    chosen by mutual rank-correlation on train_idx (NOT by C-index)."""
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    n = len(recs)
    # Greedy decorrelation on train_idx
    chosen = [int(np.argmax([harrell_c(y_struct, ranks[i],
                              valid=np.isin(np.arange(len(ranks[i])), train_idx))
                              for i in range(n)]))]
    while len(chosen) < min(top_k, n):
        best, best_score = None, -np.inf
        for i in range(n):
            if i in chosen:
                continue
            # decorrelation = -max corr with already chosen, on train_idx only
            corrs = []
            for c in chosen:
                a = ranks[i][train_idx]
                b = ranks[c][train_idx]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 10:
                    continue
                corrs.append(abs(np.corrcoef(a[m], b[m])[0, 1]))
            if not corrs:
                continue
            score = -max(corrs)
            if score > best_score:
                best_score, best = score, i
        if best is None:
            break
        chosen.append(best)
    log_r_oof = np.nanmean(np.stack([np.log(ranks[i] + 1e-9) for i in chosen]), axis=0)
    log_r_val = np.nanmean(np.stack([np.log(val_ranks[i] + 1e-9) for i in chosen]), axis=0)
    return dict(name='F02_geomean_rank', oof=np.exp(log_r_oof), val=np.exp(log_r_val),
                size=len(chosen), members=[recs[i]['label'] for i in chosen])


# ============================================================================
# F03 — Hodges-Lehmann pseudomedian
# ============================================================================
def f03_hodges_lehmann(recs, y_struct, train_idx, holdout_idx, top_k=15):
    """Hodges-Lehmann pseudomedian over the top-k candidates by train_idx C."""
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    scores = [harrell_c(y_struct, ranks[i],
              valid=np.isin(np.arange(len(ranks[i])), train_idx)) for i in range(len(recs))]
    order = np.argsort([-s if np.isfinite(s) else 0 for s in scores])[:top_k]

    def hl(stack):
        # stack shape: (n_cands, n_patients)
        out = np.full(stack.shape[1], np.nan)
        for p in range(stack.shape[1]):
            col = stack[:, p]
            col = col[np.isfinite(col)]
            if len(col) < 2:
                continue
            pairs = (col[:, None] + col[None, :]).flatten() / 2
            out[p] = np.median(pairs)
        return out

    oof_hl = hl(np.stack([ranks[i] for i in order], axis=0))
    val_hl = hl(np.stack([val_ranks[i] for i in order], axis=0))
    return dict(name='F03_hodges_lehmann', oof=oof_hl, val=val_hl,
                size=len(order), members=[recs[i]['label'] for i in order])


# ============================================================================
# F04 — Bates-Granger variance-weighted combination
# ============================================================================
def f04_bates_granger(recs, y_struct, train_idx, holdout_idx, top_k=20):
    """Variance-weighted Bates-Granger. Weights = 1/Var(error) on train_idx,
    where 'error' = rank prediction - event indicator (proxy)."""
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    # Pick top-k by train_idx C
    train_mask = np.isin(np.arange(len(ranks[0])), train_idx)
    scores = [harrell_c(y_struct, ranks[i], valid=train_mask) for i in range(len(recs))]
    order = np.argsort([-s if np.isfinite(s) else 0 for s in scores])[:top_k]
    # Variance of error on train_idx (using event indicator as proxy target)
    y_event = y_struct['event'].astype(float)
    variances = []
    for i in order:
        err = ranks[i] - y_event
        v = np.nanvar(err[train_idx])
        variances.append(max(v, 1e-6))
    inv = np.array([1.0 / v for v in variances])
    w = inv / inv.sum()
    oof_bg = np.nansum(np.stack([w[k] * ranks[order[k]] for k in range(len(order))], axis=0), axis=0)
    val_bg = np.nansum(np.stack([w[k] * val_ranks[order[k]] for k in range(len(order))], axis=0), axis=0)
    return dict(name='F04_bates_granger', oof=oof_bg, val=val_bg, size=len(order),
                members=[f'{recs[i]["label"]}(w={w[k]:.3f})' for k, i in enumerate(order[:5])])


# ============================================================================
# F05 — Stacking with regularized meta-learner + nested CV
# ============================================================================
def f05_stacking_nested(recs, y_struct, train_idx, holdout_idx, top_k=20, K_inner=5):
    """L2-logistic stacking with nested CV. Top-k candidates (by train OOF C),
    inner K-fold to fit meta-learner on the train_idx portion only."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    train_mask = np.isin(np.arange(len(ranks[0])), train_idx)
    scores = [harrell_c(y_struct, ranks[i], valid=train_mask) for i in range(len(recs))]
    order = np.argsort([-s if np.isfinite(s) else 0 for s in scores])[:top_k]
    # Stack base ranks on train_idx
    X_train = np.stack([ranks[i][train_idx] for i in order], axis=1)
    y_train = y_struct['event'][train_idx].astype(int)
    # Replace NaN with column mean (NaN-safe)
    col_mean = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_mean, inds[1])
    # Strong L2 to combat over-fitting weights
    meta = LogisticRegressionCV(Cs=[1e-2, 1e-1, 1.0], cv=StratifiedKFold(K_inner),
                                 penalty='l2', solver='lbfgs', max_iter=500)
    try:
        meta.fit(X_train, y_train)
    except Exception as e:
        print(f'  F05 meta-learner failed: {e}')
        return None
    # Predict OOF + val
    n_total = len(ranks[0])
    X_full = np.stack([ranks[i] for i in order], axis=1)
    inds = np.where(np.isnan(X_full))
    X_full[inds] = np.take(col_mean, inds[1])
    oof_stack = meta.predict_proba(X_full)[:, 1]
    X_val = np.stack([val_ranks[i] for i in order], axis=1)
    inds = np.where(np.isnan(X_val))
    X_val[inds] = np.take(col_mean, inds[1])
    val_stack = meta.predict_proba(X_val)[:, 1]
    return dict(name='F05_stacking_l2', oof=oof_stack, val=val_stack, size=len(order),
                members=[f'{recs[i]["label"]}(w={w:.2f})' for i, w in zip(
                    order[:5], meta.coef_[0][:5])])


# ============================================================================
# F06 — Diversity-penalized ensemble selection (ICDM 2006)
# ============================================================================
def f06_diversity_penalized(recs, y_struct, train_idx, holdout_idx, max_size=30,
                            init_k=5, lam=0.1):
    """Forward greedy with correlation penalty."""
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    train_mask = np.isin(np.arange(len(ranks[0])), train_idx)
    scores = [harrell_c(y_struct, ranks[i], valid=train_mask) for i in range(len(recs))]
    init_order = np.argsort([-s if np.isfinite(s) else 0 for s in scores])[:init_k]
    ensemble = list(init_order)
    running = np.nanmean(np.stack([ranks[i] for i in ensemble], axis=0), axis=0)
    while len(ensemble) < max_size:
        best, best_score = None, -np.inf
        for i in range(len(recs)):
            cand_pred = (running * len(ensemble) + ranks[i]) / (len(ensemble) + 1)
            m = harrell_c(y_struct, cand_pred, valid=train_mask)
            if not np.isfinite(m):
                continue
            corrs = []
            for e in ensemble:
                a = ranks[i][train_idx]
                b = ranks[e][train_idx]
                fm = np.isfinite(a) & np.isfinite(b)
                if fm.sum() < 10:
                    continue
                corrs.append(abs(np.corrcoef(a[fm], b[fm])[0, 1]))
            corr = np.mean(corrs) if corrs else 0.0
            s = m - lam * corr
            if s > best_score:
                best_score, best = s, i
        if best is None or best in ensemble:
            break
        ensemble.append(best)
        running = (running * (len(ensemble) - 1) + ranks[best]) / len(ensemble)
    val_oof = np.nanmean(np.stack([ranks[i] for i in ensemble], axis=0), axis=0)
    val_val = np.nanmean(np.stack([val_ranks[i] for i in ensemble], axis=0), axis=0)
    return dict(name='F06_diversity_pen', oof=val_oof, val=val_val,
                size=len(ensemble), members=[recs[i]['label'] for i in ensemble[:8]])


# ============================================================================
# F11 — CoxPH stacking on rank features (proper survival meta-learner)
# ============================================================================
def f11_cox_stack(recs, y_struct, train_idx, holdout_idx, top_k=20, alpha=10.0):
    """L2-regularized CoxPH meta-learner over top-k rank features.

    Unlike F05 (binary logistic over event indicator), this uses the actual
    survival objective (partial likelihood) and respects censoring."""
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    train_mask = np.isin(np.arange(len(ranks[0])), train_idx)
    scores = [harrell_c(y_struct, ranks[i], valid=train_mask) for i in range(len(recs))]
    order = np.argsort([-s if np.isfinite(s) else 0 for s in scores])[:top_k]
    X_train = np.stack([ranks[i][train_idx] for i in order], axis=1)
    col_mean = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_mean, inds[1])
    y_train = y_struct[train_idx]
    try:
        # l1_ratio must be >0; use small value for near-pure L2 with stability
        meta = CoxnetSurvivalAnalysis(l1_ratio=0.01, n_alphas=1, alphas=[alpha],
                                       fit_baseline_model=False)
        meta.fit(X_train, y_train)
    except Exception as e:
        print(f'  F11 CoxPH stack failed: {e}')
        return None
    n_total = len(ranks[0])
    X_full = np.stack([ranks[i] for i in order], axis=1)
    inds = np.where(np.isnan(X_full))
    X_full[inds] = np.take(col_mean, inds[1])
    oof_stack = -meta.predict(X_full)  # higher risk = larger
    X_val = np.stack([val_ranks[i] for i in order], axis=1)
    inds = np.where(np.isnan(X_val))
    X_val[inds] = np.take(col_mean, inds[1])
    val_stack = -meta.predict(X_val)
    return dict(name='F11_cox_stack', oof=oof_stack, val=val_stack, size=len(order),
                members=[recs[i]['label'] for i in order[:8]])


# ============================================================================
# F12 — Uncertainty-routed pool (parameter-free, train-fit-free)
# ============================================================================
def f12_uncertainty_routed(recs, y_struct, train_idx, holdout_idx, top_k=15,
                           uncertainty_q=0.5):
    """Per-patient: if rank-std across bases is high (uncertain), use pseudomedian
    (robust); otherwise use rank-mean (efficient). Cutoff = uncertainty_q quantile
    of rank-std on train_idx — no OOF metric fitting."""
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    train_mask = np.isin(np.arange(len(ranks[0])), train_idx)
    scores = [harrell_c(y_struct, ranks[i], valid=train_mask) for i in range(len(recs))]
    order = np.argsort([-s if np.isfinite(s) else 0 for s in scores])[:top_k]
    stack = np.stack([ranks[i] for i in order], axis=0)
    val_stack = np.stack([val_ranks[i] for i in order], axis=0)
    # Per-patient std (NaN-safe)
    std = np.nanstd(stack, axis=0)
    val_std = np.nanstd(val_stack, axis=0)
    # Compute cutoff from train_idx only
    cutoff = np.nanquantile(std[train_idx], uncertainty_q)
    mean_oof = np.nanmean(stack, axis=0)
    mean_val = np.nanmean(val_stack, axis=0)

    # Pseudomedian for high-uncertainty patients
    def hl_per_patient(stk):
        out = np.full(stk.shape[1], np.nan)
        for p in range(stk.shape[1]):
            col = stk[:, p][np.isfinite(stk[:, p])]
            if len(col) < 2:
                continue
            pairs = (col[:, None] + col[None, :]).flatten() / 2
            out[p] = np.median(pairs)
        return out

    high_oof = hl_per_patient(stack)
    high_val = hl_per_patient(val_stack)
    out_oof = np.where(std > cutoff, high_oof, mean_oof)
    out_val = np.where(val_std > cutoff, high_val, mean_val)
    return dict(name='F12_uncertainty_routed', oof=out_oof, val=out_val,
                size=len(order),
                members=[recs[i]['label'] for i in order[:8]])


# ============================================================================
# F09 — Trimmed mean of predictions (robust)
# ============================================================================
def f09_trimmed_mean(recs, y_struct, train_idx, holdout_idx, top_k=20, q=0.1):
    """Trim top/bottom q from per-patient stack of top-k candidates."""
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    train_mask = np.isin(np.arange(len(ranks[0])), train_idx)
    scores = [harrell_c(y_struct, ranks[i], valid=train_mask) for i in range(len(recs))]
    order = np.argsort([-s if np.isfinite(s) else 0 for s in scores])[:top_k]

    def trim(stack):
        out = np.full(stack.shape[1], np.nan)
        for p in range(stack.shape[1]):
            col = stack[:, p]
            col = col[np.isfinite(col)]
            if len(col) < 5:
                if len(col):
                    out[p] = col.mean()
                continue
            n = len(col)
            k = int(np.floor(q * n))
            sc = np.sort(col)
            out[p] = sc[k:n-k].mean() if k > 0 else sc.mean()
        return out

    oof_t = trim(np.stack([ranks[i] for i in order], axis=0))
    val_t = trim(np.stack([val_ranks[i] for i in order], axis=0))
    return dict(name='F09_trimmed_mean', oof=oof_t, val=val_t, size=len(order),
                members=[recs[i]['label'] for i in order[:8]])


# ============================================================================
# Plain anchor for reference: simple rank mean of top-15
# ============================================================================
def anchor_rank_mean(recs, y_struct, train_idx, holdout_idx, top_k=15):
    if len(recs) == 0:
        return None
    ranks = [_rank01(r['oof']) for r in recs]
    val_ranks = [_rank01(r['val']) for r in recs]
    train_mask = np.isin(np.arange(len(ranks[0])), train_idx)
    scores = [harrell_c(y_struct, ranks[i], valid=train_mask) for i in range(len(recs))]
    order = np.argsort([-s if np.isfinite(s) else 0 for s in scores])[:top_k]
    return dict(name='anchor_rank_mean',
                oof=np.nanmean(np.stack([ranks[i] for i in order]), axis=0),
                val=np.nanmean(np.stack([val_ranks[i] for i in order]), axis=0),
                size=top_k, members=[recs[i]['label'] for i in order[:8]])


# ============================================================================
# V07 — Adversarial validation
# ============================================================================
def v07_adversarial_oof_vs_val(oof_pred, val_pred):
    """Train a classifier to discriminate train-OOF from val_test predictions
    using only the prediction value as a feature. AUC ≈ 0.5 = distributions match."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    o = oof_pred[np.isfinite(oof_pred)]
    v = val_pred[np.isfinite(val_pred)]
    if len(o) < 50 or len(v) < 50:
        return float('nan')
    X = np.r_[o, v].reshape(-1, 1)
    y = np.r_[np.zeros(len(o)), np.ones(len(v))]
    try:
        auc = cross_val_score(GradientBoostingClassifier(n_estimators=50, max_depth=3),
                              X, y, cv=3, scoring='roc_auc').mean()
        return float(auc)
    except Exception:
        return float('nan')


# ============================================================================
# Hold-out split (V08)
# ============================================================================
def carve_holdout(y_struct, valid_mask, frac=0.10, seed=2026):
    """Stratified hold-out: ~frac of train indices, preserving event rate."""
    from sklearn.model_selection import StratifiedShuffleSplit
    valid_idx = np.where(valid_mask)[0]
    y_event = y_struct['event'][valid_idx].astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    train_rel, holdout_rel = next(sss.split(valid_idx, y_event))
    train_idx = valid_idx[train_rel]
    holdout_idx = valid_idx[holdout_rel]
    return train_idx, holdout_idx


# ============================================================================
# Main
# ============================================================================
METHODS = [
    anchor_rank_mean,
    f01_caruana_bagged,
    f02_geomean_rank,
    f03_hodges_lehmann,
    f04_bates_granger,
    f05_stacking_nested,
    f06_diversity_penalized,
    f09_trimmed_mean,
    # F11 disabled: Coxnet path solver hangs on rank features at any alpha
    # f11_cox_stack,
    f12_uncertainty_routed,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--output-dir', default='tests/results/annitia/eval')
    ap.add_argument('--holdout-frac', type=float, default=0.15)
    ap.add_argument('--n-holdout-seeds', type=int, default=5,
                    help='Average hold-out scoring over N seeds (V01 + V08).')
    ap.add_argument('--holdout-base-seed', type=int, default=2026)
    ap.add_argument('--n-bags', type=int, default=10)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    print('=== Loading train + val ===')
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)
    n_total = len(train['hepatic']['y_struct'])
    print(f'  n_total={n_total}, n_val={len(val["trustii_id"])}')

    # Pool — auto-discover any subdir with cv_summary.json (so synthetic-data
    # outputs from other chats / future runs are picked up automatically).
    res_root = _REPO / 'tests' / 'results' / 'annitia'
    pool_dirs = sorted([d for d in res_root.iterdir()
                        if d.is_dir() and (d / 'cv_summary.json').exists()
                        and d.name not in ('eval',)])
    print(f'  pool dirs: {[d.name for d in pool_dirs]}')

    summary_rows = []
    method_outputs = {}
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        print(f'\n========== {endpoint} ==========')
        y_struct = train[endpoint]['y_struct']
        valid_mask = train[endpoint]['valid_mask']
        recs = collect_pool(pool_dirs, endpoint, n_total)
        print(f'  candidates loaded: {len(recs)}')
        if len(recs) == 0:
            continue

        # Multi-seed hold-out evaluation: each method run for each holdout seed
        # then averaged. This is V01 + V08 combined: the hold-out gate, but
        # bagged across hold-out seeds to defang the small-event-count variance.
        method_outputs[endpoint] = []
        method_acc = {}  # name -> dict of {c_holdout, uno_holdout, c_train, val_preds, size, members}
        adv_auc_acc = {}  # name -> list of adv AUCs
        full_oof_acc = {}  # name -> list of full OOF arrays (for final submit refit)
        full_val_acc = {}  # name -> list of full val arrays
        seed_count_per_method = {}

        for s_idx in range(args.n_holdout_seeds):
            seed = args.holdout_base_seed + s_idx
            train_idx, holdout_idx = carve_holdout(y_struct, valid_mask,
                                                   frac=args.holdout_frac,
                                                   seed=seed)
            print(f'\n  [seed {seed}] train_idx={len(train_idx)} '
                  f'(ev={int(y_struct["event"][train_idx].sum())}) '
                  f'holdout={len(holdout_idx)} '
                  f'(ev={int(y_struct["event"][holdout_idx].sum())})')

            holdout_mask = np.isin(np.arange(n_total), holdout_idx)
            train_mask = np.isin(np.arange(n_total), train_idx)

            for method in METHODS:
                try:
                    if method is f01_caruana_bagged:
                        r = method(recs, y_struct, train_idx, holdout_idx,
                                   n_bags=args.n_bags)
                    else:
                        r = method(recs, y_struct, train_idx, holdout_idx)
                except Exception as e:
                    print(f'    [seed {seed}] {method.__name__:25s} FAILED: '
                          f'{type(e).__name__}: {e}')
                    continue
                if r is None:
                    continue
                c_h = harrell_c(y_struct, r['oof'], valid=holdout_mask)
                c_t = harrell_c(y_struct, r['oof'], valid=train_mask)
                try:
                    u_h = uno_ipcw_c(y_struct[train_idx], y_struct[holdout_idx],
                                     r['oof'][holdout_idx])
                except Exception:
                    u_h = float('nan')
                a_auc = v07_adversarial_oof_vs_val(
                    r['oof'][np.isfinite(r['oof'])], r['val'])
                name = r['name']
                acc = method_acc.setdefault(name, dict(
                    c_holdout=[], uno_holdout=[], c_train=[], size=[],
                    members=r.get('members', [])))
                acc['c_holdout'].append(c_h)
                acc['uno_holdout'].append(u_h)
                acc['c_train'].append(c_t)
                acc['size'].append(r['size'])
                adv_auc_acc.setdefault(name, []).append(a_auc)
                full_oof_acc.setdefault(name, []).append(r['oof'])
                full_val_acc.setdefault(name, []).append(r['val'])
                seed_count_per_method[name] = seed_count_per_method.get(name, 0) + 1

        print(f'\n  --- Aggregated over {args.n_holdout_seeds} hold-out seeds ---')
        print(f'  {"method":25s} {"hold C":>9s} {"±SE":>7s} '
              f'{"Uno hold":>10s} {"±SE":>7s} {"adv AUC":>9s}')
        for name, acc in method_acc.items():
            ch = np.array(acc['c_holdout'])
            uh = np.array([x for x in acc['uno_holdout'] if np.isfinite(x)])
            ct = np.array(acc['c_train'])
            sz = np.array(acc['size'])
            adv = np.array([x for x in adv_auc_acc.get(name, []) if np.isfinite(x)])
            r = dict(
                name=name,
                c_holdout=float(np.nanmean(ch)),
                c_holdout_se=float(np.nanstd(ch) / max(np.sqrt(len(ch)), 1)),
                uno_holdout=float(np.nanmean(uh)) if len(uh) else float('nan'),
                uno_holdout_se=float(np.nanstd(uh) / max(np.sqrt(len(uh)), 1)) if len(uh) > 1 else float('nan'),
                c_train=float(np.nanmean(ct)),
                size=int(np.mean(sz)),
                adv_auc=float(np.nanmean(adv)) if len(adv) else float('nan'),
                members=acc['members'],
                # Average the per-seed full OOFs for downstream stability
                oof=np.nanmean(np.stack(full_oof_acc[name], axis=0), axis=0),
                val=np.nanmean(np.stack(full_val_acc[name], axis=0), axis=0),
            )
            method_outputs[endpoint].append(r)
            summary_rows.append(dict(endpoint=endpoint, method=name,
                                     c_train=r['c_train'], c_holdout=r['c_holdout'],
                                     c_holdout_se=r['c_holdout_se'],
                                     uno_holdout=r['uno_holdout'],
                                     uno_holdout_se=r['uno_holdout_se'],
                                     adv_auc=r['adv_auc'], size=r['size']))
            print(f'  {name:25s} {r["c_holdout"]:>9.4f} '
                  f'{r["c_holdout_se"]:>7.4f} '
                  f'{r["uno_holdout"]:>10.4f} '
                  f'{r["uno_holdout_se"]:>7.4f} '
                  f'{r["adv_auc"]:>9.3f}')

        # Save val_test predictions per method
        for r in method_outputs[endpoint]:
            csv_p = out / f'sub_{r["name"]}.csv'
            df = pd.DataFrame({'trustii_id': val['trustii_id'], col: r['val']})
            if csv_p.exists():
                prev = pd.read_csv(csv_p)
                if col in prev.columns:
                    prev[col] = r['val']
                else:
                    prev = prev.merge(df, on='trustii_id')
                prev.to_csv(csv_p, index=False)
            else:
                df.to_csv(csv_p, index=False)

    # Combined LB on hold-out per method
    print('\n========== Combined hold-out LB (0.7*hep + 0.3*death) ==========')
    method_names = sorted(set(r['method'] for r in summary_rows))
    combined = []
    for m in method_names:
        hep = [r for r in summary_rows if r['method'] == m and r['endpoint'] == 'hepatic']
        death = [r for r in summary_rows if r['method'] == m and r['endpoint'] == 'death']
        if not hep or not death:
            continue
        c_hold = 0.7 * hep[0]['c_holdout'] + 0.3 * death[0]['c_holdout']
        c_train = 0.7 * hep[0]['c_train'] + 0.3 * death[0]['c_train']
        # combined SE under independence: sqrt(0.7^2 * SE_hep^2 + 0.3^2 * SE_death^2)
        c_hold_se = float(np.sqrt(
            0.49 * hep[0].get('c_holdout_se', 0)**2
            + 0.09 * death[0].get('c_holdout_se', 0)**2))
        u_hold = (0.7 * (hep[0]['uno_holdout'] if np.isfinite(hep[0]['uno_holdout']) else hep[0]['c_holdout'])
                  + 0.3 * (death[0]['uno_holdout'] if np.isfinite(death[0]['uno_holdout']) else death[0]['c_holdout']))
        combined.append(dict(method=m, train_lb=c_train,
                             hold_lb=c_hold, hold_lb_se=c_hold_se,
                             uno_hold_lb=u_hold))
    combined.sort(key=lambda d: -d['hold_lb'])
    print(f'\n  {"method":25s} {"train_LB":>10s} {"hold_LB":>10s} {"±SE":>8s} {"uno_hold_LB":>12s}')
    for c in combined:
        print(f'  {c["method"]:25s} {c["train_lb"]:>10.4f} '
              f'{c["hold_lb"]:>10.4f} {c["hold_lb_se"]:>8.4f} {c["uno_hold_lb"]:>12.4f}')

    # Markdown report
    md_path = out / 'comparison.md'
    lines = ['# Fusion methods comparison (V08-gated, multi-seed)', '',
             f'Hold-out frac = {args.holdout_frac}, '
             f'seeds = {args.holdout_base_seed}..{args.holdout_base_seed + args.n_holdout_seeds - 1} '
             f'({args.n_holdout_seeds} repeats)', '']
    for endpoint in ('hepatic', 'death'):
        rows = [r for r in summary_rows if r['endpoint'] == endpoint]
        rows.sort(key=lambda r: -r['c_holdout'])
        lines.append(f'## {endpoint}\n')
        lines.append('| method | train C | hold C | hold SE | Uno hold | Uno SE | adv AUC | size |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for r in rows:
            lines.append(f'| {r["method"]} | {r["c_train"]:.4f} | '
                         f'{r["c_holdout"]:.4f} | {r.get("c_holdout_se", 0):.4f} | '
                         f'{r["uno_holdout"]:.4f} | {r.get("uno_holdout_se", 0):.4f} | '
                         f'{r["adv_auc"]:.3f} | {r["size"]} |')
        lines.append('')
    lines.append('## Combined LB (0.7*hep + 0.3*death) on hold-out\n')
    lines.append('| method | train LB | hold LB | ±SE | Uno hold LB |')
    lines.append('|---|---|---|---|---|')
    for c in combined:
        lines.append(f'| {c["method"]} | {c["train_lb"]:.4f} | '
                     f'{c["hold_lb"]:.4f} | {c["hold_lb_se"]:.4f} | '
                     f'{c["uno_hold_lb"]:.4f} |')
    md_path.write_text('\n'.join(lines))
    print(f'\nWrote {md_path}')

    # JSON summary too
    (out / 'comparison.json').write_text(json.dumps(
        dict(summary_rows=summary_rows, combined=combined,
             holdout_frac=args.holdout_frac,
             holdout_base_seed=args.holdout_base_seed,
             n_holdout_seeds=args.n_holdout_seeds),
        indent=2))


if __name__ == '__main__':
    main()
