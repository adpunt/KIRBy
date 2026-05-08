"""Six different diverse-fusion strategies, all honestly evaluated on
training OOFs, and uncertainty-aware routing where uncertainty values
are real signals (we measured r=0.47 between IsolationForest anomaly
and ensemble miss rate for hep, r=0.57 between ensemble disagreement
and miss rate for death).

Variants:

1. rank_mean        — average rank-transformed risks (current best honest)
2. rank_median      — median rank (robust to outlier branches)
3. rank_trimmed     — trim top/bottom 20%, average the middle
4. residual_greedy  — at each step, pick the model with the LOWEST
                      mean discordance on the patients the current
                      ensemble currently gets wrong (top-quartile by
                      discordance on the running ensemble). Rather
                      than 'pick the model that maximises ensemble C
                      overall', this targets the failure cases
                      directly.
5. uncertainty_route — split patients by anomaly score (IsolationForest
                       on raw clinical+snapshot features). For each
                       quantile bucket, pick the per-bucket-best model
                       (lowest discordance on bucket members). At
                       prediction time route by per-val-patient
                       anomaly score.
6. anomaly_blend     — soft mix of two specialists: a 'normal' model
                       (best on low-anomaly half) and an 'anomalous'
                       specialist (best on high-anomaly half), weighted
                       per-patient by sigmoid(anomaly_score - median).

All evaluated by HONEST OOF C and the best variant emitted as a
submission CSV.

Usage:
    python tests/experiments/annitia/refuse_variants.py
    python tests/experiments/annitia/refuse_variants.py --max-size 12

Outputs:
    tests/results/annitia/full/sub_master_<variant>.csv
    tests/results/annitia/full/refuse_variants_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia                      # noqa: E402
from experiments.annitia.views import build_views                          # noqa: E402
from experiments.annitia.survival import concordance                       # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _rank01(x):
    finite = np.isfinite(x); out = np.full_like(x, np.nan, dtype=float)
    if finite.sum():
        out[finite] = rankdata(x[finite]) / max(finite.sum(), 1)
    return out


def _c(y, valid, scores):
    m = valid & np.isfinite(scores) & np.isfinite(y['time'])
    if m.sum() < 5:
        return float('nan')
    return float(concordance(y[m], scores[m]))


def _per_patient_disc(time, event, risk):
    """Per-patient discordance fraction. NaN where no comparable pairs."""
    finite = np.isfinite(time) & np.isfinite(risk)
    t = np.where(finite, time, np.nan); e = event.astype(bool) & finite
    ti = t[:, None]; tj = t[None, :]; ei = e[:, None]
    finite_pair = np.isfinite(ti) & np.isfinite(tj) & (ti < tj)
    comparable = finite_pair & ei
    ri = risk[:, None]; rj = risk[None, :]
    discordant = comparable & (ri < rj)
    tied = comparable & (ri == rj)
    pi = comparable.sum(axis=1); pj = comparable.sum(axis=0)
    di = discordant.sum(axis=1) + 0.5 * tied.sum(axis=1)
    dj = discordant.sum(axis=0) + 0.5 * tied.sum(axis=0)
    cnt = pi + pj; disc = di + dj
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, disc / cnt, np.nan)


def _anomaly_scores(train, val):
    """Per-patient IsolationForest anomaly score on raw features.
    Higher = more anomalous. Computed jointly so val scores share scale."""
    from sklearn.ensemble import IsolationForest
    views_t = build_views(train, view_names=['snapshot', 'clinical_pro'])
    views_v = build_views(val, view_names=['snapshot', 'clinical_pro'])
    Xt = np.concatenate([views_t[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    Xv = np.concatenate([views_v[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
    iso = IsolationForest(n_estimators=300, contamination='auto',
                          random_state=42).fit(Xt)
    return -iso.score_samples(Xt), -iso.score_samples(Xv)


def _collect(pool_dirs, endpoint):
    out = []
    for d in pool_dirs:
        cv = d / 'cv_summary.json'
        if not cv.exists(): continue
        sm = json.loads(cv.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r.get('config_id', '')
            mc = r.get('mean_cindex', float('nan'))
            if not np.isfinite(mc): continue
            oof_p = d / f'oof_{endpoint}_{cid}.npy'
            val_p = d / f'val_{endpoint}_{cid}.npy'
            if not (oof_p.exists() and val_p.exists()): continue
            out.append({
                'label': f'{d.name}__{cid}', 'cv': float(mc),
                'oof': np.load(oof_p), 'val': np.load(val_p),
            })
    out.sort(key=lambda r: -r['cv'])
    return out


# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------
def variant_mean(recs, y, valid, max_size, anom_train, anom_val):
    """Greedy add-the-model-that-most-improves-rank-mean-C."""
    if not recs: return None
    chosen = [recs[0]]; rks = [_rank01(recs[0]['oof'])]; vrs = [_rank01(recs[0]['val'])]
    best_c = _c(y, valid, np.mean(np.stack(rks, axis=1), axis=1))
    pool = list(recs[1:])
    while len(chosen) < max_size and pool:
        best_score, best_i = -np.inf, None
        for i, cand in enumerate(pool):
            tmp_score = np.mean(np.stack(rks + [_rank01(cand['oof'])], axis=1), axis=1)
            c = _c(y, valid, tmp_score)
            if c > best_score: best_score, best_i = c, i
        if best_i is None or best_score - best_c < 1e-4: break
        chosen.append(pool[best_i])
        rks.append(_rank01(pool[best_i]['oof']))
        vrs.append(_rank01(pool[best_i]['val']))
        best_c = best_score
        pool.pop(best_i)
    return {
        'name': 'rank_mean', 'oof_c': best_c, 'size': len(chosen),
        'val': np.mean(np.stack(vrs, axis=1), axis=1),
        'oof': np.mean(np.stack(rks, axis=1), axis=1),
        'members': [c['label'] for c in chosen],
    }


def variant_median(recs, y, valid, max_size, *_):
    """Greedy add-to-rank-median."""
    if not recs: return None
    chosen = [recs[0]]; rks = [_rank01(recs[0]['oof'])]; vrs = [_rank01(recs[0]['val'])]
    best_c = _c(y, valid, np.median(np.stack(rks, axis=1), axis=1))
    pool = list(recs[1:])
    while len(chosen) < max_size and pool:
        best_score, best_i = -np.inf, None
        for i, cand in enumerate(pool):
            tmp = np.median(np.stack(rks + [_rank01(cand['oof'])], axis=1), axis=1)
            c = _c(y, valid, tmp)
            if c > best_score: best_score, best_i = c, i
        if best_i is None or best_score - best_c < 1e-4: break
        chosen.append(pool[best_i])
        rks.append(_rank01(pool[best_i]['oof']))
        vrs.append(_rank01(pool[best_i]['val']))
        best_c = best_score
        pool.pop(best_i)
    return {
        'name': 'rank_median', 'oof_c': best_c, 'size': len(chosen),
        'val': np.median(np.stack(vrs, axis=1), axis=1),
        'oof': np.median(np.stack(rks, axis=1), axis=1),
        'members': [c['label'] for c in chosen],
    }


def variant_trimmed(recs, y, valid, max_size, *_, trim=0.2):
    """Trim top/bottom `trim` fraction, average the middle."""
    if not recs: return None
    def _trim(arr):
        n = arr.shape[1]
        if n < 5: return arr.mean(axis=1)
        k = int(np.floor(n * trim / 2))
        s = np.sort(arr, axis=1)
        return s[:, k:n - k].mean(axis=1) if k > 0 else s.mean(axis=1)
    chosen = [recs[0]]; rks = [_rank01(recs[0]['oof'])]; vrs = [_rank01(recs[0]['val'])]
    best_c = _c(y, valid, _trim(np.stack(rks, axis=1)))
    pool = list(recs[1:])
    while len(chosen) < max_size and pool:
        best_score, best_i = -np.inf, None
        for i, cand in enumerate(pool):
            tmp = _trim(np.stack(rks + [_rank01(cand['oof'])], axis=1))
            c = _c(y, valid, tmp)
            if c > best_score: best_score, best_i = c, i
        if best_i is None or best_score - best_c < 1e-4: break
        chosen.append(pool[best_i])
        rks.append(_rank01(pool[best_i]['oof']))
        vrs.append(_rank01(pool[best_i]['val']))
        best_c = best_score
        pool.pop(best_i)
    return {
        'name': 'rank_trimmed', 'oof_c': best_c, 'size': len(chosen),
        'val': _trim(np.stack(vrs, axis=1)),
        'oof': _trim(np.stack(rks, axis=1)),
        'members': [c['label'] for c in chosen],
    }


def variant_residual_greedy(recs, y, valid, max_size, *_, hard_q=0.75):
    """Pick model that's best on the patients current ensemble most fails on.

    At each step:
      1. Compute current ensemble per-patient discordance.
      2. Identify the top-quartile-hardest patients (>= hard_q quantile).
      3. Pick the candidate with lowest mean discordance on THOSE patients.
      4. Add to ensemble. Stop when overall ensemble C stops improving.
    """
    if not recs: return None
    chosen = [recs[0]]; rks = [_rank01(recs[0]['oof'])]; vrs = [_rank01(recs[0]['val'])]
    best_c = _c(y, valid, np.mean(np.stack(rks, axis=1), axis=1))
    pool = list(recs[1:])
    while len(chosen) < max_size and pool:
        ens = np.mean(np.stack(rks, axis=1), axis=1)
        ens_disc = _per_patient_disc(y['time'], y['event'], ens)
        keep = valid & np.isfinite(ens_disc)
        if keep.sum() < 50: break
        thresh = np.nanquantile(ens_disc[keep], hard_q)
        hard_mask = keep & (ens_disc >= thresh)
        if hard_mask.sum() < 20: break
        # Pick candidate with lowest hard-set discordance
        best_hard, best_i = np.inf, None
        for i, cand in enumerate(pool):
            d_c = _per_patient_disc(y['time'], y['event'], cand['oof'])
            sub = d_c[hard_mask]; sub = sub[np.isfinite(sub)]
            if len(sub) < 5: continue
            md = np.mean(sub)
            if md < best_hard:
                best_hard, best_i = md, i
        if best_i is None: break
        # Add and re-check overall C
        rks_try = rks + [_rank01(pool[best_i]['oof'])]
        c_try = _c(y, valid, np.mean(np.stack(rks_try, axis=1), axis=1))
        if c_try - best_c < 1e-4: break
        chosen.append(pool[best_i])
        rks.append(rks_try[-1])
        vrs.append(_rank01(pool[best_i]['val']))
        best_c = c_try
        pool.pop(best_i)
    return {
        'name': 'residual_greedy', 'oof_c': best_c, 'size': len(chosen),
        'val': np.mean(np.stack(vrs, axis=1), axis=1),
        'oof': np.mean(np.stack(rks, axis=1), axis=1),
        'members': [c['label'] for c in chosen],
    }


def variant_uncertainty_route(recs, y, valid, max_size, anom_train, anom_val,
                              n_buckets=4):
    """Split patients into anomaly quantile buckets; pick best model per
    bucket; route val patients by their own anomaly score."""
    if not recs: return None
    a = anom_train
    keep = valid & np.isfinite(a)
    if keep.sum() < 100: return None
    edges = np.nanquantile(a[keep], np.linspace(0, 1, n_buckets + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    best_per_bucket = []
    for b in range(n_buckets):
        bucket_mask = keep & (a >= edges[b]) & (a < edges[b + 1])
        if bucket_mask.sum() < 20:
            best_per_bucket.append(recs[0]); continue
        best_c, best_r = -np.inf, recs[0]
        for r in recs[:max_size * 4]:  # only consider top candidates by CV
            d = _per_patient_disc(y['time'], y['event'], r['oof'])
            sub_d = d[bucket_mask]; sub_d = sub_d[np.isfinite(sub_d)]
            if len(sub_d) < 5: continue
            mean_disc_in_bucket = np.mean(sub_d)
            score = 1.0 - mean_disc_in_bucket
            if score > best_c:
                best_c, best_r = score, r
        best_per_bucket.append(best_r)
    # Build train OOF: for each train row, use the bucket model's OOF[row]
    oof_route = np.full(len(y), np.nan)
    for b in range(n_buckets):
        bucket_mask = (a >= edges[b]) & (a < edges[b + 1])
        oof_route[bucket_mask] = _rank01(best_per_bucket[b]['oof'])[bucket_mask]
    # For val: assign by val anomaly score
    val_route = np.full(len(anom_val), np.nan)
    for b in range(n_buckets):
        m = (anom_val >= edges[b]) & (anom_val < edges[b + 1])
        val_route[m] = _rank01(best_per_bucket[b]['val'])[m]
    c = _c(y, valid, oof_route)
    return {
        'name': 'uncertainty_route', 'oof_c': c, 'size': n_buckets,
        'val': val_route, 'oof': oof_route,
        'members': [r['label'] for r in best_per_bucket],
        'edges': edges.tolist(),
    }


def variant_anomaly_blend(recs, y, valid, max_size, anom_train, anom_val):
    """Soft 2-specialist blend: low-anomaly specialist + high-anomaly
    specialist, mixed per-patient by sigmoid(anomaly - median)."""
    if not recs: return None
    a = anom_train
    keep = valid & np.isfinite(a)
    med = np.nanmedian(a[keep])
    low_mask = keep & (a < med)
    high_mask = keep & (a >= med)
    # Pick best on low half
    best_low, best_high = recs[0], recs[0]
    best_low_c, best_high_c = -np.inf, -np.inf
    for r in recs[:max_size * 4]:
        d = _per_patient_disc(y['time'], y['event'], r['oof'])
        for half_mask, label in ((low_mask, 'low'), (high_mask, 'high')):
            sub = d[half_mask]; sub = sub[np.isfinite(sub)]
            if len(sub) < 10: continue
            score = 1.0 - np.mean(sub)
            if label == 'low' and score > best_low_c:
                best_low_c, best_low = score, r
            if label == 'high' and score > best_high_c:
                best_high_c, best_high = score, r
    # Mix by sigmoid(a - med)
    spread = max(np.nanstd(a[keep]), 1e-3)
    w_train = 1.0 / (1.0 + np.exp(-(a - med) / spread))
    w_val = 1.0 / (1.0 + np.exp(-(anom_val - med) / spread))
    low_oof = _rank01(best_low['oof'])
    high_oof = _rank01(best_high['oof'])
    low_val = _rank01(best_low['val'])
    high_val = _rank01(best_high['val'])
    blend_oof = (1.0 - w_train) * low_oof + w_train * high_oof
    blend_val = (1.0 - w_val) * low_val + w_val * high_val
    return {
        'name': 'anomaly_blend', 'oof_c': _c(y, valid, blend_oof), 'size': 2,
        'val': blend_val, 'oof': blend_oof,
        'members': [best_low['label'], best_high['label']],
    }


def variant_disagreement_route(recs, y, valid, max_size, anom_train, anom_val,
                               n_buckets=4):
    """Route by per-patient ENSEMBLE DISAGREEMENT (rank-std across all
    pool models) rather than by anomaly score. The disagreement signal
    correlates strongly with miss rate (r=0.57 death, r=0.20 hep)."""
    if not recs: return None
    R_train = np.stack([_rank01(r['oof']) for r in recs], axis=1)
    R_val = np.stack([_rank01(r['val']) for r in recs], axis=1)
    n_finite = np.sum(np.isfinite(R_train), axis=1)
    with np.errstate(invalid='ignore'):
        unc_train = np.nanstd(R_train, axis=1)
        unc_val = np.nanstd(R_val, axis=1)
    unc_train[n_finite < 5] = np.nan
    keep = valid & np.isfinite(unc_train)
    if keep.sum() < 100: return None
    edges = np.nanquantile(unc_train[keep], np.linspace(0, 1, n_buckets + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    best_per_bucket = []
    for b in range(n_buckets):
        bm = keep & (unc_train >= edges[b]) & (unc_train < edges[b + 1])
        if bm.sum() < 20:
            best_per_bucket.append(recs[0]); continue
        best_score, best_r = -np.inf, recs[0]
        for r in recs[:max_size * 4]:
            d = _per_patient_disc(y['time'], y['event'], r['oof'])
            sub = d[bm]; sub = sub[np.isfinite(sub)]
            if len(sub) < 5: continue
            score = 1.0 - np.mean(sub)
            if score > best_score: best_score, best_r = score, r
        best_per_bucket.append(best_r)
    oof_route = np.full(len(y), np.nan)
    for b in range(n_buckets):
        bm = (unc_train >= edges[b]) & (unc_train < edges[b + 1])
        oof_route[bm] = _rank01(best_per_bucket[b]['oof'])[bm]
    val_route = np.full(len(unc_val), np.nan)
    for b in range(n_buckets):
        m = (unc_val >= edges[b]) & (unc_val < edges[b + 1])
        val_route[m] = _rank01(best_per_bucket[b]['val'])[m]
    return {
        'name': 'disagreement_route', 'oof_c': _c(y, valid, oof_route),
        'size': n_buckets, 'val': val_route, 'oof': oof_route,
        'members': [r['label'] for r in best_per_bucket],
    }


def variant_cluster_route(recs, y, valid, max_size, anom_train, anom_val,
                          n_clusters=5):
    """KMeans cluster patients on raw clinical+snapshot features; pick
    best model per cluster by lowest mean discordance within the cluster.
    At inference, route val patients by cluster assignment."""
    if not recs: return None
    from sklearn.cluster import KMeans
    train = load_annitia()
    views_t = build_views(train, view_names=['snapshot', 'clinical_pro'])
    Xt = np.concatenate([views_t[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    val_path = '/Volumes/seagate/kirby/annitia/val_test.csv'
    val = load_annitia(data_dir=str(Path(val_path).parent),
                       filename=Path(val_path).name, has_outcomes=False)
    views_v = build_views(val, view_names=['snapshot', 'clinical_pro'])
    Xv = np.concatenate([views_v[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(Xt)
    cl_train = km.labels_
    cl_val = km.predict(Xv)
    best_per_cluster = []
    for c in range(n_clusters):
        cm = valid & (cl_train == c)
        if cm.sum() < 20:
            best_per_cluster.append(recs[0]); continue
        best_score, best_r = -np.inf, recs[0]
        for r in recs[:max_size * 4]:
            d = _per_patient_disc(y['time'], y['event'], r['oof'])
            sub = d[cm]; sub = sub[np.isfinite(sub)]
            if len(sub) < 5: continue
            s = 1.0 - np.mean(sub)
            if s > best_score: best_score, best_r = s, r
        best_per_cluster.append(best_r)
    oof_route = np.full(len(y), np.nan)
    val_route = np.full(len(cl_val), np.nan)
    for c in range(n_clusters):
        oof_route[cl_train == c] = _rank01(best_per_cluster[c]['oof'])[cl_train == c]
        val_route[cl_val == c] = _rank01(best_per_cluster[c]['val'])[cl_val == c]
    return {
        'name': 'cluster_route', 'oof_c': _c(y, valid, oof_route),
        'size': n_clusters, 'val': val_route, 'oof': oof_route,
        'members': [r['label'] for r in best_per_cluster],
    }


def variant_specialist_override(recs, y, valid, max_size, anom_train, anom_val,
                                top_pct=0.20):
    """Generalist (rank_mean diversity ensemble) for everyone, but override
    the top-X% most anomalous patients with the per-bucket specialist that
    had the lowest discordance on similarly-anomalous training patients."""
    if not recs: return None
    # 1. Build the diversity rank_mean ensemble
    base = variant_mean(recs, y, valid, max_size, anom_train, anom_val)
    if base is None: return None
    base_oof = base['oof']; base_val = base['val']
    # 2. For training patients in the top-X% anomaly bracket, pick the
    #    single model with lowest discordance on that bracket.
    a = anom_train; keep = valid & np.isfinite(a)
    if keep.sum() < 50: return base
    thresh = np.nanquantile(a[keep], 1.0 - top_pct)
    high_mask = keep & (a >= thresh)
    if high_mask.sum() < 20: return base
    best_score, best_r = -np.inf, None
    for r in recs[:max_size * 4]:
        d = _per_patient_disc(y['time'], y['event'], r['oof'])
        sub = d[high_mask]; sub = sub[np.isfinite(sub)]
        if len(sub) < 5: continue
        s = 1.0 - np.mean(sub)
        if s > best_score: best_score, best_r = s, r
    if best_r is None: return base
    spec_oof = _rank01(best_r['oof'])
    spec_val = _rank01(best_r['val'])
    # 3. Override top-X% anomalous on training side
    out_oof = base_oof.copy()
    out_oof[high_mask] = spec_oof[high_mask]
    # 4. Override on val: top-X% by VAL anomaly score
    val_thresh = np.nanquantile(anom_val, 1.0 - top_pct)
    val_high = anom_val >= val_thresh
    out_val = base_val.copy()
    out_val[val_high] = spec_val[val_high]
    return {
        'name': 'specialist_override', 'oof_c': _c(y, valid, out_oof),
        'size': base['size'] + 1, 'val': out_val, 'oof': out_oof,
        'members': base['members'] + [f'SPECIALIST_top{int(top_pct*100)}pct: ' + best_r['label']],
    }


def variant_soft_blend_uncertainty(recs, y, valid, max_size, anom_train, anom_val):
    """Soft per-patient mix between the diversity ensemble and a
    'most-robust-on-uncertain-rows' specialist, weighted by ensemble
    DISAGREEMENT (rank-std) — patients where the ensemble disagrees
    the most get pulled towards the specialist."""
    if not recs: return None
    base = variant_mean(recs, y, valid, max_size, anom_train, anom_val)
    if base is None: return None
    R_train = np.stack([_rank01(r['oof']) for r in recs], axis=1)
    R_val = np.stack([_rank01(r['val']) for r in recs], axis=1)
    with np.errstate(invalid='ignore'):
        unc_train = np.nanstd(R_train, axis=1)
        unc_val = np.nanstd(R_val, axis=1)
    keep = valid & np.isfinite(unc_train)
    if keep.sum() < 50: return base
    med = np.nanmedian(unc_train[keep])
    high_mask = keep & (unc_train > med)
    if high_mask.sum() < 20: return base
    best_score, best_r = -np.inf, None
    for r in recs[:max_size * 4]:
        d = _per_patient_disc(y['time'], y['event'], r['oof'])
        sub = d[high_mask]; sub = sub[np.isfinite(sub)]
        if len(sub) < 5: continue
        s = 1.0 - np.mean(sub)
        if s > best_score: best_score, best_r = s, r
    if best_r is None: return base
    # Soft mix: w = sigmoid((unc - med) / spread)
    spread = max(np.nanstd(unc_train[keep]), 1e-3)
    w_t = 1.0 / (1.0 + np.exp(-(unc_train - med) / spread))
    w_v = 1.0 / (1.0 + np.exp(-(unc_val - med) / spread))
    spec_oof = _rank01(best_r['oof']); spec_val = _rank01(best_r['val'])
    out_oof = (1.0 - w_t) * base['oof'] + w_t * spec_oof
    out_val = (1.0 - w_v) * base['val'] + w_v * spec_val
    return {
        'name': 'soft_blend_disagreement',
        'oof_c': _c(y, valid, out_oof),
        'size': base['size'] + 1, 'val': out_val, 'oof': out_oof,
        'members': base['members'] + [f'SPECIALIST_high_disagreement: ' + best_r['label']],
    }


VARIANTS = [variant_mean, variant_median, variant_trimmed,
            variant_residual_greedy, variant_uncertainty_route,
            variant_anomaly_blend, variant_disagreement_route,
            variant_cluster_route, variant_specialist_override,
            variant_soft_blend_uncertainty]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--max-size', type=int, default=10)
    ap.add_argument('--output-dir', default='tests/results/annitia/full')
    args = ap.parse_args()

    res_root = _REPO / 'tests' / 'results' / 'annitia'
    pool_dirs = [res_root / p for p in
                 ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam', 'research', 'research_v2')
                 if (res_root / p).exists()]
    print(f'Pooling from: {[d.name for d in pool_dirs]}')

    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)
    anom_train, anom_val = _anomaly_scores(train, val)
    print(f'  anomaly: train range [{anom_train.min():.3f},{anom_train.max():.3f}] '
          f'med={np.median(anom_train):.3f}; val range '
          f'[{anom_val.min():.3f},{anom_val.max():.3f}]')

    out = Path(args.output_dir)
    summary = {}
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        recs = _collect(pool_dirs, endpoint)
        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        print(f'\n=== {endpoint} (pool={len(recs)}) ===')
        endpoint_results = []
        for variant_fn in VARIANTS:
            try:
                r = variant_fn(recs, y, valid, args.max_size, anom_train, anom_val)
            except Exception as e:
                print(f'  {variant_fn.__name__:30s} FAILED: {type(e).__name__}: {e}')
                continue
            if r is None:
                continue
            endpoint_results.append(r)
            print(f'  {r["name"]:25s} OOF C={r["oof_c"]:.4f}  size={r["size"]:2d}'
                  f'  members={[m[:35] for m in r["members"][:5]]}')
        # Best variant: write submission per-endpoint into its own CSV
        endpoint_results.sort(key=lambda r: -r['oof_c'])
        best = endpoint_results[0] if endpoint_results else None
        if best is None:
            continue
        summary[endpoint] = {
            'best_variant': best['name'],
            'best_oof_c': best['oof_c'],
            'best_size': best['size'],
            'best_members': best['members'],
            'all_variants': [{'name': r['name'], 'oof_c': r['oof_c'],
                              'size': r['size']} for r in endpoint_results],
        }
        # Save each variant's val predictions + OOF arrays for downstream stats.
        for r in endpoint_results:
            csv_p = out / f'sub_master_{r["name"]}.csv'
            df = pd.DataFrame({'trustii_id': val['trustii_id']})
            df[col] = r['val']
            if csv_p.exists():
                prev = pd.read_csv(csv_p)
                if col in prev.columns:
                    prev[col] = r['val']
                else:
                    prev = prev.merge(df, on='trustii_id')
                prev.to_csv(csv_p, index=False)
            else:
                df.to_csv(csv_p, index=False)
            # Save the per-variant OOF for dig_deeper / bootstrap-CI stats.
            oof_arr = r.get('oof')
            if oof_arr is not None and isinstance(oof_arr, np.ndarray):
                # The OOF array from refuse_variants is the rank-mean / median /
                # bucket-routed prediction restricted to valid+nonnan rows. Pad
                # to full-train length with NaN so downstream alignment is easy.
                full = np.full(len(y_full := train[endpoint]['y_struct']), np.nan)
                valid_mask = train[endpoint]['valid_mask']
                # If oof_arr is the size of valid_mask, copy directly; otherwise
                # try the size of valid only.
                if oof_arr.size == valid_mask.size:
                    full = oof_arr
                elif oof_arr.size == int(valid_mask.sum()):
                    full[valid_mask] = oof_arr
                np.save(out / f'oof_{endpoint}_{r["name"]}.npy', full)

    # Combined "best per endpoint" submission
    combined = pd.DataFrame({'trustii_id': val['trustii_id']})
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        if endpoint in summary:
            best_name = summary[endpoint]['best_variant']
            sub = pd.read_csv(out / f'sub_master_{best_name}.csv')
            combined[col] = sub[col]
    combined_path = out / 'sub_master_best_per_endpoint.csv'
    combined.to_csv(combined_path, index=False)
    summary_path = out / 'refuse_variants_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'\nDONE')
    print(f'  combined best: {combined_path}')
    print(f'  per-variant: {out}/sub_master_<name>.csv')
    for ep, s in summary.items():
        print(f'  {ep}: best={s["best_variant"]} OOF-C={s["best_oof_c"]:.4f}')


if __name__ == '__main__':
    main()
