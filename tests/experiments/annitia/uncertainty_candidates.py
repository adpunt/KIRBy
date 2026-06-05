"""Uncertainty-aware ensemble candidates (sub_master_unc_*).

Five distinct uncertainty signals, each producing a separate fusion CSV:

  --variant mahal_blend    : Mahalanobis distance to training centroid drives a
                             continuous gate between anchor (in-distribution) and
                             rank_median (defensive) per patient.
  --variant bootstrap_var  : Per-base-model bootstrap std on OOF -> per-model
                             confidence weights; aggregate base preds inverse-var.
  --variant disagree_soft  : Per-patient ensemble rank-std drives a soft mix of
                             anchor and the model that performs best on
                             high-disagreement training rows (no buckets).
  --variant conformal      : OOF residual-width conformal calibrator per base;
                             use 1/width as fusion weight per base (NOT per
                             patient), pooled across the cohort.
  --variant specialist     : Hybrid override -- top-20% high-uncertainty val
                             patients go to most-trusted model (tabpfn-landmark
                             for death, synaug-none-rsf for hep); rest = anchor.

Reads:
  tests/results/annitia/branches/{oof,val}_<endpoint>_<config>.npy
  tests/results/annitia/full/sub_master_meta_avg_aware_weighted.csv (anchor)
  tests/results/annitia/analysis/ensemble_uncertainty_<endpoint>.npy

Writes:
  tests/results/annitia/full/sub_master_unc_<variant>.csv

Run all:
  for v in mahal_blend bootstrap_var disagree_soft conformal specialist; do
      python uncertainty_candidates.py --variant $v
  done
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia                 # noqa: E402
from experiments.annitia.views import build_views                     # noqa: E402
from experiments.annitia.survival import concordance                  # noqa: E402

RES = REPO / 'tests/results/annitia'
BRANCHES = RES / 'branches'
ANALYSIS = RES / 'analysis'
FULL = RES / 'full'
ANCHOR = FULL / 'sub_master_meta_avg_aware_weighted.csv'
VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'

# Most-trusted base model per endpoint (from oof_diagnostics.md)
SPECIALIST = {
    'hepatic': 'synaug-none-rsf-500-n0__snapshot+clinical_pro+fpca+gp_smooth',
    'death':   'tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth',
}
ENDPOINTS = (('hepatic', 'risk_hepatic_event'), ('death', 'risk_death'))


# ---------------------------------------------------------------------------
def _rank01(x):
    finite = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if finite.sum():
        out[finite] = rankdata(x[finite]) / max(finite.sum(), 1)
    return out


def _c(y, valid, scores):
    m = valid & np.isfinite(scores) & np.isfinite(y['time'])
    if m.sum() < 5:
        return float('nan')
    return float(concordance(y[m], scores[m]))


def _load_pool(endpoint):
    """Returns list of {label, oof, val, cv}, sorted by mean_cindex desc."""
    cv_p = BRANCHES / 'cv_summary.json'
    sm = json.loads(cv_p.read_text())
    out = []
    for r in sm.get('per_condition', {}).get(endpoint, []):
        cid = r['config_id']
        mc = r.get('mean_cindex', float('nan'))
        if not np.isfinite(mc):
            continue
        oof_p = BRANCHES / f'oof_{endpoint}_{cid}.npy'
        val_p = BRANCHES / f'val_{endpoint}_{cid}.npy'
        if not (oof_p.exists() and val_p.exists()):
            continue
        out.append({
            'label': cid, 'cv': float(mc),
            'oof': np.load(oof_p), 'val': np.load(val_p),
        })
    out.sort(key=lambda r: -r['cv'])
    return out


def _val_int_ids(val):
    """Map GUID patient_ids -> integer trustii_id, sorted by trustii_id 1..423."""
    val_meta = pd.read_csv(VAL_CSV)[['trustii_id', 'patient_id_anon']]
    guid_to_int = dict(zip(val_meta['patient_id_anon'].astype(str),
                           val_meta['trustii_id'].astype(int)))
    int_ids = np.array([guid_to_int[str(g)] for g in val['patient_ids']])
    assert set(int_ids) == set(range(1, len(int_ids) + 1)), 'incomplete trustii_id map'
    return int_ids


def _features(d):
    views = build_views(d, view_names=['snapshot', 'clinical_pro'])
    X = np.concatenate([views[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _anchor():
    a = pd.read_csv(ANCHOR)
    return a.sort_values('trustii_id').reset_index(drop=True)


def _write(int_ids, hep, death, name):
    df = pd.DataFrame({'trustii_id': int_ids,
                       'risk_hepatic_event': hep,
                       'risk_death': death}).sort_values('trustii_id').reset_index(drop=True)
    dest = FULL / f'sub_master_unc_{name}.csv'
    df.to_csv(dest, index=False)
    return dest


# ---------------------------------------------------------------------------
# (a) Mahalanobis-distance-aware blend
# ---------------------------------------------------------------------------
def variant_mahal_blend(train, val, anchor):
    """Continuous gate by Mahalanobis distance.
    Low distance (in-dist) -> anchor.  High distance (extrapolation) ->
    rank_median across full pool (defensive).  Sigmoid gate centered at
    training median distance.

    OOF training-side: same gate using leave-one-out Mahalanobis on training,
    anchor approximated by its own variants stored values.  We approximate
    OOF C by computing: gate is patient-specific; for OOF we use the median
    of base-model OOFs as the 'defensive' arm.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Xt = _features(train); Xv = _features(val)
    sc = StandardScaler().fit(Xt)
    Xt_s = sc.transform(Xt); Xv_s = sc.transform(Xv)
    n_pc = min(20, Xt_s.shape[1], Xt_s.shape[0] - 1)
    pca = PCA(n_components=n_pc, random_state=42).fit(Xt_s)
    Tt = pca.transform(Xt_s); Tv = pca.transform(Xv_s)
    cov = np.cov(Tt.T) + 1e-3 * np.eye(n_pc)
    inv = np.linalg.pinv(cov)
    mu = Tt.mean(axis=0)
    def _md(M):
        d = M - mu
        return np.sqrt(np.einsum('ij,jk,ik->i', d, inv, d))
    md_t = _md(Tt); md_v = _md(Tv)
    # gate = sigmoid((md - median(md_t)) / std(md_t))
    med = np.median(md_t); spread = max(np.std(md_t), 1e-3)
    g_t = 1.0 / (1.0 + np.exp(-(md_t - med) / spread))
    g_v = 1.0 / (1.0 + np.exp(-(md_v - med) / spread))

    out = {}
    for ep, col in ENDPOINTS:
        recs = _load_pool(ep)
        y = train[ep]['y_struct']; valid = train[ep]['valid_mask']
        # 'defensive' = rank_median across pool
        oof_stack = np.stack([_rank01(r['oof']) for r in recs], axis=1)
        val_stack = np.stack([_rank01(r['val']) for r in recs], axis=1)
        defensive_oof = np.nanmedian(oof_stack, axis=1)
        defensive_val = np.nanmedian(val_stack, axis=1)
        # 'anchor' for OOF approximation: use rank_mean (best high-confidence)
        anchor_oof = np.nanmean(oof_stack, axis=1)
        # Final OOF blend
        oof = (1 - g_t) * _rank01(anchor_oof) + g_t * _rank01(defensive_oof)
        # Val blend uses real anchor file
        anc_v = _rank01(anchor[col].values)
        def_v = _rank01(defensive_val)
        val_pred = (1 - g_v) * anc_v + g_v * def_v
        out[ep] = {'oof': oof, 'val': val_pred,
                   'oof_c': _c(y, valid, oof)}
    return out, {'gate_train_summary': {'med': float(med), 'spread': float(spread),
                                        'mean_g_v': float(g_v.mean())}}


# ---------------------------------------------------------------------------
# (b) Bootstrap-variance-weighted aggregation
# ---------------------------------------------------------------------------
def variant_bootstrap_var(train, val, anchor, B=100, seed=42):
    """Per-base-model OOF bootstrap -> per-base std (scalar). 1/(std+eps)
    becomes that model's WEIGHT in the final fusion. Lower-variance models
    contribute more. Distinct from CV-weighted because bootstrap captures
    OOF-prediction stability, not CV-fold stability.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for ep, col in ENDPOINTS:
        recs = _load_pool(ep)
        y = train[ep]['y_struct']; valid = train[ep]['valid_mask']
        n = len(train['patient_ids'])
        # Bootstrap each base model's OOF: resample patients, recompute
        # rank-correlation with self -> variance proxy. Cheap+consistent:
        # for each model, std of OOF C across B bootstraps.
        weights = []
        for r in recs:
            oof = r['oof']
            cs = []
            valid_idx = np.where(valid & np.isfinite(oof) & np.isfinite(y['time']))[0]
            if len(valid_idx) < 50:
                weights.append(1e-6); continue
            for _ in range(B):
                bs = rng.choice(valid_idx, size=len(valid_idx), replace=True)
                try:
                    c = float(concordance(y[bs], oof[bs]))
                except Exception:
                    c = np.nan
                if np.isfinite(c):
                    cs.append(c)
            sd = float(np.std(cs)) if cs else 1.0
            weights.append(1.0 / (sd + 1e-3))
        weights = np.array(weights)
        weights = weights / weights.sum()
        oof_stack = np.stack([_rank01(r['oof']) for r in recs], axis=1)
        val_stack = np.stack([_rank01(r['val']) for r in recs], axis=1)
        # Weighted average
        oof = np.nansum(oof_stack * weights[None, :], axis=1)
        val_pred = np.nansum(val_stack * weights[None, :], axis=1)
        out[ep] = {'oof': oof, 'val': val_pred,
                   'oof_c': _c(y, valid, oof),
                   'weights': weights.tolist()}
    return out, {'B': B,
                 'top_weights_hep': out.get('hepatic', {}).get('weights', [])[:5]}


# ---------------------------------------------------------------------------
# (c) Ensemble-disagreement soft-route
# ---------------------------------------------------------------------------
def variant_disagree_soft(train, val, anchor):
    """Per-patient ensemble rank-std on training -> identify the single
    base model that performs best on the top-quartile-disagreement training
    patients. Then for val patients with high disagreement, smoothly blend
    that specialist with anchor.

    NEW vs disagreement_router: continuous sigmoid gate (no buckets), and
    identifies the SPECIALIST by performance on hard cases (not best variant
    among 10 fixed routes).
    """
    out = {}
    for ep, col in ENDPOINTS:
        recs = _load_pool(ep)
        y = train[ep]['y_struct']; valid = train[ep]['valid_mask']
        oof_stack = np.stack([_rank01(r['oof']) for r in recs], axis=1)
        val_stack = np.stack([_rank01(r['val']) for r in recs], axis=1)
        # disagreement
        dis_t = np.nanstd(oof_stack, axis=1)
        dis_v = np.nanstd(val_stack, axis=1)
        # On training, the upper quartile of disagreement = "hard" patients
        keep = valid & np.isfinite(dis_t)
        thresh = np.nanquantile(dis_t[keep], 0.75)
        hard = keep & (dis_t >= thresh)
        # Pick specialist: base model with best C on hard subset
        best_c, best_i = -np.inf, 0
        for i, r in enumerate(recs):
            c = _c(y, hard, r['oof'])
            if np.isfinite(c) and c > best_c:
                best_c, best_i = c, i
        spec_oof = _rank01(recs[best_i]['oof'])
        spec_val = _rank01(recs[best_i]['val'])
        # rank_mean as anchor for OOF
        anchor_oof = np.nanmean(oof_stack, axis=1)
        # gate: sigmoid centered at median disagreement
        med = np.nanmedian(dis_t[keep]); spread = max(np.nanstd(dis_t[keep]), 1e-3)
        g_t = 1.0 / (1.0 + np.exp(-(dis_t - med) / spread))
        g_v = 1.0 / (1.0 + np.exp(-(dis_v - med) / spread))
        oof = (1 - g_t) * _rank01(anchor_oof) + g_t * spec_oof
        anc_v = _rank01(anchor[col].values)
        val_pred = (1 - g_v) * anc_v + g_v * spec_val
        out[ep] = {'oof': oof, 'val': val_pred,
                   'oof_c': _c(y, valid, oof),
                   'specialist': recs[best_i]['label'], 'specialist_hard_c': best_c}
    return out, {'specialists': {ep: out[ep]['specialist'] for ep, _ in ENDPOINTS}}


# ---------------------------------------------------------------------------
# (d) Conformal-prediction wrapper
# ---------------------------------------------------------------------------
def variant_conformal(train, val, anchor, alpha=0.2):
    """For each base model, compute a conformal interval width on OOF rank
    residuals: width_i = quantile_(1-alpha)(|rank(oof_i) - rank(target_proxy)|).
    Target proxy = rank of (event-time inverse / censored downweight).

    Use 1/width as fusion weight ACROSS bases (not per patient). This is a
    distinct signal from (b): conformal width is a calibration-driven
    notion of reliability, not bootstrap variance.
    """
    out = {}
    for ep, col in ENDPOINTS:
        recs = _load_pool(ep)
        y = train[ep]['y_struct']; valid = train[ep]['valid_mask']
        # Target proxy: rank by (event=1 -> -time, censored -> +time)
        # Higher risk if shorter event time. Build a rank target.
        time = y['time']; event = y['event'].astype(bool)
        proxy = np.where(event, -time, time + 1e6)  # events first by time
        proxy_r = _rank01(proxy)
        weights = []
        for r in recs:
            oof = r['oof']
            m = valid & np.isfinite(oof) & np.isfinite(time)
            if m.sum() < 50:
                weights.append(1e-6); continue
            res = np.abs(_rank01(oof)[m] - proxy_r[m])
            w = float(np.quantile(res, 1 - alpha))
            weights.append(1.0 / (w + 1e-3))
        weights = np.array(weights); weights = weights / weights.sum()
        oof_stack = np.stack([_rank01(r['oof']) for r in recs], axis=1)
        val_stack = np.stack([_rank01(r['val']) for r in recs], axis=1)
        oof = np.nansum(oof_stack * weights[None, :], axis=1)
        val_pred = np.nansum(val_stack * weights[None, :], axis=1)
        out[ep] = {'oof': oof, 'val': val_pred,
                   'oof_c': _c(y, valid, oof),
                   'weights': weights.tolist()}
    return out, {'alpha': alpha}


# ---------------------------------------------------------------------------
# (e) Uncertainty-targeted specialist
# ---------------------------------------------------------------------------
def variant_specialist(train, val, anchor, top_q=0.20):
    """For top-20% high-uncertainty (ensemble rank-std) val patients, OVERRIDE
    anchor with the most-trusted single base model from oof_diagnostics
    (tabpfn-landmark for death, synaug-none-rsf for hep). For the rest, use
    anchor unchanged.

    Hard switch (not soft) — distinct from (a)/(c) because it is targeted at
    a small high-uncertainty tail with a model selected by EXTERNAL trust
    flags, not by performance on hard training rows.
    """
    out = {}
    for ep, col in ENDPOINTS:
        recs = _load_pool(ep)
        y = train[ep]['y_struct']; valid = train[ep]['valid_mask']
        # locate the specialist in the pool
        spec_label = SPECIALIST[ep]
        spec = next((r for r in recs if r['label'] == spec_label), None)
        if spec is None:
            # fallback: highest-CV model
            spec = recs[0]
        oof_stack = np.stack([_rank01(r['oof']) for r in recs], axis=1)
        val_stack = np.stack([_rank01(r['val']) for r in recs], axis=1)
        dis_t = np.nanstd(oof_stack, axis=1)
        dis_v = np.nanstd(val_stack, axis=1)
        thresh_t = np.nanquantile(dis_t[np.isfinite(dis_t)], 1 - top_q)
        thresh_v = np.nanquantile(dis_v[np.isfinite(dis_v)], 1 - top_q)
        # OOF: rank_mean baseline; override top-q by specialist OOF
        anchor_oof = _rank01(np.nanmean(oof_stack, axis=1))
        spec_oof_r = _rank01(spec['oof'])
        oof = anchor_oof.copy()
        m = np.isfinite(dis_t) & (dis_t >= thresh_t)
        oof[m] = spec_oof_r[m]
        # Val: anchor.csv, override top-q by spec val
        anc_v = _rank01(anchor[col].values)
        spec_val_r = _rank01(spec['val'])
        val_pred = anc_v.copy()
        mv = np.isfinite(dis_v) & (dis_v >= thresh_v)
        val_pred[mv] = spec_val_r[mv]
        out[ep] = {'oof': oof, 'val': val_pred,
                   'oof_c': _c(y, valid, oof),
                   'specialist': spec['label'], 'n_overrides_val': int(mv.sum())}
    return out, {'top_q': top_q,
                 'specialists': {ep: out[ep]['specialist'] for ep, _ in ENDPOINTS},
                 'n_overrides': {ep: out[ep]['n_overrides_val'] for ep, _ in ENDPOINTS}}


# ---------------------------------------------------------------------------
VARIANTS = {
    'mahal_blend':   variant_mahal_blend,
    'bootstrap_var': variant_bootstrap_var,
    'disagree_soft': variant_disagree_soft,
    'conformal':     variant_conformal,
    'specialist':    variant_specialist,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(VARIANTS.keys()))
    args = ap.parse_args()

    train = load_annitia()
    val = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                       filename=Path(VAL_CSV).name, has_outcomes=False)
    int_ids = _val_int_ids(val)
    anchor = _anchor()

    fn = VARIANTS[args.variant]
    out, meta = fn(train, val, anchor)

    hep = out['hepatic']['val']
    death = out['death']['val']
    dest = _write(int_ids, hep, death, args.variant)

    hep_c = out['hepatic']['oof_c']; d_c = out['death']['oof_c']
    weighted = 0.7 * hep_c + 0.3 * d_c
    summary = {
        'variant': args.variant,
        'oof_c_hepatic': hep_c, 'oof_c_death': d_c,
        'oof_weighted_07_03': weighted,
        'csv': str(dest), 'meta': meta,
    }
    print(json.dumps(summary, indent=2, default=str))

    # Persist summary
    sjson = FULL / f'sub_master_unc_{args.variant}.summary.json'
    sjson.write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == '__main__':
    main()
