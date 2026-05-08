"""Hierarchical fusion for the expanded pool (post-Round-4 + Round-5).

Motivation: with ~70 NHANES T2.x branches + ~14 hparam-tuned models
joining the existing 117 hep / 85 death pool, a flat rank-mean lets
one large source dominate (BE has 80 models alone). This script
fuses HIERARCHICALLY:

  LAYER 1 — per-source pre-fusion
    For each source dir in {be, cd, hep_boost, branches, external,
    hparam}, build a single super-OOF + super-val per endpoint by
    rank_median across all valid models in that source.

  LAYER 2 — quality-stratified weighting
    Group sources by their best-internal CV-C: ≥0.85 → weight 1.0;
    0.70-0.85 → 0.5; <0.70 → 0.2. Weak sources can't drag good ones
    down.

  LAYER 3 — inter-source meta
    Compute rank_mean / rank_median / rank_trimmed of the (5-7)
    super-models. Plus a small Coxnet stack_meta (well-conditioned
    at 5-7 features × ~1000 rows). Plus stack_meta_xe with the
    other endpoint's super-models as cross-features.

  LAYER 4 — patient-aware route on super-models
    Anomaly-bucket and disagreement-bucket routing across the
    super-models (small bucket counts so each bucket has plenty of
    patients).

Final submission CSV: rank-mean of all of LAYER 3 + LAYER 4 outputs
(rank-rescaled). Honest OOF C reported per layer.

Usage:
    python tests/experiments/annitia/refuse_hierarchical.py
    python tests/experiments/annitia/refuse_hierarchical.py --output sub_master_HIER.csv
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
from experiments.annitia.fusion import (                                   # noqa: E402
    rank_matrix, fuse_equal, fuse_rrf, StackMetaFusion,
)


SOURCE_NAMES = ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam', 'research', 'research_v2')


def _rank01(x):
    finite = np.isfinite(x); out = np.full_like(x, np.nan, dtype=float)
    if finite.sum():
        out[finite] = rankdata(x[finite]) / max(finite.sum(), 1)
    return out


def _c(y, valid, scores):
    m = valid & np.isfinite(scores) & np.isfinite(y['time'])
    if m.sum() < 5: return float('nan')
    return float(concordance(y[m], scores[m]))


def _per_patient_disc(time, event, risk):
    finite = np.isfinite(time) & np.isfinite(risk)
    t = np.where(finite, time, np.nan); e = event.astype(bool) & finite
    ti = t[:, None]; tj = t[None, :]; ei = e[:, None]
    finite_pair = np.isfinite(ti) & np.isfinite(tj) & (ti < tj)
    comparable = finite_pair & ei
    ri = risk[:, None]; rj = risk[None, :]
    discordant = comparable & (ri < rj); tied = comparable & (ri == rj)
    pi = comparable.sum(axis=1); pj = comparable.sum(axis=0)
    di = discordant.sum(axis=1) + 0.5 * tied.sum(axis=1)
    dj = discordant.sum(axis=0) + 0.5 * tied.sum(axis=0)
    cnt = pi + pj; disc = di + dj
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, disc / cnt, np.nan)


def _collect_source(d, endpoint):
    """Return list of (label, cv, oof_array, val_array) for one source dir."""
    cv = d / 'cv_summary.json'
    if not cv.exists(): return []
    sm = json.loads(cv.read_text())
    out = []
    for r in sm.get('per_condition', {}).get(endpoint, []):
        cid = r.get('config_id', '')
        mc = r.get('mean_cindex', float('nan'))
        if not np.isfinite(mc): continue
        op = d / f'oof_{endpoint}_{cid}.npy'
        vp = d / f'val_{endpoint}_{cid}.npy'
        if not (op.exists() and vp.exists()): continue
        out.append({'label': cid, 'cv': float(mc),
                    'oof': np.load(op), 'val': np.load(vp)})
    return out


def _build_super_model(models, kind='top10_mean', top_k=10):
    """Aggregate a list of {oof, val, cv} dicts into one super-model.

    Default 'top10_mean': sort by CV C, take top-K, rank_mean. This
    avoids letting weaker models dilute the strong ones at the
    within-source level.
    """
    if not models: return None
    if kind == 'top10_mean' or kind == 'top10_median':
        ranked = sorted(models, key=lambda m: -m.get('cv', 0.0))[:top_k]
        oofs = np.stack([_rank01(m['oof']) for m in ranked], axis=1)
        vals = np.stack([_rank01(m['val']) for m in ranked], axis=1)
        if kind == 'top10_mean':
            super_oof = np.nanmean(oofs, axis=1)
            super_val = np.nanmean(vals, axis=1)
        else:
            super_oof = np.nanmedian(oofs, axis=1)
            super_val = np.nanmedian(vals, axis=1)
        cv_internal = max(m['cv'] for m in ranked)
        return {'oof': super_oof, 'val': super_val,
                'cv_internal': cv_internal, 'n_models': len(ranked),
                'n_total_in_source': len(models)}
    oofs = np.stack([_rank01(m['oof']) for m in models], axis=1)
    vals = np.stack([_rank01(m['val']) for m in models], axis=1)
    if kind == 'median':
        super_oof = np.nanmedian(oofs, axis=1)
        super_val = np.nanmedian(vals, axis=1)
    elif kind == 'mean':
        super_oof = np.nanmean(oofs, axis=1)
        super_val = np.nanmean(vals, axis=1)
    elif kind == 'trimmed':
        n = oofs.shape[1]; k = max(1, int(np.floor(n * 0.2 / 2)))
        s_oof = np.sort(oofs, axis=1); s_val = np.sort(vals, axis=1)
        super_oof = s_oof[:, k:n-k].mean(axis=1) if n > 2*k else s_oof.mean(axis=1)
        super_val = s_val[:, k:n-k].mean(axis=1) if n > 2*k else s_val.mean(axis=1)
    else:
        raise ValueError(kind)
    cv_internal = max(m['cv'] for m in models)
    return {'oof': super_oof, 'val': super_val,
            'cv_internal': cv_internal, 'n_models': len(models)}


def _quality_weight(cv):
    if cv >= 0.85: return 1.0
    if cv >= 0.70: return 0.5
    if cv >= 0.55: return 0.2
    return 0.0   # exclude failed sources entirely


def _gate_pretraining_source(super_models, src='external', floor=0.55, ceil_dom=0.97):
    """Subjective rules for the pretraining-derived source:

      - if its super-OOF C is below `floor`, drop it (epic fail).
      - if it dominates everyone else by > 0.05 AND > ceil_dom, double
        its weight (it's discovered something nobody else has).
    """
    if src not in super_models:
        return None
    src_c = super_models[src]['oof_c']
    others = [m['oof_c'] for s, m in super_models.items() if s != src]
    if src_c < floor:
        print(f'  >> dropping {src} (super OOF C={src_c:.4f} < floor {floor})')
        del super_models[src]
        return 'dropped'
    if others and src_c > max(others) + 0.05 and src_c > ceil_dom:
        print(f'  >> {src} dominates: super C={src_c:.4f} '
              f'vs next best={max(others):.4f}; boost weight 2x')
        return 'boosted'
    return 'kept'


def _coxnet_predict_safe(X_train, y_train, X_test, valid_mask=None,
                         alpha_index_frac=0.5):
    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
    except ImportError:
        return np.full(X_test.shape[0], np.nan)
    finite = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train['time'])
    if valid_mask is not None: finite = finite & valid_mask
    if finite.sum() < X_train.shape[1] + 5 or not y_train[finite]['event'].any():
        return np.full(X_test.shape[0], np.nan)
    try:
        m = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                   max_iter=2000, fit_baseline_model=False)
        m.fit(X_train[finite], y_train[finite])
        coefs = np.asarray(m.coef_)
        if coefs.ndim == 2:
            idx = max(0, min(coefs.shape[1] - 1,
                             int(round((coefs.shape[1] - 1) * alpha_index_frac))))
            w = coefs[:, idx]
        else: w = coefs
        if not np.any(np.abs(w) > 1e-12):
            w = np.ones(X_train.shape[1]) / X_train.shape[1]
        # Defensive impute test
        Xt = X_test.copy()
        col_means = np.nanmean(X_train[finite], axis=0)
        for j in range(Xt.shape[1]):
            mask = ~np.isfinite(Xt[:, j])
            if mask.any(): Xt[mask, j] = col_means[j]
        return Xt @ w.astype(np.float64)
    except Exception:
        return np.full(X_test.shape[0], np.nan)


def _nested_oof_stack(super_models_self, super_models_cross, y, valid):
    """Honest OOF for a Coxnet stacker on super-models (5-7 features → safe)."""
    R_self = np.column_stack([_rank01(s['oof']) for s in super_models_self])
    if super_models_cross:
        R_cross = np.column_stack([_rank01(s['oof']) for s in super_models_cross])
        R = np.concatenate([R_self, R_cross], axis=1)
    else:
        R = R_self
    finite = np.all(np.isfinite(R), axis=1) & valid & np.isfinite(y['time'])
    idx = np.where(finite)[0]
    if idx.size < R.shape[1] + 10:
        return np.full(len(y), np.nan)
    R_k = R[idx]; y_k = y[idx]
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(y), np.nan)
    for tr, va in skf.split(idx, y_k['event'].astype(int)):
        oof[idx[va]] = _coxnet_predict_safe(R_k[tr], y_k[tr], R_k[va])
    return oof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--output', default='tests/results/annitia/full/sub_master_HIER.csv')
    args = ap.parse_args()

    res_root = _REPO / 'tests' / 'results' / 'annitia'
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)
    n_train = len(train['patient_ids'])
    n_val = len(val['patient_ids'])

    # ---- Anomaly + disagreement signals (computed once, used in Layer 4) ----
    from sklearn.ensemble import IsolationForest
    views_t = build_views(train, view_names=['snapshot', 'clinical_pro'])
    views_v = build_views(val,   view_names=['snapshot', 'clinical_pro'])
    Xt_raw = np.nan_to_num(np.concatenate([views_t[v][0] for v in ['snapshot','clinical_pro']], axis=1))
    Xv_raw = np.nan_to_num(np.concatenate([views_v[v][0] for v in ['snapshot','clinical_pro']], axis=1))
    iso = IsolationForest(n_estimators=300, random_state=42).fit(Xt_raw)
    anom_t = -iso.score_samples(Xt_raw); anom_v = -iso.score_samples(Xv_raw)

    summary = {}
    master = pd.DataFrame({'trustii_id': val['trustii_id']})
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        y = train[endpoint]['y_struct']; valid = train[endpoint]['valid_mask']
        print(f'\n=== {endpoint} ===')

        # ---- LAYER 1: per-source pre-fusion ----
        super_models = {}
        for src in SOURCE_NAMES:
            d = res_root / src
            models = _collect_source(d, endpoint)
            if not models:
                print(f'  {src:12s} (skipped — 0 models)')
                continue
            # rank_median super-model for this source
            sm = _build_super_model(models, kind='median')
            sm['source'] = src
            sm['oof_c'] = _c(y, valid, sm['oof'])
            super_models[src] = sm
            print(f'  {src:12s} {len(models):3d} models → super OOF C={sm["oof_c"]:.4f} '
                  f'(internal best CV={sm["cv_internal"]:.4f})')

        if not super_models:
            print(f'  {endpoint}: no sources, filling zeros')
            master[col] = 0.0; continue

        # ---- LAYER 1.5: subjective gate on pretraining sources ----
        # Drop epic fails, boost overwhelming wins
        for pretrain_src in ('external', 'hparam', 'research', 'research_v2'):
            outcome = _gate_pretraining_source(super_models, pretrain_src)
            if outcome == 'boosted':
                # Mark for boost (handled below)
                super_models[pretrain_src]['_boost'] = 2.0

        # ---- LAYER 2: per-source quality-stratified weights ----
        weights = {}
        for src, sm in super_models.items():
            base = _quality_weight(sm['cv_internal'])
            boost = sm.get('_boost', 1.0)
            weights[src] = base * boost
        wsum = sum(weights.values()) or 1.0
        weights = {k: v / wsum for k, v in weights.items()}
        print(f'  weights: {dict([(k, round(v, 3)) for k, v in weights.items()])}')

        # ---- Disagreement signal across the super-models ----
        R_super_oof = np.column_stack([_rank01(super_models[src]['oof'])
                                        for src in super_models])
        R_super_val = np.column_stack([_rank01(super_models[src]['val'])
                                        for src in super_models])
        disag_t = np.nanstd(R_super_oof, axis=1)
        disag_v = np.nanstd(R_super_val, axis=1)

        # ---- LAYER 3: inter-source meta aggregations ----
        layer3 = {}
        # rank_mean (weighted by quality)
        srcs = list(super_models.keys())
        w_arr = np.array([weights[s] for s in srcs])
        layer3['rank_mean_w'] = {
            'oof': R_super_oof @ w_arr,
            'val': R_super_val @ w_arr,
        }
        layer3['rank_mean'] = {
            'oof': np.mean(R_super_oof, axis=1),
            'val': np.mean(R_super_val, axis=1),
        }
        layer3['rank_median'] = {
            'oof': np.median(R_super_oof, axis=1),
            'val': np.median(R_super_val, axis=1),
        }
        # rank_trimmed: drop top/bottom 1 super-model
        if R_super_oof.shape[1] >= 4:
            s_o = np.sort(R_super_oof, axis=1); s_v = np.sort(R_super_val, axis=1)
            layer3['rank_trimmed'] = {
                'oof': s_o[:, 1:-1].mean(axis=1),
                'val': s_v[:, 1:-1].mean(axis=1),
            }
        # stack_meta (Coxnet on 5-7 super-models — safe)
        try:
            stack_oof = _nested_oof_stack(list(super_models.values()), None, y, valid)
            sm_full = StackMetaFusion(l1_ratio=0.5)
            sm_full.fit([s['oof'] for s in super_models.values()], y, valid_mask=valid)
            stack_val = sm_full.transform([s['val'] for s in super_models.values()])
            layer3['stack_meta'] = {'oof': stack_oof, 'val': stack_val}
        except Exception as e:
            print(f'  stack_meta failed: {e}')

        # rrf
        layer3['rrf'] = {
            'oof': fuse_rrf([super_models[s]['oof'] for s in srcs], k=60.0),
            'val': fuse_rrf([super_models[s]['val'] for s in srcs], k=60.0),
        }

        for name, d in layer3.items():
            d['oof_c'] = _c(y, valid, d['oof'])

        # ---- LAYER 4: patient-aware routing across super-models ----
        layer4 = {}
        # 4a. anomaly-bucket (3 buckets — small, robust at this granularity)
        keep_a = valid & np.isfinite(anom_t)
        if keep_a.sum() >= 100:
            edges = np.nanquantile(anom_t[keep_a], np.linspace(0, 1, 4))
            edges[0] = -np.inf; edges[-1] = np.inf
            chosen = []
            r_oof = np.full(n_train, np.nan); r_val = np.full(n_val, np.nan)
            for b in range(3):
                bm = keep_a & (anom_t >= edges[b]) & (anom_t < edges[b + 1])
                if bm.sum() < 30:
                    chosen.append(srcs[0]); continue
                # pick super-model with lowest discordance on bucket
                best, best_src = -np.inf, srcs[0]
                for s in srcs:
                    d_s = _per_patient_disc(y['time'], y['event'], super_models[s]['oof'])
                    sub = d_s[bm]; sub = sub[np.isfinite(sub)]
                    if len(sub) < 5: continue
                    sc = 1.0 - np.mean(sub)
                    if sc > best: best, best_src = sc, s
                chosen.append(best_src)
                r_oof[bm] = _rank01(super_models[best_src]['oof'])[bm]
                vm = (anom_v >= edges[b]) & (anom_v < edges[b + 1])
                r_val[vm] = _rank01(super_models[best_src]['val'])[vm]
            layer4['anomaly_route'] = {'oof': r_oof, 'val': r_val,
                                        'oof_c': _c(y, valid, r_oof),
                                        'meta': chosen}
        # 4b. disagreement-bucket (3 buckets)
        keep_d = valid & np.isfinite(disag_t)
        if keep_d.sum() >= 100:
            edges = np.nanquantile(disag_t[keep_d], np.linspace(0, 1, 4))
            edges[0] = -np.inf; edges[-1] = np.inf
            chosen = []
            r_oof = np.full(n_train, np.nan); r_val = np.full(n_val, np.nan)
            for b in range(3):
                bm = keep_d & (disag_t >= edges[b]) & (disag_t < edges[b + 1])
                if bm.sum() < 30:
                    chosen.append(srcs[0]); continue
                best, best_src = -np.inf, srcs[0]
                for s in srcs:
                    d_s = _per_patient_disc(y['time'], y['event'], super_models[s]['oof'])
                    sub = d_s[bm]; sub = sub[np.isfinite(sub)]
                    if len(sub) < 5: continue
                    sc = 1.0 - np.mean(sub)
                    if sc > best: best, best_src = sc, s
                chosen.append(best_src)
                r_oof[bm] = _rank01(super_models[best_src]['oof'])[bm]
                vm = (disag_v >= edges[b]) & (disag_v < edges[b + 1])
                r_val[vm] = _rank01(super_models[best_src]['val'])[vm]
            layer4['disagreement_route'] = {'oof': r_oof, 'val': r_val,
                                             'oof_c': _c(y, valid, r_oof),
                                             'meta': chosen}

        # ---- FINAL: rank-mean across all Layer 3 + Layer 4 outputs ----
        all_outputs = list(layer3.values()) + list(layer4.values())
        finite_outputs = [o for o in all_outputs if np.isfinite(o['oof_c'])]
        print(f'  Layer 3+4 component OOF Cs:')
        for name, d in {**layer3, **layer4}.items():
            print(f'    {name:24s} OOF C={d["oof_c"]:.4f}')
        # Final = mean of rank-transformed components
        oof_stack = np.column_stack([_rank01(o['oof']) for o in finite_outputs])
        val_stack = np.column_stack([_rank01(o['val']) for o in finite_outputs])
        final_oof = np.nanmean(oof_stack, axis=1)
        final_val = np.nanmean(val_stack, axis=1)
        final_c = _c(y, valid, final_oof)
        print(f'  HIERARCHICAL FINAL OOF C={final_c:.4f}')

        master[col] = final_val
        summary[endpoint] = {
            'final_oof_c': final_c,
            'sources': {s: {'oof_c': sm['oof_c'], 'cv_internal': sm['cv_internal'],
                            'n_models': sm['n_models'], 'weight': weights[s]}
                        for s, sm in super_models.items()},
            'layer3_components': {n: d['oof_c'] for n, d in layer3.items()},
            'layer4_components': {n: d['oof_c'] for n, d in layer4.items()},
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out, index=False)
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nDONE')
    print(f'  CSV: {out}')
    for ep, s in summary.items():
        print(f'  {ep}: HIER OOF C={s["final_oof_c"]:.4f}')


if __name__ == '__main__':
    main()
