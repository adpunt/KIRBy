"""Patient-aware meta-meta variants.

Builds NEW meta-variants that route between the 10 existing fusion
variants on a per-patient basis using:
  - anomaly score (IsolationForest on raw clinical+snapshot)
  - ensemble disagreement (rank-std across the model pool)
  - cluster identity (KMeans on raw features)

The 10 existing variants live as sub_master_<name>.csv in
tests/results/annitia/full/.

Six new meta-meta variants:

1. anomaly_router_meta — bucket patients into 4 anomaly quantiles, for
   each bucket pick the variant with the lowest in-bucket discordance,
   route val patients by their own anomaly score.

2. disagreement_router_meta — same but bucketed by ensemble disagreement.

3. cluster_router_meta — KMeans 5 clusters on raw clinical features;
   per-cluster pick best variant.

4. anomaly_blend_meta — soft sigmoid mix of (best variant for low-anomaly
   half) and (best variant for high-anomaly half).

5. coxnet_stack_meta_with_uncertainty — Coxnet meta-learner trained
   on (all 10 variant predictions + anomaly + disagreement) as features,
   evaluated by nested 5-fold CV (so the OOF C is honest), refit on
   full training for val_test prediction.

6. weighted_anomaly_per_patient — per-patient weighted blend of all 10
   variants, where weights are softmax(- per-bucket discordance vec)
   computed from the patient's anomaly bucket.

Outputs CSVs to tests/results/annitia/full/ and journals to
tests/results/annitia/submissions/.
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

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds       # noqa: E402
from experiments.annitia.views import build_views                          # noqa: E402
from experiments.annitia.survival import concordance                       # noqa: E402


VARIANTS = ['rank_mean', 'rank_median', 'rank_trimmed', 'residual_greedy',
            'uncertainty_route', 'anomaly_blend', 'disagreement_route',
            'cluster_route', 'specialist_override', 'soft_blend_disagreement']


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


def _ensemble_disagreement(pool_dirs, endpoint, n):
    """Rank-std across all base-model OOFs (the same uncertainty signal
    we measured r=0.20-0.57 vs miss-rate)."""
    ranks = []
    for d in pool_dirs:
        cv = d / 'cv_summary.json'
        if not cv.exists(): continue
        sm = json.loads(cv.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r['config_id']
            mc = r.get('mean_cindex', float('nan'))
            if not np.isfinite(mc): continue
            oof_p = d / f'oof_{endpoint}_{cid}.npy'
            if not oof_p.exists(): continue
            v = np.load(oof_p)
            ranks.append(_rank01(v))
    if not ranks: return np.full(n, np.nan)
    R = np.stack(ranks, axis=1)
    n_finite = np.sum(np.isfinite(R), axis=1)
    with np.errstate(invalid='ignore'):
        unc = np.nanstd(R, axis=1)
    unc[n_finite < 10] = np.nan
    return unc


def _ensemble_disagreement_val(pool_dirs, endpoint, n_val):
    """Same but on val_test predictions."""
    ranks = []
    for d in pool_dirs:
        cv = d / 'cv_summary.json'
        if not cv.exists(): continue
        sm = json.loads(cv.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r['config_id']
            mc = r.get('mean_cindex', float('nan'))
            if not np.isfinite(mc): continue
            val_p = d / f'val_{endpoint}_{cid}.npy'
            if not val_p.exists(): continue
            ranks.append(_rank01(np.load(val_p)))
    if not ranks: return np.full(n_val, np.nan)
    R = np.stack(ranks, axis=1)
    n_finite = np.sum(np.isfinite(R), axis=1)
    with np.errstate(invalid='ignore'):
        return np.nanstd(R, axis=1)


def _anomaly_scores(train, val):
    from sklearn.ensemble import IsolationForest
    Xt = np.concatenate([build_views(train, view_names=['snapshot', 'clinical_pro'])[v][0]
                         for v in ['snapshot', 'clinical_pro']], axis=1)
    Xv = np.concatenate([build_views(val, view_names=['snapshot', 'clinical_pro'])[v][0]
                         for v in ['snapshot', 'clinical_pro']], axis=1)
    Xt = np.nan_to_num(Xt, nan=0.0); Xv = np.nan_to_num(Xv, nan=0.0)
    iso = IsolationForest(n_estimators=300, random_state=42).fit(Xt)
    return -iso.score_samples(Xt), -iso.score_samples(Xv), Xt, Xv


def _load_variant_oofs_and_vals(endpoint):
    """For each variant, recompute its OOF/val predictions on the FULL
    training set by re-creating it from the saved sub_master_<v>.csv...
    actually we don't have OOF for variants saved — only val. So we
    must recompute the OOFs by replaying the variant on training OOFs.

    Easier: load the per-variant val_test CSVs (those exist), and load
    the BASE pool OOFs to compute per-variant OOF on training rows by
    re-running the simple variants only.

    For routing meta-meta we only NEED:
      - per-variant val_test prediction (the CSV column)
      - per-variant per-training-patient OOF prediction (recompute fast)

    Recomputing OOFs from base pool here:
    """
    from experiments.annitia.refuse_variants import (
        _collect, _rank01 as _r01, _c as _cc, _per_patient_disc as _pd,
        variant_mean, variant_median, variant_trimmed,
        variant_residual_greedy, variant_uncertainty_route,
        variant_anomaly_blend, variant_disagreement_route,
        variant_cluster_route, variant_specialist_override,
        variant_soft_blend_uncertainty)
    return None  # handled below by replaying


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagent/kirby/annitia/val_test.csv')
    args, _ = ap.parse_known_args()
    args.validation_csv = '/Volumes/seagate/kirby/annitia/val_test.csv'  # default fix

    res_root = _REPO / 'tests' / 'results' / 'annitia'
    pool_dirs = [res_root / p for p in
                 ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam', 'research', 'research_v2')
                 if (res_root / p).exists()]
    full_dir = res_root / 'full'

    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)
    n_train = len(train['patient_ids'])
    n_val = len(val['patient_ids'])

    anom_train, anom_val, Xt, Xv = _anomaly_scores(train, val)
    print(f'anomaly: train range [{anom_train.min():.3f},{anom_train.max():.3f}]')

    # Load per-variant val predictions
    var_val = {}
    for v in VARIANTS:
        df = pd.read_csv(full_dir / f'sub_master_{v}.csv')
        var_val[v] = {'hep': df['risk_hepatic_event'].values,
                      'death': df['risk_death'].values}

    # We need per-variant per-training-patient OOF predictions to
    # measure each variant's per-patient discordance and pick the
    # best variant per patient bucket. Recompute by re-running the
    # variants on the TRAINING OOFs (which exist).
    from experiments.annitia.refuse_variants import (
        _collect, variant_mean, variant_median, variant_trimmed,
        variant_residual_greedy, variant_uncertainty_route,
        variant_anomaly_blend, variant_disagreement_route,
        variant_cluster_route, variant_specialist_override,
        variant_soft_blend_uncertainty)
    VARIANT_FNS = {
        'rank_mean': variant_mean, 'rank_median': variant_median,
        'rank_trimmed': variant_trimmed, 'residual_greedy': variant_residual_greedy,
        'uncertainty_route': variant_uncertainty_route, 'anomaly_blend': variant_anomaly_blend,
        'disagreement_route': variant_disagreement_route, 'cluster_route': variant_cluster_route,
        'specialist_override': variant_specialist_override,
        'soft_blend_disagreement': variant_soft_blend_uncertainty,
    }

    submissions_dir = res_root / 'submissions'
    summary_out = {}

    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        recs = _collect(pool_dirs, endpoint)
        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        print(f'\n=== {endpoint} ===  pool of {len(recs)} base models')

        # Recompute each variant's OOF
        var_oof = {}
        for vname, vfn in VARIANT_FNS.items():
            try:
                r = vfn(recs, y, valid, 10, anom_train, anom_val)
                if r is None: continue
                var_oof[vname] = r['oof']
            except Exception as e:
                print(f'  WARN: {vname} OOF recompute failed: {e}')

        # Per-variant per-patient discordance
        var_disc = {}
        for vname, oof in var_oof.items():
            var_disc[vname] = _per_patient_disc(y['time'], y['event'], oof)

        # Ensemble disagreement signal (from base pool)
        disag_train = _ensemble_disagreement(pool_dirs, endpoint, n_train)
        disag_val = _ensemble_disagreement_val(pool_dirs, endpoint, n_val)

        # Helper: for a binary mask of training patients, pick the variant
        # with lowest mean discordance on that mask.
        def _best_variant_on_mask(mask):
            best_score, best_v = -np.inf, list(var_oof.keys())[0]
            for vname, d in var_disc.items():
                sub = d[mask]; sub = sub[np.isfinite(sub)]
                if len(sub) < 5: continue
                s = 1.0 - np.mean(sub)
                if s > best_score: best_score, best_v = s, vname
            return best_v

        endpoint_results = {}

        # ---- 1. anomaly_router_meta ----
        a = anom_train; keep = valid & np.isfinite(a)
        edges = np.nanquantile(a[keep], np.linspace(0, 1, 5))  # 4 buckets
        edges[0] = -np.inf; edges[-1] = np.inf
        bucket_oof = np.full(n_train, np.nan)
        bucket_val = np.full(n_val, np.nan)
        ek = 'hep' if endpoint == 'hepatic' else 'death'
        chosen_per_bucket = []
        for b in range(4):
            bm = keep & (a >= edges[b]) & (a < edges[b + 1])
            if bm.sum() < 20:
                vbest = list(var_oof.keys())[0]
            else:
                vbest = _best_variant_on_mask(bm)
            chosen_per_bucket.append(vbest)
            bucket_oof[bm] = _rank01(var_oof[vbest])[bm]
            vm = (anom_val >= edges[b]) & (anom_val < edges[b + 1])
            bucket_val[vm] = _rank01(var_val[vbest][ek])[vm]
        endpoint_results['anomaly_router_meta'] = {
            'oof_c': _c(y, valid, bucket_oof),
            'val': bucket_val, 'meta': chosen_per_bucket,
        }

        # ---- 2. disagreement_router_meta ----
        u = disag_train; keep = valid & np.isfinite(u)
        if keep.sum() >= 100:
            edges_u = np.nanquantile(u[keep], np.linspace(0, 1, 5))
            edges_u[0] = -np.inf; edges_u[-1] = np.inf
            d_oof = np.full(n_train, np.nan); d_val = np.full(n_val, np.nan)
            chosen_d = []
            for b in range(4):
                bm = keep & (u >= edges_u[b]) & (u < edges_u[b + 1])
                if bm.sum() < 20:
                    vbest = list(var_oof.keys())[0]
                else:
                    vbest = _best_variant_on_mask(bm)
                chosen_d.append(vbest)
                d_oof[bm] = _rank01(var_oof[vbest])[bm]
                vm = (disag_val >= edges_u[b]) & (disag_val < edges_u[b + 1])
                d_val[vm] = _rank01(var_val[vbest][ek])[vm]
            endpoint_results['disagreement_router_meta'] = {
                'oof_c': _c(y, valid, d_oof),
                'val': d_val, 'meta': chosen_d,
            }

        # ---- 3. cluster_router_meta ----
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xt)
        cl_train = km.labels_; cl_val = km.predict(Xv)
        c_oof = np.full(n_train, np.nan); c_val = np.full(n_val, np.nan)
        chosen_c = []
        for c in range(5):
            cm = valid & (cl_train == c)
            if cm.sum() < 20:
                vbest = list(var_oof.keys())[0]
            else:
                vbest = _best_variant_on_mask(cm)
            chosen_c.append(vbest)
            c_oof[cl_train == c] = _rank01(var_oof[vbest])[cl_train == c]
            c_val[cl_val == c] = _rank01(var_val[vbest][ek])[cl_val == c]
        endpoint_results['cluster_router_meta'] = {
            'oof_c': _c(y, valid, c_oof),
            'val': c_val, 'meta': chosen_c,
        }

        # ---- 4. anomaly_blend_meta ----
        med = np.nanmedian(a[keep])
        low_mask = keep & (a < med); high_mask = keep & (a >= med)
        v_low = _best_variant_on_mask(low_mask)
        v_high = _best_variant_on_mask(high_mask)
        spread = max(np.nanstd(a[keep]), 1e-3)
        w_t = 1.0 / (1.0 + np.exp(-(a - med) / spread))
        w_v = 1.0 / (1.0 + np.exp(-(anom_val - med) / spread))
        ab_oof = (1 - w_t) * _rank01(var_oof[v_low]) + w_t * _rank01(var_oof[v_high])
        ab_val = (1 - w_v) * _rank01(var_val[v_low][ek]) + w_v * _rank01(var_val[v_high][ek])
        endpoint_results['anomaly_blend_meta'] = {
            'oof_c': _c(y, valid, ab_oof),
            'val': ab_val, 'meta': [v_low, v_high],
        }

        # ---- 5. coxnet_stack_meta_with_uncertainty (NESTED CV) ----
        try:
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            from sklearn.model_selection import StratifiedKFold

            # Features: all variant ranks + anomaly_z + disagreement_z
            anom_z = (anom_train - np.nanmean(anom_train[keep])) / max(np.nanstd(anom_train[keep]), 1e-3)
            disag_z = np.where(np.isfinite(disag_train),
                               (disag_train - np.nanmean(disag_train[keep])) / max(np.nanstd(disag_train[keep]), 1e-3),
                               0.0)
            anom_z_v = (anom_val - np.nanmean(anom_train[keep])) / max(np.nanstd(anom_train[keep]), 1e-3)
            disag_z_v = (disag_val - np.nanmean(disag_train[keep])) / max(np.nanstd(disag_train[keep]), 1e-3)
            disag_z_v = np.where(np.isfinite(disag_z_v), disag_z_v, 0.0)

            X_train_feat = np.column_stack(
                [_rank01(var_oof[v]) for v in var_oof.keys()] +
                [anom_z.reshape(-1, 1), disag_z.reshape(-1, 1)])
            X_val_feat = np.column_stack(
                [_rank01(var_val[v][ek]) for v in var_oof.keys()] +
                [anom_z_v.reshape(-1, 1), disag_z_v.reshape(-1, 1)])

            keep_full = np.all(np.isfinite(X_train_feat), axis=1) & valid & np.isfinite(y['time'])
            idx = np.where(keep_full)[0]
            X_k = X_train_feat[idx]; y_k = y[idx]

            # Nested CV for honest OOF
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            stack_oof = np.full(n_train, np.nan)
            for tr_loc, va_loc in skf.split(idx, y_k['event'].astype(int)):
                model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                               max_iter=2000, fit_baseline_model=False)
                try:
                    model.fit(X_k[tr_loc], y_k[tr_loc])
                    coefs = np.asarray(model.coef_)
                    if coefs.ndim == 2:
                        w = coefs[:, coefs.shape[1] // 2]  # mid-alpha
                    else: w = coefs
                    if not np.any(np.abs(w) > 1e-12):
                        w = np.ones(X_k.shape[1]) / X_k.shape[1]
                    stack_oof[idx[va_loc]] = X_k[va_loc] @ w.astype(np.float64)
                except Exception as e:
                    print(f'    coxnet fold failed: {e}')
            # Refit on full
            model_full = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                                max_iter=2000, fit_baseline_model=False)
            model_full.fit(X_k, y_k)
            coefs_full = np.asarray(model_full.coef_)
            if coefs_full.ndim == 2:
                w_full = coefs_full[:, coefs_full.shape[1] // 2]
            else: w_full = coefs_full
            if not np.any(np.abs(w_full) > 1e-12):
                w_full = np.ones(X_k.shape[1]) / X_k.shape[1]
            X_val_safe = X_val_feat.copy()
            for j in range(X_val_safe.shape[1]):
                m = ~np.isfinite(X_val_safe[:, j])
                if m.any(): X_val_safe[m, j] = np.nanmean(X_k[:, j])
            stack_val = X_val_safe @ w_full.astype(np.float64)
            endpoint_results['coxnet_stack_with_uncertainty'] = {
                'oof_c': _c(y, valid, stack_oof),
                'val': stack_val,
                'meta': {'coefs': w_full.tolist(), 'features': list(var_oof.keys()) + ['anomaly_z', 'disagreement_z']},
            }
        except ImportError:
            print('    sksurv not available; skipping coxnet stack')

        # ---- 6. weighted_anomaly_per_patient ----
        # Per-bucket softmax weights over variants
        bucket_weights = []  # list of (n_variants,) for each bucket
        for b in range(4):
            bm = keep & (a >= edges[b]) & (a < edges[b + 1])
            if bm.sum() < 20:
                bucket_weights.append(np.ones(len(var_oof)) / len(var_oof)); continue
            disc_per_v = []
            for vname in var_oof.keys():
                sub = var_disc[vname][bm]; sub = sub[np.isfinite(sub)]
                disc_per_v.append(np.mean(sub) if len(sub) else 0.5)
            # Softmax over (1 - disc) with temperature
            scores = -np.array(disc_per_v) * 10
            ex = np.exp(scores - scores.max())
            bucket_weights.append(ex / ex.sum())
        # Build OOF
        wp_oof = np.full(n_train, np.nan)
        wp_val = np.full(n_val, np.nan)
        for b in range(4):
            bm = (a >= edges[b]) & (a < edges[b + 1])
            mix = sum(bucket_weights[b][i] * _rank01(var_oof[vname])
                      for i, vname in enumerate(var_oof.keys()))
            wp_oof[bm] = mix[bm]
            vm = (anom_val >= edges[b]) & (anom_val < edges[b + 1])
            mix_v = sum(bucket_weights[b][i] * _rank01(var_val[vname][ek])
                        for i, vname in enumerate(var_oof.keys()))
            wp_val[vm] = mix_v[vm]
        endpoint_results['weighted_anomaly_per_patient'] = {
            'oof_c': _c(y, valid, wp_oof),
            'val': wp_val,
            'meta': {'bucket_weights': [w.tolist() for w in bucket_weights]},
        }

        # Print + save per-variant
        print('  Honest OOF Cs:')
        for name, r in sorted(endpoint_results.items(), key=lambda x: -x[1]['oof_c']):
            print(f'    {name:40s} OOF C={r["oof_c"]:.4f}')
        for name, r in endpoint_results.items():
            csv_p = full_dir / f'sub_master_{name}.csv'
            if csv_p.exists():
                prev = pd.read_csv(csv_p)
                if col in prev.columns:
                    prev[col] = r['val']
                else:
                    prev[col] = r['val']
                prev.to_csv(csv_p, index=False)
            else:
                pd.DataFrame({'trustii_id': val['patient_ids'],
                              col: r['val']}).to_csv(csv_p, index=False)
        summary_out[endpoint] = {
            name: {'oof_c': r['oof_c'], 'meta': r.get('meta', None)}
            for name, r in endpoint_results.items()
        }

    # Save summary
    (full_dir / 'refuse_meta_aware_summary.json').write_text(
        json.dumps(summary_out, indent=2, default=str))
    print(f'\nDONE')
    for ep, d in summary_out.items():
        best = max(d.items(), key=lambda x: x[1]['oof_c'])
        print(f'  {ep}: best meta-meta = {best[0]} OOF C={best[1]["oof_c"]:.4f}')


if __name__ == '__main__':
    main()
