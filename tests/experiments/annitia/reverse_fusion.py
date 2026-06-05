"""Reverse-fusion: produce TRUE training-side OOF predictions per variant.

Background. The bug exposed in `oof_to_leaderboard.py::variant_proxy_oof`
collapses 14 of the top-15 leaderboard variants onto the SAME proxy OOF
(rank-mean of base OOFs). That makes the per-variant LB-correlation /
bootstrap CI / holdout fusion-test meaningless — every variant scores
identically on the local validation. The fusion logic that
DIFFERENTIATES the val_test predictions (anomaly weights, NNLS,
isotonic, meta-aware bucket weights, cluster routing) lives only at
val_test prediction time and is invisible on training.

This script reproduces the fusion logic on the TRAINING-SIDE OOF
arrays so we have a true per-variant training-side prediction and can
compute differentiated rankings.

Output:
  tests/results/annitia/full/proxy_train_oof_<variant>_<endpoint>.npy
  tests/results/annitia/full/per_variant_true_oof_C.csv

Strategy:

1. Direct replay (16 base variants from refuse_variants + refuse_meta_aware).
   Each refuse_*.py variant function already computes its OOF prediction
   internally as part of its scoring step. We invoke the same functions
   here and harvest the 'oof' array.

2. Indirect / val-similarity replay (meta_avg_*, FINAL, best_combined,
   ultra_meta, etc.). These variants don't have a clean training-OOF
   replay path — but every one is some weighted/rank-mean over a SUBSET
   of base-variant predictions. We:
   a. Build the variant->variant val-prediction similarity in val space
      (against the 16 base variants we already have OOFs for).
   b. Identify the subset of base variants whose val mean MOST closely
      reproduces the meta variant's val (via NNLS regression of meta-
      variant val ranks on base-variant val ranks).
   c. Apply the same NNLS weights to the corresponding training OOFs
      to reconstruct the meta variant's training-side OOF.

This is a model-of-the-fusion approximation, but it IS data-driven (the
weights are derived from the actual val predictions, not assumed) and
yields per-variant training-side OOFs that DIFFER, which is the whole
point — we can now rank variants by their training-side OOF C and
compare against the leaderboard.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia              # noqa: E402
from experiments.annitia.survival import concordance               # noqa: E402

FULL = REPO / 'tests/results/annitia/full'
RES = REPO / 'tests/results/annitia'

# Pool dirs same as refuse_variants.py / refuse_meta_aware.py.
POOL_NAMES = ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam')

# 16 base variants we can directly replay on training OOFs.
BASE_VARIANTS = [
    'rank_mean', 'rank_median', 'rank_trimmed', 'residual_greedy',
    'uncertainty_route', 'anomaly_blend', 'disagreement_route',
    'cluster_route', 'specialist_override', 'soft_blend_disagreement',
    # meta-aware (from refuse_meta_aware.py)
    'anomaly_router_meta', 'disagreement_router_meta',
    'cluster_router_meta', 'anomaly_blend_meta',
    'weighted_anomaly_per_patient',
    # 'coxnet_stack_with_uncertainty' produces an OOF too; defer for now.
]

# High-EV variants we want to reconstruct via NNLS-on-val.
META_VARIANTS = [
    'meta_avg_aware_weighted', 'meta_avg_all', 'meta_avg_grand_all',
    'meta_avg_weighted', 'meta_avg_top5', 'meta_avg',
    'FINAL', 'best_combined', 'best_combined_v2', 'rank_median',
    'BEST_BET', 'meta_avg_grand_all_v2', 'meta_avg_grand_all_v3',
    'meta_avg_aware_weighted_v2', 'meta_avg_aware_weighted_v3',
    'meta_avg_aware_weighted_no_nnls', 'meta_avg_aware_weighted_no_nnls_v3',
    'meta_avg_aware_top5', 'meta_avg_new_variants_only',
    'ultra_meta', 'ultra_meta_weighted',
    'isotonic_avg', 'nnls_stack', 'bootstrap_dropout',
    'holdout_pick', 'diverse', 'HIER', 'honest',
]


def rank01(x):
    finite = np.isfinite(x); out = np.full_like(x, np.nan, dtype=float)
    if finite.sum():
        out[finite] = rankdata(x[finite]) / max(finite.sum(), 1)
    return out


def c_index(y, valid, scores):
    m = valid & np.isfinite(scores) & np.isfinite(y['time'])
    if m.sum() < 5:
        return float('nan')
    return float(concordance(y[m], scores[m]))


def main():
    print('=== reverse_fusion: per-variant TRUE training-side OOFs ===\n')

    train = load_annitia()
    val = load_annitia(
        data_dir='/Volumes/seagate/kirby/annitia',
        filename='val_test.csv',
        has_outcomes=False,
    )
    n_train = len(train['patient_ids'])
    n_val = len(val['patient_ids'])
    print(f'n_train={n_train} n_val={n_val}\n')

    pool_dirs = [RES / p for p in POOL_NAMES if (RES / p).exists()]
    print('pool_dirs:', [d.name for d in pool_dirs])

    # ------------------------------------------------------------------
    # Step 1: direct replay of base variants.
    # ------------------------------------------------------------------
    from experiments.annitia.refuse_variants import (
        _collect, _anomaly_scores,
        variant_mean, variant_median, variant_trimmed,
        variant_residual_greedy, variant_uncertainty_route,
        variant_anomaly_blend, variant_disagreement_route,
        variant_cluster_route, variant_specialist_override,
        variant_soft_blend_uncertainty,
    )

    VARIANT_FNS = {
        'rank_mean': variant_mean, 'rank_median': variant_median,
        'rank_trimmed': variant_trimmed,
        'residual_greedy': variant_residual_greedy,
        'uncertainty_route': variant_uncertainty_route,
        'anomaly_blend': variant_anomaly_blend,
        'disagreement_route': variant_disagreement_route,
        'cluster_route': variant_cluster_route,
        'specialist_override': variant_specialist_override,
        'soft_blend_disagreement': variant_soft_blend_uncertainty,
    }

    print('\nComputing anomaly scores...')
    anom_train, anom_val = _anomaly_scores(train, val)

    train_oofs = {ep: {} for ep in ('hepatic', 'death')}
    val_preds = {ep: {} for ep in ('hepatic', 'death')}

    for endpoint in ('hepatic', 'death'):
        print(f'\n--- {endpoint} ---')
        recs = _collect(pool_dirs, endpoint)
        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        print(f'  pool={len(recs)} candidates')
        for vname, fn in VARIANT_FNS.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = fn(recs, y, valid, 10, anom_train, anom_val)
                if r is None:
                    print(f'    {vname}: returned None')
                    continue
                oof = r['oof']
                vp = r['val']
                train_oofs[endpoint][vname] = oof
                val_preds[endpoint][vname] = vp
                cval = c_index(y, valid, oof)
                print(f'    {vname:30s} OOF C={cval:.4f}  size={r.get("size", "?")}')
                np.save(FULL / f'proxy_train_oof_{vname}_{endpoint}.npy', oof)
            except Exception as e:
                print(f'    {vname}: FAILED {type(e).__name__}: {e}')

        # Meta-aware variants from refuse_meta_aware (we replay the
        # arithmetic inline — they share dependencies on var_oof).
        try:
            _replay_meta_aware(
                endpoint, recs, y, valid, anom_train, anom_val,
                train_oofs[endpoint], val_preds[endpoint],
                train, val, n_train, n_val, pool_dirs,
            )
        except Exception as e:
            print(f'  meta-aware replay FAILED: {type(e).__name__}: {e}')

    # ------------------------------------------------------------------
    # Step 2: NNLS-based reconstruction for meta_avg_* family.
    # ------------------------------------------------------------------
    print('\n--- META_VARIANTS reconstruction via NNLS-on-val ---')
    for endpoint in ('hepatic', 'death'):
        col = 'risk_hepatic_event' if endpoint == 'hepatic' else 'risk_death'
        # Base variant val matrix.
        base_names = list(val_preds[endpoint].keys())
        if not base_names:
            print(f'  {endpoint}: no base variants; skipping')
            continue
        V_base = np.stack([rank01(val_preds[endpoint][v]) for v in base_names], axis=1)
        # Impute NaN with column median.
        col_med = np.nanmedian(V_base, axis=0)
        nan_idx = np.where(np.isnan(V_base))
        V_base[nan_idx] = np.take(col_med, nan_idx[1])
        T_base = np.stack([rank01(train_oofs[endpoint][v]) for v in base_names], axis=1)
        col_med_t = np.nanmedian(T_base, axis=0)
        nan_idx_t = np.where(np.isnan(T_base))
        T_base[nan_idx_t] = np.take(col_med_t, nan_idx_t[1])

        for mv in META_VARIANTS:
            csv_p = FULL / f'sub_master_{mv}.csv'
            if not csv_p.exists():
                continue
            try:
                df = pd.read_csv(csv_p)
                if col not in df.columns:
                    continue
                target_val = rank01(df[col].values)
                target_val_filled = np.where(
                    np.isnan(target_val),
                    np.nanmedian(target_val),
                    target_val,
                )
                # NNLS: target_val ≈ V_base @ w
                w, _ = nnls(V_base, target_val_filled)
                if w.sum() <= 0:
                    # uniform fallback.
                    w = np.ones(len(base_names)) / len(base_names)
                else:
                    w = w / w.sum()
                # Apply same weights to training-side ranks.
                train_pred = T_base @ w
                np.save(FULL / f'proxy_train_oof_{mv}_{endpoint}.npy', train_pred)
                # Val-side reconstruction quality.
                val_pred_recon = V_base @ w
                rho_recon, _ = spearmanr(val_pred_recon, target_val_filled)
                # Capture the implied weights for diagnostics.
                top_w_idx = np.argsort(-w)[:3]
                top_w_str = ', '.join(
                    f'{base_names[i]}={w[i]:.2f}' for i in top_w_idx
                )
                print(f'  [{endpoint}] {mv:40s} val_recon_rho={rho_recon:.4f}  '
                      f'top: {top_w_str}')
            except Exception as e:
                print(f'  [{endpoint}] {mv}: FAILED {type(e).__name__}: {e}')

    # ------------------------------------------------------------------
    # Step 3: per-variant true OOF C-index.
    # ------------------------------------------------------------------
    print('\n--- per-variant TRUE OOF C-index ---')
    rows = []
    all_variants = sorted(set(BASE_VARIANTS + META_VARIANTS))
    for v in all_variants:
        h_p = FULL / f'proxy_train_oof_{v}_hepatic.npy'
        d_p = FULL / f'proxy_train_oof_{v}_death.npy'
        if not h_p.exists() or not d_p.exists():
            continue
        h_arr = np.load(h_p)
        d_arr = np.load(d_p)
        h_c = c_index(train['hepatic']['y_struct'],
                      train['hepatic']['valid_mask'], h_arr)
        d_c = c_index(train['death']['y_struct'],
                      train['death']['valid_mask'], d_arr)
        lb_proxy = 0.7 * h_c + 0.3 * d_c
        rows.append({'variant': v, 'hep_C': h_c, 'death_C': d_c,
                     'weighted_LB': lb_proxy})

    rows.sort(key=lambda r: -r['weighted_LB'])
    out_df = pd.DataFrame(rows)
    out_p = FULL / 'per_variant_true_oof_C.csv'
    out_df.to_csv(out_p, index=False)
    print(f'\nwrote {out_p}')
    print('\nTop 15 by weighted_LB:')
    for r in rows[:15]:
        print(f'  {r["variant"]:40s} hep={r["hep_C"]:.4f} death={r["death_C"]:.4f} '
              f'LB={r["weighted_LB"]:.4f}')

    # ------------------------------------------------------------------
    # Step 4: cross-check against leaderboard_history.
    # ------------------------------------------------------------------
    hist = json.loads((REPO / 'tests/experiments/annitia/leaderboard_history.json').read_text())
    lb_map = {s['stem'].replace('sub_master_', ''): s['score'] for s in hist['submissions']}
    print('\nLeaderboard cross-check:')
    rho_pairs = []
    for r in rows:
        if r['variant'] in lb_map:
            print(f'  {r["variant"]:40s} TRUE_LB_proxy={r["weighted_LB"]:.4f}  REAL_LB={lb_map[r["variant"]]:.4f}  '
                  f'gap={r["weighted_LB"] - lb_map[r["variant"]]:+.4f}')
            rho_pairs.append((r['weighted_LB'], lb_map[r['variant']]))
    if len(rho_pairs) >= 3:
        x = np.array([p[0] for p in rho_pairs]); yv = np.array([p[1] for p in rho_pairs])
        rho, _ = spearmanr(x, yv)
        print(f'\nSpearman rho(true_oof_LB, real_LB) = {rho:.4f}  (vs proxy ~0 from collapse)')


def _replay_meta_aware(endpoint, recs, y, valid, anom_train, anom_val,
                       train_oofs_ep, val_preds_ep,
                       train, val, n_train, n_val, pool_dirs):
    """Inline replay of refuse_meta_aware.py for a single endpoint.

    Mirrors the bucket-routing / weighted-blend / cluster-routing
    arithmetic, populating train_oofs_ep[name] for each meta-meta variant.
    """
    from experiments.annitia.refuse_variants import _per_patient_disc
    from experiments.annitia.refuse_meta_aware import _ensemble_disagreement
    print(f'  meta-aware replay for {endpoint}...')

    # Build var_oof / var_disc from already-computed base variants.
    var_oof = {v: train_oofs_ep[v] for v in [
        'rank_mean', 'rank_median', 'rank_trimmed', 'residual_greedy',
        'uncertainty_route', 'anomaly_blend', 'disagreement_route',
        'cluster_route', 'specialist_override', 'soft_blend_disagreement',
    ] if v in train_oofs_ep}
    var_disc = {v: _per_patient_disc(y['time'], y['event'], oof)
                for v, oof in var_oof.items()}

    a = anom_train; keep = valid & np.isfinite(a)
    if keep.sum() < 50: return
    edges = np.nanquantile(a[keep], np.linspace(0, 1, 5))
    edges[0] = -np.inf; edges[-1] = np.inf

    def _best_variant_on_mask(mask):
        best_score, best_v = -np.inf, list(var_oof.keys())[0]
        for vn, dd in var_disc.items():
            sub = dd[mask]; sub = sub[np.isfinite(sub)]
            if len(sub) < 5: continue
            s = 1.0 - np.mean(sub)
            if s > best_score: best_score, best_v = s, vn
        return best_v

    # ---- anomaly_router_meta ----
    bucket_oof = np.full(n_train, np.nan)
    for b in range(4):
        bm = keep & (a >= edges[b]) & (a < edges[b + 1])
        if bm.sum() < 20:
            vbest = list(var_oof.keys())[0]
        else:
            vbest = _best_variant_on_mask(bm)
        bucket_oof[bm] = rank01(var_oof[vbest])[bm]
    train_oofs_ep['anomaly_router_meta'] = bucket_oof
    np.save(FULL / f'proxy_train_oof_anomaly_router_meta_{endpoint}.npy', bucket_oof)

    # ---- disagreement_router_meta ----
    disag_train = _ensemble_disagreement(pool_dirs, endpoint, n_train)
    keep_u = valid & np.isfinite(disag_train)
    if keep_u.sum() >= 100:
        edges_u = np.nanquantile(disag_train[keep_u], np.linspace(0, 1, 5))
        edges_u[0] = -np.inf; edges_u[-1] = np.inf
        d_oof = np.full(n_train, np.nan)
        for b in range(4):
            bm = keep_u & (disag_train >= edges_u[b]) & (disag_train < edges_u[b + 1])
            if bm.sum() < 20:
                vbest = list(var_oof.keys())[0]
            else:
                vbest = _best_variant_on_mask(bm)
            d_oof[bm] = rank01(var_oof[vbest])[bm]
        train_oofs_ep['disagreement_router_meta'] = d_oof
        np.save(FULL / f'proxy_train_oof_disagreement_router_meta_{endpoint}.npy', d_oof)

    # ---- cluster_router_meta ----
    from sklearn.cluster import KMeans
    from experiments.annitia.views import build_views
    views_t = build_views(train, view_names=['snapshot', 'clinical_pro'])
    Xt = np.concatenate([views_t[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    Xt = np.nan_to_num(Xt, nan=0.0)
    km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xt)
    cl_train = km.labels_
    c_oof = np.full(n_train, np.nan)
    for c in range(5):
        cm = valid & (cl_train == c)
        if cm.sum() < 20:
            vbest = list(var_oof.keys())[0]
        else:
            vbest = _best_variant_on_mask(cm)
        c_oof[cl_train == c] = rank01(var_oof[vbest])[cl_train == c]
    train_oofs_ep['cluster_router_meta'] = c_oof
    np.save(FULL / f'proxy_train_oof_cluster_router_meta_{endpoint}.npy', c_oof)

    # ---- anomaly_blend_meta ----
    med = np.nanmedian(a[keep])
    low_mask = keep & (a < med); high_mask = keep & (a >= med)
    v_low = _best_variant_on_mask(low_mask)
    v_high = _best_variant_on_mask(high_mask)
    spread = max(np.nanstd(a[keep]), 1e-3)
    w_t = 1.0 / (1.0 + np.exp(-(a - med) / spread))
    ab_oof = (1 - w_t) * rank01(var_oof[v_low]) + w_t * rank01(var_oof[v_high])
    train_oofs_ep['anomaly_blend_meta'] = ab_oof
    np.save(FULL / f'proxy_train_oof_anomaly_blend_meta_{endpoint}.npy', ab_oof)

    # ---- weighted_anomaly_per_patient ----
    bucket_weights = []
    for b in range(4):
        bm = keep & (a >= edges[b]) & (a < edges[b + 1])
        if bm.sum() < 20:
            bucket_weights.append(np.ones(len(var_oof)) / len(var_oof)); continue
        disc_per_v = []
        for vn in var_oof.keys():
            sub = var_disc[vn][bm]; sub = sub[np.isfinite(sub)]
            disc_per_v.append(np.mean(sub) if len(sub) else 0.5)
        scores_v = -np.array(disc_per_v) * 10
        ex = np.exp(scores_v - scores_v.max())
        bucket_weights.append(ex / ex.sum())
    wp_oof = np.full(n_train, np.nan)
    for b in range(4):
        bm = (a >= edges[b]) & (a < edges[b + 1])
        mix = sum(bucket_weights[b][i] * rank01(var_oof[vn])
                  for i, vn in enumerate(var_oof.keys()))
        wp_oof[bm] = mix[bm]
    train_oofs_ep['weighted_anomaly_per_patient'] = wp_oof
    np.save(FULL / f'proxy_train_oof_weighted_anomaly_per_patient_{endpoint}.npy', wp_oof)


if __name__ == '__main__':
    main()
