"""Patient subgroup discovery + OOD score.

Builds two new representations to break the variant-similarity ceiling:

A. **Subgroup IDs**: GMM clustering on PCA-reduced features with BIC-selected
   k. Each patient gets a cluster assignment. Per-cluster Cox PH gives
   cluster-conditional survival predictions — qualitatively different from
   the existing pool because base models pool all patients into one fit.

B. **OOD score**: Mahalanobis distance from training centroid + per-feature
   z-score + isolation-forest score. Tells us which val_test patients are
   extrapolation-zone (where any model is suspect).

Outputs:
  tests/results/annitia/full/patient_representation_train.csv
  tests/results/annitia/full/patient_representation_val.csv
  tests/results/annitia/full/sub_master_subgroup_routed.csv  (new submission)
  tests/results/annitia/full/patient_representation.md       (summary)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia          # noqa: E402
from experiments.annitia.views import build_views              # noqa: E402
from experiments.annitia.survival import concordance           # noqa: E402

FULL = REPO / 'tests/results/annitia/full'


def features_matrix(d):
    views = build_views(d, view_names=['snapshot', 'clinical_pro'])
    X = np.concatenate([views[v][0] for v in ['snapshot', 'clinical_pro']], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def select_k_via_bic(X, k_min=2, k_max=8, seed=42):
    bics = {}
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, random_state=seed, n_init=3,
                              covariance_type='diag').fit(X)
        bics[k] = gmm.bic(X)
    best = min(bics, key=bics.get)
    return best, bics


def fit_per_cluster_cox(X_train, y_struct_train, valid_train, cluster_train,
                       X_val, cluster_val):
    """For each cluster, fit a Cox PH on training rows in that cluster, predict
    on val rows in that cluster. For clusters with too few events, fall back
    to global Cox."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    n_val = X_val.shape[0]
    val_pred = np.full(n_val, np.nan)

    # Global fallback fit
    finite_train = valid_train & np.isfinite(y_struct_train['time'])
    try:
        global_cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(
            X_train[finite_train], y_struct_train[finite_train])
        global_pred_val = global_cox.predict(X_val)
    except Exception as e:
        print(f'  global Cox failed: {e}')
        global_pred_val = np.zeros(n_val)

    n_clusters = max(cluster_train.max(), cluster_val.max()) + 1
    cluster_stats = {}
    for c in range(n_clusters):
        train_mask = (cluster_train == c) & finite_train
        val_mask = (cluster_val == c)
        n_train = train_mask.sum()
        n_events = int(y_struct_train['event'][train_mask].sum()) if n_train else 0
        n_val_c = int(val_mask.sum())
        cluster_stats[int(c)] = {'n_train': int(n_train), 'n_events': int(n_events),
                                 'n_val': n_val_c}
        if n_events < 5 or n_train < 30:
            # Fallback to global
            val_pred[val_mask] = global_pred_val[val_mask]
            cluster_stats[int(c)]['fitted'] = False
            continue
        try:
            cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(
                X_train[train_mask], y_struct_train[train_mask])
            val_pred[val_mask] = cox.predict(X_val[val_mask])
            cluster_stats[int(c)]['fitted'] = True
        except Exception as e:
            val_pred[val_mask] = global_pred_val[val_mask]
            cluster_stats[int(c)]['fitted'] = False

    return val_pred, cluster_stats, global_pred_val


def per_cluster_cox_oof(X_train, y_struct_train, valid_train, cluster_train, seed=42):
    """5-fold OOF prediction using cluster-routed Cox. For honest C-index."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sklearn.model_selection import KFold

    n = X_train.shape[0]
    oof = np.full(n, np.nan)
    finite = valid_train & np.isfinite(y_struct_train['time'])
    valid_idx = np.where(finite)[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for tr, te in kf.split(valid_idx):
        train_rows = valid_idx[tr]
        test_rows = valid_idx[te]
        # Global fallback for this fold
        try:
            global_cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(
                X_train[train_rows], y_struct_train[train_rows])
            global_pred = global_cox.predict(X_train[test_rows])
        except Exception:
            global_pred = np.zeros(len(test_rows))

        # Per-cluster
        for c in np.unique(cluster_train):
            tr_c = train_rows[cluster_train[train_rows] == c]
            te_c_mask = cluster_train[test_rows] == c
            te_c_rows = test_rows[te_c_mask]
            if len(te_c_rows) == 0:
                continue
            n_events_c = int(y_struct_train['event'][tr_c].sum()) if len(tr_c) else 0
            if n_events_c < 5 or len(tr_c) < 30:
                # Fallback: use global prediction
                oof[te_c_rows] = global_pred[te_c_mask]
                continue
            try:
                cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(
                    X_train[tr_c], y_struct_train[tr_c])
                oof[te_c_rows] = cox.predict(X_train[te_c_rows])
            except Exception:
                oof[te_c_rows] = global_pred[te_c_mask]
    return oof


def main():
    train = load_annitia()
    val_path = '/Volumes/seagate/kirby/annitia/val_test.csv'
    val = load_annitia(data_dir=str(Path(val_path).parent),
                       filename=Path(val_path).name, has_outcomes=False)

    Xt = features_matrix(train)
    Xv = features_matrix(val)
    print(f'train features: {Xt.shape}, val features: {Xv.shape}')

    # Standardise + PCA (95% variance, capped at 30 dims)
    scaler = StandardScaler().fit(Xt)
    Xt_s = scaler.transform(Xt); Xv_s = scaler.transform(Xv)
    pca = PCA(n_components=min(30, Xt_s.shape[1])).fit(Xt_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pc = max(2, int(np.searchsorted(cumvar, 0.95) + 1))
    print(f'PCA: {n_pc} components for 95% variance (cumvar={cumvar[n_pc-1]:.3f})')
    pca = PCA(n_components=n_pc).fit(Xt_s)
    Xt_pc = pca.transform(Xt_s); Xv_pc = pca.transform(Xv_s)

    # === A. Subgroup discovery via GMM with BIC ===
    print('\n--- A. Subgroup discovery (GMM + BIC) ---')
    best_k, bics = select_k_via_bic(Xt_pc, k_min=2, k_max=6)
    print(f'BIC-selected k = {best_k}')
    print('  BIC by k:', {k: f'{v:.1f}' for k, v in bics.items()})
    gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=5,
                          covariance_type='diag').fit(Xt_pc)
    cluster_train = gmm.predict(Xt_pc)
    cluster_val = gmm.predict(Xv_pc)

    # Distribution
    print('\nCluster sizes:')
    for c in range(best_k):
        n_t = (cluster_train == c).sum()
        n_v = (cluster_val == c).sum()
        events_h = int(train['hepatic']['event'][cluster_train == c].sum())
        events_d = int(train['death']['event'][cluster_train == c].sum())
        print(f'  cluster {c}: train n={n_t} (hep_evt={events_h}, dth_evt={events_d}), val n={n_v}')

    # === B. OOD scores ===
    print('\n--- B. OOD scores ---')
    # Mahalanobis distance from training centroid
    cov = EmpiricalCovariance().fit(Xt_pc)
    mah_train = cov.mahalanobis(Xt_pc)
    mah_val = cov.mahalanobis(Xv_pc)
    train_p99 = np.percentile(mah_train, 99)
    print(f'Mahalanobis: train p99={train_p99:.2f}, val p99={np.percentile(mah_val, 99):.2f}')
    print(f'  val patients beyond train p99: {(mah_val > train_p99).sum()}/{len(mah_val)}')

    # Isolation forest
    iso = IsolationForest(n_estimators=300, random_state=42, contamination='auto').fit(Xt_pc)
    iso_train = -iso.score_samples(Xt_pc)
    iso_val = -iso.score_samples(Xv_pc)

    # Composite OOD: average rank across mahalanobis and iso
    def rank01(x): return rankdata(x) / len(x)
    ood_train = (rank01(mah_train) + rank01(iso_train)) / 2
    ood_val = (rank01(mah_val) + rank01(iso_val)) / 2

    # Save per-patient representations
    train_repr = pd.DataFrame({
        'patient_id': train['patient_ids'],
        'cluster': cluster_train,
        'mahalanobis': mah_train,
        'iso_score': iso_train,
        'ood_composite': ood_train,
    })
    val_repr = pd.DataFrame({
        'trustii_id': val['patient_ids'],
        'cluster': cluster_val,
        'mahalanobis': mah_val,
        'iso_score': iso_val,
        'ood_composite': ood_val,
    })
    train_repr.to_csv(FULL / 'patient_representation_train.csv', index=False)
    val_repr.to_csv(FULL / 'patient_representation_val.csv', index=False)
    print(f'\nwrote {FULL}/patient_representation_train.csv')
    print(f'wrote {FULL}/patient_representation_val.csv')

    # === Cluster-routed survival predictions ===
    print('\n--- Cluster-routed survival predictions ---')
    h_pred, h_stats, h_global = fit_per_cluster_cox(
        Xt_pc, train['hepatic']['y_struct'], train['hepatic']['valid_mask'],
        cluster_train, Xv_pc, cluster_val)
    d_pred, d_stats, d_global = fit_per_cluster_cox(
        Xt_pc, train['death']['y_struct'], train['death']['valid_mask'],
        cluster_train, Xv_pc, cluster_val)

    # OOF C of cluster-routed Cox (for sanity)
    print('\nOOF C of cluster-routed Cox PH:')
    h_oof = per_cluster_cox_oof(Xt_pc, train['hepatic']['y_struct'],
                                 train['hepatic']['valid_mask'], cluster_train)
    h_finite = train['hepatic']['valid_mask'] & np.isfinite(h_oof)
    h_oof_c = float(concordance(train['hepatic']['y_struct'][h_finite], h_oof[h_finite])) if h_finite.sum() >= 5 else float('nan')
    print(f'  hepatic cluster-routed OOF C = {h_oof_c:.4f}')

    d_oof = per_cluster_cox_oof(Xt_pc, train['death']['y_struct'],
                                 train['death']['valid_mask'], cluster_train)
    d_finite = train['death']['valid_mask'] & np.isfinite(d_oof)
    d_oof_c = float(concordance(train['death']['y_struct'][d_finite], d_oof[d_finite])) if d_finite.sum() >= 5 else float('nan')
    print(f'  death   cluster-routed OOF C = {d_oof_c:.4f}')

    # Comparison: global Cox OOF C
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sklearn.model_selection import KFold
    def global_cox_oof(X, y_struct, valid):
        n = X.shape[0]; oof = np.full(n, np.nan)
        finite = valid & np.isfinite(y_struct['time'])
        idx = np.where(finite)[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in kf.split(idx):
            try:
                cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(
                    X[idx[tr]], y_struct[idx[tr]])
                oof[idx[te]] = cox.predict(X[idx[te]])
            except Exception:
                continue
        return oof
    g_h_oof = global_cox_oof(Xt_pc, train['hepatic']['y_struct'],
                              train['hepatic']['valid_mask'])
    g_h_finite = train['hepatic']['valid_mask'] & np.isfinite(g_h_oof)
    g_h_c = float(concordance(train['hepatic']['y_struct'][g_h_finite], g_h_oof[g_h_finite])) if g_h_finite.sum() >= 5 else float('nan')
    g_d_oof = global_cox_oof(Xt_pc, train['death']['y_struct'],
                              train['death']['valid_mask'])
    g_d_finite = train['death']['valid_mask'] & np.isfinite(g_d_oof)
    g_d_c = float(concordance(train['death']['y_struct'][g_d_finite], g_d_oof[g_d_finite])) if g_d_finite.sum() >= 5 else float('nan')
    print(f'  vs GLOBAL Cox OOF C: hep={g_h_c:.4f}, death={g_d_c:.4f}')

    # Build submission CSV from cluster-routed predictions
    trustii_id = val['patient_ids']
    out = pd.DataFrame({
        'trustii_id': trustii_id,
        'risk_hepatic_event': h_pred,
        'risk_death': d_pred,
    })
    dest = FULL / 'sub_master_subgroup_routed.csv'
    out.to_csv(dest, index=False)
    print(f'\nwrote {dest}')

    # Sanity vs anchor
    anc_path = FULL / 'sub_master_meta_avg_aware_weighted.csv'
    if anc_path.exists():
        from scipy.stats import spearmanr
        anc = pd.read_csv(anc_path)
        rho_h, _ = spearmanr(out['risk_hepatic_event'], anc['risk_hepatic_event'])
        rho_d, _ = spearmanr(out['risk_death'], anc['risk_death'])
        print(f'  spearman vs 0.9302 anchor: hep={rho_h:.4f} death={rho_d:.4f}')

    # Cross-reference: are contested val_test patients in specific clusters?
    print('\n--- Cluster distribution of contested val_test patients ---')
    cont_h = pd.read_csv(FULL / 'valtest_contention_hepatic.csv')
    cont_h_top20 = cont_h.head(20).merge(val_repr, on='trustii_id', how='left')
    print('  Top 20 contested hep, cluster distribution:')
    for c in range(best_k):
        n = (cont_h_top20['cluster'] == c).sum()
        print(f'    cluster {c}: {n}/20')

    cont_d = pd.read_csv(FULL / 'valtest_contention_death.csv')
    cont_d_top20 = cont_d.head(20).merge(val_repr, on='trustii_id', how='left')
    print('  Top 20 contested death, cluster distribution:')
    for c in range(best_k):
        n = (cont_d_top20['cluster'] == c).sum()
        print(f'    cluster {c}: {n}/20')

    # OOD overlap with contested
    print('\n--- OOD score correlation with contention ---')
    from scipy.stats import spearmanr
    ch_merge = cont_h.merge(val_repr, on='trustii_id', how='left')
    rho_oh, _ = spearmanr(ch_merge['contention'], ch_merge['ood_composite'])
    print(f'  hepatic contention ↔ OOD: ρ={rho_oh:.4f}')
    cd_merge = cont_d.merge(val_repr, on='trustii_id', how='left')
    rho_od, _ = spearmanr(cd_merge['contention'], cd_merge['ood_composite'])
    print(f'  death contention ↔ OOD: ρ={rho_od:.4f}')

    # Summary
    summary = {
        'pca_components': int(n_pc),
        'best_k_bic': int(best_k),
        'bic_by_k': {int(k): float(v) for k, v in bics.items()},
        'cluster_stats_hep': h_stats,
        'cluster_stats_death': d_stats,
        'cluster_routed_oof_c': {'hepatic': h_oof_c, 'death': d_oof_c},
        'global_cox_oof_c': {'hepatic': g_h_c, 'death': g_d_c},
        'mahalanobis_train_p99': float(train_p99),
        'val_patients_beyond_train_p99': int((mah_val > train_p99).sum()),
        'contention_vs_ood_spearman': {'hepatic': float(rho_oh), 'death': float(rho_od)},
    }
    (FULL / 'patient_representation.json').write_text(json.dumps(summary, indent=2))
    print(f'\nwrote {FULL}/patient_representation.json')


if __name__ == '__main__':
    main()
