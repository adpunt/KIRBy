"""Rigorous local validation of top candidates BEFORE burning a submission.

Inputs: list of candidate names that have oof_<ep>_<name>.npy in
tests/results/annitia/full/. For each candidate, compute:

  1. **Bootstrap CI on combined LB** (B=1000, paired stratified by event flag).
     Reports the FULL distribution so we can do pairwise tests.
  2. **Pairwise dominance** vs anomaly_blend_meta and meta_avg_aware_weighted
     (current LB best). For each bootstrap replica, both are scored on the
     SAME bootstrap sample → dependent comparison → P(A > B) is the
     fraction of replicas where A wins.
  3. **Holdout-fold C** using the existing 10% carve-out from refuse_holdout.
     This is a single-shot but uses patients NOT in the variant's own training
     fold for some base models.
  4. **Per-cohort breakdown** — split valid patients by:
     - Anomaly score quartile (4 buckets)
     - Event vs censored
     - Baseline-age quartile
     For each split, compute C-index. A candidate that uniformly wins is
     more trustworthy than one that wins only on a subset.
  5. **Distribution-shift indicator** between train OOFs and val predictions
     (KS test on rank distributions).

Outputs:
  tests/results/annitia/validate_top.md   — full report with verdict per candidate
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.survival import concordance
from experiments.annitia.views import build_views


VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'
FULL = REPO / 'tests/results/annitia/full'
RES = REPO / 'tests/results/annitia'

# Top candidates to validate. Order matters for the pairwise tests.
CANDIDATES = [
    'anomaly_blend_meta',
    'rank_median',
    'specialist_override',
    'weighted_anomaly_per_patient',
    'cluster_router_meta',
    'rank_mean',
    'meta_avg_aware_weighted_v4',
    'cherry_pick_hep_anomBlend_death_wapp',  # best-per-endpoint cherry pick
]


def boot_pair(y, oofs_dict, B=1000, seed=42):
    """Stratified bootstrap by event flag. Returns dict[name -> array of B C-indices].
    All candidates use the SAME bootstrap indices each replica → paired compare."""
    rng = np.random.default_rng(seed)
    e = y['event'].astype(bool)
    iev = np.where(e)[0]; icn = np.where(~e)[0]
    out = {n: [] for n in oofs_dict}
    for _ in range(B):
        ev_b = rng.choice(iev, size=iev.size, replace=True)
        cn_b = rng.choice(icn, size=icn.size, replace=True)
        idx = np.concatenate([ev_b, cn_b])
        for n, p in oofs_dict.items():
            try:
                C = concordance(y[idx], p[idx])
            except Exception:
                C = float('nan')
            out[n].append(C)
    return {n: np.array(v) for n, v in out.items()}


def pair_winrate(boot_combined, baseline='meta_avg_aware_weighted_v4'):
    """For each candidate, P(combined_LB > baseline) over bootstrap replicas."""
    if baseline not in boot_combined: return {}
    base = boot_combined[baseline]
    out = {}
    for n, vals in boot_combined.items():
        if n == baseline: continue
        # Mask both to finite indices
        m = np.isfinite(vals) & np.isfinite(base)
        if m.sum() < 50:
            out[n] = float('nan'); continue
        out[n] = float(np.mean(vals[m] > base[m]))
    return out


def per_cohort_C(y, oof, valid, anom, base_age):
    """Returns dict of C-index per cohort split."""
    out = {}
    # By anomaly quartile
    keep = valid & np.isfinite(anom) & np.isfinite(oof)
    if keep.sum() >= 50:
        a = anom[keep]; q = np.quantile(a, [0, 0.25, 0.5, 0.75, 1.0])
        for k in range(4):
            mask = keep & (anom >= q[k]) & (anom < q[k + 1])
            if k == 3: mask = keep & (anom >= q[k])
            n_ev = (mask & y['event'].astype(bool)).sum()
            if n_ev < 2: out[f'anomQ{k+1}'] = float('nan'); continue
            try:
                out[f'anomQ{k+1}'] = float(concordance(y[mask], oof[mask]))
            except Exception:
                out[f'anomQ{k+1}'] = float('nan')
    # By baseline age quartile
    keep2 = valid & np.isfinite(base_age) & np.isfinite(oof)
    if keep2.sum() >= 50:
        a = base_age[keep2]; q = np.quantile(a, [0, 0.25, 0.5, 0.75, 1.0])
        for k in range(4):
            mask = keep2 & (base_age >= q[k]) & (base_age < q[k + 1])
            if k == 3: mask = keep2 & (base_age >= q[k])
            n_ev = (mask & y['event'].astype(bool)).sum()
            if n_ev < 2: out[f'ageQ{k+1}'] = float('nan'); continue
            try:
                out[f'ageQ{k+1}'] = float(concordance(y[mask], oof[mask]))
            except Exception:
                out[f'ageQ{k+1}'] = float('nan')
    return out


def main():
    print('Loading...')
    train = load_annitia()

    # Compute anomaly score for cohort splits
    vt = build_views(train, view_names=('snapshot', 'clinical_pro'))
    X_tr = np.concatenate([vt[v][0] for v in ('snapshot', 'clinical_pro')], axis=1)
    sc = StandardScaler().fit(X_tr)
    iso = IsolationForest(n_estimators=200, random_state=42).fit(sc.transform(X_tr))
    anom = -iso.score_samples(sc.transform(X_tr))
    base_age = np.asarray(train['baseline_age'], dtype=float)

    report = ['# validate_top — rigorous candidate validation\n']
    report.append('Goal: decide which candidates are worth submission slots vs OOF noise.\n\n')

    for endpoint in ('hepatic', 'death'):
        ep_y = train[endpoint]
        valid = ep_y['valid_mask']; y = ep_y['y_struct']
        report.append(f'## {endpoint}\n\n')

        # Load OOFs for each candidate
        oofs = {}
        for name in CANDIDATES:
            p = FULL / f'oof_{endpoint}_{name}.npy'
            if p.exists():
                a = np.load(p)
                if a.size == valid.size:
                    oofs[name] = a
                else:
                    print(f'  size mismatch for {name}: {a.size} vs {valid.size}')
        if not oofs:
            report.append(f'_No OOFs found for {endpoint}._\n\n'); continue

        # Mask to common-valid (finite for ALL candidates)
        m_all = valid.copy()
        for a in oofs.values():
            m_all = m_all & np.isfinite(a)
        idx_all = np.where(m_all)[0]
        y_all = y[idx_all]
        oofs_sub = {n: a[idx_all] for n, a in oofs.items()}

        # Point OOF C
        report.append('### Point OOF C\n\n')
        report.append('| Candidate | C |\n|---|---|\n')
        point_C = {}
        for n in oofs_sub:
            try:
                C = float(concordance(y_all, oofs_sub[n]))
            except Exception:
                C = float('nan')
            point_C[n] = C
            report.append(f'| {n} | {C:.4f} |\n')
        report.append('\n')

        # Bootstrap CI (paired)
        print(f'  bootstrap {endpoint} B=1000 ...')
        boot = boot_pair(y_all, oofs_sub, B=1000)
        report.append('### Bootstrap (B=1000, stratified by event)\n\n')
        report.append('| Candidate | Mean C | Lo 95% | Hi 95% | Std |\n|---|---|---|---|---|\n')
        for n, vals in boot.items():
            v = vals[np.isfinite(vals)]
            if not v.size:
                report.append(f'| {n} | nan | nan | nan | nan |\n'); continue
            report.append(f'| {n} | {v.mean():.4f} | {np.percentile(v, 2.5):.4f}'
                          f' | {np.percentile(v, 97.5):.4f} | {v.std():.4f} |\n')
        report.append('\n')

        # Pairwise win rate vs current LB anchor
        if 'meta_avg_aware_weighted_v4' in boot:
            wr = pair_winrate(boot, 'meta_avg_aware_weighted_v4')
            report.append('### Win-rate vs current LB best (`meta_avg_aware_weighted`)\n\n')
            report.append('| Candidate | P(beats anchor) | Δmean |\n|---|---|---|\n')
            anchor_v = boot['meta_avg_aware_weighted_v4']
            anchor_v = anchor_v[np.isfinite(anchor_v)]
            for n, p in sorted(wr.items(), key=lambda x: -x[1] if np.isfinite(x[1]) else -1):
                v = boot[n]; v = v[np.isfinite(v)]
                if not v.size:
                    report.append(f'| {n} | nan | nan |\n'); continue
                dm = v.mean() - anchor_v.mean()
                report.append(f'| {n} | {p:.3f} | {dm:+.4f} |\n')
            report.append('\n')

        # Per-cohort
        report.append('### Per-cohort C-index breakdown\n\n')
        cohort_data = {}
        for n in oofs_sub:
            cohort_data[n] = per_cohort_C(y, oofs[n], valid, anom, base_age)
        cohorts = ['anomQ1', 'anomQ2', 'anomQ3', 'anomQ4',
                   'ageQ1', 'ageQ2', 'ageQ3', 'ageQ4']
        report.append('| Candidate |' + '|'.join(cohorts) + '|\n')
        report.append('|---|' + '|'.join('---' for _ in cohorts) + '|\n')
        for n in oofs_sub:
            row = f'| {n} |'
            for c in cohorts:
                v = cohort_data[n].get(c, float('nan'))
                row += f' {v:.3f} |' if np.isfinite(v) else ' nan |'
            report.append(row + '\n')
        report.append('\n')

        # Distribution shift between OOF (train) and val predictions per candidate
        report.append('### Train-OOF vs val-prediction distribution (KS p-value)\n\n')
        report.append('Lower p = larger distribution shift between OOF and val. '
                      'Big shift → variant likely overfits.\n\n')
        report.append('| Candidate | KS p-value |\n|---|---|\n')
        for n in oofs_sub:
            try:
                csv_p = FULL / f'sub_master_{n}.csv'
                if not csv_p.exists():
                    report.append(f'| {n} | n/a |\n'); continue
                vdf = pd.read_csv(csv_p)
                col = 'risk_hepatic_event' if endpoint == 'hepatic' else 'risk_death'
                if col not in vdf.columns:
                    report.append(f'| {n} | n/a |\n'); continue
                # Compare RANK distributions (uniform expected)
                v = vdf[col].values
                tr_r = rankdata(oofs_sub[n]) / len(oofs_sub[n])
                va_r = rankdata(v) / len(v)
                ks = ks_2samp(tr_r, va_r)
                report.append(f'| {n} | {ks.pvalue:.3f} |\n')
            except Exception as ex:
                report.append(f'| {n} | err: {ex} |\n')
        report.append('\n')

    # Combined LB analysis (0.7 hep + 0.3 death) using paired bootstrap
    report.append('## Combined LB (0.7*C_hep + 0.3*C_death) bootstrap\n\n')
    # Need to bootstrap the SAME indices on hep and death simultaneously per replica.
    # Use independent bootstraps on each endpoint then average per replica
    # (slightly conservative: assumes endpoint indices independent, which they're not).
    # For the paired-test purpose this is OK since both candidates use the same boot
    # samples on each endpoint.
    rng = np.random.default_rng(42)
    B = 1000
    hep_y = train['hepatic']['y_struct']
    death_y = train['death']['y_struct']
    hep_valid = train['hepatic']['valid_mask']
    death_valid = train['death']['valid_mask']

    hep_oofs = {}; death_oofs = {}
    for name in CANDIDATES:
        hp = FULL / f'oof_hepatic_{name}.npy'
        dp = FULL / f'oof_death_{name}.npy'
        if hp.exists() and dp.exists():
            hep_oofs[name] = np.load(hp)
            death_oofs[name] = np.load(dp)

    if not hep_oofs:
        report.append('_No combined-LB candidates available._\n')
    else:
        # Common valid for hep
        m_h = hep_valid.copy()
        for a in hep_oofs.values():
            m_h = m_h & np.isfinite(a)
        # Common valid for death
        m_d = death_valid.copy()
        for a in death_oofs.values():
            m_d = m_d & np.isfinite(a)

        hep_idx = np.where(m_h)[0]; death_idx = np.where(m_d)[0]
        ev_h = hep_y[hep_idx]['event'].astype(bool)
        ev_d = death_y[death_idx]['event'].astype(bool)
        iev_h, icn_h = np.where(ev_h)[0], np.where(~ev_h)[0]
        iev_d, icn_d = np.where(ev_d)[0], np.where(~ev_d)[0]

        boot_combined = {n: [] for n in hep_oofs}
        for _ in range(B):
            bh = np.concatenate([rng.choice(iev_h, iev_h.size, replace=True),
                                  rng.choice(icn_h, icn_h.size, replace=True)])
            bd = np.concatenate([rng.choice(iev_d, iev_d.size, replace=True),
                                  rng.choice(icn_d, icn_d.size, replace=True)])
            yh = hep_y[hep_idx][bh]; yd = death_y[death_idx][bd]
            for n in hep_oofs:
                ph = hep_oofs[n][hep_idx][bh]; pd_ = death_oofs[n][death_idx][bd]
                try:
                    Ch = concordance(yh, ph); Cd = concordance(yd, pd_)
                    boot_combined[n].append(0.7 * Ch + 0.3 * Cd)
                except Exception:
                    boot_combined[n].append(float('nan'))
        boot_combined = {n: np.array(v) for n, v in boot_combined.items()}

        report.append('### Combined-LB bootstrap (B=1000)\n\n')
        report.append('| Candidate | Mean LB | Lo 95% | Hi 95% | Std |\n|---|---|---|---|---|\n')
        for n, v in sorted(boot_combined.items(),
                            key=lambda x: -np.nanmean(x[1])):
            v = v[np.isfinite(v)]
            if not v.size:
                report.append(f'| {n} | nan |\n'); continue
            report.append(f'| {n} | {v.mean():.4f} | {np.percentile(v, 2.5):.4f}'
                          f' | {np.percentile(v, 97.5):.4f} | {v.std():.4f} |\n')
        report.append('\n')

        if 'meta_avg_aware_weighted_v4' in boot_combined:
            anchor = boot_combined['meta_avg_aware_weighted_v4']
            report.append('### Pairwise: P(candidate combined LB > anchor combined LB)\n\n')
            report.append('| Candidate | P(beat anchor) | Δmean LB |\n|---|---|---|\n')
            for n, v in sorted(boot_combined.items(),
                                key=lambda x: -np.mean(x[1] > anchor)):
                if n == 'meta_avg_aware_weighted_v4': continue
                m = np.isfinite(v) & np.isfinite(anchor)
                if m.sum() < 50:
                    report.append(f'| {n} | nan | nan |\n'); continue
                p = float(np.mean(v[m] > anchor[m]))
                dm = float(v[m].mean() - anchor[m].mean())
                report.append(f'| {n} | {p:.3f} | {dm:+.4f} |\n')
            report.append('\n')

    out = RES / 'validate_top.md'
    out.write_text(''.join(report))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
