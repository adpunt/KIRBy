"""ANNITIA distribution-shift + per-cohort + per-horizon diagnostic.

Pure local diagnostic. NO submission CSVs are written.

Outputs:
  tests/results/annitia/full/distribution_shift.csv
  tests/results/annitia/full/distribution_shift_summary.json
  tests/results/annitia/full/per_cohort_C.csv
  tests/results/annitia/full/per_horizon_C.csv
  AUDIT_DISTRIBUTION_AND_COHORTS.md  (in repo root)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, LONGITUDINAL_VARS, NIT_VARS  # noqa
from experiments.annitia.survival import concordance  # noqa

BRANCHES = REPO / 'tests/results/annitia/branches'
FULL = REPO / 'tests/results/annitia/full'
FULL.mkdir(parents=True, exist_ok=True)
AUDIT_MD = REPO / 'AUDIT_DISTRIBUTION_AND_COHORTS.md'


# ---------- snapshot feature builder ----------
def baseline_snapshot(data: dict) -> pd.DataFrame:
    """First non-NaN value per longitudinal var + static features.

    Each row is a patient. Columns: static (gender,T2DM,...) + base_<var>
    for every longitudinal var (the value at v1, fall back to first
    finite visit if v1 is NaN).
    """
    n = len(data['patient_ids'])
    cols: dict[str, np.ndarray] = {}
    for s, arr in data['static'].items():
        cols[s] = arr.astype(float)
    cols['baseline_age'] = data['baseline_age']
    cols['last_followup_age'] = data['last_followup_age']
    cols['followup_years'] = data['last_followup_age'] - data['baseline_age']
    for var in LONGITUDINAL_VARS:
        if var == 'Age':
            continue
        wide = data['wide'][var]
        first = np.full(n, np.nan)
        for i in range(n):
            row = wide[i]
            finite = row[np.isfinite(row)]
            if finite.size:
                first[i] = float(finite[0])
        cols[f'base_{var}'] = first
    df = pd.DataFrame(cols)
    df['patient_id_anon'] = data['patient_ids']
    return df


# ---------- A. distribution shift ----------
def distribution_shift(train_df: pd.DataFrame, val_df: pd.DataFrame):
    feature_cols = [c for c in train_df.columns
                    if c not in ('patient_id_anon',) and pd.api.types.is_numeric_dtype(train_df[c])]
    rows = []
    for c in feature_cols:
        a = train_df[c].to_numpy(dtype=float)
        b = val_df[c].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 10:
            rows.append({'feature': c, 'n_train': len(a), 'n_val': len(b),
                         'ks_stat': np.nan, 'ks_p': np.nan,
                         'wasserstein': np.nan,
                         'mean_train': np.nan, 'mean_val': np.nan,
                         'mean_diff': np.nan, 'std_train': np.nan,
                         'std_val': np.nan, 'std_diff': np.nan,
                         'miss_train': np.nan, 'miss_val': np.nan})
            continue
        ks = stats.ks_2samp(a, b)
        # standardise before wasserstein for cross-feature comparability
        sd = float(np.std(np.concatenate([a, b])) or 1.0)
        w = float(stats.wasserstein_distance(a / sd, b / sd))
        rows.append({
            'feature': c,
            'n_train': len(a), 'n_val': len(b),
            'ks_stat': float(ks.statistic), 'ks_p': float(ks.pvalue),
            'wasserstein': w,
            'mean_train': float(np.mean(a)), 'mean_val': float(np.mean(b)),
            'mean_diff': float(np.mean(b) - np.mean(a)),
            'std_train': float(np.std(a)), 'std_val': float(np.std(b)),
            'std_diff': float(np.std(b) - np.std(a)),
            'miss_train': float(train_df[c].isna().mean()),
            'miss_val': float(val_df[c].isna().mean()),
        })
    out = pd.DataFrame(rows).sort_values('ks_stat', ascending=False, na_position='last')
    return out, feature_cols


def mahalanobis_distances(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]):
    Xt = train_df[cols].to_numpy(dtype=float)
    Xv = val_df[cols].to_numpy(dtype=float)
    keep = [c for ci, c in enumerate(cols) if np.isfinite(Xt[:, ci]).mean() > 0.5
            and np.isfinite(Xv[:, ci]).mean() > 0.5]
    Xt = train_df[keep].to_numpy(dtype=float)
    Xv = val_df[keep].to_numpy(dtype=float)
    means = np.nanmean(Xt, axis=0)
    Xt_imp = np.where(np.isfinite(Xt), Xt, means)
    Xv_imp = np.where(np.isfinite(Xv), Xv, means)
    cov = np.cov(Xt_imp.T) + 1e-3 * np.eye(Xt_imp.shape[1])
    inv = np.linalg.pinv(cov)
    d2_train = np.sum(((Xt_imp - means) @ inv) * (Xt_imp - means), axis=1)
    d2_val = np.sum(((Xv_imp - means) @ inv) * (Xv_imp - means), axis=1)
    return np.sqrt(np.maximum(d2_train, 0)), np.sqrt(np.maximum(d2_val, 0)), keep


def domain_classifier_auroc(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            cols: list[str], seed: int = 42):
    Xt = train_df[cols].to_numpy(dtype=float)
    Xv = val_df[cols].to_numpy(dtype=float)
    means = np.nanmean(np.vstack([Xt, Xv]), axis=0)
    Xt = np.where(np.isfinite(Xt), Xt, means)
    Xv = np.where(np.isfinite(Xv), Xv, means)
    X = np.vstack([Xt, Xv])
    y = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xv))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    auroc = float(roc_auc_score(y, oof))
    # Per-feature LR coefficient (fit on full data) to identify discriminating features
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)
    coefs = pd.Series(clf.coef_.ravel(), index=cols).sort_values(key=np.abs, ascending=False)
    return auroc, coefs


def optimal_transport_mass(train_df: pd.DataFrame, val_df: pd.DataFrame,
                           cols: list[str], n_max: int = 400):
    """Approximation: average min Mahalanobis distance from each val point
    to the nearest train point. Avoids POT dependency."""
    rng = np.random.default_rng(0)
    Xt = train_df[cols].to_numpy(dtype=float)
    Xv = val_df[cols].to_numpy(dtype=float)
    means = np.nanmean(Xt, axis=0)
    Xt = np.where(np.isfinite(Xt), Xt, means)
    Xv = np.where(np.isfinite(Xv), Xv, means)
    if len(Xt) > n_max:
        Xt = Xt[rng.choice(len(Xt), n_max, replace=False)]
    cov = np.cov(Xt.T) + 1e-3 * np.eye(Xt.shape[1])
    inv = np.linalg.pinv(cov)
    diffs = Xv[:, None, :] - Xt[None, :, :]
    d2 = np.einsum('ijk,kl,ijl->ij', diffs, inv, diffs)
    nn = np.sqrt(np.min(np.maximum(d2, 0), axis=1))
    return float(np.mean(nn)), float(np.median(nn)), float(np.percentile(nn, 90))


# ---------- B. cohort + variant scoring ----------
def define_cohorts(snap: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks per cohort definition. Indexed against snap rows."""
    n = len(snap)
    cohorts: dict[str, np.ndarray] = {}

    # Diabetes: T2DM == 1 OR fasting glucose >= 7 mmol/L (~126 mg/dL).
    # No HbA1c in this dataset; gluc_fast is the closest proxy.
    t2dm = (snap['T2DM'] == 1).fillna(False).to_numpy()
    gf = snap['base_gluc_fast'].to_numpy(dtype=float)
    diab_proxy = (gf >= 7.0)
    diab = t2dm | (np.isfinite(gf) & diab_proxy)
    cohorts['diabetes_yes'] = diab
    cohorts['diabetes_no'] = (~diab) & (np.isfinite(gf) | (snap['T2DM'].isin([0, 1])))

    # Sex: gender encoded numerically (0/1). Use both arms.
    g = snap['gender'].to_numpy(dtype=float)
    if np.isfinite(g).sum() > 0:
        u = np.unique(g[np.isfinite(g)])
        if len(u) == 2:
            cohorts[f'sex_{int(u[0])}'] = (g == u[0])
            cohorts[f'sex_{int(u[1])}'] = (g == u[1])

    # Age tertiles
    age = snap['baseline_age'].to_numpy(dtype=float)
    cohorts['age_lt50'] = (age < 50)
    cohorts['age_50_65'] = (age >= 50) & (age < 65)
    cohorts['age_ge65'] = (age >= 65)

    # FIB-4 = age * AST / (PLT * sqrt(ALT)). Standard cutoffs 1.3 / 2.67.
    ast = snap['base_ast'].to_numpy(dtype=float)
    alt = snap['base_alt'].to_numpy(dtype=float)
    plt = snap['base_plt'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        fib4 = age * ast / (plt * np.sqrt(np.maximum(alt, 1e-6)))
    snap['fib4'] = fib4  # store for shift later
    valid = np.isfinite(fib4)
    cohorts['fib4_low'] = valid & (fib4 < 1.3)
    cohorts['fib4_intermediate'] = valid & (fib4 >= 1.3) & (fib4 < 2.67)
    cohorts['fib4_high'] = valid & (fib4 >= 2.67)

    # Follow-up duration tertiles -> short/long
    fu = snap['followup_years'].to_numpy(dtype=float)
    fu_finite = fu[np.isfinite(fu)]
    if fu_finite.size:
        med = float(np.median(fu_finite))
        cohorts['followup_short'] = (fu < med) & np.isfinite(fu)
        cohorts['followup_long'] = (fu >= med) & np.isfinite(fu)

    cohorts['ALL'] = np.ones(n, dtype=bool)
    return cohorts


def list_variant_basenames():
    """Variant base name = config_id (stem of oof_<endpoint>_<config_id>.npy)."""
    hep = {f.name[len('oof_hepatic_'):-4] for f in BRANCHES.glob('oof_hepatic_*.npy')}
    de = {f.name[len('oof_death_'):-4] for f in BRANCHES.glob('oof_death_*.npy')}
    return sorted(hep & de)


def load_variant(endpoint: str, config_id: str) -> np.ndarray | None:
    p = BRANCHES / f'oof_{endpoint}_{config_id}.npy'
    if not p.exists():
        return None
    return np.load(p)


def weighted_lb(c_hep: float, c_death: float) -> float:
    if not np.isfinite(c_hep) or not np.isfinite(c_death):
        return float('nan')
    return 0.7 * c_hep + 0.3 * c_death


def select_top15(variants: list[str], data: dict) -> list[str]:
    """Rank variants by weighted train OOF (0.7 hep + 0.3 death) and take top 15."""
    rows = []
    yh = data['hepatic']
    yd = data['death']
    for v in variants:
        sh = load_variant('hepatic', v)
        sd = load_variant('death', v)
        if sh is None or sd is None:
            continue
        mh = yh['valid_mask'] & np.isfinite(sh) & np.isfinite(yh['time'])
        md = yd['valid_mask'] & np.isfinite(sd) & np.isfinite(yd['time'])
        ch = concordance(yh['y_struct'][mh], sh[mh]) if mh.sum() > 20 else np.nan
        cd = concordance(yd['y_struct'][md], sd[md]) if md.sum() > 20 else np.nan
        rows.append((v, ch, cd, weighted_lb(ch, cd)))
    rows.sort(key=lambda r: (-(r[3] if np.isfinite(r[3]) else -1)))
    return [r[0] for r in rows[:15]]


def per_cohort_table(top_variants: list[str], data: dict, cohorts: dict[str, np.ndarray]):
    rows = []
    yh = data['hepatic']
    yd = data['death']
    for v in top_variants:
        sh = load_variant('hepatic', v)
        sd = load_variant('death', v)
        if sh is None or sd is None:
            continue
        for cname, mask in cohorts.items():
            mh = mask & yh['valid_mask'] & np.isfinite(sh) & np.isfinite(yh['time'])
            md = mask & yd['valid_mask'] & np.isfinite(sd) & np.isfinite(yd['time'])
            ch = concordance(yh['y_struct'][mh], sh[mh]) if (mh.sum() >= 20 and yh['event'][mh].sum() >= 3) else np.nan
            cd = concordance(yd['y_struct'][md], sd[md]) if (md.sum() >= 20 and yd['event'][md].sum() >= 3) else np.nan
            rows.append({
                'variant': v, 'cohort': cname,
                'n_hep': int(mh.sum()), 'n_death': int(md.sum()),
                'events_hep': int(yh['event'][mh].sum()),
                'events_death': int(yd['event'][md].sum()),
                'C_hep': ch, 'C_death': cd,
                'weighted_LB': weighted_lb(ch, cd),
            })
    return pd.DataFrame(rows)


# ---------- C. per-horizon ----------
def censor_at_horizon(y_struct: np.ndarray, valid_mask: np.ndarray, horizon: float):
    """Right-censor at horizon years. Returns new structured array.
    Patients with time > horizon: event=False, time=horizon.
    Patients censored before horizon: kept as-is.
    """
    new = y_struct.copy()
    over = valid_mask & np.isfinite(new['time']) & (new['time'] > horizon)
    new['event'] = new['event'] & ~over
    new['time'] = np.where(over, horizon, new['time'])
    return new


def per_horizon_table(top_variants: list[str], data: dict, horizons: list[float]):
    rows = []
    yh = data['hepatic']
    yd = data['death']
    for v in top_variants:
        sh = load_variant('hepatic', v)
        sd = load_variant('death', v)
        if sh is None or sd is None:
            continue
        for H in horizons:
            yh_h = censor_at_horizon(yh['y_struct'], yh['valid_mask'], H)
            yd_h = censor_at_horizon(yd['y_struct'], yd['valid_mask'], H)
            mh = yh['valid_mask'] & np.isfinite(sh) & np.isfinite(yh_h['time'])
            md = yd['valid_mask'] & np.isfinite(sd) & np.isfinite(yd_h['time'])
            ch = concordance(yh_h[mh], sh[mh]) if (mh.sum() > 20 and yh_h['event'][mh].sum() >= 3) else np.nan
            cd = concordance(yd_h[md], sd[md]) if (md.sum() > 20 and yd_h['event'][md].sum() >= 3) else np.nan
            rows.append({
                'variant': v, 'horizon_yr': H,
                'events_hep': int(yh_h['event'][mh].sum()),
                'events_death': int(yd_h['event'][md].sum()),
                'C_hep': ch, 'C_death': cd, 'weighted_LB': weighted_lb(ch, cd),
            })
    return pd.DataFrame(rows)


# ---------- driver ----------
def main():
    val_path = os.environ.get('ANNITIA_VAL_TEST', '/Volumes/seagate/kirby/annitia/val_test.csv')
    val_dir = str(Path(val_path).parent)
    val_name = Path(val_path).name
    print(f'Loading train + val ({val_path})')
    train = load_annitia()
    val = load_annitia(data_dir=val_dir, filename=val_name, has_outcomes=False)

    train_snap = baseline_snapshot(train)
    val_snap = baseline_snapshot(val)

    # Restrict shift analysis to columns present in both, numeric, non-id
    common_cols = [c for c in train_snap.columns if c in val_snap.columns
                   and c not in ('patient_id_anon',)
                   and pd.api.types.is_numeric_dtype(train_snap[c])]
    shift_df, _ = distribution_shift(train_snap[common_cols + ['patient_id_anon']],
                                     val_snap[common_cols + ['patient_id_anon']])
    shift_df.to_csv(FULL / 'distribution_shift.csv', index=False)
    print(f'Wrote {FULL/"distribution_shift.csv"} ({len(shift_df)} features)')

    # Restrict heavy analyses to features with reasonable coverage
    ok_cols = shift_df[(shift_df['miss_train'] < 0.4) & (shift_df['miss_val'] < 0.4)
                       & shift_df['ks_stat'].notna()]['feature'].tolist()
    if not ok_cols:
        ok_cols = [c for c in common_cols if train_snap[c].notna().mean() > 0.6]

    d2_train, d2_val, used_cols = mahalanobis_distances(train_snap, val_snap, ok_cols)
    auroc, coef_series = domain_classifier_auroc(train_snap, val_snap, used_cols)
    ot_mean, ot_med, ot_p90 = optimal_transport_mass(train_snap, val_snap, used_cols)

    summary = {
        'n_train': int(len(train_snap)),
        'n_val': int(len(val_snap)),
        'n_features_total': int(len(common_cols)),
        'n_features_used_multivar': int(len(used_cols)),
        'mahalanobis': {
            'train_mean': float(np.mean(d2_train)), 'train_median': float(np.median(d2_train)),
            'val_mean': float(np.mean(d2_val)), 'val_median': float(np.median(d2_val)),
            'val_p90': float(np.percentile(d2_val, 90)),
            'val_pct_above_train_p95': float(np.mean(d2_val > np.percentile(d2_train, 95))),
        },
        'domain_classifier_auroc': auroc,
        'top_discriminating_features': coef_series.head(15).to_dict(),
        'optimal_transport_nn_mahalanobis': {
            'mean_min_dist': ot_mean, 'median_min_dist': ot_med, 'p90_min_dist': ot_p90,
        },
        'top_shifted_features_by_ks': shift_df.head(15)[
            ['feature', 'ks_stat', 'wasserstein', 'mean_diff', 'std_diff']
        ].to_dict('records'),
    }
    with open(FULL / 'distribution_shift_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {FULL/"distribution_shift_summary.json"} (AUROC={auroc:.3f})')

    # ---------- B + C ----------
    variants = list_variant_basenames()
    top15 = select_top15(variants, train)
    print(f'Top 15 variants by weighted train OOF: {top15}')
    cohorts = define_cohorts(train_snap)
    per_cohort = per_cohort_table(top15, train, cohorts)
    per_cohort.to_csv(FULL / 'per_cohort_C.csv', index=False)
    print(f'Wrote {FULL/"per_cohort_C.csv"} ({len(per_cohort)} rows)')

    horizons = [1.0, 3.0, 5.0, 10.0]
    per_horizon = per_horizon_table(top15, train, horizons)
    per_horizon.to_csv(FULL / 'per_horizon_C.csv', index=False)
    print(f'Wrote {FULL/"per_horizon_C.csv"} ({len(per_horizon)} rows)')

    # ---------- audit markdown ----------
    write_audit_md(summary, shift_df, per_cohort, per_horizon, top15, cohorts)


def write_audit_md(summary, shift_df, per_cohort, per_horizon, top15, cohorts):
    lines = [
        '# ANNITIA distribution & cohort audit',
        '',
        f'Train n = {summary["n_train"]} (hep-valid 1106, death-valid 966).  '
        f'Val_test n = {summary["n_val"]}.',
        f'Features compared: {summary["n_features_total"]}.  '
        f'Used in multivariate shift: {summary["n_features_used_multivar"]}.',
        '',
        '## A. Distribution shift summary',
        '',
        f'- **Domain classifier AUROC** (5-fold OOF logistic regression, '
        f'standardised features, mean-imputed): **{summary["domain_classifier_auroc"]:.3f}**.',
        '  - >=0.7 indicates a meaningful shift; >=0.8 is severe; ~0.5 means no shift.',
        f'- **Mahalanobis (val vs train centroid)**: train median '
        f'{summary["mahalanobis"]["train_median"]:.2f}, '
        f'val median {summary["mahalanobis"]["val_median"]:.2f}, '
        f'val p90 {summary["mahalanobis"]["val_p90"]:.2f}.',
        f'  - Fraction of val patients above train p95 distance: '
        f'**{summary["mahalanobis"]["val_pct_above_train_p95"]*100:.1f}%** '
        '(no shift => ~5%).',
        f'- **OT proxy** (mean nearest-neighbour Mahalanobis from val -> train, '
        f'sub-sampled): mean {summary["optimal_transport_nn_mahalanobis"]["mean_min_dist"]:.2f}, '
        f'median {summary["optimal_transport_nn_mahalanobis"]["median_min_dist"]:.2f}, '
        f'p90 {summary["optimal_transport_nn_mahalanobis"]["p90_min_dist"]:.2f}.',
        '',
        '## A.1 Most-shifted features (top 15 by KS statistic)',
        '',
        '| feature | KS | Wasserstein* | mean_train | mean_val | std_train | std_val |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in shift_df.head(15).itertuples():
        lines.append(
            f'| {r.feature} | {r.ks_stat:.3f} | {r.wasserstein:.3f} | '
            f'{r.mean_train:.2f} | {r.mean_val:.2f} | '
            f'{r.std_train:.2f} | {r.std_val:.2f} |'
        )
    lines += [
        '',
        '*Wasserstein computed on standardised values (pooled SD), so values are '
        'comparable across features.',
        '',
        '## A.2 Top discriminating features (logistic-regression weights)',
        '',
        'Sign convention: positive = more likely val_test. Coefficients on '
        'standardised features.',
        '',
        '| feature | LR coef |',
        '|---|---|',
    ]
    for f, w in list(summary['top_discriminating_features'].items())[:15]:
        lines.append(f'| {f} | {w:+.3f} |')

    # B. cohort
    lines += [
        '',
        '## B. Per-cohort C-index (top 15 variants by weighted train OOF)',
        '',
        f'Top 15 variants ranked by 0.7*C_hep + 0.3*C_death on training OOF:',
        '',
    ]
    for v in top15:
        lines.append(f'  - `{v}`')

    # Cohort sizes
    lines += [
        '',
        '### Cohort sizes (training set)',
        '',
        '| cohort | n |',
        '|---|---|',
    ]
    for cname, mask in cohorts.items():
        lines.append(f'| {cname} | {int(mask.sum())} |')

    # Identify weakest variant per cohort (largest LB drop vs ALL)
    if 'ALL' in per_cohort['cohort'].unique():
        all_lb = per_cohort[per_cohort['cohort'] == 'ALL'].set_index('variant')['weighted_LB']
        per_cohort = per_cohort.copy()
        per_cohort['LB_vs_all'] = per_cohort.apply(
            lambda r: r['weighted_LB'] - all_lb.get(r['variant'], np.nan), axis=1)
        weakest = (per_cohort[per_cohort['cohort'] != 'ALL']
                   .dropna(subset=['LB_vs_all'])
                   .sort_values('LB_vs_all'))
        lines += [
            '',
            '### Weakest cohort drops (variant x cohort), top 15 by LB delta vs ALL',
            '',
            '| variant | cohort | n_hep | events_hep | C_hep | C_death | weighted_LB | delta_vs_all |',
            '|---|---|---|---|---|---|---|---|',
        ]
        for r in weakest.head(15).itertuples():
            lines.append(
                f'| {r.variant} | {r.cohort} | {r.n_hep} | {r.events_hep} | '
                f'{r.C_hep:.3f} | {r.C_death:.3f} | {r.weighted_LB:.3f} | '
                f'{r.LB_vs_all:+.3f} |'
            )

        # Cohort-specific weakness summary
        lines += [
            '',
            '### Cohort-specific failures (LB drop > 0.05 vs ALL)',
            '',
        ]
        bad = weakest[weakest['LB_vs_all'] < -0.05]
        if len(bad):
            for r in bad.itertuples():
                lines.append(
                    f'- `{r.variant}` drops **{abs(r.LB_vs_all):.3f}** on '
                    f'`{r.cohort}` (n_hep={r.n_hep}, ev_hep={r.events_hep}).'
                )
        else:
            lines.append('- None.')

    # C. horizon
    lines += [
        '',
        '## C. Per-time-horizon C-index',
        '',
        '| variant | horizon | C_hep | C_death | weighted_LB |',
        '|---|---|---|---|---|',
    ]
    ph_sorted = per_horizon.sort_values(['variant', 'horizon_yr'])
    for r in ph_sorted.itertuples():
        lines.append(
            f'| {r.variant} | {r.horizon_yr:.0f}y | '
            f'{r.C_hep:.3f} | {r.C_death:.3f} | {r.weighted_LB:.3f} |'
        )

    # Horizon stability
    lines += ['', '### Horizon stability (max - min weighted_LB per variant)', '']
    horizon_pivot = per_horizon.pivot(index='variant', columns='horizon_yr', values='weighted_LB')
    horizon_pivot['range'] = horizon_pivot.max(axis=1) - horizon_pivot.min(axis=1)
    horizon_pivot = horizon_pivot.sort_values('range')
    lines += [
        '| variant | LB range across horizons |',
        '|---|---|',
    ]
    for v in horizon_pivot.index:
        lines.append(f'| {v} | {horizon_pivot.loc[v, "range"]:.3f} |')
    most_stable = horizon_pivot.head(3).index.tolist()
    least_stable = horizon_pivot.tail(3).index.tolist()
    lines += [
        '',
        f'**Most horizon-stable**: {", ".join(most_stable)}.',
        f'**Most horizon-sensitive**: {", ".join(least_stable)}.',
        '',
        '## Notes on cohort proxies',
        '',
        '- **Diabetes**: T2DM column OR fasting glucose >= 7 mmol/L. '
        '(No HbA1c column present in this dataset.)',
        '- **Sex**: `gender` column (binary).',
        '- **FIB-4**: age * AST / (PLT * sqrt(ALT)) using baseline-visit '
        'values; cutoffs 1.3 / 2.67.',
        '- **Follow-up**: median split on (last_followup_age - baseline_age).',
        '',
    ]

    AUDIT_MD.write_text('\n'.join(lines))
    print(f'Wrote {AUDIT_MD}')


if __name__ == '__main__':
    main()
