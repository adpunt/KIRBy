"""Per-patient residual analysis across all OOF .npy files.

Runs offline against `tests/results/annitia/{be,cd,hep_boost,branches,external}/`,
does not touch any in-flight pipeline. Cheap (vectorised, O(N^2 x C) on 1.3k
patients x ~150 conditions ~ 250M ops, finishes in seconds).

Outputs to `tests/results/annitia/analysis/`:

    per_patient_<endpoint>.csv
        Row = patient. Columns = mean_disc, max_disc, std_disc,
        n_comparable, best_model, worst_model. One file per endpoint.

    per_model_<endpoint>.csv
        Row = model. Columns = mean_disc_per_patient, n_patients_worst,
        n_patients_best, mean_cindex (recomputed). One file per endpoint.

    error_corr_<endpoint>.csv
        Pairwise Pearson correlation of patient-level discordance vectors
        across all models. Models with low corr are picking up different
        signal — good ensemble candidates.

    hard_patients_<endpoint>.csv
        Top-50 patients ranked by mean discordance (intrinsically hardest).

    anomaly_corr_<endpoint>.csv
        Correlation of mean per-patient discordance with IsolationForest
        and LocalOutlierFactor scores on raw features. If nontrivially
        positive: anomaly detection is a useful auxiliary feature.

    router_<endpoint>.csv
        Per-patient best-model assignment: which model has lowest
        per-patient discordance. Useful for "mixture of experts" gating.

Usage:
    python tests/experiments/annitia/analyse_residuals.py
    python tests/experiments/annitia/analyse_residuals.py --endpoint death
    python tests/experiments/annitia/analyse_residuals.py --pools be,branches
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia          # noqa: E402
from experiments.annitia.views import build_views              # noqa: E402


def _load_oofs(pool_dirs, endpoint):
    """Collect all (label, oof_vec) pairs across pool dirs for one endpoint."""
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
            p = d / f'oof_{endpoint}_{cid}.npy'
            if not p.exists():
                continue
            try:
                v = np.load(p)
            except Exception:
                continue
            out.append({'label': f'{d.name}__{cid}', 'cv': float(mc),
                        'oof': v.astype(np.float64), 'source': d.name})
    return out


def per_patient_discordance(time, event, risk):
    """Return (disc_count_i, comparable_count_i) length-N arrays.

    Pair (i, j) is comparable when one has event before the other's time:
        time_i < time_j AND event_i == True   (i is the earlier event)
    For each such pair, the *correct* ordering is risk_i > risk_j (higher
    risk for the earlier event). The model is *discordant on pair (i, j)*
    if risk_i <= risk_j (treating ties as half-discordant).

    Per-patient discordance attributes the error to BOTH i and j (the pair
    is shared) — this gives a symmetric "how hard is patient i" measure.
    """
    n = len(time)
    finite = np.isfinite(time) & np.isfinite(risk)
    time = np.where(finite, time, np.nan)
    event = event.astype(bool) & finite
    # Pair-level: (i, j) where i has event AND time_i < time_j AND finite
    # both. Use broadcast.
    t = time[:, None]; tj = time[None, :]
    e_i = event[:, None]
    finite_pair = np.isfinite(t) & np.isfinite(tj) & (t < tj)
    comparable = finite_pair & e_i  # (n, n) — i is the earlier event
    # Concordant: risk_i > risk_j; tie counts half.
    r = risk[:, None]; rj = risk[None, :]
    discordant = comparable & (r < rj)
    tied = comparable & (r == rj)
    # Each pair (i, j) with comparable=True attributes 1 to BOTH i and j.
    pairs_per_i = comparable.sum(axis=1)              # i is earlier-event
    pairs_per_j = comparable.sum(axis=0)              # j is later (no event needed)
    discord_i = discordant.sum(axis=1) + 0.5 * tied.sum(axis=1)
    discord_j = discordant.sum(axis=0) + 0.5 * tied.sum(axis=0)
    comparable_count = pairs_per_i + pairs_per_j
    discord_count = discord_i + discord_j
    return discord_count, comparable_count


def compute_endpoint(train, endpoint, pool_dirs):
    y = train[endpoint]['y_struct']
    valid = train[endpoint]['valid_mask']
    time = np.asarray(y['time'], dtype=np.float64)
    event = np.asarray(y['event'], dtype=bool)
    n = len(time)

    runs = _load_oofs(pool_dirs, endpoint)
    if not runs:
        print(f'  no OOFs found for {endpoint}; skipping')
        return None
    print(f'  {endpoint}: {len(runs)} models, n={n} valid={int(valid.sum())} events={int(event.sum())}')

    M = len(runs)
    # Per-patient miss rate matrix (n x M); NaN where no comparable pairs.
    miss = np.full((n, M), np.nan)
    cmp_counts = np.zeros((n, M))
    for j, r in enumerate(runs):
        d_i, c_i = per_patient_discordance(time, event, r['oof'])
        with np.errstate(invalid='ignore', divide='ignore'):
            mr = np.where(c_i > 0, d_i / c_i, np.nan)
        miss[:, j] = mr
        cmp_counts[:, j] = c_i

    # Restrict to valid patients with at least one comparable pair somewhere
    keep = valid & np.isfinite(np.nanmean(miss, axis=1))
    miss_keep = miss[keep]
    return {
        'runs': runs,
        'miss': miss,             # (n, M) per-patient miss rate
        'cmp': cmp_counts,
        'keep': keep,
        'miss_keep': miss_keep,   # (n_kept, M)
        'time': time, 'event': event, 'valid': valid,
    }


def write_per_patient(res, train, endpoint, out_dir):
    runs = res['runs']
    miss = res['miss']; keep = res['keep']
    pid = np.asarray(train['patient_ids'])
    labels = [r['label'] for r in runs]
    with np.errstate(invalid='ignore'):
        mean_d = np.nanmean(miss, axis=1)
        max_d = np.nanmax(miss, axis=1)
        std_d = np.nanstd(miss, axis=1)
    best_idx = np.nanargmin(np.where(np.isnan(miss), 1.0, miss), axis=1)
    worst_idx = np.nanargmax(np.where(np.isnan(miss), -1.0, miss), axis=1)
    df = pd.DataFrame({
        'patient_id': pid,
        'event': res['event'],
        'time': res['time'],
        'kept': keep,
        'mean_disc': mean_d,
        'max_disc': max_d,
        'std_disc': std_d,
        'best_model': [labels[i] for i in best_idx],
        'worst_model': [labels[i] for i in worst_idx],
        'n_models_with_signal': np.sum(np.isfinite(miss), axis=1),
    })
    p = out_dir / f'per_patient_{endpoint}.csv'
    df.to_csv(p, index=False)
    print(f'  wrote {p}')
    # Hard patients = top-50 by mean_disc (only kept)
    hard = df[df['kept']].sort_values('mean_disc', ascending=False).head(50)
    hp = out_dir / f'hard_patients_{endpoint}.csv'
    hard.to_csv(hp, index=False)
    print(f'  wrote {hp}')
    return df


def write_per_model(res, endpoint, out_dir):
    runs = res['runs']
    miss = res['miss']; keep = res['keep']
    labels = [r['label'] for r in runs]
    cv = [r['cv'] for r in runs]
    miss_keep = res['miss_keep']
    with np.errstate(invalid='ignore'):
        mean_per = np.nanmean(miss_keep, axis=0)
    # For each patient (kept), which model is its best/worst?
    best_idx = np.nanargmin(np.where(np.isnan(miss_keep), 1.0, miss_keep), axis=1)
    worst_idx = np.nanargmax(np.where(np.isnan(miss_keep), -1.0, miss_keep), axis=1)
    n_best = np.bincount(best_idx, minlength=len(runs))
    n_worst = np.bincount(worst_idx, minlength=len(runs))
    df = pd.DataFrame({
        'model': labels,
        'mean_cindex': cv,
        'mean_disc_per_patient': mean_per,
        'n_patients_best': n_best,
        'n_patients_worst': n_worst,
        'source': [r['source'] for r in runs],
    }).sort_values('mean_disc_per_patient', ascending=True)
    p = out_dir / f'per_model_{endpoint}.csv'
    df.to_csv(p, index=False)
    print(f'  wrote {p}')


def write_error_corr(res, endpoint, out_dir):
    miss_keep = res['miss_keep']
    labels = [r['label'] for r in res['runs']]
    # Use Pearson on per-patient miss rates; impute nans with column mean.
    M = miss_keep.copy()
    for j in range(M.shape[1]):
        col = M[:, j]
        cmean = np.nanmean(col)
        col[np.isnan(col)] = cmean
        M[:, j] = col
    if M.shape[0] < 5 or M.shape[1] < 2:
        print(f'  too few rows/columns for error corr ({M.shape})')
        return
    corr = np.corrcoef(M, rowvar=False)
    df = pd.DataFrame(corr, index=labels, columns=labels)
    p = out_dir / f'error_corr_{endpoint}.csv'
    df.to_csv(p)
    print(f'  wrote {p}')
    # Find the most-uncorrelated pairs (low |r| = orthogonal signal)
    pairs = []
    n = corr.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            pairs.append((labels[a], labels[b], corr[a, b]))
    pairs.sort(key=lambda x: abs(x[2]))
    df_orth = pd.DataFrame(pairs[:30], columns=['model_a', 'model_b', 'pearson'])
    df_orth.to_csv(out_dir / f'error_orth_pairs_{endpoint}.csv', index=False)


def write_anomaly_corr(res, train, endpoint, out_dir):
    """Train IsolationForest + LocalOutlierFactor on raw clinical+snapshot
    features, correlate score with mean per-patient discordance.

    Strong positive correlation => anomaly score predicts which patients
    the ensemble gets wrong, so it's a useful auxiliary feature for fusion.
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        print('  sklearn missing; skipping anomaly cross-ref')
        return
    miss = res['miss']; keep = res['keep']
    with np.errstate(invalid='ignore'):
        mean_d = np.nanmean(miss, axis=1)
    # Build raw feature matrix from snapshot+clinical_pro
    views = build_views(train, view_names=['snapshot', 'clinical_pro'])
    X = np.concatenate([views[v][0] for v in ['snapshot', 'clinical_pro']],
                       axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # IsolationForest higher = more normal — flip for anomaly-ness
    iso = IsolationForest(n_estimators=200, contamination='auto',
                          random_state=42).fit(X)
    iso_score = -iso.score_samples(X)   # higher = more anomalous
    # LOF: novelty=False so we get fit_predict on the same data
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    lof.fit_predict(X)
    lof_score = -lof.negative_outlier_factor_   # higher = more anomalous
    keep_finite = keep & np.isfinite(mean_d)
    rows = []
    for name, sc in (('isolation_forest', iso_score), ('lof', lof_score)):
        s = sc[keep_finite]; d = mean_d[keep_finite]
        if s.size < 5:
            continue
        # Pearson + Spearman
        from scipy.stats import pearsonr, spearmanr
        rp, pp = pearsonr(s, d)
        rs, ps = spearmanr(s, d)
        rows.append({'detector': name,
                     'pearson_r': rp, 'pearson_p': pp,
                     'spearman_r': rs, 'spearman_p': ps,
                     'n': int(s.size)})
    if rows:
        df = pd.DataFrame(rows)
        p = out_dir / f'anomaly_corr_{endpoint}.csv'
        df.to_csv(p, index=False)
        print(f'  wrote {p}')


def write_router(res, train, endpoint, out_dir):
    """Per-patient best-model assignment.

    The 'router' answers: 'if I had to pick ONE model for this patient,
    which would be best?' A useful diagnostic before turning this into
    an actual mixture-of-experts gate.
    """
    miss = res['miss']; keep = res['keep']
    runs = res['runs']
    labels = [r['label'] for r in runs]
    pid = np.asarray(train['patient_ids'])
    best_idx = np.nanargmin(np.where(np.isnan(miss), 1.0, miss), axis=1)
    df = pd.DataFrame({
        'patient_id': pid,
        'kept': keep,
        'best_model': [labels[i] for i in best_idx],
        'best_model_disc': miss[np.arange(len(miss)), best_idx],
        'event': res['event'], 'time': res['time'],
    })
    p = out_dir / f'router_{endpoint}.csv'
    df.to_csv(p, index=False)
    print(f'  wrote {p}')
    # Distribution: how concentrated is the routing?
    counts = df[df['kept']]['best_model'].value_counts()
    summary = pd.DataFrame({'best_model': counts.index,
                            'n_patients_chose': counts.values})
    summary.to_csv(out_dir / f'router_summary_{endpoint}.csv', index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--endpoint', default='both', choices=('hepatic', 'death', 'both'))
    ap.add_argument('--pools', default='be,cd,hep_boost,branches,external')
    ap.add_argument('--output-dir', default='tests/results/annitia/analysis')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    res_root = _REPO / 'tests' / 'results' / 'annitia'
    pool_dirs = [res_root / p.strip() for p in args.pools.split(',')
                 if p.strip()]
    pool_dirs = [d for d in pool_dirs if d.exists()]
    print(f'Pooling from: {[d.name for d in pool_dirs]}')

    train = load_annitia()
    eps = ('hepatic', 'death') if args.endpoint == 'both' else (args.endpoint,)
    for ep in eps:
        print(f'\n=== {ep} ===')
        res = compute_endpoint(train, ep, pool_dirs)
        if res is None:
            continue
        write_per_patient(res, train, ep, out)
        write_per_model(res, ep, out)
        write_error_corr(res, ep, out)
        write_anomaly_corr(res, train, ep, out)
        write_router(res, train, ep, out)
    print(f'\nDONE. Inspect CSVs under {out}')


if __name__ == '__main__':
    main()
