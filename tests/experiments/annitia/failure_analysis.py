"""Training-side failure analysis: per-patient FP/FN-style errors.

Concordance is rank-based, but we can decompose it patient-by-patient:
  - FN-like: patient HAD an event but ensemble predicted LOW risk.
  - FP-like: patient had NO event in followup but ensemble predicted HIGH risk
    (might be censored — could still get event later, but on the data we have,
    the ensemble is wasting "high risk" budget on them).

Outputs per endpoint:
  - per-patient discordance (mean across base models)
  - hardest 30 patients (top discordance)
  - FP/FN bucket assignments
  - feature deltas of failing patients vs population

Outputs:
  tests/results/annitia/full/failure_analysis_<endpoint>.json
  tests/results/annitia/full/failure_analysis.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia       # noqa: E402

BRANCHES = REPO / 'tests/results/annitia/branches'
FULL = REPO / 'tests/results/annitia/full'


def rank01_nan(x):
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum():
        out[m] = rankdata(x[m]) / max(m.sum(), 1)
    return out


def per_patient_disc(time, event, risk, valid):
    """Per-patient discordance fraction: fraction of comparable pairs this
    patient is on the wrong side of."""
    finite = valid & np.isfinite(risk) & np.isfinite(time)
    n = len(time)
    out = np.full(n, np.nan)
    if finite.sum() < 5:
        return out
    t = np.where(finite, time, np.nan)
    e = event.astype(bool) & finite
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
        out[:] = np.where(cnt > 0, disc / cnt, np.nan)
    return out


def consensus_risk(endpoint, n):
    """Mean rank01 across base OOF arrays."""
    files = list(BRANCHES.glob(f'oof_{endpoint}_*.npy'))
    arrs = []
    for f in files:
        try:
            a = np.load(f)
            if a.shape[0] != n: continue
            if np.isfinite(a).sum() < 50: continue
            if np.std(a[np.isfinite(a)]) <= 1e-10: continue
            arrs.append(rank01_nan(a))
        except Exception:
            continue
    return np.nanmean(np.stack(arrs, axis=1), axis=1) if arrs else None


def analyse_endpoint(endpoint, train_data, top_k=30):
    y_struct = train_data[endpoint]['y_struct']
    valid = train_data[endpoint]['valid_mask']
    time = train_data[endpoint]['time']
    event = train_data[endpoint]['event']
    n = len(time)

    risk = consensus_risk(endpoint, n)
    if risk is None:
        print(f'  {endpoint}: no usable OOF arrays')
        return None

    disc = per_patient_disc(time, event, risk, valid)

    # Build patient table
    rec = pd.DataFrame({
        'idx': np.arange(n),
        'patient_id': train_data['patient_ids'],
        'baseline_age': train_data['baseline_age'],
        'last_followup_age': train_data['last_followup_age'],
        'event': event,
        'time': time,
        'valid': valid,
        'consensus_risk': risk,
        'risk_pctile': rank01_nan(risk),
        'disc': disc,
    })
    rec_valid = rec[rec['valid'] & rec['disc'].notna()].copy()

    # Buckets
    high_risk = rec_valid['risk_pctile'] > 0.75
    low_risk = rec_valid['risk_pctile'] < 0.25
    has_event = rec_valid['event'] == 1
    no_event = rec_valid['event'] == 0

    fn_like = rec_valid[low_risk & has_event].copy()    # had event, predicted low
    fp_like = rec_valid[high_risk & no_event].copy()    # no event, predicted high
    correct_pos = rec_valid[high_risk & has_event].copy()
    correct_neg = rec_valid[low_risk & no_event].copy()

    print(f'\n=== {endpoint} ===')
    print(f'  n_valid={len(rec_valid)}, n_events={has_event.sum()}, '
          f'event_rate={has_event.mean():.3f}')
    print(f'  FN-like (event, low pred): {len(fn_like)}/{has_event.sum()} '
          f'= {len(fn_like)/max(has_event.sum(),1):.1%} of events')
    print(f'  FP-like (no event, high pred): {len(fp_like)}/{no_event.sum()} '
          f'= {len(fp_like)/max(no_event.sum(),1):.1%} of non-events')
    print(f'  Correctly high-risk (event, high pred): {len(correct_pos)}/{has_event.sum()} '
          f'= {len(correct_pos)/max(has_event.sum(),1):.1%}')
    print(f'  Mean disc on FN-like: {fn_like["disc"].mean():.3f}')
    print(f'  Mean disc on FP-like: {fp_like["disc"].mean():.3f}')
    print(f'  Mean disc overall:   {rec_valid["disc"].mean():.3f}')

    print(f'\n  Hardest {top_k} patients (by disc):')
    print(f'  {"idx":>5} {"event":>5} {"time":>7} {"risk_pc":>8} {"disc":>6} {"age":>5}')
    for _, r in rec_valid.nlargest(top_k, 'disc').iterrows():
        print(f'  {int(r["idx"]):>5} {int(r["event"]):>5} {r["time"]:>7.1f} '
              f'{r["risk_pctile"]:>8.3f} {r["disc"]:>6.3f} {r["baseline_age"]:>5.1f}')

    # Feature comparison: FN-like vs correctly-flagged events
    feature_summary = {}
    if len(fn_like) > 0:
        for col in ['baseline_age', 'last_followup_age', 'time']:
            try:
                pop_mean = rec_valid[col].mean()
                fn_mean = fn_like[col].mean()
                fp_mean = fp_like[col].mean() if len(fp_like) else float('nan')
                feature_summary[col] = {
                    'population_mean': float(pop_mean),
                    'fn_like_mean': float(fn_mean),
                    'fp_like_mean': float(fp_mean),
                }
            except Exception:
                continue

    out = {
        'endpoint': endpoint,
        'n_valid': int(len(rec_valid)),
        'n_events': int(has_event.sum()),
        'event_rate': float(has_event.mean()),
        'fn_like_count': int(len(fn_like)),
        'fp_like_count': int(len(fp_like)),
        'correct_pos_count': int(len(correct_pos)),
        'fn_like_idx': fn_like['idx'].astype(int).tolist(),
        'fp_like_idx': fp_like['idx'].astype(int).tolist(),
        'hardest_idx': rec_valid.nlargest(top_k, 'disc')['idx'].astype(int).tolist(),
        'mean_disc_overall': float(rec_valid['disc'].mean()),
        'mean_disc_fn_like': float(fn_like['disc'].mean()) if len(fn_like) else None,
        'mean_disc_fp_like': float(fp_like['disc'].mean()) if len(fp_like) else None,
        'feature_summary': feature_summary,
    }
    return out


def main():
    train = load_annitia()
    h_out = analyse_endpoint('hepatic', train)
    d_out = analyse_endpoint('death', train)

    diag = {'hepatic': h_out, 'death': d_out}
    (FULL / 'failure_analysis.json').write_text(json.dumps(diag, indent=2))
    print(f'\nwrote {FULL}/failure_analysis.json')

    md = ['# Training-side failure analysis',
          '',
          'Per-patient discordance on consensus OOF (mean rank across base models).',
          '',
          'Definitions:',
          '- **FN-like**: patient had an event but ensemble predicted bottom-quartile risk',
          '- **FP-like**: patient had NO event in followup but ensemble predicted top-quartile risk',
          '',
          '## Per-endpoint summary', '']
    for ep, r in diag.items():
        if r is None: continue
        md.append(f'### {ep}')
        md.append('')
        md.append(f'- n_valid={r["n_valid"]}, n_events={r["n_events"]}, '
                  f'event_rate={r["event_rate"]:.3f}')
        md.append(f'- FN-like: **{r["fn_like_count"]}/{r["n_events"]} '
                  f'= {r["fn_like_count"]/max(r["n_events"],1):.1%}** of events missed')
        md.append(f'- FP-like: **{r["fp_like_count"]}/{r["n_valid"]-r["n_events"]} '
                  f'= {r["fp_like_count"]/max(r["n_valid"]-r["n_events"],1):.1%}** of non-events flagged')
        md.append(f'- correctly-flagged events (high-risk): **{r["correct_pos_count"]}/{r["n_events"]} '
                  f'= {r["correct_pos_count"]/max(r["n_events"],1):.1%}**')
        md.append(f'- mean disc overall: {r["mean_disc_overall"]:.3f}')
        fn_disc = r["mean_disc_fn_like"]
        fp_disc = r["mean_disc_fp_like"]
        md.append(f'- mean disc on FN-like: {fn_disc:.3f}' if fn_disc is not None else '- mean disc on FN-like: NA')
        md.append(f'- mean disc on FP-like: {fp_disc:.3f}' if fp_disc is not None else '- mean disc on FP-like: NA')
        md.append('')
        if r['feature_summary']:
            md.append('Feature comparison (FN-like vs population vs FP-like):')
            md.append('')
            md.append('| Feature | Population | FN-like | FP-like |')
            md.append('|---|---|---|---|')
            for k, v in r['feature_summary'].items():
                md.append(f'| {k} | {v["population_mean"]:.2f} | '
                          f'{v["fn_like_mean"]:.2f} | {v["fp_like_mean"]:.2f} |')
            md.append('')

    (FULL / 'failure_analysis.md').write_text('\n'.join(md))
    print(f'wrote {FULL}/failure_analysis.md')


if __name__ == '__main__':
    main()
