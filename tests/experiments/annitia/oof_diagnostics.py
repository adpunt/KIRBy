"""Per-base-model OOF diagnostics: bootstrap CIs, per-fold variance, trust flags.

The aim: stop trusting point estimates of OOF C. With 47 hepatic events the
standard error on C is ~0.025 — a 0.97 OOF estimate has a 95% CI down to ~0.92.

Outputs:
  tests/results/annitia/full/oof_diagnostics.json — per-model dict
  tests/results/annitia/full/oof_diagnostics.md   — human-readable report

Trust flags:
  HIGH    — OOF C lower-CI ≥ 0.85 AND per-fold std ≤ 0.04
  MEDIUM  — OOF C lower-CI ≥ 0.75 AND per-fold std ≤ 0.06
  LOW     — anything else (use only as diversity filler at low weight)
  SUSPECT — pooled C > 0.97 with per-fold std > 0.05 (likely overfit on a fold)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia              # noqa: E402
from experiments.annitia.survival import concordance               # noqa: E402

BRANCHES = REPO / 'tests/results/annitia/branches'
FULL = REPO / 'tests/results/annitia/full'


def bootstrap_ci(y_struct, scores, valid_mask, n_boot=1000, seed=42, alpha=0.05):
    """Bootstrap 95% CI on concordance index.

    Resamples patient indices with replacement. Returns (mean, lo, hi).
    """
    rng = np.random.default_rng(seed)
    finite = valid_mask & np.isfinite(scores) & np.isfinite(y_struct['time'])
    idx = np.where(finite)[0]
    if len(idx) < 20:
        return float('nan'), float('nan'), float('nan')
    cs = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        try:
            c = float(concordance(y_struct[b], scores[b]))
            if np.isfinite(c):
                cs.append(c)
        except Exception:
            continue
    if len(cs) < 10:
        return float('nan'), float('nan'), float('nan')
    arr = np.array(cs)
    return float(arr.mean()), float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def per_fold_c(y_struct, scores, valid_mask, n_splits=5, seed=42):
    """Per-fold C across the same KFold split used in training. Returns list."""
    finite = valid_mask & np.isfinite(scores) & np.isfinite(y_struct['time'])
    idx = np.where(finite)[0]
    if len(idx) < 20:
        return []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cs = []
    for _, te in kf.split(idx):
        ti = idx[te]
        if len(ti) < 5 or y_struct['event'][ti].sum() < 2:
            continue
        try:
            c = float(concordance(y_struct[ti], scores[ti]))
            if np.isfinite(c):
                cs.append(c)
        except Exception:
            continue
    return cs


def trust_flag(pooled_c, lo_ci, fold_std):
    if not np.isfinite(pooled_c):
        return 'NA'
    if pooled_c > 0.97 and (fold_std > 0.05 or fold_std != fold_std):  # NaN std
        return 'SUSPECT'
    if lo_ci >= 0.85 and fold_std <= 0.04:
        return 'HIGH'
    if lo_ci >= 0.75 and fold_std <= 0.06:
        return 'MEDIUM'
    return 'LOW'


def diagnose_endpoint(endpoint: str, y_struct, valid_mask):
    files = list(BRANCHES.glob(f'oof_{endpoint}_*.npy'))
    out = {}
    print(f'\n=== {endpoint} ({len(files)} base models) ===')
    print(f"  n_events = {y_struct['event'].sum()}, n_valid = {valid_mask.sum()}")
    print(f"  pooled SE_C ~ {np.sqrt(0.95 * 0.05 / max(y_struct['event'].sum(), 1)):.3f}")
    print()

    for f in sorted(files):
        cid = f.name.replace(f'oof_{endpoint}_', '').replace('.npy', '')
        try:
            scores = np.load(f)
        except Exception as e:
            out[cid] = {'error': str(e)}
            continue
        if scores.shape[0] != y_struct.shape[0]:
            out[cid] = {'error': f'shape {scores.shape} != y {y_struct.shape}'}
            continue

        # pooled C
        finite = valid_mask & np.isfinite(scores) & np.isfinite(y_struct['time'])
        if finite.sum() < 20:
            out[cid] = {'error': f'only {int(finite.sum())} finite valid rows'}
            continue
        pooled = float(concordance(y_struct[finite], scores[finite]))

        # bootstrap CI
        boot_mean, lo, hi = bootstrap_ci(y_struct, scores, valid_mask, n_boot=500)

        # per-fold
        folds = per_fold_c(y_struct, scores, valid_mask)
        fmean = float(np.mean(folds)) if folds else float('nan')
        fstd = float(np.std(folds)) if len(folds) >= 2 else float('nan')

        flag = trust_flag(pooled, lo, fstd)
        out[cid] = {
            'pooled_c': pooled, 'boot_mean': boot_mean,
            'boot_lo95': lo, 'boot_hi95': hi,
            'fold_c_mean': fmean, 'fold_c_std': fstd,
            'fold_n': len(folds), 'flag': flag,
        }
        print(f"  [{flag:7s}] {cid[:60]:60s}  C={pooled:.3f} CI[{lo:.3f},{hi:.3f}]  fold {fmean:.3f}±{fstd:.3f}")

    return out


def main():
    train = load_annitia()
    h_y = train['hepatic']['y_struct']
    h_v = train['hepatic']['valid_mask']
    d_y = train['death']['y_struct']
    d_v = train['death']['valid_mask']

    diag = {}
    diag['hepatic'] = diagnose_endpoint('hepatic', h_y, h_v)
    diag['death'] = diagnose_endpoint('death', d_y, d_v)

    # Summary stats per endpoint
    summary = {}
    for ep, d in diag.items():
        flags = [r.get('flag', 'NA') for r in d.values() if 'pooled_c' in r]
        summary[ep] = {
            'n_models': len([r for r in d.values() if 'pooled_c' in r]),
            'HIGH': flags.count('HIGH'),
            'MEDIUM': flags.count('MEDIUM'),
            'LOW': flags.count('LOW'),
            'SUSPECT': flags.count('SUSPECT'),
        }

    out_json = FULL / 'oof_diagnostics.json'
    out_json.write_text(json.dumps({'per_model': diag, 'summary': summary}, indent=2))
    print(f'\nwrote {out_json}')

    # Markdown report
    md = ['# OOF diagnostics — base model trust report',
          '',
          'Generated by `oof_diagnostics.py`. Trust flags account for sample-size noise.',
          '',
          '## Standard error context',
          '',
          f'- hepatic: {h_y["event"].sum()} events → SE on C ≈ 0.025 (95% CI half-width ≈ 0.05)',
          f'- death:   {d_y["event"].sum()} events → SE on C ≈ 0.020 (95% CI half-width ≈ 0.04)',
          '',
          '**Implication:** any difference between two models with C-index < 0.05 apart is statistically tied.',
          '',
          '## Trust flag definitions',
          '',
          '- **HIGH**: lower 95% CI ≥ 0.85 AND per-fold std ≤ 0.04. Use at face value.',
          '- **MEDIUM**: lower CI ≥ 0.75 AND per-fold std ≤ 0.06. Use with caution.',
          '- **LOW**: weaker. Only as diversity filler at low weight.',
          '- **SUSPECT**: pooled C > 0.97 with per-fold std > 0.05 — likely overfit on one lucky fold.',
          '',
          '## Per-endpoint summary',
          '']
    for ep, s in summary.items():
        md.append(f'### {ep}: {s["n_models"]} models — '
                  f'HIGH={s["HIGH"]}, MEDIUM={s["MEDIUM"]}, LOW={s["LOW"]}, SUSPECT={s["SUSPECT"]}')
        md.append('')
        md.append('| Model | Pooled C | 95% CI | Fold C ± σ | Flag |')
        md.append('|---|---|---|---|---|')
        rows = sorted(diag[ep].items(),
                      key=lambda kv: -(kv[1].get('pooled_c', 0) if 'pooled_c' in kv[1] else 0))
        for cid, r in rows:
            if 'pooled_c' not in r:
                continue
            md.append(f'| `{cid[:50]}` | {r["pooled_c"]:.3f} | '
                      f'[{r["boot_lo95"]:.3f}, {r["boot_hi95"]:.3f}] | '
                      f'{r["fold_c_mean"]:.3f} ± {r["fold_c_std"]:.3f} | '
                      f'**{r["flag"]}** |')
        md.append('')

    out_md = FULL / 'oof_diagnostics.md'
    out_md.write_text('\n'.join(md))
    print(f'wrote {out_md}')

    # Top-line print
    print('\n## TOP LINE')
    for ep, s in summary.items():
        print(f'  {ep}: {s["HIGH"]} HIGH-trust, {s["SUSPECT"]} SUSPECT (likely overfit)')


if __name__ == '__main__':
    main()
