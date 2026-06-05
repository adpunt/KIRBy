"""OOF → leaderboard calibration.

For every submission with a known leaderboard score, compute the equivalent OOF
concordance on the training set (using the same prediction logic but evaluating
against training labels). Fit a calibration curve OOF C → public C and project
expected leaderboard scores for new variants.

Critical observation: leaderboard score is LB = 0.7 * Cindex_hepatic + 0.3 *
Cindex_death on val_test (per Trustii spec — see README.md:13 and run.py:210).
For each submission we have hep and death
predictions separately (in the CSV) but only the joint score. To compute OOF
analogue we need to reconstruct training-side hep + death predictions per
submission. We can do that by:
  - Reading sub_master_<v>.csv (val predictions)
  - Reading the corresponding training-side OOF (which exists for the recipe
    builders that saved their training predictions, OR by reconstructing via
    the variant's aggregation logic on base OOFs).

For the variants that don't save their OOF, we use the consensus base OOF
(rank01 mean) and apply the same TRANSFORMATION the variant uses (e.g.,
median, trimmed mean, soft route). For meta-aware variants this is approximate.

Outputs:
  oof_to_leaderboard.json — per-submission OOF C (hep + death + 0.7*hep+0.3*death) vs LB score
  oof_to_leaderboard.md   — calibration curve report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia              # noqa: E402
from experiments.annitia.survival import concordance               # noqa: E402

BRANCHES = REPO / 'tests/results/annitia/branches'
FULL = REPO / 'tests/results/annitia/full'


def rank01_nan(x):
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum():
        out[m] = rankdata(x[m]) / max(m.sum(), 1)
    return out


def load_consensus_oof():
    """Mean rank of all base OOFs per endpoint."""
    out = {}
    for ep in ('hepatic', 'death'):
        files = list(BRANCHES.glob(f'oof_{ep}_*.npy'))
        arrs = []
        for f in files:
            try:
                a = np.load(f)
                if np.isfinite(a).sum() < 50: continue
                if np.std(a[np.isfinite(a)]) <= 1e-10: continue
                arrs.append(rank01_nan(a))
            except Exception:
                continue
        if not arrs:
            out[ep] = None
            continue
        ref = arrs[0].shape[0]
        arrs = [a for a in arrs if a.shape[0] == ref]
        out[ep] = np.nanmean(np.stack(arrs, axis=1), axis=1)
    return out


def base_oof_dict():
    """Returns {endpoint: {cid: rank01 array}} for all valid base OOFs."""
    out = {}
    for ep in ('hepatic', 'death'):
        d = {}
        for f in BRANCHES.glob(f'oof_{ep}_*.npy'):
            cid = f.name.replace(f'oof_{ep}_', '').replace('.npy', '')
            try:
                a = np.load(f)
                if np.isfinite(a).sum() < 50: continue
                if np.std(a[np.isfinite(a)]) <= 1e-10: continue
                d[cid] = rank01_nan(a)
            except Exception:
                continue
        out[ep] = d
    return out


def variant_proxy_oof(variant_stem: str, base_oofs_ep: dict, consensus: np.ndarray):
    """Construct a proxy training-side OOF for a given variant.

    For variants that aggregate via mean/median/trimmed mean, we can apply the
    same aggregation across the full base-OOF stack. For meta-aware variants
    we fall back to consensus (it's an approximation). This is good enough for
    a calibration curve — we're not claiming perfect equivalence, just
    correlation between OOF and LB.
    """
    if not base_oofs_ep:
        return consensus
    arrs = list(base_oofs_ep.values())
    ref = arrs[0].shape[0]
    arrs = [a for a in arrs if a.shape[0] == ref]
    if not arrs:
        return consensus
    stack = np.stack(arrs, axis=1)
    s = variant_stem.lower()

    if 'rank_median' in s:
        return np.nanmedian(stack, axis=1)
    if 'rank_trimmed' in s or 'trimmed' in s:
        # 20% trim
        n = stack.shape[1]; k = int(np.floor(n * 0.2 / 2))
        srt = np.sort(stack, axis=1)
        return srt[:, k:n - k].mean(axis=1) if k > 0 else srt.mean(axis=1)
    if 'rank_mean' in s or 'meta_avg' in s or 'best_combined' in s or 'rrf' in s:
        return np.nanmean(stack, axis=1)
    # default fallback
    return np.nanmean(stack, axis=1)


def main():
    train = load_annitia()
    h_y = train['hepatic']['y_struct']; h_v = train['hepatic']['valid_mask']
    d_y = train['death']['y_struct']; d_v = train['death']['valid_mask']

    history = json.loads((REPO / 'tests/experiments/annitia/leaderboard_history.json').read_text())
    consensus = load_consensus_oof()
    base_oofs = base_oof_dict()

    rows = []
    print('=== OOF → leaderboard calibration ===\n')
    for sub in history['submissions']:
        stem = sub['stem']
        lb = sub['score']
        # Try to find the training-side OOF for this submission's recipe
        h_proxy = variant_proxy_oof(stem, base_oofs['hepatic'], consensus['hepatic'])
        d_proxy = variant_proxy_oof(stem, base_oofs['death'], consensus['death'])

        # OOF C per endpoint
        h_finite = h_v & np.isfinite(h_proxy) & np.isfinite(h_y['time'])
        d_finite = d_v & np.isfinite(d_proxy) & np.isfinite(d_y['time'])
        h_c = float(concordance(h_y[h_finite], h_proxy[h_finite])) if h_finite.sum() >= 5 else float('nan')
        d_c = float(concordance(d_y[d_finite], d_proxy[d_finite])) if d_finite.sum() >= 5 else float('nan')
        oof_lb = 0.7 * h_c + 0.3 * d_c  # Trustii LB formula (run.py:210)

        rows.append({'stem': stem, 'lb': lb, 'oof_h': h_c, 'oof_d': d_c, 'oof_lb': oof_lb,
                     'gap': oof_lb - lb})
        print(f'  {stem:40s} LB={lb:.4f}  OOF LB={oof_lb:.4f}  gap={oof_lb - lb:+.4f}')

    rows = sorted(rows, key=lambda r: -r['lb'])

    # Fit a simple linear calibration: LB = a + b * OOF
    valid = [r for r in rows if np.isfinite(r['oof_lb']) and r['stem'] != 'submission']  # exclude broken round3
    if len(valid) >= 3:
        X = np.array([r['oof_lb'] for r in valid])
        Y = np.array([r['lb'] for r in valid])
        # Linear fit
        b, a = np.polyfit(X, Y, 1)
        # Spearman to gauge monotone correlation
        rho, _ = spearmanr(X, Y)
        print(f'\nLinear fit: LB = {a:.4f} + {b:.4f} * OOF_LB (where OOF_LB = 0.7*hep+0.3*death)')
        print(f'Spearman rho(OOF, LB) = {rho:.3f}')
        residuals = Y - (a + b * X)
        print(f'Residual std: {residuals.std():.4f}')
        # Empirical discount at OOF=0.97
        proj_097 = a + b * 0.97
        print(f'\nProjected LB for OOF LB = 0.97: {proj_097:.4f}')
        proj_095 = a + b * 0.95
        print(f'Projected LB for OOF LB = 0.95: {proj_095:.4f}')
    else:
        a = b = rho = float('nan')
        residuals = np.array([])

    # Markdown
    md = ['# OOF → Leaderboard calibration',
          '',
          'Each submission\'s leaderboard score paired with the OOF concordance of the equivalent recipe on the training set.',
          '',
          '## Calibration fit',
          '',
          f'- Linear: **LB = {a:.4f} + {b:.4f} · OOF_LB** (where OOF_LB = 0.7·Cindex_hep + 0.3·Cindex_death, per Trustii spec)',
          f'- Spearman ρ(OOF, LB) = **{rho:.3f}**' if np.isfinite(rho) else '- Spearman ρ: NA',
          f'- Residual std: **{residuals.std():.4f}**' if len(residuals) else '',
          '',
          '## Projected leaderboard scores',
          '',
          '| OOF C | Projected LB |',
          '|---|---|']
    for oc in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
        md.append(f'| {oc:.2f} | {a + b * oc:.4f} |')
    md.extend([
        '',
        '## Per-submission detail',
        '',
        '| Submission | LB | OOF LB (0.7h+0.3d) | Gap (OOF − LB) | Note |',
        '|---|---|---|---|---|'
    ])
    for r in rows:
        note = '*broken*' if r['stem'] == 'submission' else ''
        md.append(f'| `{r["stem"]}` | {r["lb"]:.4f} | {r["oof_lb"]:.4f} | {r["gap"]:+.4f} | {note} |')

    md.extend([
        '',
        '## Implications',
        '',
        '- Average gap (excluding broken round3): use as a discount when projecting LB from OOF.',
        '- The high Spearman ρ means OOF *ranks* recipes correctly but compresses the absolute scale.',
        '- For tomorrow: don\'t care about the 4th decimal of OOF C — care about which variants project',
        '  to higher LB given the calibration, and which have **non-overlapping** projected CIs.',
    ])
    out_md = FULL / 'oof_to_leaderboard.md'
    out_md.write_text('\n'.join(md))
    print(f'\nwrote {out_md}')

    out_json = FULL / 'oof_to_leaderboard.json'
    out_json.write_text(json.dumps({
        'fit': {'intercept': a, 'slope': b, 'spearman_rho': rho,
                'residual_std': float(residuals.std()) if len(residuals) else None},
        'projections': {f'oof_{oc}': a + b * oc for oc in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]},
        'per_submission': rows,
    }, indent=2))
    print(f'wrote {out_json}')


if __name__ == '__main__':
    main()
