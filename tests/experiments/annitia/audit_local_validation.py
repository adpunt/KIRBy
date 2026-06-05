"""Combine bootstrap_candidate_ci.csv + holdout_fusion_scores.csv + LB history
into AUDIT_LOCAL_VALIDATION.md.

Reports:
  - Per-candidate side-by-side: bootstrap mean ± CI vs holdout LB
  - Spearman ρ between bootstrap_OOF_LB and holdout_LB
  - Top-10 by each method
  - Recommendation: which method's ranking should we trust more?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
FULL = REPO / 'tests/results/annitia/full'
HIST = REPO / 'tests/experiments/annitia/leaderboard_history.json'
OUT = REPO / 'AUDIT_LOCAL_VALIDATION.md'


def main():
    boot = pd.read_csv(FULL / 'bootstrap_candidate_ci.csv')
    hold = pd.read_csv(FULL / 'holdout_fusion_scores.csv')

    df = boot.merge(hold, on='variant', how='outer')

    # LB history (when available)
    try:
        h = json.loads(HIST.read_text())
        lb = {s['stem'].replace('sub_master_', ''): s['score']
              for s in h['submissions'] if 'score' in s}
    except Exception:
        lb = {}
    df['known_LB'] = df['variant'].map(lb)

    # Spearman boot vs hold
    pair = df[['mean_weighted_LB', 'holdout_LB']].dropna()
    if len(pair) >= 3:
        rho_bh, p_bh = spearmanr(pair['mean_weighted_LB'], pair['holdout_LB'])
    else:
        rho_bh, p_bh = float('nan'), float('nan')

    pair_lb_b = df[['mean_weighted_LB', 'known_LB']].dropna()
    rho_blb = spearmanr(pair_lb_b['mean_weighted_LB'], pair_lb_b['known_LB']).correlation if len(pair_lb_b) >= 3 else float('nan')
    pair_lb_h = df[['holdout_LB', 'known_LB']].dropna()
    rho_hlb = spearmanr(pair_lb_h['holdout_LB'], pair_lb_h['known_LB']).correlation if len(pair_lb_h) >= 3 else float('nan')

    # Differentiation: how many UNIQUE values per metric
    unique_boot = df['mean_weighted_LB'].round(4).nunique()
    unique_hold = df['holdout_LB'].round(4).nunique()

    # Sorts
    top_boot = df.sort_values('mean_weighted_LB', ascending=False).head(10)
    top_hold = df.sort_values('holdout_LB', ascending=False).head(10)

    # Markdown
    n_boot = int(df['mean_weighted_LB'].notna().sum())
    n_hold = int(df['holdout_LB'].notna().sum())

    md = ['# Local Validation Audit (Bootstrap OOF + Holdout 10%)',
          '',
          f'- Total variants: **{len(df)}**',
          f'- Bootstrap-scored: **{n_boot}**, holdout-scored: **{n_hold}**',
          f'- Unique bootstrap mean LB values (4 dp): **{unique_boot}** / {n_boot}',
          f'- Unique holdout LB values       (4 dp): **{unique_hold}** / {n_hold}',
          '',]
    if n_boot < len(df):
        md.append(f'**Note:** {len(df) - n_boot} variants lack bootstrap scores '
                  '(bootstrap CSV was generated incrementally; resume the script '
                  '`tests/experiments/annitia/bootstrap_candidate_ci.py` to fill '
                  'in remaining rows).')
        md.append('')
    md += [
          '## Spearman correlations',
          '',
          f'- ρ(bootstrap_mean_LB, holdout_LB) = **{rho_bh:.3f}** (p={p_bh:.3g}, n={len(pair)})',
          f'- ρ(bootstrap_mean_LB, known_LB)   = **{rho_blb:.3f}** (n={len(pair_lb_b)})',
          f'- ρ(holdout_LB,        known_LB)   = **{rho_hlb:.3f}** (n={len(pair_lb_h)})',
          '',
          '## Differentiation diagnosis',
          '']
    if unique_boot <= 5 or unique_hold <= 5:
        md.append('**WARNING:** at least one method assigns <=5 unique values across all '
                  '70 variants. The variant_proxy_oof fallback collapses non-deterministic '
                  'fusion variants (NNLS, isotonic, anomaly routing, cluster routing) onto '
                  'a single mean-of-base-OOFs prediction. Local validation is therefore '
                  '**fundamentally unable** to differentiate those candidates — the '
                  'distinguishing fusion logic is invisible at the proxy level.')
    else:
        md.append('Both methods assign distinct scores across most variants — local '
                  'validation has signal to spend.')
    md.extend(['',
               '## Top-10 by bootstrap mean LB',
               '',
               '| Variant | mean LB | std | 95% CI | hold LB | known LB |',
               '|---|---|---|---|---|---|'])
    for _, r in top_boot.iterrows():
        ci = f'[{r.ci_weighted_low:.4f}, {r.ci_weighted_high:.4f}]' if pd.notna(r.ci_weighted_low) else '-'
        h = f'{r.holdout_LB:.4f}' if pd.notna(r.holdout_LB) else '-'
        klb = f'{r.known_LB:.4f}' if pd.notna(r.known_LB) else '-'
        ms = f'{r.mean_weighted_LB:.4f}' if pd.notna(r.mean_weighted_LB) else '-'
        sd = f'{r.std_weighted_LB:.4f}' if pd.notna(r.std_weighted_LB) else '-'
        md.append(f'| `{r.variant}` | {ms} | {sd} | {ci} | {h} | {klb} |')

    md.extend(['',
               '## Top-10 by holdout LB',
               '',
               '| Variant | holdout LB | hep | death | bootstrap mean | known LB |',
               '|---|---|---|---|---|---|'])
    for _, r in top_hold.iterrows():
        bm = f'{r.mean_weighted_LB:.4f}' if pd.notna(r.mean_weighted_LB) else '-'
        klb = f'{r.known_LB:.4f}' if pd.notna(r.known_LB) else '-'
        ml = f'{r.holdout_LB:.4f}' if pd.notna(r.holdout_LB) else '-'
        hh = f'{r.holdout_C_hep:.4f}' if pd.notna(r.holdout_C_hep) else '-'
        dd = f'{r.holdout_C_death:.4f}' if pd.notna(r.holdout_C_death) else '-'
        md.append(f'| `{r.variant}` | {ml} | {hh} | {dd} | {bm} | {klb} |')

    md.extend(['',
               '## Side-by-side (all variants, sorted by bootstrap mean LB)',
               '',
               '| Variant | boot LB ± std | 95% CI | hold LB | known LB |',
               '|---|---|---|---|---|'])
    for _, r in df.sort_values('mean_weighted_LB', ascending=False).iterrows():
        if pd.notna(r.mean_weighted_LB):
            ci = f'[{r.ci_weighted_low:.4f}, {r.ci_weighted_high:.4f}]'
            bs = f'{r.mean_weighted_LB:.4f} ± {r.std_weighted_LB:.4f}'
        else:
            ci = '-'; bs = '-'
        h = f'{r.holdout_LB:.4f}' if pd.notna(r.holdout_LB) else '-'
        klb = f'{r.known_LB:.4f}' if pd.notna(r.known_LB) else '-'
        md.append(f'| `{r.variant}` | {bs} | {ci} | {h} | {klb} |')

    md.extend(['',
               '## Recommendation',
               ''])
    if rho_bh < 0.3 or not np.isfinite(rho_bh):
        md.append('Bootstrap and holdout disagree (ρ<0.3). Treat both as weak local '
                  'signal and lean on the LB-constraint reverse-engineering instead.')
    elif unique_boot <= 5 and unique_hold <= 5:
        md.append('Both methods are degenerate — the proxy OOF cannot differentiate '
                  'data-driven fusion variants. Use LB constraints + endpoint-level '
                  'submissions to triangulate.')
    else:
        # Pick whichever method correlates better with known_LB if available
        if np.isfinite(rho_blb) and np.isfinite(rho_hlb):
            if rho_hlb > rho_blb + 0.05:
                md.append(f'**Trust holdout ranking more** (ρ_hold,LB={rho_hlb:.3f} vs '
                          f'ρ_boot,LB={rho_blb:.3f}). The holdout 10% is fitter-blind by '
                          'construction; bootstrap re-uses the same OOF rows the fusion saw.')
            elif rho_blb > rho_hlb + 0.05:
                md.append(f'**Trust bootstrap ranking more** (ρ_boot,LB={rho_blb:.3f} vs '
                          f'ρ_hold,LB={rho_hlb:.3f}). Bootstrap leverages the full training '
                          'set; the 10% holdout has too few events.')
            else:
                md.append('Both methods correlate similarly with known LB. Use the '
                          '**intersection of top-10**: candidates that rank in both '
                          'top-10 lists are the safest picks.')
        else:
            md.append('No known-LB anchor for cross-validation. Take the **intersection '
                      'of top-10 from both methods** — variants that rank highly under '
                      'both criteria are the lower-risk picks.')

    md.append('')
    OUT.write_text('\n'.join(md))
    print(f'wrote {OUT} ({len(md)} lines)')


if __name__ == '__main__':
    main()
