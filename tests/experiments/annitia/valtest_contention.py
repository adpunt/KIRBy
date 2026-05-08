"""val_test ensemble contention map.

For each val_test patient compute prediction variance across all submitted
variants. Patients with high contention (variants disagree on rank) are the
ones dragging the LB score — we don't know the true label but we know we're
not confident.

For each pair (i, j) of val_test patients, compute fraction of variants that
order i > j. Pairs near 50/50 are contested. This gives us a "stable rank"
for the consensus and identifies which orderings are model-luck.

Outputs:
  tests/results/annitia/full/valtest_contention_<endpoint>.csv  (per-patient)
  tests/results/annitia/full/valtest_contention.md             (summary)
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

FULL = REPO / 'tests/results/annitia/full'

VARIANTS = [
    'rank_mean', 'rank_median', 'rank_trimmed', 'residual_greedy',
    'uncertainty_route', 'anomaly_blend', 'disagreement_route',
    'cluster_route', 'specialist_override', 'soft_blend_disagreement',
    'anomaly_router_meta', 'disagreement_router_meta', 'anomaly_blend_meta',
    'weighted_anomaly_per_patient', 'cluster_router_meta',
    # post-Round-4 variants if present
    'bootstrap_dropout', 'isotonic_avg', 'nnls_stack', 'holdout_pick',
    # the two recipes that scored on LB
    'meta_avg_aware_weighted', 'meta_avg_grand_all', 'meta_avg_all',
    'meta_avg_weighted', 'meta_avg_top5', 'meta_avg',
]


def rank01(x):
    return rankdata(x) / len(x)


def main():
    dfs = {}
    for v in VARIANTS:
        p = FULL / f'sub_master_{v}.csv'
        if p.exists():
            dfs[v] = pd.read_csv(p)
    present = list(dfs.keys())
    print(f'val_test contention: loaded {len(present)} variant CSVs')
    if len(present) < 5:
        print('not enough variants; aborting')
        sys.exit(1)

    trustii_id = dfs[present[0]]['trustii_id'].values
    n = len(trustii_id)

    for endpoint, col in [('hepatic', 'risk_hepatic_event'), ('death', 'risk_death')]:
        # Stack all variants' rank-01 predictions
        R = np.stack([rank01(dfs[v][col].values) for v in present], axis=1)

        # Per-patient stats
        mean_rank = R.mean(axis=1)
        std_rank = R.std(axis=1)
        # contention score: rank-std normalised by max possible (variance is
        # higher near rank 0.5, so we normalise)
        # IQR of rank across variants — no normalisation needed, just absolute
        # spread
        iqr_rank = np.percentile(R, 75, axis=1) - np.percentile(R, 25, axis=1)

        # Per-patient summary
        df = pd.DataFrame({
            'trustii_id': trustii_id,
            'mean_rank': mean_rank,
            'std_rank': std_rank,
            'iqr_rank': iqr_rank,
            'min_rank': R.min(axis=1),
            'max_rank': R.max(axis=1),
        })
        df['contention'] = df['iqr_rank']
        df['stability'] = 1.0 - df['contention']
        df = df.sort_values('contention', ascending=False)
        out_csv = FULL / f'valtest_contention_{endpoint}.csv'
        df.to_csv(out_csv, index=False)
        print(f'\n{endpoint}: wrote {out_csv}')
        print(f'  mean contention (IQR rank): {df["contention"].mean():.4f}')
        print(f'  patients with IQR > 0.20: {(df["contention"] > 0.20).sum()} / {n}')
        print(f'  patients with IQR > 0.30: {(df["contention"] > 0.30).sum()} / {n}')

        # Top-30 most contested
        print(f'\n  Top 30 most contested {endpoint} patients:')
        print(f'  {"trustii_id":>10} {"mean":>6} {"min":>6} {"max":>6} {"iqr":>6}')
        for _, r in df.head(30).iterrows():
            print(f'  {int(r["trustii_id"]):>10} {r["mean_rank"]:>6.3f} '
                  f'{r["min_rank"]:>6.3f} {r["max_rank"]:>6.3f} {r["iqr_rank"]:>6.3f}')

        # Most-stable patients (consensus)
        print(f'\n  Top 10 most-stable HIGH-risk {endpoint} patients (likely true positives):')
        stable_high = df[df['mean_rank'] > 0.85].nsmallest(10, 'iqr_rank')
        for _, r in stable_high.iterrows():
            print(f'  {int(r["trustii_id"]):>10} mean={r["mean_rank"]:>6.3f} iqr={r["iqr_rank"]:>6.3f}')

        print(f'\n  Top 10 most-stable LOW-risk {endpoint} patients (likely true negatives):')
        stable_low = df[df['mean_rank'] < 0.15].nsmallest(10, 'iqr_rank')
        for _, r in stable_low.iterrows():
            print(f'  {int(r["trustii_id"]):>10} mean={r["mean_rank"]:>6.3f} iqr={r["iqr_rank"]:>6.3f}')

    # Markdown summary
    md = ['# val_test ensemble contention map',
          '',
          f'Aggregated across {len(present)} variant CSVs.',
          '',
          '## Interpretation',
          '',
          '- **High contention (IQR rank > 0.30)**: variants strongly disagree on patient ranking.',
          '  These patients dominate the LB gap. Different recipes will move them in different directions.',
          '- **Stable high-rank**: ensemble consensus — likely true high-risk patients. Submission strategy',
          '  should preserve these rankings.',
          '- **Stable low-rank**: ensemble consensus low-risk. Same.',
          '',
          '## Per-endpoint files',
          '',
          'Detailed per-patient tables: `valtest_contention_hepatic.csv` and `valtest_contention_death.csv`',
          '']

    (FULL / 'valtest_contention.md').write_text('\n'.join(md))
    print(f'\nwrote {FULL}/valtest_contention.md')


if __name__ == '__main__':
    main()
