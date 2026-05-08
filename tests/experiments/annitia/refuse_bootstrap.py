"""Bootstrap-dropout fusion: random-subset averaging.

For M iterations, average a random subset of variant CSVs. Reduces variance
relative to deterministic averaging — like dropout for ensembles.

Reads: tests/results/annitia/full/sub_master_<variant>.csv (15 existing)
Writes: tests/results/annitia/full/sub_master_bootstrap_dropout.csv

Usage:
    python tests/experiments/annitia/refuse_bootstrap.py
    python tests/experiments/annitia/refuse_bootstrap.py --M 50 --keep-frac 0.7
"""
from __future__ import annotations

import argparse
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
]


def rank01(x):
    return rankdata(x) / len(x)


def main(M: int = 50, keep_frac: float = 0.7, seed: int = 42):
    rng = np.random.default_rng(seed)
    dfs = {}
    for v in VARIANTS:
        p = FULL / f'sub_master_{v}.csv'
        if p.exists():
            dfs[v] = pd.read_csv(p)
    present = list(dfs.keys())
    if len(present) < 5:
        print(f"only {len(present)} variants present; need ≥5")
        sys.exit(1)
    print(f"bootstrap_dropout: {len(present)} variants, M={M}, keep_frac={keep_frac}")

    trustii_id = dfs[present[0]]['trustii_id'].values
    n = len(trustii_id)
    H_ranks = np.stack([rank01(dfs[v]['risk_hepatic_event'].values) for v in present], axis=1)
    D_ranks = np.stack([rank01(dfs[v]['risk_death'].values) for v in present], axis=1)
    n_keep = max(3, int(np.ceil(len(present) * keep_frac)))

    H_avg = np.zeros(n)
    D_avg = np.zeros(n)
    for _ in range(M):
        idx = rng.choice(len(present), size=n_keep, replace=False)
        H_avg += H_ranks[:, idx].mean(axis=1)
        D_avg += D_ranks[:, idx].mean(axis=1)
    H_avg /= M
    D_avg /= M

    out = pd.DataFrame({
        'trustii_id': trustii_id,
        'risk_hepatic_event': H_avg,
        'risk_death': D_avg,
    })
    dest = FULL / 'sub_master_bootstrap_dropout.csv'
    out.to_csv(dest, index=False)
    print(f'wrote {dest}')

    # quick sanity: spearman vs the 0.9302 anchor
    anc = FULL / 'sub_master_meta_avg_aware_weighted.csv'
    if anc.exists():
        from scipy.stats import spearmanr
        a = pd.read_csv(anc)
        rho_h, _ = spearmanr(out['risk_hepatic_event'], a['risk_hepatic_event'])
        rho_d, _ = spearmanr(out['risk_death'], a['risk_death'])
        print(f'  spearman vs 0.9302 anchor: hep={rho_h:.4f} death={rho_d:.4f}')

    summary = {
        'M': M, 'keep_frac': keep_frac, 'n_variants_pool': len(present),
        'n_keep_per_iter': n_keep,
    }
    (FULL / 'refuse_bootstrap_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=50)
    ap.add_argument('--keep-frac', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args.M, args.keep_frac, args.seed)
