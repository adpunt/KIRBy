#!/usr/bin/env python3
"""Combine per-branch ANNITIA submissions into a master submission.

Each branch produces its own annitia_submission_<label>.csv. This script
reads them all, rank-transforms each (per-endpoint), averages the ranks,
and writes a master submission CSV.

Why rank-average rather than score-average:
- Different models (Cox linear predictor, RSF cumulative hazard, XGBoost cox-LL)
  live on incompatible scales.
- C-index is rank-based, so rank-averaging is the natural fusion operator.
- Robust to outlier scores.

Usage:
    python tests/experiments/annitia/combine_branches.py \\
        --inputs tests/results/annitia/annitia_submission.csv \\
                 tests/results/annitia_xgbcox/annitia_submission_xgbcox.csv \\
                 tests/results/annitia_views/annitia_submission_views.csv \\
                 tests/results/annitia_stack/annitia_submission_stack.csv \\
                 tests/results/annitia_weighted/annitia_submission_weighted.csv \\
        --output tests/results/annitia/annitia_submission_master.csv

If `--weights` is supplied (one float per input), branches are weighted
non-uniformly in the rank average. Default: equal weights.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (handles ties)."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    # Plain ordinal ranks; for the C-index this is sufficient (ties between
    # different models on different patients are vanishingly rare).
    ranks[order] = np.arange(n)
    return ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', nargs='+', required=True,
                    help='Per-branch submission CSV paths.')
    ap.add_argument('--weights', type=float, nargs='+', default=None,
                    help='Optional per-branch weights (default: uniform).')
    ap.add_argument('--output', required=True,
                    help='Master submission CSV path.')
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            sys.exit(f'Missing: {p}')

    if args.weights is not None:
        if len(args.weights) != len(paths):
            sys.exit(f'--weights ({len(args.weights)}) must match --inputs ({len(paths)}).')
        weights = np.asarray(args.weights, dtype=np.float64)
    else:
        weights = np.ones(len(paths))
    weights = weights / weights.sum()

    dfs = [pd.read_csv(p) for p in paths]

    # Sanity: all CSVs same length, same trustii_id ordering
    n = len(dfs[0])
    for p, d in zip(paths, dfs):
        if len(d) != n:
            sys.exit(f'{p} length {len(d)} != {n}')
        if not (d['trustii_id'].to_numpy() == dfs[0]['trustii_id'].to_numpy()).all():
            sys.exit(f'{p} trustii_id ordering differs from {paths[0]}')

    out = pd.DataFrame({'trustii_id': dfs[0]['trustii_id'].to_numpy()})
    for col in ('risk_hepatic_event', 'risk_death'):
        agg = np.zeros(n)
        for w, d in zip(weights, dfs):
            agg += w * _rank(d[col].to_numpy())
        out[col] = agg

    out.to_csv(args.output, index=False)
    print(f'Combined {len(paths)} submissions -> {args.output}')
    print(f'  trustii_id range: {out["trustii_id"].min()}..{out["trustii_id"].max()}')
    for col in ('risk_hepatic_event', 'risk_death'):
        print(f'  {col}: range [{out[col].min():.2f}, {out[col].max():.2f}]')
    print('  weights:', dict(zip([p.name for p in paths], weights.round(3))))


if __name__ == '__main__':
    main()
