#!/usr/bin/env python3
"""Combine per-branch ANNITIA submissions into a master submission (Phase E).

Each branch produces its own annitia_submission_<label>.csv. This script
reads them all and applies one of five fusion modes to produce a master
ranking per endpoint.

Modes:
    equal       — uniform mean of within-branch ranks (legacy default).
    cv_weighted — weighted rank-mean; --weights <w1 w2 ...> required.
    single      — pick the branch with the highest CV score; --cv-scores or
                  --search-jsons (one per input) required.
    rrf         — reciprocal-rank fusion (k=60 default; --rrf-k to override).
    stack_meta  — Coxnet on rank-transformed OOF branch risks. Requires
                  per-branch OOF .npy and the training y_struct (loaded
                  from the ANNITIA dataset). --oof-inputs <a.npy b.npy ...>
                  must align positionally with --inputs.

Why rank-average rather than score-average:
- Different models live on incompatible scales (Cox linear predictor vs RSF
  cumulative hazard vs landmark-binary probability). Rank-average is the
  natural fusion operator under the C-index metric.

Usage:
    # Legacy / equal mode (back-compat)
    python combine_branches.py \\
        --inputs run_a/sub.csv run_b/sub.csv \\
        --output combined.csv

    # Weighted (manual)
    python combine_branches.py --mode cv_weighted \\
        --inputs a.csv b.csv --weights 2.0 1.0 --output combined.csv

    # Pick best by CV from per-branch search JSONs
    python combine_branches.py --mode single \\
        --inputs a.csv b.csv \\
        --search-jsons a/annitia_hepatic_search.json b/annitia_hepatic_search.json \\
        --output combined.csv

    # Reciprocal-rank fusion
    python combine_branches.py --mode rrf --rrf-k 60 \\
        --inputs a.csv b.csv --output combined.csv

    # Coxnet meta-stacking on OOF
    python combine_branches.py --mode stack_meta \\
        --inputs a.csv b.csv \\
        --oof-inputs a/oof/hepatic/<id>.npy b/oof/hepatic/<id>.npy \\
        --output combined.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from experiments.annitia.fusion import (  # noqa: E402
    SUPPORTED_MODES,
    fuse_cv_weighted,
    fuse_equal,
    fuse_rrf,
    fuse_single,
    fuse_stack_meta,
)


def _rank(x: np.ndarray) -> np.ndarray:
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)
    return ranks


def _load_cv_scores(args, n_inputs: int):
    """Resolve cv_scores from either --cv-scores or --search-jsons.

    --cv-scores: explicit list of floats, one per branch input.
    --search-jsons: list of search JSON paths; the script reads
        each `best.mean_cindex` field. Must align with --inputs by index.
    """
    if args.cv_scores is not None:
        if len(args.cv_scores) != n_inputs:
            sys.exit(f'--cv-scores ({len(args.cv_scores)}) must match '
                     f'--inputs ({n_inputs})')
        return np.asarray(args.cv_scores, dtype=np.float64)
    if args.search_jsons is not None:
        if len(args.search_jsons) != n_inputs:
            sys.exit(f'--search-jsons ({len(args.search_jsons)}) must match '
                     f'--inputs ({n_inputs})')
        out = []
        for p in args.search_jsons:
            data = json.loads(Path(p).read_text())
            best = data.get('best')
            if best is None or 'mean_cindex' not in best:
                sys.exit(f'{p} has no usable `best.mean_cindex` field')
            out.append(float(best['mean_cindex']))
        return np.asarray(out, dtype=np.float64)
    sys.exit('--mode single requires --cv-scores or --search-jsons')


def _load_oof_for_endpoint(oof_paths, endpoint: str):
    """Load OOF .npy files for one endpoint and return a list of length-N
    risk vectors. Each path is interpreted positionally relative to its
    corresponding --inputs entry (no per-endpoint disambiguation here —
    caller must supply the right .npy per branch).
    """
    out = []
    for p in oof_paths:
        arr = np.load(p)
        out.append(np.asarray(arr, dtype=np.float64))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', nargs='+', required=True,
                    help='Per-branch submission CSV paths.')
    ap.add_argument('--output', required=True,
                    help='Master submission CSV path.')
    ap.add_argument('--mode', choices=list(SUPPORTED_MODES),
                    default='equal',
                    help='Fusion mode (default: equal).')
    ap.add_argument('--weights', type=float, nargs='+', default=None,
                    help='Per-branch weights (cv_weighted mode).')
    ap.add_argument('--cv-scores', type=float, nargs='+', default=None,
                    help='Per-branch CV C-indices (single mode).')
    ap.add_argument('--search-jsons', nargs='+', default=None,
                    help='Per-branch search JSON paths; reads best.mean_cindex '
                         '(single mode alternative to --cv-scores).')
    ap.add_argument('--rrf-k', type=float, default=60.0,
                    help='RRF smoothing constant (default 60).')
    ap.add_argument('--oof-inputs-hepatic', nargs='+', default=None,
                    help='Per-branch OOF .npy paths for hepatic '
                         '(stack_meta mode).')
    ap.add_argument('--oof-inputs-death', nargs='+', default=None,
                    help='Per-branch OOF .npy paths for death (stack_meta mode).')
    ap.add_argument('--data-dir', default=None,
                    help='ANNITIA data dir (stack_meta needs y_struct from '
                         'load_annitia).')
    ap.add_argument('--filename', default='train.csv')
    ap.add_argument('--l1-ratio', type=float, default=0.5,
                    help='stack_meta Coxnet elastic-net mix.')
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            sys.exit(f'Missing: {p}')

    dfs = [pd.read_csv(p) for p in paths]
    n = len(dfs[0])
    for p, d in zip(paths, dfs):
        if len(d) != n:
            sys.exit(f'{p} length {len(d)} != {n}')
        if not (d['trustii_id'].to_numpy() == dfs[0]['trustii_id'].to_numpy()).all():
            sys.exit(f'{p} trustii_id ordering differs from {paths[0]}')

    out = pd.DataFrame({'trustii_id': dfs[0]['trustii_id'].to_numpy()})

    # Per-endpoint fusion. stack_meta needs y_struct so it's the only mode
    # that crosses the data-loading boundary.
    if args.mode == 'stack_meta':
        from kirby.datasets.annitia_maslg import load_annitia  # noqa: E402
        data = load_annitia(data_dir=args.data_dir, filename=args.filename)
        endpoints = {
            'hepatic': ('risk_hepatic_event', args.oof_inputs_hepatic),
            'death':   ('risk_death',         args.oof_inputs_death),
        }
        for endpoint, (col, oof_paths) in endpoints.items():
            if oof_paths is None:
                sys.exit(f'--mode stack_meta requires --oof-inputs-{endpoint}')
            if len(oof_paths) != len(paths):
                sys.exit(
                    f'--oof-inputs-{endpoint} ({len(oof_paths)}) must match '
                    f'--inputs ({len(paths)})')
            test_scores = [d[col].to_numpy(dtype=np.float64) for d in dfs]
            oof_scores = _load_oof_for_endpoint(oof_paths, endpoint)
            y = data[endpoint]['y_struct']
            valid = data[endpoint]['valid_mask']
            fused = fuse_stack_meta(
                oof_scores, y, test_scores, valid_mask=valid,
                l1_ratio=args.l1_ratio)
            out[col] = fused
    else:
        for col in ('risk_hepatic_event', 'risk_death'):
            test_scores = [d[col].to_numpy(dtype=np.float64) for d in dfs]
            if args.mode == 'equal':
                fused = fuse_equal(test_scores)
            elif args.mode == 'cv_weighted':
                if args.weights is None:
                    sys.exit('--mode cv_weighted requires --weights')
                if len(args.weights) != len(paths):
                    sys.exit(
                        f'--weights ({len(args.weights)}) must match '
                        f'--inputs ({len(paths)})')
                fused = fuse_cv_weighted(test_scores, args.weights)
            elif args.mode == 'single':
                cv = _load_cv_scores(args, len(paths))
                fused = fuse_single(test_scores, cv)
            elif args.mode == 'rrf':
                fused = fuse_rrf(test_scores, k=args.rrf_k)
            else:
                sys.exit(f'unknown mode {args.mode}')
            out[col] = fused

    out.to_csv(args.output, index=False)
    print(f'Combined {len(paths)} submissions ({args.mode}) -> {args.output}')
    print(f'  trustii_id range: {out["trustii_id"].min()}..{out["trustii_id"].max()}')
    for col in ('risk_hepatic_event', 'risk_death'):
        print(f'  {col}: range [{out[col].min():.2f}, {out[col].max():.2f}]')
    if args.mode == 'cv_weighted' and args.weights is not None:
        print('  weights:', dict(zip([p.name for p in paths], args.weights)))


if __name__ == '__main__':
    main()
