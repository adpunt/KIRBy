"""V10 — Permutation null distribution for ΔC between fusion methods.

For each pair (A, B) of methods, compute observed ΔC = C(A) − C(B) and the
null distribution under random per-patient swaps. Reports p-values per pair
and per metric (Harrell + Uno).

Inputs: tests/results/annitia/eval/sub_*.csv (val_test predictions per method)
        + the OOFs of each method (from competition_methods_eval.py output).

Run after competition_methods_eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'src'))

from kirby.datasets.annitia_maslg import load_annitia  # noqa: E402

from tests.experiments.annitia.competition_methods_eval import (  # noqa: E402
    METHODS, anchor_rank_mean, collect_pool, carve_holdout,
    f01_caruana_bagged, f02_geomean_rank, f03_hodges_lehmann,
    f04_bates_granger, f05_stacking_nested, f06_diversity_penalized,
    f09_trimmed_mean, harrell_c, uno_ipcw_c,
)


def permutation_paired_pvalue(p_A, p_B, y_struct, valid, R=1000, seed=0):
    """For paired predictions p_A, p_B on the same patients (subset by `valid`),
    return observed ΔC and two-sided permutation p-value under random swap."""
    rng = np.random.default_rng(seed)
    valid_idx = np.where(valid & np.isfinite(p_A) & np.isfinite(p_B))[0]
    if len(valid_idx) < 30:
        return float('nan'), float('nan')
    sub_y = y_struct[valid_idx]
    a = p_A[valid_idx]
    b = p_B[valid_idx]
    from sksurv.metrics import concordance_index_censored
    c_a = concordance_index_censored(sub_y['event'], sub_y['time'], a)[0]
    c_b = concordance_index_censored(sub_y['event'], sub_y['time'], b)[0]
    obs = c_a - c_b
    null = []
    for _ in range(R):
        mask = rng.integers(0, 2, len(valid_idx)).astype(bool)
        a2 = np.where(mask, b, a)
        b2 = np.where(mask, a, b)
        c1 = concordance_index_censored(sub_y['event'], sub_y['time'], a2)[0]
        c2 = concordance_index_censored(sub_y['event'], sub_y['time'], b2)[0]
        null.append(c1 - c2)
    null = np.array(null)
    p = float(np.mean(np.abs(null) >= abs(obs)))
    return float(obs), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=int, default=1000)
    ap.add_argument('--holdout-seed', type=int, default=2026)
    ap.add_argument('--holdout-frac', type=float, default=0.15)
    args = ap.parse_args()

    train = load_annitia()
    res_root = _REPO / 'tests' / 'results' / 'annitia'
    pool_dirs = [res_root / p for p in
                 ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam',
                  'research', 'research_v2')
                 if (res_root / p).exists()]
    n_total = len(train['hepatic']['y_struct'])

    print(f'Permutation null with R={args.R}')
    out_lines = ['# Permutation null distribution (V10)', '',
                 f'R = {args.R} permutations per pair', '']

    for endpoint in ('hepatic', 'death'):
        print(f'\n=== {endpoint} ===')
        out_lines.append(f'## {endpoint}\n')
        y_struct = train[endpoint]['y_struct']
        valid_mask = train[endpoint]['valid_mask']
        recs = collect_pool(pool_dirs, endpoint, n_total)
        train_idx, holdout_idx = carve_holdout(y_struct, valid_mask,
                                               frac=args.holdout_frac,
                                               seed=args.holdout_seed)
        holdout_mask = np.isin(np.arange(n_total), holdout_idx)
        # Build OOFs for each method on this single seed
        method_oof = {}
        for method in METHODS:
            try:
                if method is f01_caruana_bagged:
                    r = method(recs, y_struct, train_idx, holdout_idx, n_bags=10)
                else:
                    r = method(recs, y_struct, train_idx, holdout_idx)
            except Exception as e:
                print(f'  {method.__name__}: {e}')
                continue
            if r is None:
                continue
            method_oof[r['name']] = r['oof']

        names = list(method_oof.keys())
        # Use anchor as the comparison reference (V08 anchor)
        if 'anchor_rank_mean' not in names:
            print('  no anchor — skipping')
            continue
        anchor = method_oof['anchor_rank_mean']
        out_lines.append('| method | ΔC vs anchor (hold) | p-value |')
        out_lines.append('|---|---|---|')
        for name in names:
            if name == 'anchor_rank_mean':
                continue
            obs, p = permutation_paired_pvalue(
                method_oof[name], anchor, y_struct, holdout_mask, R=args.R, seed=0)
            sig = '**SIG**' if (np.isfinite(p) and p < 0.05) else ''
            out_lines.append(f'| {name} | {obs:+.4f} | {p:.3f} {sig} |')
            print(f'  {name:25s} ΔC={obs:+.4f} p={p:.3f} {sig}')
        out_lines.append('')

    out_path = res_root / 'eval' / 'permutation_null.md'
    out_path.write_text('\n'.join(out_lines))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
