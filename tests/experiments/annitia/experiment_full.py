#!/usr/bin/env python3
"""ANNITIA experiment FULL — runs experiment_be.py then experiment_cd.py and
fuses the union of OOF outputs with stack_meta for the final submission.

Estimated runtime LOCALLY: 60-120 min (sum of B/E + C/D + final fusion).

Use this when you want everything in one shot. It writes a single
sub_master.csv that fuses the strongest survivors across BOTH scripts.

Outputs:
    tests/results/annitia/be/   (from experiment_be.py)
    tests/results/annitia/cd/   (from experiment_cd.py)
    tests/results/annitia/full/
        sub_master.csv          — final fused submission
        cv_summary.json         — combined CV summary
        oof_<endpoint>_<id>.npy — copies of pooled OOFs
        val_<endpoint>_<id>.npy — copies of pooled val predictions

Usage:
    python tests/experiments/annitia/experiment_full.py
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.survival import concordance
from experiments.annitia.fusion import (
    fuse_equal, fuse_cv_weighted, fuse_single, fuse_rrf, StackMetaFusion)


def _convert(o):
    if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_convert(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o


def _run(script, extra_args):
    cmd = ['python', str(_REPO / 'tests' / 'experiments' / 'annitia' / script)] + extra_args
    print(f'\n>>> {" ".join(cmd)}\n')
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f'WARNING: {script} returned {r.returncode}; continuing')


def _collect_runs(*pool_dirs: Path):
    """Collate per-condition records + OOF/val .npy paths across all stage dirs."""
    runs = {'hepatic': [], 'death': []}
    for d in pool_dirs:
        cv_path = d / 'cv_summary.json'
        if not cv_path.exists():
            print(f'  (skipping missing {cv_path})')
            continue
        sm = json.loads(cv_path.read_text())
        for endpoint in ('hepatic', 'death'):
            for r in sm['per_condition'].get(endpoint, []):
                cid = r['config_id']
                oof_p = d / f'oof_{endpoint}_{cid}.npy'
                val_p = d / f'val_{endpoint}_{cid}.npy'
                if not (oof_p.exists() and val_p.exists()):
                    continue
                runs[endpoint].append({
                    'config_id': f'{d.name}__{cid}',
                    'mean_cindex': r['mean_cindex'],
                    'oof_path': str(oof_p),
                    'val_path': str(val_p),
                    'source': d.name,
                    'label': r.get('label', cid),
                })
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--be-args', default='', help='Extra args forwarded to experiment_be.py')
    ap.add_argument('--cd-args', default='', help='Extra args forwarded to experiment_cd.py')
    ap.add_argument('--top-k', type=int, default=6,
                    help='Top-K survivors per endpoint to fuse.')
    ap.add_argument('--output-dir', default='tests/results/annitia/full')
    ap.add_argument('--skip-be', action='store_true',
                    help='Skip the experiment_be.py invocation (use existing outputs).')
    ap.add_argument('--skip-cd', action='store_true',
                    help='Skip the experiment_cd.py invocation.')
    ap.add_argument('--extra-pools', default='hep_boost,branches,external',
                    help='Comma-separated subdirs under tests/results/annitia/ '
                         'to also pool into the final fusion '
                         '(default: hep_boost,branches,external).')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    annitia_results = _REPO / 'tests' / 'results' / 'annitia'
    be_dir = annitia_results / 'be'
    cd_dir = annitia_results / 'cd'
    extra_pools = [annitia_results / p.strip() for p in args.extra_pools.split(',') if p.strip()]
    t_start = time.time()

    val_arg = ['--validation-csv', args.validation_csv]
    if not args.skip_be:
        _run('experiment_be.py', val_arg + (args.be_args.split() if args.be_args else []))
    if not args.skip_cd:
        _run('experiment_cd.py', val_arg + (args.cd_args.split() if args.cd_args else []))

    pool_dirs = [be_dir, cd_dir] + [d for d in extra_pools if d.exists()]
    print(f'\n=== POOLING runs from {[d.name for d in pool_dirs]} ===')
    runs = _collect_runs(*pool_dirs)
    for endpoint in ('hepatic', 'death'):
        print(f'  {endpoint}: {len(runs[endpoint])} pooled conditions')

    # Load training labels
    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)

    # ------------------------------------------------------------------
    # Final fusion: sweep top-K x mode, with cross-endpoint stacking.
    # ------------------------------------------------------------------
    K_SWEEP = [3, 5, 10, 20, 40, 'all']
    MODES_NO_STACK = ('top1', 'equal', 'cv_weighted', 'single', 'rrf')
    # Pre-collect per-endpoint sorted records so we can build cross-endpoint
    # features.
    sorted_recs = {}
    for endpoint in ('hepatic', 'death'):
        rs = [r for r in runs[endpoint] if np.isfinite(r['mean_cindex'])]
        rs.sort(key=lambda r: -r['mean_cindex'])
        sorted_recs[endpoint] = rs

    def _select(recs, k):
        if k == 'all':
            return recs
        return recs[:k]

    fusion = {}
    master = pd.DataFrame({'trustii_id': val['trustii_id']})
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death',   'risk_death')):
        other = 'death' if endpoint == 'hepatic' else 'hepatic'
        own_all = sorted_recs[endpoint]
        other_all = sorted_recs[other]
        if not own_all:
            print(f'  {endpoint}: no usable conditions, filling zeros')
            master[col] = 0.0
            continue
        print(f'\n  === {endpoint}: pool of {len(own_all)} conditions, '
              f'cross-pool ({other}) of {len(other_all)} ===')

        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']

        # Trials we'll score: list of (K_self, K_cross, mode, oof_vec, val_vec)
        trials = []   # results
        per_mode_save = {}  # mode_label -> val vec (for sub_*.csv)

        for K in K_SWEEP:
            recs_self = _select(own_all, K)
            if not recs_self:
                continue
            oof_list = [np.load(r['oof_path']) for r in recs_self]
            val_list = [np.load(r['val_path']) for r in recs_self]
            cv_list  = [r['mean_cindex'] for r in recs_self]
            weights = np.maximum(np.asarray(cv_list) - 0.5, 0.0)
            if weights.sum() == 0: weights = np.ones(len(recs_self))
            n_self = len(recs_self)

            # 5 simple non-stacking modes (no meta-learner) + stack_meta + xe
            mode_vals_oof = {
                'top1':        oof_list[int(np.argmax(cv_list))],
                'equal':       fuse_equal(oof_list),
                'cv_weighted': fuse_cv_weighted(oof_list, weights),
                'single':      fuse_single(oof_list, cv_list),
                'rrf':         fuse_rrf(oof_list, k=60.0),
            }
            mode_vals_val = {
                'top1':        val_list[int(np.argmax(cv_list))],
                'equal':       fuse_equal(val_list),
                'cv_weighted': fuse_cv_weighted(val_list, weights),
                'single':      fuse_single(val_list, cv_list),
                'rrf':         fuse_rrf(val_list, k=60.0),
            }

            # stack_meta (self only)
            try:
                sm = StackMetaFusion(l1_ratio=0.5)
                sm.fit(oof_list, y, valid_mask=valid)
                mode_vals_oof['stack_meta'] = sm.transform(oof_list)
                mode_vals_val['stack_meta'] = sm.transform(val_list)
            except Exception as e:
                print(f'    K={K} stack_meta failed: {e}')

            # stack_meta_xe — append the OTHER endpoint's top-K branches as
            # extra features so death-info informs hep and vice versa.
            for K_cross in (K,) if other_all else ():
                recs_cross = _select(other_all, K_cross)
                if not recs_cross:
                    continue
                oof_xe = oof_list + [np.load(r['oof_path']) for r in recs_cross]
                val_xe = val_list + [np.load(r['val_path']) for r in recs_cross]
                try:
                    sm_xe = StackMetaFusion(l1_ratio=0.5)
                    sm_xe.fit(oof_xe, y, valid_mask=valid)
                    mode_vals_oof['stack_meta_xe'] = sm_xe.transform(oof_xe)
                    mode_vals_val['stack_meta_xe'] = sm_xe.transform(val_xe)
                except Exception as e:
                    print(f'    K={K} stack_meta_xe failed: {e}')

            for mode, ov in mode_vals_oof.items():
                vv = mode_vals_val[mode]
                try:
                    c = concordance(y[valid], ov[valid])
                except Exception:
                    c = float('nan')
                trials.append({'K': K, 'mode': mode, 'oof_c': float(c),
                               'val': vv, 'n_branches': n_self})
                tag = f'K{K}_{mode}'
                per_mode_save[tag] = vv
                print(f'    K={str(K):>3} mode={mode:14s} OOF-train C={c:.4f}')

        # Pick best overall by OOF C
        finite_trials = [t for t in trials if np.isfinite(t['oof_c'])]
        if not finite_trials:
            print(f'  {endpoint}: every trial NaN, falling back to zeros')
            master[col] = 0.0
            continue
        best = max(finite_trials, key=lambda t: t['oof_c'])
        # Also report the best NON-stacking trial so user sees the no-stack answer
        no_stack_trials = [t for t in finite_trials if t['mode'] in MODES_NO_STACK]
        best_no_stack = max(no_stack_trials, key=lambda t: t['oof_c']) if no_stack_trials else None
        print(f'\n    BEST overall : K={best["K"]} mode={best["mode"]} OOF-C={best["oof_c"]:.4f}')
        if best_no_stack is not None:
            print(f'    BEST non-stack: K={best_no_stack["K"]} '
                  f'mode={best_no_stack["mode"]} OOF-C={best_no_stack["oof_c"]:.4f}')

        master[col] = best['val']
        fusion[endpoint] = {
            'best': {'K': best['K'], 'mode': best['mode'],
                     'oof_c': best['oof_c'], 'n_branches': best['n_branches']},
            'best_no_stack': (
                None if best_no_stack is None else
                {'K': best_no_stack['K'], 'mode': best_no_stack['mode'],
                 'oof_c': best_no_stack['oof_c']}),
            'pool_size': len(own_all),
            'cross_pool_size': len(other_all),
            'all_trials': [{'K': t['K'], 'mode': t['mode'], 'oof_c': t['oof_c']}
                           for t in trials],
        }
        # Save per-(K, mode) submissions for inspection
        for tag, vec in per_mode_save.items():
            pd.DataFrame({'trustii_id': val['trustii_id'], col: vec}).to_csv(
                out / f'sub_{tag}_{endpoint}.csv', index=False)

    master_path = out / 'sub_master.csv'
    master.to_csv(master_path, index=False)
    summary = {
        'pool_dirs': [str(d) for d in pool_dirs],
        'fusion': fusion, 'elapsed_sec': time.time() - t_start,
        'master': str(master_path),
    }
    (out / 'cv_summary.json').write_text(json.dumps(_convert(summary), indent=2))
    print(f'\n=== DONE in {(time.time()-t_start)/60:.1f} min ===')
    print(f'FINAL MASTER: {master_path}')


if __name__ == '__main__':
    main()
