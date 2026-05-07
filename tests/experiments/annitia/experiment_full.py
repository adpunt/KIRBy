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

    # Final fusion
    fusion = {}
    master = pd.DataFrame({'trustii_id': val['trustii_id']})
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death',   'risk_death')):
        recs = runs[endpoint]
        recs = [r for r in recs if np.isfinite(r['mean_cindex'])]
        recs.sort(key=lambda r: -r['mean_cindex'])
        recs = recs[:args.top_k]
        if not recs:
            print(f'  {endpoint}: no usable conditions, filling zeros')
            master[col] = 0.0
            continue
        print(f'\n  --- {endpoint} top-{len(recs)} ---')
        for r in recs:
            print(f'    {r["label"]:35s} C={r["mean_cindex"]:.4f}  ({r["source"]})')
        oof_list = [np.load(r['oof_path']) for r in recs]
        val_list = [np.load(r['val_path']) for r in recs]
        cv_list  = [r['mean_cindex'] for r in recs]
        weights = np.maximum(np.asarray(cv_list) - 0.5, 0.0)
        if weights.sum() == 0: weights = np.ones(len(recs))

        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        modes = {
            'top1':        val_list[int(np.argmax(cv_list))],
            'equal':       fuse_equal(val_list),
            'cv_weighted': fuse_cv_weighted(val_list, weights),
            'single':      fuse_single(val_list, cv_list),
            'rrf':         fuse_rrf(val_list, k=60.0),
        }
        sm_obj = None
        try:
            sm_obj = StackMetaFusion(l1_ratio=0.5)
            sm_obj.fit(oof_list, y, valid_mask=valid)
            modes['stack_meta'] = sm_obj.transform(val_list)
        except Exception as e:
            print(f'  stack_meta failed: {e}')

        # Score modes by OOF train C-index, pick best
        best_mode, best_c = None, -np.inf
        for mode in modes:
            try:
                if mode == 'top1':
                    ov = oof_list[int(np.argmax(cv_list))]
                elif mode == 'equal':
                    ov = fuse_equal(oof_list)
                elif mode == 'cv_weighted':
                    ov = fuse_cv_weighted(oof_list, weights)
                elif mode == 'single':
                    ov = fuse_single(oof_list, cv_list)
                elif mode == 'rrf':
                    ov = fuse_rrf(oof_list, k=60.0)
                elif mode == 'stack_meta':
                    ov = sm_obj.transform(oof_list)
                c = concordance(y[valid], ov[valid])
            except Exception:
                c = float('nan')
            print(f'    mode={mode:12s} OOF-train C={c:.4f}')
            if np.isfinite(c) and c > best_c:
                best_c, best_mode = c, mode
        print(f'    → picked {best_mode} (C={best_c:.4f})')
        master[col] = modes[best_mode]
        fusion[endpoint] = {'best_mode': best_mode, 'best_c': float(best_c),
                            'top_k': [r['label'] for r in recs]}
        # Save per-mode submissions for inspection
        for mode, vec in modes.items():
            pd.DataFrame({'trustii_id': val['trustii_id'], col: vec}).to_csv(
                out / f'sub_{mode}_{endpoint}.csv', index=False)

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
