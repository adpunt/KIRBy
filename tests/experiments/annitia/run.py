#!/usr/bin/env python3
"""ANNITIA MASLD risk-stratification experiment (Trustii challenge).

KIRBy-adjacent: instead of mixing learned representation embeddings, we mix
hand-crafted feature views (snapshot, slope, summary, delta, missingness, NIT)
of the same per-patient longitudinal record. The hybrid search enumerates all
view subsets x {Cox, RSF} and picks the best by 5-fold CV censored C-index,
per endpoint (hepatic, death).

Usage:
    # Smoke test (single view, single model, 2 folds)
    python tests/experiments/annitia_maslg.py --views nit --models rsf --n-folds 2

    # Full search, both endpoints
    python tests/experiments/annitia_maslg.py --output-dir tests/results/annitia

Outputs (per endpoint):
    - JSON with per-condition C-index table
    - Pickle with the final refitted model + view subset
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, summarize, get_cv_folds
from experiments.annitia.views import build_views, VIEW_BUILDERS
from experiments.annitia.search import search_survival_hybrid


def _convert(obj):
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run(args):
    print('=' * 70)
    print('ANNITIA MASLD risk-stratification experiment')
    print(f'  views:    {args.views}')
    print(f'  models:   {args.models}')
    print(f'  n_folds:  {args.n_folds}')
    print(f'  endpoints:{args.endpoints}')
    print(f'  data_dir: {args.data_dir or "(auto-resolved)"}')
    print('=' * 70)

    data = load_annitia(data_dir=args.data_dir, filename=args.filename)
    summarize(data)

    view_names = (args.views.split(',') if args.views
                  else list(VIEW_BUILDERS.keys()))
    views = build_views(data, view_names=view_names)
    model_names = tuple(args.models.split(','))
    endpoints = args.endpoints.split(',')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for endpoint in endpoints:
        if data[endpoint] is None:
            print(f'\nSkipping {endpoint}: no outcomes loaded.')
            continue
        print(f'\n{"-" * 70}')
        print(f'Endpoint: {endpoint}')
        print(f'  valid patients: {int(data[endpoint]["valid_mask"].sum())}')
        print(f'  events:         {int((data[endpoint]["event"] & data[endpoint]["valid_mask"]).sum())}')
        print(f'{"-" * 70}')

        folds = get_cv_folds(data, endpoint=endpoint,
                             n_folds=args.n_folds, seed=args.seed)
        prune_levels = tuple(
            (None if p in (None, '', 'none') else int(p))
            for p in args.prune_levels.split(','))
        t0 = time.time()
        result = search_survival_hybrid(
            views=views,
            y_struct=data[endpoint]['y_struct'],
            valid_mask=data[endpoint]['valid_mask'],
            folds=folds,
            endpoint=endpoint,
            model_names=model_names,
            max_subset_size=args.max_subset_size,
            prune_levels=prune_levels,
            top_k_keep=args.top_k_keep,
            verbose=True,
        )
        elapsed = time.time() - t0

        if result.best is not None:
            print(f'\n  BEST: {"+".join(result.best["views"])} / '
                  f'{result.best["model"]}  '
                  f'C={result.best["mean_cindex"]:.4f}±{result.best["std_cindex"]:.4f}')
        print(f'  ({elapsed:.1f}s)')

        # Persist the search summary
        summary = {
            'endpoint': endpoint,
            'n_valid_patients': int(data[endpoint]['valid_mask'].sum()),
            'n_events': int((data[endpoint]['event'] & data[endpoint]['valid_mask']).sum()),
            'n_folds': args.n_folds,
            'view_names': view_names,
            'model_names': list(model_names),
            'per_condition': result.per_condition,
            'best': result.best,
            'elapsed_sec': elapsed,
        }
        json_path = output_dir / f'annitia_{endpoint}_search.json'
        with open(json_path, 'w') as f:
            json.dump(_convert(summary), f, indent=2)
        print(f'  search saved: {json_path}')

        # Persist the refit model
        if result.final_model is not None:
            model_path = output_dir / f'annitia_{endpoint}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': result.final_model,
                    'view_names': result.final_view_names,
                    'model_name': result.final_model_name,
                    'prune': result.final_prune,
                    'feature_indices': result.final_feature_indices,
                    'endpoint': endpoint,
                }, f)
            print(f'  model saved:  {model_path}')

        # Persist the top-K ensemble bundle
        if result.top_k_models:
            ens_path = output_dir / f'annitia_{endpoint}_topk.pkl'
            with open(ens_path, 'wb') as f:
                pickle.dump({
                    'endpoint': endpoint,
                    'top_k': result.top_k_models,
                }, f)
            print(f'  top-{len(result.top_k_models)} ensemble saved: {ens_path}')

        results[endpoint] = summary

    # Combined score (matches the challenge weighting)
    if 'hepatic' in results and 'death' in results:
        c_h = results['hepatic']['best']['mean_cindex']
        c_d = results['death']['best']['mean_cindex']
        score = 0.7 * c_h + 0.3 * c_d
        print(f'\nQuantitative score estimate (5-fold CV):')
        print(f'  C-index hepatic = {c_h:.4f}')
        print(f'  C-index death   = {c_d:.4f}')
        print(f'  weighted        = {score:.4f}  (= 0.7*hep + 0.3*death)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views', type=str, default=None,
                   help='Comma-separated view names (default: all 6)')
    p.add_argument('--models', type=str, default='cox,coxnet,rsf,gbsa',
                   help='Comma-separated downstream models (default: cox,coxnet,rsf,gbsa)')
    p.add_argument('--prune-levels', type=str, default='none,50',
                   help='Comma-separated prune levels (e.g. "none,50,30"). '
                        '"none" = use all features; integer = top-K by RSF importance.')
    p.add_argument('--top-k-keep', type=int, default=3,
                   help='Number of top configs to retain for rank-ensemble.')
    p.add_argument('--endpoints', type=str, default='hepatic,death',
                   help='Comma-separated endpoints (default: hepatic,death)')
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--max-subset-size', type=int, default=None,
                   help='Cap on # of views to combine (default: all)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', type=str, default=None,
                   help='Override ANNITIA data dir')
    p.add_argument('--filename', type=str, default='train.csv')
    p.add_argument('--output-dir', type=str,
                   default='tests/results/annitia')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
