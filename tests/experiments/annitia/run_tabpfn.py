#!/usr/bin/env python3
"""ANNITIA TabPFN landmark runner (feat/tabpfn-landmark).

Wires together:
    - landmark.LandmarkBinarySurvival  (Layer 2 — landmark-binary framing)
    - tabpfn.TabPFNClassifierWrapper   (Layer 3 — TabPFN-v2 ICL classifier)

For each fold, for each (view subset × prune × landmark mode), trains a
LandmarkBinarySurvival with TabPFN as the per-horizon classifier, scores by
censored C-index, and dumps OOF risks under <output_dir>/oof/<endpoint>/.
The OOF format matches search.py / run_deephit.py / run_dsm.py so the
combine_branches.py Layer-5 fusion can ingest it directly.

License caveat: TabPFN-v2 is under the Prior Labs research/non-commercial
license. Trustii's terms permit competition use, but if the user wants to
redistribute the trained submission outside the contest the license should
be re-checked.

Constraints handled here:
    - TabPFN-v2 default cap is ≤100 features; we either let the wrapper's
      variance-based selector trim, OR set --ignore-pretraining-limits to
      use the full feature set with a documented degradation warning.
    - TabPFN is intolerant of NaN; the wrapper does its own zero-impute.
    - n=1253 ≪ TabPFN-v2 cap (~10k), so no train-side trimming required.

Usage:
    python tests/experiments/annitia/run_tabpfn.py \
        --endpoints hepatic,death \
        --views snapshot,slope,nit,clinical,delta \
        --horizons q50,q75 \
        --output-dir tests/results/annitia_tabpfn
"""

import argparse
import itertools
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
from experiments.annitia.survival import concordance
from experiments.annitia.landmark import LandmarkBinarySurvival
from experiments.annitia.tabpfn import TabPFNClassifierWrapper
from experiments.annitia._schema import config_id as _config_id, make_record
from experiments.annitia._oof import (
    fill_oof_fold,
    init_oof_buffer,
    save_oof,
)


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


def _concat_views(views_dict, view_names):
    return np.concatenate([views_dict[n][0] for n in view_names], axis=1)


def _parse_horizons(spec: str, time: np.ndarray, event: np.ndarray):
    """Parse a string like 'q50,q75' or '5,10' into landmark times.

    'qNN' = NN-th percentile of event times. Numeric values are absolute.
    """
    out = []
    ev_t = time[event]
    if ev_t.size == 0:
        ev_t = time
    for tok in spec.split(','):
        tok = tok.strip()
        if tok.startswith('q'):
            q = float(tok[1:]) / 100.0
            out.append(float(np.quantile(ev_t, q)))
        else:
            out.append(float(tok))
    return out


def _enumerate_subsets(view_names, max_size=None):
    from itertools import combinations
    out = []
    cap = max_size if max_size is not None else len(view_names)
    for k in range(1, cap + 1):
        for combo in combinations(view_names, k):
            out.append(combo)
    return out


def run_endpoint(endpoint, data, views_dict, args, output_dir):
    folds = get_cv_folds(data, endpoint=endpoint,
                         n_folds=args.n_folds, seed=args.seed)
    y = data[endpoint]['y_struct']
    valid_mask = data[endpoint]['valid_mask']
    n_total = len(valid_mask)

    horizons = _parse_horizons(args.horizons, y['time'][valid_mask],
                               y['event'][valid_mask])
    print(f'  endpoint={endpoint} horizons={[f"{h:.2f}" for h in horizons]}')

    view_names = list(views_dict.keys())
    subsets = _enumerate_subsets(view_names, max_size=args.max_subset_size)
    modes = args.modes.split(',')

    per_condition = []
    json_path = output_dir / f'annitia_{endpoint}_tabpfn_search.json'
    t_all = time.time()

    for subset in subsets:
        X_full = _concat_views(views_dict, subset)
        for mode in modes:
            cfg = dict(views=list(subset), mode=mode, horizons=horizons,
                       max_features=args.max_features,
                       ignore_pretraining_limits=args.ignore_pretraining_limits)
            fold_c = []
            oof_buf = init_oof_buffer(n_total) if not args.no_oof else None
            t0 = time.time()
            for k_, (tr, va) in enumerate(folds):
                try:
                    m = LandmarkBinarySurvival(
                        classifier_cls=TabPFNClassifierWrapper,
                        horizons=horizons,
                        mode=mode,
                        classifier_kwargs=dict(
                            max_features=args.max_features,
                            device=args.device,
                            ignore_pretraining_limits=args.ignore_pretraining_limits,
                        ),
                        aggregator='rank',
                    )
                    m.fit(X_full[tr], y[tr])
                    risk = m.predict_risk(X_full[va])
                    c = concordance(y[va], risk)
                    if oof_buf is not None:
                        fill_oof_fold(oof_buf, va, risk)
                except Exception as exc:
                    print(f'    {"+".join(subset):30s} mode={mode} fold {k_}: ERR {exc}')
                    c = float('nan')
                fold_c.append(c)
                print(f'    {"+".join(subset):30s} mode={mode} '
                      f'fold {k_}: C={c:.4f}')

            mean_c = float(np.nanmean(fold_c)) if any(np.isfinite(fold_c)) else float('nan')
            std_c = float(np.nanstd(fold_c)) if any(np.isfinite(fold_c)) else float('nan')

            cid = _config_id(tuple(subset), None, 'landmark', 'tabpfn',
                             [mode, f'h{len(horizons)}'])
            oof_path = None
            if oof_buf is not None:
                oof_path = save_oof(oof_buf, str(output_dir), endpoint, cid)

            rec = make_record(
                endpoint=endpoint,
                views=subset,
                model='tabpfn',
                prune=None,
                fold_cindex=fold_c,
                mean_cindex=mean_c,
                std_cindex=std_c,
                n_features=int(X_full.shape[1]),
                framing='landmark',
                tricks=[mode, f'h{len(horizons)}'],
                oof_path=oof_path,
                elapsed_sec=time.time() - t0,
            )
            rec['config'] = cfg
            per_condition.append(rec)
            print(f'  [{endpoint}] {"+".join(subset):30s} mode={mode} '
                  f'd={X_full.shape[1]:4d} C={mean_c:.4f}±{std_c:.4f}')

            partial = {
                'endpoint': endpoint,
                'view_names': view_names,
                'n_folds': args.n_folds,
                'per_condition': per_condition,
                'best': max((r for r in per_condition
                             if isinstance(r['mean_cindex'], float)
                             and r['mean_cindex'] == r['mean_cindex']),
                            key=lambda r: r['mean_cindex'], default=None),
                'elapsed_sec': time.time() - t_all,
                'in_progress': True,
            }
            tmp = json_path.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump(_convert(partial), f)
            os.replace(tmp, json_path)

    best = max((r for r in per_condition
                if isinstance(r['mean_cindex'], float)
                and r['mean_cindex'] == r['mean_cindex']),
               key=lambda r: r['mean_cindex'], default=None)
    summary = {
        'endpoint': endpoint,
        'view_names': view_names,
        'n_folds': args.n_folds,
        'per_condition': per_condition,
        'best': best,
        'elapsed_sec': time.time() - t_all,
        'in_progress': False,
    }
    with open(json_path, 'w') as f:
        json.dump(_convert(summary), f, indent=2)
    print(f'\nsearch saved: {json_path}')

    if best is not None:
        # Refit on full valid cohort.
        valid_idx = np.where(valid_mask)[0]
        Xf = _concat_views(views_dict, tuple(best['views']))[valid_idx]
        yf = y[valid_idx]
        m = LandmarkBinarySurvival(
            classifier_cls=TabPFNClassifierWrapper,
            horizons=horizons,
            mode=best['config']['mode'],
            classifier_kwargs=dict(
                max_features=args.max_features,
                device=args.device,
                ignore_pretraining_limits=args.ignore_pretraining_limits,
            ),
            aggregator='rank',
        )
        m.fit(Xf, yf)
        model_path = output_dir / f'annitia_{endpoint}_tabpfn_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': m, 'views': best['views'],
                'mode': best['config']['mode'], 'horizons': horizons,
            }, f)
        print(f'model saved: {model_path}')


def run(args):
    print('=' * 70)
    print('ANNITIA TabPFN landmark (feat/tabpfn-landmark)')
    print(f'  views:        {args.views}')
    print(f'  endpoints:    {args.endpoints}')
    print(f'  modes:        {args.modes}')
    print(f'  horizons:     {args.horizons}')
    print(f'  max_features: {args.max_features} (pretraining limits {"IGNORED" if args.ignore_pretraining_limits else "respected"})')
    print('=' * 70)

    data = load_annitia(data_dir=args.data_dir, filename=args.filename)
    summarize(data)

    view_names = (args.views.split(',') if args.views
                  else list(VIEW_BUILDERS.keys()))
    views_dict = build_views(data, view_names=view_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for endpoint in args.endpoints.split(','):
        if data[endpoint] is None:
            print(f'\nSkipping {endpoint}: no outcomes loaded.')
            continue
        print(f'\n{"-" * 70}')
        print(f'Endpoint: {endpoint}')
        print(f'{"-" * 70}')
        run_endpoint(endpoint, data, views_dict, args, output_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views', type=str, default='snapshot,slope,delta,nit,clinical')
    p.add_argument('--endpoints', type=str, default='hepatic,death')
    p.add_argument('--modes', type=str, default='drop,ipcw',
                   help='Comma-separated landmark modes (drop|ipcw)')
    p.add_argument('--horizons', type=str, default='q25,q50,q75',
                   help='Comma-separated landmarks. qNN=Nth quantile of event times; '
                        'numeric=absolute time')
    p.add_argument('--max-subset-size', type=int, default=4)
    p.add_argument('--max-features', type=int, default=100,
                   help='TabPFN-v2 default cap is 100. Set higher with --ignore-pretraining-limits.')
    p.add_argument('--ignore-pretraining-limits', action='store_true',
                   help='Bypass TabPFN feature/sample caps with degradation warning.')
    p.add_argument('--device', type=str, default='cpu',
                   help='cpu | cuda. CUDA gives ~10x speedup if available.')
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', type=str, default=None)
    p.add_argument('--filename', type=str, default='train.csv')
    p.add_argument('--output-dir', type=str, default='tests/results/annitia_tabpfn')
    p.add_argument('--no-oof', action='store_true')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
