#!/usr/bin/env python3
"""ANNITIA DSM runner (feat/dsm-cr branch).

Runs two model families and dumps their OOF risks for downstream Layer-5
fusion via combine_branches.py:

    1) Single-risk DSM, per endpoint, swept through the existing
       search_survival_hybrid grid alongside RSF/Cox/GBSA. This adds DSM as
       another (framing=cox-like, algorithm=dsm) row. We restrict the view
       sweep to a small set to keep runtime reasonable.

    2) Joint competing-risks DSM (DeepSurvivalMachinesCR). Trains once on
       the all-views concat with a small hyperparam grid; logs per-endpoint
       C-index and dumps OOF risks in the same format as run_deephit.py.

The competing-risks model is the main lever for the hepatic endpoint —
it borrows representational strength from the death endpoint's larger
event count.

Usage (smoke):
    python tests/experiments/annitia/run_dsm.py --epochs 30 --n-folds 2

Usage (full):
    python tests/experiments/annitia/run_dsm.py \
        --output-dir tests/results/annitia_dsm
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
from experiments.annitia.dsm import DeepSurvivalMachinesCR
from experiments.annitia._schema import config_id as _config_id
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


def _grid(args):
    K = [int(x) for x in args.k.split(',')]
    hidden = [tuple(int(h) for h in s.split('x')) for s in args.hidden.split(',')]
    drop = [float(x) for x in args.dropout.split(',')]
    lr = [float(x) for x in args.lr.split(',')]
    return list(itertools.product(K, hidden, drop, lr))


def run(args):
    print('=' * 70)
    print('ANNITIA DSM (competing-risks, feat/dsm-cr)')
    print(f'  views:        {args.views}')
    print(f'  n_folds:      {args.n_folds}')
    print(f'  epochs/pat:   {args.epochs}/{args.patience}')
    print('=' * 70)

    data = load_annitia(data_dir=args.data_dir, filename=args.filename)
    summarize(data)
    if data['hepatic'] is None or data['death'] is None:
        raise RuntimeError('Both endpoints must be present for DSM-CR.')

    view_names = (args.views.split(',') if args.views
                  else list(VIEW_BUILDERS.keys()))
    views_dict = build_views(data, view_names=view_names)
    X = _concat_views(views_dict, view_names)
    print(f'  concat features: {X.shape[1]}')

    folds_hep = get_cv_folds(data, endpoint='hepatic',
                             n_folds=args.n_folds, seed=args.seed)
    y_hep = data['hepatic']['y_struct']
    y_death = data['death']['y_struct']
    death_valid = data['death']['valid_mask']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'annitia_dsm_search.json'

    grid = _grid(args)
    print(f'  grid: {len(grid)} cfgs x {args.n_folds} folds = {len(grid) * args.n_folds} fits')

    per_condition = []
    t_all = time.time()
    n_total = len(data['patient_ids'])
    for cfg_i, (k_, hidden, dropout, lr) in enumerate(grid):
        cfg = dict(k=k_, hidden=list(hidden), dropout=dropout, lr=lr)
        fold_hep, fold_death, fold_w = [], [], []
        oof_hep = init_oof_buffer(n_total) if not args.no_oof else None
        oof_death = init_oof_buffer(n_total) if not args.no_oof else None
        t0 = time.time()
        for fold_i, (tr, va) in enumerate(folds_hep):
            try:
                m = DeepSurvivalMachinesCR(
                    k=k_, hidden=hidden, dropout=dropout, lr=lr,
                    batch_size=args.batch_size, epochs=args.epochs,
                    patience=args.patience,
                    horizon_quantile=args.horizon_quantile,
                    random_state=args.seed, verbose=False)
                m.fit(X[tr], y_hep[tr], y_death[tr])
                rh, rd = m.predict_risk(X[va])
                c_h = concordance(y_hep[va], rh)
                va_d = death_valid[va]
                if va_d.sum() >= 2 and bool(y_death[va][va_d]['event'].any()):
                    c_d = concordance(y_death[va][va_d], rd[va_d])
                else:
                    c_d = float('nan')
                if oof_hep is not None:
                    fill_oof_fold(oof_hep, va, rh)
                    if va_d.any():
                        fill_oof_fold(oof_death, va[va_d], rd[va_d])
            except Exception as e:
                print(f'    cfg={cfg_i} fold={fold_i} FAILED: {e}')
                c_h, c_d = float('nan'), float('nan')
            fold_hep.append(c_h)
            fold_death.append(c_d)
            if not (np.isnan(c_h) or np.isnan(c_d)):
                fold_w.append(0.7 * c_h + 0.3 * c_d)
            print(f'    cfg={cfg_i:02d} fold={fold_i} '
                  f'hep={c_h:.4f} death={c_d:.4f}')

        oof_paths = {}
        if oof_hep is not None:
            cid_hep = _config_id(tuple(view_names), None, 'dsm_cr',
                                 'dsm', [f'cfg{cfg_i}', 'hepatic'])
            oof_paths['hepatic'] = save_oof(oof_hep, str(output_dir),
                                            'hepatic', cid_hep)
            cid_death = _config_id(tuple(view_names), None, 'dsm_cr',
                                   'dsm', [f'cfg{cfg_i}', 'death'])
            oof_paths['death'] = save_oof(oof_death, str(output_dir),
                                          'death', cid_death)

        mh = float(np.nanmean(fold_hep)) if fold_hep else float('nan')
        md = float(np.nanmean(fold_death)) if fold_death else float('nan')
        mw = float(np.nanmean(fold_w)) if fold_w else float('nan')
        sh = float(np.nanstd(fold_hep)) if fold_hep else float('nan')
        sd = float(np.nanstd(fold_death)) if fold_death else float('nan')
        per_condition.append({
            'schema_version': 1,
            'config': cfg,
            'layer1': {'views': list(view_names), 'n_features': int(X.shape[1]),
                       'n_features_after_prune': int(X.shape[1]),
                       'prune': None, 'feature_set_id': None},
            'layer2': {'framing': 'dsm_cr'},
            'layer3': {'algorithm': 'dsm'},
            'layer4': {'tricks': []},
            'fold_metrics': {
                'fold_cindex_hepatic': fold_hep,
                'fold_cindex_death': fold_death,
                'mean_cindex_hepatic': mh, 'std_cindex_hepatic': sh,
                'mean_cindex_death': md, 'std_cindex_death': sd,
                'mean_weighted': mw,
            },
            'oof_paths': oof_paths,
            'mean_cindex_hepatic': mh, 'std_cindex_hepatic': sh,
            'mean_cindex_death': md, 'std_cindex_death': sd,
            'mean_weighted': mw,
            'fold_hepatic': fold_hep,
            'fold_death': fold_death,
            'elapsed_sec': time.time() - t0,
        })
        print(f'  cfg={cfg_i:02d} {cfg}  '
              f'hep={mh:.4f}±{sh:.4f}  death={md:.4f}±{sd:.4f}  '
              f'weighted={mw:.4f}  ({time.time()-t0:.1f}s)')

        partial = {
            'view_names': view_names,
            'n_features': int(X.shape[1]),
            'n_folds': args.n_folds,
            'per_condition': per_condition,
            'best': max((r for r in per_condition if not np.isnan(r['mean_weighted'])),
                        key=lambda r: r['mean_weighted'], default=None),
            'elapsed_sec': time.time() - t_all,
            'in_progress': True,
        }
        tmp = json_path.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(_convert(partial), f)
        os.replace(tmp, json_path)

    best = max((r for r in per_condition if not np.isnan(r['mean_weighted'])),
               key=lambda r: r['mean_weighted'], default=None)
    summary = {
        'view_names': view_names,
        'n_features': int(X.shape[1]),
        'n_folds': args.n_folds,
        'per_condition': per_condition,
        'best': best,
        'elapsed_sec': time.time() - t_all,
        'in_progress': False,
    }
    with open(json_path, 'w') as f:
        json.dump(_convert(summary), f, indent=2)
    print(f'\nsearch saved: {json_path}')

    if best is None:
        print('No successful fold; nothing to refit.')
        return

    print(f'\nRefitting best on full train: {best["config"]}')
    cfg = best['config']
    m = DeepSurvivalMachinesCR(
        k=cfg['k'], hidden=tuple(cfg['hidden']), dropout=cfg['dropout'],
        lr=cfg['lr'], batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, horizon_quantile=args.horizon_quantile,
        random_state=args.seed, verbose=True)
    m.fit(X, y_hep, y_death)

    model_path = output_dir / 'annitia_dsm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': m, 'view_names': view_names,
            'n_features': int(X.shape[1]), 'config': cfg,
        }, f)
    print(f'model saved: {model_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views', type=str, default=None)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', type=str, default=None)
    p.add_argument('--filename', type=str, default='train.csv')
    p.add_argument('--output-dir', type=str, default='tests/results/annitia_dsm')
    p.add_argument('--k', type=str, default='3,4,6')
    p.add_argument('--hidden', type=str, default='64x64,128x64')
    p.add_argument('--dropout', type=str, default='0.2,0.3,0.5')
    p.add_argument('--lr', type=str, default='1e-3,5e-4')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--horizon-quantile', type=float, default=0.5)
    p.add_argument('--no-oof', action='store_true')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
