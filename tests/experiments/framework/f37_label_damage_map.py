#!/usr/bin/env python3
"""f37 -- how the kinds of training-label damage differ from each other.

Owner's question, 2026-10-03: if several kinds of label damage do the same thing
to a model, the noise experiments need only one of them. This damages the
TRAINING labels of each fold with every kind the noise experiments use, at three
levels, fits a few fixed (model, representation) pairs, and keeps the held-out
score and the held-out predictions. Held-out labels are never touched.

NO HYBRID SEARCH. Only loaders, the injectors and plain model fits are called.
Data come from noise_helpers.load_noise_data, not run_all, which would start
every precompute (see f36).

Kinds (22): the 14 in noise_helpers.SWEEP_ORDER, the two shift_only_* kinds and
the four sparse_dprime* kinds (the 6 of DETECTION_IS_DEFINED that SWEEP_ORDER
lacks). Levels per kind are below (LEVELS).

The unit of the sparse_dprime* levels is the SD of lightgbm's out-of-fold
residual on the clean training labels. It is measured on ecfp4 for every pair
(one choice for all four, so the damage is identical across pairs). Rows of
those kinds say which representation set it.

Output, per dataset:
  <output>/f37_label_damage_map/<dataset>/rows.jsonl   one row per fit
  <output>/f37_label_damage_map/<dataset>/folds.json   held-out positions per fold
A rerun skips every row already on disk.

Usage:
  KIRBY_LGBM_THREADS=1 KIRBY_RF_JOBS=2 python f37_label_damage_map.py herg_ki \
      --output <dir> [--slice K/N] [--kinds a,b] [--pairs rf:ecfp4,...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'src'),
           os.path.join(_HERE, '..', '..'), os.path.join(_HERE, '..'),
           os.path.join(_HERE, '..', 'noise')):
    sys.path.insert(0, _p)

EXP_ID = 'f37_label_damage_map'
N_FOLDS = 3
DRAWS = (0, 1)
PAIRS = (('rf', 'ecfp4'), ('svm', 'pdv'), ('lightgbm', 'mhg_gnn'),
         ('dnn', 'graph_kernel'))
RESID_SD_REP = 'ecfp4'

STD_LEVELS = (0.3, 0.75, 1.5)          # fraction of the clean training-label SD
CENSORING_LEVELS = (0.10, 0.25, 0.40)  # fraction of labels clipped
SHIFT_LEVELS = (0.5, 1.0, 2.0)         # label SDs
DPRIME_LEVELS = (1.0, 2.0, 4.0)        # residual SDs


def kinds_and_levels():
    from experiments.noise import noise_helpers as nh
    out = []
    for k in nh.SWEEP_ORDER:
        out.append((k, CENSORING_LEVELS if k == 'censoring' else STD_LEVELS))
    for k in nh.SHIFT_ONLY_CONDITIONS:
        out.append((k, SHIFT_LEVELS))
    for k in nh.RELATIVE_CONDITIONS:
        if k.startswith('sparse_dprime'):
            out.append((k, DPRIME_LEVELS))
    return out


def _key(r):
    return json.dumps([r['model'], r['rep'], r['kind'], r['level'], r['draw'],
                       r['fold']])


def _done(path):
    out = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    out.add(_key(json.loads(line)))
                except ValueError:
                    continue            # last line cut off by a kill
    return out


def _append(path, row):
    with open(path, 'a') as fh:
        fh.write(json.dumps(row) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def run(data, dataset, output, slice_=None, kinds=None, pairs=None):
    from sklearn.preprocessing import StandardScaler
    import _shared as sh
    import dataset_policy as dp
    from experiments.noise import noise_helpers as nh

    labels = np.asarray(data['labels'], dtype=float).ravel()
    reps = data['reps']
    metric_name, metric_fn, lower_is_better = sh.get_domain_metric(data)
    if data.get('task_type', 'regression') != 'regression':
        raise SystemExit(f'f37 is regression only; {dataset} is '
                         f'{data.get("task_type")}')
    noise_groups = dp.noise_groups_for(data)[0]
    splits = list(sh.cv_split(data['groups'], N_FOLDS, reps[PAIRS[0][1]],
                              labels, task_type='regression'))

    out_dir = os.path.join(output, EXP_ID, dataset)
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, 'rows.jsonl')
    with open(os.path.join(out_dir, 'folds.json'), 'w') as fh:
        json.dump({'val_idx': [np.asarray(v).tolist() for _, v in splits],
                   'train_idx': [np.asarray(t).tolist() for t, _ in splits]},
                  fh)
    done = _done(rows_path)

    wanted = kinds_and_levels()
    if kinds:
        wanted = [(k, l) for k, l in wanted if k in kinds]
    pairs = tuple(pairs) if pairs else PAIRS

    # Units: one per (fold, kind, level, draw) plus one clean unit per fold.
    # Each unit fits every pair, so the damage is built once per unit.
    units = [(f, 'clean', 0.0, 0) for f in range(N_FOLDS)]
    for f in range(N_FOLDS):
        for kind, levels in wanted:
            for lvl in levels:
                for d in DRAWS:
                    units.append((f, kind, lvl, d))
    if slice_:
        k, n = (int(x) for x in slice_.split('/'))
        units = units[k::n]

    resid_cache = {}
    scalers = {}
    factories = {}
    t_start = time.time()
    for ui, (fold, kind, lvl, draw) in enumerate(units):
        todo = [(m, r) for m, r in pairs if _key(
            {'model': m, 'rep': r, 'kind': kind, 'level': lvl, 'draw': draw,
             'fold': fold}) not in done]
        if not todo:
            continue
        tr, va = splits[fold]
        tr, va = np.asarray(tr), np.asarray(va)
        prov = {}
        if kind == 'clean':
            y_tr = labels[tr].copy()
        else:
            if kind in nh.SHIFT_ONLY_CONDITIONS:
                y_tr, _, prov = nh.inject_shift_train_labels(
                    labels, train_idx=tr, condition=kind, delta_sd=lvl,
                    draw=draw)
            elif kind in nh.RELATIVE_CONDITIONS:
                if fold not in resid_cache:
                    Xr = StandardScaler().fit_transform(
                        np.asarray(reps[RESID_SD_REP][tr], dtype=np.float64))
                    g_tr = (np.asarray(data['groups'])[tr])
                    resid_cache[fold] = nh.oof_residual_sd(
                        Xr, labels[tr], groups=g_tr)[0]
                y_tr, _, prov = nh.inject_relative_train_labels(
                    labels, train_idx=tr, condition=kind, level=lvl,
                    draw=draw, resid_sd=resid_cache[fold])
            else:
                spec = nh.noise_spec_for(dataset, kind, lvl, noise_seed=draw)
                y_tr, _, prov = nh.inject_train_labels(
                    labels, spec, train_idx=tr, groups=noise_groups,
                    dataset_name=dataset)
        y_va = labels[va]
        for m, r in todo:
            if (r, fold) not in scalers:
                sc = StandardScaler()
                X = np.asarray(reps[r], dtype=np.float64)
                scalers[(r, fold)] = (sc.fit_transform(X[tr]), sc.transform(X[va]))
            Xt, Xv = scalers[(r, fold)]
            if m not in factories:
                factories[m] = sh.get_model_factories(
                    [m], 'regression', len(labels))[m]
            t0 = time.time()
            err = None
            try:
                model = factories[m]()
                model.fit(Xt, y_tr)
                p = np.asarray(model.predict(Xv), dtype=float).ravel()
                score = float(metric_fn(y_va, p))
            except Exception as exc:
                p, score, err = None, float('nan'), f'{type(exc).__name__}: {exc}'
            row = {'dataset': dataset, 'model': m, 'rep': r, 'kind': kind,
                   'level': lvl, 'draw': draw, 'fold': fold,
                   'metric': metric_name, 'lower_is_better': bool(lower_is_better),
                   'score': score, 'seconds': round(time.time() - t0, 2),
                   'n_train': int(len(tr)), 'n_val': int(len(va)),
                   'preds': None if p is None else [round(float(x), 5) for x in p],
                   'error': err,
                   'resid_sd': (resid_cache.get(fold)
                                if kind in nh.RELATIVE_CONDITIONS else None),
                   'resid_sd_rep': (RESID_SD_REP
                                    if kind in nh.RELATIVE_CONDITIONS else None)}
            _append(rows_path, row)
            done.add(_key(row))
        print(f'[f37 {dataset}] unit {ui + 1}/{len(units)} fold{fold} {kind} '
              f'{lvl} d{draw}  elapsed {time.time() - t_start:.0f}s',
              flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='f37 on one dataset')
    ap.add_argument('dataset')
    ap.add_argument('--output', required=True)
    ap.add_argument('--slice', default=None, help='K/N: every Nth unit from K')
    ap.add_argument('--kinds', default=None)
    ap.add_argument('--pairs', default=None, help='model:rep,model:rep')
    args = ap.parse_args()
    # NEVER A SEARCH (same guard as f36).
    os.environ['KIRBY_NO_COLD_SEARCH'] = '1.0'
    os.environ['KIRBY_COLD_SEARCH_MAX_CONFIGS'] = '0'
    from experiments.noise.noise_helpers import load_noise_data
    data, _ = load_noise_data(args.dataset)
    pairs = ([tuple(p.split(':')) for p in args.pairs.split(',')]
             if args.pairs else None)
    run(data, args.dataset, args.output, args.slice,
        args.kinds.split(',') if args.kinds else None, pairs)


if __name__ == '__main__':
    main()
