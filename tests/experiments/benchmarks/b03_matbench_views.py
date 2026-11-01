#!/usr/bin/env python3
"""Matbench view and model screen: every cached representation of each task,
scored alone under the published models on Matbench's own 5 test folds, with
out-of-fold predictions kept and correlations between views.

Object: one compound. Target: one number (MAE in the target's unit).
Folds: Matbench's 5, KFold(5, shuffle=True, random_state=18012019) over the
file's row order (load_matbench returns them; approved 2026-10-23).

Views: the representations of src/kirby/representations/materials.py, cached
by tests/tools/matbench_build_reps.py at <KIRBY_MATBENCH_REPS>/<task>/<rep>.npy
(default /Volumes/seagate/matbench/reps). expt_gap has compositions only, so
only the four composition views exist for it. The caches set missing values
to 0; the cached sine Coulomb matrix is fitted on all compounds (both noted in
NEW_DATASETS.md section 3).

Models, the two published on the leaderboard:
    rf      the leaderboard random forest: b03_matbench_rf_published.published_rf
            (500 trees, automatminer's cleaning, seed 42)
    modnet  the leaderboard MODNet: b04_matbench_modnet_published.
            modnet_published_fold (feature selection per training fold, the
            fit_preset grid with nested=5, 5-network ensemble). Needs the old
            environment (python 3.8, modnet 0.1.12); the runner calls this
            file with that interpreter. This file is kept 3.8-compatible.

Stages:
    targets    write <reps>/<task>/_targets.npz (labels and Matbench folds), so
               an environment without the matbench package can still evaluate
    evaluate   --views V --models M --tasks T: one result file per
               (task, view, model) with per-fold MAE; resumes at the next fold
    summarise  summary.md: MAE per view and model per task beside the
               published means, linear CKA between views, and the Spearman
               correlation of absolute errors between views per model

    python tests/experiments/benchmarks/b03_matbench_views.py --stage targets --tasks matbench_jdft2d
    python tests/experiments/benchmarks/b03_matbench_views.py --stage evaluate \\
        --tasks matbench_jdft2d --views magpie --models rf

Output: <KIRBY_BENCH_RESULTS>/b03_views/<task>/<view>__<model>.json and
.oof.npy (default KIRBY_BENCH_RESULTS=/Volumes/seagate/benchmarks_protocol/results).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))

REPS_DIR = Path(os.environ.get('KIRBY_MATBENCH_REPS', '/Volumes/seagate/matbench/reps'))
RESULTS_DIR = Path(os.environ.get('KIRBY_BENCH_RESULTS',
                                  '/Volumes/seagate/benchmarks_protocol/results')) / 'b03_views'
TASKS = ['matbench_jdft2d', 'matbench_phonons', 'matbench_expt_gap',
         'matbench_dielectric', 'matbench_log_gvrh', 'matbench_perovskites']
COMPOSITION_ONLY = {'matbench_expt_gap'}
COMPOSITION_VIEWS = ['magpie', 'stoichiometry', 'valence_orbital', 'ion_property']
STRUCTURE_VIEWS = ['sine_coulomb', 'soap_avg', 'site_stats_fingerprint', 'density_features',
                   'rdf', 'xrd', 'global_symmetry', 'chgnet', 'mace_mp']
MODELS = ['rf', 'modnet']
N_FOLDS = 5
SEED = 42


def views_for(task):
    """The views a task has: composition views always, structure views when
    the task has structures (every task but expt_gap)."""
    return COMPOSITION_VIEWS + ([] if task in COMPOSITION_ONLY else STRUCTURE_VIEWS)


def log(msg):
    print('%s %s' % (time.strftime('%H:%M:%S'), msg), flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'b03_views.log', 'a') as f:
        f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(HERE / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def targets_path(task):
    return REPS_DIR / task / '_targets.npz'


def view_path(task, view):
    return REPS_DIR / task / ('%s.npy' % view)


def result_path(task, view, model):
    """The file the ledger reads for one (task, view, model) unit."""
    return RESULTS_DIR / task / ('%s__%s.json' % (view, model))


# ----------------------------------------------------------------------------
# targets
# ----------------------------------------------------------------------------

def stage_targets(tasks):
    from kirby.datasets.matbench import load_matbench
    for task in tasks:
        p = targets_path(task)
        if p.exists():
            log('targets %s: already written' % task)
            continue
        d = load_matbench(task)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name('_targets.tmp.%d.npz' % os.getpid())
        np.savez(tmp, labels=np.asarray(d['labels'], float),
                 fold=np.asarray(d['matbench_fold'], int))
        os.replace(tmp, p)
        log('targets %s: %d compounds' % (task, len(d['labels'])))


def load_targets(task):
    p = targets_path(task)
    if not p.exists():
        raise SystemExit('no %s; run --stage targets --tasks %s first' % (p, task))
    z = np.load(p)
    return z['labels'], z['fold']


# ----------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------

def fit_predict(model, X_tr, y_tr, X_te, fold, names):
    if model == 'rf':
        b03 = _load_module('b03_rf', 'b03_matbench_rf_published.py')
        n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK') or -1)
        pred, n_feat_drop, n_row_drop = b03.published_rf(X_tr, y_tr, X_te, seed=SEED, n_jobs=n_jobs)
        return pred, {'features_dropped': int(n_feat_drop), 'train_rows_dropped': int(n_row_drop)}
    if model == 'modnet':
        b04 = _load_module('b04_modnet', 'b04_matbench_modnet_published.py')
        n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK') or 4)
        pred = b04.modnet_published_fold(X_tr, y_tr, X_te, feature_names=names,
                                         n_jobs=n_jobs, fast=False, k=fold)
        return np.asarray(pred, float).ravel(), {}
    raise SystemExit('unknown model %r' % model)


def evaluate_one(task, view, model):
    out = result_path(task, view, model)
    if out.exists():
        return
    vp = view_path(task, view)
    if not vp.exists():
        raise SystemExit('%s %s: no cache at %s; build it first' % (task, view, vp))
    X = np.load(vp).astype(float)
    names_p = vp.with_name('%s_names.json' % view)
    names = json.loads(names_p.read_text()) if names_p.exists() else None
    y, fold_of = load_targets(task)
    if len(X) != len(y):
        raise SystemExit('%s %s: %d rows in the cache, %d targets' % (task, view, len(X), len(y)))
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.stem + '.partial.json')
    rows, infos, oof = [], [], np.full(len(y), np.nan)
    if part.exists():
        pj = json.loads(part.read_text())
        rows, infos, oof = pj['rows'], pj['infos'], np.asarray(pj['oof'], float)
        log('%s %s %s: resuming after %d fold(s)' % (task, view, model, len(rows)))
    t_all = time.time()
    for k in range(N_FOLDS):
        if k < len(rows):
            continue
        tr, te = np.where(fold_of != k)[0], np.where(fold_of == k)[0]
        t0 = time.time()
        pred, info = fit_predict(model, X[tr], y[tr], X[te], k, names)
        oof[te] = pred
        mae = float(np.mean(np.abs(pred - y[te])))
        rows.append({'fold': k, 'mae': mae, 'n_test': int(len(te)), 'seconds': time.time() - t0})
        infos.append(info)
        log('%s %s %s fold %d: MAE %.4f (%.0f s)' % (task, view, model, k, mae, rows[-1]['seconds']))
        part.write_text(json.dumps({'rows': rows, 'infos': infos,
                                    'oof': [None if np.isnan(v) else float(v) for v in oof]}))
    res = {'task': task, 'view': view, 'model': model, 'n_features': int(X.shape[1]),
           'n_compounds': int(len(y)), 'folds': rows,
           'mean_mae': float(np.mean([r['mae'] for r in rows])),
           'fit_info': infos, 'seconds': time.time() - t_all, 'result_status': 'ok'}
    np.save(str(out.with_suffix('')) + '.oof.npy', oof)
    out.write_text(json.dumps(res, indent=1))
    if part.exists():
        part.unlink()


# ----------------------------------------------------------------------------
# summarise
# ----------------------------------------------------------------------------

def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B, 'fro') ** 2
                 / (np.linalg.norm(A.T @ A, 'fro') * np.linalg.norm(B.T @ B, 'fro') + 1e-30))


def published_means(task):
    """Leaderboard mean MAE per model, read from the leaderboard files when
    they are here; {} otherwise."""
    out = {}
    base = Path(os.environ.get('KIRBY_MATBENCH_PROTOCOL', '/Volumes/seagate/benchmarks_protocol/matbench'))
    for model, sub in (('rf', 'rf'), ('modnet', 'modnet_v0.1.12')):
        f = base / sub / 'results.json.gz'
        if not f.exists():
            continue
        import gzip
        d = json.load(gzip.open(str(f)))
        r = d['tasks'][task]['results']
        out[model] = float(np.mean([r['fold_%d' % k]['scores']['mae'] for k in range(N_FOLDS)]))
    return out


def stage_summarise(tasks):
    from scipy.stats import spearmanr
    lines = ['# Matbench view and model screen', '',
             'MAE per view and model, mean over the 5 Matbench test folds, in the target\'s unit. '
             'Lower is better.', '']
    for task in tasks:
        res = {}
        for p in sorted((RESULTS_DIR / task).glob('*__*.json')) if (RESULTS_DIR / task).is_dir() else []:
            if p.name.endswith('.partial.json'):
                continue
            r = json.loads(p.read_text())
            res[(r['view'], r['model'])] = r
        if not res:
            continue
        views = [v for v in views_for(task) if any(k[0] == v for k in res)]
        models = [m for m in MODELS if any(k[1] == m for k in res)]
        pub = published_means(task)
        lines += ['## %s' % task, '',
                  'Published means on the leaderboard\'s own features: '
                  + (', '.join('%s %.4f' % (m, v) for m, v in pub.items()) or 'not found here') + '.', '',
                  'One row is one view; a cell is mean MAE.', '',
                  '| view | features | ' + ' | '.join(models) + ' |',
                  '|---|---|' + '---|' * len(models)]
        for v in views:
            nf = next(r['n_features'] for k, r in res.items() if k[0] == v)
            cells = ['%.4f' % res[(v, m)]['mean_mae'] if (v, m) in res else '' for m in models]
            lines.append('| %s | %s | %s |' % (v, nf, ' | '.join(cells)))
        lines.append('')
        y, _ = load_targets(task)
        for m in models:
            errs = {}
            for v in views:
                f = Path(str(result_path(task, v, m).with_suffix('')) + '.oof.npy')
                if f.exists():
                    errs[v] = np.abs(np.load(f) - y)
            if len(errs) < 2:
                continue
            vs = list(errs)
            lines += ['### error correlation, %s, %s' % (m, task), '',
                      'Spearman correlation between two views\' absolute errors over every compound '
                      '(out-of-fold). 1 = the same compounds are hard for both.', '',
                      '| | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
            for a in vs:
                lines.append('| %s | %s |' % (a, ' | '.join(
                    '%.2f' % (1.0 if a == b else spearmanr(errs[a], errs[b])[0]) for b in vs)))
            lines.append('')
        mats = {}
        for v in views:
            X = np.load(view_path(task, v)).astype(float)
            X = (X - X.mean(0)) / (X.std(0) + 1e-12)
            mats[v] = X
        lines += ['### linear CKA, %s' % task, '',
                  'Linear centred kernel alignment between standardised view matrices, all compounds. '
                  '1 = the same information up to a linear map, 0 = unrelated.', '',
                  '| | ' + ' | '.join(views) + ' |', '|---|' + '---|' * len(views)]
        for a in views:
            lines.append('| %s | %s |' % (a, ' | '.join('%.2f' % linear_cka(mats[a], mats[b]) for b in views)))
        lines.append('')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'summary.md').write_text('\n'.join(lines) + '\n')
    log('summary written: %s' % (RESULTS_DIR / 'summary.md'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['targets', 'evaluate', 'summarise'], required=True)
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--views', default=None, help='comma list; default every view the task has')
    ap.add_argument('--models', default='rf')
    a = ap.parse_args()
    tasks = a.tasks.split(',')
    if a.stage == 'targets':
        stage_targets(tasks)
    elif a.stage == 'evaluate':
        for task in tasks:
            views = a.views.split(',') if a.views else views_for(task)
            for v in views:
                for m in a.models.split(','):
                    evaluate_one(task, v, m)
    else:
        stage_summarise(tasks)


if __name__ == '__main__':
    main()
