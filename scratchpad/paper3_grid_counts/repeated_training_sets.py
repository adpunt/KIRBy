#!/usr/bin/env python3
"""Which results of a noise piece fit the same training set under different names.

Adelaide, 2026-11-04: "just make sure the same thing doesn't get run multiple
times". Every result in n44, n45, n46 and n47 is scored by n07.dprime_rows ->
d_vs_dprime.score_d_dprime, which fits the model on cleaned['X'],
cleaned['y'] and cleaned['sample_weight']. This script wraps those two
functions, hashes that training set for every result, and groups results by
(piece, model, held-out molecules, training set). A group with more than one
result name is one training set fitted, or read from the saved fit, under
several names.

Modes:
  n44      one base_1 piece and one hybrid piece of n44.run_unit on made-up
           data (as _smoke_n44.py), with the neighbour settings and correction
           targets the real run builds, so neighbour cleanups are included.
  <path>   any smoke script, run as it is, recorded the same way.

Output: a summary on stdout and every group in <out>/<mode>_groups.json.
Run from the repo root:
  PYTHONPATH=src:tests:tests/experiments python scratchpad/paper3_grid_counts/repeated_training_sets.py n44
"""
import collections
import json
import os
import runpy
import sys
import tempfile
import time
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'KIRBY_LGBM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
BASE = ('/Volumes/seagate/kirby_smoke_logs/2026-11-16' if os.path.isdir('/Volumes/seagate') else '/private/tmp/claude-501/-Users-apunt-repos-KIRBy/3fc4ec58-e2f2-442a-bf1c-462c2f505d43/scratchpad/logs')
os.makedirs(BASE, exist_ok=True)
_tmp = tempfile.mkdtemp(prefix='repeats_', dir=BASE)
os.environ.setdefault('KIRBY_CACHE_DIR', os.path.join(_tmp, 'cache'))
for p in ('src', 'tests', 'tests/experiments'):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
from experiments.noise import n07_remediation as n07  # noqa: E402
from experiments.noise import d_vs_dprime as dvd  # noqa: E402
from experiments.noise.noise_cache import array_hash  # noqa: E402

RECORDS = []          # one per scored result (twins included)
_pending = []         # training-set hash from score_d_dprime, named by rows_for
_orig_score, _orig_rows = dvd.score_d_dprime, dvd.rows_for


def _score(dataset, cleaned, X_tr, y_tr_D, X_va, y_va, **kw):
    if cleaned is None:
        h = 'refused'
    else:
        sw = cleaned.get('sample_weight')
        h = '|'.join((array_hash(np.asarray(cleaned['X'])),
                      array_hash(np.asarray(cleaned['y'], dtype=float)),
                      'none' if sw is None else array_hash(np.asarray(sw, dtype=float))))
    _pending.append((h, array_hash(np.asarray(X_va)), tuple(kw.get('models') or ()),
                     array_hash(np.asarray(X_tr)), array_hash(np.asarray(y_tr_D, dtype=float))))
    return _orig_score(dataset, cleaned, X_tr, y_tr_D, X_va, y_va, **kw)


def _rows(scores, *, metric, **identity):
    if _pending:
        h, va, models, xd, yd = _pending.pop()
        piece = tuple(identity.get(k) for k in ('dataset', 'experiment', 'noise_condition',
                                                'level', 'draw', 'fold', 'model_input'))
        name = f"{identity.get('method')}|{identity.get('method_setting')}|nb={identity.get('similarity_setting')}"
        RECORDS.append({'piece': piece, 'models': models, 'X_va': va, 'train': h,
                        'is_D': h == f'{xd}|{yd}|none', 'name': name})
    return _orig_rows(scores, metric=metric, **identity)


dvd.score_d_dprime = _score
dvd.rows_for = _rows


def run_n44_pieces():
    from experiments.noise import n44_detect_and_clean as dc
    from experiments.noise import n42_similarity_noise_sweep as n42
    from experiments.noise import n23_uncertainty_cleaning as n23
    rng = np.random.RandomState(0)
    N, P = 160, 20
    X = rng.normal(size=(N, P)).astype(np.float32)
    Xh = np.hstack([X[:, :10], rng.normal(size=(N, 10)).astype(np.float32)])
    beta = rng.normal(size=P)
    y_clean = X @ beta + 0.3 * rng.normal(size=N)
    cmask = np.zeros(N, dtype=bool)
    cmask[rng.choice(N, 16, replace=False)] = True
    y0 = y_clean.copy()
    y0[cmask] += 4.0 * y_clean.std() * rng.choice([-1, 1], size=16)
    groups = np.arange(N) // 4
    tr, va = np.arange(128), np.arange(128, N)
    rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
    metric = ('rmse', rmse, True)
    # The neighbour settings a run without SMILES gets in the two spaces this
    # made-up data has, base_1 and hybrid; the screen keeps 3 either way.
    settings = [s for s in n07.sweep_settings('synthetic', N, has_smiles=False)
                if s.space in ('base_1', 'hybrid')]
    ks = n07.sweep_ks()
    for mi, Xm in (('base_1', X), ('hybrid', Xh)):
        sink = n07.RowSink(_tmp, dc.EXP_ID, 'made_up', f'outlier_p05__{mi}')
        errors = n42._Errors()
        u = n42._Unit(exp_id=dc.EXP_ID, twins=False, ds='made_up', sink=sink,
                      metric=metric, task_type='regression', is_clf=False,
                      condition='outlier_p05', level=1.0, draw=0, fold=0, model='rf',
                      mi=f'rf:{mi}', X_tr=Xm[tr], X_va=Xm[va], y0=y0[tr],
                      y_clean=y_clean[tr], y_val=y_clean[va], g_tr=groups[tr],
                      train_idx=tr, val_idx=va, prov={}, cliff_train=None,
                      rand_train=None, measurement_counts=None, row_models=('rf',),
                      takes_weight=True, learner_factories={}, errors=errors,
                      unavailable={}, counts={}, fold_similarity=None)
        u.cmask = cmask[tr]
        u.detection = {}
        spaces = {'own': dc._Space('own', Xm[tr], Xm[va], u)}
        if mi == 'base_1':
            spaces['hybrid'] = dc._Space('hybrid', Xh[tr], Xh[va], u)
        targets = {name: n23.correction_targets_cached(
            'made_up', sp.X_tr, y0[tr], groups[tr], n07.targets_needed(list(n07.CLEANUPS)))
            for name, sp in spaces.items()}
        sim = n42._Similarity(u, settings, ks,
                              {'model_input': Xm[tr], 'base_1': X[tr], 'hybrid': Xh[tr]},
                              {'model_input': Xm[va], 'base_1': X[va], 'hybrid': Xh[va]},
                              {}, {}, None, None)
        t0 = time.time()
        dc.run_unit(u, Xm[tr], Xm[va], spaces, sim, targets)
        print(f'  piece rf:{mi}: {u.rows_written()} rows in {time.time() - t0:.0f}s; '
              f'{len(sim.kept)} neighbour settings kept; '
              f'{len(errors.d)} distinct errors', flush=True)
        for k, v in errors.d.items():
            print(f'    error {k}: {v["count"]}x {v["first"][:140]}')


def family(name):
    """detector/cleanup of an n44 name, else the method."""
    m = name.split('|')[0]
    parts = m.split('/')
    return '/'.join(parts[1:3]) if len(parts) >= 3 and parts[0] == 'n44' else m


def report(mode):
    by = collections.defaultdict(list)
    refused = 0
    for r in RECORDS:
        if r['train'] == 'refused':
            refused += 1
            continue
        by[(r['piece'], r['models'], r['X_va'], r['train'])].append(r)
    pieces = sorted({r['piece'] for r in RECORDS}, key=str)
    print(f'\n{mode}: {len(RECORDS)} results, {refused} refused (no training set)')
    out = []
    for pc in pieces:
        groups = [g for k, g in by.items() if k[0] == pc]
        n_res = sum(len(g) for g in groups)
        rep = [g for g in groups if len(g) > 1]
        extra = sum(len(g) - 1 for g in rep)
        isD = sum(len(g) for g in groups if g[0]['is_D'])
        print(f'  piece {pc}: {n_res} results with a training set, {len(groups)} distinct '
              f'training sets, {extra} results repeat a training set already scored in this piece; '
              f'{isD} results are the damaged training set itself, unchanged')
        pat = collections.Counter()
        for g in rep:
            fams = tuple(sorted({family(r['name']) for r in g}))
            pat[(fams if len(fams) <= 3 else fams[:3] + (f'... {len(fams)} pairs',),
                 g[0]['is_D'])] += len(g) - 1
            out.append({'piece': list(map(str, pc)), 'is_D': g[0]['is_D'],
                        'names': [r['name'] for r in g]})
        for (fams, isd), n in pat.most_common(25):
            print(f'    {n:5d} repeats  {"[unchanged D] " if isd else ""}{" + ".join(fams)}')
    with open(os.path.join(BASE, f'{mode}_groups.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'  every group: {os.path.join(BASE, mode + "_groups.json")}')


if __name__ == '__main__':
    mode = sys.argv[1]
    t0 = time.time()
    if mode == 'n44':
        run_n44_pieces()
    else:
        sys.argv = [mode]
        try:
            runpy.run_path(mode, run_name='__main__')
        except SystemExit:
            pass
        mode = os.path.basename(mode).replace('.py', '')
    report(mode)
    print(f'{time.time() - t0:.0f}s in all')
