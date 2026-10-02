#!/usr/bin/env python3
"""f34 full sweep on the server: every kNN setting f34 defines, on each dataset's real
representation pool, the two best hybrids at a budget of 100 features, and herg as the
classification dataset.

Nothing is cut. The point of the run is to see which settings win on each
(dataset, representation), so a value left out could never be seen to win.

  weighting          all six in neighbour_weighting: uniform, inverse_sqrt,
                     inverse, inverse_square, gaussian, rank.
  neighbour count    5, 7, 10, 15, 20, 25 (f34's own), plus 30 and 40 because 25
                     was the largest f34 swept and won 7 of its 21 groups.
  coefficients       every one f18 defines for the representation's kind:
                     binary 6, count 6, continuous 10.

Uses f34's own distance code and hybrid builder, so the two runs are comparable.

ONE TASK = one (dataset, representation, fold, part). The part is 'all' for
fingerprints and hybrids; a continuous representation is split into 'main' and one
task per fractional Minkowski, because those two are the slow ones. Each task writes
<out>/<dataset>/<rep>__fold<i>__<part>.jsonl, one JSON line per coefficient as soon
as it finishes, so a task stopped by a time limit keeps what it completed and a
re-run skips it.

Usage on the server (inside an interactive session):
    python tests/experiments/framework/f34_full_sweep.py --check
    python tests/experiments/framework/f34_full_sweep.py --list-tasks tasks.txt
    python tests/experiments/framework/f34_full_sweep.py --task herg mordred 0 main
"""
import os
import sys

# f34 sets KIRBY_CACHE_DIR to the local Seagate drive when it is unset. On the
# server that would silently read an empty cache, so refuse instead.
if 'KIRBY_CACHE_DIR' not in os.environ:
    if os.path.isdir('/Volumes/seagate/kirby_simrun_cache'):
        os.environ['KIRBY_CACHE_DIR'] = '/Volumes/seagate/kirby_simrun_cache'
    else:
        sys.exit('KIRBY_CACHE_DIR is not set. Run `. setup.sh` first.')
os.environ.setdefault('KIRBY_NO_COLD_SEARCH', '1.0')

import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f34_neighbour_weighting_cross as f34  # noqa: E402
import f18_distance_metric_sweep as f18  # noqa: E402
from experiments._shared import (  # noqa: E402
    load_data, get_domain_metric, get_fold_pool_recipes)
from kirby.find_best_hybrid import nested_cv_outer_folds  # noqa: E402
from kirby.similarity import neighbour_weighting  # noqa: E402

DATASETS = ('herg', 'herg_ki', 'caco2', 'adme_mdr1')
WEIGHTING = neighbour_weighting.WEIGHTING_SCHEMES
K_VALUES = (5, 7, 10, 15, 20, 25, 30, 40)
COEFFS = {kind: tuple(f18.coefficients_for(kind))
          for kind in ('binary', 'count', 'continuous')}
HYBRIDS = dict(f34.HYBRID_LINEAGES)          # name -> distance flag
OUT_DEFAULT = (f34._ROOT / 'tests' / 'results' / 'framework'
               / 'f34_full_sweep')

# 'narrow' is what f34 asked the pool for locally: lightgbm_gain only, because the
# local result database held nothing else at a kNN budget of 100. 'full' leaves
# importance method, allocation method and representation count to the loader's
# own lists, so the best recipe is picked from everything the server holds. With
# KIRBY_NO_COLD_SEARCH=1.0 a gap raises instead of searching.
POOL_ASKS = {
    'narrow': dict(f34.POOL_ASK),
    'full': {'models': ['knn'], 'budgets': [f34.BUDGET]},
}


LOCAL_POOL = False


def load(dataset):
    # rep_names=None: the dataset's own 'general' list from DATASET_REPS, which
    # is the pool the stored searches on the server were run over. --local-pool
    # asks for f34's seven local representations instead, to test on the Seagate
    # cache.
    data = load_data(dataset, rep_names=(list(f34.LOCAL_REP_POOL)
                                         if LOCAL_POOL else None))
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    groups = None if groups is None else np.asarray(groups)
    folds = nested_cv_outer_folds(data['reps'], labels, groups, f34.N_FOLDS,
                                  f34.FOLD_SEED, data['task_type'])
    return data, labels, groups, folds


def hybrid_blueprint(lineage, fold_i, data, labels, groups, metric_name,
                     pool_ask):
    pool = get_fold_pool_recipes(
        data['reps'], labels, groups, fold_i=fold_i, n_outer_folds=f34.N_FOLDS,
        downstream_model='knn', budget=f34.BUDGET,
        dataset_name=data['dataset_name'], data=data, metric=metric_name,
        task_type=data['task_type'], domain=data.get('domain', 'molecular'),
        distance=HYBRIDS[lineage], pool_subset=f34.POOL_SUBSET, n_hybrids=1,
        internal_evaluators=f34.POOL_EVALUATOR[lineage], **pool_ask)
    n_pool = pool.get('n_folds_in_pool')
    if n_pool is not None and int(n_pool) != f34.N_FOLDS:
        raise RuntimeError(f'the pool holds {n_pool} folds, this run splits '
                           f'into {f34.N_FOLDS}')
    hybrids = pool.get('hybrids') or []
    if not hybrids:
        raise RuntimeError('the pool held no kNN recipe at a budget of '
                           f'{f34.BUDGET} features')
    top = hybrids[0]
    recipe = {k: top.get(k) for k in (
        'objective', 'tier', 'allocation_method', 'importance_method',
        'internal_evaluator', 'budget', 'metric_name', 'score_other_folds')}
    return top['blueprint'], recipe


def score(d_tr, y_tr, d_va, y_va, k, scheme, task_type, metric_name,
          metric_fn):
    k_eff = min(k, len(y_tr) - 1)
    model = f34._knn(task_type)(n_neighbors=k_eff, metric='precomputed',
                                **neighbour_weighting.sklearn_kwargs(scheme))
    model.fit(d_tr, y_tr)
    # AUROC needs a score per molecule, not a 0/1 call. f34 itself used
    # predict(), which is fine for rmse and wrong for AUROC.
    if task_type == 'classification' and metric_name == 'auc':
        p = model.predict_proba(d_va)
        pred = p[:, 1] if p.shape[1] == 2 else p
    else:
        pred = model.predict(d_va)
    return float(metric_fn(y_va, pred)), int(k_eff)


# FRACTIONAL MINKOWSKI THROUGH TORCH, NOT SCIPY. scipy's cdist computes it 4x
# slower: on 2000 molecules x 200 features, scipy 71s, this 16.7s, both on one
# thread, largest difference 1.6e-10 at p=0.5 and 2.8e-7 at p=0.3 (2026-10-01).
# At herg's size, grover's 4800 features would take ~9 hours per fold per p in
# scipy. Rows are chunked so one chunk holds at most CHUNK_ELEMENTS floats.
CHUNK_ELEMENTS = 5e7


def _fractional_minkowski(p):
    import torch

    def f(X):
        T = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64))
        n, d = T.shape
        ch = max(1, int(CHUNK_ELEMENTS // max(1, n * d)))
        D = torch.empty(n, n, dtype=T.dtype)
        for s in range(0, n, ch):
            D[s:s + ch] = ((T[s:s + ch, None, :] - T[None, :, :])
                           .abs_().pow_(p).sum(-1).pow_(1.0 / p))
        return D.numpy()
    return f


FRACTIONAL = {'minkowski_p0.5': _fractional_minkowski(0.5),
              'minkowski_p0.3': _fractional_minkowski(0.3)}


def _mahalanobis_whitened(X):
    """f18's mahalanobis, computed as euclidean after whitening.

    Same covariance as f18 (EmpiricalCovariance on the same rows) and the same
    rule: undefined when there are at least as many features as rows. scipy's
    cdist does d*d work per pair, which on grover (4800 features) and herg is
    out of reach; (u-v)' P (u-v) = |L'(u-v)|^2 with P = L L' gives the same
    distance from one matrix product.
    """
    from sklearn.covariance import EmpiricalCovariance
    from scipy.spatial.distance import cdist
    n, d = X.shape
    if d >= n:
        return None
    try:
        P = EmpiricalCovariance().fit(X).precision_
    except Exception:
        return None
    w, V = np.linalg.eigh((P + P.T) / 2)
    Z = X @ (V * np.sqrt(np.clip(w, 0, None)))
    return cdist(Z, Z, metric='euclidean')


FAST = dict(FRACTIONAL, mahalanobis=_mahalanobis_whitened)

# One task's share of the coefficients. The two fractional Minkowskis are each
# their own task because on herg's largest representations one alone takes
# hours; 'main' is every other coefficient; 'all' is everything, used where no
# coefficient is slow (fingerprints and the 100-feature hybrids).
PARTS = ('all', 'main', 'minkowski_p0.5', 'minkowski_p0.3')


def coeffs_in_part(kind, part):
    if part == 'all':
        return COEFFS[kind]
    if part == 'main':
        return tuple(c for c in COEFFS[kind] if c not in FRACTIONAL)
    return (part,) if part in COEFFS[kind] else ()


def done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            keys.add((r['representation'], r['fold'], r['coefficient']))
    return keys


def run_task(dataset, space, fold_i, part, out_dir, pool_ask_name):
    t0 = time.time()
    # One file per task: many tasks run at once and appends from several
    # processes to one file on a network filesystem are not safe.
    out = Path(out_dir) / dataset / f'{space}__fold{fold_i}__{part}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    data, labels, groups, folds = load(dataset)
    task_type = data['task_type']
    metric_name, metric_fn, lower = get_domain_metric(data)
    tr, va = folds[fold_i]
    y_tr, y_va = labels[tr], labels[va]

    recipe = None
    if space in HYBRIDS:
        blueprint, recipe = hybrid_blueprint(
            space, fold_i, data, labels, groups, metric_name,
            POOL_ASKS[pool_ask_name])
        Xt, Xv = next(
            (a, b) for (fi, a, b, _, _, why) in f34.fold_matrices(
                space, data['reps'], labels, groups, [folds[fold_i]],
                blueprints={space: {0: blueprint}}, metric_fn=metric_fn,
                lower_is_better=lower, task_type=task_type))
        if Xt is None:
            raise RuntimeError(f'{space} fold {fold_i}: hybrid not built')
    else:
        X = data['reps'][space]
        Xt, Xv = X[tr], X[va]
    Xt = np.asarray(Xt, dtype=np.float64)
    Xv = np.asarray(Xv, dtype=np.float64)

    kind = f18.classify_representation(np.vstack([Xt, Xv]))
    fns = f18.coefficients_for(kind)
    sc = StandardScaler().fit(Xt)
    Xt_std = np.nan_to_num(sc.transform(Xt), nan=0.0, posinf=0.0, neginf=0.0)
    Xv_std = np.nan_to_num(sc.transform(Xv), nan=0.0, posinf=0.0, neginf=0.0)
    print(f'[{dataset} {space} fold {fold_i}] {kind}, {Xt.shape[1]} features, '
          f'{len(tr)} train / {len(va)} held out, {metric_name}', flush=True)

    for cname in coeffs_in_part(kind, part):
        if (space, fold_i, cname) in done:
            continue
        fn, needs_nonneg = fns[cname]
        fn = FAST.get(cname, fn)
        rec = {'dataset': dataset, 'representation': space, 'fold': fold_i,
               'kind': kind, 'n_features': int(Xt.shape[1]),
               'coefficient': cname, 'metric': metric_name,
               'lower_is_better': bool(lower), 'recipe': recipe}
        tc = time.time()
        if kind in ('binary', 'count') or needs_nonneg:
            A, B = Xt, Xv
        else:
            A, B = Xt_std, Xv_std
        if needs_nonneg and kind == 'continuous' and float(np.nanmin(Xt)) < 0:
            rec['skipped'] = ('undefined: requires non-negative input, this '
                              'representation has negative values')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                blocks = f34.blocks_from_square(fn, A, B)
            if blocks is None:
                rec['skipped'] = 'undefined on this block'
            else:
                d_tr, d_va = blocks
                rec['scores'] = {
                    f'{s}|k{k}': score(d_tr, y_tr, d_va, y_va, k, s, task_type,
                                       metric_name, metric_fn)[0]
                    for k in K_VALUES for s in WEIGHTING}
        rec['seconds'] = round(time.time() - tc, 1)
        with open(out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(f'  {cname}: {rec["seconds"]}s'
              + (f' SKIPPED {rec["skipped"]}' if 'skipped' in rec else ''),
              flush=True)
    print(f'[{dataset} {space} fold {fold_i}] done in '
          f'{time.time() - t0:.0f}s', flush=True)


def check(datasets, pool_ask_name):
    """Representations, widths, kinds, and a recipe per fold for each hybrid."""
    bad = []
    for ds in datasets:
        data, labels, groups, folds = load(ds)
        metric_name, _, lower = get_domain_metric(data)
        print(f'\n=== {ds}: {len(labels)} molecules, {data["task_type"]}, '
              f'{metric_name} ===')
        for name, X in data['reps'].items():
            kind = f18.classify_representation(np.asarray(X, dtype=np.float64))
            print(f'  {name:16s} {X.shape[1]:6d} features  {kind}')
        for lineage in HYBRIDS:
            for fi in range(len(folds)):
                try:
                    _, r = hybrid_blueprint(lineage, fi, data, labels, groups,
                                            metric_name,
                                            POOL_ASKS[pool_ask_name])
                    print(f'  {lineage} fold {fi}: {r["allocation_method"]}, '
                          f'{r["importance_method"]}, evaluator '
                          f'{r["internal_evaluator"]}, '
                          f'{r["metric_name"]}={r["score_other_folds"]:.4f} '
                          f'on the other folds')
                except Exception as e:
                    print(f'  {lineage} fold {fi}: MISSING '
                          f'{type(e).__name__}: {e}')
                    bad.append(f'{ds}/{lineage}/{fi}')
    print('\nREADY' if not bad else f'\nNOT READY: {len(bad)} hybrid folds '
          f'missing: {bad}')
    return 1 if bad else 0


def list_tasks(datasets, path):
    """One line per task, the slowest first so they start earliest.

    Slowness is estimated as molecules squared x features, times 20 for a
    fractional Minkowski (20 is the scipy cityblock / torch fractional ratio
    measured locally, rounded up). Only the order uses it.
    """
    tasks = []
    for ds in datasets:
        data, labels, _, folds = load(ds)
        n = len(labels)
        for space in list(data['reps']) + list(HYBRIDS):
            if space in HYBRIDS:
                kind, d = 'hybrid', f34.BUDGET
            else:
                X = np.asarray(data['reps'][space], dtype=np.float64)
                kind, d = f18.classify_representation(X), X.shape[1]
            parts = ((('main', 1), ('minkowski_p0.5', 20),
                      ('minkowski_p0.3', 20))
                     if kind == 'continuous' else (('all', 1),))
            for fi in range(len(folds)):
                for part, w in parts:
                    tasks.append((n * n * d * w, f'{ds} {space} {fi} {part}'))
    tasks.sort(reverse=True)
    Path(path).write_text('\n'.join(t for _, t in tasks) + '\n')
    print(f'{len(tasks)} tasks written to {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--pool-ask', choices=sorted(POOL_ASKS), default='full')
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--list-tasks', metavar='PATH')
    ap.add_argument('--local-pool', action='store_true')
    ap.add_argument('--task', nargs=4,
                    metavar=('DATASET', 'REP', 'FOLD', 'PART'),
                    help=f'PART is one of {PARTS}')
    a = ap.parse_args()
    from experiments import dataset_policy  # noqa: F401  (fills the registry)
    global LOCAL_POOL
    LOCAL_POOL = a.local_pool
    if a.check:
        return check(a.datasets, a.pool_ask)
    if a.list_tasks:
        return list_tasks(a.datasets, a.list_tasks)
    if a.task:
        ds, space, fi, part = a.task
        if part not in PARTS:
            ap.error(f'PART must be one of {PARTS}')
        run_task(ds, space, int(fi), part, a.out, a.pool_ask)
        return 0
    ap.error('give --check, --list-tasks or --task')


if __name__ == '__main__':
    sys.exit(main())
