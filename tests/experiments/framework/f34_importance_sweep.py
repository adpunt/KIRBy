#!/usr/bin/env python3
"""f34 with feature importance weighting added, and neighbour counts above 40.

Same datasets, representations, folds, hybrids and scoring as f34_full_sweep. The
grid is the shortlist taken from that run, plus one new axis:

  importance weighting  each feature is multiplied by its importance score raised
                        to a power, then the distance is computed. Scores come
                        from the three methods find_best_hybrid searches over
                        (DEFAULT_IMPORTANCE_METHODS: deeplift, lightgbm_gain,
                        treeshap), at power 0.5 and power 1, plus no weighting.
                        That is 7 variants.
  neighbour weighting   rank, inverse_square
  neighbour count       10, 20, 40, 50, 60, 80, 100, 150, 200
  coefficients          binary: tanimoto, dice, euclidean
                        count: soergel, tanimoto, manhattan
                        continuous: cosine, correlation, manhattan, minkowski_p0.5

HOW THE WEIGHT IS APPLIED. The rule s05 and the other similarity experiments use
(_shared.importance_feature_weights): clip negative scores to 0, raise to the
power, multiply the features. The same power for every coefficient. Continuous
features are standardised first; fingerprints stay raw, so a weighted fingerprint
gives the weighted form of tanimoto/dice (shared bits counted by weight squared).
A feature with importance 0 drops out of the distance at any power above 0.

WHERE THE SCORES COME FROM. Computed on the training molecules of the fold only,
with hybrid.compute_feature_importance, one call per (representation, method):
the call find_best_hybrid makes. A hybrid's features take the score of the
source-representation feature they were selected from. The scores a hybrid's
recipe stores are not reused: they exist for one method only, and a
representation the hybrid takes whole stores all 1s.

Scores are cached in <out>/<dataset>/importance/, one file per (representation,
fold, method), so each is computed once whichever task gets there first.

ONE TASK = one (dataset, representation, fold, coefficient). Each writes
<out>/<dataset>/<rep>__fold<i>__<coefficient>.jsonl, one line per (coefficient,
importance variant) as it finishes; a re-run skips finished lines. A hybrid is
one task per fold with coefficient 'all': its kind is only known once it is
built (the server's were continuous; one built from fingerprint bits alone is
binary), and at 100 features every coefficient is quick.

Usage on the server (inside an interactive session):
    python tests/experiments/framework/f34_importance_sweep.py --list-tasks tasks.txt
    python tests/experiments/framework/f34_importance_sweep.py --task herg mordred 0 cosine
"""
import os
import sys
import json
import time
import fcntl
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f34_full_sweep as sweep  # noqa: E402  (sets KIRBY_CACHE_DIR rules)
import f18_distance_metric_sweep as f18  # noqa: E402
from experiments._shared import (  # noqa: E402
    get_domain_metric, build_fold_hybrid, fold_hybrid_scoring_kwargs)
from kirby.hybrid import compute_feature_importance  # noqa: E402
from kirby.find_best_hybrid import DEFAULT_IMPORTANCE_METHODS  # noqa: E402
from kirby.similarity.preprocessing import normalise_weights  # noqa: E402

f34 = sweep.f34
IMPORTANCE_METHODS = tuple(DEFAULT_IMPORTANCE_METHODS)
POWERS = (0.5, 1.0)
VARIANTS = (('none', 0.0),) + tuple(
    (m, a) for m in IMPORTANCE_METHODS for a in POWERS)
WEIGHTING = ('rank', 'inverse_square')
K_VALUES = (10, 20, 40, 50, 60, 80, 100, 150, 200)
COEFFS = {
    'binary': ('tanimoto', 'dice', 'euclidean'),
    'count': ('soergel', 'tanimoto', 'manhattan'),
    'continuous': ('cosine', 'correlation', 'manhattan', 'minkowski_p0.5'),
}
OUT_DEFAULT = sweep.OUT_DEFAULT.parent / 'f34_importance_sweep'
POOL_ASK = 'full'      # --local-pool switches to f34's local 'narrow' ask


# --variant runs one importance variant of a task, so a slow task's 7
# variants can run side by side. Each variant has its own partial directory.
ONLY_VARIANT = None


def variant_name(method, power):
    return 'none' if method == 'none' else f'{method}^{power:g}'


def importance(out_dir, dataset, rep, fold_i, method, X_tr, y_tr):
    """One (representation, fold, method) score vector, computed once.

    A lock file serialises the tasks that need the same vector: the first
    computes and saves it, the rest wait and read it.
    """
    d = Path(out_dir) / dataset / 'importance'
    d.mkdir(parents=True, exist_ok=True)
    path = d / f'{rep}__fold{fold_i}__{method}.npy'
    with open(d / f'{rep}__fold{fold_i}__{method}.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if path.exists():
            return np.load(path)
        t = time.time()
        v = np.asarray(compute_feature_importance(
            {rep: np.asarray(X_tr, dtype=np.float64)}, y_tr,
            method=method)[rep], dtype=np.float64)
        if v.shape != (X_tr.shape[1],):
            raise RuntimeError(f'{method} on {rep} gave {v.shape} scores for '
                               f'{X_tr.shape[1]} features')
        tmp = path.with_suffix('.tmp.npy')
        np.save(tmp, v)
        tmp.rename(path)
        print(f'  importance {rep} fold {fold_i} {method}: '
              f'{time.time() - t:.0f}s', flush=True)
        return v


def hybrid_with_sources(space, data, labels, groups, fold_i, tr, va,
                        metric_name, metric_fn, lower, task_type):
    """The hybrid's train/held-out matrices and, per feature, its source.

    Built with the same call f34.fold_matrices makes. Returns the list of
    (source representation, its column indices) in the order the hybrid stacks
    them, and checks that order against the matrix itself.
    """
    blueprint, recipe = sweep.hybrid_blueprint(
        space, fold_i, data, labels, groups, metric_name,
        sweep.POOL_ASKS[POOL_ASK])
    reps = data['reps']
    y_tr = labels[tr]
    Xt, Xv, feat = build_fold_hybrid(
        {n: r[tr] for n, r in reps.items()},
        {n: r[va] for n, r in reps.items()},
        y_tr, blueprint, budget_override=f34.BUDGET,
        groups=(groups[tr] if groups is not None else None),
        **fold_hybrid_scoring_kwargs(
            blueprint,
            model=f34._knn(task_type)(n_neighbors=min(f34.K_VALUES[0],
                                                      len(y_tr) - 1)),
            metric_fn=metric_fn, lower_is_better=lower))
    if feat.get('augmentation_info'):
        raise RuntimeError(f'{space} fold {fold_i}: the hybrid carries '
                           'augmentation features, which have no source '
                           'representation to take an importance score from')
    sources = [(n, np.asarray(info['selected_indices'], dtype=int))
               for n, info in feat.items()
               if isinstance(info, dict) and 'selected_indices' in info]
    width = sum(len(i) for _, i in sources)
    Xt = np.asarray(Xt, dtype=np.float64)
    if width != Xt.shape[1]:
        raise RuntimeError(f'{space} fold {fold_i}: {Xt.shape[1]} hybrid '
                           f'features, {width} selected indices')

    # Each hybrid feature must be its source feature up to a rescale. Compare
    # the two after z-scoring both; a constant feature is all zeros either way.
    def z(M):
        M = np.asarray(M, dtype=np.float64)
        sd = M.std(0)
        sd[sd == 0] = 1.0
        return (M - M.mean(0)) / sd
    src = np.hstack([np.asarray(reps[n], dtype=np.float64)[tr][:, i]
                     for n, i in sources])
    src = np.nan_to_num(np.clip(src, -1e10, 1e10), nan=0.0)
    err = float(np.abs(z(Xt) - z(src)).max())
    if err > 1e-6:
        raise RuntimeError(f'{space} fold {fold_i}: hybrid features do not '
                           f'match their source features (max {err:.2e})')
    return Xt, np.asarray(Xv, dtype=np.float64), sources, recipe


def done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            keys.add((r['coefficient'], r['importance']))
    return keys


def run_task(dataset, space, fold_i, cname, out_dir):
    t0 = time.time()
    out = Path(out_dir) / dataset / f'{space}__fold{fold_i}__{cname}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    if cname != 'all' and (len(done) >= len(VARIANTS) or
                           (cname, ONLY_VARIANT) in done):
        return
    data, labels, groups, folds = sweep.load(dataset)
    task_type = data['task_type']
    metric_name, metric_fn, lower = get_domain_metric(data)
    tr, va = folds[fold_i]
    y_tr, y_va = labels[tr], labels[va]

    recipe = None
    if space in sweep.HYBRIDS:
        Xt, Xv, sources, recipe = hybrid_with_sources(
            space, data, labels, groups, fold_i, tr, va, metric_name,
            metric_fn, lower, task_type)
    else:
        X = np.asarray(data['reps'][space], dtype=np.float64)
        Xt, Xv = X[tr], X[va]
        sources = [(space, np.arange(X.shape[1]))]

    kind = f18.classify_representation(np.vstack([Xt, Xv]))
    if cname == 'all':
        cnames = COEFFS[kind]
    elif cname in COEFFS[kind]:
        cnames = (cname,)
    else:
        raise RuntimeError(f'{cname} is not on the {kind} list')
    if kind == 'continuous':
        sc = StandardScaler().fit(Xt)
        A = np.nan_to_num(sc.transform(Xt), nan=0.0, posinf=0.0, neginf=0.0)
        B = np.nan_to_num(sc.transform(Xv), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        A, B = Xt, Xv
    print(f'[{dataset} {space} fold {fold_i} {cname}] {kind}, {Xt.shape[1]} '
          f'features, {len(tr)} train / {len(va)} held out, {metric_name}',
          flush=True)
    for c in cnames:
        run_coefficient(dataset, space, fold_i, c, kind, A, B, y_tr, y_va,
                        sources, data, tr, task_type, metric_name, metric_fn,
                        lower, recipe, done, out, out_dir, Xt.shape[1])
    print(f'[{dataset} {space} fold {fold_i} {cname}] done in '
          f'{time.time() - t0:.0f}s', flush=True)


def run_coefficient(dataset, space, fold_i, cname, kind, A, B, y_tr, y_va,
                    sources, data, tr, task_type, metric_name, metric_fn, lower,
                    recipe, done, out, out_dir, n_features):
    fn = f18.coefficients_for(kind)[cname][0]
    fn = sweep.FAST.get(cname, fn)
    for method, power in VARIANTS:
        vname = variant_name(method, power)
        if (cname, vname) in done or ONLY_VARIANT not in (None, vname):
            continue
        rec = {'dataset': dataset, 'representation': space, 'fold': fold_i,
               'kind': kind, 'n_features': int(n_features),
               'coefficient': cname, 'importance': vname,
               'importance_method': method, 'importance_power': power,
               'metric': metric_name, 'lower_is_better': bool(lower),
               'recipe': recipe}
        tc = time.time()
        if method == 'none':
            Aw, Bw = A, B
        else:
            w = np.concatenate([
                importance(out_dir, dataset, n,
                           fold_i, method,
                           np.asarray(data['reps'][n], dtype=np.float64)[tr],
                           y_tr)[i]
                for n, i in sources])
            s = normalise_weights(w, 'mean1') ** power
            rec['n_features_zero_weight'] = int((s == 0).sum())
            Aw, Bw = A * s, B * s
        f = fn
        partial = (Path(out_dir) / dataset / 'partial'
                   / f'{space}__fold{fold_i}__{cname}__{vname}')
        if cname in sweep.FRACTIONAL:
            f = sweep._fractional_minkowski(float(cname.split('_p')[1]),
                                            partial)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            blocks = f34.blocks_from_square(f, Aw, Bw)
        if blocks is None:
            rec['skipped'] = 'undefined on this block'
        else:
            d_tr, d_va = blocks
            # Weighted fingerprints are no longer whole numbers, so two
            # identical fingerprints can come out a rounding error below 0
            # (tanimoto, dice: shared bits a hair above a row's own total).
            # sklearn refuses any negative. Clip that; raise on anything
            # larger, which would be a real fault, not rounding.
            lo = min(float(d_tr.min()), float(d_va.min()))
            if lo < -1e-9:
                raise RuntimeError(f'{cname} {vname}: distance {lo:.3e}')
            rec['min_distance_before_clip'] = lo
            d_tr, d_va = np.maximum(d_tr, 0.0), np.maximum(d_va, 0.0)
            rec['scores'] = {
                f'{wt}|k{k}': sweep.score(d_tr, y_tr, d_va, y_va, k, wt,
                                          task_type, metric_name,
                                          metric_fn)[0]
                for k in K_VALUES for wt in WEIGHTING}
        rec['seconds'] = round(time.time() - tc, 1)
        with open(out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        if partial.exists():
            import shutil
            shutil.rmtree(partial, ignore_errors=True)
        print(f'  {cname} {vname}: {rec["seconds"]}s'
              + (f' SKIPPED {rec["skipped"]}' if 'skipped' in rec else ''),
              flush=True)


def list_tasks(datasets, path):
    """One line per task, cheapest first (molecules squared x features, x20 for
    the fractional Minkowski; only the order uses it)."""
    tasks = []
    for ds in datasets:
        data, labels, _, folds = sweep.load(ds)
        n = len(labels)
        for space in list(data['reps']) + list(sweep.HYBRIDS):
            if space in sweep.HYBRIDS:
                for fi in range(len(folds)):
                    tasks.append((n * n * f34.BUDGET * 20,
                                  f'{ds} {space} {fi} all'))
                continue
            X = np.asarray(data['reps'][space], dtype=np.float64)
            kind, d = f18.classify_representation(X), X.shape[1]
            for fi in range(len(folds)):
                for c in COEFFS[kind]:
                    w = 20 if c in sweep.FRACTIONAL else 1
                    tasks.append((n * n * d * w, f'{ds} {space} {fi} {c}'))
    tasks.sort()
    Path(path).write_text('\n'.join(t for _, t in tasks) + '\n')
    print(f'{len(tasks)} tasks written to {path}')


def list_unfinished(out_dir, path):
    """One line per missing (task, importance variant), slowest first, for
    the single-coefficient tasks. Run with: xargs -L 1 ... --task < path."""
    lines = []
    for f in sorted(Path(out_dir).glob('*/*.jsonl')):
        ds = f.parent.name
        rep, fold, cname = f.stem.rsplit('__', 2)
        if cname == 'all':
            continue
        have = {i for c, i in done_keys(f) if c == cname}
        slow = cname in sweep.FRACTIONAL
        for m, pw in VARIANTS:
            v = variant_name(m, pw)
            if v not in have:
                lines.append((not slow, f'{ds} {rep} {fold[4:]} {cname} '
                                        f'--variant {v}'))
    lines.sort()
    Path(path).write_text(''.join(t + '\n' for _, t in lines))
    print(f'{len(lines)} missing (task, variant) runs written to {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(sweep.DATASETS))
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    ap.add_argument('--list-tasks', metavar='PATH')
    ap.add_argument('--local-pool', action='store_true')
    ap.add_argument('--list-unfinished', metavar='PATH')
    ap.add_argument('--variant', help='run only this importance variant')
    ap.add_argument('--task', nargs=4,
                    metavar=('DATASET', 'REP', 'FOLD', 'COEFFICIENT'))
    a = ap.parse_args()
    from experiments import dataset_policy  # noqa: F401  (fills the registry)
    global POOL_ASK, ONLY_VARIANT
    ONLY_VARIANT = a.variant
    if a.list_unfinished:
        return list_unfinished(a.out, a.list_unfinished)
    sweep.LOCAL_POOL = a.local_pool
    if a.local_pool:
        POOL_ASK = 'narrow'
    if a.list_tasks:
        return list_tasks(a.datasets, a.list_tasks)
    if a.task:
        ds, space, fi, c = a.task
        run_task(ds, space, int(fi), c, a.out)
        return 0
    ap.error('give --list-tasks or --task')


if __name__ == '__main__':
    sys.exit(main())
