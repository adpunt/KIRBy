#!/usr/bin/env python3
"""f35 — does the scoring path change f34's answer?

f34 measured every (weighting scheme, coefficient, neighbour count) down one
scoring path. The repo has two, and they disagree structurally, so f34's answer
is only worth acting on if it survives the other one.

  similarity_metrics/simeval.py:161 fits
  KNeighborsRegressor(n_neighbors=kk, weights='distance', metric='precomputed')
  on a distance matrix the caller computed. The coefficient is honoured.

  s04_activity_cliffs.py:372, s05_knn_prediction.py:416 and
  s10_importance_downstream.py:586 fit on raw features and pass no metric, so
  sklearn takes its own default of Minkowski at p = 2, which is Euclidean. The
  coefficient is ignored whatever KIRBY_DISTANCE_METRIC says.

THREE PATHS, same folds, same representations, same winning cell:

  features_with_coefficient   raw features, the coefficient passed through
                              sklearn's `metric` argument. This is f34's own
                              path, and the one s04, s05 and s10 will use after
                              step 2 of the handoff's plan.
  precomputed_matrix          scipy cdist under the same coefficient, then
                              metric='precomputed'. This is what
                              similarity_metrics/ does today.
  features_no_coefficient     raw features, no metric passed. This is what s04,
                              s05 and s10 do today, and it is here so the
                              report says how far the current numbers sit from
                              either of the other two.

WHAT THE ANSWER MEANS. If the first two agree, the split between the two
scoring paths is cosmetic and can be unified whenever. If they disagree, every
number in similarity_metrics/ and every number in similarity/ sit on different
footings and the methods section has to say so.

The third path is the one that decides how much re-running step 2 costs: the
gap between features_no_coefficient and features_with_coefficient is how far
today's stored numbers move when the coefficient finally reaches the kNN.

Usage:
    python tests/experiments/framework/f35_scorer_path_check.py
    python tests/experiments/framework/f35_scorer_path_check.py --datasets caco2
"""
import os

os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_simrun_cache')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ⛔ NO SEARCH MAY RUN. get_best_search_for_downstream and get_fold_pool_recipes
# read the stored result database through find_best_hybrid, which COMPUTES any
# configuration the grid asks for and the database does not hold. On 2026-09-28
# a run of this file without this line spent minutes searching and wrote 175 new
# rows into result_db/caco2.db before it was stopped. KIRBY_NO_COLD_SEARCH makes
# find_best_hybrid raise instead, and 1.0 means it raises on ANY gap rather than
# the 0.5 the bare flag gives. Lower it only to see how short a lineage is.
os.environ.setdefault('KIRBY_NO_COLD_SEARCH', '1.0')

import sys
import json
import time
import argparse
import warnings
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments._shared import (  # noqa: E402
    load_data, get_domain_metric, save_experiment_results)
from kirby.find_best_hybrid import nested_cv_outer_folds  # noqa: E402
from kirby.similarity import distance_config  # noqa: E402
from kirby.similarity import neighbour_weighting  # noqa: E402

# f34 owns the representation pool, the fold partition, the pool read and the
# per-fold matrices. Imported rather than repeated, so the two files cannot
# drift into measuring different things.
import f34_neighbour_weighting_cross as f34  # noqa: E402

RESULTS_DIR = (_ROOT / 'tests' / 'results' / 'framework'
               / 'f34_neighbour_weighting_cross')

PATHS = ('features_with_coefficient', 'precomputed_matrix',
         'features_no_coefficient')


def winning_cell(body):
    """f34's best cell over every representation, in that dataset's metric."""
    lower = bool(body.get('lower_is_better'))
    sign = 1 if lower else -1
    rows = [(space, key, e)
            for space, cells in (body.get('cells_by_representation') or {}).items()
            for key, e in cells.items()]
    if not rows:
        return None
    rows.sort(key=lambda r: r[2]['mean_of_folds'] * sign)
    space, key, e = rows[0]
    return {'cell': key, 'won_on_representation': space,
            'weighting_scheme': e['weighting_scheme'],
            'coefficient': e['coefficient'],
            'n_neighbours': e['n_neighbours'],
            'f34_mean_of_folds': e['mean_of_folds']}


def score_three_paths(X_tr, y_tr, X_va, y_va, *, scheme, coefficient, k,
                      task_type, metric_fn):
    """One score per path, on one fold of one representation."""
    knn = f34._knn(task_type)
    k_eff = min(k, len(y_tr) - 1)
    weights = neighbour_weighting.sklearn_weights(scheme)
    out = {}

    with warnings.catch_warnings():
        # A fractional Minkowski is not a true distance and sklearn says so
        # once per fit. distance_config.is_true_metric records the same fact.
        warnings.simplefilter('ignore', UserWarning)

        m = knn(n_neighbors=k_eff, weights=weights,
                **distance_config.sklearn_kwargs(coefficient))
        m.fit(X_tr, y_tr)
        out['features_with_coefficient'] = float(
            metric_fn(y_va, m.predict(X_va)))

        # The caller computes the matrix, exactly as simeval does.
        scipy_name, scipy_kw = distance_config.scipy_metric(coefficient)
        d_tr = cdist(X_tr, X_tr, metric=scipy_name, **scipy_kw)
        d_va = cdist(X_va, X_tr, metric=scipy_name, **scipy_kw)
        finite = d_tr[np.isfinite(d_tr)]
        fill = float(finite.max()) if finite.size else 1.0
        d_tr = np.nan_to_num(d_tr, nan=fill, posinf=fill, neginf=0.0)
        d_va = np.nan_to_num(d_va, nan=fill, posinf=fill, neginf=0.0)
        m = knn(n_neighbors=k_eff, weights=weights, metric='precomputed')
        m.fit(d_tr, y_tr)
        out['precomputed_matrix'] = float(metric_fn(y_va, m.predict(d_va)))

        # No metric argument at all. This is s04:372, s05:416 and s10:586.
        m = knn(n_neighbors=k_eff, weights=weights)
        m.fit(X_tr, y_tr)
        out['features_no_coefficient'] = float(
            metric_fn(y_va, m.predict(X_va)))
    return out, int(k_eff)


def run_dataset(dataset, body, args):
    t0 = time.time()
    cell = winning_cell(body)
    if cell is None:
        raise RuntimeError(f'{dataset}: the f34 result file holds no cells')
    print(f'\n=== f35 {dataset} ===')
    print(f'  f34 winning cell: {cell["weighting_scheme"]} weighting, '
          f'{cell["coefficient"]} coefficient, {cell["n_neighbours"]} '
          f'neighbours (won on {cell["won_on_representation"]})')

    data = load_data(dataset, rep_names=list(f34.LOCAL_REP_POOL))
    reps = data['reps']
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    groups = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    fold_splits = nested_cv_outer_folds(reps, labels, groups, args.folds,
                                        f34.FOLD_SEED, task_type)

    print('  reading the pool (no search runs):', flush=True)
    _top3, blueprints, notes = f34.read_pool(
        reps, labels, groups, dataset=dataset, data=data,
        metric_name=metric_name, task_type=task_type,
        domain=data.get('domain', 'molecular'), n_folds=len(fold_splits))

    spaces = list(body.get('representations_swept') or [])
    per_space = {}
    skipped = []
    for space in spaces:
        if space not in reps and space not in dict(f34.HYBRID_LINEAGES):
            skipped.append({'representation': space,
                            'reason': 'f34 swept it but it is not in the pool '
                                      'or the hybrid lineages'})
            continue
        fold_scores = {p: [] for p in PATHS}
        for (fold_i, X_tr, X_va, y_tr, y_va, why) in f34.fold_matrices(
                space, reps, labels, groups, fold_splits,
                blueprints=blueprints, metric_fn=metric_fn,
                lower_is_better=lower_is_better, task_type=task_type):
            if X_tr is None:
                skipped.append({'representation': space, 'fold': fold_i,
                                'reason': why})
                continue
            try:
                scores, _k_eff = score_three_paths(
                    X_tr, y_tr, X_va, y_va,
                    scheme=cell['weighting_scheme'],
                    coefficient=cell['coefficient'],
                    k=cell['n_neighbours'],
                    task_type=task_type, metric_fn=metric_fn)
            except Exception as e:
                skipped.append({'representation': space, 'fold': fold_i,
                                'reason': f'raised: {type(e).__name__}: {e}'})
                continue
            for p, v in scores.items():
                fold_scores[p].append(v)
        entry = {}
        for p, vals in fold_scores.items():
            arr = np.asarray(vals, dtype=np.float64)
            if not arr.size:
                continue
            entry[p] = {'fold_scores': vals,
                        'mean_of_folds': float(arr.mean()),
                        'spread_of_folds': float(arr.max() - arr.min()),
                        'n_folds': int(arr.size)}
        if ('features_with_coefficient' in entry
                and 'precomputed_matrix' in entry):
            a = entry['features_with_coefficient']['mean_of_folds']
            b = entry['precomputed_matrix']['mean_of_folds']
            entry['two_honouring_paths_differ_by'] = abs(a - b)
        if ('features_with_coefficient' in entry
                and 'features_no_coefficient' in entry):
            a = entry['features_with_coefficient']['mean_of_folds']
            c = entry['features_no_coefficient']['mean_of_folds']
            entry['what_step_2_moves'] = abs(a - c)
        per_space[space] = entry
        print(f'    [{space}] '
              + '  '.join(f'{p}={entry[p]["mean_of_folds"]:.4f}'
                          for p in PATHS if p in entry), flush=True)

    results = {
        'experiment': 'f35_scorer_path_check',
        'dataset': dataset,
        'n_molecules': int(len(labels)),
        'n_folds': len(fold_splits),
        'fold_seed': f34.FOLD_SEED,
        'f34_winning_cell': cell,
        'paths': list(PATHS),
        'path_definitions': {
            'features_with_coefficient':
                'raw features, the coefficient passed as sklearn metric= '
                '(f34s path, and what s04/s05/s10 adopt at step 2)',
            'precomputed_matrix':
                'scipy cdist under the same coefficient, then '
                "metric='precomputed' (similarity_metrics/simeval.py:161)",
            'features_no_coefficient':
                'raw features, no metric passed, so sklearn uses Minkowski '
                'p=2 (s04:372, s05:416, s10:586 as they stand today)',
        },
        'by_representation': per_space,
        'pool_notes': notes,
        'skipped': skipped,
    }
    results['verdict'] = verdict(per_space, metric_name)
    results['elapsed_seconds'] = round(time.time() - t0, 1)
    print_dataset(results, metric_name, lower_is_better)
    save_experiment_results(results, args.output, 'f35_scorer_path_check',
                            dataset=dataset, metric_name=metric_name,
                            lower_is_better=lower_is_better)
    return results


def verdict(per_space, metric_name):
    """The two gaps, summarised over the representations of one dataset."""
    honouring = [e['two_honouring_paths_differ_by'] for e in per_space.values()
                 if 'two_honouring_paths_differ_by' in e]
    step2 = [e['what_step_2_moves'] for e in per_space.values()
             if 'what_step_2_moves' in e]
    out = {'n_representations': len(per_space)}
    if honouring:
        out['two_honouring_paths'] = {
            f'largest_gap_{metric_name}': float(np.max(honouring)),
            f'median_gap_{metric_name}': float(np.median(honouring)),
            'n_representations': len(honouring),
        }
    if step2:
        out['what_step_2_moves'] = {
            f'largest_gap_{metric_name}': float(np.max(step2)),
            f'median_gap_{metric_name}': float(np.median(step2)),
            'n_representations': len(step2),
        }
    return out


def print_dataset(results, metric_name, lower_is_better):
    direction = 'lower is better' if lower_is_better else 'higher is better'
    print(f'\n  [{results["dataset"]}] one row is one representation. Every '
          f'number is the mean over {results["n_folds"]} folds in '
          f'{metric_name} ({direction}), under f34s winning cell.')
    print(f'    {"representation":26s} {"coefficient passed":>19s} '
          f'{"precomputed":>13s} {"no coefficient":>15s} '
          f'{"first two differ":>17s} {"step 2 moves":>13s}')
    for space, e in results['by_representation'].items():
        def f(p):
            return (f'{e[p]["mean_of_folds"]:>.4f}' if p in e else 'n/a')
        gap_a = e.get('two_honouring_paths_differ_by')
        gap_b = e.get('what_step_2_moves')
        print(f'    {space:26s} {f("features_with_coefficient"):>19s} '
              f'{f("precomputed_matrix"):>13s} '
              f'{f("features_no_coefficient"):>15s} '
              + (f'{gap_a:>17.4f}' if gap_a is not None else f'{"n/a":>17s}')
              + (f' {gap_b:>13.4f}' if gap_b is not None else f' {"n/a":>13s}'))
    v = results['verdict']
    if 'two_honouring_paths' in v:
        b = v['two_honouring_paths']
        print(f'\n    The two paths that honour the coefficient differ by at '
              f'most {b[f"largest_gap_{metric_name}"]:.4f} {metric_name} and '
              f'by {b[f"median_gap_{metric_name}"]:.4f} at the median, over '
              f'{b["n_representations"]} representations.')
    if 'what_step_2_moves' in v:
        b = v['what_step_2_moves']
        print(f'    Passing the coefficient at all moves the score by at most '
              f'{b[f"largest_gap_{metric_name}"]:.4f} {metric_name} and by '
              f'{b[f"median_gap_{metric_name}"]:.4f} at the median. That is '
              f'how far s04, s05 and s10 stored numbers move at step 2.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--folds', type=int, default=f34.N_FOLDS)
    ap.add_argument('--f34-results', default=str(RESULTS_DIR))
    ap.add_argument('--output',
                    default=os.environ.get(
                        'F35_OUT',
                        str(_ROOT / 'tests' / 'results' / 'framework')))
    args = ap.parse_args()

    from experiments import dataset_policy  # noqa: F401  (fills the registry)

    bodies = {}
    for path in sorted(Path(args.f34_results).glob('*.json')):
        payload = json.loads(path.read_text())
        body = payload.get('results', payload)
        ds = body.get('dataset') or path.stem
        if args.datasets and ds not in args.datasets:
            continue
        body.setdefault('metric_name', payload.get('metric_name'))
        body.setdefault('lower_is_better', payload.get('lower_is_better'))
        bodies[ds] = body
    if not bodies:
        print(f'no f34 result files under {args.f34_results}. Run '
              f'f34_neighbour_weighting_cross.py first — f35 scores the cell '
              f'f34 picked and has nothing to score without it.')
        return 2

    failed = []
    for ds, body in bodies.items():
        try:
            run_dataset(ds, body, args)
        except Exception:
            traceback.print_exc()
            failed.append(ds)
    if failed:
        print(f'\nFAILED: {failed}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
