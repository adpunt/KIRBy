#!/usr/bin/env python3
"""Smoke for n26: label-noise variance from repeat measurements.

What has to be true, checked on a toy dataset whose noise is known:

  1. THE FITS RECOVER A KNOWN LINE. Squared residuals built as s2u + s2e / k
     with no scatter give back s2e as the slope and s2u as the intercept, under
     the least-squares fit and the binned fit.
  2. THE DIRECT ESTIMATE RECOVERS THE NOISE THE TOY WAS GIVEN. Each measurement
     carries noise of a known variance; the pooled within-molecule variance
     lands near it.
  3. THE ICC AND THE SPLIT-HALF CORRELATION AGREE WITH THEIR DEFINITIONS. Two
     identical measurements per molecule give an ICC of 1; Spearman-Brown at a
     count of 1 returns the correlation itself.
  4. THE ESTIMATOR CHECK MOVES THE SLOPE THE RIGHT WAY. Adding more noise per
     measurement raises the fitted slope.
  5. A DATASET WITH NO REPEATS IS SKIPPED, not run on made-up counts.
  6. THE HYBRID SPACE IS REACHED when the pool returns a recipe, and every
     model has one out-of-fold prediction per molecule in it.
  7. The command line parses.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Two OpenMP runtimes in one process kill it with exit 139 and no traceback --
# project_fold_hook_and_lgbm_segfault_2026_09_02. Both switches, before numpy.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import subprocess
import tempfile

import numpy as np

import experiments.noise.n26_noise_variance_from_repeats as n26
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n26_smoke_')
FAIL = []

TRUE_S2E = 0.30


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n_mol=240, n_feat=4, seed=0, s2e=TRUE_S2E):
    """Molecules measured 1 to 5 times, each measurement carrying noise s2e.

    One row per measurement. The scaffold groups hold two molecules each, so a
    molecule never sits on both sides of a fold.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_mol, n_feat))
    w = rng.normal(size=n_feat)
    y_true = X @ w
    k = rng.integers(1, 6, size=n_mol)
    rows, labels, rep_groups, groups = [], [], [], []
    for i in range(n_mol):
        for _ in range(int(k[i])):
            rows.append(i)
            labels.append(y_true[i] + rng.normal(0, np.sqrt(s2e)))
            rep_groups.append(i)
            groups.append(i // 2)
    rows = np.asarray(rows)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {
        'labels': np.asarray(labels),
        'reps': {'toy': X[rows], 'second': X2[rows]},
        'groups': np.asarray(groups),
        'replicate_groups': np.asarray(rep_groups),
        'task_type': 'regression',
        'metric': 'rmse',
    }


def fake_recipe(rep_names, budget=4):
    """A fixed recipe standing in for the precomputed pool, as _smoke_n25 does."""
    bp = Blueprint(reps=list(rep_names), allocation_method='fixed',
                   importance_method='lightgbm_gain', budget=budget,
                   internal_evaluator='lightgbm', metric='rmse',
                   lower_is_better=True, task_type='regression',
                   augmentation_strategy='none')
    meta = {'best_config': {'blueprint': bp},
            'objective': 'predictive', 'objective_choice': 'smoke_fixed_recipe',
            'pool_subset': 'predictive_and_similarity', 'tier': 'general',
            'aug_condition': 'none', 'by_objective': {}}
    return lambda *a, **k: (None, meta)


def no_recipe(*a, **k):
    return None, None


def run_it(data, name, recipe=no_recipe, **kw):
    real = n26.get_best_search_for_downstream
    n26.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_reps', {m: ['toy', 'second']
                                    for m in kw.get('models', ['rf'])})
        return n26.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n26.get_best_search_for_downstream = real


def test_fits():
    print("the fits recover a known line:")
    k = np.repeat(np.arange(1, 6), 20).astype(float)
    r2 = 0.2 + 0.5 / k
    o = n26.fit_ols(r2, 1.0 / k)
    check('ols slope is s2e', abs(o['slope_s2e'] - 0.5) < 1e-9,
          f"{o['slope_s2e']:.6f}")
    check('ols intercept is s2u', abs(o['intercept_s2u'] - 0.2) < 1e-9,
          f"{o['intercept_s2u']:.6f}")
    b = n26.fit_binned(r2, k)
    check('binned slope is s2e', abs(b['slope_s2e'] - 0.5) < 1e-9,
          f"{b['slope_s2e']:.6f}")
    g = n26.fit_gamma(r2 * np.random.default_rng(0).gamma(50, 1 / 50, r2.size),
                      1.0 / k)
    check('gamma fit returns a slope or a recorded failure',
          np.isfinite(g['slope_s2e']) or bool(g['failed']), str(g))
    flat = n26.fit_ols(r2, np.ones_like(k))
    check('a single count is a recorded failure, not a number',
          flat['failed'] is not None and not np.isfinite(flat['slope_s2e']))


def test_direct_icc_split():
    print("direct estimate, ICC and split-half:")
    data = make_data(n_mol=600, seed=1)
    vals = n26.measurements_by_molecule(data['labels'], data['replicate_groups'])
    d = n26.direct_noise_variance(vals)
    check('direct s2e near the injected per-measurement variance',
          abs(d['s2e'] - TRUE_S2E) < 0.05, f"{d['s2e']:.4f} vs {TRUE_S2E}")
    same = [np.array([v, v]) for v in np.linspace(0, 5, 30)]
    icc = n26.one_way_icc(same)
    check('identical repeats give ICC 1', abs(icc['icc'] - 1.0) < 1e-9,
          f"{icc['icc']}")
    sh = n26.split_half(vals, 3, seed=0)
    r = sh['repeats'][0]
    check('Spearman-Brown at count 1 is the correlation itself',
          abs(r['spearman_brown_projected']['1'] - r['r_one_measurement']) < 1e-12)
    check('split-half writes every repeat', len(sh['repeats']) == 3)


def test_run():
    print("a run, single representations only:")
    data = make_data(seed=2)
    res = run_it(data, 'toy_base', models=['rf'],
                 injected_variances=[0.2, 1.0], n_split_half_repeats=3)
    check('only the two single-representation spaces ran',
          res['spaces_run'] == ['base_1', 'base_2'], str(res['spaces_run']))
    check('which representations they are is recorded',
          res['base_reps_by_model'] == {'rf': ['toy', 'second']},
          str(res['base_reps_by_model']))
    row = res['per_space']['base_1']['rf']
    n_mol = len(np.unique(data['replicate_groups']))
    check('every molecule has an out-of-fold prediction',
          row['molecules_with_a_prediction'] == n_mol,
          f"{row['molecules_with_a_prediction']} of {n_mol}")
    check('base_2 is a different feature space from base_1',
          res['per_space']['base_2']['rf']['mean_squared_residual']
          != row['mean_squared_residual'])
    check('the estimator check used rf in its rank-1 representation',
          res['estimator_check']['space'] == 'base_1'
          and res['estimator_check']['rep'] == 'toy',
          str({k: res['estimator_check'].get(k) for k in ('space', 'rep')}))
    check('three folds scored', len(row['fold_scores']) == 3)
    bins = row['error_by_count']
    check('error-by-count bins hold every molecule',
          sum(b['molecules'] for b in bins.values()) == n_mol)
    chk = res['estimator_check']['per_injected_variance']
    check('estimator check ran at both injected variances', len(chk) == 2,
          str(list(chk)))
    if len(chk) == 2:
        check('more injected noise gives a larger slope change',
              chk['1']['slope_change'] > chk['0.2']['slope_change'],
              f"{chk['0.2']['slope_change']:+.4f} then "
              f"{chk['1']['slope_change']:+.4f}")
    check('the result file was written',
          os.path.exists(os.path.join(OUT, 'n26_noise_variance_from_repeats',
                                      'toy_base.json')))

    print("a run with a hybrid recipe:")
    res = run_it(data, 'toy_hybrid', recipe=fake_recipe(['toy', 'second']),
                 models=['rf', 'lightgbm'], injected_variances=[0.2],
                 n_split_half_repeats=2)
    check('the hybrid space ran', 'hybrid' in res['spaces_run'],
          str(res['spaces_run']))
    for m in ('rf', 'lightgbm'):
        got = res['per_space'].get('hybrid', {}).get(m, {})
        check(f'{m} has a prediction for every molecule in the hybrid',
              got.get('molecules_with_a_prediction') == n_mol,
              str(got.get('molecules_with_a_prediction')))
    check('the slope spread across spaces is written',
          'rf' in res['slope_across_spaces'])

    print("no stored search and no representations given:")
    res = run_it(data, 'toy_no_search', recipe=fake_recipe(['toy', 'second']),
                 models=['rf'], injected_variances=[0.2],
                 n_split_half_repeats=2, base_reps=None)
    check('no single-representation space, and the reason is recorded',
          not any(s.startswith('base_') for s in res['spaces_run'])
          and 'rf' in res['base_reps_missing'],
          f"{res['spaces_run']}, {res['base_reps_missing']}")
    check('the hybrid still ran', 'hybrid' in res['spaces_run'])

    print("a dataset with no repeats:")
    flat = make_data(seed=3)
    flat.pop('replicate_groups')
    res = run_it(flat, 'toy_flat', models=['rf'])
    check('skipped with the reason named',
          res.get('skipped') == 'no_replicate_structure', str(res))


def test_help():
    print("command line:")
    p = subprocess.run([sys.executable, n26.__file__, '--help'],
                       capture_output=True, text=True)
    check('--help exits 0', p.returncode == 0, p.stderr[-300:])
    for flag in ('--n-split-half-repeats', '--injected-variances',
                 '--tag'):
        check(f'{flag} is offered', flag in p.stdout)
    check('--base-rep is gone', '--base-rep' not in p.stdout)


if __name__ == '__main__':
    test_fits()
    test_direct_icc_split()
    test_run()
    test_help()
    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} — {', '.join(FAIL)}")
        sys.exit(1)
    print("ALL GREEN")
