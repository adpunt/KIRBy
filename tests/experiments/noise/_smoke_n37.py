#!/usr/bin/env python3
"""Smoke for n37: the label-noise share of the model's error.

What has to be true before any number n37 writes can be read:

  1. THE GAMMA INTERCEPT RECOVERS A KNOWN NOISE VARIANCE on a smooth toy
     function, at three variances, and rises by the injected variance.
  2. THE HILL INDEX of Student-t residuals sits near their degrees of freedom,
     and SD/MAD-SD is about 1 for Gaussian residuals and above 1.3 for t3.
  3. THE SPREAD REGRESSION gives R^2 near 0 when the residual spread does not
     depend on the prediction, and well above 0 when it does. No p-value key.
  4. SIMEX EXTRAPOLATION of an exactly linear curve lands on the line's value
     at lambda = -1.
  5. A FULL RUN, single representations only (base_1 'toy', base_2 'second'
     for every model): every fold has an intercept in each, the check rises
     with the level and tracks the injected variance, every model is scored,
     SIMEX runs, the out-of-fold tails are written, no errors.
  6. A FULL RUN with a hybrid recipe and repeat measurements: the hybrid space
     is scored and has its own intercept, repeat-measurement variance and its
     ratio to the MSE are written, and the count-and-spread regression is too.
  7. A classification label is skipped with its reason.
  8. The parser lists every flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import tempfile
import warnings

import numpy as np

warnings.simplefilter('ignore')

import experiments.noise.n37_noise_budget as n37
from experiments.noise.noise_helpers import gamma_test
from experiments.noise.n35_near_duplicate_disagreement import (
    euclidean_distance_matrix)
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n37_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=180, n_feat=4, seed=0, noise_sd=0.3, replicates=False):
    """A smooth function of 4 features plus Gaussian noise; 30 scaffold groups.

    With replicates=True, rows come in threes that share features exactly and
    differ only in their noise: three measurements of one molecule.
    """
    rng = np.random.default_rng(seed)
    if replicates:
        P = rng.uniform(size=(n // 3, n_feat))
        X = np.repeat(P, 3, axis=0)
        rep = np.repeat(np.arange(n // 3), 3)
    else:
        X = rng.uniform(size=(n, n_feat))
        rep = None
    f = np.sin(3 * X[:, 0]) + X[:, 1] + 0.5 * X[:, 2]
    y = f + rng.normal(0, noise_sd, len(f))
    g = (np.arange(len(f)) // 6) % 30 if rep is None else rep % 30
    X2 = X[:, ::-1] + rng.normal(0, 0.3, size=X.shape)
    d = {'labels': y, 'reps': {'toy': X, 'second': X2}, 'groups': g,
         'task_type': 'regression', 'domain': 'molecular'}
    if replicates:
        d['replicate_groups'] = rep
    return d


def fake_recipe(rep_names, budget=4):
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


BASE_REPS = {m: ['toy', 'second'] for m in n37.DOWNSTREAM_MODELS}


def run_it(data, name, recipe=no_recipe, **kw):
    real = n37.get_best_search_for_downstream
    n37.get_best_search_for_downstream = recipe
    try:
        # Every model's top 2 single representations, given directly.
        kw.setdefault('base_reps', BASE_REPS)
        kw.setdefault('models', ['rf', 'lightgbm'])
        return n37.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n37.get_best_search_for_downstream = real


def test_pure():
    print("the pure functions:")
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(600, 3))
    f = np.sin(3 * X[:, 0]) + X[:, 1]
    D = euclidean_distance_matrix(X)
    got = []
    for s2 in (0.02, 0.1, 0.3):
        est = np.mean([gamma_test(D, f + rng.normal(0, np.sqrt(s2), 600))[
            'intercept'] for _ in range(3)])
        got.append(est)
    ok = all(abs(e - s2) < 0.25 * s2 + 0.005 for e, s2 in zip(got, (0.02, 0.1, 0.3)))
    check('1. the Gamma intercept recovers the noise variance',
          ok and got[0] < got[1] < got[2],
          ', '.join(f'{s2}: {e:.4f}' for e, s2 in zip(got, (0.02, 0.1, 0.3))))

    t3 = rng.standard_t(3, size=20000)
    h = n37.hill_index(t3, share=0.02)['alpha']
    gd = n37.tail_diagnostics(rng.normal(size=5000))
    td = n37.tail_diagnostics(t3[:5000])
    check('2. Hill alpha near 3 on t3; SD/MAD-SD about 1 on Gaussian, over 1.3 '
          'on t3; t fit recovers nu',
          2.0 < h < 4.5 and abs(gd['sd_over_mad_sd'] - 1) < 0.08
          and td['sd_over_mad_sd'] > 1.3 and 2.0 < td['student_t_nu'] < 4.5,
          f"Hill {h:.2f}, SD/MAD {gd['sd_over_mad_sd']:.3f} and "
          f"{td['sd_over_mad_sd']:.3f}, nu {td['student_t_nu']:.2f}")

    p = rng.uniform(1, 3, 3000)
    flat = n37.spread_regression(rng.normal(size=3000), p[:, None])
    grow = n37.spread_regression(rng.normal(size=3000) * p, p[:, None])
    check('3. spread regression: R^2 near 0 when flat, above 0.05 when the '
          'spread grows with the prediction; no p-value',
          flat['r_squared'] < 0.01 and grow['r_squared'] > 0.05
          and not any('p' == k or 'p_value' in k for k in grow),
          f"{flat['r_squared']:.4f} and {grow['r_squared']:.4f}; n*R^2 "
          f"{grow['n_times_r_squared']:.1f}")

    ex = n37.simex_extrapolate([0, 0.5, 1, 2], [1.0, 1.2, 1.4, 1.8])
    check('4. SIMEX: a straight line lands on its own value at -1',
          abs(ex['linear']['mse_at_minus_one'] - 0.6) < 1e-9
          and abs(ex['quadratic']['mse_at_minus_one'] - 0.6) < 1e-9
          and abs(ex['linear']['drop_share_of_plain'] - 0.4) < 1e-9)


def test_runs():
    print("\nfull runs:")
    data = make_data()
    r = run_it(data, 'toy_base', levels=[0.0, 0.5, 1.0], simex_draws=[0, 1])
    pf = r['per_fold']
    check('5a. every fold has a base intercept and every model is scored',
          len(pf) == 3 and all('intercept' in f['gamma']['base_1'] for f in pf)
          and all(set(f['model_error']['base_1']) == {'rf', 'lightgbm'} for f in pf))
    ch = r['check']
    rise5, inj5 = ch['0.5']['intercept_rise_mean'], ch['0.5']['realised_injected_variance_mean']
    rise1, inj1 = ch['1']['intercept_rise_mean'], ch['1']['realised_injected_variance_mean']
    check('5b. the check rises with the level and tracks the injected variance',
          rise5 < rise1 and abs(rise1 - inj1) < 0.35 * inj1
          and abs(rise5 - inj5) < 0.35 * inj5,
          f'level 0.5: rise {rise5:.4f} vs {inj5:.4f}; level 1: rise '
          f'{rise1:.4f} vs {inj1:.4f}')
    sx = pf[0]['simex']['base_1']['rf']
    check('5c. SIMEX ran: 4 lambda points, both extrapolations',
          len(sx['lambdas']) == 4 and 'linear' in sx and 'quadratic' in sx,
          f"mse {[round(v, 4) for v in sx['mse']]}")
    o = r['out_of_fold']['base_1']['lightgbm']
    check('5d. out-of-fold tails and spread regression written over every molecule',
          o['tails']['n'] == 180 and 'student_t_nu' in o['tails']
          and 'r_squared' in o['spread_on_prediction'])
    check('5e. forest-proximity intercept written',
          'intercept' in pf[0]['gamma'].get('forest_proximity', {}))
    check('5f. no errors', not r.get('_errors'), str(r.get('_errors', ''))[:300])
    check('5g. base_2 has its own intercept and scores; each model\'s pair '
          'and the reference representation are named',
          r['spaces_run'] == ['base_1', 'base_2']
          and all('intercept' in f['gamma']['base_2'] for f in pf)
          and all(set(f['model_error']['base_2']) == {'rf', 'lightgbm'}
                  for f in pf)
          and r['base_reps_by_model'] == {m: ['toy', 'second']
                                          for m in ['rf', 'lightgbm']}
          and r['gamma_reference_rep'] == 'toy'
          and r['gamma_reference_space'] == 'base_1'
          and 'base_rep' not in r
          and all(f['model_error']['base_2']['rf']['gamma_reference_over_mse']
                  is not None for f in pf),
          f"{r['spaces_run']}, reference {r.get('gamma_reference_rep')}")

    dr = make_data(replicates=True, seed=2)
    rr = run_it(dr, 'toy_rep', recipe=fake_recipe(['toy', 'second']),
                levels=[0.0, 0.5], simex_draws=[0], models=['lightgbm'])
    f0 = rr['per_fold'][0]
    check('6a. the hybrid space is scored and has its own intercept',
          'hybrid' in f0['model_error'] and 'intercept' in f0['gamma'].get('hybrid', {}),
          str(list(f0['model_error'])))
    me = f0['model_error']['base_1']['lightgbm']
    check('6b. repeat-measurement variance and its ratio to the MSE written',
          'repeat_noise_variance_training' in f0
          and me.get('repeat_variance_over_mse') is not None
          and 'repeat_noise_variance_whole_dataset' in rr,
          f"s2e {f0['repeat_noise_variance_training']['s2e']:.4f} (true 0.09)")
    check('6c. count-and-spread regression written',
          'spread_on_prediction_count_and_spread'
          in rr['out_of_fold']['base_1']['lightgbm'])
    check('6d. no errors', not rr.get('_errors'), str(rr.get('_errors', ''))[:300])

    dc = make_data()
    dc['task_type'] = 'classification'
    rc = n37.run(dc, {'n_splits': 3}, OUT, 'toy_clf', base_reps=BASE_REPS)
    check('7. classification skipped with its reason',
          rc.get('skipped') == 'not_regression')


def test_parser():
    print("\nthe parser:")
    import subprocess
    out = subprocess.run([sys.executable, n37.__file__, '--help'],
                         capture_output=True, text=True).stdout
    flags = ('--levels', '--draws', '--simex-lambdas', '--simex-draws',
             '--k-max', '--skip-proximity', '--skip-simex', '--exp-folds',
             '--tag')
    missing = [f for f in flags if f not in out]
    check('8. every flag is in --help', not missing, f'missing {missing}')
    check('8b. --base-rep is gone', out and '--base-rep' not in out)


if __name__ == '__main__':
    test_pure()
    test_runs()
    test_parser()
    print("\n" + ("ALL GREEN" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)
