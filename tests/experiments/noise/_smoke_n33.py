#!/usr/bin/env python3
"""Smoke for n33: robust losses, target transforms, weights and mixup.

What has to be true before any number n33 writes can be read:

  1. TUKEY'S GRADIENT IS TUKEY'S. It matches a numerical derivative of the
     biweight loss inside the threshold, is zero outside it, and the second
     derivative never falls below its floor.
  2. C-MIXUP DOUBLES THE SET AND MIXES NEAR LABELS. Every new label lies between
     the two it came from, and partners sit closer in label than random pairs.
  3. THE 1/sigma^2 WEIGHTS AVERAGE ONE and a near-zero spread is floored.
  4. s IS THE RESIDUAL SD ON NORMAL RESIDUALS, and kurtosis is 3 there.
  5. EVERY LOSS SWAP FITS AND PREDICTS ON THE LABEL SCALE, including the two
     that start from the median.
  6. A FULL RUN in the base_1 space writes every label arm, every swap and the
     NGBoost pair, a record per pass with the kurtosis, and tail scores.
  7. WITH A RECIPE, the hybrid space is built and the swaps are fitted in it.
  8. THE SWEEP WRITES ONE FILE PER CONDITION and a second call skips it.
  9. The metric direction is read off the dataset.
 10. A classification label is skipped with its reason.
 11. The parser lists every flag.
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

import experiments.noise.n33_robust_training as n33
from kirby.find_best_hybrid import Blueprint
from kirby.models.dnn import DNNRegressor

# Toy speed only. NGBoost at 500 stages and the DNN at 200 epochs took 41 s and
# 26 s per fit on 100 toy molecules; neither number is what the smoke checks.
n33.NGB_ESTIMATORS = 40
_d = list(DNNRegressor.__init__.__defaults__)
_d[3], _d[5] = 15, 5          # epochs, patience
DNNRegressor.__init__.__defaults__ = tuple(_d)

OUT = tempfile.mkdtemp(prefix='n33_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=150, n_feat=4, seed=0, domain='molecular'):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {'labels': y, 'reps': {'toy': X, 'second': X2},
            'groups': np.arange(n), 'task_type': 'regression',
            'domain': domain}


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


def run_one(data, name, recipe=no_recipe, **kw):
    real = n33.get_best_search_for_downstream
    n33.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_reps', {
            m: ['toy', 'second']
            for m in (kw.get('models') or n33.DOWNSTREAM_MODELS)})
        return n33._run_one_condition(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n33.get_best_search_for_downstream = real


def test_pieces():
    print("the pieces, no model fitted:")
    c = 1.3
    obj = n33.tukey_objective(c)
    y = np.zeros(7)
    r = np.array([-2.0, -1.0, -0.5, 0.0, 0.4, 1.0, 1.5])
    g, h = obj(y, r)

    def rho(x):
        u = np.clip(x / c, -1, 1)
        return (c * c / 6.0) * (1 - (1 - u * u) ** 3)
    eps = 1e-6
    num = (rho(r + eps) - rho(r - eps)) / (2 * eps)
    inside = np.abs(r) < c
    check('1. the Tukey gradient is the derivative of the biweight inside c',
          np.allclose(g[inside], num[inside], atol=1e-5)
          and np.all(g[~inside] == 0.0),
          f'{int(inside.sum())} inside, {int((~inside).sum())} outside')
    check('1b. the Tukey hessian never falls below its floor',
          np.all(h >= n33.TUKEY_HESS_FLOOR - 1e-15))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    yy = rng.normal(size=300)
    Xa, ya = n33.c_mixup(X, yy, 0.2 * np.std(yy), np.random.default_rng(1))
    check('2. C-Mixup doubles the training set', Xa.shape == (600, 3)
          and ya.shape == (600,) and np.array_equal(ya[:300], yy))
    # Recover each new label's partner distance by re-drawing with the same rng.
    r2 = np.random.default_rng(1)
    n = 300
    partners = np.empty(n, dtype=int)
    d = yy[:, None] - yy[None, :]
    k = np.exp(-(d * d) / (2 * (0.2 * np.std(yy)) ** 2))
    np.fill_diagonal(k, 0.0)
    k /= k.sum(axis=1, keepdims=True)
    cum = np.cumsum(k, axis=1)
    u = r2.random(n)[:, None]
    partners = np.minimum((cum < u).sum(axis=1), n - 1)
    near = float(np.mean(np.abs(yy - yy[partners])))
    rand = float(np.mean(np.abs(yy - yy[rng.permutation(n)])))
    lo = np.minimum(yy, yy[partners])
    hi = np.maximum(yy, yy[partners])
    check('2b. partners are nearer in label than random pairs, and each new '
          'label lies between its two',
          near < 0.5 * rand and np.all(ya[300:] >= lo - 1e-9)
          and np.all(ya[300:] <= hi + 1e-9),
          f'mean |dy| partners {near:.3f}, random {rand:.3f}')

    w = n33.invvar_weights(np.array([1.0, 2.0, 1e-9, 0.5]), s=1.0)
    check('3. the 1/sigma^2 weights average one and a zero spread is floored',
          abs(w.mean() - 1.0) < 1e-12
          and w[2] == w.max()
          and np.isclose(w[2] / w[0], 1.0 / n33.SIGMA_FLOOR_S ** 2))

    res = rng.normal(0, 0.7, 20000)
    s = n33.robust_scale(res, np.zeros_like(res))
    kk = n33.residual_kurtosis(res, np.zeros_like(res))
    check('4. s is the residual SD and kurtosis is 3 on normal residuals',
          abs(s - 0.7) < 0.02 and abs(kk - 3.0) < 0.15,
          f's {s:.4f} against 0.7, kurtosis {kk:.3f}')


def test_swaps():
    print("\nthe loss swaps:")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    y = 2.0 * X[:, 0] + 7.0 + rng.normal(0, 0.3, 300)
    bad = []
    for name in n33.build_swaps(n33.HUBER_C):
        m = n33.make_swap(name, 0.3, n33.HUBER_C)
        if name.startswith('dnn'):
            m.set_params(epochs=40)
        m.fit(X[:200], y[:200])
        p = np.asarray(m.predict(X[200:]), dtype=float)
        err = float(np.sqrt(np.mean((p - y[200:]) ** 2)))
        if not (err < 1.2 and abs(float(np.mean(p)) - float(np.mean(y[200:]))) < 0.6):
            bad.append((name, round(err, 3)))
    check('5. every swap fits and predicts on the label scale', not bad,
          f'off scale: {bad}' if bad else
          f'{len(n33.build_swaps(n33.HUBER_C))} swaps, label mean 7')
    m = n33.make_swap('lightgbm_huber_c1', 0.3, n33.HUBER_C)
    check('5b. the Huber threshold is c * s in label units',
          np.isclose(m.get_params()['alpha'], 0.3))


def test_full_runs():
    print("\nfull runs:")
    models = ['rf', 'xgboost', 'lightgbm', 'dnn']
    res = run_one(make_data(), 'toy', models=models, levels=[0.0, 0.5],
                  draws=[0], noise_condition='laplace')
    check('6-. spaces base_1 and base_2 ran, and the representations are '
          'named', res['spaces_run'] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {m: ['toy', 'second']
                                            for m in models}
          and 'base_rep' not in res,
          f"{res['spaces_run']}, {res.get('base_reps_by_model')}")
    check('6-b. base_2 is scored for every model',
          set(res['per_level']['0.5']['base_2']['noisy']) >= set(models))
    lv = res['per_level']['0.5']['base_1']
    missing = [a for a in res['arms'] if a not in lv]
    check('6. every label arm is scored at level 0.5', not missing,
          f'missing {missing}; errors {list(res.get("_errors", {}))[:3]}'
          if missing else f"{len(res['arms'])} arms")
    per_model_missing = [(a, m) for a in res['arms'] for m in models
                         if m not in lv.get(a, {})]
    check('6b. every arm is scored for every model', not per_model_missing,
          f'{per_model_missing[:4]}; errors '
          f'{list(res.get("_errors", {}).items())[:2]}'
          if per_model_missing else '')
    swaps = [s for s in res['model_swaps'] if s in lv['noisy']]
    check('6c. every loss swap is scored on the noisy labels',
          len(swaps) == len(res['model_swaps'])
          and 'ngboost_t' in lv['noisy'] and 'ngboost_normal' in lv['noisy'],
          f'{len(swaps)} of {len(res["model_swaps"])}')
    recs = res['per_pass']['0.5']
    check('6d. one record per pass, with s and the kurtosis',
          len(recs) == 3 and all(
              np.isfinite(r['per_space']['base_1']['residual_kurtosis'])
              and r['per_space']['base_1']['robust_scale_s_label_units'] > 0
              for r in recs))
    t = res['tails_per_level']['0.5']['base_1']['yeo_johnson']['rf']
    check('6e. tail scores are written beside each score',
          all(v and 'tail_rmse_top' in v for v in t.values()))
    sw = res['summary']['per_level']['0.5']['base_1']['model_swaps']
    check('6f. each swap is paired against its parent',
          sw.get('lightgbm_tukey', {}).get('replaces') == 'lightgbm'
          and sw.get('ngboost_t', {}).get('replaces') == 'ngboost_normal'
          and sw['rf_l1']['vs_parent_on_noisy']['n_passes_paired'] == 3)
    check('6g. at level 0.0 one draw ran and clean equals noisy',
          all(np.isclose(res['per_level']['0']['base_1']['clean']['rf'][k],
                         res['per_level']['0']['base_1']['noisy']['rf'][k])
              for k in res['per_level']['0']['base_1']['noisy']['rf']))
    check('6h. no errors', not res.get('_errors'),
          str(list(res.get('_errors', {}).items())[:3]))

    res_h = run_one(make_data(), 'toy', recipe=fake_recipe(['toy', 'second']),
                    models=['rf', 'lightgbm'], levels=[0.5], draws=[0],
                    noise_condition='gaussian', skip_ngboost=True,
                    label_arms=['yeo_johnson'])
    hs = res_h['per_level'].get('0.5', {}).get('hybrid', {})
    check('7. with a recipe, the hybrid space is built and scored',
          'hybrid' in res_h['spaces_run'] and 'yeo_johnson' in hs)
    check('7b. the swaps are fitted in the hybrid space',
          'rf_l1' in hs.get('noisy', {}) and 'lightgbm_tukey' in hs.get('noisy', {}))

    res_s = run_one(make_data(domain='antibody'), 'toy_ab', models=['rf'],
                    levels=[0.5], draws=[0], noise_condition='gaussian',
                    skip_ngboost=True, label_arms=['yeo_johnson'])
    check('9. the metric direction is read off the dataset',
          res['metric'] == 'rmse' and res['lower_is_better']
          and res_s['metric'] == 'spearman' and not res_s['lower_is_better'])
    e = res_s['summary']['per_level']['0.5']['base_1']['per_model']['rf'][
        'per_arm']['clean']['vs_noisy']
    b = res_s['per_level']['0.5']['base_1']['noisy']['rf']
    g = res_s['per_level']['0.5']['base_1']['clean']['rf']
    k = sorted(set(b) & set(g))
    check('9b. a spearman gain is arm minus noisy',
          np.isclose(e['mean_gain'], np.mean([g[x] - b[x] for x in k])))


def test_sweep():
    print("\nthe sweep:")
    real = n33.get_best_search_for_downstream
    n33.get_best_search_for_downstream = no_recipe
    try:
        kw = dict(models=['rf'], levels=[0.5], draws=[0], skip_ngboost=True,
                  label_arms=['yeo_johnson'],
                  base_reps={'rf': ['toy', 'second']},
                  noise_conditions=['gaussian', 'laplace'], swaps=['rf_l1'])
        idx = n33.run(make_data(), {'n_splits': 3}, OUT, 'toy_sweep', **dict(kw))
        again = n33.run(make_data(), {'n_splits': 3}, OUT, 'toy_sweep', **dict(kw))
    finally:
        n33.get_best_search_for_downstream = real
    files = [os.path.join(OUT, n33.EXP_ID, f'toy_sweep__{c}.json')
             for c in ('gaussian', 'laplace')]
    check('8. one file per condition', all(os.path.exists(f) for f in files)
          and idx['n_conditions_run_this_time'] == 2)
    check('8b. a second call skips what is on disk',
          again['n_conditions_run_this_time'] == 0
          and again['n_conditions_on_disk'] == 2)

    d = make_data()
    dc = dict(d, labels=(d['labels'] > 5).astype(int), task_type='classification')
    rc = n33.run(dc, {'n_splits': 3}, OUT, 'toy_clf', models=['rf'])
    check('10. a classification label is skipped with its reason',
          isinstance(rc, dict) and rc.get('skipped') == 'not_regression')


def test_the_command_line():
    print("\nthe command line:")
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      'n33_robust_training.py'), '--help'],
        capture_output=True, text=True)
    wanted = ('--dataset', '--noise-condition', '--levels', '--draws',
              '--huber-c', '--skip-ngboost', '--models', '--exp-folds')
    missing = [f for f in wanted if f not in out.stdout]
    check('11. the parser lists every flag',
          out.returncode == 0 and not missing,
          f'missing {missing}' if missing else f'{len(wanted)} flags present')
    check('11b. --base-rep is gone', '--base-rep' not in out.stdout)


if __name__ == '__main__':
    test_pieces()
    test_swaps()
    test_full_runs()
    test_sweep()
    test_the_command_line()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
