#!/usr/bin/env python3
"""Smoke for n40: retuning, seed averaging, stacking and GP noise terms.

What has to be true before any number n40 writes can be read:

  1. THE CLOSED-FORM GP IS SKLEARN'S GP at the fitted noise term, so the Cp and
     CV choices move only the noise term and nothing else.
  2. Cp FOR kNN IS RSS/n + 2 sigma^2 / k, and at k = 1 the in-sample fit is exact.
  3. THE GAMMA ESTIMATE lands near a known noise variance on toy data.
  4. THE TUNER NEVER SEES A HELD-OUT MOLECULE: every matrix it is handed is
     exactly one fold's training rows, and every clean label it is handed is a
     training label of that fold and none of its held-out labels.
  5. A FULL RUN scores every arm for every model, the stack, every GP and kNN
     arm, and writes tail scores.
  6. dense_oofmse injects level x the written residual SD squared.
  7. ABOVE THE GP CAP every GP arm is skipped with its reason.
  8. WITH REPEAT MEASUREMENTS the two molecule-mean GP arms run.
  9. WITH A RECIPE the hybrid space is built and scored.
 10. THE SWEEP writes one file per condition and a second call skips it.
 11. The metric direction is read off the dataset; classification is skipped.
 12. The parser lists every flag.
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

import experiments.noise.n40_model_side_regularisation as n40
from experiments._shared import cv_split
from kirby.find_best_hybrid import Blueprint
from sklearn.preprocessing import StandardScaler

# Toy speed only.
n40.TUNE_N_ITER = 2
n40.TUNE_INNER_FOLDS = 2
n40.SEED_AVG_N = 3
n40.RF_BIG_TREES = 60
n40.OOF_SPLITS = 3

OUT = tempfile.mkdtemp(prefix='n40_smoke_')
FAIL = []
MODELS = ['lightgbm', 'rf']


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=120, n_feat=4, seed=0, domain='molecular', repeats=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    if repeats:
        X[1::2] = X[0::2]          # molecules measured twice: same features
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    d = {'labels': y, 'reps': {'toy': X, 'second': X2},
         'groups': (np.arange(n) // 2 if repeats else np.arange(n)),
         'task_type': 'regression', 'domain': domain}
    if repeats:
        d['replicate_groups'] = np.arange(n) // 2
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


# Every model's top 2 single representations, given directly: rank 1 'toy',
# rank 2 'second'.
BASE_REPS = {m: ['toy', 'second'] for m in n40.DOWNSTREAM_MODELS}


def run_one(data, name, recipe=no_recipe, **kw):
    real = n40.get_best_search_for_downstream
    n40.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_reps', BASE_REPS)
        kw.setdefault('models', MODELS)
        return n40._run_one_condition(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n40.get_best_search_for_downstream = real


def test_pieces():
    print("the pieces:")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.2, 150)
    gp = n40.make_gp().fit(X, y)
    e = n40._EigenGP(gp.kernel_.k1, X, y)
    Xq = rng.normal(size=(40, 3))
    a = float(gp.kernel_.k2.noise_level)
    diff = float(np.max(np.abs(e.predict(Xq, a) - gp.predict(Xq))))
    check('1. the closed-form GP equals sklearn\'s at the fitted noise term',
          diff < 1e-4, f'max difference {diff:.2e}')

    k, cps = n40.knn_cp_choice(X, y, 0.04)
    check('2. Cp for kNN is RSS/n + 2 sigma^2/k; exact at k = 1',
          np.isclose(cps[1], 2 * 0.04) and k in n40.KNN_K_GRID,
          f'Cp(1) {cps[1]:.4f}, chosen k {k}')

    Xg = rng.uniform(size=(600, 2))
    yg = np.sin(3 * Xg[:, 0]) + Xg[:, 1] + rng.normal(0, np.sqrt(0.1), 600)
    s2, _ = n40.gamma_sigma2(Xg, yg)
    check('3. the Gamma estimate lands near a known noise variance',
          abs(s2 - 0.1) < 0.03, f'{s2:.4f} against 0.1')


def test_tuner_never_sees_held_out():
    print("\nthe tuner:")
    data = make_data()
    calls = []
    n40.TUNER_HOOK = lambda name, X, y, g: calls.append((np.array(X), np.array(y)))
    try:
        run_one(data, 'toy_tune', levels=[0.5], draws=[0],
                noise_condition='gaussian', skip_gp=True, skip_knn=True,
                skip_stack=True)
    finally:
        n40.TUNER_HOOK = None
    labels = data['labels']
    splits = list(cv_split(data['groups'], 3, data['reps']['toy'], labels, mode='full'))
    bases = []
    for tr, va in splits:
        for rn in ('toy', 'second'):
            bases.append((StandardScaler().fit_transform(data['reps'][rn][tr]),
                          tr, va))
    bad_rows, bad_clean, n_clean = 0, 0, 0
    for X, y in calls:
        match = [(tr, va) for B, tr, va in bases
                 if B.shape == X.shape and np.allclose(B, X)]
        if not match:
            bad_rows += 1
            continue
        tr, va = match[0]
        if np.all(np.isin(y, labels)):
            n_clean += 1
            if not (set(y) == set(labels[tr]) and not set(y) & set(labels[va])):
                bad_clean += 1
    check('4. every matrix the tuner gets is exactly one fold\'s training rows',
          calls and bad_rows == 0, f'{len(calls)} calls, {bad_rows} not a training fold')
    check('4b. every clean label it gets is a training label, none held out',
          n_clean > 0 and bad_clean == 0, f'{n_clean} clean calls, {bad_clean} bad')


def test_full_runs():
    print("\nfull runs:")
    res = run_one(make_data(), 'toy', levels=[0.0, 0.5], draws=[0],
                  noise_condition='gaussian')
    lv = res['per_level']['0.5']['base_1']
    want = {'clean': MODELS, 'fixed': MODELS, 'retuned': MODELS,
            'tuned_clean_reused': MODELS, 'seed_one': ['lightgbm'],
            'seed_avg': ['lightgbm'], 'rf_1000_trees': ['rf']}
    missing = [(a, m) for a, ms in want.items() for m in ms
               if len(lv.get(a, {}).get(m, {})) != 3]
    check('5. every arm is scored for every model on every fold', not missing,
          f'{missing}; errors {list(res.get("_errors", {}).items())[:3]}'
          if missing else '')
    extras = ['stack', 'gp_ml_nugget', 'gp_nugget_from_gamma', 'gp_nugget_cp',
              'gp_nugget_cv', 'lightgbm_plus_gp_residual', 'knn_k5_uniform',
              'knn_k_cp', 'knn_k_cv']
    miss = [x for x in extras if len(lv['fixed'].get(x, {})) != 3]
    check('5b. the stack, every GP and every kNN arm is scored', not miss,
          f'missing {miss}; errors {list(res.get("_errors", {}).items())[:3]}')
    t = res['tails_per_level']['0.5']['base_1']['retuned']['rf']
    check('5c. tail scores are written beside each score',
          all(v and 'tail_rmse_top' in v for v in t.values()))
    sp = res['per_pass']['0.5'][0]['per_space']['base_1']
    check('5d. the chosen settings, the Gamma estimate and the stack weights '
          'are written', 'rf' in sp['tuned'] and 'gamma' in sp
          and 'stack_weights' in sp and 'k_cp' in sp['knn'])
    s = res['summary']['per_level']['0.5']['base_1']
    check('5e. the summary pairs arms against fixed and the extra pairs',
          'retuned' in s['per_model']['rf']['per_arm']
          and 'gp_nugget_cp_vs_gp_ml_nugget' in s['extra_pairs']
          and 'lightgbm' in s['stack'])
    check('5e2. base_1 and base_2 both ran, and each model\'s pair is named',
          res['spaces_run'] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {m: ['toy', 'second'] for m in MODELS}
          and 'base_rep' not in res
          and set(res['per_level']['0.5']['base_2']['fixed']) >= set(MODELS),
          f"{res['spaces_run']}, {res.get('base_reps_by_model')}")
    check('5f. no errors', not res.get('_errors'),
          str(list(res.get('_errors', {}).items())[:3]))
    check('5g. at level 0 clean equals fixed',
          all(np.isclose(res['per_level']['0']['base_1']['clean']['rf'][k],
                         res['per_level']['0']['base_1']['fixed']['rf'][k])
              for k in res['per_level']['0']['base_1']['fixed']['rf']))

    rd = run_one(make_data(), 'toy_rel', levels=[0.3], draws=[0],
                 noise_condition='dense_oofmse', skip_tuning=True, skip_gp=True,
                 skip_knn=True, skip_stack=True)
    ok = []
    for r in rd['per_pass']['0.3']:
        want_rms = np.sqrt(0.3) * r['resid_sd_label_units']
        ok.append(abs(r['injected_rms_label_units'] - want_rms) < 0.35 * want_rms)
    check('6. dense_oofmse injects level x the written residual SD squared',
          ok and all(ok), f'{sum(ok)} of {len(ok)} folds')
    check('6b. the residual SD is measured in the first model\'s rank-1 '
          'representation, and that is written',
          rd.get('resid_sd_rep') == 'toy' and rd.get('resid_sd_rep_chosen_as'),
          str(rd.get('resid_sd_rep')))

    real = n40.gp_max_samples
    n40.gp_max_samples = lambda: 50
    try:
        rc = run_one(make_data(), 'toy_cap', levels=[0.5], draws=[0],
                     noise_condition='gaussian', skip_tuning=True,
                     skip_knn=True, skip_stack=True)
    finally:
        n40.gp_max_samples = real
    fx = rc['per_level']['0.5']['base_1']['fixed']
    check('7. above the GP cap every GP arm is skipped with its reason',
          'gp_ml_nugget' not in fx and rc.get('arm_skips')
          and all('GP cap' in v for v in rc['arm_skips'].values()))

    rr = run_one(make_data(repeats=True), 'toy_rep', levels=[0.5], draws=[0],
                 noise_condition='gaussian', skip_tuning=True, skip_knn=True,
                 skip_stack=True)
    fx = rr['per_level']['0.5']['base_1']['fixed']
    check('8. with repeat measurements both molecule-mean GP arms run',
          rr['repeat_measurements']
          and len(fx.get('gp_means_ml_nugget', {})) == 3
          and len(fx.get('gp_means_per_molecule_alpha', {})) == 3,
          str(list(rr.get('_errors', {}).items())[:2]))

    rh = run_one(make_data(), 'toy_h', recipe=fake_recipe(['toy', 'second']),
                 levels=[0.5], draws=[0], noise_condition='gaussian',
                 skip_tuning=True, skip_stack=True)
    hs = rh['per_level'].get('0.5', {}).get('hybrid', {})
    check('9. with a recipe the hybrid space is built and scored, GP included',
          'hybrid' in rh['spaces_run'] and 'gp_ml_nugget' in hs.get('fixed', {})
          and 'rf' in hs.get('rf_1000_trees', {}),
          str(list(rh.get('_errors', {}).items())[:2]))

    ra = run_one(make_data(domain='antibody'), 'toy_ab', models=['rf'],
                 levels=[0.5], draws=[0], noise_condition='gaussian',
                 skip_tuning=True, skip_gp=True, skip_knn=True, skip_stack=True)
    check('11. the metric direction is read off the dataset',
          res['metric'] == 'rmse' and res['lower_is_better']
          and ra['metric'] == 'spearman' and not ra['lower_is_better'])


def test_sweep():
    print("\nthe sweep:")
    real = n40.get_best_search_for_downstream
    n40.get_best_search_for_downstream = no_recipe
    try:
        kw = dict(models=['rf'], levels=[0.5], draws=[0], base_reps=BASE_REPS,
                  skip_tuning=True, skip_gp=True, skip_knn=True, skip_stack=True)
        idx = n40.run(make_data(), {'n_splits': 3}, OUT, 'toy_sweep', **dict(kw))
        again = n40.run(make_data(), {'n_splits': 3}, OUT, 'toy_sweep', **dict(kw))
    finally:
        n40.get_best_search_for_downstream = real
    files = [os.path.join(OUT, n40.EXP_ID, f'toy_sweep__{c}.json')
             for c in n40.DEFAULT_CONDITIONS]
    check('10. one file per default condition', all(os.path.exists(f) for f in files)
          and idx['n_conditions_run_this_time'] == 2)
    check('10b. a second call skips what is on disk',
          again['n_conditions_run_this_time'] == 0
          and again['n_conditions_on_disk'] == 2)
    d = make_data()
    dc = dict(d, labels=(d['labels'] > 5).astype(int), task_type='classification')
    rc = n40.run(dc, {'n_splits': 3}, OUT, 'toy_clf', models=['rf'])
    check('11b. a classification label is skipped with its reason',
          isinstance(rc, dict) and rc.get('skipped') == 'not_regression')


def test_the_command_line():
    print("\nthe command line:")
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      'n40_model_side_regularisation.py'), '--help'],
        capture_output=True, text=True)
    wanted = ('--dataset', '--noise-condition', '--levels', '--draws',
              '--tune-n-iter', '--skip-tuning', '--skip-gp', '--models',
              '--exp-folds')
    missing = [f for f in wanted if f not in out.stdout]
    check('12b. --base-rep is gone', out.returncode == 0
          and '--base-rep' not in out.stdout)
    check('12. the parser lists every flag', out.returncode == 0 and not missing,
          f'missing {missing}' if missing else f'{len(wanted)} flags present')


if __name__ == '__main__':
    test_pieces()
    test_tuner_never_sees_held_out()
    test_full_runs()
    test_sweep()
    test_the_command_line()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
