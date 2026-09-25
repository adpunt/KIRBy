#!/usr/bin/env python3
"""Smoke for n38: noise sized by the model's own error.

What has to be true before any number n38 writes can be read:

  1. THE NORMAL THEORY reproduces the report's numbers (d' = 2, 5% moved:
     power 31%, false positives 1.2%, precision 58%), and the AUROC integral
     matches a Monte Carlo draw.
  2. THE SEMI-SYNTHETIC BASE takes no held-out label, returns a signal for the
     held-out molecules, and re-measures resid_sd on its own labels.
  3. THE RANDOM TWIN of winsorising applies the same changes to the same number
     of molecules.
  4. A FULL SPARSE RUN scores every arm in both label bases; the realised shift
     is level x the fold's resid_sd; the out-of-fold residual finds a 4-SD
     shift; vs_signal exists for the semi-synthetic base only; level 0 writes
     no detection.
  5. A DENSE RUN writes no detection and injects the asked variance.
  6. hetero_seen is skipped without SMILES and runs with them.
  7. With a recipe the hybrid space is built and scored.
  8. run() refuses a non-relative condition, skips a class label, and sweeps
     every relative condition by default.
  9. The parser lists every flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import inspect
import tempfile

import numpy as np
from scipy.stats import norm

import experiments.noise.n38_noise_relative_to_model_error as n38
import experiments.noise.n33_robust_training as n33
from experiments.noise.noise_helpers import RELATIVE_CONDITIONS
from kirby.find_best_hybrid import Blueprint

n33.NGB_ESTIMATORS = 40     # the smoke checks wiring, not NGBoost's fit
OUT = tempfile.mkdtemp(prefix='n38_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=150, n_feat=4, seed=0, domain='molecular', smiles=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    d = {'labels': y, 'reps': {'toy': X, 'second': X2},
         'groups': np.arange(n), 'task_type': 'regression', 'domain': domain}
    if smiles:
        d['smiles'] = ['C' * int(k) for k in rng.integers(3, 30, n)]
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


def run_it(data, name, recipe=no_recipe, **kw):
    real = n38.get_best_search_for_downstream
    n38.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_rep', 'toy')
        kw.setdefault('oof_splits', 3)
        kw.setdefault('itlm_rounds', 2)
        return n38._run_one_condition(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n38.get_best_search_for_downstream = real


def test_theory():
    print("the normal theory:")
    t = n38.theory_threshold(2.0, 0.05)
    check("1. d'=2, 5% moved: power 0.31, false positives 0.012, precision 0.57",
          abs(t['theory_power'] - 0.309) < 0.002
          and abs(t['theory_false_positive_rate'] - 0.0124) < 0.0002
          and abs(t['theory_precision'] - 0.568) < 0.003,
          f"{t['theory_power']:.3f}, {t['theory_false_positive_rate']:.4f}, "
          f"{t['theory_precision']:.3f}")
    rng = np.random.default_rng(0)
    a = np.abs(rng.normal(size=400000) + 1.5)
    b = np.abs(rng.normal(size=400000))
    mc = float(np.mean(a > b))
    th = n38.theory_auroc_abs(1.5)
    check("1b. the AUROC integral matches Monte Carlo at d'=1.5",
          abs(mc - th) < 0.004, f'integral {th:.4f}, Monte Carlo {mc:.4f}')
    check("1c. the sign does not enter the AUROC",
          abs(n38.theory_auroc_abs(-1.5) - th) < 1e-9)


def test_semi_synthetic_and_twin():
    print("\nthe semi-synthetic base and the random twin:")
    params = list(inspect.signature(n38.label_base_for_fold).parameters)
    d = make_data()
    X, y = d['reps']['toy'], d['labels']
    y_tr, y_va, sig, rsd, prov = n38.label_base_for_fold(
        'semi_synthetic', X[:100], X[100:], y[:100], np.arange(100), seed=3,
        oof_splits=3)
    check('2. the semi-synthetic base takes no held-out label and returns a '
          'held-out signal and truth',
          not any('va' in p and 'X' not in p for p in params)
          and len(y_va) == 50 and len(sig) == 50 and len(y_tr) == 100
          and rsd > 0 and prov['resid_sd_semi_synthetic'] == rsd
          and not np.allclose(y_tr, y[:100]),
          f"params {params}; resid_sd real {prov['resid_sd_real_labels']:.3f}, "
          f"semi-synthetic {rsd:.3f}")
    y0 = np.arange(20.0)
    changed = np.zeros(20, bool)
    changed[[2, 7, 11]] = True
    delta = np.zeros(20)
    delta[changed] = [-1.0, 0.5, 2.0]
    tw = n38.random_twin_of_changes(y0, changed, delta, 5)
    moved = tw != y0
    check('3. the random twin makes the same changes to the same number of '
          'molecules',
          int(moved.sum()) == 3
          and np.allclose(sorted(tw[moved] - y0[moved]), [-1.0, 0.5, 2.0]))


def test_full_runs():
    print("\nfull runs:")
    res = run_it(make_data(), 'toy', models=['rf', 'lightgbm'],
                 noise_condition='sparse_dprime_p10', levels=[0.0, 4.0],
                 draws=[0], random_seeds=[0])
    missing = {b: [a for a in res['arms']
                   if a not in res['per_level'][b]['4']['base']]
               for b in n38.LABEL_BASES}
    check('4. every arm is scored at level 4 in both label bases',
          not any(missing.values()), f'missing {missing}')
    rl = res['per_level']['real']['4']['base']['robust_loss']
    check('4a. robust_loss is scored for rf and lightgbm',
          set(rl) == {'rf', 'lightgbm'})
    ok = True
    for b in n38.LABEL_BASES:
        for r in res['method_records'][b]['4']['base']:
            f = r['pass'].split('f')[-1]
            rsd = res['fold_records'][b][f'f{f}']['resid_sd']
            ok &= (np.isclose(r['injection']['shift_label_units'], 4.0 * rsd)
                   and np.isclose(r['resid_sd'], rsd))
    check('4b. the realised shift is level x the fold\'s resid_sd', ok)
    det = res['detection']['real']['4']['base']
    th = res['theory_at_nominal_dprime']['4']
    check("4c. the out-of-fold residual finds a 4-SD shift",
          det['scores']['detection_auroc'] > 0.9,
          f"AUROC {det['scores']['detection_auroc']:.3f}, theory "
          f"{th['theory_auroc']:.3f}; d' in this space "
          f"{det['dprime_in_this_space']:.2f}")
    check('4d. vs_signal is written for the semi-synthetic base only',
          list(res['per_level_vs_signal']) == ['semi_synthetic']
          and 'no_cleaning' in res['per_level_vs_signal']['semi_synthetic']['4']['base'])
    s = res['summary']['real']['per_level']['4']['base']['rf']['per_arm']
    check('4e. winsorise and ITLM are paired against their random twins',
          len(s['winsorise_c1.5']['vs_random_twin']) == 1
          and len(s['itlm_a0.9']['vs_random_twin']) == 1)
    check('4f. at level 0 nothing moved and no detection is written',
          all('0' not in v for v in res['detection'].values()))
    check('4g. no errors were recorded', '_errors' not in res,
          str(res.get('_errors'))[:300])

    res_d = run_it(make_data(), 'toy', models=['lightgbm'],
                   noise_condition='dense_oofmse', levels=[1.0], draws=[0],
                   random_seeds=[0], label_bases=['real'])
    inj = [r['injection'] for r in res_d['method_records']['real']['1']['base']]
    ratio = [i['injected_variance_realised'] / i['injected_variance_target']
             for i in inj]
    check('5. a dense run writes no detection and injects the asked variance',
          not res_d['detection'] and all(0.7 < x < 1.3 for x in ratio),
          f'realised / target {np.round(ratio, 2).tolist()}')

    r6 = run_it(make_data(), 'toy', models=['lightgbm'],
                noise_condition='hetero_seen', levels=[1.0], draws=[0])
    r6b = run_it(make_data(smiles=True), 'toy', models=['lightgbm'],
                 noise_condition='hetero_seen', levels=[1.0], draws=[0],
                 random_seeds=[0], label_bases=['real'],
                 arms=['clean', 'no_cleaning'])
    check('6. hetero_seen is skipped without SMILES and runs with them',
          r6.get('skipped') == 'no_smiles'
          and 'no_cleaning' in r6b['per_level']['real']['1']['base'])

    res_h = run_it(make_data(), 'toy', recipe=fake_recipe(['toy', 'second']),
                   models=['lightgbm'], noise_condition='sparse_dprime_up_p05',
                   levels=[2.0], draws=[0], random_seeds=[0],
                   label_bases=['real'],
                   arms=['clean', 'no_cleaning', 'winsorise_c1.5'])
    check('7. with a recipe the hybrid space is built and scored',
          'hybrid' in res_h['spaces_run']
          and 'winsorise_c1.5' in res_h['per_level']['real']['2']['hybrid'])


def test_guards():
    print("\nthe guards:")
    d = make_data()
    r = n38.run(d, {'n_splits': 3}, OUT, 'toy', noise_condition='gaussian')
    rc = n38.run(dict(d, labels=(d['labels'] > 5).astype(int),
                      task_type='classification'), {'n_splits': 3}, OUT, 'toy')
    src = inspect.getsource(n38.run)
    check('8. run() refuses a non-relative condition, skips a class label, and '
          'sweeps every relative condition by default',
          r.get('skipped') == 'not_a_relative_condition'
          and rc.get('skipped') == 'not_regression'
          and 'default_conditions=list(RELATIVE_CONDITIONS)' in src
          and len(RELATIVE_CONDITIONS) == 9)


def test_the_command_line():
    print("\nthe command line:")
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      'n38_noise_relative_to_model_error.py'),
         '--help'], capture_output=True, text=True)
    wanted = ('--dataset', '--noise-condition', '--levels', '--draws',
              '--label-bases', '--random-seeds', '--models', '--exp-folds',
              '--base-rep', '--tag')
    missing = [f for f in wanted if f not in out.stdout]
    check('9. the parser lists every flag',
          out.returncode == 0 and not missing,
          f'missing {missing}' if missing else f'{len(wanted)} flags present')


if __name__ == '__main__':
    test_theory()
    test_semi_synthetic_and_twin()
    test_guards()
    test_full_runs()
    test_the_command_line()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
