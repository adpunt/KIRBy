#!/usr/bin/env python3
"""Smoke for n31: predicted detector ceiling, measured detector, break-even precision.

What has to be true before any number n31 writes can be read:

  1. THE TWO FORMULAS ARE THE REPORT'S. Spot values, and a simulation of
     Gaussian residuals lands on each.
  2. THE INTRACLASS CORRELATION reads high on grouped offsets, near zero on
     independent values, and is refused when no group holds two molecules.
  3. A SYNTHETIC FLAG SET holds as many molecules as moved, at the precision
     asked for.
  4. THE FOUR OPERATIONS touch the flagged molecules only.
  5. THE BREAK-EVEN PRECISION is the linear crossing, and is refused with a
     reason when the gain never crosses.
  6. THE SHIFT INJECTOR moves the share it says by delta label SDs, upward,
     and nothing else.
  7. ON TOY DATA the measured AUROC of the signed residual under a shift sits
     near the predicted one.
  8. A FULL RUN writes every arm under a subset condition, no detection and no
     synthetic arm under gaussian, tail scores on every score, and no level 0.
  9. A classification label is skipped, an unknown condition refused, a
     recipe builds the hybrid space, and the metric direction flips under
     spearman.
 10. The parser lists every flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import tempfile

import numpy as np
from sklearn.metrics import roc_auc_score

import experiments.noise.n31_regime_and_breakeven as n31
from experiments.noise.noise_helpers import inject_shift_train_labels
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n31_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=180, n_feat=4, seed=0, domain='molecular', noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, noise, n)
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


def run_it(data, name, recipe=no_recipe, **kw):
    real = n31.get_best_search_for_downstream
    n31.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_rep', 'toy')
        return n31.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n31.get_best_search_for_downstream = real


def test_formulas():
    print("the formulas:")
    a = n31.predicted_auroc_shift(1.0, 1.0 / np.sqrt(2.0))
    check('1. shift: delta 1, s_m 1/sqrt2 gives Phi(1)',
          abs(a - 0.841344746) < 1e-6, f'{a:.6f}')
    b = n31.predicted_auroc_wider(0.0, 0.5)
    check('1b. wider: rho 0 gives (2/pi) arctan(1) = 0.5', abs(b - 0.5) < 1e-12)
    c = n31.predicted_auroc_wider(2.9, 1.0)
    check('1c. wider: rho 2.9 gives about 0.8, as the report says',
          abs(c - 0.8) < 0.005, f'{c:.4f}')
    rng = np.random.default_rng(0)
    s_m, delta, sig = 0.5, 0.6, 1.2
    clean = rng.normal(0, s_m, 40000)
    shifted = rng.normal(delta, s_m, 40000)
    wide = rng.normal(0, np.sqrt(s_m ** 2 + sig ** 2), 40000)
    lab = np.r_[np.zeros(40000), np.ones(40000)]
    m1 = roc_auc_score(lab, np.r_[clean, shifted])
    m2 = roc_auc_score(lab, np.abs(np.r_[clean, wide]))
    p1 = n31.predicted_auroc_shift(delta, s_m)
    p2 = n31.predicted_auroc_wider(sig, s_m)
    check('1d. simulated shift AUROC matches the formula',
          abs(m1 - p1) < 0.01, f'simulated {m1:.4f}, formula {p1:.4f}')
    check('1e. simulated wider-error AUROC matches the formula',
          abs(m2 - p2) < 0.01, f'simulated {m2:.4f}, formula {p2:.4f}')


def test_icc():
    print("\nthe intraclass correlation:")
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(60), 5)
    off = rng.normal(0, 1.0, 60)[g]
    v = off + rng.normal(0, 0.3, g.size)
    icc, _ = n31.one_way_icc(v, g)
    check('2. grouped offsets give a high ICC', icc is not None and icc > 0.8,
          f'{icc:.3f}')
    icc0, _ = n31.one_way_icc(rng.normal(size=g.size), g)
    check('2b. independent values give an ICC near zero',
          icc0 is not None and abs(icc0) < 0.15, f'{icc0:.3f}')
    icc1, why = n31.one_way_icc(rng.normal(size=50), np.arange(50))
    check('2c. singleton groups are refused with a reason',
          icc1 is None and why, why or '')
    rs = n31.residual_shape(rng.normal(size=50), rng.normal(size=50), None)
    check('2d. no scaffold groups: ICC None with its reason',
          rs['icc_by_scaffold_group'] is None and 'icc_refused_because' in rs)


def test_flags_and_ops():
    print("\nflag sets and operations:")
    mask = np.zeros(200, dtype=bool)
    mask[:20] = True
    ok = True
    for q in n31.PRECISIONS:
        f, rp = n31.synthetic_flags(mask, q, 3)
        ok &= int(f.sum()) == 20 and abs(rp - q) < 1e-9
    check('3. every synthetic set holds 20 molecules at the precision asked', ok)
    f1, _ = n31.synthetic_flags(mask, 1.0, 0)
    f2, _ = n31.synthetic_flags(mask, 1.0, 9)
    check('3b. at precision 1.0 the set is the mask whatever the seed',
          np.array_equal(f1, mask) and np.array_equal(f2, mask))
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)
    p = rng.normal(size=200)
    flag, _ = n31.synthetic_flags(mask, 0.5, 1)
    y_r, k_r, w_r = n31.apply_operation('remove', y, flag, p, 0.3)
    y_d, k_d, w_d = n31.apply_operation('downweight', y, flag, p, 0.3)
    y_w, _, _ = n31.apply_operation('rewrite', y, flag, p, 0.3)
    y_b, _, _ = n31.apply_operation('bayes_shrink', y, flag, p, 0.3)
    check('4. remove keeps every unflagged molecule and no flagged one',
          np.array_equal(k_r, ~flag) and w_r is None)
    check('4b. downweight gives 0.25 to the flagged, 1 to the rest',
          np.all(w_d[flag] == 0.25) and np.all(w_d[~flag] == 1.0))
    check('4c. rewrite sets flagged labels to p and no other',
          np.allclose(y_w[flag], p[flag]) and np.array_equal(y_w[~flag], y[~flag]))
    check('4d. bayes_shrink moves flagged labels 30% toward p and no other',
          np.allclose(y_b[flag], 0.7 * y[flag] + 0.3 * p[flag])
          and np.array_equal(y_b[~flag], y[~flag]))


def test_break_even():
    print("\nbreak-even precision:")
    be, _ = n31.break_even_precision([0, 0.25, 0.5, 0.75, 1.0],
                                     [-0.04, -0.02, -0.01, 0.01, 0.03])
    check('5. crossing between 0.5 and 0.75 at 0.625',
          be is not None and abs(be - 0.625) < 1e-12, f'{be}')
    be2, why = n31.break_even_precision([0, 0.5, 1.0], [-0.1, -0.05, -0.01])
    check('5b. no crossing is refused with a reason', be2 is None and why, why)
    be3, why3 = n31.break_even_precision([0, 0.5, 1.0], [0.01, 0.02, 0.03])
    check('5c. a gain already positive at precision 0 is refused',
          be3 is None and why3, why3)


def test_shift_injector():
    print("\nthe shift injector:")
    y = make_data(n=300)['labels']
    tr = np.arange(250)
    y0, m, prov = inject_shift_train_labels(y, train_idx=tr,
                                            condition='shift_only_p10',
                                            delta_sd=1.0, seed=4)
    d = y0 - y[tr]
    check('6. a tenth of the training labels moved, up by one label SD, '
          'the rest untouched',
          int(m.sum()) == 25 and np.allclose(d[m], np.std(y))
          and np.all(d[~m] == 0), f'{int(m.sum())} moved')


def test_measured_near_predicted():
    print("\nmeasured against predicted, toy data:")
    res = run_it(make_data(n=900, noise=1.5, seed=5), 'toy',
                 models=['lightgbm'], conditions=['shift_only_p10'],
                 levels=[0.5, 1.0], draws=[0], precisions=[1.0],
                 flag_seeds=[0])
    rows = []
    for lvl in ('0.5', '1'):
        d = res['summary']['per_condition']['shift_only_p10'][lvl]['base'][
            'detection']
        rows.append((lvl, d['predicted_auroc_mean'],
                     d['measured_auroc_signed_residual_mean']))
    ok = all(abs(p - m) < 0.1 for _, p, m in rows)
    check('7. the signed-residual AUROC sits within 0.1 of the prediction',
          ok, '; '.join(f'delta {l}: predicted {p:.3f}, measured {m:.3f}'
                        for l, p, m in rows))


def test_full_runs():
    print("\nfull runs:")
    models = ['rf', 'lightgbm']
    res = run_it(make_data(), 'toy', models=models,
                 conditions=['outlier_only_p10', 'shift_only_p05', 'gaussian'],
                 levels=[0.0, 1.0], draws=[0], precisions=[0.0, 0.5, 1.0],
                 flag_seeds=[0, 1])
    pc = res['per_condition']
    check('8. level 0 is never run', all('0' not in pc[c] for c in pc))
    check('8a. no fit raised', not res.get('_errors'),
          str(list((res.get('_errors') or {}).items())[:2]))
    arms = pc['outlier_only_p10']['1']['base']
    want = (['clean', 'noisy'] + [f'detector/{o}' for o in n31.OPERATIONS]
            + [f'flags_p{q:g}/seed{s}/{o}' for q in (0.0, 0.5, 1.0)
               for s in (0, 1) for o in n31.OPERATIONS])
    missing = [a for a in want if a not in arms]
    check('8b. every arm scored under outlier_only_p10', not missing,
          f'missing {missing[:4]}' if missing else f'{len(want)} arms')
    g = pc['gaussian']['1']['base']
    check('8c. gaussian: detector operations run, no synthetic arm',
          'detector/remove' in g and not any(a.startswith('flags_') for a in g))
    dg = res['diagnostics']['gaussian']['1']['base']
    check('8d. gaussian: detection and prediction are absent with a reason',
          all('detection_not_defined_because' in r
              and 'predicted_auroc' not in r for r in dg.values()))
    ds = res['diagnostics']['shift_only_p05']['1']['base']
    check('8e. shift: prediction, both detections, and residual shape per pass',
          len(ds) == 3 and all(
              np.isfinite(r['predicted_auroc'])
              and r['detection_abs_residual'] is not None
              and 'detection_auroc' in r['detection_signed_residual']
              and 'excess_kurtosis' in r['residual_shape_noisy']
              for r in ds.values()))
    tl = res['tails']['outlier_only_p10']['1']['base']
    check('8f. a tail record beside every score',
          all(set(tl[a][m]) == set(arms[a][m]) for a in arms for m in arms[a]))
    syn = res['summary']['per_condition']['outlier_only_p10']['1']['base'][
        'per_model']['rf']['synthetic']
    check('8g. the summary holds a curve per operation',
          set(syn) == set(n31.OPERATIONS)
          and all('break_even_precision' in e for e in syn.values()))

    dc = dict(make_data(), labels=(make_data()['labels'] > 5).astype(int),
              task_type='classification')
    rc = run_it(dc, 'toy', models=['rf'])
    check('9. a classification label is skipped with its reason',
          rc.get('skipped') == 'not_regression')
    try:
        run_it(make_data(), 'toy', models=['rf'], conditions=['censoring'])
        ok = False
    except ValueError:
        ok = True
    check('9b. an unknown condition is refused', ok)
    res_h = run_it(make_data(), 'toy', recipe=fake_recipe(['toy', 'second']),
                   models=['lightgbm'], conditions=['shift_only_p10'],
                   levels=[1.0], draws=[0], precisions=[1.0], flag_seeds=[0])
    check('9c. with a recipe, the hybrid space is built and scored',
          'hybrid' in res_h['spaces_run']
          and 'noisy' in res_h['per_condition']['shift_only_p10']['1']['hybrid'])
    res_s = run_it(make_data(domain='antibody'), 'toy_ab', models=['rf'],
                   conditions=['shift_only_p10'], levels=[1.0], draws=[0],
                   precisions=[1.0], flag_seeds=[0])
    check('9d. the metric direction is read off the dataset',
          res['metric'] == 'rmse' and res['lower_is_better']
          and res_s['metric'] == 'spearman' and not res_s['lower_is_better'])
    row = res_s['summary']['per_condition']['shift_only_p10']['1']['base'][
        'per_model']['rf']
    b = res_s['per_condition']['shift_only_p10']['1']['base']['noisy']['rf']
    c = res_s['per_condition']['shift_only_p10']['1']['base']['clean']['rf']
    k = sorted(set(b) & set(c))
    check('9e. a spearman gain is arm minus noisy',
          np.isclose(row['clean_vs_noisy']['mean_gain'],
                     np.mean([c[x] - b[x] for x in k])))
    row2 = res['summary']['per_condition']['outlier_only_p10']['1']['base'][
        'per_model']['rf']
    b2 = arms['noisy']['rf']
    c2 = arms['clean']['rf']
    k2 = sorted(set(b2) & set(c2))
    check('9f. an rmse gain is noisy minus arm',
          np.isclose(row2['clean_vs_noisy']['mean_gain'],
                     np.mean([b2[x] - c2[x] for x in k2])))


def test_the_command_line():
    print("\nthe command line:")
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      'n31_regime_and_breakeven.py'), '--help'],
        capture_output=True, text=True)
    wanted = ('--dataset', '--conditions', '--levels', '--draws', '--precisions',
              '--flag-seeds', '--models', '--exp-folds', '--base-rep', '--tag')
    missing = [f for f in wanted if f not in out.stdout]
    check('10. the parser lists every flag',
          out.returncode == 0 and not missing,
          f'missing {missing}' if missing else f'{len(wanted)} flags present')


if __name__ == '__main__':
    test_formulas()
    test_icc()
    test_flags_and_ops()
    test_break_even()
    test_shift_injector()
    test_measured_near_predicted()
    test_full_runs()
    test_the_command_line()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
