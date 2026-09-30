#!/usr/bin/env python3
"""Smoke for n31: predicted detector ceiling, measured detector, break-even precision.

What has to be true before any number n31 writes can be read:

  1. THE TWO FORMULAS ARE THE REPORT'S. Spot values, and a simulation of
     Gaussian residuals lands on each.
  2. THE INTRACLASS CORRELATION reads high on grouped offsets, near zero on
     independent values, and is refused when no group holds two molecules.
  3. A SYNTHETIC FLAG SET holds the count it is given (a swept share of the
     training molecules, never the number moved), at the precision asked
     for, and says so when the mask is too small to reach it.
  4. THE FOUR OPERATIONS touch the flagged molecules only.
  5. THE BREAK-EVEN PRECISION is the linear crossing, and is refused with a
     reason when the gain never crosses.
  6. THE SHIFT INJECTOR moves the share it says by delta label SDs, upward,
     and nothing else.
  7. ON TOY DATA the measured AUROC of the signed residual under a shift sits
     near the predicted one.
  8. A FULL RUN writes every arm under a subset condition, no detection and no
     synthetic arm under gaussian, tail scores on every score, and at level 0
     the detector arms only (changed 2026-09-28: level 0 used to be skipped).
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

# 2026-09-25: n31 now caches through noise_cache under KIRBY_CACHE_DIR and
# checkpoints under the output directory, so the toy entries go to a
# temporary directory, as tests/smoke/smoke_noise_similarity_foundation.py
# does.
os.environ['KIRBY_CACHE_DIR'] = tempfile.mkdtemp(prefix='n31_smoke_cache_')


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
        kw.setdefault('base_reps', {
            m: ['toy', 'second']
            for m in (kw.get('models') or n31.DOWNSTREAM_MODELS)})
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
        f, rp = n31.synthetic_flags(mask, q, 3, 20)
        ok &= int(f.sum()) == 20 and abs(rp - q) < 1e-9
    check('3. every synthetic set holds 20 molecules at the precision asked', ok)
    f40, rp40 = n31.synthetic_flags(mask, 1.0, 3, 40)
    check('3a. asked for 40 at precision 1.0 with 20 moved: 40 flagged, '
          'realised precision 0.5',
          int(f40.sum()) == 40 and abs(rp40 - 0.5) < 1e-9 and f40[mask].all(),
          f'{int(f40.sum())} flagged, precision {rp40:.3f}')
    f1, _ = n31.synthetic_flags(mask, 1.0, 0, 20)
    f2, _ = n31.synthetic_flags(mask, 1.0, 9, 20)
    check('3b. at precision 1.0 the set is the mask whatever the seed',
          np.array_equal(f1, mask) and np.array_equal(f2, mask))
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)
    p = rng.normal(size=200)
    flag, _ = n31.synthetic_flags(mask, 0.5, 1, 20)
    y_r, k_r, w_r = n31.apply_operation('remove', y, flag, p, 0.3)
    y_d, k_d, w_d = n31.apply_operation('downweight', y, flag, p, 0.3)
    y_w, _, _ = n31.apply_operation('rewrite', y, flag, p, 0.3)
    y_b, _, _ = n31.apply_operation('bayes_shrink', y, flag, p, 0.3)
    # The estimated weight: flagged residuals 3x the unflagged SD give
    # sigma_n^2 = 9 - 1 = 8 times s_m^2, so w = 8/9.
    _r = np.ones(10); _f = np.zeros(10, dtype=bool); _f[:2] = True
    _r[_f] = 3.0
    check('bayes_shrink estimates its weight from the out-of-fold errors: '
          'flagged residuals 3x the rest give w = 8/9',
          abs(n31.estimated_shrink_weight(_r, np.zeros(10), _f) - 8 / 9)
          < 1e-12,
          f'{n31.estimated_shrink_weight(_r, np.zeros(10), _f):.6f}')
    check('with nothing flagged the weight is 0',
          n31.estimated_shrink_weight(_r, np.zeros(10),
                                      np.zeros(10, dtype=bool)) == 0.0)
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
                                            delta_sd=1.0, draw=4)
    d = y0 - y[tr]
    # Whole-column draw and training-label SD since 2026-09-30 (noise_helpers.
    # inject_shift_train_labels), so the share in the fold is about a tenth.
    check('6. about a tenth of the training labels moved, up by one '
          'training-label SD, the rest untouched',
          0.05 * len(tr) <= int(m.sum()) <= 0.15 * len(tr)
          and np.allclose(d[m], np.std(y[tr]))
          and np.all(d[~m] == 0), f'{int(m.sum())} moved')


def test_measured_near_predicted():
    print("\nmeasured against predicted, toy data:")
    res = run_it(make_data(n=900, noise=1.5, seed=5), 'toy',
                 models=['lightgbm'], conditions=['shift_only_p10'],
                 levels=[0.5, 1.0], draws=[0], precisions=[1.0],
                 flag_seeds=[0])
    rows = []
    for lvl in ('0.5', '1'):
        d = res['summary']['per_condition']['shift_only_p10'][lvl]['base_1'][
            'detection']
        rows.append((lvl, d['predicted_auroc_mean'],
                     d['measured_auroc_signed_residual_mean']))
    ok = all(abs(p - m) < 0.1 for _, p, m in rows)
    check('7. the signed-residual AUROC sits within 0.1 of the prediction',
          ok, '; '.join(f'delta {l}: predicted {p:.3f}, measured {m:.3f}'
                        for l, p, m in rows))


def _base1(res, cond, lvl):
    """Every per-pass diagnostic record of base_1 at (cond, lvl), keyed
    '<space key>|<pass>'. The records sit under base_1, or base_1/<model>
    when the models had separate matrices -- which they do whenever
    ensure_dprime_models adds svm and the caller's base_reps name no rank for
    it (2026-09-25). Indexing 'base_1' alone raised KeyError there."""
    return {f'{ek}|{pk}': r
            for ek, recs in res['diagnostics'][cond][lvl].items()
            if ek.split('/')[0] == 'base_1' for pk, r in recs.items()}


def test_full_runs():
    print("\nfull runs:")
    models = ['rf', 'lightgbm']
    res = run_it(make_data(), 'toy', models=models,
                 conditions=['outlier_only_p10', 'shift_only_p05', 'gaussian'],
                 levels=[0.0, 1.0], draws=[0], precisions=[0.0, 0.5, 1.0],
                 flag_seeds=[0, 1], flag_shares=[0.2])
    pc = res['per_condition']
    check('8-. spaces base_1 and base_2 ran, and the representations are named',
          res['spaces_run'] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {m: ['toy', 'second']
                                            for m in models}
          and 'base_rep' not in res,
          f"{res['spaces_run']}, {res.get('base_reps_by_model')}")
    check('8-b. base_2 is scored too',
          'noisy' in pc['outlier_only_p10']['1']['base_2'])
    # 2026-09-28: level 0 runs, one draw, detector arms only.
    l0 = pc['outlier_only_p10']['0']['base_1']
    check('8. level 0 runs the detector arms and no synthetic arm',
          all(f'detector/share0.2/{o}' in l0 for o in n31.OPERATIONS)
          and not any(a.startswith('flags_') for a in l0),
          f'{sorted(l0)[:6]}')
    # The record sits under base_1, or base_1/<model> when the models had
    # separate matrices; every one of them is read.
    d0 = list(_base1(res, 'outlier_only_p10', '0').values())
    check('8-0. level 0: no detection, and why the synthetic flags are absent',
          d0 and all('synthetic_flags_not_written_because' in r
                     and 'detection_abs_residual' not in r for r in d0),
          f'{len(d0)} records')
    check('8a. no fit raised', not res.get('_errors'),
          str(list((res.get('_errors') or {}).items())[:2]))
    arms = pc['outlier_only_p10']['1']['base_1']
    want = (['clean', 'noisy']
            + [f'detector/share0.2/{o}' for o in n31.OPERATIONS]
            + [f'flags_share0.2_p{q:g}/seed{s}/{o}' for q in (0.0, 0.5, 1.0)
               for s in (0, 1) for o in n31.OPERATIONS])
    missing = [a for a in want if a not in arms]
    check('8b. every arm scored under outlier_only_p10', not missing,
          f'missing {missing[:4]}' if missing else f'{len(want)} arms')
    g = pc['gaussian']['1']['base_1']
    check('8c. gaussian: detector operations run, no synthetic arm',
          'detector/share0.2/remove' in g
          and not any(a.startswith('flags_') for a in g))
    dg = _base1(res, 'gaussian', '1')
    check('8d. gaussian: detection and prediction are absent with a reason',
          dg and all('detection_not_defined_because' in r
                     and 'predicted_auroc' not in r for r in dg.values()))
    ds = _base1(res, 'shift_only_p05', '1')
    check('8e. shift: prediction, both detections, and residual shape per pass',
          len(ds) == 3 and all(
              np.isfinite(r['predicted_auroc'])
              and r['detection_abs_residual'] is not None
              and 'detection_auroc' in r['detection_signed_residual']
              and 'excess_kurtosis' in r['residual_shape_noisy']
              for r in ds.values()))
    tl = res['tails']['outlier_only_p10']['1']['base_1']
    check('8f. a tail record beside every score',
          all(set(tl[a][m]) == set(arms[a][m]) for a in arms for m in arms[a]))
    syn = res['summary']['per_condition']['outlier_only_p10']['1']['base_1'][
        'per_model']['rf']['synthetic']
    check('8g. the summary holds a curve per share and operation',
          set(syn) == {'share0.2'} and set(syn['share0.2']) == set(n31.OPERATIONS)
          and all('break_even_precision' in e
                  for e in syn['share0.2'].values()))
    dr = list(_base1(res, 'outlier_only_p10', '1').values())
    check('8g2. the flag count is the swept share of training molecules, not '
          'the number moved',
          dr and all(r['flag_count_by_share'] == {'0.2': int(round(
              0.2 * r['training_molecules']))} for r in dr),
          f'{len(dr)} records')

    res_x = run_it(make_data(), 'toy', models=models,
                   base_reps={'rf': ['toy', 'second'],
                              'lightgbm': ['second', 'toy']},
                   conditions=['shift_only_p10'], levels=[1.0], draws=[0],
                   precisions=[1.0], flag_seeds=[0], flag_shares=[0.1])
    dx = res_x['diagnostics']['shift_only_p10']['1']
    sx = res_x['per_condition']['shift_only_p10']['1']['base_1']['noisy']
    check('8h. models with different rank-1 representations get separate '
          'matrices in base_1, and both are scored',
          {'base_1/rf', 'base_1/lightgbm'} <= set(dx)
          and set(sx) == set(models), f'{sorted(dx)}')

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
    row = res_s['summary']['per_condition']['shift_only_p10']['1']['base_1'][
        'per_model']['rf']
    b = res_s['per_condition']['shift_only_p10']['1']['base_1']['noisy']['rf']
    c = res_s['per_condition']['shift_only_p10']['1']['base_1']['clean']['rf']
    k = sorted(set(b) & set(c))
    check('9e. a spearman gain is arm minus noisy',
          np.isclose(row['clean_vs_noisy']['mean_gain'],
                     np.mean([c[x] - b[x] for x in k])))
    row2 = res['summary']['per_condition']['outlier_only_p10']['1']['base_1'][
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
              '--flag-seeds', '--flag-shares', '--models', '--exp-folds',
              '--tag')
    missing = [f for f in wanted if f not in out.stdout]
    check('10. the parser lists every flag',
          out.returncode == 0 and not missing,
          f'missing {missing}' if missing else f'{len(wanted)} flags present')
    check('10b. --base-rep is gone', '--base-rep' not in out.stdout)


# ── the D-versus-D' rows and the checkpoint (2026-09-25) ────────────────────

def test_rows_and_checkpoint():
    """Plan section 2 on n31: the run writes D-versus-D' rows beside its
    result file, each with the metric direction and a gain signed by it, and
    leaves no per-pass checkpoint behind once the result is complete."""
    import json
    print("\nthe D-versus-D' rows and the checkpoint:")
    res = run_it(make_data(), 'toy_rows', models=['rf'], conditions=['outlier_only_p10'], levels=[1.0], draws=[0], precisions=[0.5], flag_seeds=[0])
    path = res['rows']['rows_file']
    rows = ([json.loads(l) for l in open(path)]
            if path and os.path.exists(path) else [])
    check('rows were written to the rows file', len(rows) > 0, str(path))
    methods = {x['method'] for x in rows}
    check('the rows name the methods that change the training data',
          {'remove', 'downweight', 'rewrite', 'bayes_shrink'} <= methods, str(sorted(methods)))
    settings = {x['method_setting'] for x in rows}
    check('every swept share is written, and no row is marked as using '
          'injection knowledge (bayes_shrink estimates its weight since '
          '2026-09-28)',
          all(any(f'share{sh:g}/' in st for st in settings)
              for sh in n31.FLAG_SHARES)
          and not any('uses_injection' in x['method_setting'] for x in rows),
          f'{len(settings)} settings')
    check('every row carries the direction the dataset\'s metric has',
          all(x['lower_is_better'] is bool(res['lower_is_better'])
              for x in rows), str(res['lower_is_better']))
    ok = all((x['gain'] != x['gain']) or abs(
        x['gain'] - ((x['score_D'] - x['score_Dprime'])
                     if x['lower_is_better']
                     else (x['score_Dprime'] - x['score_D']))) < 1e-12
             for x in rows)
    check('gain is score_D against score_Dprime, signed by the direction', ok)
    check('the clean fit is written beside every row',
          all(x['score_clean'] == x['score_clean'] for x in rows))
    ck = os.path.join(OUT, 'n31_regime_and_breakeven', '_checkpoints')
    check('no checkpoint is left once the result is complete',
          not (os.path.isdir(ck) and os.listdir(ck)), ck)


if __name__ == '__main__':
    test_formulas()
    test_icc()
    test_flags_and_ops()
    test_break_even()
    test_shift_injector()
    test_measured_near_predicted()
    test_full_runs()
    test_the_command_line()
    test_rows_and_checkpoint()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
