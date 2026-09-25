#!/usr/bin/env python3
"""Smoke for n41: flags with a controlled false-flag rate.

What has to be true before any number n41 writes can be read:

  1. THE CONFORMAL P-VALUE is (1 + calibration scores at least as high) /
     (calibration size + 1), and BENJAMINI-HOCHBERG agrees with statsmodels.
  2. UNDER THE NULL the realised false-flag share, averaged over 400 draws, is
     at or under q for every q in BH_QS.
  3. THE TRUSTED CALIBRATION SET holds only molecules measured 3+ times, and
     only the lower half of them by spread.
  4. WINSORISING THE FLAGGED leaves every unflagged label bit-identical and puts
     every flagged one within c*s of the prediction.
  5. RANDOM FLAGS are the asked count, drawn only from the candidates.
  6. THE JOINED MATRIX gives the residual as much variance as all the features.
  7. CLEANLAB runs and ranks a large shift above chance, on tiny data.
  8. A FULL RUN under sparse_dprime writes every arm, every detector with an
     AUROC, BH records at level 0 and at the injected level, the residual SD
     it sized the shift by, tail scores, and random twins for every flag arm.
  9. With replicate groups the trusted calibration set is run.
 10. With a recipe the hybrid space is built and scored.
 11. The guards: a condition that moves every label is refused; a class label
     is skipped; the default sweep covers every detection-defined condition.
 12. The parser lists every flag.
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

import experiments.noise.n41_conformal_fdr_detection as n41
RC = [n41.random_cal_name(s) for s in n41.CALIBRATION_SHARES]
from experiments.noise.noise_helpers import cleanlab_suspicion
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n41_smoke_')
FAIL = []
NO_CLEANLAB = [d for d in n41.DETECTORS if d != 'cleanlab']


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=180, n_feat=4, seed=0, domain='molecular', replicates=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    d = {'labels': y, 'reps': {'toy': X, 'second': X2},
         'groups': np.arange(n), 'task_type': 'regression', 'domain': domain}
    if replicates:
        # molecules of 4 rows each; the rows share features and scaffold group
        mol = np.arange(n) // 4
        base = rng.normal(size=(mol.max() + 1, n_feat))
        X = base[mol] + rng.normal(0, 0.01, size=(n, n_feat))
        y = X @ np.ones(n_feat) + 5.0 + rng.normal(0, 0.3, n)
        d.update(labels=y, reps={'toy': X, 'second': X[:, ::-1]},
                 groups=mol, replicate_groups=mol)
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
    real = n41.get_best_search_for_downstream
    n41.get_best_search_for_downstream = recipe
    try:
        # Every model's top 2 single representations, given directly.
        kw.setdefault('base_reps', {m: ['toy', 'second']
                                    for m in n41.DOWNSTREAM_MODELS})
        kw.setdefault('noise_condition', n41.DEFAULT_NOISE_CONDITION)
        kw.setdefault('detectors', NO_CLEANLAB)
        return n41._run_one_condition(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n41.get_best_search_for_downstream = real


def test_pvalues_and_bh():
    print("conformal p-values and Benjamini-Hochberg:")
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    got = n41.conformal_pvalues(cal, np.array([0.5, 2.0, 3.5, 9.0]))
    want = np.array([5, 4, 2, 1]) / 5.0
    check('1. p = (1 + calibration scores >= s) / (n_cal + 1)',
          np.allclose(got, want), f'{got} vs {want}')
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(200):
        p = np.concatenate([rng.uniform(size=40), rng.uniform(0, 0.01, 5)])
        for q in (0.05, 0.1, 0.2):
            ok &= np.array_equal(n41.benjamini_hochberg(p, q),
                                 multipletests(p, q, method='fdr_bh')[0])
    check('1b. Benjamini-Hochberg agrees with statsmodels on 600 draws', ok)
    clean = np.abs(rng.normal(size=200))
    test = np.concatenate([np.abs(rng.normal(size=100)),
                           8.0 + np.abs(rng.normal(size=60))])
    moved = np.r_[np.zeros(100, bool), np.ones(60, bool)]
    f = n41.benjamini_hochberg(n41.conformal_pvalues(clean, test), 0.1)
    prec = float(moved[f].mean()) if f.any() else 0.0
    check('1c. with a clean calibration set, 60 large outliers among 160 are '
          'flagged, most of them real',
          f.sum() >= 50 and prec >= 0.8, f'{int(f.sum())} flagged, precision {prec:.2f}')
    check('1d. the floor on how few BH can flag is (1 - s) / (s q) molecules',
          n41.min_bh_flags(96, 24, 0.1) == 39 and n41.min_bh_flags(720, 180, 0.1) == 40)


def test_null():
    print("\nthe null:")
    rng = np.random.default_rng(1)
    shares = {(sh, q): [] for sh in n41.CALIBRATION_SHARES for q in n41.BH_QS}
    for d in range(400):
        s = np.abs(rng.standard_t(4, size=200))
        for sh in n41.CALIBRATION_SHARES:
            cal = n41.random_calibration(200, sh, d)
            pv = n41.conformal_pvalues(s[cal], s[~cal])
            for q in n41.BH_QS:
                flags = np.zeros(200, dtype=bool)
                flags[np.flatnonzero(~cal)[n41.benjamini_hochberg(pv, q)]] = True
                shares[(sh, q)].append(n41.bh_record(
                    flags, np.zeros(200, bool), cal)['false_flag_share'])
    means = {k: float(np.mean(v)) for k, v in shares.items()}
    check('2. with nothing moved the false-flag share averages at or under q',
          all(means[k] <= k[1] + 0.02 for k in means),
          ', '.join(f'share {k[0]:g} q {k[1]:g}: {m:.3f}'
                    for k, m in means.items()))


def test_parts():
    print("\nthe parts:")
    g = np.repeat(np.arange(6), [1, 2, 3, 3, 4, 5])
    rng = np.random.default_rng(2)
    spread = np.array([0, 0, 0.1, 2.0, 0.2, 3.0])
    y = rng.normal(size=g.size) * spread[g]
    mask, rec = n41.trusted_calibration(y, g)
    kept = set(np.unique(g[mask]))
    check('3. the trusted set holds molecules measured 3+ times, the lower half '
          'by spread',
          kept == {2, 4} and rec['molecules_measured_enough'] == 4,
          f'kept groups {sorted(kept)}')
    m0, r0 = n41.trusted_calibration(y, None)
    check('3b. no replicate groups gives no trusted set, with the reason',
          m0 is None and 'why_none' in r0)

    y = rng.normal(size=100) * 2
    p = np.zeros(100)
    flags = np.zeros(100, bool)
    flags[:10] = True
    w = n41.winsorise_flagged(y, p, flags, c=1.0)
    s = 1.4826 * np.median(np.abs(y))
    check('4. winsorising leaves unflagged labels bit-identical, flagged within '
          'c*s of p',
          np.array_equal(w[~flags], y[~flags])
          and np.all(np.abs(w[flags] - p[flags]) <= s + 1e-12))

    cand = np.zeros(100, bool)
    cand[20:] = True
    rf = n41.random_flags(cand, 7, 3)
    check('5. random flags: the asked count, only from the candidates',
          int(rf.sum()) == 7 and not rf[:20].any())

    X = rng.normal(size=(50, 16))
    Z = n41.joined_matrix(X, rng.normal(size=50) * 5)
    check('6. the residual column carries the variance of all 16 features',
          np.isclose(np.var(Z[:, -1]), 16.0))


def test_cleanlab():
    print("\ncleanlab, on tiny data:")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(0, 0.1, 60)
    m = np.zeros(60, bool)
    m[:6] = True
    y[m] += 4.0
    from sklearn.metrics import roc_auc_score
    s = cleanlab_suspicion(X, y, cv_n_folds=3)
    auc = roc_auc_score(m, s)
    check('7. cleanlab runs, one score per molecule, ranks a 4-unit shift above '
          'chance', s.shape == (60,) and auc > 0.7, f'AUROC {auc:.2f}')


def test_full_runs():
    print("\nfull runs:")
    res = run_it(make_data(), 'toy', models=['rf', 'lightgbm'],
                 noise_condition='sparse_dprime_p10', levels=[0.0, 4.0],
                 draws=[0], random_seeds=[0, 1])
    lv = res['per_level']['4']['base_1']
    missing = [a for a in res['arms'] if a not in lv]
    check('8. every arm is scored at level 4', not missing,
          f'missing {missing}' if missing else f"{len(res['arms'])} arms")
    det = res['detection']['4']['base_1']
    no_auroc = [d for d in NO_CLEANLAB
                if d not in det or not np.isfinite(det[d]['detection_auroc'])]
    check('8b. every detector is graded, with an AUROC', not no_auroc,
          ', '.join(f"{d} {det[d]['detection_auroc']:.2f}" for d in det))
    check('8c. the conformal score finds a 4-residual-SD shift well above chance',
          det['rescaled_residual']['detection_auroc'] > 0.8)
    bh0 = res['bh']['0']['base_1']
    bh4 = res['bh']['4']['base_1']
    check('8d. BH records are written at level 0 and at level 4, for every q',
          all(f'{c}_q{q:g}' in bh0 and f'{c}_q{q:g}' in bh4
              for c in RC for q in n41.BH_QS))
    ok, info = True, []
    for c, q in [(c, q) for c in RC for q in n41.BH_QS]:
        for r in bh4[f'{c}_q{q:g}']['passes']:
            fl = r['fewest_flags_bh_can_make']
            ok &= fl is not None and (r['molecules_flagged'] == 0
                                      or r['molecules_flagged'] >= fl)
            info.append(f"{c} q {q:g}: {r['molecules_flagged']} flagged, "
                        f"floor {fl}")
    check('8e. every BH record writes its floor, and flags nothing or at least '
          'the floor', ok, '; '.join(info[::3]))
    floors = {c: bh4[f'{c}_q0.1']['passes'][0]['fewest_flags_bh_can_make']
              for c in RC}
    check('8e2. the larger calibration share lowers the fewest flags BH can make',
          floors[RC[-1]] < floors[RC[0]], str(floors))
    orc = [r for q in n41.BH_QS
           for r in bh4[f'{n41.ORACLE_CAL_NAME}_q{q:g}']['passes']]
    check('8e3. the oracle calibration set holds no moved molecule',
          orc and all(r['moved_molecules_in_calibration'] == 0 for r in orc))
    o1 = bh4[f'{n41.ORACLE_CAL_NAME}_q0.1']['passes']
    check('8e4. with the oracle set, BH flags moved molecules at level 4',
          any(r['molecules_flagged'] > 0 for r in o1),
          '; '.join(f"{r['molecules_flagged']} flagged, precision "
                    f"{r['precision']}" for r in o1))
    recs = res['fold_records']['4']
    check('8f. the residual SD the shift was sized by is written per fold',
          all(r.get('resid_sd_label_units', 0) > 0 for r in recs))
    check('8f2. it is measured in the first model\'s rank-1 representation, '
          'and that is written',
          res.get('resid_sd_rep') == 'toy' and res.get('resid_sd_rep_chosen_as'),
          str(res.get('resid_sd_rep')))
    check('8f3. base_1 and base_2 both ran, and each model\'s pair is named',
          res['spaces_run'] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {m: ['toy', 'second']
                                            for m in ['rf', 'lightgbm']}
          and 'base_rep' not in res
          and 'no_cleaning' in res['per_level']['4']['base_2'],
          f"{res['spaces_run']}, {res.get('base_reps_by_model')}")
    tails = res['tails']['4']['base_1'][f'remove_{RC[0]}_q0.1']['rf']
    check('8g. tail scores are written for every scored pass',
          len(tails) == 3 and all('tail_rmse_top' in v for v in tails.values()))
    s = res['summary']['per_level']['4']['base_1']['rf']['per_arm']
    check('8h. every flag arm is paired against its two random twins',
          all(len(s[f'{a}_{c}_q{q:g}']['vs_random_flags']) == 2
              for a in n41.ACTIONS for c in RC for q in n41.BH_QS))
    z = res['per_level']['0']['base_1']
    check('8i. an arm with no flags carries no_cleaning\'s score, not a gap',
          all(set(z[a]['rf']) == set(z['no_cleaning']['rf']) for a in res['arms']
              if a != 'clean'))

    res_t = run_it(make_data(replicates=True), 'toy_rep', models=['rf'],
                   noise_condition='sparse_dprime_p10', levels=[4.0],
                   draws=[0], random_seeds=[0], bh_qs=[0.1],
                   detectors=['rescaled_residual'])
    check('9. with replicate groups the trusted calibration set is run',
          'trusted' in res_t['calibration_sets']
          and 'remove_trusted_q0.1' in res_t['per_level']['4']['base_1'],
          str(res_t['fold_records']['4'][0].get('trusted_calibration')))

    res_h = run_it(make_data(), 'toy', recipe=fake_recipe(['toy', 'second']),
                   models=['lightgbm'], noise_condition='outlier_only_p10',
                   levels=[1.0], draws=[0], random_seeds=[0], bh_qs=[0.1],
                   detectors=['rescaled_residual'])
    check('10. with a recipe the hybrid space is built and scored',
          'hybrid' in res_h['spaces_run']
          and f'winsorise_{RC[0]}_q0.1' in res_h['per_level']['1']['hybrid'])


def test_guards():
    print("\nthe guards:")
    d = make_data()
    try:
        run_it(d, 'toy', noise_condition='gaussian', models=['rf'])
        ok = False
    except ValueError:
        ok = True
    check('11. a condition that moves every label is refused', ok)
    r = n41.run(d, {'n_splits': 3}, OUT, 'toy', noise_condition='dense_oofmse')
    check('11a. run() given such a condition writes a skip, not a crash',
          isinstance(r, dict) and r.get('skipped') == 'detection_undefined')
    import inspect
    src = inspect.getsource(n41.run)
    check('11b. with no condition named, run() sweeps every detection-defined '
          'condition, sparse_dprime and shift_only included',
          'default_conditions=list(DETECTION_IS_DEFINED)' in src
          and any(c.startswith('sparse_dprime') for c in n41.DETECTION_IS_DEFINED)
          and all(c in n41.DETECTION_IS_DEFINED
                  for c in n41.SHIFT_ONLY_CONDITIONS))
    dc = dict(d, labels=(d['labels'] > 5).astype(int),
              task_type='classification')
    rc = n41.run(dc, {'n_splits': 3}, OUT, 'toy', models=['rf'])
    check('11c. a class label is skipped with its reason',
          isinstance(rc, dict) and rc.get('skipped') == 'not_regression')


def test_the_command_line():
    print("\nthe command line:")
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      'n41_conformal_fdr_detection.py'),
         '--help'], capture_output=True, text=True)
    wanted = ('--dataset', '--noise-condition', '--levels', '--draws',
              '--bh-qs', '--detectors', '--random-seeds', '--models',
              '--exp-folds', '--tag')
    missing = [f for f in wanted if f not in out.stdout]
    check('12b. --base-rep is gone', out.returncode == 0
          and '--base-rep' not in out.stdout)
    check('12. the parser lists every flag',
          out.returncode == 0 and not missing,
          f'missing {missing}' if missing else f'{len(wanted)} flags present')


if __name__ == '__main__':
    test_pvalues_and_bh()
    test_null()
    test_parts()
    test_guards()
    test_full_runs()
    test_cleanlab()
    test_the_command_line()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
