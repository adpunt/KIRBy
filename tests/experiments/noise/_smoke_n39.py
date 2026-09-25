#!/usr/bin/env python3
"""Smoke for n39: several sources per molecule, with correlated errors and offsets.

What has to be true before any number n39 writes can be read:

  1. THE INJECTED ERROR HAS THE STRUCTURE CLAIMED. With no offsets, the error
     of a molecule's mean label has variance sigma_e^2 [rho + (1 - rho) / k].
  2. THE LEVELS NEST. At one draw, two levels move the same molecules in the
     same direction, scaled by the level.
  3. RHO_HAT RECOVERS RHO, given the injected sigma_e, at rho 0, 0.3 and 0.7.
  4. B_HAT RECOVERS B, by MixedLM and by ALS, and the two agree.
  5. THE PAIR RESIDUAL CORRELATION equals rho when the model error is zero, and
     the pair is drawn from two different documents when documents are given.
  6. The herg_ic50 count distribution loads, or falls back with a reason.
  7. A CLASS LABEL IS SKIPPED.
  8. A FULL RUN, base only and with a hybrid recipe, writes every arm for every
     model and space, with a tail record per score, and no errors.
  9. A DATASET WITH REPEAT MEASUREMENTS writes the real-data block.
 10. Under spearman a higher score counts as a gain.
 11. The parser lists every flag.
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

import experiments.noise.n39_correlated_sources as n39
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n39_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=180, n_feat=4, seed=0, domain='molecular'):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {'labels': y, 'reps': {'toy': X, 'second': X2},
            'groups': np.arange(n), 'task_type': 'regression',
            'domain': domain, 'smiles': ['C'] * n}


def make_replicate_data(n_mol=150, seed=1):
    """One row per measurement; 1-3 measurements per molecule, 12 documents."""
    rng = np.random.default_rng(seed)
    X_mol = rng.normal(size=(n_mol, 4))
    f = X_mol @ rng.normal(size=4) + 5.0
    k = rng.integers(1, 4, size=n_mol)
    rows = np.repeat(np.arange(n_mol), k)
    y = f[rows] + rng.normal(0, 0.3, rows.size)
    return {'labels': y, 'reps': {'toy': X_mol[rows],
                                  'second': X_mol[rows][:, ::-1]},
            'groups': rows.copy(), 'task_type': 'regression',
            'domain': 'molecular', 'smiles': ['C'] * rows.size,
            'replicate_groups': rows.copy(),
            'record_documents': np.array([f'D{d}' for d in
                                          rng.integers(0, 12, rows.size)])}


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


SMALL = dict(count_distributions=['uniform_1_to_4'], levels=[0.5],
             rhos=[0.0, 0.7], tau_fracs=[0.5], draws=[0])


def run_it(data, name, recipe=no_recipe, **kw):
    real = n39.get_best_search_for_downstream
    n39.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_reps', {m: ['toy', 'second']
                                    for m in n39.DOWNSTREAM_MODELS})
        return n39.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n39.get_best_search_for_downstream = real


def test_structure():
    print("the injected structure:")
    n = 6000
    shares = np.full(4, 0.25)
    st = n39.draw_structure(n, shares, seed=5)
    y0 = np.zeros(n)
    ok, det = True, []
    for rho in (0.0, 0.3, 0.7):
        _, m, _ = n39.simulate(y0, st, 1.0, rho, 0.0)
        for k in (1, 2, 4):
            emp = float(np.var(m[st['k'] == k]))
            want = rho + (1 - rho) / k
            det.append(f'rho {rho} k {k}: {emp:.3f} vs {want:.3f}')
            if abs(emp - want) > 0.12 * want + 0.02:
                ok = False
    check('1. mean-label error variance is sigma_e^2 [rho + (1-rho)/k]', ok,
          '; '.join(det[:3]) + ' ...')
    v1, _, _ = n39.simulate(y0, st, 0.5, 0.3, 0.2)
    v2, _, _ = n39.simulate(y0, st, 1.0, 0.3, 0.4)
    check('2. the levels nest: level 1.0 is exactly twice level 0.5',
          np.allclose(v2, 2 * v1))


def test_recovery():
    print("\nrho_hat and b_hat:")
    n = 2000
    st = n39.draw_structure(n, np.full(4, 0.25), seed=11)
    rng = np.random.default_rng(0)
    y = rng.normal(0, 2.0, n)
    ok, det = True, []
    for rho in (0.0, 0.3, 0.7):
        val, _, b = n39.simulate(y, st, 1.0, rho, 0.5)
        idx = np.arange(n)
        mol, src, v = n39.restrict(st, val, idx)
        b_m, s_m, info_m = n39.fit_offsets(mol, src, v, n39.N_SOURCES, 'mixedlm')
        b_a, s_a, info_a = n39.fit_offsets(mol, src, v, n39.N_SOURCES, 'als')
        r_m = n39.rho_hat_from(s_m, 1.0)
        r_a = n39.rho_hat_from(s_a, 1.0)
        used = np.unique(src)
        rec_m = n39.offset_recovery(b_m, b, used)
        rec_a = n39.offset_recovery(b_a, b, used)
        agree = float(np.corrcoef(b_m[used], b_a[used])[0, 1])
        det.append(f'rho {rho}: mixedlm {r_m:.3f}, als {r_a:.3f}; b corr '
                   f'{rec_m["corr"]:.3f}/{rec_a["corr"]:.3f}, agree {agree:.3f}')
        if not (abs(r_m - rho) < 0.06 and abs(r_a - rho) < 0.06):
            ok = False
        if not (rec_m['corr'] > 0.95 and rec_a['corr'] > 0.95 and agree > 0.98):
            ok = False
        if info_m['method'] != 'mixedlm':
            ok = False
    for d in det:
        print('      ' + d)
    check('3/4. rho_hat within 0.06 of rho; b_hat corr > 0.95 by both methods; '
          'they agree', ok)
    # The fallback path is taken above the row cap.
    real_cap = n39.MIXEDLM_MAX_ROWS
    n39.MIXEDLM_MAX_ROWS = 10
    try:
        _, _, info = n39.fit_offsets(mol, src, v, n39.N_SOURCES, 'mixedlm')
    finally:
        n39.MIXEDLM_MAX_ROWS = real_cap
    check('4b. above the row cap it falls back to ALS and says why',
          info['method'] == 'als' and 'fell_back_from_mixedlm_because' in info)


def test_pairs():
    print("\nthe real-data pair statistic:")
    rng = np.random.default_rng(2)
    n_mol, rho = 3000, 0.4
    rec_mol = np.repeat(np.arange(n_mol), 2)
    u = rng.normal(size=n_mol)
    y = (np.sqrt(rho) * u[rec_mol]
         + np.sqrt(1 - rho) * rng.normal(size=rec_mol.size))
    mols, a, b, cnt = n39.real_pairs(y, rec_mol, None, np.arange(n_mol), seed=0)
    c = n39.pair_residual_correlation(a, b)  # model error zero, f = 0
    check('5. with no model error the pair residual correlation is rho',
          abs(c - rho) < 0.05, f'{c:.3f} vs {rho}')
    docs = np.array([0, 0, 1, 2, 3, 3])
    rm = np.array([0, 0, 1, 1, 2, 2])
    yy = np.arange(6, dtype=float)
    mols, a, b, cnt = n39.real_pairs(yy, rm, docs, np.arange(3), seed=0)
    check('5b. with documents, pairs come from two documents only',
          list(mols) == [1] and cnt['molecules_with_one_document_only'] == 2
          and {a[0], b[0]} == {2.0, 3.0}, str(cnt))


def test_counts():
    print("\nthe count distribution:")
    shares, src = n39.count_shares('herg_ic50')
    check('6. herg_ic50 shares load (live or fallback) and sum to 1',
          abs(shares.sum() - 1) < 1e-9 and shares[0] > 0.5, f'{src}; '
          f'{np.round(shares, 3).tolist()}')


def test_skip():
    print("\nthe skip:")
    d = make_data()
    d['task_type'] = 'classification'
    r = run_it(d, 'toy')
    check('7. a class label is skipped', r.get('skipped') == 'not_regression')


def test_full(recipe, label):
    print(f"\na full run, {label}:")
    models = ['rf', 'lightgbm']
    res = run_it(make_data(), 'toy', recipe=recipe, models=models, **SMALL)
    errs = res.get('_errors', {})
    check(f'8. [{label}] no errors', not errs, str(list(errs.items())[:3]))
    check(f'8. [{label}] base_1 and base_2 ran, and each model\'s pair is named',
          res['spaces_run'][:2] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {m: ['toy', 'second']
                                            for m in models}
          and 'base_rep' not in res,
          f"{res['spaces_run']}, {res.get('base_reps_by_model')}")
    pc = res['per_condition']
    ok = True
    for ck in pc:
        for sp in res['spaces_run']:
            for arm in n39.ARMS:
                for m in models:
                    # A recipe that is not shareable across models is written
                    # under 'space/model', as in n34.
                    cellp = pc[ck]['0.5']
                    got = (cellp.get(sp, {}).get(arm, {}).get(m)
                           or cellp.get(f'{sp}/{m}', {}).get(arm, {}).get(m, {}))
                    if len(got) != 3:
                        ok = False
    check(f'8. [{label}] every arm x model x space x condition has 3 passes',
          ok and len(pc) == 2, f'conditions {list(pc)}, spaces '
          f"{res['spaces_run']}")
    t = res['tail_per_condition'][list(pc)[0]]['0.5']['base_1']['plain_mean']['rf']
    check(f'8. [{label}] a tail record per score',
          len(t) == 3 and all(v is None or 'tail_rmse_top' in v for v in t.values()))
    est = res['summary']['estimates']
    check(f'8. [{label}] rho_hat is written per condition',
          all(e['0.5']['rho_hat_mean'] is not None for e in est.values()),
          str({k: v['0.5']['rho_hat_mean'] for k, v in est.items()}))
    if recipe is not no_recipe:
        check(f'8. [{label}] a hybrid space ran', 'hybrid' in res['spaces_run'])
    return res


def test_replicate():
    print("\na dataset with repeat measurements:")
    res = run_it(make_replicate_data(), 'toy_rep', models=['lightgbm'], **SMALL)
    rd = res.get('real_data')
    ok = (rd is not None and len(rd['per_fold']) == 3
          and rd['documents_available']
          and all('base_1' in f['per_space'] and 'base_2' in f['per_space']
                  for f in rd['per_fold'])
          and all(f['rho_at_published_error'] is None
                  and f['rho_at_published_error_reason'] for f in rd['per_fold']))
    check('9. the real-data block is written per fold, rho at a published '
          'error refused with a reason', ok,
          str(rd['per_fold'][0] if rd else None)[:300])
    check('9. the simulation ran on one label per molecule',
          res['molecules'] == 150 and not res.get('_errors'),
          f"{res['molecules']} molecules; errors {list(res.get('_errors', {}))[:2]}")


def test_direction():
    print("\nthe direction under spearman:")
    res = run_it(make_data(domain='antibody'), 'toy', models=['rf'],
                 count_distributions=['uniform_1_to_4'], levels=[0.5],
                 rhos=[0.3], tau_fracs=[0.0], draws=[0])
    ck = list(res['per_condition'])[0]
    raw = res['per_condition'][ck]['0.5']['base_1']
    e = res['summary']['per_condition'][ck]['0.5']['base_1']['rf']['per_arm']['clean']
    gains = [raw['clean']['rf'][k] - raw['plain_mean']['rf'][k]
             for k in raw['plain_mean']['rf']]
    check('10. under spearman a higher score counts as a gain',
          res['lower_is_better'] is False
          and np.allclose(sorted(e['vs_plain_mean']['pass_gains']), sorted(gains)),
          f"metric {res['metric']}")


def test_parser():
    print("\nthe parser:")
    import subprocess
    out = subprocess.run([sys.executable, n39.__file__, '--help'],
                         capture_output=True, text=True).stdout
    flags = ['--levels', '--rhos', '--tau-fracs', '--count-distributions',
             '--draws', '--offset-method', '--hybrid-labels', '--exp-folds',
             '--tag']
    missing = [f for f in flags if f not in out]
    check('11. the parser lists every flag', not missing, f'missing {missing}')
    check('11b. --base-rep is gone', '--base-rep' not in out)


if __name__ == '__main__':
    test_structure()
    test_recovery()
    test_pairs()
    test_counts()
    test_skip()
    test_full(no_recipe, 'base only')
    test_full(fake_recipe(['toy', 'second']), 'with a hybrid recipe')
    test_replicate()
    test_direction()
    test_parser()
    print("\n" + ("ALL GREEN" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)
