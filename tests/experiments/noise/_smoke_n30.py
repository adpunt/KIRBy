#!/usr/bin/env python3
"""Smoke for n30: clipped labels -- lift, drop, winsorise, or a censored likelihood.

What has to be true before any number n30 writes can be read:

  1. THE CLIP DOES NOT DEPEND ON THE DRAW. Noise seeds 0, 1 and 2 give the same
     clipped column, which is why n30 runs one draw.
  2. THE ROWS AT THE LIMIT ARE THE ROWS THE CLIP MOVED, on continuous toy labels,
     and the limit comes back on the provenance row.
  3. THE CONSTANT LIFT IS THE FORMULA. Every row at the limit moves up by the
     mean of sd * phi(a) / (1 - Phi(a)); no other row moves.
  4. THE SHUFFLED LIFTS ARE THE SAME AMOUNTS ON OTHER MOLECULES. Same multiset,
     different order, other rows untouched.
  5. WINSORISATION MOVES ONLY LABELS BEYOND c * s, and moves them exactly to
     the threshold.
  6. THE AFT PREDICTIONS ARE ON THE LABEL SCALE. On uncensored toy labels they
     track the labels in their own units, not log units and not shifted.
  7. THE RECOVERY SHARE IS SIGNED BY THE METRIC AND REFUSED UNDER THE 2% FLOOR.
  8. A CONDITION OTHER THAN censoring, AND A CLASSIFICATION LABEL, ARE REFUSED.
  9. A FULL RUN, single representations only and with a hybrid recipe, writes every arm, every
     swap, and a clip record per pass. Under spearman the direction flips.
 10. The parser lists every flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Two OpenMP runtimes in one process exit with 139 and no traceback --
# project_fold_hook_and_lgbm_segfault_2026_09_02. Both switches, before numpy.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import tempfile

import numpy as np
from scipy.stats import norm

import experiments.noise.n30_censored_likelihood as n30
from experiments.noise.noise_helpers import noise_spec_for, inject_train_labels
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n30_smoke_')
FAIL = []

# 2026-09-25: n30 now caches through noise_cache under KIRBY_CACHE_DIR and
# checkpoints under the output directory, so the toy entries go to a
# temporary directory, as tests/smoke/smoke_noise_similarity_foundation.py
# does.
os.environ['KIRBY_CACHE_DIR'] = tempfile.mkdtemp(prefix='n30_smoke_cache_')


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=180, n_feat=4, seed=0, domain='molecular'):
    """Two representations and a label built from the first. Singleton groups."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {'labels': y, 'reps': {'toy': X, 'second': X2},
            'groups': np.arange(n), 'task_type': 'regression',
            'domain': domain}


def fake_recipe(rep_names, budget=4):
    """A fixed recipe standing in for the pool, as _smoke_n25 does."""
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
    real = n30.get_best_search_for_downstream
    n30.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_reps', {m: ['toy', 'second']
                                    for m in kw.get('models', ['rf'])})
        return n30.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n30.get_best_search_for_downstream = real


def test_injector():
    print("what the injector does:")
    y = make_data()['labels']
    tr = np.arange(140)
    cols, lims = [], []
    for d in (0, 1, 2):
        spec = noise_spec_for('toy', 'censoring', 0.25, noise_seed=d)
        y0, m, prov = inject_train_labels(y, spec, train_idx=tr)
        cols.append(np.asarray(y0))
        lims.append(prov.get('censoring_limit'))
    check('1. draws 0, 1 and 2 give the same clipped column',
          all(np.array_equal(cols[0], c) for c in cols[1:]))
    at = n30.rows_at_limit(cols[0], lims[0])
    moved = cols[0] != y[tr]
    check('2. the rows at the limit are the rows the clip moved',
          lims[0] is not None and np.array_equal(at, moved),
          f'{int(at.sum())} at the limit, {int(moved.sum())} moved')


def test_lifts_and_winsor():
    print("\nthe label arms, no model fitted:")
    rng = np.random.default_rng(1)
    n = 200
    y = rng.normal(0, 1, n)
    limit = 0.8
    y0 = np.minimum(y, limit)
    at = n30.rows_at_limit(y0, limit)
    p = y + rng.normal(0, 0.4, n)
    sd = n30.residual_sd(y0, p, at)
    cols, info = n30.lift_columns(y0, p, at, limit, sd, (0, 1), (0, 0, 0))
    a = (limit - p[at]) / sd
    expect = float(np.mean(np.maximum(0.0, sd * norm.pdf(a) / (1 - norm.cdf(a)))))
    c = cols['constant_lift']
    check('3. the constant lift raises every row at the limit by the formula',
          np.allclose(c[at] - limit, expect) and np.array_equal(c[~at], y0[~at]),
          f'lift {expect:.4f} label units on {int(at.sum())} rows')
    pm = cols['per_molecule_lift']
    sh = cols['per_molecule_lift_shuffled/seed0']
    check('4. the shuffled lifts are the same amounts on other molecules',
          np.allclose(np.sort(pm[at]), np.sort(sh[at]))
          and not np.allclose(pm[at], sh[at])
          and np.array_equal(sh[~at], y0[~at]))
    tn = cols['truncated_normal_mean']
    check('3b. the truncated-normal mean is floored at the limit and differs '
          'from the probe formula',
          np.all(tn[at] >= limit - 1e-12) and not np.allclose(tn[at], pm[at]))

    w, s, moved = n30.winsorise(y0, p, 1.0)
    r = y0 - p
    beyond = np.abs(r) > s
    check('5. winsorisation moves only labels beyond c * s, to the threshold',
          np.array_equal(w[~beyond], y0[~beyond])
          and np.allclose(np.abs(w[beyond] - p[beyond]), s)
          and moved == int(beyond.sum()),
          f'{moved} of {n} labels moved, s = {s:.4f}')


def test_aft_scale():
    print("\nthe censored likelihood:")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    y = 2.0 * X[:, 0] + 7.0 + rng.normal(0, 0.2, 300)
    at = np.zeros(300, dtype=bool)
    pred = n30.fit_aft_predict(X[:200], y[:200], at[:200], X[200:], 0.2)
    err = float(np.sqrt(np.mean((pred - y[200:]) ** 2)))
    check('6. AFT predictions are on the label scale',
          err < 1.0 and abs(float(np.mean(pred)) - float(np.mean(y[200:]))) < 0.5,
          f'held-out rmse {err:.3f} label units, label sd {np.std(y):.3f}')
    y0 = np.minimum(y, np.quantile(y, 0.75))
    at = n30.rows_at_limit(y0, float(np.quantile(y, 0.75)))
    pc = n30.fit_aft_predict(X[:200], y0[:200], at[:200], X[200:], 0.2)
    check('6b. with rows at the limit, AFT predicts above the limit somewhere',
          float(np.max(pc)) > float(np.quantile(y, 0.75)),
          f'max prediction {np.max(pc):.3f}, limit {np.quantile(y, 0.75):.3f}')


def test_recovery_share():
    print("\nthe recovery share:")
    s_lo, _ = n30.recovery_share(0.60, 0.70, 0.65, True)
    s_hi, _ = n30.recovery_share(0.60, 0.50, 0.55, False)
    check('7. half the gap is 0.5 under rmse and under spearman',
          abs(s_lo - 0.5) < 1e-12 and abs(s_hi - 0.5) < 1e-12)
    s, why = n30.recovery_share(0.60, 0.605, 0.60, True)
    check('7b. a gap under 2% of the clean score is refused with a reason',
          s is None and why is not None, why or '')


def test_guards():
    print("\nthe guards:")
    d = make_data()
    try:
        run_it(d, 'toy', noise_condition='outlier_p10', models=['rf'])
        ok = False
    except ValueError:
        ok = True
    check('8. a condition other than censoring is refused', ok)
    dc = dict(d, labels=(d['labels'] > 5).astype(int),
              task_type='classification')
    rc = run_it(dc, 'toy', models=['rf'])
    check('8b. a classification label is skipped with its reason',
          isinstance(rc, dict) and rc.get('skipped') == 'not_regression',
          str(rc)[:80])


def test_full_runs():
    print("\nfull runs:")
    models = ['rf', 'xgboost', 'lightgbm']
    res = run_it(make_data(), 'toy', models=models, levels=[0.0, 0.25])
    check('9. both single-representation spaces ran, and are named',
          res['spaces_run'] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {m: ['toy', 'second']
                                            for m in models},
          f"{res['spaces_run']}, {res.get('base_reps_by_model')}")
    check('9. base_2 is scored too', 'clipped' in
          res['per_level']['0.25'].get('base_2', {}))
    lv = res['per_level']['0.25']['base_1']
    missing = [a for a in res['arms'] if a not in lv]
    check('9. every label arm is scored at fraction 0.25', not missing,
          f'missing {missing}' if missing else f"{len(res['arms'])} arms")
    swaps = [s for s in ('xgboost_aft', 'rf_l1', 'lightgbm_huber')
             if s in lv['clipped']]
    check('9b. every model swap is scored on the clipped labels',
          len(swaps) == 3, f'{swaps}')
    l0 = res['per_level']['0']['base_1']
    check('9c. at fraction 0.0 the lift and drop arms are absent, not zero',
          all(a not in l0 for a in n30.NEEDS_A_LIMIT)
          and 'winsor_c1' in l0)
    recs = res['clip_per_pass']['0.25']
    check('9d. one clip record per pass, with the limit and both counts',
          len(recs) == 3 and all(r['limit'] is not None
                                  and r['molecules_at_the_limit'] > 0
                                  for r in recs))
    sw = res['summary']['per_level']['0.25']['base_1']['model_swaps']
    check('9e. xgboost_aft is paired against xgboost on the constant lift too',
          'vs_parent_on_constant_lift' in sw.get('xgboost_aft', {}))
    n_pass = sum(v['vs_clipped']['n_passes_paired'] for v in
                 res['summary']['per_level']['0.25']['base_1']['per_model'][
                     'rf']['per_arm'].values())
    check('9f. paired counts are bounded by the passes that ran',
          n_pass <= 3 * len(res['arms']))

    res_h = run_it(make_data(), 'toy', recipe=fake_recipe(['toy', 'second']),
                   models=['rf', 'lightgbm'], levels=[0.25], skip_aft=True)
    check('9g. with a recipe, the hybrid space is built and scored',
          'hybrid' in res_h['spaces_run']
          and 'hybrid' in res_h['per_level']['0.25']
          and 'constant_lift' in res_h['per_level']['0.25']['hybrid'])
    hs = res_h['per_level']['0.25']['hybrid']['clipped']
    check('9h. rf_l1 and lightgbm_huber are fitted in the hybrid space',
          'rf_l1' in hs and 'lightgbm_huber' in hs)

    res_s = run_it(make_data(domain='antibody'), 'toy_ab', models=['rf'],
                   levels=[0.25], skip_aft=True)
    rmse_row = res['summary']['per_level']['0.25']['base_1']['per_model']['rf']
    sp_row = res_s['summary']['per_level']['0.25']['base_1']['per_model']['rf']
    check('9i. the metric direction is read off the dataset',
          res['metric'] == 'rmse' and res['lower_is_better']
          and res_s['metric'] == 'spearman' and not res_s['lower_is_better'])
    # Under spearman, positive gain means the arm's correlation is higher.
    e = sp_row['per_arm']['clean']['vs_clipped']
    b = res_s['per_level']['0.25']['base_1']['clipped']['rf']
    g = res_s['per_level']['0.25']['base_1']['clean']['rf']
    k = sorted(set(b) & set(g))
    check('9j. a spearman gain is arm minus clipped',
          np.isclose(e['mean_gain'], np.mean([g[x] - b[x] for x in k])))
    e2 = rmse_row['per_arm']['clean']['vs_clipped']
    b2 = res['per_level']['0.25']['base_1']['clipped']['rf']
    g2 = res['per_level']['0.25']['base_1']['clean']['rf']
    k2 = sorted(set(b2) & set(g2))
    check('9k. an rmse gain is clipped minus arm',
          np.isclose(e2['mean_gain'], np.mean([b2[x] - g2[x] for x in k2])))


def test_the_command_line():
    print("\nthe command line:")
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      'n30_censored_likelihood.py'), '--help'],
        capture_output=True, text=True)
    wanted = ('--dataset', '--noise-condition', '--levels', '--draws',
              '--winsor-c', '--models', '--exp-folds', '--tag')
    missing = [f for f in wanted if f not in out.stdout]
    check('10. the parser lists every flag',
          out.returncode == 0 and not missing,
          f'missing {missing}' if missing else f'{len(wanted)} flags present')
    check('10b. --base-rep is gone', '--base-rep' not in out.stdout)


# ── the D-versus-D' rows and the checkpoint (2026-09-25) ────────────────────

def test_rows_and_checkpoint():
    """Plan section 2 on n30: the run writes D-versus-D' rows beside its
    result file, each with the metric direction and a gain signed by it, and
    leaves no per-pass checkpoint behind once the result is complete."""
    import json
    print("\nthe D-versus-D' rows and the checkpoint:")
    res = run_it(make_data(), 'toy_rows', models=['rf'], levels=[0.25])
    path = res['rows']['rows_file']
    rows = ([json.loads(l) for l in open(path)]
            if path and os.path.exists(path) else [])
    check('rows were written to the rows file', len(rows) > 0, str(path))
    methods = {x['method'] for x in rows}
    check('the rows name the methods that change the training data',
          {'constant_lift', 'drop_clipped', 'truncated_normal_mean'} <= methods, str(sorted(methods)))
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
    ck = os.path.join(OUT, 'n30_censored_likelihood', '_checkpoints')
    check('no checkpoint is left once the result is complete',
          not (os.path.isdir(ck) and os.listdir(ck)), ck)


if __name__ == '__main__':
    test_injector()
    test_lifts_and_winsor()
    test_aft_scale()
    test_recovery_share()
    test_guards()
    test_full_runs()
    test_the_command_line()
    test_rows_and_checkpoint()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
