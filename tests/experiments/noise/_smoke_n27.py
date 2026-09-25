#!/usr/bin/env python3
"""Smoke for n27: removal cost and how label error reaches the test error.

What has to be true for the numbers n27 writes to be readable:

  1. THE KEPT MOLECULES ARE THE SAME ON EVERY RUN AND IN EVERY SPACE. The draw
     is seeded by crc32, not hash(), and the space is not in the key.
  2. A SHARE KEEPS THAT SHARE. 0.7 of 100 training molecules keeps 70.
  3. ALPHA 0 MOVES NOTHING AND ALPHA 1 IS THE PREDICTION.
  4. THE RATIO IS ITS NUMERATOR OVER ITS DENOMINATOR, and both are on the row.
  5. THE LOSS IS SIGNED BY THE METRIC'S DIRECTION. The same two scores give
     opposite signs under rmse and under spearman.
  6. A PASS PAIRS WITH ITS OWN FOLD. An injected pass is compared against the
     correct-label fit of the fold it ran on, never another fold.
  7. THE INJECTED VARIANCE GROWS WITH THE LEVEL, and the held-out labels are
     never touched.
  8. Single-representation-only and hybrid runs both finish, a classification label is refused,
     and the CLI parses.
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

import experiments.noise.n27_removal_cost_and_propagation as n27
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n27_smoke_')
FAIL = []

# 2026-09-25: n27 now caches through noise_cache under KIRBY_CACHE_DIR and
# checkpoints under the output directory, so the toy entries go to a
# temporary directory, as tests/smoke/smoke_noise_similarity_foundation.py
# does.
os.environ['KIRBY_CACHE_DIR'] = tempfile.mkdtemp(prefix='n27_smoke_cache_')
N = int(os.environ.get('KIRBY_SMOKE_N', '180'))


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=N, n_feat=4, seed=0, domain='molecular'):
    """Two representations and a label built from the first one.

    Groups are singletons: the fold split is not what this file tests.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    w = rng.normal(size=n_feat)
    y = X @ w + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {
        'labels': y,
        'reps': {'toy': X, 'second': X2},
        'groups': np.arange(n),
        'task_type': 'regression',
        'domain': domain,
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
    """What the pool returns on a dataset it was never built for."""
    return None, None


def run_it(data, name, recipe=no_recipe, **kw):
    real = n27.get_best_search_for_downstream
    n27.get_best_search_for_downstream = recipe
    try:
        return n27.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n27.get_best_search_for_downstream = real


CHEAP = dict(models=['rf'], base_reps={'rf': ['toy', 'second']}, shares=[0.25, 0.7, 0.9], repeats=2,
             levels=[0.5, 1.0], draws=[0], alphas=[0.0, 0.5, 1.0])


def test_draws():
    print("which molecules are kept:")
    a = n27.kept_rows(0.7, 3, 1, 100)
    b = n27.kept_rows(0.7, 3, 1, 100)
    c = n27.kept_rows(0.7, 4, 1, 100)
    check('the same (share, draw, fold) keeps the same molecules', np.array_equal(a, b))
    check('a different draw keeps different molecules', not np.array_equal(a, c))
    check('0.7 of 100 keeps 70', len(a) == 70, f'{len(a)} kept')
    check('no molecule kept twice', len(set(a.tolist())) == len(a))
    check('the default shares reach down to a quarter and a half (D3)',
          {0.25, 0.5} <= set(n27.SHARES) and 1.0 in n27.SHARES,
          str(n27.SHARES))
    q = n27.kept_rows(0.25, 3, 1, 100)
    check('0.25 of 100 keeps 25', len(q) == 25, f'{len(q)} kept')
    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "sys.path.insert(0, %r); "
            "import experiments.noise.n27_removal_cost_and_propagation as m; "
            "print(m.kept_rows(0.7, 3, 1, 100).tolist())") % (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                     '..', 'src'))
    p = subprocess.run([sys.executable, '-c', code], capture_output=True,
                       text=True)
    other = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''
    check('a second process keeps the same molecules',
          other == str(a.tolist()), p.stderr[-200:] if p.returncode else '')


def test_move_and_sign():
    print("moving labels, and the sign of a loss:")
    y = np.array([1.0, 2.0, 3.0])
    oof = np.array([2.0, 2.0, 2.0])
    check('alpha 0 moves nothing', np.allclose(n27.moved_toward(y, oof, 0.0), y))
    check('alpha 1 is the prediction',
          np.allclose(n27.moved_toward(y, oof, 1.0), oof))
    check('alpha 0.5 is half way',
          np.allclose(n27.moved_toward(y, oof, 0.5), [1.5, 2.0, 2.5]))
    check('rmse 0.5 -> 0.6 is a loss of +0.1',
          np.isclose(n27._loss(0.5, 0.6, True), 0.1))
    check('spearman 0.5 -> 0.6 is a loss of -0.1',
          np.isclose(n27._loss(0.5, 0.6, False), -0.1))


def test_run(domain, recipe, label):
    print(f"a full run, {label}:")
    data = make_data(domain=domain)
    res = run_it(data, 'toy', recipe=recipe, **CHEAP)
    s = res['summary']
    want_lib = domain == 'molecular'
    check(f'{label}: metric direction read off the data',
          res['lower_is_better'] is want_lib,
          f"{res['metric']}, lower_is_better={res['lower_is_better']}")

    spaces = res['spaces_run']
    if recipe is not no_recipe:
        check(f'{label}: the hybrid space ran', 'hybrid' in spaces, str(spaces))
    else:
        check(f'{label}: single representations only when the pool is empty',
              spaces == ['base_1', 'base_2'], str(spaces))
    check(f'{label}: which representations is recorded',
          res['base_reps_by_model'] == {'rf': ['toy', 'second']},
          str(res.get('base_reps_by_model')))

    ref = res['per_fold']['reference']
    for sp in spaces:
        check(f'{label}: {sp} has a correct-label fit on every fold',
              len(ref.get(sp, {}).get('rf', {})) == 3,
              str(sorted(ref.get(sp, {}).get('rf', {}))))

    # 4 and 6: every injected row pairs with its own fold, ratio = num / den
    ok_ratio, ok_fold, n_rows = True, True, 0
    for sp, by_m in s['propagation'].items():
        for m, blk in by_m.items():
            for lvl, r in blk['injected_by_level'].items():
                for row in r['per_pass']:
                    n_rows += 1
                    fk = 'f' + row['pass'].split('f')[-1]
                    raw = res['per_fold']['injected'][sp][lvl][m][row['pass']]
                    want = raw['mse'] - ref[sp][m][fk]['mse']
                    ok_fold &= np.isclose(row['held_out_mse_change'], want)
                    ok_ratio &= np.isclose(
                        row['ratio'], row['held_out_mse_change']
                        / row['training_label_variance_added'])
    check(f'{label}: an injected pass pairs with its own fold', ok_fold and n_rows > 0,
          f'{n_rows} rows')
    check(f'{label}: every ratio is its numerator over its denominator', ok_ratio)

    # 7: injected variance grows with the level
    blk = s['propagation']['base_1']['rf']['injected_by_level']
    v05 = blk['0.5']['mean_training_label_variance_added']
    v10 = blk['1']['mean_training_label_variance_added']
    check(f'{label}: more variance injected at level 1.0 than at 0.5',
          v10 > v05, f'{v05:.4f} then {v10:.4f} label units squared')

    # 3 in the run: alpha 0 adds no variance and scores the reference
    rel = s['propagation']['base_1']['rf']['relabelled_by_alpha']
    r0 = rel['0']
    check(f'{label}: alpha 0 adds zero variance',
          r0['mean_training_label_variance_added'] == 0.0)
    check(f'{label}: alpha 0 has an undefined ratio, not a number',
          all(not np.isfinite(r['ratio']) for r in r0['per_pass']))

    # 2 in the run: shares keep that share
    rc = res['per_fold']['removal_cost']['base_1']['0.7']['rf']['r0']
    kept = {v['training_molecules_kept'] / v['training_molecules']
            for v in rc.values()}
    check(f'{label}: 0.7 keeps 0.7 of each fold',
          all(abs(k - 0.7) < 0.01 for k in kept), str(sorted(kept)))
    rcq = res['per_fold']['removal_cost']['base_1']['0.25']['rf']['r0']
    kq = {v['training_molecules_kept'] / v['training_molecules']
          for v in rcq.values()}
    check(f'{label}: 0.25 keeps a quarter of each fold',
          all(abs(k - 0.25) < 0.01 for k in kq), str(sorted(kq)))
    row = s['removal_cost']['base_1']['0.7']['rf']
    check(f'{label}: one removal row per draw', row['draws_read'] == 2,
          f"{row['draws_read']} rows")
    check(f'{label}: the worst draw is no better than the best',
          row['worst_draw_mean_loss'] >= row['best_draw_mean_loss'])
    check(f'{label}: no errors recorded', not res.get('_errors'),
          str(res.get('_errors'))[:300])
    return res


def test_labels_untouched():
    print("the held-out labels:")
    data = make_data()
    before = data['labels'].copy()
    run_it(data, 'toy', **CHEAP)
    check('the label array is unchanged after a run',
          np.array_equal(before, data['labels']))


def test_refuses_classification():
    print("a class label:")
    data = make_data()
    data['labels'] = (data['labels'] > 0).astype(int)
    data['task_type'] = 'classification'
    res = run_it(data, 'toy', **CHEAP)
    check('a classification label is skipped with a reason',
          res.get('skipped') == 'not_regression', str(res)[:120])


def test_cli():
    print("the command line:")
    p = subprocess.run([sys.executable, n27.__file__, '--help'],
                       capture_output=True, text=True)
    for flag in ('--shares', '--repeats', '--levels', '--alphas', '--draws',
                 '--tag', '--parts'):
        check(f'{flag} is a flag', flag in p.stdout)
    check('--base-rep is gone', '--base-rep' not in p.stdout)


# ── the D-versus-D' rows and the checkpoint (2026-09-25) ────────────────────

def test_rows_and_checkpoint():
    """Plan section 2 on n27: the run writes D-versus-D' rows beside its
    result file, each with the metric direction and a gain signed by it, and
    leaves no per-pass checkpoint behind once the result is complete."""
    import json
    print("\nthe D-versus-D' rows and the checkpoint:")
    res = run_it(make_data(), 'toy_rows', models=['rf'], base_reps={'rf': ['toy', 'second']}, shares=[0.9], repeats=1, levels=[0.5], draws=[0], alphas=[0.5], parts=['removal_cost', 'relabelled'])
    path = res['rows']['rows_file']
    rows = ([json.loads(l) for l in open(path)]
            if path and os.path.exists(path) else [])
    check('rows were written to the rows file', len(rows) > 0, str(path))
    methods = {x['method'] for x in rows}
    check('the rows name the methods that change the training data',
          {'removal_cost', 'move_toward_oof'} <= methods, str(sorted(methods)))
    check('every row carries the direction the dataset\'s metric has',
          all(x['lower_is_better'] is bool(res['lower_is_better'])
              for x in rows), str(res['lower_is_better']))
    ok = all((x['gain'] != x['gain']) or abs(
        x['gain'] - ((x['score_D'] - x['score_Dprime'])
                     if x['lower_is_better']
                     else (x['score_Dprime'] - x['score_D']))) < 1e-12
             for x in rows)
    check('gain is score_D against score_Dprime, signed by the direction', ok)
    check('D is the correct-label fit, so score_D equals score_clean',
          all(x['score_D'] == x['score_clean'] for x in rows))
    ck = os.path.join(OUT, 'n27_removal_cost_and_propagation', '_checkpoints')
    check('no checkpoint is left once the result is complete',
          not (os.path.isdir(ck) and os.listdir(ck)), ck)


if __name__ == '__main__':
    test_draws()
    test_move_and_sign()
    test_run('molecular', no_recipe, 'rmse, single representations only')
    test_run('antibody', no_recipe, 'spearman, single representations only')
    test_run('molecular', fake_recipe(['toy', 'second']), 'rmse, hybrid')
    test_labels_untouched()
    test_refuses_classification()
    test_cli()
    test_rows_and_checkpoint()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
