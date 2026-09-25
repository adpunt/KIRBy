#!/usr/bin/env python3
"""Smoke for n36: which molecules to measure a second time.

What has to be true, on a toy dataset with known repeat noise:

  1. THE START AND THE SPARE ARE TWO DIFFERENT REAL MEASUREMENTS of the same
     molecule, and a molecule measured once has no spare.
  2. THE BUDGET HANDS OUT EXACTLY B SECOND MEASUREMENTS, to candidates only,
     and a chosen molecule's label becomes the mean of its two.
  3. EACH RULE RANKS BY WHAT IT SAYS: disagreement puts the largest
     |label - prediction| first; sparsest puts the most isolated molecule first;
     leverage counts reverse neighbours. The leverage rules give the same
     order as the old variance_x_leverage rules, whose extra factor was the
     same for every candidate.
  4. A RUN WRITES A PAIRED DIFFERENCE for every rule at every budget against
     every random draw, on the same (fold, seed) passes, plus both references
     and the tail scores.
  5. THE HYBRID SPACE IS REACHED when the pool returns a recipe.
  6. A dataset with no repeats is skipped; the command line parses.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# 2026-09-25: today's similarity (model_input) unless a check sets its own
# list, and the noise cache in a temporary directory, never the real one.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
import tempfile as _tf
os.environ['KIRBY_CACHE_DIR'] = _tf.mkdtemp(prefix='n36_smoke_cache_')

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import subprocess
import tempfile

import numpy as np

import experiments.noise.n36_remeasure_priority as n36
from _smoke_n26 import make_data, fake_recipe, no_recipe

OUT = tempfile.mkdtemp(prefix='n36_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def run_it(data, name, recipe=no_recipe, **kw):
    real = n36.get_best_search_for_downstream
    n36.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_reps', {
            m: ['toy', 'second']
            for m in (kw.get('models') or n36.DOWNSTREAM_MODELS)})
        return n36.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n36.get_best_search_for_downstream = real


def test_helpers():
    print("start, spare, budget and rules:")
    vals = [np.array([1.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    start, spare = n36.draw_start_and_spare(vals, seed=0)
    check('a molecule measured once has no spare', np.isnan(spare[0]))
    ok = all(start[j] in vals[j] and spare[j] in vals[j] and start[j] != spare[j]
             for j in (1, 2))
    check('start and spare are two different real measurements', ok,
          f'{start} {spare}')

    start_tr = np.array([0.0, 1.0, 2.0, 3.0])
    spare_tr = np.array([np.nan, 3.0, 4.0, 5.0])
    y = n36.labels_after_budget(start_tr, spare_tr, np.array([1, 3]))
    check('chosen molecules get the mean of two, others keep the start',
          np.allclose(y, [0.0, 2.0, 2.0, 4.0]), str(y))

    cand = np.array([1, 2, 3])
    dis = np.array([9.0, 0.1, 5.0, 0.3])
    r = n36.rank_candidates('disagreement', cand, disagreement=dis,
                            leverage=np.ones(4), sparsity=np.zeros(4),
                            random_seed=0)
    check('disagreement ranks the largest first', list(r) == [2, 3, 1], str(r))
    spars = np.array([0.0, 7.0, 1.0, 3.0])
    r = n36.rank_candidates('sparsest', cand, disagreement=dis,
                            leverage=np.ones(4), sparsity=spars,
                            random_seed=0)
    check('sparsest ranks the most isolated first', list(r) == [1, 3, 2], str(r))

    # Four points in a tight cluster and one far away: the far one is the
    # sparsest and feeds nobody's neighbourhood beyond the minimum.
    X = np.array([[0, 0], [0, .1], [.1, 0], [.1, .1], [10, 10]], dtype=float)
    lev, sp = n36.neighbour_stats(X, n_neighbours=2)
    check('the isolated molecule is the sparsest', int(np.argmax(sp)) == 4,
          str(sp))
    check('the isolated molecule has leverage 1 (no one\'s neighbour)',
          lev[4] == 1.0, str(lev))
    check('leverage counts every neighbour slot once',
          lev.sum() == 5 + 5 * 2, str(lev))

    check('the old rule names are gone',
          'variance_x_leverage' not in ' '.join(n36.RULES)
          and 'leverage' in n36.RULES
          and 'leverage_x_disagreement' in n36.RULES, str(n36.RULES))

    # The old rules, written out as they were before the rename, with the
    # constant factor s2e/1 - s2e/2 still in and the same seeded tie-break.
    def old_rank(rule, cand, dis, lev, s2e, seed):
        tie = np.random.default_rng(int(seed)).random(len(cand))
        score = (s2e / 1.0 - s2e / 2.0) * lev[cand]
        if rule == 'variance_x_leverage_x_disagreement':
            score = score * dis[cand]
        return cand[np.lexsort((tie, -score))]

    rng = np.random.default_rng(11)
    same = []
    for trial in range(200):
        n = 60
        cand = np.sort(rng.choice(n, size=35, replace=False))
        # Integer leverage like neighbour_stats gives, so ties are common.
        lev_t = 1.0 + rng.integers(0, 6, size=n).astype(float)
        dis_t = np.abs(rng.normal(size=n))
        s2e = float(rng.uniform(0.01, 3.0))
        for old, new in (('variance_x_leverage', 'leverage'),
                         ('variance_x_leverage_x_disagreement',
                          'leverage_x_disagreement')):
            a = old_rank(old, cand, dis_t, lev_t, s2e, trial)
            b = n36.rank_candidates(new, cand, disagreement=dis_t,
                                    leverage=lev_t, sparsity=np.zeros(n),
                                    random_seed=trial)
            same.append(np.array_equal(a, b))
    check('leverage and leverage_x_disagreement give the old order exactly, '
          '200 toy draws each', all(same),
          f'{sum(same)} of {len(same)} orders identical')


def test_run():
    print("a run, single representations only:")
    data = make_data(n_mol=180, seed=4)
    res = run_it(data, 'toy_base', models=['rf'], budget_shares=[0.25, 0.5],
                 seeds=[0, 1], random_draws=2, oof_splits=3)
    check('only base_1 and base_2 ran', res['spaces_run'] == ['base_1', 'base_2'],
          str(res['spaces_run']))
    check('the representations are named per model',
          res['base_reps_by_model'] == {'rf': ['toy', 'second']}
          and 'base_rep' not in res, str(res.get('base_reps_by_model')))
    check('base_2 is scored', bool(
        res['per_space']['base_2']['rf']['rule_vs_random'].get('0.5')))
    row = res['per_space']['base_1']['rf']
    passes = set(row['scores']['zero_budget'])
    check('3 folds x 2 seeds = 6 passes', len(passes) == 6, str(sorted(passes)))
    check('all-measurements reference paired on every pass',
          len(row['all_measurements_vs_zero_budget']) == 6)
    for b in ('0.25', '0.5'):
        for rule in n36.RULES:
            if rule == 'random':
                continue
            per = row['rule_vs_random'].get(b, {}).get(rule, {})
            check(f'{rule} at budget {b}: paired vs both random draws on 6 '
                  f'passes', len(per) == 2 and all(len(v) == 6
                                                    for v in per.values()),
                  str({k: len(v) for k, v in per.items()}))
    one = row['scores']['zero_budget'][sorted(passes)[0]]
    check('tail scores are written on a pass',
          'tail_rmse_top' in one and 'calibration_slope' in one, str(one))
    # Zero budget and all measurements must differ, or the toy has no repeats
    # to hand out and every comparison is empty.
    am = list(row['all_measurements_vs_zero_budget'].values())
    check('the two references differ', any(abs(v) > 0 for v in am), str(am))
    check('the result file was written',
          os.path.exists(os.path.join(OUT, 'n36_remeasure_priority',
                                      'toy_base.json')))

    print("a run with a hybrid recipe:")
    res = run_it(data, 'toy_hybrid', recipe=fake_recipe(['toy', 'second']),
                 models=['rf', 'lightgbm'], budget_shares=[0.5], seeds=[0],
                 random_draws=1, oof_splits=3)
    check('the hybrid space ran', 'hybrid' in res['spaces_run'],
          str(res['spaces_run']))
    for m in ('rf', 'lightgbm'):
        got = res['per_space'].get('hybrid', {}).get(m, {})
        check(f'{m} has paired rule scores in the hybrid',
              bool(got.get('rule_vs_random', {}).get('0.5')))
    check('no errors recorded', '_errors' not in res, str(res.get('_errors')))

    print("two models with different rank-1 representations:")
    res = run_it(data, 'toy_split', models=['rf', 'lightgbm'],
                 base_reps={'rf': ['toy', 'second'],
                            'lightgbm': ['second', 'toy']},
                 budget_shares=[0.5], seeds=[0], random_draws=1, oof_splits=3)
    check('each model is scored in base_1 and base_2',
          all(res['per_space'][s].get(m, {}).get('rule_vs_random', {}).get('0.5')
              for s in ('base_1', 'base_2') for m in ('rf', 'lightgbm')))
    check('no errors recorded', '_errors' not in res, str(res.get('_errors')))

    print("a dataset with no repeats:")
    flat = make_data(seed=3)
    flat.pop('replicate_groups')
    res = run_it(flat, 'toy_flat', models=['rf'])
    check('skipped with the reason named',
          res.get('skipped') == 'no_replicate_structure', str(res))


def test_help():
    print("command line:")
    p = subprocess.run([sys.executable, n36.__file__, '--help'],
                       capture_output=True, text=True)
    check('--help exits 0', p.returncode == 0, p.stderr[-300:])
    for flag in ('--budget-shares', '--seeds', '--random-draws',
                 '--oof-splits', '--tag'):
        check(f'{flag} is offered', flag in p.stdout)
    check('--base-rep is gone', '--base-rep' not in p.stdout)



def _rows_of(res):
    """The D-versus-D' rows a run wrote (2026-09-25)."""
    import json
    p = (res.get('d_vs_dprime') or {}).get('rows_file')
    if not p or not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def _with_settings(names, fn):
    """Run fn() under KIRBY_SIMILARITY_SETTINGS=names, then restore it."""
    old = os.environ.get('KIRBY_SIMILARITY_SETTINGS')
    os.environ['KIRBY_SIMILARITY_SETTINGS'] = names
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop('KIRBY_SIMILARITY_SETTINGS', None)
        else:
            os.environ['KIRBY_SIMILARITY_SETTINGS'] = old



def _close(a, b, rtol=1e-12):
    """a == b, floats to rtol. rf's default factory predicts with n_jobs=-1,
    and its threads sum the trees in no fixed order, so two fits of the same
    forest can differ in the last bit (measured 2026-09-25: 1.402832974397575
    against 1.4028329743975747). That is not a change of number."""
    import math
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rtol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (a == b or (math.isnan(a) and math.isnan(b))
                or math.isclose(a, b, rel_tol=rtol, abs_tol=0.0))
    return a == b

def test_rows_and_resume():
    """2026-09-25: the D-versus-D' rows (or, for a measurement-only experiment,
    the per-pass checkpoint), a rerun that resumes every pass, and -- where
    the experiment finds neighbours -- the similarity sweep."""
    print("\nrows, resume and the similarity sweep:")
    from experiments.noise.d_vs_dprime import ROW_FIELDS
    kw = dict(models=['rf', 'svm'], budget_shares=[0.5], seeds=[0], random_draws=1, oof_splits=3)
    res = run_it(make_data(n_mol=120, seed=4), 'toy_rows', **dict(kw))
    rows = _rows_of(res)
    check('R1. D-vs-D\' rows are written, each with exactly the d_vs_dprime '
          'columns, D and D\' both scored', bool(rows) and all(
              set(r) == set(ROW_FIELDS) and r['score_D'] is not None
              for r in rows) and {'rf', 'svm'} <= {r['model'] for r in rows},
          f'{len(rows)} rows, methods {sorted({r["method"] for r in rows})[:4]}')
    again = run_it(make_data(n_mol=120, seed=4), 'toy_rows', **dict(kw))
    got = again['d_vs_dprime']
    check('R2. a rerun resumes every pass from the checkpoint and returns the '
          'same scores', got['passes_resumed_from_checkpoint'] > 0
          and got['rows_written_this_run'] == 0
          and again['per_space'] == res['per_space'],
          f"{got['passes_resumed_from_checkpoint']} passes resumed")
    # The same dataset name: the injected noise is seeded on it, so another
    # name would compare two different draws. The settings are in the
    # checkpoint signature, so this run recomputes rather than resumes.
    names = 'model_input,base_2|none|manhattan,forest_proximity|none|forest'
    r_sw = _with_settings(names, lambda: run_it(make_data(n_mol=120, seed=4),
                                                'toy_rows', **dict(kw)))
    row = r_sw['per_space']['base_1']['rf']['scores']
    base = res['per_space']['base_1']['rf']['scores']
    new = [a for a in row if '@base_2|none|manhattan|k5' in a]
    check('R3. every neighbour rule is also scored under each other setting '
          'and k, and model_input\'s own columns are unchanged',
          bool(new) and any('@forest_proximity|none|forest|k20' in a for a in row)
          and all(_close(row[a], base[a]) for a in base), f'{len(row)} columns')


if __name__ == '__main__':
    test_helpers()
    test_run()
    test_rows_and_resume()
    test_help()
    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} — {', '.join(FAIL)}")
        sys.exit(1)
    print("ALL GREEN")
