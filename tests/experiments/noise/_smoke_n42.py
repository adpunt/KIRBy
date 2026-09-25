#!/usr/bin/env python3
"""Smoke for n42: every mitigation method x similarity setting x model input.

WRITTEN 2026-09-25, NOT RUN. Run it on the server (srun), not on the owner's
laptop:
    srun --clusters=htc -p interactive --cpus-per-task=4 --mem=16G \
        --time=02:00:00 python tests/experiments/noise/_smoke_n42.py

Toy data, 150 molecules, two representations, a fixed recipe; the pool and the
single-representation ranking are replaced so nothing reads the precompute.

What has to be true before any row n42 writes can be read:

  1. REGISTERED: 'n42' is in SINGLE_DATASET_EXPERIMENTS before 'n06', in both
     import maps and in HONOURS_NOISE_CONDITION, and imports.
  2. A RUN WRITES ROWS for every family asked, under all three model inputs
     (rf:base_1, rf:base_2, rf:hybrid), and every row carries the metric's own
     lower_is_better.
  3. THE SIMILARITY AXIS: rows of a method that reads neighbours carry each
     asked setting and a k; rows of one that does not carry 'none'.
  4. MODEL_INPUT AT k=9 IS TODAY'S label_smooth: the D' n07 builds from the
     MODEL_INPUT lists at k=9 equals the one it builds with no lists, bit for
     bit, on the same scores.
  5. RESUME: a second identical run computes no unit and writes no row.
  6. CONTROLS: under KIRBY_NOISE_CONTROLS=1 random and shuffled twins appear.
  7. A CLASSIFICATION SET skips the six residual families and says why.
  8. families_asked refuses an unknown name.
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Today's similarity plus one swept setting, unless the caller sets a list;
# the noise cache in a temporary directory, never the real one.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS',
                      'model_input,base_1|none|manhattan')
os.environ['KIRBY_CACHE_DIR'] = tempfile.mkdtemp(prefix='n42_smoke_cache_')
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.pop('KIRBY_NOISE_CONTROLS', None)

import numpy as np

import experiments.noise.n42_similarity_noise_sweep as n42
import experiments.noise.noise_helpers as NH
from experiments.noise import n07_remediation as n07
from experiments.noise import similarity_space as ss
from experiments.noise import noise_cache as nc
from kirby.find_best_hybrid import Blueprint

FAIL = []
# Families for the full-run checks. Everything by default; a narrower list
# through KIRBY_N42_FAMILIES keeps a first smoke short.
FAMS = n42.families_asked()


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=150, n_feat=6, seed=0, clf=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    if clf:
        y = (y > np.median(y)).astype(int)
    return {'labels': y, 'reps': {'toy': X, 'second': X2},
            'groups': np.arange(n) // 3,
            'task_type': 'classification' if clf else 'regression',
            'domain': 'molecular', 'augmentations': None}


def fake_recipe(clf=False):
    bp = Blueprint(reps=['toy', 'second'], allocation_method='fixed',
                   importance_method='lightgbm_gain', budget=6,
                   internal_evaluator='lightgbm',
                   metric='auc' if clf else 'rmse',
                   lower_is_better=not clf,
                   task_type='classification' if clf else 'regression',
                   augmentation_strategy='none')
    meta = {'best_config': {'blueprint': bp}, 'objective': 'predictive',
            'objective_choice': 'smoke_fixed_recipe',
            'pool_subset': 'predictive_and_similarity', 'tier': 'general',
            'aug_condition': 'none', 'by_objective': {}}
    return lambda *a, **k: (None, meta)


def run_it(data, out, name, condition='outlier_p10', clf=False, **kw):
    real_pool = n42.get_best_search_for_downstream
    real_base = n42.noise_base_reps_by_model
    n42.get_best_search_for_downstream = fake_recipe(clf)
    n42.noise_base_reps_by_model = (
        lambda searches, models, **k: ({m: ['toy', 'second'] for m in models},
                                       {}))
    try:
        return n42._run_one_condition(
            data, {'n_splits': 3, 'models': ['rf']}, out, name,
            noise_condition=condition, **kw)
    finally:
        n42.get_best_search_for_downstream = real_pool
        n42.noise_base_reps_by_model = real_base


def read_rows(out, name, condition):
    path = n07.rows_path(out, n42.EXP_ID, name, condition)
    if not os.path.exists(path):
        return path, []
    with open(path) as fh:
        return path, [json.loads(l) for l in fh if l.strip()]


def test_registered():
    print("registration:")
    L = NH.SINGLE_DATASET_EXPERIMENTS
    check('1. n42 listed before n06',
          'n42' in L and L.index('n42') < L.index('n06'))
    check('1b. n42 in HONOURS_NOISE_CONDITION',
          'n42' in NH.HONOURS_NOISE_CONDITION)
    check('1c. both import maps name it',
          NH._experiment_full_name('n42') == n42.EXP_ID
          and NH._import_experiment('n42') is n42)


def test_run_and_resume():
    print("\na full run on toy regression data:")
    out = tempfile.mkdtemp(prefix='n42_smoke_')
    data = make_data()
    res = run_it(data, out, 'toy', families=list(FAMS))
    path, rows = read_rows(out, 'toy', 'outlier_p10')
    check('2. rows written', len(rows) > 0, f'{len(rows)} rows -> {path}')
    fams = {r['method'].split('/')[0] for r in rows}
    check('2b. every asked family wrote rows', set(FAMS) <= fams,
          f'missing {sorted(set(FAMS) - fams)}; errors {res.get("errors")}')
    mis = {r['model_input'] for r in rows}
    check('2c. all three model inputs',
          {'rf:base_1', 'rf:base_2', 'rf:hybrid'} <= mis, f'{sorted(mis)}')
    check('2d. every row carries lower_is_better=True (rmse)',
          all(r['lower_is_better'] is True for r in rows))
    check('2e. rf and svm are fitted on every D-prime',
          {'rf', 'svm'} <= {r['model'] for r in rows})

    print("\nthe similarity axis:")
    want = {s.name for s in n07.sweep_settings()}
    sims = {r['similarity_setting'] for r in rows}
    check('3. swept settings appear', want <= sims | {ss.MODEL_INPUT.name},
          f'{sorted(sims)}')
    ls = [r for r in rows if r['method'] == 'n07/label_smooth']
    check('3b. label_smooth rows carry a setting and a k',
          bool(ls) and all(r['similarity_setting'] != n07.NO_NEIGHBOURS
                           and str(r['method_setting']).startswith('k=')
                           for r in ls))
    rm = [r for r in rows if r['method'] == 'n07/remove'
          and 'detector=cross_val' in str(r['method_setting'])]
    check('3c. remove under cross_val carries similarity none',
          bool(rm) and all(r['similarity_setting'] == n07.NO_NEIGHBOURS
                           for r in rm))
    check('3d. n25 exemption arms are written',
          any('arm=cliff_exempt' in str(r['method_setting']) for r in rows)
          or res.get('n25_cliff_reference_error') is not None,
          f"cliff reference error: {res.get('n25_cliff_reference_error')}")

    print("\nresume:")
    res2 = run_it(data, out, 'toy', families=list(FAMS))
    _, rows2 = read_rows(out, 'toy', 'outlier_p10')
    check('5. second run computes no unit',
          res2['units_computed_this_run'] == 0,
          f"{res2['units_computed_this_run']} computed, "
          f"{res2['units_resumed']} resumed")
    check('5b. and writes no row', len(rows2) == len(rows),
          f'{len(rows)} -> {len(rows2)}')


def test_model_input_is_today():
    print("\nMODEL_INPUT at k=9 is today's label_smooth:")
    data = make_data()
    X = data['reps']['toy'][:100]
    y = np.asarray(data['labels'][:100], dtype=float)
    rng = np.random.default_rng(1)
    scores = rng.uniform(size=100)
    L = ss.neighbour_lists(ss.MODEL_INPUT, spaces_train={'model_input': X},
                           spaces_val=None, y_train=y)
    for st in ('label_smooth', 'knn_smooth'):
        a = n07.cleaned_training_set(st, X, y, scores, 0.1)
        b = n07.cleaned_training_set(st, X, y, scores, 0.1,
                                     similarity_neighbours=L, similarity_k=9)
        check(f'4. {st}: lists at k=9 give the same labels bit for bit',
              a is not None and b is not None
              and np.array_equal(np.asarray(a['y']), np.asarray(b['y'])))


def test_controls():
    print("\ncontrols:")
    out = tempfile.mkdtemp(prefix='n42_smoke_ctl_')
    os.environ['KIRBY_NOISE_CONTROLS'] = '1'
    try:
        run_it(make_data(seed=2), out, 'toyc', families=['n07', 'n31'])
    finally:
        os.environ.pop('KIRBY_NOISE_CONTROLS', None)
    _, rows = read_rows(out, 'toyc', 'outlier_p10')
    meth = {r['method'] for r in rows}
    check('6. random and shuffled twins written',
          any(m.endswith('__random_twin') for m in meth)
          and any(m.endswith('__shuffled_twin') for m in meth))


def test_classification():
    print("\nclassification:")
    out = tempfile.mkdtemp(prefix='n42_smoke_clf_')
    data = make_data(clf=True, seed=3)
    res = run_it(data, out, 'toyclf', condition='uniform', clf=True,
                 families=list(FAMS))
    skipped = set(res.get('families_skipped') or {})
    check('7. the residual families are skipped on a class label',
          (set(FAMS) & set(n42.REGRESSION_ONLY_FAMILIES)) <= skipped,
          f'{sorted(skipped)}')
    _, rows = read_rows(out, 'toyclf', 'uniform')
    check('7b. rows carry lower_is_better=False (auc)',
          bool(rows) and all(r['lower_is_better'] is False for r in rows),
          f'{len(rows)} rows')


def test_guard():
    print("\nguards:")
    try:
        n42.families_asked({'families': 'n07,nope'})
        ok = False
    except ValueError:
        ok = True
    check('8. an unknown family is refused', ok)


if __name__ == '__main__':
    print(f"families: {list(FAMS)}; settings: "
          f"{[s.name for s in n07.sweep_settings()]}")
    test_registered()
    test_model_input_is_today()
    test_guard()
    test_run_and_resume()
    test_controls()
    test_classification()
    print(f"\ncache: {nc.report()}")
    if FAIL:
        print(f"\n{len(FAIL)} FAILED: {FAIL}")
        sys.exit(1)
    print("\nALL PASSED")
