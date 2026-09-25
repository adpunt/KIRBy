#!/usr/bin/env python3
"""Smoke for n18: cleaning the labels as they come.

The thing that has to be true: on data where some labels really ARE wrong, the
cleanups win folds; on data where none are, they do not. If a smoke cannot tell
those two apart, the experiment cannot either.

  1. nothing is injected — the uncleaned model trains on the labels handed in
  2. every cleanup is compared against the SAME uncleaned score per fold
  3. on labels with real errors in them, at least one cleanup wins most folds
  4. on clean labels, no cleanup sweeps every fold at every setting
  5. the headline is a fold count, and it never exceeds the folds that ran

CHECKED IN BOTH FEATURE SPACES since 2026-09-17. The wrong-label set and the
clean set are each run in the base representation and in the hybrid, and the
separation between them has to hold in both. Before that there was no hybrid
space, so a cleanup that only works in one fixed representation would have
passed this file.

Two more, added with it:

  * the labels handed to the hybrid build are the fold's training labels, as the
    loader produced them. No cleanup reaches the feature ranking or the budget
    split, and neither does a held-out label.
  * a run where no recipe comes back still produces every base-space number
    rather than being dropped.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# The hybrid build ranks features with LightGBM, which splits its work across
# features, and this smoke's molecules carry three. One thread is far faster
# here than every core; the switch is the repository's own, read by
# kirby/hybrid.py:78.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')

import tempfile

import numpy as np

import experiments.noise.n18_clean_the_real_labels as n18
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n18_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=600, n_feat=3, wrong_fraction=0.0, seed=0):
    """A dataset whose labels carry errors already, before anything touches them.

    wrong_fraction of the molecules get a label from a different molecule, which
    is what a transcription error looks like. Nothing is injected at run time --
    the errors are in the labels the loader hands over.

    TWO representations, because a hybrid needs something to split a budget
    across. `toy` carries the features the label was built from. `second` carries
    the same features in a different order with noise added, so it predicts too,
    but worse.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    w = rng.normal(size=n_feat)
    y = X @ w + rng.normal(0, 0.1, n)
    if wrong_fraction > 0:
        k = int(n * wrong_fraction)
        hit = rng.choice(n, size=k, replace=False)
        y[hit] = y[rng.choice(n, size=k, replace=False)]
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {
        'labels': y,
        'reps': {'toy': X, 'second': X2},
        'groups': np.arange(n),
        'task_type': 'regression',
        'metric': 'rmse',
    }


# ── A recipe, so the hybrid space exists without a precomputed pool ──────────
# The pool is built by a real precompute on a real dataset and there is none
# here, so the lookup is replaced with a fixed recipe. 'fixed' allocation splits
# the budget evenly across the two representations, which is what makes the
# hybrid a genuinely different feature space from either of them.

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
    """What the pool returns on a dataset it was never built for."""
    return None, None


def run_it(data, name, captured_labels=None, **kw):
    """One n18 run with a fixed recipe standing in for the precomputed pool."""
    real_lookup = n18.get_best_search_for_downstream
    real_build = n18.hybrid_for_fold
    n18.get_best_search_for_downstream = fake_recipe(sorted(data['reps']))
    if captured_labels is not None:
        def spy(train_reps, val_reps, y_train, bp, groups_train, model,
                metric_fn, lower_is_better):
            captured_labels.append(np.asarray(y_train, dtype=float).copy())
            return real_build(train_reps, val_reps, y_train, bp, groups_train,
                              model, metric_fn, lower_is_better)
        n18.hybrid_for_fold = spy
    try:
        return n18.run(data, {'n_splits': 4}, OUT, name,
                       models=['rf'], detectors=['neighbor_residual'],
                       base_rep='toy', **kw)
    finally:
        n18.get_best_search_for_downstream = real_lookup
        n18.hybrid_for_fold = real_build


def test_nothing_injected():
    print("nothing is injected:")
    data = make_data(wrong_fraction=0.0, seed=1)
    seen = []
    r = run_it(data, 'toy_clean', captured_labels=seen)
    check('the run records that it injected nothing',
          r['noise_injection'] == 'none' and
          r['noise_regime'] == 'whatever_the_assay_carries')
    check('it says the recipe never saw a clean label column',
          r['recipe_saw_clean_labels'] is False)
    check('it says no cleanup reached the feature choice',
          r['hybrid_sees_cleaned_labels'] is False
          and r['weights_reach_feature_selection'] is False)
    check('both spaces ran, and the base representation is named',
          r['hybrid_available'] is True
          and r['spaces_run'] == ['base', 'hybrid']
          and r['base_rep'] == 'toy',
          f"{r['spaces_run']}, base {r['base_rep']}")

    for space in r['spaces_run']:
        blk = r['per_space'][space]['per_detector']['neighbor_residual']
        rates = sorted(blk)
        base = [blk[x]['per_model']['rf']['no_cleaning_mean'] for x in rates]
        check(f'{space}: the uncleaned score is the same at every flag rate',
              max(base) - min(base) < 1e-12,
              f'{[round(b, 6) for b in base]}')
    # Read the rate off the result rather than naming one. The grid moved from
    # (0.05, 0.10, 0.20) to (0.01, 0.05, 0.20) on 2026-09-18, and a smoke that
    # hardcodes a value fails on the grid rather than on the behaviour.
    _rate = str(r['flag_rates'][0])
    check('the two spaces are genuinely different feature spaces',
          abs(r['per_space']['base']['per_detector']['neighbor_residual']
              [_rate]['per_model']['rf']['no_cleaning_mean']
              - r['per_space']['hybrid']['per_detector']['neighbor_residual']
              [_rate]['per_model']['rf']['no_cleaning_mean']) > 1e-9)

    # What the feature ranking was handed must be this fold's training labels,
    # as the loader produced them.
    seen_all = np.concatenate(seen) if seen else np.array([])
    check('the hybrid was built at all', seen_all.size > 0,
          f'{len(seen)} builds, {seen_all.size} labels')
    check('every label the hybrid saw is one of the labels the loader handed in',
          seen_all.size > 0 and bool(np.all(np.isin(seen_all, data['labels']))),
          f'{int((~np.isin(seen_all, data["labels"])).sum())} of '
          f'{seen_all.size} were not')
    n_train = len(data['labels']) - len(data['labels']) // 4
    check('each build saw one fold\'s training molecules, not the whole dataset',
          all(abs(len(s) - n_train) <= 2 for s in seen),
          f'{[len(s) for s in seen]} against about {n_train}')
    return r


def test_finds_a_real_error(clean_result):
    print("\nlabels that really are wrong:")
    data = make_data(wrong_fraction=0.20, seed=2)
    r = run_it(data, 'toy_wrong')
    check('it ran in both spaces', r['spaces_run'] == ['base', 'hybrid'],
          str(r['spaces_run']))

    for space in r['spaces_run']:
        s = r['summary']['per_space'][space]['per_cleanup']
        any_majority = {c: v.get('settings_winning_most_folds', 0)
                        for c, v in s.items()}
        check(f'{space}: at least one cleanup wins most folds somewhere',
              sum(any_majority.values()) > 0, str(any_majority))
        best = max((v['best_setting']['folds_won']
                    / max(v['best_setting']['folds_scored'], 1)
                    for v in s.values() if v.get('settings_scored')), default=0)
        check(f'{space}: the best cleanup wins at least half its folds',
              best >= 0.5, f'best fold win rate {best:.2f}')

    # The separation. The control must be quieter than the dirty set, in BOTH
    # spaces. This is what the file exists to check, and until 2026-09-17 it was
    # only ever checked in one fixed representation.
    print("\nthe separation, in both spaces:")
    for space in r['spaces_run']:
        s = r['summary']['per_space'][space]['per_cleanup']
        cs = clean_result['summary']['per_space'][space]['per_cleanup']
        dirty = sum(v.get('settings_winning_every_fold', 0) for v in s.values())
        clean = sum(v.get('settings_winning_every_fold', 0) for v in cs.values())
        check(f'{space}: cleaning sweeps more settings when the labels really '
              f'are wrong',
              dirty >= clean,
              f'wrong-label sweeps {dirty}, clean-label sweeps {clean}')

    check('the sweep counts sit side by side in the result file',
          set(r['summary']['sweeps_by_space']) == set(r['cleanups']),
          str(sorted(r['summary']['sweeps_by_space'])))
    return r


def test_counts_are_sane(r):
    print("\nthe counts:")
    ok_bound, ok_paired, seen = True, True, 0
    for space, sblk in r['per_space'].items():
        for det, by_rate in sblk['per_detector'].items():
            for rate, blk in by_rate.items():
                n_ran = blk['n_folds_ran']
                for m, row in blk['per_model'].items():
                    for c in r['cleanups']:
                        # Absent, not zero. A cleanup that reads neither swept
                        # axis is scored at the first slot only, so it has no
                        # row at the others -- quantile_clip is the only one.
                        cell = row.get(c)
                        if cell is None:
                            continue
                        seen += 1
                        if cell['folds_won'] > cell['folds_scored']:
                            ok_bound = False
                        if cell['folds_scored'] > n_ran:
                            ok_paired = False
                        if len(cell['fold_gains']) != cell['folds_scored']:
                            ok_paired = False
    check('folds won never exceeds folds scored', ok_bound, f'{seen} readings')
    check('folds scored never exceeds folds run, and the gains line up',
          ok_paired)
    check('every cleanup carries a spread over folds beside its mean',
          all(np.isfinite(row[c]['spread_over_folds'])
              for sblk in r['per_space'].values()
              for by_rate in sblk['per_detector'].values()
              for blk in by_rate.values()
              for row in blk['per_model'].values()
              for c in r['cleanups']
              if c in row and row[c]['folds_scored'] > 1))


def test_shuffled_control(dirty, clean):
    """The detector's own ranking must beat the same scores dealt at random
    where the labels really are wrong, and must not where none are."""
    print("\nthe shuffled-score control (2026-09-25):")
    for space in dirty['spaces_run']:
        ctrl = dirty['per_space'][space].get('shuffled_control')
        check(f'{space}: the control block is written', bool(ctrl))
        if not ctrl:
            continue
        rows = [by_seed
                for by_rate in ctrl.values()
                for by_model in by_rate.values()
                for by_c in by_model.values()
                for c, by_seed in by_c.items()]
        check(f'{space}: every seed is its own row',
              all(set(bs) == {str(s) for s in dirty['shuffle_seeds']}
                  for bs in rows), f'{len(rows)} settings')
        check(f'{space}: score-free cleanups are not re-run',
              not any(c in n18.SCORE_FREE_CLEANUPS
                      for by_rate in ctrl.values()
                      for by_model in by_rate.values()
                      for by_c in by_model.values() for c in by_c))
        # remove at the highest flag rate: 20% wrong labels, 20% removed. A
        # detector that finds them has to beat removing the same count at random.
        rate = str(max(dirty['flag_rates']))
        rm = ctrl['neighbor_residual'][rate]['rf'].get('remove', {})
        beat = [row['real_minus_shuffled'] for row in rm.values()]
        check(f'{space}: on wrong labels, remove beats random removal for '
              f'every seed', bool(beat) and all(b > 0 for b in beat),
              str([round(b, 4) for b in beat]))
        s_d = dirty['summary']['shuffled_control']['per_space'][space]
        s_c = clean['summary']['shuffled_control']['per_space'][space]
        d_beat = sum(v['settings_where_real_beat_shuffled_on_most_folds']
                     for by_seed in s_d.values() for v in by_seed.values())
        c_beat = sum(v['settings_where_real_beat_shuffled_on_most_folds']
                     for by_seed in s_c.values() for v in by_seed.values())
        check(f'{space}: the real ranking beats random more often on wrong '
              f'labels than on clean ones', d_beat > c_beat,
              f'wrong labels {d_beat}, clean labels {c_beat}')

    r = run_it(make_data(n=200, wrong_fraction=0.0, seed=4), 'toy_noctrl',
               flag_rates=[0.10], shuffle_seeds=[])
    check('an empty seed list switches the control off',
          all(not any(by_rate[str(0.1)]['rf']
                      for by_rate in [v for v in sp['shuffled_control'].values()])
              for sp in r['per_space'].values()))


def test_qm9_is_marked():
    print("\nthe control:")
    data = make_data(n=200, wrong_fraction=0.0, seed=3)
    r = run_it(data, 'qm9', flag_rates=[0.10])
    check('qm9 is marked as the control in the result file',
          r['is_computed_label_control'] is True)
    data2 = make_data(n=200, seed=3)
    r2 = run_it(data2, 'logd', flag_rates=[0.10])
    check('a measured dataset is not marked as the control',
          r2['is_computed_label_control'] is False)


def test_runs_without_a_recipe():
    print("\nno recipe from the pool:")
    data = make_data(n=400, wrong_fraction=0.20, seed=4)
    real = n18.get_best_search_for_downstream
    n18.get_best_search_for_downstream = no_recipe
    try:
        r = n18.run(data, {'n_splits': 3}, OUT, 'toy_no_recipe',
                    models=['rf'], detectors=['neighbor_residual'],
                    base_rep='toy', flag_rates=[0.10])
    finally:
        n18.get_best_search_for_downstream = real

    check('the run is not dropped', 'summary' in r)
    check('it records that no hybrid was available, and why',
          r['hybrid_available'] is False
          and 'precomputed pool' in r['hybrid_unavailable_reason'])
    check('the base representation still ran', r['spaces_run'] == ['base'],
          str(r['spaces_run']))
    check('no hybrid row was written',
          'hybrid' not in r['per_space']
          and 'hybrid' not in r['summary']['per_space'])

    blk = (r['per_space']['base']['per_detector']['neighbor_residual']
           [str(r['flag_rates'][0])])
    row = blk['per_model']['rf']
    # `cleanups` is what this task type ran, which is seven on a regression set
    # since 2026-09-18 and was four before. A cleanup that reads neither swept
    # axis is scored at the first slot only, and this IS that slot.
    check('every cleanup still carries a fold count in the base space',
          all(row[c]['folds_scored'] == blk['n_folds_ran']
              for c in r['cleanups']),
          str({c: row[c]['folds_scored'] for c in r['cleanups']}))
    check('every cleanup still carries a finite mean gain',
          all(np.isfinite(row[c]['mean_gain']) for c in r['cleanups']),
          str({c: round(row[c]['mean_gain'], 4) for c in r['cleanups']}))
    check('the uncleaned score is still there to beat',
          np.isfinite(row['no_cleaning_mean']),
          f"{row['no_cleaning_mean']:.4f}")
    check('the summary still counts settings in the base space',
          all(v.get('settings_scored', 0) > 0
              for v in r['summary']['per_space']['base']['per_cleanup'].values()),
          str({c: v.get('settings_scored')
               for c, v in r['summary']['per_space']['base']['per_cleanup'].items()}))


def test_one_build_per_recipe_not_per_model():
    """Which downstream models may share one built feature matrix.

    n18 looks the recipe up per downstream model, so five models mean five
    builds per fold per space. Under a recipe from the similarity family that is
    waste -- no downstream model is fitted inside the build -- so two models
    holding the same such recipe share one. Sharing wrongly is the same act as
    handing a model another model's features, which is the defect the per-model
    lookup exists to remove, so every way the comparison can fail to give a
    clean answer has to land on building twice.

    NOTHING ELSE HERE REACHES THIS. The fixture's pool holds no
    similarity-objective search -- the run above prints that it could not build
    that space for any model -- so the smoke runs green through the predictive
    path without entering the grouping at all. These recipes are made up for
    that reason.
    """
    from experiments.noise.noise_helpers import group_models_by_recipe
    models = ['rf', 'xgboost', 'svm', 'dnn', 'lightgbm']

    class Geo:
        objective = 'similarity'
        def __init__(self, tag): self.tag = tag
        def __eq__(self, o): return isinstance(o, Geo) and o.tag == self.tag

    class Pred:
        objective = 'predictive'
        def __init__(self, tag): self.tag = tag
        def __eq__(self, o): return isinstance(o, Pred) and o.tag == self.tag

    class Raises:
        objective = 'similarity'
        def __eq__(self, o): raise RuntimeError('this comparison raises')

    class ReturnsAnArray:
        objective = 'similarity'
        def __eq__(self, o): return np.array([True, False])

    one = Geo('a')
    g = group_models_by_recipe({m: one for m in models}, models)
    check('five models sharing one geometry recipe object build it once',
          [(len(x[1]), x[2]) for x in g] == [(5, True)], str(len(g)) + ' build(s)')

    g = group_models_by_recipe({m: Geo('a') for m in models}, models)
    check('five equal but separate geometry recipes build it once',
          [(len(x[1]), x[2]) for x in g] == [(5, True)], str(len(g)) + ' build(s)')

    g = group_models_by_recipe(
        {'rf': Geo('a'), 'xgboost': Geo('a'), 'svm': Geo('b'),
         'dnn': Geo('b'), 'lightgbm': Geo('c')}, models)
    check('three distinct geometry recipes build three matrices',
          [sorted(x[1]) for x in g] == [['rf', 'xgboost'], ['dnn', 'svm'],
                                        ['lightgbm']],
          str([sorted(x[1]) for x in g]))

    g = group_models_by_recipe({m: Pred('a') for m in models}, models)
    check('equal PREDICTIVE recipes are never shared, the model is fitted inside',
          [(len(x[1]), x[2]) for x in g] == [(1, False)] * 5,
          str(len(g)) + ' build(s)')

    for cls, why in ((Raises, 'raises'), (ReturnsAnArray, 'returns an array')):
        g = group_models_by_recipe({m: cls() for m in models}, models)
        check(f'a recipe whose comparison {why} is built per model',
              [len(x[1]) for x in g] == [1] * 5, str(len(g)) + ' build(s)')

    g = group_models_by_recipe({'rf': Geo('a'), 'svm': None, 'dnn': Geo('a')},
                               models)
    check('a model with no recipe gets no build, not another model\'s',
          [sorted(x[1]) for x in g] == [['dnn', 'rf']],
          str([sorted(x[1]) for x in g]))

    check('no recipes at all builds nothing',
          group_models_by_recipe({}, models) == [], '0 build(s)')

    g = group_models_by_recipe(
        {'rf': Geo('a'), 'xgboost': Pred('a'), 'svm': Geo('a')}, models)
    check('a predictive recipe never joins a geometry group',
          [(sorted(x[1]), x[2]) for x in g] == [(['rf', 'svm'], True),
                                                (['xgboost'], False)],
          str([(sorted(x[1]), x[2]) for x in g]))


if __name__ == '__main__':
    clean = test_nothing_injected()
    dirty = test_finds_a_real_error(clean)
    test_counts_are_sane(dirty)
    test_shuffled_control(dirty, clean)
    test_qm9_is_marked()
    test_runs_without_a_recipe()
    test_one_build_per_recipe_not_per_model()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
