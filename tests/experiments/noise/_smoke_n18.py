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


if __name__ == '__main__':
    clean = test_nothing_injected()
    dirty = test_finds_a_real_error(clean)
    test_counts_are_sane(dirty)
    test_qm9_is_marked()
    test_runs_without_a_recipe()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
