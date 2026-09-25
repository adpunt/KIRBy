#!/usr/bin/env python3
"""Smoke for n20: does the cleaning have to come before the recipe search.

What has to be true, and why each one is here.

  1. THE TWO SEARCHES ARE ONE CODE PATH. Under `no_cleaning` both are handed
     byte-identical label columns, so the four arms are four identical fits and
     every fold must give one number four times. A gap there is the two
     searches choosing from lists of different lengths, which is the defect n15
     carries at NOISE_AUDIT_2026_09_10.md section 4, n15 defect 3. If it
     appears it is REPORTED, not absorbed into a wider tolerance.

     This is the check the level-0.0 row was meant to make, and `no_cleaning` is
     what makes it exact. Running the REAL cleanups at level 0.0 does not test
     it: at level 0.0 nothing is CORRUPTED, but `remove` still drops rows and
     `knn_smooth` still overwrites labels, so the two searches are handed
     different columns and the four arms are not expected to agree. That
     four-way spread is printed beside the control so both readings are there.

  2. The engine refuses what it said it would: train_labels together with
     label_noise, train_labels of the wrong length, train_labels together with
     rep_factories, and a train_labels column holding a non-finite value.

  3. Two different label columns land on two different cache identities, so one
     column's search can never read the other's stored rows and no wipe is
     needed.

  4. `downweight` has no searched-on-cleaned arm, and that is recorded rather
     than left as an absence. Nothing in the feature selection reads a weight.

  5. `remove` produces all four arms, including the one where the features came
     from the kept rows and the model was fitted on every row.

  6. Both regimes run, the counts are bounded by the folds that ran, and a
     detection score is written where a corrupted subset exists.

  7. THE MENU IS WHAT n07 OFFERS, MINUS WHAT THE SEARCH CANNOT BE HANDED.
     Every cleanup n20 names hands back a row mask or a full-length cleaned
     label column; every one it leaves out hands back a per-row weight and
     neither of those, or declines outright. Checked by calling each cleanup,
     not by reading the comment beside it. The four that are not defined on a
     class label are dropped on a classification dataset, and every one that
     survives that gate really runs on one.

RUN COST. The grid is cut to what the machinery needs. The search is cut to one
internal evaluator and one representation count, and tuning is off -- on 200
molecules a fast-tier tuned search is about 250s and this file runs eight of
them. The three fast-tier allocation methods and the pruning schedule are KEPT,
because they are what test 1 is about.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Two OpenMP runtimes in one process. torch links libiomp5 and LightGBM links
# libomp, and a multi-threaded fit of one after the other kills the process with
# exit 139 and no traceback. Both switches are needed: KIRBY_LGBM_THREADS caps
# LightGBM's own threads and OMP_NUM_THREADS caps the pool size, and the crash
# runs in both orders. Measured here 2026-09-17: with OMP_NUM_THREADS unset this
# file dies at exit 139 in the first search.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
# The similarity sweep and the noise cache, both added 2026-09-25. A smoke runs
# today's behaviour -- neighbours in the model-input matrix only -- unless the
# caller names settings in KIRBY_SIMILARITY_SETTINGS, and it never reads or
# writes $KIRBY_CACHE_DIR/noise_cache: a cached build would hide the spy.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
os.environ.setdefault('KIRBY_NOISE_CACHE', '0')

import re
import tempfile

import numpy as np

import experiments.noise.n20_clean_before_search as n20

OUT = tempfile.mkdtemp(prefix='n20_smoke_')
CACHE = tempfile.mkdtemp(prefix='n20_smoke_cache_')
FAIL = []

# What the smoke narrows the search to. The three fast-tier allocation methods
# and the pruning schedule stay: test 1 is about whether the two searches prune
# the same list, so leaving one allocation method would make it vacuous.
SMOKE_SEARCH = {
    'tune': False,
    'internal_evaluators': ['rf'],
    'rep_counts': [None],
}


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=200, seed=0):
    """Two representations, one of which carries the signal.

    A hybrid needs two to split a budget across. `aaa` holds the features the
    label was built from; `bbb` holds the same features reordered with noise on
    top, so it predicts too and predicts worse.

    Groups are singletons: the fold split is not what this file is testing.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 6))
    y = a @ rng.normal(size=6) + rng.normal(0, 0.25, n)
    b = a[:, ::-1] + rng.normal(0, 0.6, size=a.shape)
    return {
        'labels': y,
        'reps': {'aaa': a, 'bbb': b},
        'groups': np.arange(n),
        'task_type': 'regression',
        'metric': 'rmse',
    }


def run_it(data, name, **kw):
    """One tiny n20 run.

    The detector and the cleaning run in rf's top 2 single representations,
    ranked inside each fold from that fold's own search on the uncleaned
    labels (owner's decision 2026-09-25). `aaa` and `bbb` are the only two, so
    every fold runs both as base_1 and base_2 in some order.
    """
    return n20.run(
        data, {'n_splits': 2}, OUT, name,
        models=['rf'], flag_rates=[0.10],
        cleanups=['remove', 'knn_smooth', 'downweight'],
        budgets=[10], search_n_folds=2, cache_dir=CACHE,
        search_settings=dict(SMOKE_SEARCH), **kw)


# ── 1. the control: the two searches are one code path ──────────────────────

def test_search_path_identity():
    print("the two searches are one code path:")
    data = make_data(seed=1)
    r = run_it(data, 'toy', regimes=['injected'], levels=[0.0], draws=[0])
    idt = r['summary']['search_path_identity']
    n_read, worst = idt['n_cells_read'], idt['max_spread_over_arms_anywhere']
    check('the control was actually read',
          n_read > 0 and idt['n_folds_probed'] > 0,
          f"{idt['n_folds_probed']} folds, {n_read} score cells")
    check('the control really did hand the two searches one label column',
          all(p['label_columns_identical'] for p in idt['per_fold'].values()))

    # The defect stated directly. n15's third defect is two searches choosing
    # from lists of DIFFERENT LENGTHS, so that is what is counted, not inferred
    # from a score.
    check('both searches scored the same number of candidates on every fold',
          idt['n_folds_where_both_searched_the_same_number_of_candidates']
          == idt['n_folds_probed'],
          ', '.join(f"{k.split('|')[-1]}: "
                    f"{p['n_candidates_searched_asis']} vs "
                    f"{p['n_candidates_searched_cleaned']}"
                    for k, p in idt['per_fold'].items()))
    check('both searches picked the same recipe everywhere',
          not idt['recipes_that_did_not_match'],
          f"{idt['n_recipes_that_matched']} matched"
          + (f", DID NOT MATCH: {idt['recipes_that_did_not_match']}"
             if idt['recipes_that_did_not_match'] else ''))
    if idt['recipes_that_did_not_match']:
        print("        ^ REPORT THIS. Two identical label columns picked two "
              "different\n          recipes. Nothing about cleaning can be "
              "read until it is fixed, and\n          the tolerance below is "
              "not the thing to widen.")

    # And the consequence on the scores. The four arms are four identical fits,
    # so this cannot be asserted at zero, and the reason is not generic float
    # dust -- it is that the model does not predict the same number twice.
    #
    # MEASURED 2026-09-17. One RandomForestRegressor(n_jobs=-1, random_state=42)
    # fitted once on 160 rows, then asked to predict the same 40 rows forty
    # times: 0 of 39 repeats were bit-identical to the first, the largest
    # difference being 8.88e-16. At n_jobs=1, 39 of 39 were identical and the
    # largest difference was exactly 0. sklearn sums each tree's prediction into
    # one shared array from worker threads, so the summation order is the thread
    # completion order and floating-point addition is not associative
    # (sklearn.ensemble._forest._accumulate_prediction). The roster's rf is
    # n_jobs=-1 (find_best_hybrid.py:2382).
    #
    # So this control, run three times over on one process with a fresh cache
    # each time, gave 2.22e-16, 2.22e-16 and exactly 0.0. An equality check here
    # is a coin flip. What IS exact is the candidate count and the recipe, and
    # both are checked above at equality. For scale, the same four arms under a
    # REAL cleanup sit 0.24 apart in this smoke.
    print(f"        four-way score spread: {worst:.3g} (the rf roster model's "
          f"own prediction noise is ~1e-15; see above)")
    check('the four arms give one number four times, to the last bit',
          worst < 1e-12, f'largest four-way spread {worst:.6g}'
          + (f' at {idt["worst_cell"]}' if idt['worst_cell'] else ''))
    check('the search narrowing is written to the file, so a short run reads short',
          r['search_settings'].get('tune') is False
          and r['search_settings'].get('speed') == 'fast'
          and r['adaptive_pruning_schedule'] is not None,
          f"{r['search_settings']}")
    return r


def test_level_zero_reading(r):
    """The level-0.0 four-way spread for the REAL cleanups: printed, not gated.

    At level 0.0 nothing is corrupted but the cleanups still fire, so the four
    arms are not handed identical labels and are not expected to agree. The
    number is here so it can be read beside the control.
    """
    print("\nlevel 0.0, the real cleanups (a reading, not a gate):")
    lvl = r['summary']['per_regime']['injected']['per_level'].get('0')
    if not lvl:
        check('level 0.0 ran', False, 'no level-0 block')
        return
    seen = []
    for cleanup, crow in ((c, cr) for brow in lvl['per_base'].values()
                          for c, cr in brow['per_cleanup'].items()):
        if cleanup == n20.IDENTITY_CONTROL:
            continue
        for rk, rrow in crow['per_flag_rate'].items():
            for m, mrow in rrow['per_model'].items():
                vals = [v for v in mrow['arm_means'].values() if np.isfinite(v)]
                if len(vals) >= 2:
                    seen.append((cleanup, m, len(vals), max(vals) - min(vals)))
    check('level 0.0 produced a four-way reading for the real cleanups',
          len(seen) > 0, f'{len(seen)} cells')
    for cleanup, m, n_arms, spread in seen:
        print(f"        {cleanup:12s} {m:4s} {n_arms} arms, "
              f"four-way spread {spread:.4f}")


# ── 2. what the engine refuses ──────────────────────────────────────────────

def test_engine_refusals():
    print("\nwhat find_best_hybrid refuses:")
    from kirby.find_best_hybrid import find_best_hybrid
    from kirby.noise_spec import default_spec

    rng = np.random.default_rng(3)
    n = 60
    reps = {'aaa': rng.normal(size=(n, 4)), 'bbb': rng.normal(size=(n, 3))}
    y = reps['aaa'] @ rng.normal(size=4)
    base = dict(base_reps=reps, labels=y, groups=np.arange(n), budgets=[5],
                models=['rf'], metric='rmse', task_type='regression',
                n_folds=2, verbose=False, cache=False, speed='fast')

    cases = [
        ('train_labels together with label_noise',
         dict(train_labels=y,
              label_noise=default_spec(False, target_effective_noise=0.3)),
         'not both'),
        ('train_labels of the wrong length',
         dict(train_labels=y[:10]), 'entries'),
        ('train_labels together with rep_factories',
         dict(train_labels=y, rep_factories={'zzz': lambda *a: (None, None)},
              rep_factories_id='zzz'),
         'rep_factories'),
        ('a non-finite value in train_labels',
         dict(train_labels=np.concatenate([[np.nan], y[1:]])), 'non-finite'),
    ]
    for name, kw, needle in cases:
        try:
            find_best_hybrid(**{**base, **kw})
            check(f'{name} raises', False, 'it did not raise')
        except ValueError as exc:
            check(f'{name} raises', needle in str(exc), f'{str(exc)[:64]}...')
        except Exception as exc:
            check(f'{name} raises ValueError', False,
                  f'{type(exc).__name__}: {str(exc)[:50]}')


# ── 3. two columns, two cache identities ────────────────────────────────────

def test_two_columns_two_identities():
    print("\ntwo label columns cannot read each other's stored rows:")
    from kirby.find_best_hybrid import find_best_hybrid

    rng = np.random.default_rng(4)
    n = 90
    reps = {'aaa': rng.normal(size=(n, 5)), 'bbb': rng.normal(size=(n, 3))}
    y = reps['aaa'] @ rng.normal(size=5) + rng.normal(0, .2, n)
    y2 = y.copy()
    y2[:15] = y2[:15] + 4.0
    cdir = tempfile.mkdtemp(prefix='n20_ident_')
    base = dict(base_reps=reps, labels=y, groups=np.arange(n), budgets=[5],
                models=['rf'], metric='rmse', task_type='regression',
                n_folds=2, verbose=False, cache=True, cache_dir=cdir,
                dataset_name='smoke', tier='general', speed='fast',
                allocation_methods=['performance_weighted'],
                internal_evaluators=['rf'], rep_counts=[None])
    find_best_hybrid(train_labels=y, **base)
    find_best_hybrid(train_labels=y2, **base)

    tokens, files = set(), []
    for root, _dirs, names in os.walk(cdir):
        for f in names:
            files.append(f)
            tokens |= set(re.findall(r'trainlabels__lb[0-9a-f]{8}', f))
    check('every stored file names the column it was built from',
          len(tokens) > 0, f'{len(files)} files')
    check('the two columns land on two identities, not one',
          len(tokens) == 2, f'identities {sorted(tokens)}')
    # And the identity is the column's own content, not a name for it.
    per_rep = [f for f in files if f.startswith('smoke__aaa__')]
    check('the same representation has one stored score per column',
          len(per_rep) == 2, f'{sorted(per_rep)}')


# ── 4 and 5. what each cleanup can and cannot reach ─────────────────────────

def test_downweight_is_refused_not_faked(r):
    print("\ndownweight cannot reach the feature selection:")
    why = r['cleanup_reaches_feature_selection'].get('downweight', '')
    check('the result file says why, in the file',
          'sample_weight' in why and 'zero times' in why, why[:64] + '...')

    import kirby.hybrid as _h
    import kirby.find_best_hybrid as _f
    src = open(_h.__file__).read() + open(_f.__file__).read()
    check('sample_weight really is absent from both engine files',
          'sample_weight' not in src)

    ran, leaked = False, []
    for reg in r['per_regime'].values():
        for lvl in (b for lv in reg.values() for b in lv.values()):
            for rate in (lvl.get('downweight') or {}).values():
                for arms in rate.values():
                    ran = ran or bool(arms.get('search_asis__fit_cleaned'))
                    for a in ('search_cleaned__fit_cleaned',
                              'search_cleaned__fit_asis'):
                        if arms.get(a):
                            leaked.append(a)
    check('downweight was run and scored the two arms it does have', ran)
    check('downweight writes no searched-on-cleaned score', not leaked,
          f'{leaked}')


def test_remove_has_all_four(r):
    print("\nremove reaches the feature selection by dropping rows:")
    got = {}
    for reg in r['per_regime'].values():
        for lvl in (b for lv in reg.values() for b in lv.values()):
            for rate in (lvl.get('remove') or {}).values():
                for arms in rate.values():
                    for a in n20.ARMS:
                        got[a] = got.get(a, 0) + len(arms.get(a) or {})
    check('all four arms produced a score for remove',
          all(got.get(a, 0) > 0 for a in n20.ARMS),
          ', '.join(f'{a}={got.get(a, 0)}' for a in n20.ARMS))
    check('the fourth arm exists: features from the kept rows, fit on every row',
          got.get('search_cleaned__fit_asis', 0) > 0)


# ── 7. the menu: which of n07's cleanups n20 runs, and why not the rest ─────

def test_the_menu_matches_n07():
    """Every name n20 runs is one n07 offers, and every name it does not run
    returns something the recipe search cannot be handed.

    This is the invariant behind n20.CLEANUPS, checked by CALLING the cleanup
    rather than by reading the comment next to it. `keep` is a row mask the
    cleaned search can run on and `label_column` is a full-length column it can
    take as train_labels=; a cleanup handing back neither has no cleaned search
    and so no searched-on-cleaned arm.

    Nothing here fits a hybrid or runs a search, so it costs one small fit per
    cleanup.
    """
    print("\nwhich of n07's cleanups n20 runs:")
    from experiments.noise.n07_remediation import (
        CLEANUPS as N07_CLEANUPS, REGRESSION_ONLY_CLEANUPS,
        SCORE_FREE_CLEANUPS, cleaned_training_set)

    check('every cleanup n20 names is one n07 offers',
          all(c in N07_CLEANUPS for c in n20.CLEANUPS),
          f'{[c for c in n20.CLEANUPS if c not in N07_CLEANUPS]}')

    rng = np.random.default_rng(7)
    n = 60
    X = rng.normal(size=(n, 4))
    y = X @ rng.normal(size=4) + rng.normal(0, .2, n)
    scores = rng.random(n)
    g = np.arange(n)

    def built(name, is_clf=False, labels=None):
        try:
            return cleaned_training_set(name, X, y if labels is None else labels,
                                        scores, 0.10, is_clf=is_clf, groups=g)
        except Exception as exc:
            return exc

    reaches, refused = [], []
    for name in n20.CLEANUPS:
        if name == 'downweight':
            continue
        out = built(name)
        ok = (isinstance(out, dict)
              and (out.get('keep') is not None
                   or out.get('label_column') is not None))
        (reaches if ok else refused).append(name)
    check('every cleanup n20 runs hands back a row mask or a label column',
          not refused,
          f'{len(reaches)} do: {reaches}' + (f'; these do not: {refused}'
                                             if refused else ''))

    # And the ones left out, by the value they return rather than by argument.
    for name in N07_CLEANUPS:
        if name in n20.CLEANUPS:
            continue
        out = built(name)
        if out is None:
            check(f'{name} is left out: it declines outright here', True,
                  'returned None, so there is nothing to search on')
            continue
        check(f'{name} is left out: it returns a weight and nothing else',
              isinstance(out, dict) and out.get('keep') is None
              and out.get('label_column') is None
              and out.get('sample_weight') is not None,
              'keep None, label_column None, sample_weight set')

    # downweight is in the list and is the one exception: it is scored on its
    # two fit-side arms and its two searched-on-cleaned arms are recorded as
    # undefined. test_downweight_is_refused_not_faked checks that end.
    dw = built('downweight')
    check('downweight, the one in the list that cannot reach the search, '
          'returns a weight and nothing else',
          isinstance(dw, dict) and dw.get('keep') is None
          and dw.get('label_column') is None
          and dw.get('sample_weight') is not None)

    # The classification gate.
    run_clf, dropped_clf = n20.cleanups_for(True)
    check('a regression-only cleanup is not offered on a class label',
          set(dropped_clf) == {c for c in n20.CLEANUPS
                               if c in REGRESSION_ONLY_CLEANUPS}
          and not any(c in REGRESSION_ONLY_CLEANUPS for c in run_clf),
          f'dropped {list(dropped_clf)}, kept {list(run_clf)}')
    run_reg, dropped_reg = n20.cleanups_for(False)
    check('nothing is dropped on a regression label',
          run_reg == n20.CLEANUPS and dropped_reg == ())

    # Every cleanup that survives the gate really does run on a class label.
    y_clf = (y > np.median(y)).astype(int)
    broke = [c for c in run_clf
             if isinstance(built(c, is_clf=True, labels=y_clf), Exception)]
    check('every cleanup offered on a class label really runs on one',
          not broke, f'{broke}')

    # The permuted-flag control, one entry per cleanup the shuffled score can
    # reach.
    want_control = {c for c in n20.CLEANUPS
                    if c not in ('downweight',)
                    and c not in SCORE_FREE_CLEANUPS}
    have_control = {v for v in n20.RANDOM_FLAG_CLEANUPS.values()}
    check('every cleanup the suspicion score decides has a permuted-flag twin',
          want_control == have_control,
          f'without one: {sorted(want_control - have_control)}; '
          f'with one and not expected: {sorted(have_control - want_control)}')
    check('a permuted-flag twin runs the cleanup it is named after',
          all(v in n20.CLEANUPS for v in n20.RANDOM_FLAG_CLEANUPS.values()),
          f'{len(n20.RANDOM_FLAG_CLEANUPS)} twins')


# ── 6. both regimes, the counts, and what was graded against ────────────────

def test_regimes_counts_and_grading():
    print("\nboth regimes, the counts, and what was graded against:")
    data = make_data(seed=5)
    r = run_it(data, 'toy2', regimes=['injected', 'as_they_come'],
               levels=[0.3], draws=[0])
    check('both regimes ran and wrote a block',
          set(r['per_regime']) == {'injected', 'as_they_come'},
          f'{sorted(r["per_regime"])}')
    check('the uninjected regime is keyed as having no level',
          list(r['per_regime']['as_they_come']) == ['none'],
          f'{list(r["per_regime"]["as_they_come"])}')

    n_folds = 2
    ok_bound = ok_paired = True
    cells = 0
    for reg in r['summary']['per_regime'].values():
        for lvl in (b for lv in reg['per_level'].values()
                    for b in lv['per_base'].values()):
            for crow in lvl['per_cleanup'].values():
                for rrow in crow['per_flag_rate'].values():
                    for mrow in rrow['per_model'].values():
                        for key in ('ordering_cost', 'feature_choice_only'):
                            cell = mrow.get(key)
                            if not cell:
                                continue
                            cells += 1
                            if cell['folds_won'] > cell['n_folds_paired']:
                                ok_bound = False
                            if cell['n_folds_paired'] > n_folds:
                                ok_paired = False
                            if len(cell['fold_gains']) != cell['n_folds_paired']:
                                ok_paired = False
    check('folds won never exceeds folds paired', ok_bound, f'{cells} readings')
    check('folds paired never exceeds folds run, and the gains line up',
          ok_paired)
    check('the run records what every score was graded against',
          'never cleaned and never corrupted' in r['claim_scope'])
    check('the two arms the question is about are named in the file',
          r['arm_the_suite_runs_today'] == 'search_asis__fit_cleaned'
          and r['arm_the_question_is_about'] == 'search_cleaned__fit_cleaned')

    det = {f'{b}/{k}': v for b, by_lvl in (r.get('detection') or {}).items()
           for k, v in by_lvl.items()}
    read = [k for k, v in det.items()
            if v and np.isfinite(v.get('detection_average_precision',
                                       float('nan')))]
    check('a detection score is written under the injected condition',
          len(read) > 0, f'levels with a reading: {sorted(read)}')
    if read:
        d = det[read[0]]
        check('the detection score names what its fractions count',
              d['detection_n_corrupted'] > 0 and d['detection_n_training'] > 0,
              f"{d['detection_n_corrupted']} molecules moved of "
              f"{d['detection_n_training']} in the training fold")
    return r


def test_quick_two_single_representations():
    """The quick screen: one fold pair, no injection, both single
    representations run the detector and the cleanups."""
    print("\nthe single representations come from each fold's own search:")
    r = run_it(make_data(seed=7), 'toy_quick', regimes=['as_they_come'])
    per_fold = r['base_reps_by_fold']
    check("every fold names rf's top 2 from its own search",
          per_fold and all(sorted(v) == ['base_1', 'base_2']
                           and sorted(v.values()) == ['aaa', 'bbb']
                           for v in per_fold.values()), str(per_fold))
    lvl = r['per_regime']['as_they_come']['none']
    got = {b: sum(len(f) for c in lvl[b].values() for rt in c.values()
                  for arms in rt.values() for f in arms.values())
           for b in ('base_1', 'base_2') if b in lvl}
    check('both single representations produced arm scores',
          set(got) == {'base_1', 'base_2'} and all(got.values()), str(got))


if __name__ == '__main__':
    if '--quick' in sys.argv:
        test_quick_two_single_representations()
        print("\n" + "=" * 60)
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}" if FAIL else "ALL GREEN")
        sys.exit(1 if FAIL else 0)
    res = test_search_path_identity()
    test_level_zero_reading(res)
    test_engine_refusals()
    test_two_columns_two_identities()
    test_downweight_is_refused_not_faked(res)
    test_remove_has_all_four(res)
    test_the_menu_matches_n07()
    test_regimes_counts_and_grading()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
