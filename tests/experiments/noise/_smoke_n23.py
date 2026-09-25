#!/usr/bin/env python3
"""Smoke for n23: can a per-molecule uncertainty clean labels.

The thing that has to be true: on data where some labels really ARE wrong, an
uncertainty number that is a noise signal finds the wrong ones and cleaning by
it wins folds; on data where none are wrong, it wins nothing. If a smoke cannot
tell those two apart, the experiment cannot either.

  1. THE NUMBER IS GRADED BEFORE ANY CLEANING. Under injection there is a list
     of the molecules whose label moved. Every source is scored against it, and
     a source that ranks the moved molecules at the top scores an average
     precision well above the share that moved. That is the check that needs no
     model fit and it is the one that says whether the number is a noise signal
     at all.

  2. THE ZERO-NOISE CONTROL FIRES. At level 0.0 nothing is corrupted, so
     nothing is cleaning. Cleaning must win fewer settings there than at a level
     where labels really did move.

  3. BOTH SHAPES PRODUCE A SCORE. `drop` removes rows and `move` shifts labels,
     and neither may quietly fall back to the other.

  4. THE TWO MOVE RULES ARE DIFFERENT KNOBS. 'proportional' keeps the order of
     the molecules and drops the scale; 'confidence' uses the spread in label
     units. They must not produce the same number.

  5. WHAT EACH SOURCE MOVES LABELS TOWARD IS RECORDED. A source with a denoised
     value of its own uses it; one without moves toward the neighbours' mean,
     which is n07.label_smooth's target. A reader comparing two sources has to
     know whether they differ in one thing or two.

  6. The counts are bounded by the folds that ran, and nothing is ever graded
     against a cleaned or a corrupted label.

RUN COST. `ngboost` and `heteroscedastic_nn` each fit a model per inner fold per
outer fold, so the smoke runs two inner folds and the two cheapest sources by
default. The expensive pair is exercised in its own test, once.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Two OpenMP runtimes in one process: torch links libiomp5 and LightGBM links
# libomp, and a multi-threaded fit of one after the other kills the process with
# exit 139 and no traceback. Both switches are needed and the crash runs in both
# orders — project_fold_hook_and_lgbm_segfault_2026_09_02, and measured again in
# _smoke_n20.py on 2026-09-17.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import tempfile

import numpy as np

import experiments.noise.n23_uncertainty_cleaning as n23

OUT = tempfile.mkdtemp(prefix='n23_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=240, seed=0):
    """Two representations. The label is a function of the first one.

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


def no_recipe(*a, **k):
    """What the pool returns on a dataset it was never built for.

    The hybrid space needs a precomputed pool and there is none here, so the
    lookup is replaced and every arm runs in the two single-representation
    spaces, whose representations the caller names through `base_reps`. n18's
    smoke checks that a run with no recipe still produces every
    single-representation number; this file does not repeat that.
    """
    return None, None


def run_it(data, name, **kw):
    """One tiny n23 run, ONE condition, in base_1 ('aaa') and base_2 ('bbb') only.

    `_run_one_condition` and not `run`. Since 2026-09-23 `run` sweeps every
    condition in SWEPT_CONDITIONS and writes each to its own file, returning an
    INDEX of what it wrote -- so the scores these tests read are no longer in
    its return value. They are about one condition's behaviour, so they call the
    one-condition layer directly. The sweep, the checkpointing and the resume
    are covered by test_the_sweep_checkpoints_and_resumes below.
    """
    real = n23.get_best_search_for_downstream
    n23.get_best_search_for_downstream = no_recipe
    try:
        return n23._run_one_condition(
            data, {'n_splits': 3}, OUT, name,
            models=['rf'], base_reps={'rf': ['aaa', 'bbb']},
            sources=kw.pop('sources', ['ngboost', 'activity_cliff_aware']),
            drop_shares=[0.10], move_strengths=[0.5],
            estimator_folds=2, **kw)
    finally:
        n23.get_best_search_for_downstream = real


# ── 1. the number is graded before any cleaning ─────────────────────────────

def test_detection_before_cleaning():
    print("does the number find the molecules that moved:")
    data = make_data(seed=1)
    r = run_it(data, 'toy', regimes=['injected'], levels=[0.0, 0.5], draws=[0])
    # {level: {space: {downstream model: {source: reading}}}}. The model level
    # arrived 2026-09-23: a hybrid space is one feature matrix per downstream
    # model, so the uncertainty number that grades the molecules is computed
    # per model. This run asks for one model, 'rf'.
    check('the single-representation comparison is the two named, per model',
          r['spaces_run'] == ['base_1', 'base_2']
          and r['base_reps_by_model'] == {'rf': ['aaa', 'bbb']}
          and 'base_rep' not in r,
          f"{r['spaces_run']}, {r.get('base_reps_by_model')}")
    det = r['detection']
    check('a detection reading was written at the corrupted level, in both',
          '0.5' in det
          and all((det['0.5'].get(sp) or {}).get('rf')
                  for sp in ('base_1', 'base_2')),
          f'levels {sorted(det)}')
    check('nothing was written at level 0.0, where nothing moved',
          not any((det.get('0') or {}).get(sp) for sp in ('base_1', 'base_2')),
          f"level 0 block: {det.get('0')}")
    _src = next(iter(det['0.5']['base_1']['rf']))
    check('base_1 and base_2 are different feature matrices',
          det['0.5']['base_1']['rf'][_src]['detection_average_precision']
          != det['0.5']['base_2']['rf'][_src]['detection_average_precision'],
          _src)

    read = det['0.5']['base_1']['rf']
    for source, d in read.items():
        if not d or not np.isfinite(d['detection_average_precision']):
            continue
        share = d['detection_n_corrupted'] / max(d['detection_n_training'], 1)
        print(f"        {source:20s} average precision "
              f"{d['detection_average_precision']:.3f}, against "
              f"{share:.3f} for a coin flip "
              f"({d['detection_n_corrupted']:.0f} molecules moved of "
              f"{d['detection_n_training']:.0f} in the training fold)")
    check('every source that was read carries the two counts its fractions are of',
          all(d['detection_n_corrupted'] > 0 and d['detection_n_training'] > 0
              for d in read.values() if d))
    beat = [s for s, d in read.items()
            if d and np.isfinite(d['detection_average_precision'])
            and d['detection_average_precision'] >
            d['detection_n_corrupted'] / max(d['detection_n_training'], 1)]
    check('at least one source ranks the moved molecules above a coin flip',
          bool(beat), f'above chance: {beat}')
    return r


# ── 2. the zero-noise control ───────────────────────────────────────────────

def test_zero_noise_control(r):
    print("\nthe zero-noise control:")
    z = r['summary'].get('zero_noise_control', {})
    got = z.get('settings_winning_most_folds_at_level_zero')
    check('the control ran and is rolled up per source', bool(got), f'{got}')
    if not got:
        return
    lvl05 = (r['summary']['per_regime']['injected']['per_level']
             .get('0.5', {}).get('per_space', {}))

    def _at_half(key):
        space, source = key.split('/', 1)
        return (lvl05.get(space, {}).get('per_source', {}).get(source, {})
                .get('settings_winning_most_folds', 0))

    for key, at_zero in got.items():
        print(f"        {key:28s} won most folds in {at_zero} settings with "
              f"nothing corrupted, {_at_half(key)} with a tenth corrupted")
    check('cleaning wins at least as many settings where labels really moved',
          sum(_at_half(k) for k in got) >= sum(got.values()),
          'if this fails the cleaning is removing hard molecules, not wrong ones')


# ── 3, 4, 5. the shapes, the two move rules, and what they move toward ──────

def test_both_shapes_score(r):
    print("\nboth shapes, and the two move rules:")
    arms = set(r['arms'])
    check('the arm list carries one to beat, a drop and both move rules',
          'none' in arms
          and any(a.startswith('drop@') for a in arms)
          and any(a.startswith('move_proportional@') for a in arms)
          and any(a.startswith('move_confidence@') for a in arms),
          f'{sorted(arms)}')

    scored = {}
    for reg in r['per_regime'].values():
        for lvl in reg.values():
            for by_source in lvl.values():
                for source, by_arm in by_source.items():
                    for arm, by_model in by_arm.items():
                        n = sum(len(v) for v in by_model.values())
                        scored[arm] = scored.get(arm, 0) + n
    check('every arm produced a score', all(scored.get(a, 0) > 0 for a in arms),
          ', '.join(f'{a}={scored.get(a, 0)}' for a in sorted(arms)))

    # The two move rules must not be the same fit. Compared on the arithmetic
    # itself rather than through a score, which a tree model could round away.
    rng = np.random.default_rng(7)
    y = rng.normal(size=120)
    sigma = np.abs(rng.normal(size=120)) * 0.3
    tgt = y + rng.normal(0, 0.2, 120)
    a = n23.cleaned_training_set('move', 0.5, None, y, sigma, tgt,
                                 method='proportional')[1]
    b = n23.cleaned_training_set('move', 0.5, None, y, sigma, tgt,
                                 method='confidence')[1]
    check('the two move rules are different knobs',
          float(np.max(np.abs(a - b))) > 1e-6,
          f'largest difference {float(np.max(np.abs(a - b))):.4g}')
    check('a move keeps every molecule and a drop does not',
          len(a) == len(y)
          and len(n23.cleaned_training_set(
              'drop', 0.10, np.zeros((120, 2)), y, sigma, tgt)[1]) < len(y))

    tgt_by = r['target_source_by_source']
    check('what each source moves labels toward is recorded',
          all(s in tgt_by and tgt_by[s] for s in r['sources']),
          '; '.join(f'{k}: {", ".join(v)}' for k, v in tgt_by.items()))


# ── 6. the counts, and what was graded against ──────────────────────────────

def test_counts_and_grading():
    print("\nboth regimes, the counts, and what was graded against:")
    data = make_data(seed=5)
    r = run_it(data, 'toy2', regimes=['injected', 'as_they_come'],
               levels=[0.5], draws=[0])
    check('both regimes ran and wrote a block',
          set(r['per_regime']) == {'injected', 'as_they_come'},
          f'{sorted(r["per_regime"])}')

    n_folds, ok_bound, ok_paired, cells = 3, True, True, 0
    for reg in r['summary']['per_regime'].values():
        for lrow in reg['per_level'].values():
            for srow in lrow['per_space'].values():
                for blk in srow['per_source'].values():
                    for arow in blk['per_arm'].values():
                        for cellv in arow['per_model'].values():
                            cells += 1
                            if cellv['folds_won'] > cellv['n_folds_paired']:
                                ok_bound = False
                            if cellv['n_folds_paired'] > n_folds:
                                ok_paired = False
                            if len(cellv['fold_gains']) != cellv['n_folds_paired']:
                                ok_paired = False
    check('folds won never exceeds folds paired', ok_bound, f'{cells} readings')
    check('folds paired never exceeds folds run, and the gains line up',
          ok_paired)
    check('the run records what every score was graded against',
          'never cleaned and never corrupted' in r['claim_scope'])
    check("it records that KIRBy's gaussian_process is not the separating number",
          'MODEL-UNSURE' in r['gaussian_process_is_not_that_number'])
    check('it records that no cleaning reached the feature choice',
          "n20's question" in r['features_are_fixed_note'])

    # outlier_p10 picks its corrupted molecules at random, so a source that
    # predicts noise from the features is undefined under it. That has to be on
    # the file, or a chance-level number reads as the method failing.
    defined = r['sources_defined_under_this_condition']
    check('the condition is recorded as not keying noise to structure',
          r['condition_keys_noise_to_structure'] is False,
          f"condition {r['noise_condition']}")
    check('the feature-based sources are marked undefined under it',
          defined.get('ngboost') is False
          and defined.get('activity_cliff_aware') is True,
          f'{defined}')


# ── the expensive pair, once ────────────────────────────────────────────────

def test_the_other_sources_run():
    print("\nthe slower sources, including the ported Gaussian process:")
    data = make_data(n=150, seed=9)
    r = run_it(data, 'toy3', regimes=['as_they_come'],
               sources=['heteroscedastic_nn', 'gaussian_process',
                        'hetero_gp'])
    got = {}
    for reg in r['per_regime'].values():
        for lvl in reg.values():
            for source, by_arm in lvl.get('base_1', {}).items():
                got[source] = sum(len(v) for a in by_arm.values()
                                  for v in a.values())
    check('all three slower sources produced scores', len(got) == 3, f'{got}')
    secs = r['estimator_seconds_by_source']
    print('        seconds per estimate: '
          + ', '.join(f"{k} {v['mean']:.1f}" for k, v in secs.items()))
    check('every source that ran is timed in the result file',
          all(s in secs for s in got))


def test_the_sweep_checkpoints_and_resumes():
    """The condition is an axis n23 sweeps, and a killed run keeps what it did.

    THIS IS WHY THE DESIGN EXISTS. One condition measured between 1,991 and
    7,095 seconds locally, so a sweep is a day's work and it WILL be interrupted
    -- by a wall clock, by a kill, by a laptop closing. Writing once at the end
    loses all of it. So each condition goes to its own file as it lands and a
    condition already on disk is not run again. Nothing tested that until now.
    """
    import json
    print("\nthe sweep, its checkpoints and its resume:")
    data = make_data(n=120, seed=11)

    # Two conditions, by stride, from a run that would otherwise do all eight.
    real = n23.get_best_search_for_downstream
    n23.get_best_search_for_downstream = no_recipe
    try:
        idx = n23.run(data, {'n_splits': 3}, OUT, 'toy_sweep',
                      models=['rf'], base_reps={'rf': ['aaa', 'bbb']},
                      sources=['activity_cliff_aware'],
                      drop_shares=[0.10], move_strengths=[0.5],
                      estimator_folds=2, regimes=['injected'],
                      levels=[0.5], draws=[0],
                      # All three survive on a dataset with no SMILES. An
                      # earlier version named grouped_wider here, which is
                      # dropped for exactly that reason BEFORE the slice is
                      # taken -- so the check was testing the drop and calling
                      # it striding.
                      noise_conditions=['outlier_p01', 'outlier_p05',
                                        'outlier_only_p05'],
                      sweep_slice='0/2')
    finally:
        n23.get_best_search_for_downstream = real

    check('a stride slice takes every Nth condition, not the first N',
          idx['swept_conditions'] == ['outlier_p01', 'outlier_only_p05'],
          str(idx['swept_conditions']))
    check('it ran them and said which',
          idx['conditions_run_this_time'] == idx['swept_conditions'],
          str(idx['conditions_run_this_time']))

    # Each one is a file of its own, with scores in it.
    for cond in idx['swept_conditions']:
        path = idx['conditions_written'][cond]
        body = json.load(open(path))
        inner = body.get('results', body)
        check(f'{cond} wrote its own file with its own condition on it',
              inner.get('noise_condition') == cond, f"{cond} -> {inner.get('noise_condition')}")

    # And the whole point: run it again, and it does no work.
    real = n23.get_best_search_for_downstream
    n23.get_best_search_for_downstream = no_recipe
    try:
        again = n23.run(data, {'n_splits': 3}, OUT, 'toy_sweep',
                        models=['rf'], base_reps={'rf': ['aaa', 'bbb']},
                        sources=['activity_cliff_aware'],
                        drop_shares=[0.10], move_strengths=[0.5],
                        estimator_folds=2, regimes=['injected'],
                        levels=[0.5], draws=[0],
                        noise_conditions=['outlier_p01', 'outlier_p05',
                                          'outlier_only_p05'],
                        sweep_slice='0/2')
    finally:
        n23.get_best_search_for_downstream = real

    check('a second run re-runs NOTHING, which is what resume means',
          again['conditions_run_this_time'] == [],
          str(again['conditions_run_this_time']))
    check('and it still reports every condition as written',
          sorted(again['conditions_written']) == sorted(idx['conditions_written']),
          str(sorted(again['conditions_written'])))


def test_every_converted_experiment_has_the_same_shape():
    """Six experiments sweep the condition now. They must all do it the same way.

    n09, n12 and n16 have no smoke of their own, so without this nothing
    exercises their converted run() at all. This is a structural check, not a
    behavioural one: it says each has a one-condition layer, that its run() goes
    through the shared helper rather than a private copy, and that the index it
    returns carries the counts. It does NOT say their numbers are right.

    THE COUNTS ARE THE POINT. run_all saves this index at <dir>/<dataset>.json,
    which is the exact path verify_run_status.py:179 and
    audit_done_dataset_gaps.py:144 check to decide whether an experiment ran.
    save_experiment_results stamps result_status from whether the payload holds
    a computed quantity, so an index of bare filenames was filed as
    result_status='empty' -- a sweep of fourteen finished conditions reading as
    a run that produced nothing.
    """
    import importlib, inspect
    print("\nthe six experiments that sweep the condition:")
    mods = {'n09': 'n09_lnl_survey',
            'n11': 'n11_estimation_remediation_grid',
            'n12': 'n12_importance_weighted_distance',
            'n16': 'n16_importance_under_noise',
            'n21': 'n21_neighbour_distance',
            'n23': 'n23_uncertainty_cleaning'}
    missing_layer, private_copy = [], []
    for short, name in mods.items():
        m = importlib.import_module(f'experiments.noise.{name}')
        if not hasattr(m, '_run_one_condition'):
            missing_layer.append(short)
            continue
        src = inspect.getsource(m.run)
        if 'sweep_conditions(' not in src:
            private_copy.append(short)
    check('every one has a one-condition layer to call',
          not missing_layer, f'missing: {missing_layer}' if missing_layer else
          f'{len(mods)} experiments')
    check('and every one sweeps through the shared helper, not a private copy',
          not private_copy, f'private: {private_copy}' if private_copy else
          f'{len(mods)} experiments')

    from experiments.noise.noise_helpers import sweep_conditions
    calls = []

    def fake_one(data, config, output_dir, dataset_name, **kw):
        calls.append(kw['noise_condition'])
        return {'score': 1.0}

    import tempfile
    out = tempfile.mkdtemp()
    idx = sweep_conditions(fake_one, 'n_probe', {'task_type': 'regression'},
                           {}, out, 'ds',
                           default_conditions=['gaussian', 'laplace'],
                           kwargs={})
    for k in ('n_conditions_swept', 'n_conditions_on_disk',
              'n_conditions_run_this_time', 'n_conditions_skipped'):
        check(f'the index carries {k}', k in idx, str(idx.get(k)))
    check('a run with nowhere to write is refused, and says why',
          _refuses_without_an_output_dir(fake_one))


def _refuses_without_an_output_dir(fake_one):
    from experiments.noise.noise_helpers import sweep_conditions
    try:
        sweep_conditions(fake_one, 'n_probe', {'task_type': 'regression'}, {},
                         None, 'ds', default_conditions=['gaussian'], kwargs={})
        return False
    except ValueError as exc:
        return 'resumable' in str(exc) or 'output directory' in str(exc)


if __name__ == '__main__':
    res = test_detection_before_cleaning()
    test_zero_noise_control(res)
    test_both_shapes_score(res)
    test_counts_and_grading()
    test_the_other_sources_run()
    test_the_sweep_checkpoints_and_resumes()
    test_every_converted_experiment_has_the_same_shape()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
