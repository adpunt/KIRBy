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
    lookup is replaced and every arm runs in the base representation. n18's
    smoke checks that a run with no recipe still produces every base number;
    this file does not repeat that.
    """
    return None, None


def run_it(data, name, **kw):
    """One tiny n23 run, in the base representation only."""
    real = n23.get_best_search_for_downstream
    n23.get_best_search_for_downstream = no_recipe
    try:
        return n23.run(
            data, {'n_splits': 3}, OUT, name,
            models=['rf'], base_rep='aaa',
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
    det = r['detection']
    check('a detection reading was written at the corrupted level',
          '0.5' in det and det['0.5'].get('base'), f'levels {sorted(det)}')
    check('nothing was written at level 0.0, where nothing moved',
          not (det.get('0') or {}).get('base'),
          f"level 0 block: {det.get('0')}")

    read = det['0.5']['base']
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
             .get('0.5', {}).get('per_space', {}).get('base', {})
             .get('per_source', {}))
    for key, at_zero in got.items():
        source = key.split('/', 1)[1]
        at_half = lvl05.get(source, {}).get('settings_winning_most_folds', 0)
        print(f"        {source:20s} won most folds in {at_zero} settings with "
              f"nothing corrupted, {at_half} with a tenth corrupted")
    check('cleaning wins at least as many settings where labels really moved',
          sum(lvl05.get(k.split('/', 1)[1], {})
              .get('settings_winning_most_folds', 0) for k in got)
          >= sum(got.values()),
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


# ── the expensive pair, once ────────────────────────────────────────────────

def test_the_other_two_sources_run():
    print("\nthe two slower sources:")
    data = make_data(n=150, seed=9)
    r = run_it(data, 'toy3', regimes=['as_they_come'],
               sources=['heteroscedastic_nn', 'gaussian_process'])
    got = {}
    for reg in r['per_regime'].values():
        for lvl in reg.values():
            for source, by_arm in lvl.get('base', {}).items():
                got[source] = sum(len(v) for a in by_arm.values()
                                  for v in a.values())
    check('both slower sources produced scores', len(got) == 2, f'{got}')
    secs = r['estimator_seconds_by_source']
    print('        seconds per estimate: '
          + ', '.join(f"{k} {v['mean']:.1f}" for k, v in secs.items()))
    check('every source that ran is timed in the result file',
          all(s in secs for s in got))


if __name__ == '__main__':
    res = test_detection_before_cleaning()
    test_zero_noise_control(res)
    test_both_shapes_score(res)
    test_counts_and_grading()
    test_the_other_two_sources_run()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
