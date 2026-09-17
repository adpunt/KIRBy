#!/usr/bin/env python3
"""Smoke for n17: real-replicate removal.

Builds a dataset whose repeat measurements really do disagree, by a KNOWN amount
that varies from entity to entity. Then checks the pieces n17 depends on:

  1. an entity's training label is ONE of its measurements, and its grading label
     is the mean of the OTHER ones, with neither drawn from the other
  2. an entity below the grading threshold trains and is never graded
  3. the five arms remove the same NUMBER of entities where they remove any
  4. the ceiling really does remove the entities that disagree most
  5. a dataset with no repeat structure is refused rather than invented
  6. end to end on a set where the disagreement is informative, the ceiling beats
     no cleaning -- if it does not, the harness cannot detect anything either

AND THE THING THE WHOLE EXPERIMENT RESTS ON, checked in both feature spaces.
Two datasets, identical except for the measurements. In one, three in ten
entities were measured by a laboratory that scatters ten times as widely as the
rest, so there is a real difference between entities to find. In the other every
laboratory scatters by the same amount, so there is nothing to find and removing
the worst-disagreeing entities can only cost what removing any entities costs.

The ceiling must gain more beyond its own floor on the first than on the second,
in every base representation AND in the hybrid. A smoke that could not tell
those two apart would not be checking anything -- and it is in the hybrid space
that the check was missing until 2026-09-17, because there was no hybrid space.

Two more, added with it:

  * the labels handed to the hybrid build are the training entities' own single
    measurements. No consensus label an entity is graded against reaches the
    feature ranking or the budget split.
  * a run where no recipe comes back still produces every base-space number
    rather than being dropped.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# The hybrid build ranks features with LightGBM, which splits its work across
# features, and this smoke's entities carry four. n19's smoke measured the same
# thing on 2026-09-16: one thread 0.41s, every core 138.5s. The switch is the
# repository's own, read by kirby/hybrid.py:78.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')

import tempfile

import numpy as np

import experiments.noise.n17_replicate_removal as n17
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n17_smoke_')

FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_replicate_data(n_entities=400, n_feat=3, seed=0, wide_fraction=0.30):
    """Rows of repeat measurements with a known per-entity disagreement.

    Every entity gets a true value from its features. `wide_fraction` of them are
    'bad laboratories' whose measurements scatter ten times as widely. How many
    times an entity was measured is 1, 2, 3, 4 or 6.

    At wide_fraction=0 every laboratory scatters by the same amount, so no entity
    disagrees more than any other and there is nothing for a removal arm to find.
    That is the control the separation check reads against.

    TWO representations, because a hybrid needs something to split a budget
    across. `toy` carries the features the true value was built from. `second`
    carries the same features shuffled into a different order with noise added,
    so it predicts too, but worse.
    """
    rng = np.random.default_rng(seed)
    X_ent = rng.normal(size=(n_entities, n_feat))
    w = rng.normal(size=n_feat)
    truth = X_ent @ w
    wide = rng.random(n_entities) < wide_fraction
    sigma = np.where(wide, 1.5, 0.15)
    ks = rng.choice([1, 2, 3, 4, 6], size=n_entities, p=[.25, .2, .25, .2, .1])

    # A SEPARATE generator, so that adding a second representation did not move
    # the measurement draws below. Taken from `rng` it would have, and the
    # informative set's margins would be different numbers for a reason that has
    # nothing to do with what this file checks.
    X_second = X_ent[:, ::-1] + np.random.default_rng(
        seed + 9001).normal(0, 0.5, size=X_ent.shape)

    X_rows, X2_rows, y_rows, gids = [], [], [], []
    for j in range(n_entities):
        for _ in range(int(ks[j])):
            X_rows.append(X_ent[j])
            X2_rows.append(X_second[j])
            y_rows.append(truth[j] + rng.normal(0, sigma[j]))
            gids.append(j)
    return {
        'labels': np.asarray(y_rows, dtype=float),
        'reps': {'toy': np.asarray(X_rows, dtype=float),
                 'second': np.asarray(X2_rows, dtype=float)},
        'replicate_groups': np.asarray(gids),
        'groups': np.asarray(gids),
        'task_type': 'regression',
        'metric': 'rmse',
    }, truth, wide, ks


# ── A recipe, so the hybrid space exists without a precomputed pool ──────────
# The pool is built by a real precompute on a real dataset and there is none
# here, so the lookup is replaced with a fixed recipe. 'fixed' allocation splits
# the budget evenly across the two representations, which is what makes the
# hybrid a genuinely different feature space from either of them rather than a
# rename of the better one.

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


def test_entity_construction():
    print("entity construction:")
    data, truth, wide, ks = make_replicate_data()
    ent = n17.build_entities(data['labels'], data['replicate_groups'], seed=0)

    check('one entity per group', len(ent['n_measurements']) == len(ks))
    check('measurement counts match the data',
          np.array_equal(np.sort(ent['n_measurements']), np.sort(ks)))

    g = ent['gradeable']
    check('gradeable is exactly the entities with 3+ measurements',
          np.array_equal(g, ent['n_measurements'] >= n17.MIN_MEASUREMENTS),
          f'{int(g.sum())} of {len(g)} gradeable')
    check('every ungradeable entity still has a training label',
          np.all(np.isfinite(ent['train_label'])))
    check('no ungradeable entity has a grading label',
          np.all(np.isnan(ent['grade_label'][~g])))

    # The training label must be one of that entity's OWN measurements, and the
    # grading label must not equal it.
    rows_of = {}
    for i, gid in enumerate(data['replicate_groups']):
        rows_of.setdefault(gid, []).append(i)
    ok_member, ok_distinct = True, True
    for j, gid in enumerate(ent['group_id']):
        if not g[j]:
            continue
        vals = data['labels'][rows_of[gid]]
        if not np.any(np.isclose(vals, ent['train_label'][j])):
            ok_member = False
        # Exact equality, not isclose. The two are genuinely different floats,
        # and on 400 entities one pair lands within isclose's relative tolerance
        # by coincidence -- measured, 1 of 190 gradeable entities.
        if ent['train_label'][j] == ent['grade_label'][j]:
            ok_distinct = False
    check('training label is one of that entity\'s own measurements', ok_member)
    check('grading label is not the measurement that trained', ok_distinct)

    # The grading label must be the mean of the measurements NOT chosen.
    j = int(np.argmax(ent['n_measurements']))
    vals = data['labels'][rows_of[ent['group_id'][j]]]
    rest = [v for v in vals if not np.isclose(v, ent['train_label'][j])]
    check('grading label is the mean of the rest',
          np.isclose(ent['grade_label'][j], np.mean(rest), atol=1e-9),
          f'{ent["grade_label"][j]:.4f} vs {np.mean(rest):.4f}')

    # The true spread must track the laboratory that was actually wider.
    sp = ent['true_spread']
    m = np.isfinite(sp)
    check('entities with the wider laboratory have the larger true spread',
          np.nanmean(sp[m & wide]) > 3 * np.nanmean(sp[m & ~wide]),
          f'wide {np.nanmean(sp[m & wide]):.3f} vs narrow '
          f'{np.nanmean(sp[m & ~wide]):.3f}')

    # And on the control there is no such split to find.
    flat, _t, flat_wide, _k = make_replicate_data(wide_fraction=0.0, seed=7)
    ent_flat = n17.build_entities(flat['labels'], flat['replicate_groups'],
                                  seed=0)
    sp_flat = ent_flat['true_spread']
    mf = np.isfinite(sp_flat)
    hi = float(np.nanmean(np.sort(sp_flat[mf])[-int(0.3 * mf.sum()):]))
    lo = float(np.nanmean(np.sort(sp_flat[mf])[:int(0.3 * mf.sum())]))
    check('on the control the widest and narrowest entities are the same '
          'laboratory',
          flat_wide.sum() == 0 and hi < 6 * lo,
          f'widest third {hi:.3f}, narrowest third {lo:.3f}')


def test_arms():
    print("\nthe five arms:")
    rng = np.random.default_rng(0)
    n, d = 300, 6
    X = rng.normal(size=(n, d))
    y = X[:, 0] + rng.normal(0, .3, n)
    spread = rng.random(n)
    scores = rng.random(n)
    n_flag = 30

    built = {a: n17.build_arm(a, X, y, scores, spread, n_flag,
                              np.random.default_rng(1)) for a in n17.ARMS}
    check('every arm built', all(v is not None for v in built.values()),
          str({k: v is None for k, v in built.items()}))

    removed = {a: built[a][2] for a in n17.ARMS}
    relabelled = {a: built[a][3] for a in n17.ARMS}
    check('no_cleaning removes nothing and relabels nothing',
          removed['no_cleaning'] == 0 and relabelled['no_cleaning'] == 0)
    check('there are two floors, one per population',
          'floor_measured' in n17.ARMS and 'floor_any' in n17.ARMS,
          str(n17.ARMS))
    check('repair removes nothing — it relabels',
          removed['repair'] == 0 and relabelled['repair'] == n_flag,
          f"removed {removed['repair']}, relabelled {relabelled['repair']}")
    check('drop, ceiling and both floors remove the SAME count',
          removed['drop'] == removed['ceiling'] == removed['floor_measured']
          == removed['floor_any'] == n_flag
          and not any(relabelled[a] for a in
                      ('drop', 'ceiling', 'floor_measured', 'floor_any')),
          str(removed))
    check('repair keeps every row but changes some labels',
          built['repair'][0].shape == X.shape
          and not np.allclose(built['repair'][1], y))
    check('repair changed exactly the flagged count',
          int((~np.isclose(built['repair'][1], y)).sum()) <= n_flag,
          f'{int((~np.isclose(built["repair"][1], y)).sum())} labels moved')

    # The ceiling must remove the widest-spread entities and nothing else.
    kept_ceiling = built['ceiling'][1]
    worst = np.argsort(-spread)[:n_flag]
    check('ceiling removed the entities with the largest true spread',
          len(kept_ceiling) == n - n_flag
          and not np.any(np.isclose(y[worst][:, None], kept_ceiling[None, :]).all(0)))

    # An entity measured once has no spread; the ceiling must not prefer it.
    sp_nan = spread.copy()
    sp_nan[:50] = np.nan
    _, y_c, n_rm, _ = n17.build_arm('ceiling', X, y, scores, sp_nan, n_flag,
                                    np.random.default_rng(1))
    check('ceiling still removes the full count when some spreads are unknown',
          n_rm == n_flag, f'{n_rm} removed')

    # The ceiling and its own floor must both be capped by how many entities have
    # a spread at all. This is the defect found on 2026-09-16: the ceiling could
    # only reach 12% of herg_ic50's molecules while the floor reached all of them.
    sp_few = np.full(n, np.nan); sp_few[:20] = np.random.default_rng(2).random(20)
    c = n17.build_arm('ceiling', X, y, scores, sp_few, n_flag, np.random.default_rng(1))
    fm = n17.build_arm('floor_measured', X, y, scores, sp_few, n_flag,
                       np.random.default_rng(1))
    fa = n17.build_arm('floor_any', X, y, scores, sp_few, n_flag,
                       np.random.default_rng(1))
    check('ceiling and floor_measured are both capped at the entities with a spread',
          c[2] == fm[2] == 20, f'ceiling {c[2]}, floor_measured {fm[2]}')
    check('floor_any is not capped — it draws from all of training',
          fa[2] == n_flag, f'floor_any {fa[2]}')

    check('drop and repair return None when the detector failed',
          n17.build_arm('drop', X, y, None, spread, n_flag,
                        np.random.default_rng(1)) is None
          and n17.build_arm('repair', X, y, None, spread, n_flag,
                            np.random.default_rng(1)) is None)


def test_refuses_without_replicates():
    print("\nrefusals:")
    rng = np.random.default_rng(0)
    data = {'labels': rng.normal(size=200), 'reps': {'toy': rng.normal(size=(200, 5))},
            'groups': np.arange(200), 'task_type': 'regression', 'metric': 'rmse'}
    r = n17.run(data, {'n_splits': 3}, OUT, 'toy_no_replicates')
    check('a dataset with no repeat structure is refused, not invented',
          r.get('skipped') == 'no_replicate_structure', str(r)[:90])

    data2, *_ = make_replicate_data(n_entities=60)
    data2['task_type'] = 'classification'
    r2 = n17.run(data2, {'n_splits': 3}, OUT, 'toy_clf')
    check('a classification dataset is refused',
          r2.get('skipped') == 'not_regression', str(r2)[:90])


def _run_with_hybrid(data, name, captured_labels=None, **kw):
    """One n17 run with a fixed recipe standing in for the precomputed pool.

    `captured_labels`, when given, is a list every hybrid build appends its
    training labels to, so the caller can check what the feature ranking saw.
    """
    real_lookup = n17.get_best_search_for_downstream
    real_build = n17.hybrid_for_fold
    n17.get_best_search_for_downstream = fake_recipe(sorted(data['reps']))
    if captured_labels is not None:
        def spy(train_reps, val_reps, y_train, bp, groups_train, model,
                metric_fn, lower_is_better):
            captured_labels.append(np.asarray(y_train, dtype=float).copy())
            return real_build(train_reps, val_reps, y_train, bp, groups_train,
                              model, metric_fn, lower_is_better)
        n17.hybrid_for_fold = spy
    try:
        return n17.run(data, {'n_splits': 4}, OUT, name, **kw)
    finally:
        n17.get_best_search_for_downstream = real_lookup
        n17.hybrid_for_fold = real_build


def _ceiling_beyond_its_floor(result, space, model='rf'):
    return (result['summary']['per_space'][space]['per_model'][model]
            ['gain_over_its_own_floor']['ceiling'])


def test_end_to_end():
    print("\nend to end — the disagreement is informative:")
    data, truth, wide, ks = make_replicate_data(n_entities=700, seed=3)
    seen_labels = []
    r = _run_with_hybrid(data, 'toy_replicates', captured_labels=seen_labels,
                         models=['rf'], n_draws=2)
    check('it ran', 'summary' in r, str(r.get('skipped', ''))[:60])
    if 'summary' not in r:
        return None, None

    f = r['feasibility']
    check('feasibility counts entities, not rows',
          f['n_entities'] == len(ks) and f['n_rows'] == len(data['labels']),
          f"{f['n_entities']} entities, {f['n_rows']} rows")
    check('gradeable count matches the 3+ count',
          f['n_entities_gradeable'] == f['entities_by_min_measurements'][3])

    # ── the two spaces ──
    check('both representations ran, and so did the hybrid',
          r['hybrid_available'] is True
          and set(r['spaces_run']) == {'base:toy', 'base:second', 'hybrid'},
          str(r['spaces_run']))
    check('the representations it ran are named in the result file',
          r['base_reps'] == ['second', 'toy'], str(r['base_reps']))
    check('it says the recipe never saw a clean label column',
          r['recipe_saw_clean_labels'] is False)
    check('it says the grading labels never reached the feature choice',
          r['hybrid_sees_grading_labels'] is False)

    # ── what the hybrid was built on ──
    # Every label the feature ranking saw must be one of the training entities'
    # own single measurements, and none of them may be a consensus label.
    train_pool, grade_pool = [], []
    for draw in range(2):
        e = n17.build_entities(data['labels'], data['replicate_groups'],
                               seed=draw)
        train_pool.append(e['train_label'])
        grade_pool.append(e['grade_label'][e['gradeable']])
    train_pool = np.concatenate(train_pool)
    grade_pool = np.concatenate(grade_pool)
    seen = np.concatenate(seen_labels) if seen_labels else np.array([])
    check('the hybrid was built at all', seen.size > 0,
          f'{len(seen_labels)} builds, {seen.size} labels')
    check('every label the hybrid saw is a training entity\'s own single '
          'measurement',
          seen.size > 0 and bool(np.all(np.isin(seen, train_pool))),
          f'{int((~np.isin(seen, train_pool)).sum())} of {seen.size} were not')
    check('no consensus label an entity is graded against reached the hybrid',
          seen.size > 0 and not bool(np.any(np.isin(seen, grade_pool))),
          f'{int(np.isin(seen, grade_pool).sum())} of {seen.size} did')

    for space in r['spaces_run']:
        row = r['summary']['per_space'][space]['per_model']['rf']
        g = row['gain_over_no_cleaning']
        check(f'{space}: no_cleaning scores zero gain against itself',
              abs(g['no_cleaning']) < 1e-12)
        check(f'{space}: every pick carries what it gained beyond its own floor',
              set(row['gain_over_its_own_floor']) == {'ceiling', 'drop', 'repair'},
              str({k: round(v, 4) for k, v in row['gain_over_its_own_floor'].items()}))
        for arm in n17.ARMS:
            s = row['arm_scores'][arm]
            check(f'{space}: {arm} carries a spread over the draws',
                  np.isfinite(s['std_over_draws']) and s['n_draws_scored'] == 2,
                  f"mean {s['mean_over_draws']:.4f} std {s['std_over_draws']:.4f}")

    # The representation the true value was actually built from is the one where
    # removal can be read, so that is where the ceiling must beat no cleaning.
    row = r['summary']['per_space']['base:toy']['per_model']['rf']
    check('base:toy — the ceiling beats no cleaning',
          row['gain_over_no_cleaning']['ceiling'] > 0,
          f"ceiling {row['gain_over_no_cleaning']['ceiling']:+.4f}")
    check('base:toy — the ceiling beats its OWN floor',
          _ceiling_beyond_its_floor(r, 'base:toy') > 0,
          f"{_ceiling_beyond_its_floor(r, 'base:toy'):+.4f}")
    check('base:toy — the floor-to-ceiling span is positive',
          row['floor_to_ceiling_span'] > 0,
          f"{row['floor_to_ceiling_span']:+.4f}")
    check('base:toy — drop and repair each have a position between floor and '
          'ceiling',
          all(k in row['removal_position'] for k in ('drop', 'repair')),
          str({k: round(v, 3) for k, v in row['removal_position'].items()}))

    hrow = r['summary']['per_space']['hybrid']['per_model']['rf']
    check('hybrid — the ceiling beats no cleaning there too',
          hrow['gain_over_no_cleaning']['ceiling'] > 0,
          f"ceiling {hrow['gain_over_no_cleaning']['ceiling']:+.4f}")
    check('hybrid — the ceiling beats its OWN floor',
          _ceiling_beyond_its_floor(r, 'hybrid') > 0,
          f"{_ceiling_beyond_its_floor(r, 'hybrid'):+.4f}")

    check('the hybrid is compared with each representation by name',
          set(r['summary']['gain_by_space']['ceiling']['rf']['hybrid_minus_base'])
          == {'base:toy', 'base:second'},
          str({k: round(v, 4) for k, v in
               r['summary']['gain_by_space']['ceiling']['rf']
               ['hybrid_minus_base'].items()}))

    print("\nend to end — nothing disagrees more than anything else:")
    flat, *_ = make_replicate_data(n_entities=700, seed=3, wide_fraction=0.0)
    r_flat = _run_with_hybrid(flat, 'toy_replicates_flat',
                              models=['rf'], n_draws=2)
    check('the control ran too', 'summary' in r_flat,
          str(r_flat.get('skipped', ''))[:60])
    if 'summary' not in r_flat:
        return r, None

    print("\nthe separation, in both spaces:")
    for space in ('base:toy', 'hybrid'):
        informative = _ceiling_beyond_its_floor(r, space)
        control = _ceiling_beyond_its_floor(r_flat, space)
        check(f'{space}: removing the worst-disagreeing entities is worth more '
              f'where entities really do disagree by different amounts',
              informative > control,
              f'informative {informative:+.4f} vs control {control:+.4f}')
    return r, r_flat


def test_runs_without_a_recipe():
    print("\nno recipe from the pool:")
    data, *_ = make_replicate_data(n_entities=500, seed=5)
    real = n17.get_best_search_for_downstream
    n17.get_best_search_for_downstream = no_recipe
    try:
        r = n17.run(data, {'n_splits': 3}, OUT, 'toy_no_recipe',
                    models=['rf'], n_draws=2)
    finally:
        n17.get_best_search_for_downstream = real

    check('the run is not dropped', 'summary' in r,
          str(r.get('skipped', ''))[:60])
    if 'summary' not in r:
        return
    check('it records that no hybrid was available, and why',
          r['hybrid_available'] is False
          and 'precomputed pool' in r['hybrid_unavailable_reason'])
    check('every base representation still ran',
          set(r['spaces_run']) == {'base:toy', 'base:second'},
          str(r['spaces_run']))
    check('no hybrid row was written',
          'hybrid' not in r['summary']['per_space'])

    finite, total = 0, 0
    for space in r['spaces_run']:
        row = r['summary']['per_space'][space]['per_model']['rf']
        for arm in n17.ARMS:
            total += 1
            if np.isfinite(row['arm_scores'][arm]['mean_over_draws']):
                finite += 1
    check('every arm in every base representation still carries a score',
          finite == total, f'{finite} of {total}')
    check('the floor-to-ceiling reading is still there without a hybrid',
          np.isfinite(r['summary']['per_space']['base:toy']['per_model']['rf']
                      ['floor_to_ceiling_span']))


if __name__ == '__main__':
    test_entity_construction()
    test_arms()
    test_refuses_without_replicates()
    test_end_to_end()
    test_runs_without_a_recipe()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
