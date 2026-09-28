#!/usr/bin/env python3
"""Smoke for n19: the measurement count and spread as weight and as shrinkage.

The whole experiment rests on one claim: a label built from more measurements is
more precise, and using that helps. So the smoke builds datasets where the right
answer is KNOWN and different in each, and checks that n19 gives a different
answer in each.

  count_carries_precision  every molecule's measurements scatter by the SAME
                           amount, and only HOW MANY of them differs. So the
                           mean of six is six times as precise as the mean of
                           one, and weighting by the count is the right weight.
                           Weighting by the spread is not -- every spread is the
                           same.
  spread_carries_precision every molecule was measured the same NUMBER of times,
                           and only how far its measurements scatter differs. So
                           the count carries nothing and the spread carries
                           everything, which is the opposite way round.

If n19 cannot tell those two apart it is not measuring what it says it is.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# The out-of-fold prediction the two relabelling conditions shrink toward is a
# LightGBM fit on the training molecules of a fold. LightGBM splits its work
# across features, and this smoke's molecules carry four. MEASURED 2026-09-16 on
# 250 rows by 4 features: one thread 0.41s, every core 138.5s. So a smoke that
# left the thread count alone would spend its whole life in thread contention.
# The switch is the repository's own, read by kirby/hybrid.py:78 and now by
# kirby/utils/noisy_labels/core.py.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
# The similarity sweep and the noise cache, both added 2026-09-25. A smoke runs
# today's behaviour -- neighbours in the model-input matrix only -- unless the
# caller names settings in KIRBY_SIMILARITY_SETTINGS, and it never reads or
# writes $KIRBY_CACHE_DIR/noise_cache: a cached build would hide the spy.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
os.environ.setdefault('KIRBY_NOISE_CACHE', '0')

import tempfile

import numpy as np

import experiments.noise.n17_replicate_removal as n17
import experiments.noise.n19_measurement_count as n19

OUT = tempfile.mkdtemp(prefix='n19_smoke_')

# ⛔ THE POOL LOOKUP IS STUBBED. There is no precomputed pool for these toy
# datasets, and the real lookup then runs a cold search over them. MEASURED
# 2026-09-25: that search reached a DNN fit (kirby/models/dnn.py:194) and hung
# for over four hours at 0% CPU. The single representations are named through
# `base_reps`, so nothing here needs the stored search; every run is in base_1
# (and base_2 where two are named) only, as _smoke_n23 does.
n19.get_best_search_for_downstream = lambda *a, **k: (None, None)
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def _rows(X_ent, truth, sigma, ks, rng):
    X_rows, y_rows, gids = [], [], []
    for j in range(len(ks)):
        for _ in range(int(ks[j])):
            X_rows.append(X_ent[j])
            y_rows.append(truth[j] + rng.normal(0, sigma[j]))
            gids.append(j)
    gids = np.asarray(gids)
    return {
        'labels': np.asarray(y_rows, dtype=float),
        'reps': {'toy': np.asarray(X_rows, dtype=float)},
        'replicate_groups': gids,
        'groups': gids,
        'task_type': 'regression',
        'metric': 'rmse',
    }


def count_carries_precision(n_ent=400, n_feat=4, seed=0):
    """Same scatter for every molecule; only the measurement count differs.

    So the count IS the precision, but only for a label built by averaging.
    Every molecule scatters by 4 against a signal whose standard deviation is
    about 2, and the counts run 4, 8, 20 and 60 -- so an averaged label carries
    an error variance of 4 at the low end and 0.27 at the high end, a spread of
    fifteen times. A one-measurement label carries 16 whatever the count, which
    is the thing the two label builds exist to separate.
    """
    rng = np.random.default_rng(seed)
    X_ent = rng.normal(size=(n_ent, n_feat))
    truth = X_ent @ rng.normal(size=n_feat)
    sigma = np.full(n_ent, 4.0)
    ks = rng.choice([4, 8, 20, 60], size=n_ent, p=[.30, .30, .25, .15])
    return _rows(X_ent, truth, sigma, ks, rng), ks, sigma


def spread_carries_precision(n_ent=400, n_feat=4, seed=1):
    """Same measurement count for every molecule; only the scatter differs.

    Thirty measurements each, so the count carries nothing and the CONSENSUS
    every molecule is graded against is clean whichever laboratory measured it.
    Three in ten molecules came from a laboratory that scatters by 5 against a
    signal standard deviation of about 2; the rest scatter by 0.2. So a
    one-measurement training label is worthless for those three in ten and
    almost exact for the rest, while every grading label is good -- which is
    what lets a weight be read at all.
    """
    rng = np.random.default_rng(seed)
    X_ent = rng.normal(size=(n_ent, n_feat))
    truth = X_ent @ rng.normal(size=n_feat)
    wide = rng.random(n_ent) < 0.30
    sigma = np.where(wide, 5.0, 0.2)
    ks = np.full(n_ent, 30)
    return _rows(X_ent, truth, sigma, ks, rng), ks, sigma


def test_stats_line_up_with_n17():
    print("measurement_stats against build_entities:")
    data, ks, sigma = count_carries_precision(n_ent=200)
    stats = n19.measurement_stats(data['labels'], data['replicate_groups'])
    ent = n17.build_entities(data['labels'], data['replicate_groups'], seed=0)

    check('same number of molecules',
          len(stats['n_measurements']) == len(ent['n_measurements']))
    check('same measurement count per molecule, in the same order',
          np.array_equal(stats['n_measurements'], ent['n_measurements']))
    check('same representative row per molecule',
          np.array_equal(stats['row_index'], ent['row_index']))
    check('the counts are the ones the data was built with',
          np.array_equal(np.sort(stats['n_measurements']), np.sort(ks)))

    # var_all is the PER-MEASUREMENT variance, so it must track sigma and NOT
    # the count. Everything downstream divides it by the count itself.
    v = stats['var_all']
    check('every molecule measured twice or more has a variance',
          np.all(np.isfinite(v[stats['n_measurements'] >= 2])))
    built = float(np.mean(sigma ** 2))
    check('the measured variance sits near the variance it was built with',
          abs(float(np.nanmean(v)) - built) < 0.15 * built,
          f"measured {np.nanmean(v):.3f} vs built {built:.3f}")


def test_label_builds():
    print("\ntraining labels:")
    data, ks, sigma = count_carries_precision(n_ent=150)
    stats = n19.measurement_stats(data['labels'], data['replicate_groups'])
    ent = n17.build_entities(data['labels'], data['replicate_groups'], seed=0)

    y_single, n_single = n19.training_labels(ent, stats, 'single')
    y_avg, n_avg = n19.training_labels(ent, stats, 'averaged')

    check('a single label rests on one measurement where the molecule is gradeable',
          np.all(n_single[ent['gradeable']] == 1.0))
    check('and on all of them where it is not, because there it IS the mean',
          np.array_equal(n_single[~ent['gradeable']],
                         stats['n_measurements'][~ent['gradeable']].astype(float)),
          f"{int((~ent['gradeable']).sum())} ungradeable molecules")
    check('an averaged label rests on every one of them',
          np.array_equal(n_avg, stats['n_measurements'].astype(float)))
    check('the averaged label is the mean of that molecule\'s measurements',
          np.allclose(y_avg, stats['mean_all']))

    rows_of = {}
    for i, g in enumerate(data['replicate_groups']):
        rows_of.setdefault(g, []).append(i)
    member = all(np.any(np.isclose(data['labels'][rows_of[g]], y_single[j]))
                 for j, g in enumerate(ent['group_id']))
    check('a single label is one of that molecule\'s own measurements', member)

    # The point of the two builds: precision falls with the count for one and
    # not for the other.
    var_single, _, _ = n19.label_variance(stats, n_single)
    var_avg, _, _ = n19.label_variance(stats, n_avg)
    k = stats['n_measurements']
    hi, lo = k >= 20, k <= 8
    check('a single label is no more precise when the molecule was measured more',
          abs(float(np.mean(var_single[hi]) - np.mean(var_single[lo])))
          < 0.25 * float(np.mean(var_single)),
          f"{np.mean(var_single[hi]):.3f} at 20+ vs "
          f"{np.mean(var_single[lo]):.3f} at 8 or fewer")
    check('an averaged label IS more precise when the molecule was measured more',
          float(np.mean(var_avg[hi])) < 0.5 * float(np.mean(var_avg[lo])),
          f"{np.mean(var_avg[hi]):.3f} at 20+ vs "
          f"{np.mean(var_avg[lo]):.3f} at 8 or fewer")


def test_variance_fill_and_calibration():
    print("\nfilling and calibration:")
    data, ks, sigma = count_carries_precision(n_ent=200)
    stats = n19.measurement_stats(data['labels'], data['replicate_groups'])
    # Force some molecules to have been measured once, so there is something
    # to fill.
    stats['var_all'][:30] = np.nan
    stats['n_measurements'][:30] = 1
    n_behind = np.ones(len(stats['var_all']))

    var, n_filled, floor = n19.label_variance(stats, n_behind)
    check('every molecule with no variance of its own is filled',
          n_filled == 30 and np.all(np.isfinite(var)), f'{n_filled} filled')
    check('nothing is left at zero, so no weight is infinite',
          float(np.min(var)) >= floor, f'floor {floor:.2e}')

    var_cal, fit, n_trusted = n19.calibrated_variance(stats, n_behind)
    check('the noise function is fitted only on the well-measured molecules',
          n_trusted == int((stats['n_measurements'] >= 3).sum()),
          f'{n_trusted} molecules')
    check('the noise function carries one entry per measurement count it saw',
          set(fit) == {int(c) for c in
                       np.unique(stats['n_measurements'][
                           stats['n_measurements'] >= 3])},
          str(sorted(fit)))
    check('a molecule measured once still gets a variance from the curve',
          np.all(np.isfinite(var_cal[:30])))

    # A molecule measured once takes the nearest count the trusted subset saw.
    nearest = min(fit)
    check('a molecule measured once takes the smallest fitted count',
          np.allclose(var_cal[:30], fit[nearest]),
          f'count {nearest}, variance {fit[nearest]:.3f}')


def test_weights():
    print("\nweights:")
    k = np.array([1., 2., 4., 8.])
    n_behind = k.copy()
    per_measurement = np.array([1.0, 1.0, 4.0, 1.0])
    variance = per_measurement / n_behind

    w_count = n19.weights_for('count_weighted', k, variance, n_behind)
    w_spread = n19.weights_for('spread_weighted', k, variance, n_behind)
    w_prec = n19.weights_for('precision_weighted', k, variance, n_behind)

    for name, w in (('count', w_count), ('spread', w_spread),
                    ('precision', w_prec)):
        check(f'{name} weights average one',
              abs(float(np.mean(w)) - 1.0) < 1e-12, f'{np.mean(w):.6f}')

    check('the count weight follows the count and nothing else',
          np.allclose(w_count / w_count[0], k / k[0]))
    check('the spread weight ignores the count',
          np.allclose(w_spread / w_spread[0],
                      per_measurement[0] / per_measurement))
    check('the precision weight is the count over the spread',
          np.allclose(w_prec / w_prec[0],
                      (k / per_measurement) / (k[0] / per_measurement[0])))
    check('the molecule whose measurements disagree most is weighted least',
          int(np.argmin(w_spread)) == 2, f'lightest is molecule {np.argmin(w_spread)}')


def test_shrinkage():
    print("\nshrinkage:")
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    oof = y + rng.normal(0, 0.5, size=300)

    tiny = np.full(300, 1e-9)
    y_t, s_t, _ = n19.shrink_toward(y, oof, tiny)
    check('a precise label barely moves',
          np.allclose(y_t, y, atol=1e-6) and float(np.max(s_t)) < 1e-6,
          f'largest shrinkage {np.max(s_t):.2e}')

    huge = np.full(300, 1e6)
    y_h, s_h, _ = n19.shrink_toward(y, oof, huge)
    check('a hopeless label moves almost all the way to the prediction',
          np.allclose(y_h, oof, atol=1e-3) and float(np.min(s_h)) > 0.999,
          f'smallest shrinkage {np.min(s_h):.6f}')

    var = np.where(np.arange(300) < 150, 1e-9, 1e6)
    _, s_mixed, tau2 = n19.shrink_toward(y, oof, var)
    check('a less precise label moves further than a precise one',
          float(np.mean(s_mixed[150:])) > float(np.mean(s_mixed[:150])),
          f'{np.mean(s_mixed[150:]):.4f} vs {np.mean(s_mixed[:150]):.4f}')
    check('tau2 is how far the labels sit from the prediction',
          abs(tau2 - float(np.var(y - oof))) < 1e-12)


def test_refusals():
    print("\nrefusals:")
    rng = np.random.default_rng(0)
    bare = {'labels': rng.normal(size=200),
            'reps': {'toy': rng.normal(size=(200, 4))},
            'groups': np.arange(200), 'task_type': 'regression',
            'metric': 'rmse'}
    r = n19.run(bare, {'n_splits': 3}, OUT, 'toy_no_replicates')
    check('a dataset with no repeat structure is refused, not invented',
          r.get('skipped') == 'no_replicate_structure', str(r)[:90])

    data, *_ = count_carries_precision(n_ent=60)
    data['task_type'] = 'classification'
    r2 = n19.run(data, {'n_splits': 3}, OUT, 'toy_clf')
    check('a classification dataset is refused',
          r2.get('skipped') == 'not_regression', str(r2)[:90])


def _cell(res, build, cond, model='rf', space='base_1'):
    return (res['summary']['per_label_build'][build]['per_space'][space]
            ['per_model'][model]['per_condition'][cond])


def test_end_to_end():
    print("\nend to end — the count carries the precision:")
    data, ks, sigma = count_carries_precision(n_ent=400, seed=3)
    r = n19.run(data, {'n_splits': 4}, OUT, 'toy_count',
                models=['rf'], base_reps={'rf': ['toy']}, n_draws=2, oof_splits=3,
                model_arms=[])
    check('it ran', 'summary' in r, str(r.get('skipped', ''))[:60])
    if 'summary' not in r:
        return

    check('it says the weights never reached the feature selection',
          r['weights_reach_feature_selection'] is False)
    check('it says the recipe never saw a clean label column',
          r['recipe_saw_clean_labels'] is False)
    check('every comparison is paired per fold',
          r['summary']['comparison'] == 'paired per fold')

    u = _cell(r, 'averaged', 'untouched')
    check('untouched gains nothing against itself and wins no fold',
          abs(u['mean_gain']) < 1e-12 and u['folds_won'] == 0,
          f"mean {u['mean_gain']:+.2e}, won {u['folds_won']}")
    check('untouched is paired with every fold it scored',
          u['n_folds_paired'] == u['n_folds'] == 8,
          f"{u['n_folds_paired']} paired of {u['n_folds']}")

    # An AVERAGED label's precision IS the measurement count here, so the two
    # weights that read the count should win most folds.
    for cond in ('count_weighted', 'precision_weighted'):
        c = _cell(r, 'averaged', cond)
        check(f'{cond} wins most folds when the count IS the precision',
              c['folds_won'] * 2 > c['n_folds_paired'],
              f"{c['folds_won']} of {c['n_folds_paired']}, "
              f"mean {c['mean_gain']:+.4f}")

    print("\nend to end — the spread carries the precision:")
    data2, ks2, sigma2 = spread_carries_precision(n_ent=400, seed=4)
    r2 = n19.run(data2, {'n_splits': 4}, OUT, 'toy_spread',
                 models=['rf'], base_reps={'rf': ['toy']}, n_draws=2, oof_splits=3,
                model_arms=[])
    check('it ran', 'summary' in r2, str(r2.get('skipped', ''))[:60])
    if 'summary' not in r2:
        return

    # Every molecule was measured the same number of times, so a weight read off
    # the count is the same number for every molecule and the fit is unchanged.
    for build in ('single', 'averaged'):
        c = _cell(r2, build, 'count_weighted')
        check(f'weighting by the count is the identical fit on the {build} '
              f'label when every count is the same',
              c['folds_tied'] == c['n_folds_paired'] and c['folds_won'] == 0,
              f"tied {c['folds_tied']} of {c['n_folds_paired']}, "
              f"won {c['folds_won']}, mean {c['mean_gain']:+.2e}")

    # A ONE-MEASUREMENT label from the wide laboratory is worthless and from the
    # narrow one is nearly exact, so the spread is the whole of the precision.
    for cond in ('spread_weighted', 'precision_weighted'):
        c = _cell(r2, 'single', cond)
        check(f'{cond} wins most folds when the spread IS the precision',
              c['folds_won'] * 2 > c['n_folds_paired'],
              f"{c['folds_won']} of {c['n_folds_paired']}, "
              f"mean {c['mean_gain']:+.4f}")

    # The whole check: the two datasets put the two weights the opposite way
    # round, so n19 is reading the data and not the arm's name.
    count_on_count = _cell(r, 'averaged', 'count_weighted')['folds_won']
    count_on_spread = _cell(r2, 'single', 'count_weighted')['folds_won']
    spread_on_spread = _cell(r2, 'single', 'spread_weighted')['folds_won']
    check('the two datasets give opposite answers, which is the whole check',
          count_on_count > count_on_spread and spread_on_spread > count_on_spread,
          f'count weight wins {count_on_count} folds where the count is the '
          f'precision and {count_on_spread} where it is not; the spread weight '
          f'wins {spread_on_spread} there')

    for cond in ('shrunk', 'calibrated'):
        c = _cell(r2, 'single', cond)
        check(f'{cond} produced a score',
              np.isfinite(c['mean']) and c['n_folds'] > 0,
              f"{c['mean']:.4f} over {c['n_folds']} folds")

    # Fitted per outer fold on its training molecules since 2026-09-28; the
    # first fold's curve is read here.
    def _first_fit(res):
        per_fold = res['per_draw']['0']['per_label_build']['averaged'][
            'variance_fit_per_fold']
        return per_fold[sorted(per_fold)[0]][
            'fitted_variance_by_measurement_count']
    check('the fitted noise function is written out',
          bool(_first_fit(r2)), str(_first_fit(r2)))
    fitc = _first_fit(r)
    check('the noise function is flat in the count when the scatter is flat',
          max(fitc.values()) < 2.0 * min(fitc.values()),
          str({k: round(v, 2) for k, v in fitc.items()}))
    # Every molecule here was measured the same number of times, so every
    # single label rests on one measurement and 1/count does not vary. The
    # three conditions that need the slope have nothing to read it from.
    check('every condition that needs no fitted slope carries the spread of '
          'its own per-fold gains',
          all(np.isfinite(_cell(r2, 'single', c)['spread_over_fold_gains'])
              for c in n19.CONDITIONS
              if c not in n19.FITTED_VARIANCE_CONDITIONS))
    for c in n19.FITTED_VARIANCE_CONDITIONS:
        check(f'{c} is absent, not faked, when 1/count does not vary',
              _cell(r2, 'single', c)['n_folds'] == 0,
              f"{_cell(r2, 'single', c)['n_folds']} folds")
    errs = r2['per_draw']['0']['per_label_build']['single'].get('_errors', {})
    check('and the reason is written down',
          any(k.endswith('noise_variance') for k in errs), str(list(errs))[:120])


def test_fit_noise_variance():
    print("\nsigma_u^2 and sigma_e^2 from squared error against 1/count:")
    rng = np.random.default_rng(7)
    n = 20000
    k = rng.integers(1, 11, size=n).astype(float)
    truth = rng.normal(size=n)
    # sigma_u^2 = 0.5, sigma_e^2 = 2.0 by construction.
    y = truth + rng.normal(0, np.sqrt(0.5), n) + rng.normal(0, np.sqrt(2.0 / k))
    s2u, s2e, info = n19.fit_noise_variance(y, truth, k)
    check('the slope recovers the variance of one measurement',
          abs(s2e - 2.0) < 0.2, f'{s2e:.3f} against 2.0')
    check('the intercept recovers the error left at infinite count',
          abs(s2u - 0.5) < 0.1, f'{s2u:.3f} against 0.5')
    none_u, none_e, info2 = n19.fit_noise_variance(y, truth, np.ones(n))
    check('one count everywhere gives no slope, and says why',
          none_u is None and none_e is None and 'reason' in info2)
    # Error that GROWS with the count: the slope is negative and is floored.
    y_bad = truth + rng.normal(0, np.sqrt(0.2 * k))
    u3, e3, info3 = n19.fit_noise_variance(y_bad, truth, k)
    check('a negative slope is floored to zero and the floor is recorded',
          e3 == 0.0 and info3['slope_floored_to_zero'] is True,
          f"fitted slope {info3['slope_fitted']:.3f}")

    w = n19.inverse_variance_weights(0.5, 2.0 / k)
    check('the inverse-variance weights average one',
          abs(w.mean() - 1.0) < 1e-12)
    ratio = w[k == 10].mean() / w[k == 1].mean()
    check('ten measurements weigh (0.5 + 2) / (0.5 + 0.2) times one, not ten',
          abs(ratio - 2.5 / 0.7) < 1e-9, f'{ratio:.4f}')

    var = np.array([1.0, 4.0, np.nan, 0.25])
    cnt = np.array([2, 11, 1, 5])
    pulled, pooled = n19.pooled_spread_variance(var, cnt, prior=3)
    want_pooled = (1 * 1.0 + 10 * 4.0 + 4 * 0.25) / 15
    check('the pooled variance weights each molecule by count - 1',
          abs(pooled - want_pooled) < 1e-12, f'{pooled:.4f}')
    check('a molecule measured once gets the pooled variance',
          abs(pulled[2] - pooled) < 1e-12)
    check('ten degrees of freedom sit closer to their own variance than one',
          abs(pulled[1] - 4.0) / abs(pooled - 4.0)
          < abs(pulled[0] - 1.0) / abs(pooled - 1.0))

    X = rng.normal(size=(50, 3))
    a, b = n19.with_count_feature(X[:40], X[40:], rng.integers(1, 6, 40),
                                  rng.integers(1, 6, 10))
    check('the count feature adds exactly one feature to each side',
          a.shape == (40, 4) and b.shape == (10, 4))
    check('and is standardised on the training molecules',
          abs(a[:, -1].mean()) < 1e-9)


def test_new_conditions_end_to_end():
    print("\nend to end — the four added conditions and the model arms:")
    data, ks, sigma = count_carries_precision(n_ent=240, seed=5)
    # A second representation, so base_2 runs too: the features plus noise.
    _rng = np.random.default_rng(55)
    data['reps']['toy_noisy'] = (data['reps']['toy']
                                 + _rng.normal(0, 1.0, data['reps']['toy'].shape))
    r = n19.run(data, {'n_splits': 3}, OUT, 'toy_new_arms',
                models=['rf'], base_reps={'rf': ['toy', 'toy_noisy']},
                n_draws=1, oof_splits=3, label_builds=['averaged'])
    check('it ran', 'summary' in r, str(r.get('skipped', ''))[:60])
    if 'summary' not in r:
        return
    check('the single-representation comparison is the two named, in order',
          r['spaces_run'][:2] == ['base_1', 'base_2']
          and r['base_reps_by_model'] == {'rf': ['toy', 'toy_noisy']}
          and 'base_rep' not in r,
          f"{r['spaces_run']}, {r.get('base_reps_by_model')}")
    check('the model arms name the representation they ran on in each',
          r['model_arms_base_reps'] == {'base_1': 'toy', 'base_2': 'toy_noisy'},
          str(r.get('model_arms_base_reps')))
    u1 = _cell(r, 'averaged', 'untouched', space='base_1')['mean']
    u2 = _cell(r, 'averaged', 'untouched', space='base_2')['mean']
    check('base_1 and base_2 are different feature matrices',
          np.isfinite(u1) and np.isfinite(u2) and abs(u1 - u2) > 1e-9,
          f'untouched {u1:.4f} in base_1, {u2:.4f} in base_2')
    for cond in ('invvar_weighted', 'count_feature',
                 'invvar_and_count_feature', 'spread_pooled_weighted'):
        c = _cell(r, 'averaged', cond)
        check(f'{cond} scored every fold untouched scored',
              c['n_folds_paired'] == 3, f"{c['n_folds_paired']} of 3")
    fits = (r['per_draw']['0']['per_label_build']['averaged']['per_space']
            ['base_1']['fitted_noise_variance'])
    slopes = [f['slope_fitted'] for f in fits]
    # Every measurement scatters by 4, so one measurement carries variance 16
    # and the slope should sit near it. Out-of-fold error from 160 molecules is
    # rough, so the check is the order of magnitude, not the digits.
    check('one fit per fold is written out', len(fits) == 3)
    check('the fitted slope sits near the variance the data was built with',
          all(4.0 < s < 64.0 for s in slopes),
          f"{[round(s, 1) for s in slopes]} against 16")
    arms = r['summary']['model_arms']['per_label_build']['averaged']['base_1']
    for arm, plain in n19.MODEL_ARM_PAIRS.items():
        check(f'{arm} is paired against {plain} on every fold',
              arms.get(arm, {}).get('n_folds_paired') == 3,
              str(arms.get(arm, {}).get('n_folds_paired')))
    arms2 = r['summary']['model_arms']['per_label_build']['averaged']['base_2']
    check('the model arms ran in base_2 too',
          all(arms2.get(a, {}).get('n_folds_paired') == 3
              for a in n19.MODEL_ARM_PAIRS),
          str({a: arms2.get(a, {}).get('n_folds_paired')
               for a in n19.MODEL_ARM_PAIRS}))
    check('nothing in the model arms is paired against a downstream model',
          all(v['against'] in n19.MODEL_ARMS for v in arms.values()))
    bb = r['per_draw']['0']['per_label_build']['averaged']
    nz = bb.get('n_molecules_with_interval_width_above_zero')
    check('the interval arm says how many molecules have a real interval',
          isinstance(nz, int) and 0 < nz <= 240, str(nz))

    try:
        n19.run(data, {'n_splits': 3}, OUT, 'toy_bad_arm', models=['rf'],
                n_draws=1, model_arms=['svm_plain'])
        check('an unknown model arm is refused', False)
    except ValueError:
        check('an unknown model arm is refused', True)


def test_interval_aft():
    print("\ninterval AFT (2026-09-24):")
    labels = np.array([1.0, 3.0, 2.0, 5.0, 5.0, 4.0])
    groups = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
    st = n19.measurement_stats(labels, groups)
    check('min and max per molecule, in entity order',
          np.allclose(st['min_all'], [1, 2, 4]) and np.allclose(st['max_all'], [3, 2, 5]))
    ent = {'gradeable': np.array([False, False, True])}
    y_single = np.array([2.0, 2.0, 5.0])
    lo, hi = n19.interval_bounds(ent, st, 'single', y_single)
    check('single: a gradeable molecule rests on one measurement, zero width',
          lo[2] == hi[2] == 5.0)
    check('single: the others span all their measurements',
          lo[0] == 1.0 and hi[0] == 3.0)
    lo_a, hi_a = n19.interval_bounds(ent, st, 'averaged', st['mean_all'])
    check('averaged: every interval contains the training label',
          np.all(lo_a <= st['mean_all']) and np.all(st['mean_all'] <= hi_a))

    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    y = 2 * X[:, 0] + rng.normal(0, 0.3, 300)
    pred = n19.fit_interval_aft_predict(X[:200], y[:200], y[:200], X[200:], 0.3)
    r = float(np.corrcoef(pred, y[200:])[0, 1])
    check('point-interval AFT predicts on the label scale',
          r > 0.8 and abs(pred.mean() - y[200:].mean()) < 0.5,
          f'pearson {r:.3f}, mean {pred.mean():.3f} vs {y[200:].mean():.3f}')
    pw = n19.fit_interval_aft_predict(X[:200], y[:200] - 1, y[:200] + 1,
                                      X[200:], 0.3)
    check('a symmetric interval keeps predictions near the centre',
          abs(pw.mean() - pred.mean()) < 0.5,
          f'{pw.mean():.3f} vs {pred.mean():.3f}')


def test_interval_aft_skip_under_single():
    print("\ninterval AFT is not fitted where it would repeat the point arm:")
    aft = ['xgboost_aft_point', 'xgboost_interval_aft']
    # Every molecule measured 4 or more times, so every one is gradeable and
    # every single label rests on one measurement: zero width everywhere.
    data, ks, _ = count_carries_precision(n_ent=180, seed=6)
    r = n19.run(data, {'n_splits': 3}, OUT, 'toy_aft_skip', models=['rf'],
                base_reps={'rf': ['toy']}, n_draws=1, oof_splits=3, model_arms=aft,
                label_builds=['single', 'averaged'])
    check('it ran', 'summary' in r, str(r.get('skipped', ''))[:60])
    if 'summary' not in r:
        return
    blk_s = r['per_draw']['0']['per_label_build']['single']
    skips = blk_s.get('model_arm_skips', {}).get('xgboost_interval_aft', {})
    check('single: a skip record for every fold scored',
          len(skips) == blk_s['n_folds_scored'] == 3, f'{len(skips)} records')
    check('single: each record carries a reason and the measured count, '
          'which is every training molecule',
          all(v.get('reason') and
              v['n_training_molecules_with_zero_width_interval']
              == v['n_training_molecules'] > 0 for v in skips.values()),
          str([(v['n_training_molecules_with_zero_width_interval'],
                v['n_training_molecules']) for v in skips.values()]))
    arms_s = blk_s['per_space']['base_1']['model_arms']
    check('single: the interval arm has no score, the point arm has all',
          arms_s['xgboost_interval_aft']['n_folds'] == 0
          and arms_s['xgboost_aft_point']['n_folds'] == 3)
    row_s = r['summary']['model_arms']['per_label_build']['single']['base_1'][
        'xgboost_interval_aft']
    check('single: the summary pairs nothing and counts the skipped folds',
          row_s['n_folds_paired'] == 0 and row_s['n_folds_skipped'] == 3
          and not np.isfinite(row_s['mean_gain']), str(row_s)[:120])

    blk_a = r['per_draw']['0']['per_label_build']['averaged']
    check('averaged: no skip record',
          not blk_a.get('model_arm_skips'), str(blk_a.get('model_arm_skips')))
    row_a = r['summary']['model_arms']['per_label_build']['averaged']['base_1'][
        'xgboost_interval_aft']
    check('averaged: the interval arm is still fitted and paired on every fold',
          row_a['n_folds_paired'] == 3 and row_a['n_folds_skipped'] == 0,
          f"{row_a['n_folds_paired']} paired")

    # Some molecules measured twice: they cannot be graded, keep both
    # measurements under single, and give a real interval. The arm must run.
    rng = np.random.default_rng(8)
    n_ent = 180
    X_ent = rng.normal(size=(n_ent, 4))
    truth = X_ent @ rng.normal(size=4)
    ks2 = rng.choice([2, 4, 8], size=n_ent, p=[.4, .3, .3])
    data2 = _rows(X_ent, truth, np.full(n_ent, 1.0), ks2, rng)
    r2 = n19.run(data2, {'n_splits': 3}, OUT, 'toy_aft_twice', models=['rf'],
                 base_reps={'rf': ['toy']}, n_draws=1, oof_splits=3, model_arms=aft,
                 label_builds=['single'])
    check('it ran with molecules measured twice', 'summary' in r2,
          str(r2.get('skipped', ''))[:60])
    if 'summary' not in r2:
        return
    blk2 = r2['per_draw']['0']['per_label_build']['single']
    row2 = r2['summary']['model_arms']['per_label_build']['single']['base_1'][
        'xgboost_interval_aft']
    check('single with real intervals: no skip, paired on every fold',
          not blk2.get('model_arm_skips') and row2['n_folds_paired'] == 3,
          f"{row2['n_folds_paired']} paired, "
          f"{blk2['n_molecules_with_interval_width_above_zero']} molecules "
          f"with a real interval")


if __name__ == '__main__':
    test_interval_aft()
    test_fit_noise_variance()
    test_new_conditions_end_to_end()
    test_interval_aft_skip_under_single()
    test_stats_line_up_with_n17()
    test_label_builds()
    test_variance_fill_and_calibration()
    test_weights()
    test_shrinkage()
    test_refusals()
    test_end_to_end()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
