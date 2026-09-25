#!/usr/bin/env python3
"""Smoke for n21: which distance are the nearest neighbours found in.

Six things it checks, in order.

  1. The suspicion score n21 computes from a neighbour list reproduces
     `NeighborResidualEstimator` and `LocalVarianceRatioEstimator` to the last
     bit, under euclidean and manhattan and under both task types. n21 carries
     its own copy of that arithmetic because a forest proximity is a pairwise
     distance and those classes take a feature space; this is what stops the two
     copies drifting apart.
  2. n21's feature weighting at exponent 0.5 IS n12's `_weight_features`, the
     same numbers. So the cell labelled `imp_pow0.5` is n12's space and the
     comparison against `imp_pow1` is the comparison n12's defect 1 needs.
  3. The forest proximity distance is what it says: one minus the share of trees
     in which two molecules land in the same leaf. Checked against a brute-force
     recount, and a molecule never appears in its own neighbour list even when
     another molecule sits at distance zero from it.
  4. THE HEADLINE. On a set where two of twenty features carry the whole signal
     and the other eighteen are noise, the importance-weighted and
     forest-proximity distances find the corrupted molecules better than the
     plain one.
  5. And on a set where every one of the twenty features matters equally, that
     advantage is gone.
  6. The two guards: a condition where every molecule's label moves is refused,
     and level 0.0 is dropped with the reason recorded.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
# The similarity sweep and the noise cache, both added 2026-09-25. A smoke runs
# today's behaviour -- neighbours in the model-input matrix only -- unless the
# caller names settings in KIRBY_SIMILARITY_SETTINGS, and it never reads or
# writes $KIRBY_CACHE_DIR/noise_cache: a cached build would hide the spy.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
os.environ.setdefault('KIRBY_NOISE_CACHE', '0')

import tempfile
import warnings

import numpy as np

warnings.filterwarnings('ignore')

import experiments.noise.n21_neighbour_distance as n21
import experiments.noise.n12_importance_weighted_distance as n12
from kirby.utils.noise_estimation import (
    NeighborResidualEstimator, LocalVarianceRatioEstimator,
)

OUT = tempfile.mkdtemp(prefix='n21_smoke_')
FAIL = []

# What the end-to-end runs sweep. One importance method (treeshap adds nothing
# the smoke checks and costs minutes), one level, both metrics, both exponents,
# both detectors, one model.
CHEAP = {
    'importance_methods': ['lightgbm_gain'],
    'k_values': [5, 10],
    'models': ['rf'],
    'levels': [0.5],
    'include_hybrid': False,
    'forest_trees': 60,
    # The single representation is named here, because there is no stored
    # search to rank one from. n21 takes the first downstream model's top 2;
    # with one representation there is only base_1.
    'base_reps': {'rf': ['ecfp4']},
}


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=400, seed=0, signal='sparse'):
    """Twenty features, and a label built from two of them or from all twenty.

    signal='sparse'  features 0 and 1 carry the whole signal; 2 to 19 are noise
                     that no label depends on.
    signal='equal'   all twenty carry the same weight.

    The grouping is twelve molecules per group, so a group-aware split has
    enough groups to work with. Same shape `_smoke_n20.make_data` uses.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 20))
    if signal == 'sparse':
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.1, n)
    elif signal == 'equal':
        y = X.sum(axis=1) * (3.6 / np.sqrt(20.0)) + rng.normal(0, 0.1, n)
    else:
        raise ValueError(signal)
    return {
        'labels': y,
        'reps': {'ecfp4': X},
        'groups': np.arange(n) // 12,
        'task_type': 'regression',
        'metric': 'rmse',
        'dataset_name': 'n21smoke',
    }


# ---------------------------------------------------------------------------

def test_transcription_matches_the_estimators():
    print("n21's suspicion score against the two estimator classes:")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(150, 12))
    y_reg = 2.0 * X[:, 0] + rng.normal(0, 0.3, 150)
    y_clf = (y_reg > np.median(y_reg)).astype(int)

    pairs = (
        ('neighbor_residual', NeighborResidualEstimator),
        ('local_variance_ratio', LocalVarianceRatioEstimator),
    )
    for metric in ('euclidean', 'manhattan'):
        for detector, cls in pairs:
            for is_clf, y in ((False, y_reg), (True, y_clf)):
                for k in (5, 10):
                    k_eff = int(min(k, len(y) - 1))
                    theirs = cls(k=k, metric=metric,
                                 is_clf=is_clf).estimate(X, y)
                    nbrs = n21.neighbours_in_space(X, k_eff, metric)
                    mine = n21.suspicion_from_neighbours(nbrs, y, detector,
                                                         is_clf)
                    same = (theirs.shape == mine.shape
                            and np.array_equal(theirs, mine))
                    check(f'{detector} k{k} {metric} '
                          f'{"classification" if is_clf else "regression"}',
                          same,
                          '' if same else
                          f'max difference {np.max(np.abs(theirs - mine)):.3e}')


def test_weighting_is_n12s_at_exponent_half():
    print("\nthe weighting at exponent 0.5 against n12's:")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 25))
    for name, imp in (
        ('ordinary importance', rng.random(25)),
        ('some features at zero', np.where(rng.random(25) < 0.4, 0.0,
                                           rng.random(25))),
        ('every feature at zero', np.zeros(25)),
        ('a negative entry', np.concatenate([[-1.0], rng.random(24)])),
    ):
        theirs = n12._weight_features(X, imp)
        mine = n21.weight_features(X, imp, 0.5)
        check(f'{name}: same matrix as n12',
              theirs.shape == mine.shape and np.array_equal(theirs, mine))

    # And exponent 1.0 is a different space wherever the importance is not flat,
    # or the two cells the convention question compares would be the same cell.
    imp = rng.random(25)
    half = n21.weight_features(X, imp, 0.5)
    one = n21.weight_features(X, imp, 1.0)
    check('exponent 1.0 is a different space from exponent 0.5',
          not np.allclose(half, one))
    check('a flat importance gives the same neighbours at either exponent',
          np.array_equal(
              n21.neighbours_in_space(
                  n21.weight_features(X, np.ones(25), 0.5), 5, 'manhattan'),
              n21.neighbours_in_space(
                  n21.weight_features(X, np.ones(25), 1.0), 5, 'manhattan')))


def test_forest_proximity_is_what_it_says():
    print("\nthe forest proximity distance:")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 6))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.2, 80)
    leaves = n21.forest_leaf_ids(X, y, is_clf=False, n_trees=25,
                                 min_samples_leaf=5, seed=0)
    check('one leaf id per tree per molecule',
          leaves.shape == (80, 25), str(leaves.shape))

    D = n21.forest_proximity_distance(leaves, chunk=17)
    brute = np.empty((80, 80))
    for i in range(80):
        for j in range(80):
            brute[i, j] = 1.0 - np.mean(leaves[i] == leaves[j])
    check('it is one minus the share of trees that put the two in one leaf',
          np.allclose(D, brute, atol=1e-6),
          f'largest difference {np.max(np.abs(D - brute)):.3e}')
    check('a molecule is at distance zero from itself',
          np.allclose(np.diag(D), 0.0))
    check('it is symmetric', np.allclose(D, D.T))
    check('every entry sits between 0 and 1',
          float(D.min()) >= 0.0 and float(D.max()) <= 1.0,
          f'{D.min():.3f} to {D.max():.3f}')
    # The chunk boundary must not change a single entry.
    check('the chunk size changes nothing',
          np.array_equal(D, n21.forest_proximity_distance(leaves, chunk=80)))

    # A molecule never takes itself as a neighbour, even when another molecule
    # is also at distance zero from it. Two identical rows produce exactly that.
    leaves_tied = leaves.copy()
    leaves_tied[3] = leaves_tied[2]
    D_tied = n21.forest_proximity_distance(leaves_tied)
    nbrs = n21.neighbours_in_distance_matrix(D_tied, 4)
    self_in_own_list = int(sum(i in nbrs[i] for i in range(len(nbrs))))
    check('no molecule is its own neighbour, ties and all',
          self_in_own_list == 0,
          f'{self_in_own_list} of {len(nbrs)} molecules were')
    check('the two molecules put in identical leaves are each other\'s nearest',
          nbrs[2][0] == 3 and nbrs[3][0] == 2,
          f'{nbrs[2][0]} and {nbrs[3][0]}')


# ---------------------------------------------------------------------------

def _ap_by_space(results, level='0.5', base='base_1'):
    """Average precision per space inside one single representation, with
    the 'base_1/' prefix taken off so the checks read the family names."""
    row = results['summary']['per_level'][level]
    pre = f'{base}/'
    return {k[len(pre):]: v for k, v in row['average_precision_by_space'].items()
            if k.startswith(pre)}


def _gap_over_plain(ap, space, plain):
    """How much better this distance ranks the corrupted molecules.

    Both numbers are average precision, so the difference is in average
    precision too: 0 means the two distances rank the corrupted molecules
    equally well, and a positive number means `space` puts them higher up.
    """
    return float(ap[space] - ap[plain])


def _run(signal, seed, folds=3, second_rep=False):
    """ONE condition. `_run_one_condition`, not `run`.

    Since 2026-09-23 n21.run sweeps every condition in DETECTION_IS_DEFINED,
    writes each to its own file and returns an INDEX of what it wrote, so the
    scores these checks read are no longer in its return value. They are about
    one condition's behaviour, so they call the one-condition layer. The sweep
    itself is covered in _smoke_n23, which tests the shared helper both now use.
    """
    data = make_data(n=400, seed=seed, signal=signal)
    kw = dict(CHEAP)
    if second_rep:
        # base_2: the same features with noise on top, so it is a different
        # matrix and every space is built twice.
        rng = np.random.default_rng(seed + 100)
        data['reps']['ecfp4_noisy'] = (data['reps']['ecfp4']
                                       + rng.normal(0, 0.3, (400, 20)))
        kw['base_reps'] = {'rf': ['ecfp4', 'ecfp4_noisy']}
    return n21._run_one_condition(data, {'n_splits': folds}, OUT,
                                  f'n21smoke_{signal}', **kw)


def test_sparse_signal_rewards_the_better_distance():
    print("\ntwo of twenty features carry the whole signal:")
    r = _run('sparse', seed=1)
    check('it ran', 'summary' in r, str(r)[:100])
    if 'summary' not in r:
        return None
    ap = _ap_by_space(r)
    block = r['per_level']['0.5']
    check('some molecules were corrupted and some were not',
          0 < block['molecules_corrupted_mean'] <
          block['molecules_in_training_fold_mean'],
          f"{block['molecules_corrupted_mean']:.0f} corrupted of "
          f"{block['molecules_in_training_fold_mean']:.0f} training molecules")

    gaps = {}
    for metric in ('euclidean', 'manhattan'):
        plain = f'plain|{metric}'
        e = n21.EXPONENT_MATCHED_TO_METRIC[metric]
        weighted = f'imp_pow{e:g}|lightgbm_gain|{metric}'
        gaps[f'weighted|{metric}'] = _gap_over_plain(ap, weighted, plain)
        check(f'the importance-weighted distance beats the plain one under '
              f'{metric}',
              gaps[f'weighted|{metric}'] > 0.05,
              f'average precision {ap[weighted]:.4f} against {ap[plain]:.4f}')
        gaps[f'forest|{metric}'] = _gap_over_plain(ap, 'forest_proximity',
                                                   plain)
        check(f'the forest proximity beats the plain distance under {metric}',
              gaps[f'forest|{metric}'] > 0.05,
              f"average precision {ap['forest_proximity']:.4f} against "
              f"{ap[plain]:.4f}")
    return gaps


def test_equal_signal_shows_no_such_gap(sparse_gaps):
    print("\nevery one of the twenty features matters equally:")
    r = _run('equal', seed=1)
    check('it ran', 'summary' in r, str(r)[:100])
    if 'summary' not in r or not sparse_gaps:
        return
    ap = _ap_by_space(r)
    for metric in ('euclidean', 'manhattan'):
        plain = f'plain|{metric}'
        e = n21.EXPONENT_MATCHED_TO_METRIC[metric]
        weighted = f'imp_pow{e:g}|lightgbm_gain|{metric}'
        g = _gap_over_plain(ap, weighted, plain)
        s = sparse_gaps[f'weighted|{metric}']
        check(f'the importance weighting buys almost nothing under {metric}',
              g < 0.5 * s,
              f'{g:+.4f} in average precision here against {s:+.4f} when two '
              f'features carried the signal')
        gf = _gap_over_plain(ap, 'forest_proximity', plain)
        sf = sparse_gaps[f'forest|{metric}']
        check(f'the forest proximity buys almost nothing under {metric}',
              gf < 0.5 * sf,
              f'{gf:+.4f} in average precision here against {sf:+.4f} when two '
              f'features carried the signal')


def test_the_spaces_and_the_convention_flags():
    print("\nwhat one run records:")
    r = _run('sparse', seed=2, folds=2, second_rep=True)
    check('it ran', 'summary' in r, str(r)[:100])
    if 'summary' not in r:
        return
    check('the single representations are the two named, and recorded',
          r['base_reps_used'] == {'base_1': 'ecfp4', 'base_2': 'ecfp4_noisy'}
          and r['base_reps_ranked_by_model'] == 'rf' and 'base_rep' not in r,
          str(r.get('base_reps_used')))
    check('every space is built under both, and no key lacks a prefix',
          all(k.split('/', 1)[0] in ('base_1', 'base_2') for k in r['spaces'])
          and len([k for k in r['spaces'] if k.startswith('base_1/')])
          == len([k for k in r['spaces'] if k.startswith('base_2/')]),
          f"{len(r['spaces'])} spaces")
    _pb = r['per_level']['0.5']['per_base']
    check('each single representation has its own clean and corrupted scores',
          _pb['base_1']['noisy_score_by_model']['rf']
          != _pb['base_2']['noisy_score_by_model']['rf']
          and r['per_level']['0.5']['noisy_score_by_model']
          == _pb['base_1']['noisy_score_by_model'],
          f"{_pb['base_1']['noisy_score_by_model']['rf']:.4f} against "
          f"{_pb['base_2']['noisy_score_by_model']['rf']:.4f}")
    spaces = {k[len('base_1/'):]: v for k, v in r['spaces'].items()
              if k.startswith('base_1/')}
    families = sorted(set(s['family'] for s in spaces.values()))
    check('three families of distance are present, the hybrid being switched off',
          families == ['forest_proximity', 'importance_weighted', 'plain'],
          str(families))
    check('the hybrid says why it is absent',
          'absent_because' in r['hybrid'], r['hybrid'].get('absent_because', ''))

    matched = {k: v['convention_matches_metric'] for k, v in spaces.items()
               if v['family'] == 'importance_weighted'}
    check('exponent 0.5 is the matched one under euclidean',
          matched['imp_pow0.5|lightgbm_gain|euclidean'] is True)
    check('exponent 1 is the matched one under manhattan',
          matched['imp_pow1|lightgbm_gain|manhattan'] is True)
    check('exponent 0.5 under manhattan is the mismatched cell n12 runs today',
          matched['imp_pow0.5|lightgbm_gain|manhattan'] is False)

    conv = r['summary']['per_level']['0.5']['weighting_convention']
    check('the convention question is answered at both metrics, in both',
          sorted(conv) == [f'{b}/lightgbm_gain|{m}'
                           for b in ('base_1', 'base_2')
                           for m in ('euclidean', 'manhattan')],
          str(sorted(conv)))
    for name, row in conv.items():
        cmp_key = '0.5 against 1'
        check(f'{name}: the two exponents were paired per cell',
              row[cmp_key]['detection_average_precision']['cells_paired'] > 0,
              f"{row[cmp_key]['detection_average_precision']['cells_paired']} "
              f"cells")

    block = r['per_level']['0.5']
    # base_2 is the noisier copy of the features, and on it the corrupted fit
    # can score no worse than the clean one. n21 then refuses to divide by that
    # gap and writes why (GAP_MIN_FRAC), which is a recorded refusal, not a
    # missing number. Every base_1 space must still give a finite figure.
    for space in r['spaces']:
        row = block['per_space'][space]
        finite = bool(row['repair']) and np.isfinite(
            row['recovery_percent_of_gap_mean'])
        refused = bool(row['repair']) and all(
            (v.get('recovery_note') or '').startswith('not computed: the '
                                                      'clean-to-corrupted gap')
            for v in row['repair'].values())
        check(f'{space}: the repair figure was computed, or refused with '
              f'the gap-floor reason',
              finite or (space.startswith('base_2/') and refused),
              f"recovery {row['recovery_percent_of_gap_mean']:+.1f}% of the gap"
              if finite else 'refused: the clean-to-corrupted gap is under '
                             'GAP_MIN_FRAC of the clean score')


def test_the_two_guards():
    print("\nthe two guards:")
    data = make_data(n=200, seed=4)
    try:
        n21.run(data, {'n_splits': 2}, OUT, 'n21smoke_guard',
                noise_condition='gaussian', **CHEAP)
        check('a condition where every label moves is refused', False,
              'it ran instead')
    except ValueError as exc:
        check('a condition where every label moves is refused',
              'average precision' in str(exc), str(exc)[:90])

    kw = dict(CHEAP)
    kw.pop('levels')
    kw['k_values'] = [5]
    kw['detectors'] = ['neighbor_residual']
    kw['metrics'] = ['manhattan']
    kw['importance_exponents'] = [1.0]
    kw['noise_seeds'] = [0]
    r = n21._run_one_condition(data, {'n_splits': 2}, OUT,
                               'n21smoke_levels', **kw)
    check('level 0.0 is not in the default grid',
          0.0 not in r['levels'], str(r['levels']))
    check('and the result file says why it is gone',
          '0' in r['levels_dropped'],
          str(r['levels_dropped'])[:120])


if __name__ == '__main__':
    test_transcription_matches_the_estimators()
    test_weighting_is_n12s_at_exponent_half()
    test_forest_proximity_is_what_it_says()
    sparse_gaps = test_sparse_signal_rewards_the_better_distance()
    test_equal_signal_shows_no_such_gap(sparse_gaps)
    test_the_spaces_and_the_convention_flags()
    test_the_two_guards()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
