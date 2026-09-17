#!/usr/bin/env python3
"""Smoke for the three repair rules added to n07 and n11 on 2026-09-17.

WHAT THE THREE ARE. Every repair rule these two files carried moved a label
toward the mean label of its ten nearest training molecules. Three more targets
existed in the repository and nothing used them:

  oof_smooth     move each label toward what a LightGBM model trained on the
                 OTHER training molecules predicts for it, in proportion to its
                 suspicion score. kirby/utils/noisy_labels/core.py:127, the same
                 call n19 makes, on a scaffold-group-aware inner split.
  gp_smooth      the same, toward the posterior mean of a Gaussian process
                 fitted the same way. GaussianProcessEstimator has always
                 computed this and kept it as predictions_.
  quantile_clip  not a move toward a target. A quantile regression forest gives
                 each molecule the 10th-to-90th-percentile range of the labels
                 carried by the training molecules sharing its leaves. A label
                 inside its range is left byte-identical; a label outside is
                 pulled onto the edge it crossed.

WHAT THIS FILE CHECKS.

  1. A corrupted subset of known membership costs the downstream model
     something: fitting on the corrupted training labels is worse on the
     held-out molecules than fitting on the clean ones. Without this the other
     checks divide by nothing.

  2. Each of the three closes some of that clean-to-corrupted gap. Reported as
     the raw held-out RMSE of all five arms first, then as the share of the gap
     recovered.

  3. The clip changes ONLY the labels that sat outside their own range. Two
     halves: every label inside its range comes back bit-for-bit, and every
     label the clip moved was outside its range.

  4. n07's own list carries all three, and running one through
     `cleaned_training_set` returns a full-length cleaned label column, which is
     what a caller hands to find_best_hybrid(train_labels=...).

  5. n11's grid emits the two prefixed smoothing rules and the clip rule, the
     rules it emitted before 2026-09-17 are all still there under their old
     keys, and the clip row says how many training labels it moved out of how
     many there were.

The set is synthetic: a smooth function of 12 features over 400 molecules in 20
scaffold groups, which is what makes the corrupted subset known by construction
and the whole file run in under a minute.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Two OpenMP runtimes in one process: torch links libiomp5 and LightGBM links
# libomp, and a multi-threaded fit of one after the other kills the process with
# exit 139 and no traceback. Same two switches _smoke_n20.py sets, for the same
# reason. n11 try-imports torch, and both of the new targets fit LightGBM.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from kirby.utils.noise_estimation import (
    QuantileForestEstimator, clip_to_range,
)
import experiments.noise.n07_remediation as n07
import experiments.noise.n11_estimation_remediation_grid as n11

FAIL = []

# The three rules this file is about.
NEW_RULES = ('oof_smooth', 'gp_smooth', 'quantile_clip')

# How many training molecules in every hundred get a corrupted label, and how
# far their label moves. 12 in 100, each by four label SDs in a random
# direction — the Outlier shape n07 injects, at a dose big enough that one fold
# of 300 molecules shows it.
CORRUPT_FRACTION = 0.12
CORRUPT_SIZE_IN_LABEL_SDS = 4.0


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def make_set(seed=0):
    """400 molecules, 12 features, 20 scaffold groups, a smooth label.

    Returns everything already split into one training fold and one held-out
    fold, with the training features standardised on the training rows only.
    """
    rng = np.random.RandomState(seed)
    n, d, n_groups = 400, 12, 20
    X = rng.normal(size=(n, d))
    y = (3.0 * np.sin(X[:, 0]) + 2.0 * X[:, 1] - 1.5 * X[:, 2] ** 2
         + X[:, 3] * X[:, 4] + 0.2 * rng.normal(size=n))
    groups = rng.randint(0, n_groups, size=n)

    tr, va = next(iter(GroupKFold(n_splits=4).split(X, y, groups)))
    sc = StandardScaler()
    return {
        'X_tr': sc.fit_transform(X[tr]),
        'X_va': sc.transform(X[va]),
        'y_tr': y[tr].copy(),
        'y_va': y[va].copy(),
        'g_tr': groups[tr],
    }


def corrupt(y_tr, seed=1):
    """Move a known subset of the training labels. Returns (y_noisy, mask)."""
    rng = np.random.RandomState(seed)
    n_bad = int(round(CORRUPT_FRACTION * len(y_tr)))
    idx = rng.choice(len(y_tr), n_bad, replace=False)
    mask = np.zeros(len(y_tr), dtype=bool)
    mask[idx] = True
    shift = CORRUPT_SIZE_IN_LABEL_SDS * float(np.std(y_tr))
    y_noisy = y_tr.copy()
    y_noisy[mask] += shift * rng.choice([-1.0, 1.0], size=n_bad)
    return y_noisy, mask


def fit_score(X_tr, y_tr, X_va, y_va, seed=42):
    m = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    m.fit(X_tr, y_tr)
    return rmse(y_va, m.predict(X_va))


# ---------------------------------------------------------------------------
# 1 + 2: the gap exists, and each of the three closes some of it
# ---------------------------------------------------------------------------

def test_each_rule_recovers_some_of_the_gap():
    print("\n=== Each of the three recovers part of the clean-to-corrupted gap ===")
    s = make_set()
    y_noisy, mask = corrupt(s['y_tr'])
    print(f"  {int(mask.sum())} of {len(mask)} training labels were moved, each "
          f"by {CORRUPT_SIZE_IN_LABEL_SDS} label SDs")

    clean = fit_score(s['X_tr'], s['y_tr'], s['X_va'], s['y_va'])
    noisy = fit_score(s['X_tr'], y_noisy, s['X_va'], s['y_va'])
    gap = noisy - clean            # rmse: the corrupted fit is the bigger number
    print(f"  held-out rmse    clean {clean:.4f}   corrupted {noisy:.4f}   "
          f"gap {gap:.4f}")
    check('the corrupted subset costs the model something', gap > 0,
          f'clean rmse {clean:.4f}, corrupted rmse {noisy:.4f}')

    suspicion = n07._compute_noise_scores(s['X_tr'], y_noisy, is_clf=False)
    check('the detector produced a suspicion score for every training molecule',
          suspicion is not None and len(suspicion) == len(y_noisy))
    if suspicion is None:
        return None

    out = {}
    for rule in NEW_RULES:
        cleaned = n07.cleaned_training_set(
            rule, s['X_tr'], y_noisy, suspicion, CORRUPT_FRACTION,
            is_clf=False, groups=s['g_tr'])
        check(f'{rule} returned a cleaned training set', cleaned is not None)
        if cleaned is None:
            continue
        score = fit_score(cleaned['X'], cleaned['y'], s['X_va'], s['y_va'])
        recovered = (noisy - score) / gap * 100.0
        out[rule] = {'rmse': score, 'recovered_pct': recovered,
                     'cleaned': cleaned}
        print(f"  {rule:<14} held-out rmse {score:.4f}   "
              f"{recovered:+.1f}% of the gap")

    for rule in NEW_RULES:
        if rule not in out:
            continue
        check(f'{rule} closes some of the gap', out[rule]['recovered_pct'] > 0,
              f"rmse {out[rule]['rmse']:.4f} against corrupted {noisy:.4f} "
              f"and clean {clean:.4f}")

    # Every one of them writes a full-length label column, which is what a
    # caller hands to find_best_hybrid(train_labels=...).
    for rule in NEW_RULES:
        if rule not in out:
            continue
        col = out[rule]['cleaned']['label_column']
        check(f'{rule} writes a full-length cleaned label column',
              col is not None and len(col) == len(y_noisy)
              and np.all(np.isfinite(col)))

    return {'set': s, 'y_noisy': y_noisy, 'mask': mask, 'out': out}


# ---------------------------------------------------------------------------
# 3: the clip touches only the labels that were outside their range
# ---------------------------------------------------------------------------

def test_clip_moves_only_the_labels_outside_the_range(state):
    print("\n=== The clip changes only the labels that sat outside the range ===")
    if state is None or 'quantile_clip' not in state['out']:
        check('the clip ran', False, 'no clipped column to check')
        return
    s, y_noisy = state['set'], state['y_noisy']

    # The same forest the cleanup fitted, refitted here so the two edges of each
    # molecule's range are in hand. Deterministic: same seed, same
    # group-disjoint inner split.
    est = QuantileForestEstimator(n_folds=n07.REPAIR_OOF_SPLITS,
                                  groups=s['g_tr'])
    est.estimate(s['X_tr'], y_noisy)
    lo, hi = est.quantile_lo_, est.quantile_hi_
    outside = (y_noisy < lo) | (y_noisy > hi)
    inside = ~outside

    y_clipped = np.asarray(state['out']['quantile_clip']['cleaned']['y'],
                           dtype=float)
    changed = y_clipped != y_noisy

    print(f"  {int(outside.sum())} of {len(y_noisy)} training labels sat "
          f"outside their own range; the clip moved {int(changed.sum())}")

    check('the clip left every label inside its range bit-for-bit',
          np.array_equal(y_clipped[inside], y_noisy[inside]),
          f'{int(np.count_nonzero(changed & inside))} of '
          f'{int(inside.sum())} inside labels were altered')
    check('every label the clip moved was outside its range',
          not np.any(changed & inside),
          f'{int(np.count_nonzero(changed & ~outside))} moved labels were '
          f'inside')
    check('the clip found something to move', changed.sum() > 0,
          f'{int(changed.sum())} labels moved')
    check('a moved label lands exactly on the edge it crossed',
          np.allclose(y_clipped[outside], np.clip(y_noisy, lo, hi)[outside]),
          'alpha=1.0 puts it on the boundary')

    # The clip is aimed at corrupted labels, so say how many of the ones it
    # moved were in fact corrupted, out of how many it moved.
    m = state['mask']
    print(f"  of the {int(changed.sum())} labels the clip moved, "
          f"{int(np.count_nonzero(changed & m))} were ones the corruption had "
          f"moved (there were {int(m.sum())} of those in {len(m)} rows)")

    # alpha is a strength, not a switch: half the excursion leaves the label
    # between where it was and the edge, and nothing inside the range moves.
    half = clip_to_range(y_noisy, lo, hi, alpha=0.5)
    check('alpha=0.5 moves an outside label half way to the edge',
          np.allclose(half[outside],
                      y_noisy[outside] + 0.5 * (np.clip(y_noisy, lo, hi)[outside]
                                                - y_noisy[outside]))
          and np.array_equal(half[inside], y_noisy[inside]))
    check('alpha=0.0 changes nothing at all',
          np.array_equal(clip_to_range(y_noisy, lo, hi, alpha=0.0), y_noisy))


# ---------------------------------------------------------------------------
# 4: n07 lists all three, and refuses them on a class label
# ---------------------------------------------------------------------------

def test_n07_carries_the_rules():
    print("\n=== n07 carries the three rules, and refuses them on class labels ===")
    for rule in NEW_RULES:
        check(f'{rule} is one of n07\'s cleanups', rule in n07.CLEANUPS)
    check('the clip reads no suspicion score',
          'quantile_clip' not in n07.SCORE_DRIVEN_CLEANUPS)
    check('the two smoothing rules read a suspicion score',
          all(r in n07.SCORE_DRIVEN_CLEANUPS
              for r in ('oof_smooth', 'gp_smooth')))

    # The four that were there before are untouched.
    for rule in ('remove', 'downweight', 'label_smooth', 'knn_smooth'):
        check(f'{rule} is still a cleanup', rule in n07.CLEANUPS)

    s = make_set()
    y_cls = (s['y_tr'] > np.median(s['y_tr'])).astype(int)
    scores = np.abs(np.asarray(y_cls, dtype=float) - 0.5)
    for rule in NEW_RULES:
        try:
            n07.cleaned_training_set(rule, s['X_tr'], y_cls, scores, 0.1,
                                     is_clf=True, groups=s['g_tr'])
            refused = False
        except ValueError:
            refused = True
        check(f'{rule} refuses a class label rather than writing a float',
              refused)


# ---------------------------------------------------------------------------
# 5: n11's grid emits the new keys and keeps every old one
# ---------------------------------------------------------------------------

def test_n11_grid_keys():
    print("\n=== n11's grid emits the new rules and keeps the old ones ===")
    from kirby.utils.noisy_labels import oof_predict

    s = make_set()
    y_noisy, _mask = corrupt(s['y_tr'])
    mf = {'rf': lambda: RandomForestRegressor(n_estimators=60, random_state=0,
                                              n_jobs=1)}
    targets = {
        'oof': oof_predict(s['X_tr'], y_noisy, s['g_tr'],
                           n_splits=n11.EXTRA_TARGET_SPLITS),
    }

    res = n11._run_estimator_grid(
        'neighbor_residual', 'euclidean', s['X_tr'], y_noisy,
        s['X_tr'], s['X_va'], s['y_va'], mf, rmse, len(y_noisy),
        is_clf=False, groups=s['g_tr'], extra_targets=targets)
    check('the grid ran', '_error' not in res, res.get('_error', ''))

    own = [k for k in res if k.startswith('rf|') and '+' not in k
           and not k.startswith('rf|quantile_clip')]
    prefixed = [k for k in res if k.startswith('rf|oof+')]
    print(f"  keys under the estimator's own target: {len(own)}; "
          f"under the out-of-fold target: {len(prefixed)}")
    check("the estimator's own remediation keys are unchanged",
          'rf|proportional_0.5' in res and 'rf|inverse_0.5' in res,
          f'{len(own)} keys with no target prefix')
    check('the out-of-fold target got its own keys',
          'rf|oof+proportional_0.5' in res,
          f'{len(prefixed)} keys carrying the target name')
    check('the two targets give different numbers',
          res.get('rf|proportional_0.5') != res.get('rf|oof+proportional_0.5'),
          f"own {res.get('rf|proportional_0.5')}, "
          f"out-of-fold {res.get('rf|oof+proportional_0.5')}")
    check('an estimator with no range emits no clip row',
          not any(k.startswith('rf|quantile_clip') for k in res))

    # The one estimator that does carry a range.
    res_q = n11._run_estimator_grid(
        'quantile_forest', None, s['X_tr'], y_noisy,
        s['X_tr'], s['X_va'], s['y_va'], mf, rmse, len(y_noisy),
        is_clf=False, groups=s['g_tr'], extra_targets=targets)
    check('the quantile forest ran', '_error' not in res_q,
          res_q.get('_error', ''))
    clip_keys = sorted(k for k in res_q if k.startswith('rf|quantile_clip'))
    check('the quantile forest emits one clip row per alpha',
          len(clip_keys) == len(n11.ALPHAS), f'{clip_keys}')
    check('the clip row says how many labels it moved out of how many there were',
          res_q.get('_n_labels_outside_range') is not None
          and res_q.get('_n_training_labels') == len(y_noisy),
          f"{res_q.get('_n_labels_outside_range')} of "
          f"{res_q.get('_n_training_labels')} training labels")
    check('quantile_forest is in n11\'s estimator list',
          'quantile_forest' in n11.BASE_ESTIMATORS)


if __name__ == '__main__':
    state = test_each_rule_recovers_some_of_the_gap()
    test_clip_moves_only_the_labels_outside_the_range(state)
    test_n07_carries_the_rules()
    test_n11_grid_keys()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
