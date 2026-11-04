#!/usr/bin/env python3
"""The held-out cliff rules and the agreement row, against brute force.

WHY THIS EXISTS. Both pieces are new on 2026-11-04 and both are written
vectorised, so neither can be read off the code:

  held_out_cliff_flag answers her open DECIDE item two ways. The looser rule
  asks whether two of the training molecules similar to a held-out one differ
  by more than the gap, and the code answers it from the largest and smallest
  similar training label in one pass. That is only the same question if the
  widest difference in a set is always between its largest and smallest
  member, which this checks pair by pair.

  the agreement row leaves out the molecules a comparison set has no
  prediction for, because a missing prediction is not a prediction of "not a
  cliff". This checks that leaving them out is what happens, and that the
  counts are the counts over what is left.

    python tests/experiments/cliffs/_smoke_held_out_and_agreement.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.noise import cliff_rules as cr                   # noqa: E402
from experiments.cliffs import definition_agreement as A          # noqa: E402
from experiments.cliffs import error_by_cliff_status as E         # noqa: E402


def _brute_same_as_training(y, similar, train_idx, val_idx, cut, is_clf=False):
    out = np.zeros(len(val_idx), dtype=bool)
    for a, h in enumerate(val_idx):
        for t in train_idx:
            if not similar[h, t]:
                continue
            gap = (1.0 if y[h] != y[t] else 0.0) if is_clf \
                else abs(y[h] - y[t])
            if gap > cut:
                out[a] = True
                break
    return out


def _brute_two_similar(y, similar, train_idx, val_idx, cut, is_clf=False):
    out = np.zeros(len(val_idx), dtype=bool)
    for a, h in enumerate(val_idx):
        partners = [t for t in train_idx if similar[h, t]]
        for t1, t2 in combinations(partners, 2):
            gap = (1.0 if y[t1] != y[t2] else 0.0) if is_clf \
                else abs(y[t1] - y[t2])
            if gap > cut:
                out[a] = True
                break
    return out


def check_held_out_rules(n=60, seed=3, is_clf=False):
    rng = np.random.default_rng(seed)
    y = (rng.integers(0, 2, n).astype(float) if is_clf
         else rng.normal(size=n) * 1.4)
    bits = (rng.random((n, 40)) < 0.3).astype(float)
    ctx = cr.CliffContext(y, reps={'fp': bits}, is_clf=is_clf,
                          dataset_name='smoke')
    train_idx = np.arange(0, n, 2)
    val_idx = np.arange(1, n, 2)

    checked = 0
    for pct in (2.0, 10.0, 30.0):
        fn = cr.coefficient_criterion('fp', 'tanimoto', pct=pct)
        similar, _ = fn(ctx)
        similar = np.asarray(similar, dtype=bool).copy()
        np.fill_diagonal(similar, False)
        cuts = ([0.5] if is_clf else [0.3, 1.0, 2.5])
        for cut in cuts:
            got = E.held_out_cliff_flag(ctx, train_idx, val_idx, similar, cut,
                                        'same_as_training')
            want = _brute_same_as_training(y, similar, train_idx, val_idx, cut,
                                           is_clf)
            assert np.array_equal(got, want), (
                f'same_as_training at pct {pct} cut {cut}: '
                f'{int((got != want).sum())} of {len(val_idx)} held-out '
                f'molecules differ')
            got2 = E.held_out_cliff_flag(ctx, train_idx, val_idx, similar, cut,
                                         'two_similar_training_molecules')
            want2 = _brute_two_similar(y, similar, train_idx, val_idx, cut,
                                       is_clf)
            assert np.array_equal(got2, want2), (
                f'two_similar_training_molecules at pct {pct} cut {cut}: '
                f'{int((got2 != want2).sum())} of {len(val_idx)} held-out '
                f'molecules differ')
            checked += 2
    label = 'a class label' if is_clf else 'a regression label'
    print(f'  {label}: {checked} held-out flags match the pair-by-pair count '
          f'on {len(val_idx)} held-out and {len(train_idx)} training molecules')


def check_the_looser_rule_is_looser(n=80, seed=5):
    """Her own record has the looser rule reaching about 40 per cent of
    held-out molecules where the training rule aims at 15. It cannot be
    stricter: every molecule the training rule flags has a similar training
    molecule far from it, and the two of them are two similar training
    molecules far from each other only if BOTH are training molecules. So the
    two rules are not nested, and this checks that rather than assuming it."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n) * 1.4
    bits = (rng.random((n, 40)) < 0.3).astype(float)
    ctx = cr.CliffContext(y, reps={'fp': bits}, dataset_name='smoke')
    train_idx, val_idx = np.arange(0, n, 2), np.arange(1, n, 2)
    fn = cr.coefficient_criterion('fp', 'tanimoto', pct=20.0)
    similar = np.asarray(fn(ctx)[0], dtype=bool).copy()
    np.fill_diagonal(similar, False)
    a = E.held_out_cliff_flag(ctx, train_idx, val_idx, similar, 1.0,
                              'same_as_training')
    b = E.held_out_cliff_flag(ctx, train_idx, val_idx, similar, 1.0,
                              'two_similar_training_molecules')
    print(f'  on the same pairs and the same gap the training rule flags '
          f'{int(a.sum())} of {len(val_idx)} held-out molecules and the '
          f'looser rule flags {int(b.sum())}; '
          f'{int((a & ~b).sum())} are flagged by the first only and '
          f'{int((b & ~a).sum())} by the second only')


def check_agreement_row():
    ours = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    theirs = np.array([1, 0, 1, 0, 1, 1], dtype=bool)
    r = A._agreement_row(ours, theirs)
    assert r['n_molecules_compared'] == 6, r
    assert (r['both'], r['ours_only'], r['theirs_only'], r['neither']) == \
        (2, 1, 2, 1), r
    assert r['n_molecules_with_no_prediction'] == 0

    # Two molecules the comparison set has no prediction for. They must leave
    # the comparison entirely, not count as non-cliffs.
    have = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    r2 = A._agreement_row(ours, theirs, have)
    assert r2['n_molecules_compared'] == 4, r2
    assert r2['n_molecules_with_no_prediction'] == 2, r2
    assert (r2['both'], r2['ours_only'], r2['theirs_only'], r2['neither']) == \
        (1, 1, 1, 1), r2
    # Counting them as non-cliffs instead would have given a different answer,
    # which is the whole reason they are dropped.
    wrong = A._agreement_row(ours & have, theirs & have)
    assert wrong['neither'] == 3 and r2['neither'] == 1, (
        f"calling an unpredicted molecule a non-cliff puts it in `neither`: "
        f"{wrong['neither']} against {r2['neither']}, and both should not be "
        f"the same number")
    print(f"  the agreement row: 6 molecules give 2 both, 1 ours only, "
          f"2 theirs only, 1 neither; with 2 of them unpredicted it compares "
          f"4 and says so")


def check_external_flag_file_is_absent():
    """No file for the three published models, so no row against them."""
    got, record = A.read_external_flags('herg_ki', 'graphcliff', ['C', 'CC'])
    assert got is None, 'a file for graphcliff exists; this check assumes none'
    assert record['available'] is False
    assert 'has not been run' in record['reason']
    print(f"  the three published models: {A.EXTERNAL_DIR} holds no flag "
          f"file, so the run records it as missing and writes no row")


def main():
    print('\nthe two held-out cliff rules against a pair-by-pair count')
    check_held_out_rules()
    check_held_out_rules(is_clf=True)
    print('\nhow far apart the two rules are on the same pairs')
    check_the_looser_rule_is_looser()
    print('\nthe agreement row')
    check_agreement_row()
    check_external_flag_file_is_absent()
    print('\nOK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
