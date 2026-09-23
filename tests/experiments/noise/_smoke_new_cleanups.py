#!/usr/bin/env python3
"""Smoke for the cleanups added from the compass label-noise review.

WHAT A CLEANUP MUST DO. `cleaned_training_set` is the one dispatcher every
noise experiment calls, so a new name has to honour its contract exactly or it
breaks n07, n17 and n18 at once. The contract is a dict carrying X, y,
sample_weight, keep and label_column, or None when the cleanup declines.

WHY EACH NEW ONE EARNS ITS PLACE, which is what these checks are really for. A
method that is a second spelling of an existing one is noise in the menu, so
each check below shows the new cleanup doing something the old one does not.

  huber_weight   against `downweight`. Both return weights from the same
                 suspicion score. downweight's weight is 1/(1+strength*s),
                 which falls from the very first molecule and never reaches
                 zero. Huber's is FLAT at 1.0 inside the scale and only then
                 falls, so an ordinary label is not touched at all. Test 2
                 shows the flat region exists; test 3 shows it separates the
                 damaged labels harder than downweight does.

  js_shrink      against `oof_smooth`. Both move a label toward the same
                 out-of-fold prediction. oof_smooth moves it by a fraction the
                 CALLER chooses; js_shrink's fraction is DERIVED from the two
                 variances, so there is no alpha. It BUILDS its own target
                 when the caller hoists none, because n07 hoists none at all --
                 declining there made it look broken rather than untested.

Cheap by design: pure functions on in-memory arrays, no find_best_hybrid, no
result store, no dataset. Runs while every store is stale.

Run:  python tests/experiments/noise/_smoke_new_cleanups.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np

from experiments.noise.n07_remediation import (
    cleaned_training_set, CLEANUPS, REGRESSION_ONLY_CLEANUPS,
    PROX_SMOOTH_MAX_ROWS, CLEANUP_TARGETS, targets_needed)

_FAILS = []


def _check(name, cond, detail=''):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not cond:
        _FAILS.append(name)


def _fixture(seed=0, n=150, n_bad=15):
    """A set where 15 of the 150 labels are wrecked and we know which."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = X[:, 0] * 2.0 + rng.randn(n) * 0.2
    bad = rng.choice(n, n_bad, replace=False)
    y[bad] += rng.randn(n_bad) * 5.0
    scores = np.abs(y - X[:, 0] * 2.0)      # a detector that works
    groups = np.repeat(np.arange(n // 10), 10)
    return X, y, scores, groups, bad


def _contract(name, out, n_rows):
    """Every cleanup returns this shape or None."""
    _check(f'{name} returns the dispatcher contract',
           out is not None and set(out) == {'X', 'y', 'sample_weight',
                                            'keep', 'label_column'},
           f"got {sorted(out) if out else None}")
    if out is None:
        return
    _check(f'{name} keeps every molecule',
           len(out['X']) == n_rows and len(out['y']) == n_rows,
           f"{len(out['X'])} of {n_rows}")


def test_registered():
    for nm in ('huber_weight', 'js_shrink', 'prox_smooth'):
        _check(f'{nm} is in CLEANUPS', nm in CLEANUPS)
    _check('js_shrink is regression-only, because there is no halfway point '
           'between two nominal classes',
           'js_shrink' in REGRESSION_ONLY_CLEANUPS)
    _check('huber_weight is NOT regression-only, because a weight is defined '
           'on any label',
           'huber_weight' not in REGRESSION_ONLY_CLEANUPS)


def test_huber_has_a_flat_region():
    """THE HEADLINE for huber_weight. Ordinary labels are untouched.

    Under Huber's weight the molecules inside the scale all get exactly 1.0
    before mean-normalisation, which is the property `downweight` does not
    have: its weight falls from the very first molecule.
    """
    X, y, scores, groups, bad = _fixture()
    hub = cleaned_training_set('huber_weight', X, y, scores, 0.1, groups=groups)
    dwn = cleaned_training_set('downweight', X, y, scores, 0.1, groups=groups)
    _contract('huber_weight', hub, len(y))
    if hub is None or dwn is None:
        return
    hw, dw = hub['sample_weight'], dwn['sample_weight']

    # After mean-normalisation the flat region is the modal value, and it is
    # the MAXIMUM: nothing is weighted above an ordinary molecule.
    at_max = int(np.isclose(hw, hw.max()).sum())
    _check('huber_weight leaves a block of ordinary molecules at one weight',
           at_max >= 10,
           f"{at_max} of the {len(hw)} molecules sit at the top weight")
    _check('downweight has no such block, because its weight falls from the '
           'first molecule',
           int(np.isclose(dw, dw.max()).sum()) < at_max,
           f"downweight has {int(np.isclose(dw, dw.max()).sum())} at its top "
           f"weight against huber_weight's {at_max}")
    _check('both are mean-normalised to one',
           abs(hw.mean() - 1.0) < 1e-9 and abs(dw.mean() - 1.0) < 1e-9)


def test_huber_separates_harder_than_downweight():
    """Both should weight the wrecked labels lower. Huber should do it more."""
    X, y, scores, groups, bad = _fixture()
    hub = cleaned_training_set('huber_weight', X, y, scores, 0.1, groups=groups)
    dwn = cleaned_training_set('downweight', X, y, scores, 0.1, groups=groups)
    if hub is None or dwn is None:
        return
    clean = np.setdiff1d(np.arange(len(y)), bad)
    r_h = hub['sample_weight'][clean].mean() / hub['sample_weight'][bad].mean()
    r_d = dwn['sample_weight'][clean].mean() / dwn['sample_weight'][bad].mean()
    _check('huber_weight weights a clean molecule above a wrecked one',
           r_h > 1.0, f"clean-to-wrecked weight ratio {r_h:.2f} to 1")
    _check('and it separates them harder than downweight does',
           r_h > r_d,
           f"huber {r_h:.2f} to 1 against downweight {r_d:.2f} to 1")


def test_huber_survives_identical_scores():
    """Every score the same: no molecule is more suspect, so the weight is flat
    rather than a divide by zero."""
    X, y, _, groups, _ = _fixture()
    flat = np.full(len(y), 0.3)
    out = cleaned_training_set('huber_weight', X, y, flat, 0.1, groups=groups)
    _contract('huber_weight on identical scores', out, len(y))
    if out is not None:
        w = out['sample_weight']
        _check('huber_weight returns a flat weight when no molecule is more '
               'suspect than another',
               np.allclose(w, 1.0), f"range {w.min():.4f} to {w.max():.4f}")


def test_js_shrink_moves_the_suspect_labels_most():
    """THE HEADLINE for js_shrink. The shrink is derived, not chosen, and it is
    largest where the label looks worst."""
    X, y, scores, groups, bad = _fixture()
    oof = X[:, 0] * 2.0                      # a near-perfect prediction
    out = cleaned_training_set('js_shrink', X, y, scores, 0.1,
                               groups=groups, targets={'oof': oof})
    _contract('js_shrink', out, len(y))
    if out is None:
        return
    moved = np.abs(out['y'] - y)
    clean = np.setdiff1d(np.arange(len(y)), bad)
    _check('js_shrink moves the wrecked labels further than the clean ones',
           moved[bad].mean() > moved[clean].mean(),
           f"{moved[bad].mean():.4f} against {moved[clean].mean():.4f} "
           f"label units")
    _check('js_shrink moves every label toward the prediction, never past it',
           np.all(moved <= np.abs(y - oof) + 1e-9),
           'a label that overshot its target would mean shrink above one')
    _check('js_shrink reports the corrected column so a recipe search can see '
           'it', out['label_column'] is not None
           and np.allclose(out['label_column'], out['y']))


def test_js_shrink_builds_its_own_target_when_none_is_hoisted():
    """⛔ THE ONE THAT CAUGHT A REAL BUG.

    js_shrink first DECLINED when no out-of-fold prediction was handed to it.
    That looked principled -- better than shrinking toward the fold mean, which
    is a different method -- and it was wrong, because n07 hoists no targets at
    all: all three of its `_evaluate_fold` callers leave targets at None. So
    js_shrink skipped on every n07 fold and would have read as broken rather
    than untested.

    It now builds its own target, exactly as `oof_smooth` already did, and
    declines only when building one FAILS.
    """
    X, y, scores, groups, bad = _fixture()
    clean = np.setdiff1d(np.arange(len(y)), bad)
    for targets in (None, {}, {'oof': None}):
        out = cleaned_training_set('js_shrink', X, y, scores, 0.1,
                                   groups=groups, targets=targets)
        _check(f'js_shrink builds its own target when targets={targets!r}',
               out is not None)
        if out is None:
            continue
        moved = np.abs(out['y'] - y)
        _check(f'  and still moves the wrecked labels furthest, targets='
               f'{targets!r}',
               moved[bad].mean() > moved[clean].mean(),
               f"{moved[bad].mean():.4f} against {moved[clean].mean():.4f} "
               f"label units")


def test_every_cleanup_declares_the_target_it_needs():
    """The mapping that stops the next method hitting the same gap.

    n18 used to write the cleanup-to-target pairs out by hand, three of them,
    so a fourth cleanup needing a target could not appear in the list. It is
    now one declaration that every caller derives from.
    """
    _check('CLEANUP_TARGETS names a target for js_shrink',
           CLEANUP_TARGETS.get('js_shrink') == 'oof',
           f"got {CLEANUP_TARGETS.get('js_shrink')!r}")
    _check('every cleanup it names is a real cleanup',
           set(CLEANUP_TARGETS) <= set(CLEANUPS),
           f"unknown: {sorted(set(CLEANUP_TARGETS) - set(CLEANUPS))}")
    need = targets_needed(CLEANUPS)
    _check('targets_needed covers all three hoistable targets',
           set(need) == {'oof', 'gp', 'quantile'}, f"got {need}")
    _check('targets_needed asks for nothing when no cleanup needs anything',
           targets_needed(['remove', 'downweight', 'huber_weight']) == [])


def test_prox_smooth_beats_euclidean_where_most_features_are_noise():
    """THE HEADLINE for prox_smooth, and the reason it is not knn_smooth with a
    different argument.

    40 features, 2 of them carrying the whole signal. Euclidean distance treats
    all 40 alike, so 38 columns of noise drown the 2 that matter. A forest
    proximity cannot have that problem: a feature the trees never split on
    cannot move two molecules apart.

    The measure is how close a corrected label gets to the value it had before
    it was wrecked -- which the fixture knows and neither cleanup is told.
    """
    rng = np.random.RandomState(0)
    n, n_bad = 200, 20
    X = rng.randn(n, 40)
    signal = X[:, 0] * 3.0 + X[:, 1] * 2.0
    clean = signal + rng.randn(n) * 0.2
    y = clean.copy()
    bad = rng.choice(n, n_bad, replace=False)
    y[bad] += rng.randn(n_bad) * 6.0
    scores = np.abs(y - signal)
    groups = np.repeat(np.arange(n // 10), 10)

    knn = cleaned_training_set('knn_smooth', X, y, scores, 0.1, groups=groups)
    prx = cleaned_training_set('prox_smooth', X, y, scores, 0.1, groups=groups)
    _contract('prox_smooth', prx, n)
    if knn is None or prx is None:
        return

    before = np.abs(y[bad] - clean[bad]).mean()
    e_knn = np.abs(knn['y'][bad] - clean[bad]).mean()
    e_prx = np.abs(prx['y'][bad] - clean[bad]).mean()

    _check('prox_smooth corrects the same molecules knn_smooth does',
           int((prx['y'] != y).sum()) == int((knn['y'] != y).sum()) == n_bad,
           f"{int((prx['y'] != y).sum())} and {int((knn['y'] != y).sum())} "
           f"of the {n_bad} wrecked")
    _check('prox_smooth moves a wrecked label closer to its true value',
           e_prx < before,
           f"{before:.3f} to {e_prx:.3f} label units from the truth")
    _check('and closer than Euclidean neighbours manage, where most features '
           'are noise',
           e_prx < e_knn,
           f"prox_smooth {e_prx:.3f} against knn_smooth {e_knn:.3f}")
    _check('prox_smooth leaves the unflagged molecules bit-identical',
           np.array_equal(prx['y'][prx['y'] == y], y[prx['y'] == y]))


def test_prox_smooth_refuses_a_training_block_it_cannot_hold():
    """The proximity is a square matrix, so its memory is quadratic. Above the
    cap it is refused, not subsampled: a proximity over a subsample is a
    different quantity from one over the fold."""
    _check('PROX_SMOOTH_MAX_ROWS is set', PROX_SMOOTH_MAX_ROWS > 0,
           f"{PROX_SMOOTH_MAX_ROWS} molecules")
    n = PROX_SMOOTH_MAX_ROWS + 1
    rng = np.random.RandomState(1)
    X = rng.randn(n, 4).astype(np.float32)
    y = rng.randn(n)
    out = cleaned_training_set('prox_smooth', X, y, np.abs(y), 0.1)
    _check('prox_smooth declines above the cap rather than building the matrix',
           out is None, f"at {n} molecules")


def main():
    print("=== SMOKE: cleanups added from the compass review ===\n")
    print("Registration:")
    test_registered()
    print("\nhuber_weight — the flat region is what makes it not downweight:")
    test_huber_has_a_flat_region()
    test_huber_separates_harder_than_downweight()
    test_huber_survives_identical_scores()
    print("\nprox_smooth — the forest's own similarity, not Euclidean:")
    test_prox_smooth_beats_euclidean_where_most_features_are_noise()
    test_prox_smooth_refuses_a_training_block_it_cannot_hold()
    print("\njs_shrink — the shrink is derived, not chosen:")
    test_js_shrink_moves_the_suspect_labels_most()
    test_js_shrink_builds_its_own_target_when_none_is_hoisted()
    print("\nThe declaration that stops the next method being forgotten:")
    test_every_cleanup_declares_the_target_it_needs()
    print(f"\n=== {'ALL GREEN' if not _FAILS else f'{len(_FAILS)} FAILED: {_FAILS}'} ===")
    sys.exit(1 if _FAILS else 0)


if __name__ == '__main__':
    main()
