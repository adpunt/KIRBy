#!/usr/bin/env python3
"""Fast, self-contained audit smoke tests for the n13 noise experiment.

Covers the AUDIT_NOISE.md (2026-05-23) findings that were fixed in n13.
These are pure-function / synthetic-data tests — no server data, no
find_best_hybrid — so they run in well under a second and can gate CI.

Run:
    python tests/experiments/noise/_smoke_audit_n04_n05_n13.py
Exit 0 = all green; non-zero on first failure.
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

from experiments.noise import n13_noise_distributions as n13

_failures = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ''))
    if not cond:
        _failures.append(name)


# ───────────────────────── n13 ─────────────────────────
# n13's private `_inject_noise` and its four hand-rolled distributions are gone.
# Every condition now comes from noiseInject, is dose-matched, and reports what it
# delivered. These checks moved with it.

def _n13_inject(condition, y, level=0.3, seed=1, groups=None, is_clf=False):
    from experiments.noise.noise_helpers import inject_train_labels, noise_spec_for
    from kirby.noise_spec import GROUPED_STRATEGIES
    spec = noise_spec_for('qm9', condition, level)
    import numpy as _np
    return inject_train_labels(
        y, spec, train_idx=_np.arange(len(y)), is_clf=is_clf,
        groups=groups if condition in GROUPED_STRATEGIES else None)


def test_n13_private_injector_is_gone():
    """The five ad-hoc definitions the suite grew are replaced by one."""
    check('n13 has no private _inject_noise', not hasattr(n13, '_inject_noise'))
    check('n13 no longer carries a separate partial axis',
          not hasattr(n13, 'PARTIAL_FRACTIONS'))


def test_n13_outlier_mask_marks_the_contaminated_subset():
    """The outlier conditions hit a sparse random subset far harder than the rest."""
    y = np.random.RandomState(0).normal(5.0, 2.0, 1000)
    _, mask, prov = _n13_inject('outlier_p10', y, level=0.6)
    n_corrupt = int(mask.sum())
    check('n13 outlier_p10 marks ~10% of rows',
          80 <= n_corrupt <= 130, f"n_corrupt={n_corrupt} (expect ~100)")
    check('n13 outlier_p10 records the fraction it actually hit',
          abs(prov['affected_molecule_fraction'] - n_corrupt / len(y)) < 1e-9)


def test_n13_shape_only_conditions_perturb_every_row():
    """For the shape-only conditions every molecule gets the SAME scale, so
    "which rows are unreliable" is undefined there -- not empty."""
    y = np.random.RandomState(1).normal(5.0, 2.0, 500)
    for nt in ('gaussian', 'laplace', 'student_t_nu5', 'student_t_nu3'):
        _, mask, prov = _n13_inject(nt, y)
        check(f'n13 {nt} mask all-True', bool(mask.all()),
              f"sum={int(mask.sum())}/{len(y)}")
        check(f'n13 {nt} records affected fraction 1.0',
              abs(prov['affected_molecule_fraction'] - 1.0) < 1e-9)


def test_n13_conditions_distinct_at_matched_dose():
    """Distinct realisations, and the SAME amount of noise in each.

    Distinctness alone was the old check, and it passed throughout the years the
    conditions were delivering between 0.49x and 2.00x each other's dose.
    """
    from kirby.noise_spec import DOSE_MATCHED_STRATEGIES
    y = np.random.RandomState(2).normal(5.0, 2.0, 2000)
    groups = np.arange(len(y)) // 4
    outs, doses = {}, {}
    for nt in DOSE_MATCHED_STRATEGIES:
        yn, _, prov = _n13_inject(nt, y, level=0.3, seed=7, groups=groups)
        outs[nt] = yn
        doses[nt] = prov['realised_dose_fraction_of_spread']
    names = list(outs)
    distinct = all(not np.allclose(outs[names[i]], outs[names[j]])
                   for i in range(len(names)) for j in range(i + 1, len(names)))
    check(f'n13 {len(names)} conditions distinct', distinct)
    # Student-t at nu = 3 has an infinite fourth moment, so its per-run dose is
    # unstable by construction; it is excluded from the flatness band, not from
    # the sweep.
    flat = {k: v for k, v in doses.items() if k != 'student_t_nu3'}
    spread = max(flat.values()) / min(flat.values()) - 1.0
    check('n13 dose is flat across conditions', spread < 0.15,
          f"spread={spread:.1%} over {sorted(flat)}")


def test_n13_clf_flip_models_stay_in_vocabulary():
    """A flipped class label has no magnitude, so classification is not
    dose-matched; it takes the noiseInject flip models instead."""
    y = np.array([0, 1, 2] * 40)
    for nt in ('uniform', 'class_dependent', 'confusion_directed'):
        yn, mask, _ = _n13_inject(nt, y, level=0.5, seed=3, is_clf=True)
        flipped = int((yn != y).sum())
        check(f'n13 clf {nt} flips some labels', flipped > 0, f"flipped={flipped}")
        check(f'n13 clf {nt} labels stay in vocabulary',
              set(np.unique(yn)).issubset({0, 1, 2}))
        check(f'n13 clf {nt} mask matches the flips', int(mask.sum()) == flipped)


def test_n13_dnn_in_unweighted_set():
    """P1 #9: dnn is registered as a Track-B unweighted (non-remediable) model."""
    check('n13 dnn flagged unweighted-track-B',
          'dnn' in n13.UNWEIGHTED_TRACK_B_MODELS)


def test_n13_one_axis_for_every_condition():
    """The separate 'partial' axis is gone.

    'partial' held its magnitude fixed and swept a corrupted-subset fraction while
    every other type swept magnitude, so its cells were not comparable with theirs
    and it contributed several times as many rows to the win tally. A corrupted
    subset is what outlier_p01/p05/p10 now are, with the fraction carried by the
    condition, so every condition sweeps one shared level grid.
    """
    check('n13 PARTIAL_STRENGTH gone', not hasattr(n13, 'PARTIAL_STRENGTH'))
    check('n13 PARTIAL_FRACTIONS gone', not hasattr(n13, 'PARTIAL_FRACTIONS'))
    summary = n13._compute_summary(
        {'per_noise_type': {'gaussian': {}, 'outlier_p05': {}}})
    check('n13 every condition shares one sweep axis',
          set(summary['sweep_axis_by_noise_type'].values()) == {'noise_level'},
          str(summary['sweep_axis_by_noise_type']))
    check('n13 partial_axis_note removed', 'partial_axis_note' not in summary)


def main():
    tests = [
        test_n13_private_injector_is_gone,
        test_n13_outlier_mask_marks_the_contaminated_subset,
        test_n13_shape_only_conditions_perturb_every_row,
        test_n13_conditions_distinct_at_matched_dose,
        test_n13_clf_flip_models_stay_in_vocabulary,
        test_n13_dnn_in_unweighted_set,
        test_n13_one_axis_for_every_condition,
    ]
    for t in tests:
        print(f"\n== {t.__name__} ==")
        t()
    print("\n" + "=" * 50)
    if _failures:
        print(f"FAILURES ({len(_failures)}): {_failures}")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == '__main__':
    main()
