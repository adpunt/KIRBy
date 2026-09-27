#!/usr/bin/env python
"""A molecule's injected noise must not depend on which fold it lands in.

THE GATE for RERUN_PLAN.md 3.3b, settled by the author on 2026-09-04: the
laboratory matches QM9. QM9 has one train/test split and injects once over the
training label column before anything is split, so a molecule's corruption is a
property of the molecule. The laboratory has five folds; the only fold-free
equivalent is to draw once over the whole column and index by each molecule's
position in the dataset.

Before the change the injection ran on `y_train`, a different vector in every
fold. RERUN_PLAN.md 3.3a measured the damage on a 600-molecule five-fold split:
1,688 of 2,400 molecule-fold pairs received different noise from the same
molecule in an earlier fold.

AMENDED 2026-09-27. The whole-column draw also took the noise amount, the
censoring limit and the grouped-wider families from every label, held-out ones
included, where the NoiseInject paper's Methods take all three from the clean
TRAINING labels. The runner now calls kirby.noise_spec.inject_training_rows: the
random DRAW is still one per molecule, but the constants that scale it come from
each fold's training labels, so the injected VALUE differs between folds by that
constant. What stays a property of the molecule is the draw.

This test fits nothing, needs no data files and does not import torch, so it runs
on a login node. It runs both schemes on the same folds and asserts:

  * the OLD per-fold scheme draws differently across folds -- so the test is
    measuring something real and would not pass vacuously;
  * the NEW scheme gives every molecule exactly one random draw (the injected
    error divided by its noise scale), in every condition;
  * the NEW scheme's training noise does not change when the held-out labels do,
    and its amount is the level times the TRAINING labels' standard deviation;
  * _ColumnView refuses a mis-threaded index rather than silently handing a
    molecule someone else's noise -- which is the Gaussian-process subsample
    trap in 3.3b.

    python tests/test_noise_is_a_property_of_the_molecule.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# noise_column.py, NOT the runner: the runner imports torch, so this gate could
# not run on a login node at all -- `libtorch_cuda.so: failed to map segment`,
# reported 2026-09-04. It fits nothing and needs no data, so it should run
# anywhere. The runner imports the same three objects from the same module, so
# there is still one definition of each.
import noise_column as runner

sys.path.insert(0, str(HERE.parent / 'src'))
from kirby.noise_spec import inject_training_rows  # noqa: E402

N_MOL = 600
N_FOLDS = 5
LEVEL = 0.5
# One condition of each kind: plain, and the two that key on scaffold groups.
CONDITIONS = ['gaussian', 'grouped_wider', 'grouped_shifted']

FAILURES = []


def fixture():
    rng = np.random.default_rng(11)
    y = rng.normal(2.5, 0.9, N_MOL)
    # 60 scaffold families, so GroupKFold has something to hold together.
    # INTEGER ids: the grouped conditions count group sizes with np.bincount, so
    # a string id raises inside the injector. The pipeline passes integer codes
    # (`ng_train`), and the fixture has to match what actually runs.
    groups = np.array([i % 60 for i in range(N_MOL)], dtype=np.int64)
    return y, groups


def draw_per_fold(y, groups, condition, per_fold):
    """{molecule position: set of random draws it received across the folds}.

    The draw is the injected error divided by the molecule's noise scale: the
    level-free random number, with the training-derived constants divided out.

    per_fold=True reproduces the old scheme -- inject on the fold's own training
    labels, a fresh draw per fold. per_fold=False is what the runner does now,
    inject_training_rows.
    """
    seen = {}
    gkf = GroupKFold(n_splits=N_FOLDS)
    for train_rows, _test_rows in gkf.split(y, y, groups):
        inj = runner._injector_for(condition, LEVEL)
        if per_fold:
            res = inj.inject_verbose(y[train_rows], LEVEL * float(np.std(y[train_rows])),
                                     groups=groups[train_rows])
        else:
            res = inject_training_rows(inj, y, train_rows, LEVEL, groups=groups)
        draw = np.asarray(res.epsilon) / np.asarray(res.noise_scale)
        for pos, d in zip(train_rows, draw):
            seen.setdefault(int(pos), set()).add(round(float(d), 9))
    return seen


def held_out_labels_leak(y, groups, condition):
    """Problems found when the held-out labels change: [] if none."""
    problems = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (train_rows, test_rows) in enumerate(gkf.split(y, y, groups)):
        base = inject_training_rows(runner._injector_for(condition, LEVEL),
                                    y, train_rows, LEVEL, groups=groups)
        y2 = y.copy()
        y2[test_rows] = y2[test_rows] * 3.0 + 10.0
        moved = inject_training_rows(runner._injector_for(condition, LEVEL),
                                     y2, train_rows, LEVEL, groups=groups)
        if not np.array_equal(base.epsilon, moved.epsilon):
            problems.append(f"{condition} fold {fold}: the training noise changed "
                            f"when only the held-out labels did")
        want = LEVEL * float(np.std(y[train_rows]))
        if not np.isclose(base.target_dose_label_units, want):
            problems.append(f"{condition} fold {fold}: asked for "
                            f"{base.target_dose_label_units:.6f}, the level times the "
                            f"training standard deviation is {want:.6f}")
    return problems


def report(condition, seen, label):
    disagreeing = sum(1 for v in seen.values() if len(v) > 1)
    pairs = sum(len(v) for v in seen.values())
    print(f"    {label:16s} {condition:18s} "
          f"{disagreeing:4d} of {len(seen)} molecules carry more than one value "
          f"({pairs} molecule-fold values)")
    return disagreeing


def main():
    y, groups = fixture()
    print(f"  {N_MOL} molecules, {N_FOLDS} folds, level {LEVEL}, "
          f"{len(set(groups))} scaffold groups\n")

    for condition in CONDITIONS:
        old = report(condition, draw_per_fold(y, groups, condition, True), 'per fold (old)')
        new = report(condition, draw_per_fold(y, groups, condition, False), 'training (new)')

        if new != 0:
            FAILURES.append(
                f"{condition}: {new} molecule(s) received more than one distinct "
                f"random draw. A molecule's draw is supposed to be a property of "
                f"the molecule (3.3b).")
        leaks = held_out_labels_leak(y, groups, condition)
        print(f"    held-out labels changed: "
              f"{'training noise unchanged, amount from training labels' if not leaks else leaks[0]}")
        FAILURES.extend(leaks)
        if old == 0:
            FAILURES.append(
                f"{condition}: the OLD per-fold scheme agreed with itself, so this "
                f"test proves nothing here. Either the fixture is degenerate or "
                f"the injector stopped depending on its input vector.")
        print()

    # The Gaussian-process trap: positions that do not point at these labels.
    print("  _ColumnView on a mis-threaded index:")
    rows = np.arange(100)
    try:
        runner._ColumnView(y, groups, rows + 7, np.arange(100, 150),
                           y[rows], y[100:150], 'the GP path')
    except runner.RunIntegrityError as exc:
        first = str(exc).splitlines()[0]
        print(f"    refused, correctly: {first[:96]}")
    else:
        FAILURES.append(
            "_ColumnView accepted train_rows that point at the wrong molecules. "
            "That is the Gaussian-process subsample defect going through silently.")

    # ...and that it accepts the right one.
    try:
        runner._ColumnView(y, groups, rows, np.arange(100, 150),
                           y[rows], y[100:150], 'the ordinary path')
        print("    accepted a correct index")
    except Exception as exc:                                   # noqa: BLE001
        FAILURES.append(f"_ColumnView rejected a correct index: {exc}")

    print()
    if FAILURES:
        print(f"FAIL — {len(FAILURES)} problem(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PASS — every molecule carries one random draw across all folds, in every "
          "condition; the training noise ignores the held-out labels; and a "
          "mis-threaded index is refused.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
