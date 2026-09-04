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

This test fits nothing, needs no data files and does not import torch, so it runs
on a login node. It runs both schemes on the same folds and asserts:

  * the OLD scheme disagrees with itself across folds -- so the test is measuring
    something real and would not pass vacuously;
  * the NEW scheme gives every molecule exactly one value, in every condition;
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


def epsilon_per_fold(y, groups, condition, per_fold):
    """{molecule position: set of epsilons it received across the folds}.

    per_fold=True reproduces the old scheme -- inject on the fold's own training
    labels. per_fold=False is the new one -- inject over the whole column and
    index.
    """
    seen = {}
    gkf = GroupKFold(n_splits=N_FOLDS)
    sd = float(np.std(y))
    for train_rows, _test_rows in gkf.split(y, y, groups):
        inj = runner._injector_for(condition, LEVEL)
        if per_fold:
            res = inj.inject_verbose(y[train_rows], LEVEL * float(np.std(y[train_rows])),
                                     groups=groups[train_rows])
            eps = np.asarray(res.epsilon)
        else:
            res = inj.inject_verbose(y, LEVEL * sd, groups=groups)
            eps = np.asarray(res.epsilon)[train_rows]
        for pos, e in zip(train_rows, eps):
            seen.setdefault(int(pos), set()).add(round(float(e), 12))
    return seen


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
        old = report(condition, epsilon_per_fold(y, groups, condition, True), 'per fold (old)')
        new = report(condition, epsilon_per_fold(y, groups, condition, False), 'one column (new)')

        if new != 0:
            FAILURES.append(
                f"{condition}: {new} molecule(s) received more than one distinct "
                f"noise value under the one-column draw. A molecule's corruption "
                f"is supposed to be a property of the molecule (3.3b).")
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
    print("PASS — every molecule carries one injected noise value across all folds, "
          "in every condition, and a mis-threaded index is refused.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
