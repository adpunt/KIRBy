"""The noise column: the injector, the level key, and where a fold's rows sit in it.

WHY THIS IS ITS OWN MODULE. These three things are what the fold-independence gate
needs, and nothing else. They used to live in
`alternative_data_noise_robustness.py`, which imports torch -- so the gate could
not run on a login node (`libtorch_cuda.so: failed to map segment`) or in any
environment without a GPU build, even though it fits nothing and needs no data.
A gate that cannot be run where the operator is standing is not a gate.

The runner imports all of it from here, so there is still ONE definition of each.
"""
import zlib

import numpy as np

from noiseInject import CONDITIONS, NoiseInjectorRegression


class RunIntegrityError(RuntimeError):
    """The run is misconfigured, or produced output that cannot be interpreted.

    A separate class because the per-condition loop below wraps every experiment
    in `except Exception`, logs it and carries on -- which is right for a model
    that failed to fit on one representation, and completely wrong for "the
    zero-level negative control is missing" or "this condition wrote no rows".
    Those must stop the job, so they are raised as this and re-raised there.
    """


def _level_key(level):
    """The canonical text form of a noise level, used to seed from its VALUE.

    `%.12g` round-trips every level any grid in this file uses, and gives the
    same text for 0.6, 0.60 and 6e-1, so a level named three ways is one level.
    """
    return f"{float(level):.12g}"


class _ColumnView:
    """The full label column, plus where this fold's rows sit inside it.

    WHY THIS EXISTS. Noise used to be injected on `y_train` -- a different vector
    in every fold -- so a molecule drew different noise depending on which fold it
    landed in: 1,688 of 2,400 molecule-fold pairs on a 600-molecule five-fold
    split (RERUN_PLAN.md 3.3a). QM9 draws once over its training column before
    anything is split, so a molecule's corruption is a property of the molecule.
    With five folds the only fold-free equivalent is to draw over the whole
    column and index (3.3b, settled by the author 2026-09-04).

    The check below is the point of the class. Indexing is only correct if
    `y_column[train_rows]` really is the labels being fitted -- and the Gaussian
    process path SUBSAMPLES its training rows, so its positions are
    `train_idx[train_idx_local][gp_row_idx]` and not `train_idx[train_idx_local]`.
    Get that wrong and every GP molecule silently receives another molecule's
    noise, which is the shape of the QM9 defect in RERUN_PLAN.md 2.10b. So the
    labels are compared rather than trusted.
    """

    def __init__(self, y_column, groups_column, train_rows, test_rows,
                 y_train, y_test, who, val_rows=None, y_val=None):
        if y_column is None or train_rows is None or test_rows is None:
            raise ValueError(
                f"{who} needs y_column, train_rows and test_rows: the noise is "
                f"drawn once over the whole label column and indexed by each "
                f"molecule's position in the dataset (RERUN_PLAN.md 3.3b). "
                f"There is no per-fold fallback -- that was the defect.")
        self.y = np.asarray(y_column, dtype=float)
        self.groups = None if groups_column is None else np.asarray(groups_column)
        self.train_rows = np.asarray(train_rows, dtype=int)
        self.test_rows = np.asarray(test_rows, dtype=int)
        if self.groups is not None and len(self.groups) != len(self.y):
            raise ValueError(
                f"{who}: groups_column has {len(self.groups)} entries against "
                f"{len(self.y)} labels; they must both span the whole dataset.")
        self.val_rows = None if val_rows is None else np.asarray(val_rows, dtype=int)
        pairs = [(self.train_rows, y_train, 'train'), (self.test_rows, y_test, 'test')]
        if self.val_rows is not None:
            pairs.append((self.val_rows, y_val, 'val'))
        for rows, y_fold, name in pairs:
            got = self.y[rows]
            want = np.asarray(y_fold, dtype=float)
            if got.shape != want.shape or not np.allclose(got, want, equal_nan=True):
                raise RunIntegrityError(
                    f"{who}: y_column[{name}_rows] is not the {name} labels this "
                    f"call was given ({got.shape} vs {want.shape}). The row "
                    f"positions do not point at these molecules, so every one of "
                    f"them would be given another molecule's noise. If this is "
                    f"the Gaussian-process path, the positions are "
                    f"train_idx[train_idx_local][gp_row_idx].")


def _injector_for(condition, level, stream='training'):
    """A fresh injector for one condition at one noise LEVEL.

    Three properties this exists to guarantee, all learned the hard way.

    REPRODUCIBILITY. `NoiseInjectorRegression` advances its generator on every
    draw, so re-using one injector across the level grid makes the realisation at
    a level depend on how many levels ran before it.

    GRID-INDEPENDENCE. The seed is derived from the level's VALUE, never from its
    POSITION in the grid. It used to be the position: `--sigmas 0.0 0.6` gave 0.6
    index 1 while `--sigmas 0.0 0.3 0.6` gave it index 2, so the two runs
    corrupted the same molecules differently at the same nominal level, and any
    gap-filling resubmission drew noise unrelated to the original run's. Merging
    those files put two different corruptions of the same molecule into one
    column with nothing to tell them apart.

    SEPARATE STREAMS. The validation split draws from its own stream, so
    corrupting validation cannot change the training corruption.

    NOTE: this changes the realised noise relative to any run made before it,
    because the seed formula changed. Results from the two are not mergeable.
    """
    base = 42 if stream == 'training' else 1337
    # zlib.crc32, NOT hash(): Python randomises string hashing per process, so
    # hash() here would have made the noise different on every run while the
    # docstring above promised the opposite.
    digest = zlib.crc32(f"{stream}|{condition}|{_level_key(level)}".encode('utf-8'))
    seed = (base * 1000003 + digest) & 0x7FFFFFFF
    # WHO gets hit is seeded WITHOUT the level. The shape draw varies per level;
    # the affected set must not, or `noise_pattern` -- the level-free column the
    # whole zero-level subtraction rests on -- describes a different set of
    # molecules than the injection at that level. Measured on a 120-molecule
    # fixture before this: for outlier_p10 the pattern and the realised scale had
    # Spearman -0.10 and their top-15% sets overlapped 3 of 18; for
    # grouped_wider, 0.375 at one level and -0.250 at another
    # (RERUN_PLAN.md 2.13). The Rust injector has always been level-independent
    # here.
    selection = (base * 1000003
                 + zlib.crc32(f"{stream}|{condition}|selection".encode('utf-8'))) \
        & 0x7FFFFFFF
    return NoiseInjectorRegression.from_condition(condition, random_state=seed,
                                                  selection_state=selection)
