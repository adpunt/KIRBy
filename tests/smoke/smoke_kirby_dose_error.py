"""If an injection is ever made fatal, it must reach the surface.

    python tests/smoke/smoke_kirby_dose_error.py

The injector checks how much noise it actually delivered. **By default it warns
and carries on** -- the band is three standard errors, so a working injector lands
outside it on about 1% of draws at these dataset sizes, and stopping there would
discard a sound run over an unlucky draw (author's decision 2026-08-28,
RERUN_PLAN.md 2.14a). The delivered amount is on every row, so nothing is lost.

`DoseError` is what it raises if a caller ever makes a missed draw fatal, and this
file guards the one thing that would make that setting useless. Every cell in
`run_dataset` is wrapped in `except Exception`, which prints one line and carries
on -- right for a model that failed to fit, and wrong for a wrong injection: caught
there, the cell would simply be missing from the results while the job finished
green, and the run would report a noise level it never received.
`RunIntegrityError` is already re-raised above that handler for exactly this
reason; `DoseError` sits beside it.

Four checks, driven through the real `run_dataset` rather than read off the source:

  1. a DoseError raised inside the injection escapes run_dataset, and is not merely
     printed at the cell on the way past
  2. CONTROL -- an ordinary model failure at the same point is still caught at the
     cell and printed, so check 1 is testing the re-raise and not just "the runner
     propagates errors". The run then stops on its own integrity check, because a
     swallowed cell wrote no rows -- which is the very thing a swallowed DoseError
     would have looked like, only with a level that silently never arrived.

See RERUN_PLAN.md 2.14a and NOISE_DESIGN.md 5.1d in qsar_qm_models.
"""
import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

_argv, sys.argv = sys.argv, ['smoke_kirby_dose_error.py']
_spec = importlib.util.spec_from_file_location(
    'runner', ROOT / 'tests' / 'alternative_data_noise_robustness.py')
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
sys.argv = _argv

from noiseInject import DoseError                                   # noqa: E402

T0 = time.time()
FAILS = []
NCHECK = [0]


def check(name, ok, detail=''):
    NCHECK[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ''))
    if not ok:
        FAILS.append(name)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# A small, real, parseable set: enough molecules for five scaffold folds without
# spending a minute on featurisation.
RNG = np.random.default_rng(0)
SMILES = []
for i in range(150):
    ring = ['c1ccccc1', 'C1CCCCC1', 'c1ccncc1', 'C1CCNCC1', 'c1ccsc1'][i % 5]
    SMILES.append(f"{ring}" + "C" * (1 + i % 7) + ['O', 'N', 'F', 'Cl'][i % 4])
LABELS = 7.0 + 0.4 * RNG.normal(size=len(SMILES))


class _Boom:
    """Stands in for an injector and fails the way we want to test."""

    def __init__(self, exc):
        self.exc = exc

    def inject_verbose(self, *a, **k):
        raise self.exc

    def noise_scale(self, y, *a, **k):
        return np.zeros(len(np.asarray(y)))


def run_with(exc):
    """Run one model on one representation with the injection failing.

    Returns (what escaped, what the run printed).
    """
    real = P._injector_for
    tmp = Path(tempfile.mkdtemp(prefix='dose_error_smoke_'))
    log = io.StringIO()
    try:
        P._injector_for = lambda condition, sigma, **k: (
            _Boom(exc) if float(sigma) > 0 else real(condition, sigma, **k))
        with contextlib.redirect_stdout(log):
            P.run_dataset('ChEMBL-hERG-Ki', SMILES, LABELS, tmp,
                          model_filter=['RF'], rep_filter=['ECFP4'],
                          sigma_levels=[0.0, 0.5], conditions=['gaussian'],
                          oof_folds=0, noise_validation=False)
        return None, log.getvalue()
    except BaseException as e:                       # noqa: BLE001 - that is the check
        return e, log.getvalue()
    finally:
        P._injector_for = real
        shutil.rmtree(tmp, ignore_errors=True)


section("A failed injection must reach the surface")

MARK = 'the injector delivered an amount nobody asked for'
raised, out = run_with(DoseError(MARK))
check("a DoseError from the injection escapes run_dataset",
      isinstance(raised, DoseError),
      f"got {type(raised).__name__ if raised else 'no exception -- the cell vanished'}")
check("and it is not merely printed at the cell on the way past",
      f"ERROR: {MARK}" not in out,
      'the broad handler did not see it')

section("CONTROL -- an ordinary model failure is still caught at the cell")

MARK2 = 'this model could not be fitted'
raised, out = run_with(ValueError(MARK2))
check("a plain ValueError at the same point is caught and printed, not raised",
      not isinstance(raised, ValueError) and f"ERROR: {MARK2}" in out,
      f"escaped as {type(raised).__name__}" if isinstance(raised, ValueError)
      else 'caught at the cell, as it should be')
check("the run then stops on its own integrity check, having written no rows",
      isinstance(raised, RuntimeError),
      f"got {type(raised).__name__ if raised else 'a clean finish, which would hide the gap'}")

print(f"\n{'=' * 70}")
print(f"{NCHECK[0]} checks in {time.time() - T0:.1f}s")
if FAILS:
    print("FAILED: " + '; '.join(FAILS))
    sys.exit(1)
print("ALL CHECKS PASSED")
