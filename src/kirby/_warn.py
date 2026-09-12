"""One warning class the engine's own blanket filters do not swallow.

`warnings.filterwarnings('ignore')` runs at import in `kirby/find_best_hybrid.py`
and again in `tests/experiments/_shared.py`, for sklearn's convergence and
deprecation noise. It takes KIRBy's own warnings with it, and those are the ones
that say a representation could not be scored, an internal evaluator was
substituted, or a budget was split equally under another method's name. Every one
of those went unread.

Raise engine warnings as `KirbyWarning` and they survive the filter. PRINT them
as well: a warning shown once per process is not enough when the thing it is
reporting happened on every fold, and a cluster log is read long after the fact.

Printing EVERY time stopped working once the hybrid's feature selection was
refitted inside each fold: that runs a whole search per fold per model, and each
one emits the budget-not-filled line for every budget, allocation, importance
method and evaluator it tries. The first few of each distinct message print in
full, the rest are counted, and the totals print at exit — so the fact and its
frequency both survive without burying the scores. KIRBY_WARN_REPEATS sets the
limit; 0 restores printing every one.

Stdlib only, and it imports nothing from the rest of the package, so both
`hybrid.py` and `find_best_hybrid.py` can import it at module scope without a
cycle — and without pulling lightgbm in before torch, which segfaults on macOS.
"""
import atexit
import os
import re
import warnings
from collections import Counter


class KirbyWarning(UserWarning):
    """A warning about the engine's own behaviour, not a library's."""


# How many times each distinct warning is printed before it is only counted.
# Printing every time was right while a search ran once per dataset. Scoring a
# hybrid with the feature selection refitted inside each fold runs it once per
# fold per model, and every budget, allocation, importance method and evaluator
# inside each of those emits the budget-not-filled line -- thousands of lines
# that bury the per-fold scores. The first few still print, the rest are
# counted, and the totals are printed at exit, so nothing is lost.
_PRINT_LIMIT = int(os.environ.get('KIRBY_WARN_REPEATS', '3'))
_counts: Counter = Counter()
_shapes: dict = {}


def _shape(message: str) -> str:
    """Collapse a message to its family, so numbers do not make it distinct."""
    return re.sub(r'\d+', '#', message)


def warn_once(message: str, *, quiet: bool = False) -> None:
    """Warn as a KirbyWarning and print it, so it survives any filter.

    The first `KIRBY_WARN_REPEATS` occurrences of each distinct message print in
    full; after that they are counted and reported at exit. Set
    KIRBY_WARN_REPEATS=0 for no limit.

    `quiet=True` warns without printing — for the handful of places that already
    print their own, better-formatted line.
    """
    warnings.warn(message, KirbyWarning, stacklevel=2)
    if quiet:
        return
    key = _shape(message)
    _counts[key] += 1
    _shapes.setdefault(key, message)
    n = _counts[key]
    if _PRINT_LIMIT <= 0 or n <= _PRINT_LIMIT:
        print(f"  !! {message}")
    elif n == _PRINT_LIMIT + 1:
        print(f"  !! (further occurrences of that warning are counted, not "
              f"printed; the total is reported at exit)")


def warning_counts() -> dict:
    """Every distinct warning raised so far, and how many times."""
    return {_shapes[k]: v for k, v in _counts.items()}


@atexit.register
def _report_suppressed() -> None:
    hidden = {k: v for k, v in _counts.items()
              if _PRINT_LIMIT > 0 and v > _PRINT_LIMIT}
    if not hidden:
        return
    print("\n  !! warnings raised more than "
          f"{_PRINT_LIMIT} times (printed once each here):")
    for k, v in sorted(hidden.items(), key=lambda kv: -kv[1]):
        # The collapsed shape, with # where the numbers were. Quoting the first
        # message instead would attach one occurrence's numbers to all of them.
        print(f"     {v:6d}x  {k}")


# The blanket filters are installed at import of the modules above, so this has
# to be re-applied by anything that wants KirbyWarning shown. Applied here too so
# importing this module is enough.
warnings.filterwarnings('once', category=KirbyWarning)
