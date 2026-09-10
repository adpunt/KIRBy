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

Stdlib only, and it imports nothing from the rest of the package, so both
`hybrid.py` and `find_best_hybrid.py` can import it at module scope without a
cycle — and without pulling lightgbm in before torch, which segfaults on macOS.
"""
import warnings


class KirbyWarning(UserWarning):
    """A warning about the engine's own behaviour, not a library's."""


def warn_once(message: str, *, quiet: bool = False) -> None:
    """Warn as a KirbyWarning and print it, so it survives any filter.

    `quiet=True` warns without printing — for the handful of places that already
    print their own, better-formatted line.
    """
    warnings.warn(message, KirbyWarning, stacklevel=2)
    if not quiet:
        print(f"  !! {message}")


# The blanket filters are installed at import of the modules above, so this has
# to be re-applied by anything that wants KirbyWarning shown. Applied here too so
# importing this module is enough.
warnings.filterwarnings('once', category=KirbyWarning)
