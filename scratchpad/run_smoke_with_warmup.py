"""Run tests/scripts/smoke_real_dataset.py with the local lightgbm warm-up applied.

On this laptop the first model fit after loading data kills the interpreter with no
traceback (two copies of a threading library get loaded). Fitting a two-tree model
BEFORE any data is loaded avoids it. The smoke script itself has no warm-up, so it
cannot be run directly here; on the server it can.

Usage: python scratchpad/run_smoke_with_warmup.py [dataset]
"""
import os
import runpy
import sys

try:
    import numpy as _np
    import lightgbm as _lgb
    _lgb.LGBMRegressor(n_estimators=2, verbose=-1).fit(
        _np.zeros((8, 2)), _np.arange(8.0))
    print("[warmup] lightgbm warmed up")
except Exception as _e:
    print(f"[warmup] skipped: {_e}")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SMOKE = os.path.join(_HERE, '..', 'tests', 'scripts', 'smoke_real_dataset.py')

sys.argv = [_SMOKE] + sys.argv[1:]
runpy.run_path(_SMOKE, run_name='__main__')
