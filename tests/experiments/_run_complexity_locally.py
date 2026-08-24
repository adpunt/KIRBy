"""Local end-to-end runner for complexity experiments (small real dataset).

Actually executes each experiment's run() on a small dataset with locally-loadable
reps (aac/blosum/onehot — no foundation models, no server) and a lightweight
'fast' search config. Verifies the fixes produce real output end-to-end.

Usage: python tests/experiments/_run_complexity_locally.py <exp1> <exp2> ...
       (no args = run the default fix-verification set)

Env knobs (smoke):
  KIRBY_LOCAL_DS   dataset name (default flab_thermo_jain)
  KIRBY_LOCAL_REPS comma-separated rep list (default aac,blosum,onehot)
  KIRBY_SMOKE_N    row cap (honored by _shared._smoke_subsample)
"""
import sys
import os
import warnings
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import experiments._shared as _shared  # noqa: E402
from experiments._shared import load_data  # noqa: E402
import kirby.find_best_hybrid as _fbh  # noqa: E402

# ── SMOKE-ONLY runtime shrink (local; never touches server runs) ──────────
# precompute_hybrid_searches hardcodes models=None and
# FULL_BUDGETS, ignoring config. On a laptop the full grid (8 model types ×
# 5 evaluators × 9 allocations × 3 importance methods, incl. O(n^3) GP) makes
# even a 200-row/420-dim search take many minutes. To prove the code paths we
# trim the in-memory search axes the precompute reads. These are runtime
# monkeypatches of imported objects (NOT edits to _shared.py / src), and only
# take effect in this driver process — server jobs import fresh and are
# unaffected.
_shared.FULL_BUDGETS = [25]
# c02 imports INTERNAL_EVALUATORS from _shared at its (lazy) import time; patch
# the source so the copy it binds is the trimmed list (drops slow svm/mlp/match).
_shared.INTERNAL_EVALUATORS = ['rf']
# Keep only the fast tree model so precompute trains one model type, not eight.
_fbh.DEFAULT_MODELS = {'rf': _fbh.DEFAULT_MODELS['rf']}
_fbh.DEFAULT_INTERNAL_EVALUATORS = ['rf']
_fbh.DEFAULT_ALLOCATION_METHODS = _fbh.DEFAULT_ALLOCATION_METHODS[:1]
# treeshap works for both regression and classification (the list's first
# entry 'deeplift' is slower and clf-flaky). NOTE: on herg classification the
# hybrid search still returns no scored config even with treeshap (see the
# COMPLEXITY_REPORT shared-code item) — that is a find_best_hybrid clf-path
# issue, not something this trim can fix.
_fbh.DEFAULT_IMPORTANCE_METHODS = ['treeshap']

DATASET = os.environ.get('KIRBY_LOCAL_DS', 'flab_thermo_jain')
REPS = os.environ.get('KIRBY_LOCAL_REPS', 'aac,blosum,onehot').split(',')


def fast_config(n_splits=2, budgets=(25,)):
    # Only the 'full' defaults exist; keep it tractable locally via tiny
    # budgets, 2 folds, and a 1-model set. Trim the search axes to 1-2 each so
    # the local run finishes in minutes (this is a smoke run, not a real result).
    return {
        'n_splits': n_splits,
        'allocations': _fbh.DEFAULT_ALLOCATION_METHODS[:2],
        'importance_methods': _fbh.DEFAULT_IMPORTANCE_METHODS[:1],
        'internal_evaluators': _fbh.DEFAULT_INTERNAL_EVALUATORS[:1],
        'budgets': list(budgets),
        'models': ['rf'],
    }


# Per-experiment lean kwargs so the local run stays fast.
def kw_for(exp):
    if exp == 'c10_noise_robustness':
        return {'sigmas': [0.0, 0.3], 'strategies': ['legacy']}
    if exp == 'c06_within_family_gradient':
        return {'families': ['RF']}
    if exp == 'c19_importance_weighted_training':
        # c19 defaults to budget=100; precompute in smoke only builds budget 25
        # (FULL_BUDGETS=[25]), so the pool lookup at 100 misses. Pin to 25 so the
        # hybrid input space is actually populated. (Server uses full budgets.)
        return {'budget': 25}
    return {}


def main(argv):
    exps = argv or ['c01_rep_characterization', 'c10_noise_robustness']
    data = load_data(DATASET, rep_names=REPS, rep_preset='fast')
    print(f"\nLoaded {DATASET}: n={len(data['labels'])}, reps={list(data['reps'])}, "
          f"task={data['task_type']}\n")
    config = fast_config()
    out = os.environ.get('KIRBY_LOCAL_OUT', 'tests/results/_local_smoke')
    os.makedirs(out, exist_ok=True)

    for exp in exps:
        print('=' * 70)
        print(f"RUNNING {exp}  (ds={DATASET}, task={data['task_type']})")
        print('=' * 70)
        mod = importlib.import_module(f'experiments.complexity.{exp}')
        kw = kw_for(exp)
        try:
            res = mod.run(data, config, out, DATASET, **kw)
            if isinstance(res, dict):
                if 'error' in res:
                    print(f"  -> RETURNED ERROR: {res.get('error')} "
                          f"| extra={ {k: res[k] for k in res if k != 'error'} }")
                else:
                    print(f"  -> OK keys={list(res.keys())[:12]}")
            else:
                print(f"  -> OK (returned {type(res).__name__})")
            print()
        except Exception as e:
            import traceback
            print(f"  -> RAISED {type(e).__name__}: {e}")
            traceback.print_exc()
            print()


if __name__ == '__main__':
    main(sys.argv[1:])
