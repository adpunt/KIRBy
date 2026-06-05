#!/usr/bin/env python3
"""Tiny local smoke harness for noise experiments (n01-n13).

NOT a results run. Loads a SMALL real dataset with 2-3 cheap, locally-computable
reps (no foundation models / no server data), builds a minimal find_best_hybrid
config, and runs one experiment's run() end-to-end to surface code breakage.

Usage:
    python tests/experiments/noise/_smoke_harness.py n05            # reg (buchwald)
    python tests/experiments/noise/_smoke_harness.py n05 --clf      # clf (humanness)
    python tests/experiments/noise/_smoke_harness.py n12 --reg --subsample 300
"""
import os
# Single-thread native libs BEFORE numpy/torch/lightgbm import — avoids the
# macOS OpenMP double-load segfault (exit 139) seen running find_best_hybrid
# locally. Must precede all scientific imports.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import argparse
import warnings
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import experiments._shared as _shared  # noqa: E402
from experiments._shared import (  # noqa: E402
    load_data, INTERNAL_EVALUATORS,
)

# Smoke-only: shrink the budget grid so every find_best_hybrid path (precompute
# AND experiments' own budget=None search calls, which default to FULL_BUDGETS)
# stays tiny. This is a SPEED hack for breakage-hunting, NOT a results config.
_shared.FULL_BUDGETS = [25]

# Smoke-only: shrink the 'full' SPEED_TIER search space so BOTH precompute AND
# the experiments' own create_hybrid rebuilds stay cheap. Without this the
# winning blueprint can pick importance_method='deeplift'/'treeshap' (TF/SHAP)
# and internal_evaluator='mlp'/'svm', and get_fitted_pool_hybrid →
# create_hybrid then RE-RUNS that expensive importance/eval on the full data
# every time an experiment builds its hybrid — burning ~4-5 min of setup per
# experiment BEFORE its sweep loop even starts (observed locally 2026-06-05).
# Restricting to lightgbm_gain importance + rf evaluator + 2 allocations + 2
# search folds makes the rebuild ~seconds. This ONLY mutates the in-process
# SPEED_TIERS dict for this smoke harness; server (full) runs import the
# module fresh and are unaffected. NOT a results config.
try:
    from kirby.find_best_hybrid import SPEED_TIERS as _ST
    _ST['full']['importance_methods'] = ['lightgbm_gain']
    _ST['full']['allocation_methods'] = ['rank_linear', 'performance_weighted']
    _ST['full']['internal_evaluators'] = ['rf']
    _ST['full']['n_folds'] = 2
except Exception as _e:  # pragma: no cover
    print(f"  (smoke) could not shrink SPEED_TIERS['full']: {_e}")

# Cheap, local, no-server rep sets per task type.
# Low-dim, locally-computable reps so create_hybrid / importance stay fast.
REG_DATASET = 'buchwald_hartwig'
REG_REPS = ['diff_maccs', 'reactant_pdv', 'onehot_components']  # 167/208/45 D
CLF_DATASET = 'humanness_therapeutic'
CLF_REPS = ['blosum', 'onehot']  # 5600 D each (no cheaper antibody clf rep)


def tiny_config():
    """Minimal find_best_hybrid config — keeps precompute fast."""
    return {
        'n_splits': 2,
        # 'full' is the ONLY valid SPEED_TIERS key — the legacy 'fast' tier was
        # removed 2026-04-25 and now raises KeyError inside find_best_hybrid,
        # which silently broke n03's get_hybrid_features_and_weights path
        # (caught by the 2026-05-23 audit). Speed comes from the [25]-budget
        # hack below, not from a tier name. Keep this 'full' so the smoke
        # exercises the SAME tier production runs use.
        'search_speed': 'full',
        'allocations': ['greedy'],
        'importance_methods': ['lightgbm_gain'],
        'internal_evaluators': ['rf'],
        'budgets': [25],
        'models': ['rf'],
    }


def build_data(clf=False, subsample=None, seed=0):
    ds = CLF_DATASET if clf else REG_DATASET
    reps = CLF_REPS if clf else REG_REPS
    t0 = time.time()
    data = load_data(ds, rep_preset='all', rep_names=reps)
    # NaN guard (mirror noise_helpers)
    for k, X in list(data['reps'].items()):
        if np.isnan(X).any():
            data['reps'][k] = np.nan_to_num(X, nan=0.0)
    if subsample and len(data['labels']) > subsample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(data['labels']), subsample, replace=False))
        data['labels'] = np.asarray(data['labels'])[idx]
        data['reps'] = {k: X[idx] for k, X in data['reps'].items()}
        if data.get('groups') is not None:
            data['groups'] = np.asarray(data['groups'])[idx]
        for sk in ('heavy_seqs', 'light_seqs'):
            if data.get(sk) is not None:
                data[sk] = [data[sk][i] for i in idx]
    data['augmentations'] = None
    print(f"  loaded {ds} ({reps}) n={len(data['labels'])} task={data['task_type']} "
          f"in {time.time()-t0:.1f}s")
    return data, ds


def precompute(data, dataset_name, config):
    from experiments._shared import precompute_hybrid_searches
    reps = data['reps']
    # rf-only evaluator => no TensorFlow/mlp in the search loop (the dominant
    # local cost). Experiments read data['precomputed_searches'] so they reuse
    # this rather than re-running a full mlp search.
    data['precomputed_searches'] = precompute_hybrid_searches(
        reps=reps, labels=data['labels'], groups=data.get('groups'),
        config=config, augmentations=None, task_type=data['task_type'],
        domain=data.get('domain', 'molecular'),
        dataset_name=dataset_name, tier='general',
        internal_evaluators=['rf'])
    data['precomputed_searches_top'] = {}
    data['top_tier_reps'] = {}
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp')
    ap.add_argument('--clf', action='store_true')
    ap.add_argument('--reg', action='store_true')
    ap.add_argument('--subsample', type=int, default=400)
    ap.add_argument('--no-precompute', action='store_true')
    args = ap.parse_args()

    clf = args.clf
    print(f"=== SMOKE {args.exp} ({'clf' if clf else 'reg'}) ===")
    data, ds = build_data(clf=clf, subsample=args.subsample)
    config = tiny_config()

    if not args.no_precompute:
        t0 = time.time()
        data = precompute(data, ds, config)
        print(f"  precompute done in {time.time()-t0:.1f}s")
    else:
        data['precomputed_searches'] = {}
        data['precomputed_searches_top'] = {}
        data['top_tier_reps'] = {}

    from experiments.noise.noise_helpers import _import_experiment
    mod = _import_experiment(args.exp)
    out_dir = '/tmp/noise_smoke'
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    try:
        result = mod.run(data, config, out_dir, ds)
        print(f"\n=== {args.exp} RAN OK in {time.time()-t0:.1f}s ===")
        return result
    except Exception as e:
        import traceback
        print(f"\n=== {args.exp} CRASHED: {type(e).__name__}: {e} ===")
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
