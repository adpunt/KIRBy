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
# default search constants for this smoke harness; server (full) runs import the
# module fresh and are unaffected. NOT a results config.
try:
    import kirby.find_best_hybrid as _fbh
    _fbh.DEFAULT_IMPORTANCE_METHODS = ['lightgbm_gain']
    _fbh.DEFAULT_ALLOCATION_METHODS = ['rank_linear', 'performance_weighted']
    _fbh.DEFAULT_INTERNAL_EVALUATORS = ['rf']
    _fbh.DEFAULT_N_FOLDS = 2
except Exception as _e:  # pragma: no cover
    print(f"  (smoke) could not shrink the default search config: {_e}")

# Cheap, local, no-server rep sets per task type.
# Low-dim, locally-computable reps so create_hybrid / importance stay fast.
#
# Smoke-only env override (server runs never set these): the default reg path
# (buchwald_hartwig + reactant_pdv) recomputes RDKit physicochemical descriptors
# over all 3955 reactions on a COLD rep cache, which takes ~7 min before the
# experiment even starts — far too slow for a breakage smoke. KIRBY_SMOKE_REG_DS
# lets the §V smoke pick a tiny, fast-rep reg dataset (e.g. ginkgo_heparin n=94
# with aac/blosum/onehot — no slow reaction descriptors). Default is unchanged.
REG_DATASET = os.environ.get('KIRBY_SMOKE_REG_DS', 'buchwald_hartwig')
_default_reg_reps = ('diff_maccs', 'reactant_pdv', 'onehot_components')  # 167/208/45 D
REG_REPS = list(os.environ.get('KIRBY_SMOKE_REG_REPS',
                               ','.join(_default_reg_reps)).split(','))
CLF_DATASET = os.environ.get('KIRBY_SMOKE_CLF_DS', 'humanness_therapeutic')
CLF_REPS = list(os.environ.get('KIRBY_SMOKE_CLF_REPS', 'blosum,onehot').split(','))


def tiny_config():
    """Minimal find_best_hybrid config — keeps precompute fast."""
    return {
        'n_splits': 2,
        # Speed tiers were removed entirely. Search cheapness here comes from
        # the [25]-budget hack below, exercising the same default path as production.
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
        # Subsample every row-aligned list-like field, not just antibody seqs —
        # molecular experiments (e.g. n14's ECFP4-Tanimoto reference) read
        # data['smiles'], which must stay aligned with the subsampled labels.
        for sk in ('heavy_seqs', 'light_seqs', 'smiles', 'sequences', 'seqs',
                   'sample_ids'):
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

    # NOISY lineage — a hybrid blueprint searched on the noisy TRAIN labels and
    # scored on clean ones, so the hybrid gets no more access to the truth than a
    # base rep does. n15 compares it against the clean blueprint. ONE spec here
    # (the smoke wants seconds, not the full 3-level grid); the real precompute
    # builds noise_helpers.NOISY_SEARCH_LEVELS.
    data['precomputed_searches_noisy'] = {}
    data['precomputed_searches_top_noisy'] = {}
    if os.environ.get('KIRBY_SMOKE_NOISY_SEARCH', '1') != '0':
        for spec in smoke_noise_specs(data['task_type']):
            data['precomputed_searches_noisy'][spec.spec_id] = (
                precompute_hybrid_searches(
                    reps=reps, labels=data['labels'], groups=data.get('groups'),
                    config=config, augmentations=None, task_type=data['task_type'],
                    domain=data.get('domain', 'molecular'),
                    dataset_name=dataset_name, tier='general',
                    internal_evaluators=['rf'], label_noise=spec))
    return data


def smoke_noise_specs(task_type):
    """One noise condition for the smoke (the real grid is 3 levels)."""
    from kirby.noise_spec import default_spec
    return [default_spec(is_clf=(task_type == 'classification'),
                         target_effective_noise=0.20, noise_seed=0)]


def _shrink_experiment_module(mod):
    """Smoke-only: collapse an experiment's module-level grid constants so its
    noise-seed × sigma × method sweeps complete in seconds rather than minutes.

    Several noise experiments (n05/n09/n10/n11/n12/n13) loop over
    NOISE_SEEDS=[0,1,2] and a SIGMAS list, running ~15 distance/GP-LOO detectors
    per (sigma, seed) over several rep spaces. On a tiny smoke dataset the
    per-detector work is still GP-LOO-heavy, so a single sigma's 3 seeds can take
    minutes — the sweep never reaches its first printed summary line within a
    breakage smoke's wall budget. This GATE only fires when KIRBY_SMOKE_N is set
    (the §V smoke env); a server/production run never sets it, so the full grids
    are untouched. We mutate ONLY the imported module's module-level attributes
    in-process (NOT the source files), keeping every other family's runs and the
    real grids unaffected. Purely a speed hack for breakage hunting — NOT a
    results config.
    """
    if not os.environ.get('KIRBY_SMOKE_N'):
        return
    shrunk = []
    if hasattr(mod, 'NOISE_SEEDS') and len(getattr(mod, 'NOISE_SEEDS')) > 1:
        mod.NOISE_SEEDS = [mod.NOISE_SEEDS[0]]
        shrunk.append('NOISE_SEEDS->[0]')
    if hasattr(mod, 'SIGMAS') and len(getattr(mod, 'SIGMAS')) > 1:
        # Keep a single representative non-zero sigma where possible so the
        # detection task is still non-degenerate.
        _sig = [s for s in mod.SIGMAS if s] or list(mod.SIGMAS)
        mod.SIGMAS = [_sig[0]]
        shrunk.append(f'SIGMAS->[{mod.SIGMAS[0]}]')
    if shrunk:
        print(f"  (smoke) shrank {mod.__name__.split('.')[-1]} grids: "
              f"{', '.join(shrunk)}")


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
    _shrink_experiment_module(mod)
    out_dir = '/tmp/noise_smoke'
    os.makedirs(out_dir, exist_ok=True)
    # Smoke-only run kwargs: some experiments read their sigma/seed grids from
    # run() **kwargs (n05 'sigmas', n11 'noise_seeds') rather than module-level
    # constants, so the module-attr shrink above can't reach them. All noise
    # run() signatures take **kwargs and ignore unknown keys, so passing these
    # is safe for every experiment. GATED on KIRBY_SMOKE_N — server runs never
    # pass them and keep their full grids.
    run_kwargs = {}
    if os.environ.get('KIRBY_SMOKE_N'):
        run_kwargs = {'sigmas': [0.3], 'noise_seeds': [0]}
        # n15 sweeps NoiseSpecs, not sigmas — hand it the same single smoke spec the
        # noisy precompute above actually built, or it would ask the pool for a spec
        # with no cached search and cold-build one live.
        run_kwargs['noise_specs'] = smoke_noise_specs(data['task_type'])
    t0 = time.time()
    try:
        result = mod.run(data, config, out_dir, ds, **run_kwargs)
        print(f"\n=== {args.exp} RAN OK in {time.time()-t0:.1f}s ===")
        return result
    except Exception as e:
        import traceback
        print(f"\n=== {args.exp} CRASHED: {type(e).__name__}: {e} ===")
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
