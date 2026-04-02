"""Shared configuration and runner for per-dataset noise experiments.

Each dataset gets its own script (e.g. herg.py, logd.py) that calls
run_all() with its rep list from the pipeline readiness tiers.
12 experiments (N01-N05, N07-N13) are single-dataset and standalone
with no inter-dependencies. N06 (BindingDB) is run separately.

Experiments:
  N01-N08: Original noise suite (cliffs, weighting, SAS, GP-LOO, detection,
           BindingDB, remediation, estimation survey)
  N09: LNL method survey with hybrids (12 training-loop methods)
  N10: Hybrid neighborhood for similarity-dependent LNL
  N11: Noise estimation × remediation grid (full confirmation report grid)
  N12: Feature importance-weighted distance for noise estimation
  N13: Alternative noise distributions (uniform, multiplicative, heavy-tailed, partial)

Config: RF / LightGBM / MLP, 1-fold scaffold CV, fast search.
Rep lists reused from complexity_helpers.DATASET_REPS.
"""

import sys
import os
import time
from pathlib import Path

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))         # tests/experiments/

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy,
    DOWNSTREAM_MODELS_NOISE, INTERNAL_EVALUATORS,
)

# Reuse dataset rep configs — single source of truth
from experiments.complexity.complexity_helpers import DATASET_REPS


# ── Experiment list ────────────────────────────────────────────────────────
# All standalone, no ordering constraints.

SINGLE_DATASET_EXPERIMENTS = [
    'n01',   # activity cliff framework (Dablander)
    'n02',   # importance-weighted sample weighting
    'n03',   # SAS landscape analysis
    'n04',   # GP-LOO cliff-aware noise detection
    'n05',   # noise injection benchmark (all 15 detectors)
    'n07',   # noise remediation strategies
    'n08',   # noise estimation survey (all 15 estimators)
    'n09',   # LNL method survey with hybrids (12 methods)
    'n10',   # hybrid neighborhood for similarity-dependent LNL
    'n11',   # noise estimation x remediation grid
    'n12',   # feature importance-weighted distance
    'n13',   # alternative noise distributions
]

# N06 (BindingDB ground-truth) is dataset-specific — not in the per-dataset list.
# Run separately: python tests/experiments/noise/n06_bindingdb_ground_truth.py


def _import_experiment(exp_id: str):
    """Dynamically import a noise experiment module by ID (e.g. 'n01')."""
    module_map = {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n03': 'n03_sas_landscape',
        'n04': 'n04_gp_loo_detection',
        'n05': 'n05_noise_injection_benchmark',
        'n06': 'n06_bindingdb_ground_truth',
        'n07': 'n07_remediation',
        'n08': 'n08_estimation_survey',
        'n09': 'n09_lnl_survey',
        'n10': 'n10_hybrid_neighborhood_lnl',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.noise.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name used by save_experiment_results."""
    name_map = {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n03': 'n03_sas_landscape',
        'n04': 'n04_gp_loo_detection',
        'n05': 'n05_noise_injection_benchmark',
        'n07': 'n07_remediation',
        'n08': 'n08_estimation_survey',
        'n09': 'n09_lnl_survey',
        'n10': 'n10_hybrid_neighborhood_lnl',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
    }
    return name_map.get(exp_id, exp_id)


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None):
    """Load data once and run all single-dataset noise experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: fast tier from SPEED_TIERS).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
    """
    if config is None:
        from kirby.find_best_hybrid import SPEED_TIERS
        from experiments._shared import SCREENING_BUDGETS
        tier = SPEED_TIERS['fast']
        config = {
            'n_splits': tier['n_folds'],
            'search_speed': 'fast',
            'allocations': tier['allocation_methods'],
            'importance_methods': tier['importance_methods'],
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': SCREENING_BUDGETS,
            'models': DOWNSTREAM_MODELS_NOISE,
        }
    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)

    # Resolve rep list
    if rep_names is None:
        ds_reps = DATASET_REPS.get(dataset_name)
        if ds_reps is None:
            raise ValueError(
                f"No rep list for '{dataset_name}'. Pass rep_names explicitly "
                f"or add it to DATASET_REPS in complexity_helpers.py."
            )
        rep_names = ds_reps['general']

    print(f"\n{'=' * 70}")
    print(f"NOISE EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Config:      {config['n_splits']} fold, {config['models']}, speed={config['search_speed']}")
    print(f"Reps ({len(rep_names)}): {', '.join(rep_names)}")
    print(f"Experiments: {experiments}")
    print(f"Output:      {output_dir}")

    # ── Load data ONCE ──
    t0 = time.time()
    print(f"\nLoading {dataset_name}...")
    data = load_data(dataset_name, rep_preset='all', rep_names=rep_names)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # NaN guard: some reps produce NaN for certain sequences. Replace with 0 and warn.
    import numpy as np
    reps_data = data.get('reps', {})
    for rname, X in list(reps_data.items()):
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            pct = 100 * nan_count / X.size
            print(f"  WARNING: {rname} has {nan_count} NaN values ({pct:.1f}%) — imputing 0")
            reps_data[rname] = np.nan_to_num(X, nan=0.0)

    # ── Load augmentation reps ──
    aug_rep_names = DATASET_REPS.get(dataset_name, {}).get('augmentation', [])
    if aug_rep_names:
        missing = [r for r in aug_rep_names if r not in reps_data]
        if missing:
            print(f"  Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing)
            aug_reps = {}
            for rn, arr in aug_data['reps'].items():
                if int(np.isnan(arr).sum()) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                aug_reps[rn] = arr
            data['augmentations'] = aug_reps
        else:
            data['augmentations'] = {r: reps_data[r] for r in aug_rep_names}
        print(f"  Augmentations ({len(data['augmentations'])}): "
              f"{', '.join(f'{k}={v.shape[1]}' for k, v in data['augmentations'].items())}")
    else:
        data['augmentations'] = None

    # ── Precompute hybrid searches (general reps) ──
    _task_type = data.get('task_type', 'regression')
    if len(reps_data) >= 2:
        from experiments._shared import precompute_hybrid_searches
        print(f"\n  Precomputing hybrid searches (general, {len(reps_data)} reps)...")
        precompute_results = precompute_hybrid_searches(
            reps=reps_data,
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
        )
        data['precomputed_searches'] = precompute_results
    else:
        data['precomputed_searches'] = {}

    # ── Precompute hybrid searches (top-tier reps) ──
    top_tier_names = DATASET_REPS.get(dataset_name, {}).get('top_tier', [])
    top_tier_reps = {k: v for k, v in reps_data.items() if k in top_tier_names}
    data['top_tier_reps'] = top_tier_reps
    if len(top_tier_reps) >= 2:
        print(f"\n  Precomputing hybrid searches (top-tier, {len(top_tier_reps)} reps)...")
        precompute_top = precompute_hybrid_searches(
            reps=top_tier_reps,
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
        )
        data['precomputed_searches_top'] = precompute_top
    else:
        data['precomputed_searches_top'] = {}

    # ── Run experiments ──
    results = {}
    for exp_id in experiments:
        print(f"\n{'─' * 70}")
        t_exp = time.time()

        try:
            mod = _import_experiment(exp_id)
        except (ImportError, KeyError) as e:
            print(f"  SKIP {exp_id}: import failed ({e})")
            continue

        try:
            exp_result = mod.run(data, config, output_dir, dataset_name)
            results[exp_id] = exp_result

            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(exp_result, output_dir, exp_full_name, dataset_name)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': str(e)}

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE: {dataset_name} — {len(results)} experiments in {total:.0f}s")
    print(f"{'=' * 70}")

    return results
