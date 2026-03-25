"""Shared configuration and runner for per-dataset similarity experiments.

Each dataset gets its own script (e.g. herg.py, logd.py) that calls
run_all() with its rep list from the pipeline readiness tiers.
Cross-dataset experiment (S09) stays in its own file and consumes
the per-dataset JSON outputs.

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
    get_hybrid_features_and_weights, load_experiment_results,
)

# Reuse dataset rep configs — single source of truth
from experiments.complexity.complexity_helpers import (
    DATASET_REPS, PRELIMINARY,
)


# ── Similarity-specific config overrides ───────────────────────────────────
# Similarity experiments use the same preliminary config as complexity,
# plus some similarity-specific parameters.

SIMILARITY_CONFIG = dict(PRELIMINARY)
# Similarity experiments don't sweep allocations/importance — they use the
# best from find_best_hybrid. These are similarity-specific defaults:
SIMILARITY_CONFIG.update({
    'weight_methods': ['lightgbm_gain'],
    'k_values': [5, 10],
    'n_sample': 5000,
    'knn_conditions': ['hybrid_unweighted', 'hybrid_weighted'],
})


# ── Single-dataset experiment list ─────────────────────────────────────────
# In order. S02 first (produces shared_search for S03-S07).
# S09 is cross-dataset — run separately.

SINGLE_DATASET_EXPERIMENTS = [
    's01',   # similarity baseline (standalone)
    's02',   # hybrid similarity (produces shared_search)
    's03',   # feature concentration (uses shared_search)
    's04',   # activity cliffs (uses shared_search)
    's05',   # kNN prediction (uses shared_search)
    's06',   # applicability domain (uses shared_search)
    's07',   # noise bridge (uses shared_search)
    's08',   # optimal dimension (standalone)
    's10',   # importance downstream (standalone)
    's11',   # bayesian optimization (standalone)
    's12',   # cross-domain cliffs (standalone, generic)
    's13',   # clustering + retrieval (standalone)
]


def _import_experiment(exp_id: str):
    """Dynamically import a similarity experiment module by ID (e.g. 's01')."""
    module_map = {
        's01': 's01_similarity_baseline',
        's02': 's02_hybrid_similarity',
        's03': 's03_feature_concentration',
        's04': 's04_activity_cliffs',
        's05': 's05_knn_prediction',
        's06': 's06_applicability_domain',
        's07': 's07_noise_bridge',
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.similarity.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name used by save_experiment_results."""
    name_map = {
        's01': 's01_similarity_baseline',
        's02': 's02_hybrid_similarity',
        's03': 's03_feature_concentration',
        's04': 's04_activity_cliffs',
        's05': 's05_knn_prediction',
        's06': 's06_applicability_domain',
        's07': 's07_noise_bridge',
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    }
    return name_map.get(exp_id, exp_id)


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None):
    """Load data once and run all single-dataset similarity experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: SIMILARITY_CONFIG).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
    """
    if config is None:
        config = dict(SIMILARITY_CONFIG)
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
    print(f"SIMILARITY EXPERIMENTS: {dataset_name.upper()}")
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

    # ── Shared search (produced by S02, consumed by S03-S07) ──
    # Run the search once up front if any of S02-S07 are in the experiment list.
    shared_search = None
    needs_search = any(e in experiments for e in ['s02', 's03', 's04', 's05', 's06', 's07'])
    if needs_search:
        print(f"\n  Running shared hybrid search (speed={config['search_speed']})...")
        try:
            shared_search = get_hybrid_features_and_weights(
                data['reps'], data['labels'], data.get('groups'),
                search_speed=config['search_speed'],
                weight_method=config['weight_methods'][0],
            )
            print(f"  Shared search done (dim={shared_search[0].shape[1]})")
        except Exception as e:
            print(f"  Shared search FAILED: {e}")

    # ── Run experiments in order ──
    results = {}
    for exp_id in experiments:
        print(f"\n{'─' * 70}")
        t_exp = time.time()

        try:
            mod = _import_experiment(exp_id)
        except (ImportError, KeyError) as e:
            print(f"  SKIP {exp_id}: import failed ({e})")
            continue

        # Build kwargs — pass shared_search and similarity-specific params
        kwargs = {}
        if exp_id in ('s02', 's03', 's04', 's05', 's06', 's07'):
            kwargs['shared_search'] = shared_search
        if exp_id in ('s01', 's02', 's08'):
            kwargs['n_sample'] = config.get('n_sample', 5000)
        if exp_id == 's02':
            kwargs['weight_methods'] = config.get('weight_methods', ['lightgbm_gain'])
        if exp_id == 's05':
            kwargs['k_values'] = config.get('k_values', [5, 10])
            kwargs['conditions'] = config.get('knn_conditions')
        if exp_id == 's08':
            kwargs['budgets'] = config.get('budgets', [25, 50, 100, 200])

        try:
            exp_result = mod.run(data, config, output_dir, dataset_name, **kwargs)
            results[exp_id] = exp_result

            # Save per-experiment JSON
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
