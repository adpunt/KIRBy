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
    load_data, save_experiment_results, convert_numpy, should_skip_experiment,
    DOWNSTREAM_MODELS_NOISE, INTERNAL_EVALUATORS,
)

# Reuse dataset rep configs — single source of truth
from experiments.complexity.complexity_helpers import DATASET_REPS


# ── Experiment list ────────────────────────────────────────────────────────
# All standalone, no ordering constraints.

SINGLE_DATASET_EXPERIMENTS = [
    'n01',   # cliff-stratified prediction error (sub-4 only; sub-1/2/3 cut)
    'n02',   # importance-weighted sample weighting
    'n07',   # noise remediation strategies
    'n09',   # LNL method survey with hybrids (12 methods)
    'n11',   # noise estimation x remediation grid
    'n12',   # feature importance-weighted distance
    'n13',   # alternative noise distributions
    'n14',   # activity-cliff landscape survey (gate; no downstream models)
    'n15',   # clean-searched vs noise-searched hybrid blueprint (oracle advantage)
]
# CUT 2026-07-07 (irredeemable by design — see AUDIT_NOISE.md):
# n03 sas_landscape, n04 gp_loo_detection, n05 noise_injection_benchmark,
# n08 estimation_survey, n10 hybrid_neighborhood_lnl.

# N06 (BindingDB ground-truth) is dataset-specific — not in the per-dataset list.
# Run separately: python tests/experiments/noise/n06_bindingdb_ground_truth.py


# ── Noisy-label hybrid search grid ─────────────────────────────────────────
# THE noise conditions for which a fairly-searched (noise-aware) hybrid blueprint
# is precomputed and cached. This list is the single source of truth: the
# precompute builds exactly these, and an experiment may only ask for a spec that
# is in here. If the two ever diverge, an experiment gets a blueprint fitted to a
# noise draw that exists nowhere in its own run — and nothing crashes. Hence: one
# list, imported by both.
#
# Levels are TARGET EFFECTIVE NOISE (mean |dy| as a fraction of the label SD), so a
# level means the same thing on every dataset — unlike the four ad-hoc, mutually
# incompatible `_inject_noise` definitions the suite grew.
#
# ONE noise_seed. The blueprint should not swing with the random draw; if it does,
# that is itself the finding, and it is what the clean-vs-noisy experiment measures.
NOISY_SEARCH_LEVELS = (0.05, 0.10, 0.20)
NOISY_SEARCH_SEED = 0


def noisy_search_specs(task_type: str):
    """The NoiseSpec list the noisy precompute builds, for this task type."""
    from kirby.noise_spec import default_spec
    is_clf = (task_type == 'classification')
    return [default_spec(is_clf=is_clf, target_effective_noise=lvl,
                         noise_seed=NOISY_SEARCH_SEED)
            for lvl in NOISY_SEARCH_LEVELS]


def _import_experiment(exp_id: str):
    """Dynamically import a noise experiment module by ID (e.g. 'n01')."""
    module_map = {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n06': 'n06_bindingdb_ground_truth',
        'n07': 'n07_remediation',
        'n09': 'n09_lnl_survey',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
        'n14': 'n14_activity_cliff_landscape',
        'n15': 'n15_clean_vs_noisy_hybrid',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.noise.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name used by save_experiment_results."""
    name_map = {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n07': 'n07_remediation',
        'n09': 'n09_lnl_survey',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
        'n14': 'n14_activity_cliff_landscape',
        'n15': 'n15_clean_vs_noisy_hybrid',
    }
    return name_map.get(exp_id, exp_id)


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None,
            pruning_mode: str = 'default_pruning',
            force_rerun: bool = False):
    """Load data once and run all single-dataset noise experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: built from the DEFAULT_* constants).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
        pruning_mode: One of PRUNING_MODES. 'default_pruning' = legacy
            single-stage prune. 'aggressive_pruning' = F16-validated multi-
            stage pruning (checkpoints=[25,50,100], td=0.5) applied globally
            to all models, no n_samples gate. F16: ~58% fold-0 eval reduction.
    """
    if config is None:
        from kirby.find_best_hybrid import (
            DEFAULT_ALLOCATION_METHODS, DEFAULT_IMPORTANCE_METHODS, DEFAULT_N_FOLDS,
        )
        # Production precompute uses FULL_BUDGETS (SCREENING_BUDGETS was
        # test-only). FULL_BUDGETS = [10,25,50,100,250,500,1000,2000] CONTAINS
        # the hybrid budgets n10 requests ([25,100,250]); SCREENING_BUDGETS =
        # [50,500,1000,2000] does not, so n10's strict budget guard skipped all
        # three and the hybrid-vs-base comparison (n10's whole point) came back
        # silently empty. Both the in-memory precompute below and the SLURM
        # disk cache (tests/precompute_hybrids.py, speed='full') already build
        # FULL_BUDGETS, so this only makes config['budgets'] truthful and keeps
        # n10's requested hybrid budgets reachable.
        from experiments._shared import FULL_BUDGETS
        config = {
            'n_splits': DEFAULT_N_FOLDS,
            'search_speed': 'full',
            'allocations': DEFAULT_ALLOCATION_METHODS,
            'importance_methods': DEFAULT_IMPORTANCE_METHODS,
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': FULL_BUDGETS,
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

    # ── Resolve pruning_mode + apply n_samples gate ──
    from experiments._shared import resolve_pruning_mode, apply_pruning_gate
    _sched, _ovr_template, _min_n = resolve_pruning_mode(pruning_mode)
    _model_overrides = apply_pruning_gate(
        _ovr_template, len(data['labels']), _min_n)
    print(f"\n  Pruning mode: {pruning_mode}")
    if _model_overrides is not None:
        print(f"    overrides active for: {sorted(_model_overrides.keys())}")
    elif _ovr_template is not None:
        print(f"    overrides gated off (n={len(data['labels'])} < {_min_n})")

    # ── Precompute hybrid searches (general reps) ──
    _task_type = data['task_type']
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
            dataset_name=dataset_name, tier='general',
            adaptive_pruning_schedule=_sched,
            model_pruning_aggressiveness=_model_overrides,
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
            dataset_name=dataset_name, tier='top',
            adaptive_pruning_schedule=_sched,
            model_pruning_aggressiveness=_model_overrides,
        )
        data['precomputed_searches_top'] = precompute_top
    else:
        data['precomputed_searches_top'] = {}

    # ── Precompute NOISY-LABEL hybrid searches ──
    # Every search above was run on the CLEAN labels. An experiment that then
    # corrupts the labels and reports "the hybrid copes better" cannot separate a
    # genuinely better hybrid from one that was BUILT knowing the answers — the base
    # representations get no such look. (This is why the July audit made n11/n12
    # carry an "ORACLE-BIASED / best-case, not a fair baseline" warning, and why n09
    # re-searched under noise inline with cache=False, paying for it with a crippled
    # rf-only search.)
    #
    # These searches TRAIN on a deterministic noisy copy of the labels and SCORE on
    # the clean ones, at full search fidelity, cached. find_best_hybrid derives its
    # own cache tier from the spec, so the noisy lineage sits BESIDE the clean one
    # and never overwrites it.
    #
    # Cost: |NOISE_SPECS| x the clean precompute. Opt out with KIRBY_NOISY_SEARCH=0.
    data['precomputed_searches_noisy'] = {}
    data['precomputed_searches_top_noisy'] = {}
    if os.environ.get('KIRBY_NOISY_SEARCH', '1') != '0':
        specs = noisy_search_specs(data['task_type'])
        for spec in specs:
            if len(reps_data) >= 2:
                print(f"\n  Precomputing NOISY hybrid searches "
                      f"(general, noise={spec.spec_id})...")
                data['precomputed_searches_noisy'][spec.spec_id] = (
                    precompute_hybrid_searches(
                        reps=reps_data, labels=data['labels'],
                        groups=data.get('groups'), config=config,
                        augmentations=data.get('augmentations'),
                        task_type=_task_type,
                        domain=data.get('domain', 'molecular'),
                        dataset_name=dataset_name, tier='general',
                        adaptive_pruning_schedule=_sched,
                        model_pruning_aggressiveness=_model_overrides,
                        label_noise=spec,
                    ))
            if len(top_tier_reps) >= 2:
                print(f"\n  Precomputing NOISY hybrid searches "
                      f"(top-tier, noise={spec.spec_id})...")
                data['precomputed_searches_top_noisy'][spec.spec_id] = (
                    precompute_hybrid_searches(
                        reps=top_tier_reps, labels=data['labels'],
                        groups=data.get('groups'), config=config,
                        augmentations=data.get('augmentations'),
                        task_type=_task_type,
                        domain=data.get('domain', 'molecular'),
                        dataset_name=dataset_name, tier='top',
                        adaptive_pruning_schedule=_sched,
                        model_pruning_aggressiveness=_model_overrides,
                        label_noise=spec,
                    ))

    # ── Run experiments ──
    results = {}
    for exp_id in experiments:
        # Resume: skip an experiment whose output JSON already exists (mirror
        # complexity). Lets a walled/restarted job advance instead of re-running
        # finished experiments. force_rerun=True recomputes regardless.
        _out_json = os.path.join(
            output_dir, _experiment_full_name(exp_id), f"{dataset_name}.json")
        if not force_rerun and os.path.exists(_out_json):
            print(f"  RESUME {exp_id}: already done — skipping ({_out_json})")
            results[exp_id] = {'resumed': True}
            continue
        print(f"\n{'─' * 70}")
        t_exp = time.time()

        # Large-n / domain scope policy: write a {'skipped':...} marker so the
        # wave-audit counts it as intentionally-skipped (see _shared.should_skip_
        # experiment + REPRODUCIBILITY_LEDGER.md §E).
        _skip = should_skip_experiment(exp_id, data)
        if _skip is not None:
            print(f"  SKIP {exp_id}: {_skip['reason']}")
            results[exp_id] = _skip
            save_experiment_results(_skip, output_dir,
                                    _experiment_full_name(exp_id), dataset_name)
            continue

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
