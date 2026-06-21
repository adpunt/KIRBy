#!/usr/bin/env python3
"""Precompute hybrid searches for all datasets × rep tiers × aug conditions.

For each dataset in DATASET_REPS, runs find_best_hybrid with cache=True for:
  1. general reps + no aug
  2. general reps + uncapped aug
  3. top_tier reps + no aug
  4. top_tier reps + uncapped aug
  5. top_tier reps + capped aug

(general/capped dropped 2026-05-22 — the iterative growth loop was ~9× more
expensive than uncapped on the large general-rep pool; consumers redirected
to top/capped which still provides capped semantics on the curated subset.)

Results are cached to ~/.cache/kirby/searches/{dataset}__{tier}__{aug}__{speed}.pkl
(or under $KIRBY_CACHE_DIR). Per-rep baselines are deduped under
~/.cache/kirby/baselines/ and shared across all conditions of the same
dataset. Experiments load these via get_precomputed_search() — instant,
no recompute.

Usage:
    python tests/precompute_hybrids.py caco2                  # one dataset
    python tests/precompute_hybrids.py caco2 herg_ki logd     # multiple
    python tests/precompute_hybrids.py --domain molecular     # all molecular
    python tests/precompute_hybrids.py --domain antibody      # all antibody
    python tests/precompute_hybrids.py --all                  # everything
    python tests/precompute_hybrids.py caco2 --tier top       # top_tier only
    python tests/precompute_hybrids.py caco2 --speed full     # full search
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from experiments._shared import (
    load_data, SCREENING_BUDGETS, FULL_BUDGETS, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, TOP_CAPPED_MAX_N,
)
from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.find_best_hybrid import find_best_hybrid, _cache_stats_summary, _reset_cache_stats


# Pruning modes live in tests/experiments/_shared.py so both this script
# and the experiment helpers (run_all in complexity/similarity/noise) share
# a single source of truth.
from experiments._shared import (
    PRUNING_MODES, resolve_pruning_mode, apply_pruning_gate,
)
_resolve_pruning_mode = resolve_pruning_mode  # backwards-compat alias


# =============================================================================
# HYBRID REGISTRY — maps dataset name → published best configs
# =============================================================================

def _registry_path():
    cache_root = Path(os.environ.get('KIRBY_CACHE_DIR',
                      str(Path.home() / '.cache' / 'kirby')))
    return cache_root / 'hybrid_registry.json'


def _load_registry():
    path = _registry_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_registry(registry):
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _update_registry(dataset_name, tier, aug_strategy, result):
    """Add a SearchResult's best configs to the registry.

    Registry schema:
    {
      "dataset_name": {
        "metric": "rmse",
        "lower_is_better": true,
        "configs": {
          "tier/aug_strategy": {
            "model_name": {
              "budget": {
                "score_mean": float,
                "score_std": float,
                "blueprint": {...}  (Blueprint.to_dict())
              }
            }
          }
        }
      }
    }

    Every entry carries metric + lower_is_better. No guessing.
    """
    registry = _load_registry()
    ds_entry = registry.setdefault(dataset_name, {
        'metric': result.metric_name,
        'lower_is_better': result.lower_is_better,
        'configs': {},
    })
    # Update metric info (in case it changed)
    ds_entry['metric'] = result.metric_name
    ds_entry['lower_is_better'] = result.lower_is_better

    config_key = f"{tier}/{aug_strategy}"
    config_entry = ds_entry['configs'].setdefault(config_key, {})

    for model_name, budget_dict in result.best_by_model_budget.items():
        model_entry = config_entry.setdefault(model_name, {})
        for budget, r in budget_dict.items():
            bp = r.get('blueprint')
            model_entry[str(budget)] = {
                'score_mean': round(r['score_mean'], 6),
                'score_std': round(r.get('score_std', 0), 6),
                'n_features': r.get('n_features', 0),
                'allocation_method': r.get('allocation_method', ''),
                'importance_method': r.get('importance_method', ''),
                'internal_evaluator': r.get('internal_evaluator', ''),
                'blueprint': bp.to_dict() if bp else None,
            }

    _save_registry(registry)


# Domain detection (matches complexity_helpers.get_aug_groups)
def _get_domain(dataset_name):
    if dataset_name.startswith(('tcga_', 'luad', 'brca')):
        return 'tcga'
    elif dataset_name.startswith(('davis', 'kiba', 'bindingdb')):
        return 'dta'
    elif dataset_name.startswith(('buchwald', 'doyle', 'suzuki', 'bh_')):
        return 'reaction'
    elif any(dataset_name.startswith(p) for p in (
            'flab_', 'abagym_', 'abcov_', 'ddg_', 'tz_', 'humanness',
            'ginkgo_', 'graphinity_')):
        return 'antibody'
    elif dataset_name.startswith(('nt_', 'gb_')):
        return 'nucleotide'
    return 'molecular'


def _get_metric(domain, task_type, labels, dataset_name=None):
    """Resolve metric for precompute.

    Prefers DATASET_REGISTRY when dataset_name is given and registered.
    Falls back to the historical task_type+domain heuristic otherwise —
    keeps unregistered datasets working without blocking precompute.
    """
    if dataset_name is not None:
        from kirby.registry import lookup as _registry_lookup
        info = _registry_lookup(dataset_name)
        if info is not None:
            if info.task_type != task_type:
                raise ValueError(
                    f"Registry/loader task_type mismatch for "
                    f"'{dataset_name}': loader={task_type!r}, "
                    f"registry={info.task_type!r}."
                )
            return info.metric
    # Fallback heuristic.
    if domain == 'nucleotide':
        return 'mcc'
    if task_type == 'classification':
        n_classes = len(np.unique(labels))
        return 'f1_weighted' if n_classes > 2 else 'auc'
    if domain == 'dta':
        return 'ci'
    if domain == 'antibody':
        return 'spearman'
    return 'rmse'


def _datasets_for_domain(domain):
    result = []
    for ds in DATASET_REPS:
        if _get_domain(ds) == domain:
            result.append(ds)
    return sorted(result)


def _condition_cost_rank(condition):
    """Deterministic cost proxy for a precompute condition.

    Higher = more expensive (should run FIRST when cost-ordering so a
    wall-clock timeout sacrifices the cheap tail, not the costly variants).

    Pure ordering signal — does NOT change which conditions are computed.
    The proxy is built from the same inputs every run, so the sort is
    stable and reproducible:

      1. augmentation strategy:  capped (refits the model up to
         max_candidates per slot AND runs a phase-B replace pass) is the
         most expensive, then uncapped (single fill pass), then none.
      2. augmentation-pool width: more aug features = a wider candidate
         ranking + more A/B-test fits.
      3. base-rep count: more reps = more per-rep baselines + allocation work.

    Ties broken by the (tier, strat) label so equal-cost conditions keep a
    deterministic relative order.
    """
    tier, reps, augs, strat = condition
    strat_cost = {'none': 0, 'uncapped': 1, 'capped': 2}.get(strat, 1)
    n_aug = len(augs) if augs else 0
    n_reps = len(reps) if reps else 0
    return (strat_cost, n_aug, n_reps, tier, strat)


def _order_conditions_expensive_first(conditions):
    """Return conditions sorted most-expensive-first (descending cost rank).

    Behaviour-preserving: returns a permutation of the SAME list — same
    elements, same count — only the order differs. Verified by
    test_condition_ordering_preserves_set below.
    """
    return sorted(conditions, key=_condition_cost_rank, reverse=True)


def test_condition_ordering_preserves_set():
    """Cost-ordering must be a pure permutation (same SET, different order).

    Run with: python tests/precompute_hybrids.py --selftest-ordering
    """
    import numpy as _np
    r = {'a': _np.zeros((4, 3)), 'b': _np.zeros((4, 2))}
    aug_small = {'x': _np.zeros((4, 5))}
    aug_big = {'x': _np.zeros((4, 5)), 'y': _np.zeros((4, 7)), 'z': _np.zeros((4, 9))}
    conditions = [
        ('general', r, None, 'none'),
        ('general', r, aug_small, 'uncapped'),
        ('top', r, None, 'none'),
        ('top', r, aug_small, 'uncapped'),
        ('top', r, aug_small, 'capped'),
        ('top+borderline', r, aug_big, 'uncapped'),
    ]
    ordered = _order_conditions_expensive_first(conditions)

    # 1. SET preserved (same elements, same count) — compare by identity
    #    because the tuples contain unhashable numpy-array dicts.
    assert sorted(id(c) for c in conditions) == sorted(id(c) for c in ordered), \
        "ordering changed the set of conditions"
    assert len(ordered) == len(conditions)

    # 2. capped runs before its uncapped sibling, which runs before none
    labels = [(t, s) for t, _, _, s in ordered]
    assert labels.index(('top', 'capped')) < labels.index(('top', 'uncapped')), \
        "capped should sort before uncapped"
    assert labels.index(('top', 'uncapped')) < labels.index(('top', 'none')), \
        "uncapped should sort before none"
    assert labels.index(('general', 'uncapped')) < labels.index(('general', 'none')), \
        "uncapped should sort before none (general tier)"

    # 3. capped (most expensive strategy) is first overall
    assert labels[0] == ('top', 'capped'), \
        f"expected top/capped first, got {labels[0]}"

    # 4. idempotent + deterministic
    assert _order_conditions_expensive_first(ordered) == ordered, \
        "ordering is not idempotent"
    print("test_condition_ordering_preserves_set: PASS")
    return True


def precompute_dataset(dataset_name, speed='full', tiers=('general', 'top'),
                       models=None, condition_filter=None, distance=False,
                       adaptive_pruning=True, adaptive_pruning_schedule=None,
                       model_pruning_aggressiveness=None,
                       min_samples_for_overrides=0):
    """Run all precompute conditions for one dataset.

    Args:
        distance: If True, run distance-evaluator conditions (for similarity/noise
            experiments that use DISTANCE_EVALUATORS + DISTANCE_BUDGETS).
        adaptive_pruning: Legacy single-stage threshold (True=0.15, float=custom).
            Used as fallback when adaptive_pruning_schedule is None.
        adaptive_pruning_schedule: Multi-stage pruning schedule dict.
            Schema: {'checkpoints': [int, ...],
                     'tail_drop_per_checkpoint': float,
                     'final_strategy_threshold': float,
                     'min_configs_per_model': int}
        model_pruning_aggressiveness: Per-model schedule overrides.
            Schema: {model_name: {schedule fields}}.
        min_samples_for_overrides: If the loaded dataset has n_samples
            below this threshold, model_pruning_aggressiveness is dropped
            (set to None) for this dataset. Mid-flight pruning is unsafe
            on small datasets where fold-0 noise dominates.

        See PRUNING_MODES + _resolve_pruning_mode for named-mode entry point.
    """
    ds_config = DATASET_REPS.get(dataset_name)
    if ds_config is None:
        print(f"  {dataset_name}: not in DATASET_REPS, skipping")
        return

    domain = _get_domain(dataset_name)
    print(f"\n{'='*70}")
    print(f"  {dataset_name} (domain={domain})")
    print(f"{'='*70}")
    _reset_cache_stats()

    # Only load the reps we actually need (general + top_tier + augmentation)
    general_names = ds_config.get('general', [])
    top_names = ds_config.get('top_tier', [])
    aug_names = ds_config.get('augmentation', [])
    needed_reps = sorted(set(general_names + top_names + aug_names))

    try:
        data = load_data(dataset_name, rep_names=needed_reps)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return

    all_reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data.get('task_type')
    if task_type not in ('classification', 'regression'):
        raise ValueError(
            f"precompute_hybrids: loader for '{dataset_name}' returned "
            f"task_type={task_type!r}. Must be 'classification' or "
            f"'regression' — silent defaults would cause wrong metrics."
        )
    metric = _get_metric(domain, task_type, labels, dataset_name=dataset_name)

    general_reps = {k: all_reps[k] for k in general_names if k in all_reps}
    top_reps = {k: all_reps[k] for k in top_names if k in all_reps}
    aug_reps = {k: all_reps[k] for k in aug_names if k in all_reps}

    print(f"  general={len(general_reps)}, top={len(top_reps)}, aug={len(aug_reps)}")
    print(f"  metric={metric}, task={task_type}, n={len(labels)}")

    # Apply n_samples gate to per-model overrides. Mid-flight pruning on
    # small datasets risks dropping the per-model best (fold-0 noise
    # dominates). f16 calibration found this fails on n=132/720 antibody.
    if (model_pruning_aggressiveness is not None
            and len(labels) < min_samples_for_overrides):
        print(f"  n_samples={len(labels)} < {min_samples_for_overrides} threshold "
              f"— disabling per-model pruning overrides for this dataset.")
        model_pruning_aggressiveness = None
    elif model_pruning_aggressiveness is not None:
        print(f"  per-model pruning overrides active for: "
              f"{sorted(model_pruning_aggressiveness.keys())}")

    # NOTE: pass models=None so find_best_hybrid uses speed-tier defaults.
    # This matches how get_precomputed_search() calls find_best_hybrid,
    # ensuring identical cache keys. Do NOT pass DOWNSTREAM_MODELS explicitly.
    budgets = FULL_BUDGETS if speed == 'full' else SCREENING_BUDGETS

    # Build conditions
    conditions = []

    if distance:
        # Distance conditions for similarity/noise experiments
        # Uses DISTANCE_EVALUATORS + DISTANCE_BUDGETS (separate cache keys)
        if 'general' in tiers and len(general_reps) >= 2:
            conditions.append(('general_distance', general_reps, None, 'none'))
            if aug_reps:
                conditions.append(('general_distance', general_reps, aug_reps, 'uncapped'))
        if 'top' in tiers and len(top_reps) >= 2:
            conditions.append(('top_distance', top_reps, None, 'none'))
            if aug_reps:
                conditions.append(('top_distance', top_reps, aug_reps, 'uncapped'))
    else:
        # Standard conditions. General tier emits {none, uncapped} —
        # capped dropped 2026-05-22 (cost vs. consumer benefit; consumers
        # redirected to top/capped). Top tier still emits capped below.
        if 'general' in tiers and len(general_reps) >= 2:
            conditions.append(('general', general_reps, None, 'none'))
            if aug_reps:
                conditions.append(('general', general_reps, aug_reps, 'uncapped'))
        elif 'general' in tiers and len(general_reps) == 1 and aug_reps:
            # Single general rep — use augmentation mode
            conditions.append(('general', general_reps, aug_reps, 'uncapped'))

        # Top/capped is only emitted when n_samples <= TOP_CAPPED_MAX_N
        # (imported from experiments._shared as single source of truth).
        # Above that threshold the capped strategy projects past the 7-day
        # partition limit (empirically: top/capped has only ever completed
        # for n<=242; n=429 ddg_combined has never finished within a 7d
        # window). Experiments fall back to top/uncapped on larger datasets.
        n_samples = len(labels)

        if 'top' in tiers and len(top_reps) >= 2:
            conditions.append(('top', top_reps, None, 'none'))
            if aug_reps:
                conditions.append(('top', top_reps, aug_reps, 'uncapped'))
                if n_samples > TOP_CAPPED_MAX_N:
                    print(f"  SKIP top/capped for {dataset_name} "
                          f"(n={n_samples} > {TOP_CAPPED_MAX_N}; "
                          f"projects past 7-day partition)")
                else:
                    conditions.append(('top', top_reps, aug_reps, 'capped'))
            # Also test with general-minus-top as augmentation (borderline reps)
            borderline = {k: v for k, v in general_reps.items() if k not in top_reps}
            if borderline:
                merged_aug = {**borderline, **aug_reps}
                conditions.append(('top+borderline', top_reps, merged_aug, 'uncapped'))
        elif 'top' in tiers and len(top_reps) == 1:
            # Single top rep — augment with everything else
            pool = {k: v for k, v in general_reps.items() if k not in top_reps}
            pool.update(aug_reps)
            if pool:
                conditions.append(('top', top_reps, pool, 'uncapped'))

    # gb_demo_* (n=100k): even trimmed to tree models, the 'uncapped' A/B refit
    # loop over 100k rows projects past the 7-day wall. Skip uncapped ENTIRELY
    # for these two; they still get the cheap 'none' (no-augmentation) caches.
    # (§D Hole #2, user 2026-06-10.) c[3] is the strategy; this catches the
    # general / top / top+borderline AND distance uncapped variants in one place.
    UNCAPPED_SKIP_DATASETS = {
        'gb_demo_coding_vs_intergenomic_seqs',
        'gb_demo_human_or_worm',
        # DTA n>=50k: even trimmed to gradient boosters, the uncapped A/B refit
        # loop projects far past the 7-day wall (general/none alone was ~4 days
        # on bindingdb_kd). Skip uncapped entirely; keep the cheap 'none' caches.
        # (2026-06-21 cost-cut plan, Lever 4.) davis (n~29k) is NOT skipped.
        'bindingdb_kd',
        'kiba',
    }
    if dataset_name in UNCAPPED_SKIP_DATASETS:
        _before = len(conditions)
        conditions = [c for c in conditions if c[3] != 'uncapped']
        if len(conditions) != _before:
            print(f"  SKIP all 'uncapped' conditions for {dataset_name} "
                  f"({_before - len(conditions)} dropped; n=100k projects past "
                  f"7-day wall) — keeping {len(conditions)} non-uncapped")

    # Filter to single condition if requested (for per-job SLURM)
    if condition_filter:
        _cond_map = {
            'general_none': ('general', 'none'),
            'general_uncapped': ('general', 'uncapped'),
            'general_capped': ('general', 'capped'),
            'top_none': ('top', 'none'),
            'top_uncapped': ('top', 'uncapped'),
            'top_capped': ('top', 'capped'),
            'top_borderline': ('top+borderline', 'uncapped'),
            'general_distance_none': ('general_distance', 'none'),
            'general_distance_uncapped': ('general_distance', 'uncapped'),
            'top_distance_none': ('top_distance', 'none'),
            'top_distance_uncapped': ('top_distance', 'uncapped'),
        }
        if condition_filter in _cond_map:
            want_tier, want_strat = _cond_map[condition_filter]
            conditions = [(t, r, a, s) for t, r, a, s in conditions
                          if t == want_tier and s == want_strat]

    # Optional iteration-order change. ORDER-ONLY (the SET of conditions
    # computed is identical — only the sequence changes; the cached output is
    # byte-identical either way):
    #
    #   KIRBY_PRECOMPUTE_EXPENSIVE_FIRST=1  — cost-sort expensive-first so a
    #       wall-clock timeout sacrifices the CHEAP tail (general/none) rather
    #       than the costly variants (top/capped, top+borderline). Addresses
    #       the "expensive-last → capped never reached at the wall" failure.
    #
    # (The legacy KIRBY_PRECOMPUTE_REVERSE_ORDER two-wave flag was REMOVED
    # 2026-06-10: runs are single-writer now — no concurrent forward/reverse
    # waves — so the flag's only purpose (a second batch racing the first on
    # the same cache files) is gone, and it was the O3 clobber source.)
    if os.environ.get('KIRBY_PRECOMPUTE_EXPENSIVE_FIRST') == '1':
        _before = list(conditions)
        conditions = _order_conditions_expensive_first(conditions)
        # Safety assertion: ordering must never drop or add a condition.
        # Condition tuples carry numpy-array-valued dicts (not orderable /
        # hashable), so compare the multiset by object identity instead.
        assert (sorted(id(c) for c in _before)
                == sorted(id(c) for c in conditions)), (
            "condition reordering changed the SET of conditions — bug")
        print("  KIRBY_PRECOMPUTE_EXPENSIVE_FIRST=1 → cost-sorted conditions expensive-first")

    print(f"  {len(conditions)} conditions to precompute:")
    for tier, reps, augs, strat in conditions:
        aug_str = f"{len(augs)} aug reps" if augs else "no aug"
        print(f"    {tier:20s} {len(reps):2d} reps, {aug_str:20s}, strategy={strat}")

    # Run each condition
    n_ok = 0
    t_total = time.time()

    for tier, reps, augs, strat in conditions:
        label = f"{tier}/{strat}"
        print(f"\n  --- {label} ({len(reps)} reps) ---")

        t0 = time.time()
        try:
            # Distance conditions use DISTANCE_EVALUATORS + DISTANCE_BUDGETS
            is_distance = tier.endswith('_distance')
            _budgets = DISTANCE_BUDGETS if is_distance else budgets
            _extra = {}
            if is_distance:
                _extra['internal_evaluators'] = DISTANCE_EVALUATORS

            result = find_best_hybrid(
                base_reps=reps,
                labels=labels,
                groups=groups,
                budgets=_budgets,
                models=models,
                augmentations=augs,
                augmentation_strategy=strat,
                speed=speed,
                metric=metric,
                task_type=task_type,
                cache=True,
                verbose=True,
                dataset_name=dataset_name,
                tier=tier,
                adaptive_pruning=adaptive_pruning,
                adaptive_pruning_schedule=adaptive_pruning_schedule,
                model_pruning_aggressiveness=model_pruning_aggressiveness,
                **_extra,
            )
            elapsed = time.time() - t0
            n_models = len(result.best_by_model)
            n_results = len(result.all_results)
            print(f"  {label}: {n_results} configs, {n_models} models, "
                  f"{elapsed:.0f}s")

            # Update the hybrid registry
            try:
                _update_registry(dataset_name, tier, strat, result)
            except Exception as e:
                print(f"  WARNING: registry update failed: {e}")

            n_ok += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {label}: FAILED after {elapsed:.0f}s — {e}")

    total = time.time() - t_total
    print(f"\n  {dataset_name}: {n_ok}/{len(conditions)} conditions "
          f"precomputed in {total:.0f}s")
    print(f"  {_cache_stats_summary()}")


def main():
    parser = argparse.ArgumentParser(
        description='Precompute hybrid searches for experiment datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('datasets', nargs='*', help='Dataset names')
    parser.add_argument('--selftest-ordering', action='store_true',
                        help='Run the condition-ordering permutation self-test '
                             'and exit (no compute).')
    parser.add_argument('--domain', type=str, default=None,
                        choices=['molecular', 'antibody', 'tcga', 'dta', 'reaction'],
                        help='Precompute all datasets in a domain')
    parser.add_argument('--all', action='store_true',
                        help='Precompute ALL datasets')
    parser.add_argument('--tier', type=str, default='both',
                        choices=['general', 'top', 'both'],
                        help='Which rep tiers to precompute')
    parser.add_argument('--condition', type=str, default=None,
                        choices=['general_none', 'general_uncapped', 'general_capped',
                                 'top_none', 'top_uncapped', 'top_capped',
                                 'top_borderline',
                                 'general_distance_none', 'general_distance_uncapped',
                                 'top_distance_none', 'top_distance_uncapped'],
                        help='Run a single specific condition (for per-job SLURM)')
    parser.add_argument('--speed', type=str, default='full',
                        choices=['full'],
                        help="Search speed tier. Only 'full' is supported for "
                             "production precompute. To run faster searches "
                             "use --pruning-preset (Phase 2 multi-stage "
                             "pruning). 'fast' tier was retired here to prevent "
                             "the silent-cache-bypass regression from "
                             "EXPERIMENT_TRACKER.md.")
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: speed-tier models)')
    parser.add_argument('--distance', action='store_true',
                        help='Run distance-evaluator conditions (DISTANCE_EVALUATORS + '
                             'DISTANCE_BUDGETS) for similarity/noise experiments')
    parser.add_argument('--pruning', type=float, default=None,
                        help='Legacy single-stage adaptive pruning threshold '
                             'override (0.15=default). Mostly superseded by '
                             '--pruning-mode. True if omitted.')
    parser.add_argument('--pruning-mode', type=str, default='default_pruning',
                        choices=list(PRUNING_MODES.keys()),
                        help="Adaptive-pruning mode. 'default_pruning' = "
                             'existing single-stage 0.15 strategy prune only. '
                             "'aggressive_pruning' = multi-stage tail-drop "
                             '(td=0.5 at b=50) on expensive models (gp/svm/mlp), '
                             'gated by n_samples >= 1000. Cheap models always '
                             'use the existing default. See PRUNING_MODES for '
                             'descriptions.')
    parser.add_argument('--pruning-checkpoints', type=str, default=None,
                        help='Comma-separated budgets at which to fire '
                             'mid-flight tail-drop (e.g. "50" or "25,50,100"). '
                             'Overrides the default checkpoints in the chosen '
                             'pruning-mode. Applies to expensive-model overrides.')
    parser.add_argument('--pruning-tail-drop', type=float, default=None,
                        help='Bottom-fraction of (alloc, importance) pairs '
                             'to drop at each checkpoint (e.g. 0.5). Overrides '
                             "the default in the chosen pruning-mode's "
                             'expensive-model overrides.')
    parser.add_argument('--pruning-min-samples-gate', type=int, default=None,
                        help='Override the dataset n_samples threshold below '
                             'which per-model pruning overrides are dropped. '
                             "default_pruning uses 0 (always off); "
                             'aggressive_pruning uses 1000.')

    args = parser.parse_args()

    if args.selftest_ordering:
        test_condition_ordering_preserves_set()
        sys.exit(0)

    # Resolve dataset list
    if args.all:
        datasets = sorted(DATASET_REPS.keys())
    elif args.domain:
        datasets = _datasets_for_domain(args.domain)
    elif args.datasets:
        datasets = args.datasets
    else:
        parser.print_help()
        sys.exit(1)

    tiers = ('general', 'top') if args.tier == 'both' else (args.tier,)
    models = args.models.split(',') if args.models else None
    pruning = args.pruning if args.pruning is not None else True
    # Resolve pruning mode + apply individual-knob overrides
    override_checkpoints = None
    if args.pruning_checkpoints:
        override_checkpoints = [int(b) for b in args.pruning_checkpoints.split(',')
                                if b.strip()]
    pruning_schedule, model_overrides, min_samples_gate = _resolve_pruning_mode(
        args.pruning_mode,
        override_checkpoints=override_checkpoints,
        override_tail_drop=args.pruning_tail_drop,
        override_min_samples=args.pruning_min_samples_gate,
    )

    print(f"Precompute: {len(datasets)} datasets, tiers={tiers}, speed={args.speed}")
    if args.condition:
        print(f"Condition: {args.condition}")
    if args.distance:
        print(f"Mode: DISTANCE (evaluators={DISTANCE_EVALUATORS}, budgets={DISTANCE_BUDGETS})")
    print(f"Pruning mode: {args.pruning_mode}")
    print(f"  schedule = {pruning_schedule}")
    print(f"  model_overrides = {model_overrides}")
    print(f"  min_samples gate = {min_samples_gate}")
    if args.pruning is not None:
        print(f"  legacy --pruning override = {args.pruning}")
    print(f"Datasets: {datasets}")

    t_start = time.time()
    for ds in datasets:
        precompute_dataset(ds, speed=args.speed, tiers=tiers, models=models,
                           condition_filter=args.condition, distance=args.distance,
                           adaptive_pruning=pruning,
                           adaptive_pruning_schedule=pruning_schedule,
                           model_pruning_aggressiveness=model_overrides,
                           min_samples_for_overrides=min_samples_gate)

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"All done: {len(datasets)} datasets in {total:.0f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
