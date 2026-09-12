#!/usr/bin/env python3
"""Precompute hybrid searches for all datasets × aug conditions.

For each dataset in DATASET_REPS, runs find_best_hybrid with cache=True on the
one 10-rep pool ('general') for:
  1. general reps + no aug
  2. general reps + uncapped aug
  3. general reps + capped aug (every dataset; the 250-row gate was removed 2026-08-27)

(The 'top' tier is retired (2026-08-24): there is one 10-rep pool per dataset.
capped now rides on 'general' — it only lived on the old 'top' subset because
'general' used to be a big expensive pool, which is now those same 10 reps.)

Results are cached to ~/.cache/kirby/searches/{dataset}__{tier}__{aug}.pkl
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
    load_data, FULL_BUDGETS, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS,
    MAIN_MODELS_WITH_KNN, RIDGE_MODELS,
)
from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.find_best_hybrid import (
    find_best_hybrid, _cache_stats_summary, _reset_cache_stats,
    OBJECTIVE_PREDICTIVE, OBJECTIVE_SIMILARITY)


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


# Metric names only a geometry or blend objective produces. Mirrors
# _GEOMETRY_PRIMARY_KEY and _objective_metric_name in find_best_hybrid.
_GEOMETRY_METRIC_NAMES = ('readacross', 'struct_fidelity')


def _is_geometry_metric_name(metric_name):
    m = str(metric_name or '')
    return m in _GEOMETRY_METRIC_NAMES or m.startswith('blend_')


def _registry_tier(tier, objective):
    """The tier as find_best_hybrid files it, so the registry key matches disk.

    Kept deliberately identical to find_best_hybrid.py:6975-6976. A predictive
    run returns the tier unchanged, so every registry written before this
    existed still reads back under the same key.
    """
    from kirby.find_best_hybrid import (
        OBJECTIVE_PREDICTIVE as _OP, _objective_cache_token)
    if tier and objective and objective != _OP:
        return f'{tier}__obj-{_objective_cache_token(objective)}'
    return tier


def _update_registry(dataset_name, tier, aug_strategy, result, objective=None):
    """Add a SearchResult's best configs to the registry.

    Registry schema:
    {
      "dataset_name": {
        "metric": "rmse",                  # the PREDICTIVE metric only
        "lower_is_better": true,
        "configs": {
          "tier/aug_strategy": {
            "_metric": "rmse",             # this config's own metric
            "_lower_is_better": true,
            "_objective": "predictive",
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

    ★ THE OBJECTIVE IS PART OF THE KEY, and the dataset-level metric belongs to
    the PREDICTIVE objective alone.

    find_best_hybrid namespaces its own caches by appending `__obj-<token>` to
    the tier (find_best_hybrid.py:6975-6976), but that happens INSIDE the search
    and never comes back out, so the tier reaching this function is the plain
    one. Writing a geometry result under it did two things, both silent:
    the configs collided with the predictive ones, and the two unconditional
    assignments below stamped `metric='struct_fidelity', lower_is_better=False`
    onto the whole dataset — after which get_best_hybrid read every PREDICTIVE
    blueprint under a geometry metric and reported the worst as the best.

    So: the objective goes in the config key exactly as it goes in the cache
    tier, each config carries its own metric, and the dataset-level metric is
    only ever written by a predictive run.
    """
    objective = objective or OBJECTIVE_PREDICTIVE
    metric_name = result.metric_name
    is_predictive = (objective == OBJECTIVE_PREDICTIVE)

    # Belt and braces. If a caller ever forgets to pass the objective through,
    # the metric name gives it away: a geometry search stores 'readacross' or
    # 'struct_fidelity', a blend stores 'blend_...'. Raise rather than write the
    # corruption this docstring describes.
    if is_predictive and _is_geometry_metric_name(metric_name):
        raise ValueError(
            f"_update_registry: objective={objective!r} but the search stored "
            f"metric_name={metric_name!r}, which only a geometry or blend "
            f"objective produces. Pass the objective through, or this call "
            f"would overwrite {dataset_name}'s predictive metric.")

    registry = _load_registry()
    ds_entry = registry.setdefault(dataset_name, {
        'metric': metric_name if is_predictive else None,
        'lower_is_better': result.lower_is_better if is_predictive else None,
        'configs': {},
    })
    # Only a predictive run owns the dataset-level metric.
    if is_predictive:
        ds_entry['metric'] = metric_name
        ds_entry['lower_is_better'] = result.lower_is_better

    config_key = f"{_registry_tier(tier, objective)}/{aug_strategy}"
    config_entry = ds_entry['configs'].setdefault(config_key, {})
    # This config's own metric, whatever the dataset-level one says.
    config_entry['_objective'] = (dict(objective) if isinstance(objective, dict)
                                  else objective)
    config_entry['_metric'] = metric_name
    config_entry['_lower_is_better'] = bool(result.lower_is_better)

    for model_name, budget_dict in result.best_by_model_budget.items():
        if model_name.startswith('_'):
            raise ValueError(
                f"_update_registry: model name {model_name!r} starts with '_', "
                f"which collides with this config's own metadata keys.")
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
        ('general', r, aug_small, 'capped'),
        ('general_distance', r, aug_big, 'uncapped'),
    ]
    ordered = _order_conditions_expensive_first(conditions)

    # 1. SET preserved (same elements, same count) — compare by identity
    #    because the tuples contain unhashable numpy-array dicts.
    assert sorted(id(c) for c in conditions) == sorted(id(c) for c in ordered), \
        "ordering changed the set of conditions"
    assert len(ordered) == len(conditions)

    # 2. capped runs before its uncapped sibling, which runs before none
    labels = [(t, s) for t, _, _, s in ordered]
    assert labels.index(('general', 'capped')) < labels.index(('general', 'uncapped')), \
        "capped should sort before uncapped"
    assert labels.index(('general', 'uncapped')) < labels.index(('general', 'none')), \
        "uncapped should sort before none"

    # 3. capped (most expensive strategy) is first overall
    assert labels[0] == ('general', 'capped'), \
        f"expected general/capped first, got {labels[0]}"

    # 4. idempotent + deterministic
    assert _order_conditions_expensive_first(ordered) == ordered, \
        "ordering is not idempotent"
    print("test_condition_ordering_preserves_set: PASS")
    return True


def precompute_dataset(dataset_name, tiers=('general',),
                       models=None, knn_models=False, ridge_models=False,
                       condition_filter=None, distance=False,
                       adaptive_pruning=False, adaptive_pruning_schedule=None,
                       model_pruning_aggressiveness=None,
                       min_samples_for_overrides=0,
                       speed='full', allocation_methods=None,
                       importance_methods=None, internal_evaluators=None,
                       objectives=None):
    """Run all precompute conditions for one dataset.

    Args:
        knn_models: If True, score the main (non-distance) conditions over
            MAIN_MODELS_WITH_KNN instead of the default set. The model list is
            part of the search cache key, so this writes an ADDITIONAL search
            file per condition and replaces nothing. s02 asks for a recipe
            chosen for five-nearest-neighbour prediction and reads that same
            list; without this pass the file it asks for is on no disk.
            Ignored when models= is given explicitly, and never applied to the
            distance conditions, which name their own list.
        ridge_models: Same shape as knn_models, for RIDGE_MODELS (the default
            set plus 'ridge'). c16 asks for 'ridge' for the Ridge family, 'svm'
            for SVR and 'rf' for RandomForest; 'ridge' is not auto-selected, so
            models=None scores none and that lookup finds nothing. Also an
            extra search file per condition — the sorted model names are in the
            cache key — invalidating nothing already on disk. Setting both
            knn_models and ridge_models runs BOTH lists, one search each.
        objectives: which search objectives to run, as a tuple. None means
            (OBJECTIVE_PREDICTIVE,) — exactly what this script did before.
            Adding OBJECTIVE_SIMILARITY runs a SECOND search per condition
            whose features are chosen so that each molecule's activity is
            predicted from its five nearest neighbours in the distance the
            features define. find_best_hybrid appends the objective to the
            cache tier, so that search is a separate file and the predictive
            one is untouched. Without it, every pool lookup that asks for a
            similarity-built hybrid misses and rebuilds the search live inside
            the experiment.
        distance: If True, run distance-evaluator conditions (for similarity/noise
            experiments that use DISTANCE_EVALUATORS + DISTANCE_BUDGETS).
        speed: 'full' (5 allocation x 3 importance x 3 evaluator, the default
            model set) or 'fast' (3 x 2 x 2, rf + lightgbm only). Any of the
            three lists passed explicitly wins over it -- that is the custom
            case. The recipes a dataset is precomputed over come from HERE and
            nowhere else, so two runs at the same speed cover the same recipes.
        allocation_methods / importance_methods / internal_evaluators: custom
            lists; None means take the speed's list.
        adaptive_pruning: Legacy single-stage threshold (True=0.15, float=custom).
            Used as fallback when adaptive_pruning_schedule is None. Precompute
            defaults to OFF so a model precomputed later covers the same
            recipes as the ones before it.
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

    # Only load the reps we actually need (the 10-rep pool + augmentation)
    general_names = ds_config.get('general', [])
    aug_names = ds_config.get('augmentation', [])
    needed_reps = sorted(set(general_names + aug_names))

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
    aug_reps = {k: all_reps[k] for k in aug_names if k in all_reps}

    print(f"  general={len(general_reps)}, aug={len(aug_reps)}")
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

    # NOTE: pass models=None so find_best_hybrid uses the default model set.
    # This matches how get_precomputed_search() calls find_best_hybrid,
    # ensuring identical cache keys. Do NOT pass DOWNSTREAM_MODELS explicitly.
    budgets = FULL_BUDGETS

    # Build conditions
    conditions = []

    # The 'top' tier is retired (2026-08-24): one 10-rep pool ('general') per
    # dataset. capped now rides on 'general' (it only lived on 'top' because
    # 'general' used to be a big expensive pool — now they are the same reps).
    if distance:
        # Distance conditions for similarity/noise experiments
        # Uses DISTANCE_EVALUATORS + DISTANCE_BUDGETS (separate cache keys)
        if len(general_reps) >= 2:
            conditions.append(('general_distance', general_reps, None, 'none'))
            if aug_reps:
                conditions.append(('general_distance', general_reps, aug_reps, 'uncapped'))
    else:
        # Standard conditions on the 10-rep pool: {none, uncapped, capped}.
        # The 250-row gate on capped was REMOVED 2026-08-27. It existed because
        # the old capped refitted a model per candidate feature; the rebuilt one
        # costs about a tenth of that. capped now goes to every dataset.
        # _shared.get_precomputed_search asks for it unconditionally too -- the
        # two must stay in step or the lookup misses and it cold-builds.
        if len(general_reps) >= 2:
            conditions.append(('general', general_reps, None, 'none'))
            if aug_reps:
                conditions.append(('general', general_reps, aug_reps, 'uncapped'))
                conditions.append(('general', general_reps, aug_reps, 'capped'))
        elif len(general_reps) == 1 and aug_reps:
            # Single general rep — use augmentation mode
            conditions.append(('general', general_reps, aug_reps, 'uncapped'))

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
            'general_distance_none': ('general_distance', 'none'),
            'general_distance_uncapped': ('general_distance', 'uncapped'),
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

    # Which objectives to search. None = predictive only, which is what this
    # script ran before the parameter existed.
    objectives = tuple(objectives) if objectives else (OBJECTIVE_PREDICTIVE,)

    # Run each condition. One search per (condition, objective, model list);
    # each combination is a separate cache entry, so none of them replaces
    # another. With the defaults there is exactly one model list (None) and one
    # objective, which is the single search this loop always ran.
    n_ok = 0
    n_runs = 0
    t_total = time.time()

    for tier, reps, augs, strat in conditions:
        # Distance conditions use DISTANCE_EVALUATORS + DISTANCE_BUDGETS
        is_distance = tier.endswith('_distance')
        _budgets = DISTANCE_BUDGETS if is_distance else budgets
        # knn is tier 'extra' as of 2026-09-04 and ridge is tier 'off', so
        # models=None scores neither. The distance lineage names knn back in;
        # s02/s04-s08/s10/s13 all read a knn-scored config, and c16 reads a
        # ridge-scored one. Each named list is an extra search file: the sorted
        # model names go into the search cache key, so it is a separate search
        # from the models=None one and invalidates nothing already precomputed.
        if is_distance:
            model_lists = [DISTANCE_MODELS]
        elif models is not None:
            model_lists = [models]
        else:
            model_lists = []
            if knn_models:
                model_lists.append(MAIN_MODELS_WITH_KNN)
            if ridge_models:
                model_lists.append(RIDGE_MODELS)
            if not model_lists:
                model_lists = [None]

        # speed picks the default three lists; an explicit list wins.
        _extra = {'speed': speed}
        if allocation_methods is not None:
            _extra['allocation_methods'] = allocation_methods
        if importance_methods is not None:
            _extra['importance_methods'] = importance_methods
        if internal_evaluators is not None:
            _extra['internal_evaluators'] = internal_evaluators
        # Distance conditions fix their own evaluators, over both.
        if is_distance:
            _extra['internal_evaluators'] = DISTANCE_EVALUATORS

        for objective in objectives:
            for _models in model_lists:
                label = f"{tier}/{strat}"
                if objective != OBJECTIVE_PREDICTIVE:
                    label += f" obj={objective}"
                if _models is not None:
                    label += f" models={'+'.join(sorted(_models))}"
                print(f"\n  --- {label} ({len(reps)} reps) ---")
                n_runs += 1

                t0 = time.time()
                try:
                    result = find_best_hybrid(
                        base_reps=reps,
                        labels=labels,
                        groups=groups,
                        budgets=_budgets,
                        models=_models,
                        augmentations=augs,
                        augmentation_strategy=strat,
                        metric=metric,
                        task_type=task_type,
                        cache=True,
                        verbose=True,
                        dataset_name=dataset_name,
                        tier=tier,
                        # Under a geometry objective find_best_hybrid overrides
                        # `metric` with its own ('readacross') and fits no
                        # downstream model, so the metric passed above is the
                        # dataset's and is discarded there.
                        objective=objective,
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

                    # Update the hybrid registry. The objective is part of the
                    # registry key exactly as it is part of the cache tier, and
                    # only a predictive run owns the dataset-level metric — so
                    # it MUST be passed or a geometry result overwrites the
                    # predictive configs and stamps 'readacross' on the dataset.
                    try:
                        _update_registry(dataset_name, tier, strat, result,
                                         objective=objective)
                    except Exception as e:
                        print(f"  WARNING: registry update failed: {e}")

                    n_ok += 1
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  {label}: FAILED after {elapsed:.0f}s — {e}")

    total = time.time() - t_total
    print(f"\n  {dataset_name}: {n_ok}/{n_runs} searches over "
          f"{len(conditions)} conditions precomputed in {total:.0f}s")
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
    parser.add_argument('--tier', type=str, default='general',
                        choices=['general'],
                        help="Retired flag — one 10-rep pool ('general') per dataset")
    parser.add_argument('--condition', type=str, default=None,
                        choices=['general_none', 'general_uncapped', 'general_capped',
                                 'general_distance_none', 'general_distance_uncapped'],
                        help='Run a single specific condition (for per-job SLURM)')
    parser.add_argument('--with-knn', action='store_true',
                        help='Score the main conditions over the default model '
                             'set plus knn (MAIN_MODELS_WITH_KNN). Writes an '
                             'extra search file per condition; replaces '
                             'nothing. Required before s02 can read a recipe '
                             'chosen for five-nearest-neighbour prediction.')
    parser.add_argument('--with-ridge', action='store_true',
                        help='Score the main conditions over the default model '
                             "set plus ridge (RIDGE_MODELS). Writes an extra "
                             'search file per condition; replaces nothing. '
                             'Required before c16 can read a recipe chosen for '
                             'the Ridge family. Combines with --with-knn: each '
                             'list is searched once.')
    parser.add_argument('--with-similarity', action='store_true',
                        help='Also search each condition under '
                             "objective='similarity' — features chosen so each "
                             "molecule's activity is predicted from its five "
                             'nearest neighbours in the distance the features '
                             'define. An extra search file per condition (the '
                             'objective rides on the cache tier); the '
                             'predictive search is untouched. Required before a '
                             'pool lookup that admits similarity-built hybrids '
                             'can hit instead of rebuilding the search inside '
                             'the experiment.')
    parser.add_argument('--objectives', type=str, default=None,
                        help='Comma-separated search objectives to run '
                             "(e.g. 'predictive,similarity' or 'similarity'). "
                             'Overrides --with-similarity. Default: predictive '
                             'only.')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: the default model set)')
    parser.add_argument('--speed', type=str, default='full',
                        choices=['fast', 'full'],
                        help="Which recipe list to precompute over. 'full' = 5 "
                             'allocation x 3 importance x 5 evaluator, default '
                             "model set. 'fast' = 3 x 2 x 2, rf + lightgbm only. "
                             'Naming any of the three lists below overrides '
                             'that list (the custom case).')
    parser.add_argument('--allocation-methods', type=str, default=None,
                        help='Comma-separated allocation methods. Overrides '
                             "--speed's list.")
    parser.add_argument('--importance-methods', type=str, default=None,
                        help='Comma-separated importance methods. Overrides '
                             "--speed's list.")
    parser.add_argument('--evaluators', type=str, default=None,
                        help='Comma-separated internal evaluators. Overrides '
                             "--speed's list. Ignored for --distance "
                             'conditions, which fix their own.')
    parser.add_argument('--distance', action='store_true',
                        help='Run distance-evaluator conditions (DISTANCE_EVALUATORS + '
                             'DISTANCE_BUDGETS) for similarity/noise experiments')
    parser.add_argument('--pruning', type=float, default=None,
                        help='Legacy single-stage adaptive pruning threshold '
                             'override (0.15 = the old default). Superseded by '
                             '--pruning-mode, which wins when it names a '
                             'schedule. OFF if omitted.')
    parser.add_argument('--pruning-mode', type=str, default='no_pruning',
                        choices=list(PRUNING_MODES.keys()),
                        help="Adaptive-pruning mode. DEFAULT 'no_pruning' = "
                             'nothing is dropped, so a model precomputed later '
                             'covers the same recipes as the ones before it. '
                             "'default_pruning' = "
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

    def _csv(v):
        return [x.strip() for x in v.split(',') if x.strip()] if v else None

    tiers = ('general',)
    models = args.models.split(',') if args.models else None
    pruning = args.pruning if args.pruning is not None else False
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

    print(f"Precompute: {len(datasets)} datasets, tiers={tiers}")
    if args.condition:
        print(f"Condition: {args.condition}")
    if args.distance:
        print(f"Mode: DISTANCE (evaluators={DISTANCE_EVALUATORS}, budgets={DISTANCE_BUDGETS})")
    print(f"Speed: {args.speed}" + (f" (custom allocation={args.allocation_methods})" if args.allocation_methods else "")
          + (f" (custom importance={args.importance_methods})" if args.importance_methods else "")
          + (f" (custom evaluators={args.evaluators})" if args.evaluators else ""))
    print(f"Pruning mode: {args.pruning_mode}")
    print(f"  schedule = {pruning_schedule}")
    print(f"  model_overrides = {model_overrides}")
    print(f"  min_samples gate = {min_samples_gate}")
    if args.pruning is not None:
        print(f"  legacy --pruning override = {args.pruning}")
    if args.with_knn:
        print(f"Model set: {MAIN_MODELS_WITH_KNN} (--with-knn; extra search "
              f"files, nothing replaced)")
    if args.with_ridge:
        print(f"Model set: {RIDGE_MODELS} (--with-ridge; extra search "
              f"files, nothing replaced)")

    # Objectives. --objectives wins over --with-similarity; the default is the
    # predictive search alone, which is what this script always ran.
    if args.objectives:
        objectives = tuple(_csv(args.objectives) or ())
    elif args.with_similarity:
        objectives = (OBJECTIVE_PREDICTIVE, OBJECTIVE_SIMILARITY)
    else:
        objectives = (OBJECTIVE_PREDICTIVE,)
    _known_objectives = (OBJECTIVE_PREDICTIVE, OBJECTIVE_SIMILARITY)
    _bad = [o for o in objectives if o not in _known_objectives]
    if _bad:
        parser.error(
            f"--objectives {_bad} — this script precomputes "
            f"{list(_known_objectives)}. 'struct_fid' needs the MCS pair set "
            f"and the whole-dataset row numbers, which nothing here supplies.")
    print(f"Objectives: {list(objectives)}")
    print(f"Datasets: {datasets}")

    t_start = time.time()
    for ds in datasets:
        precompute_dataset(ds, tiers=tiers, models=models,
                           knn_models=args.with_knn,
                           ridge_models=args.with_ridge,
                           objectives=objectives,
                           condition_filter=args.condition, distance=args.distance,
                           speed=args.speed,
                           allocation_methods=_csv(args.allocation_methods),
                           importance_methods=_csv(args.importance_methods),
                           internal_evaluators=_csv(args.evaluators),
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
