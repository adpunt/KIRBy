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
                       rep_counts=None, objectives=None,
                       tune=False, tune_n_iter=20, tune_knn_k=True):
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
        tune: tune each model's hyperparameters PER FOLD, on the feature matrix
            that fold is about to fit (_tune_estimator_on_fold). OFF by default.

            ⛔ IT IS AN IDENTITY COLUMN OF THE RESULT DATABASE, not a setting on
            a row. A tuned run and an untuned run share NO row: the canonical
            token is '' when off and 'tuned<n_iter>' with '_k' appended when the
            tuned neighbour count is kept (find_best_hybrid._tune_identity_token).
            So this is a SECOND precompute of the same size, not a variation on
            the first, and it has to be run for every condition the untuned pass
            was run for or the experiments asking for it find nothing.

            ⛔ AND IT IS NOT A SMALL MULTIPLIER. RandomizedSearchCV runs
            n_iter candidates over 3 inner folds, then fits the winner: 61 fits
            per configuration for rf, xgboost and lightgbm at n_iter=20, 25 for
            svm (its grid holds only 8 points, so n_iter is clamped), and 1 for
            dnn, which has NO entry in _HP_SEARCH_SPACES and is therefore
            silently untuned. Weighted over the 17 (model, evaluator) pairs a
            full search runs, about 42 times the fits of an untuned pass.
            Measured from the grids 2026-09-14; see --tune-n-iter.

        tune_n_iter: RandomizedSearchCV candidates per model. 20 by default,
            which is what n15 asks for, and n15 is the one experiment that
            already runs tuned. It is in the identity token, so lowering it
            here makes a set of rows nothing asks for.

        tune_knn_k: keep the neighbour count the tuning chose instead of forcing
            it back to 5. True by default, again to match n15 ('tuned20_k').
            Only reaches a search whose model list holds k-nearest-neighbours,
            which is the distance lineage and the knn pass.

        rep_counts: how many base representations a configuration may use, as a
            list where None means the whole pool. None here means take the
            speed's list (DEFAULT_REP_COUNTS = [2, 3, 5, None] at full,
            FAST_REP_COUNTS = [3, None] at fast). FORCED to [None] under a
            non-predictive objective, which computes no per-fold allocation
            scores to rank representations by -- see the note at the objective
            loop.
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
        return 0, 0

    domain = _get_domain(dataset_name)
    print(f"\n{'='*70}")
    print(f"  {dataset_name} (domain={domain})")
    print(f"{'='*70}")
    _reset_cache_stats()

    # Only load the reps we actually need (the 10-rep pool + augmentation)
    general_names = ds_config.get('general', [])
    aug_names = ds_config.get('augmentation', [])

    # ⛔ A NO-AUGMENTATION JOB MUST NOT ASK FOR THE AUGMENTATION
    # REPRESENTATIONS. load_data BUILDS a representation it cannot find rather
    # than failing (_shared._cache_rep), so naming one that is not on disk buys
    # however long it takes to compute, inside this job's wall, for a condition
    # that will never use it.
    #
    # Measured 2026-09-14: `precompute_hybrids.py caco2 --condition general_none`
    # on a cache holding 1 of caco2's 5 augmentation representations went
    # straight to "Computing 4 missing rep(s): ['coulomb_matrix',
    # 'pharmacophore3d', 'usr', 'usrcat']" and started generating 3D conformers
    # for 2,158 molecules. Three of those four need conformers, which is the
    # expensive kind.
    #
    # On ARC today this fires on qm9, whose 4 augmentation representations are
    # absent while all 10 pool representations are present.
    #
    # This is a SCOPE fix, not a cut: the augmented conditions still load and
    # still run. It only stops a job that was filtered down to 'none' from
    # paying for representations outside its own condition.
    _none_only = condition_filter in ('general_none', 'general_distance_none')
    if _none_only and aug_names:
        print(f"  --condition {condition_filter}: not loading the "
              f"{len(aug_names)} augmentation representations. They are not "
              f"used by this condition, and load_data BUILDS a representation "
              f"it cannot find.")
        aug_names = []

    needed_reps = sorted(set(general_names + aug_names))

    try:
        data = load_data(dataset_name, rep_names=needed_reps)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return 0, None

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

    # REFUSE a partly-loaded pool. The loader drops a representation it could
    # not build, prints a warning and returns the rest, so a search over 7 of
    # the 10 runs and reports success. Which representations the search ran over
    # is part of the result database's identity (result_cache.IDENTITY_COLS
    # holds 'rep_pool' as sorted name-and-width pairs), so those rows are never
    # served to an experiment that loads all 10 -- it finds nothing and rebuilds
    # the whole search inside its own wall. Added 2026-09-13.
    _missing = [k for k in general_names if k not in all_reps]
    if _missing:
        print(f"  REFUSED: {len(_missing)} of {len(general_names)} pool "
              f"representations did not load: {_missing}. A search over the "
              f"rest is a DIFFERENT search and no experiment asking for the "
              f"full pool will ever read it. Build them, then re-run.")
        return 0, None
    # A partly-loaded AUGMENTATION set drops the augmented conditions only --
    # 'none' is still a complete, readable search and is the one every other
    # experiment needs first.
    _missing_aug = [k for k in aug_names if k not in all_reps]
    if _missing_aug and aug_reps:
        print(f"  {len(_missing_aug)} of {len(aug_names)} augmentation "
              f"representations did not load: {_missing_aug}. Dropping the "
              f"'uncapped' and 'capped' conditions for this dataset -- a "
              f"partial set is a different search. 'none' still runs.")
        aug_reps = {}

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
            # The representation-count axis, per objective.
            #
            # ⛔ IT IS UNDEFINED UNDER A NON-PREDICTIVE OBJECTIVE. A
            # configuration restricted to the best N representations ranks them
            # by that fold's per-representation allocation scores. A geometry
            # objective never computes those -- _skip_alloc_precompute is true
            # for it (find_best_hybrid.py:5944) and alloc_cache stays empty --
            # so _evaluate_hybrid_config raises (find_best_hybrid.py:4412) for
            # every count below the pool size.
            #
            # The raise is caught and printed per configuration, so before this
            # the job ran the whole grid, failed on three of the four counts,
            # stored only the whole-pool rows and exited 0. On a 10-rep pool at
            # full speed that is 6,120 printed failures of 8,160 configurations
            # per condition. Measured 2026-09-14.
            #
            # This is not a cut for compute. The axis has no meaning without a
            # per-fold ranking, which is 'undefined' -- one of the three stated
            # reasons a cut is allowed.
            _obj_extra = dict(_extra)
            if objective != OBJECTIVE_PREDICTIVE:
                _obj_extra['rep_counts'] = [None]
                print(f"\n  objective={objective}: representation counts forced "
                      f"to the whole pool. The count axis needs per-fold "
                      f"allocation scores and this objective computes none.")
            elif rep_counts is not None:
                _obj_extra['rep_counts'] = rep_counts

            for _models in model_lists:
                label = f"{tier}/{strat}"
                if objective != OBJECTIVE_PREDICTIVE:
                    label += f" obj={objective}"
                if tune:
                    label += f" TUNED(n_iter={tune_n_iter}"
                    label += ", knn_k)" if tune_knn_k else ")"
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
                        tune=tune,
                        tune_n_iter=tune_n_iter,
                        tune_knn_k=tune_knn_k,
                        **_obj_extra,
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
    return n_ok, n_runs


# =============================================================================
# HAND A QUEUED PRECOMPUTE JOB OVER TO THE EXPERIMENTS
# =============================================================================
# A job already in the queue cannot be re-pointed: SLURM copied its command at
# submit time and `scontrol update` cannot change it. But that command is
#
#     python -u tests/precompute_hybrids.py <dataset> --condition <c> ...
#
# and THIS FILE is read from disk when the job starts. So a slot queued days ago
# -- carrying days of accrued priority -- can be spent on something else, by
# leaving a marker on disk before it starts.
#
# WHAT IT IS FOR. 2026-09-17: 55 precompute jobs had been queued three days
# behind 522 higher-priority jobs, with the precompute being done by hand in
# interactive sessions instead. Those 55 slots were about to run work that was
# already finished. This spends them on the experiments that come next.
#
# HOW A JOB CLAIMS WORK. One marker file per (dataset, category) under
# <cache>/handoff/. A job claims one by RENAMING it, which is atomic on a POSIX
# filesystem, so exactly one of the eleven jobs on a dataset can take a given
# category however many start at once. A job that claims nothing simply ends,
# which is what every job does today.
#
# ⛔ THE PRECOMPUTE STILL HAS TO BE DONE. The handoff runs the experiments with
# KIRBY_NO_COLD_SEARCH=1, so if a dataset's precompute is not on disk the
# experiment refuses in minutes and names the command that would fill it,
# instead of rebuilding the whole search inside the job's wall. The marker is
# put back when that happens, so the work is not lost from the queue.

def _handoff_dir():
    cache_root = Path(os.environ.get('KIRBY_CACHE_DIR',
                      str(Path.home() / '.cache' / 'kirby')))
    return cache_root / 'handoff'


def _claim_handoff(dataset_name):
    """Atomically take one (dataset, category) marker, or return None."""
    d = _handoff_dir()
    if not d.is_dir():
        return None, None
    jobid = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    for marker in sorted(d.glob(f'{dataset_name}__*')):
        if marker.suffix == '.claimed' or '.claimed.' in marker.name:
            continue
        category = marker.name.split('__', 1)[1]
        claimed = marker.with_name(f'{marker.name}.claimed.{jobid}')
        try:
            os.rename(str(marker), str(claimed))   # atomic; loser gets OSError
        except OSError:
            continue
        return category, claimed
    return None, None


def _run_experiments_for(dataset_name, category):
    """Run one experiment category for one dataset, in this job's remaining wall."""
    import subprocess
    env = dict(os.environ)
    # Fail fast rather than cold-building a search this job was not sized for.
    env['KIRBY_NO_COLD_SEARCH'] = env.get('KIRBY_NO_COLD_SEARCH', '1')
    cmd = [sys.executable, '-u', 'tests/experiments/run_dataset.py',
           category, dataset_name]
    print(f"\n{'='*70}")
    print(f"  HANDOFF: this queued slot is running the {category} experiments")
    print(f"  on {dataset_name} instead of a precompute that is already done.")
    print(f"  {' '.join(cmd)}")
    print(f"  KIRBY_NO_COLD_SEARCH={env['KIRBY_NO_COLD_SEARCH']} — it refuses in "
          f"minutes if the precompute is missing.")
    print(f"{'='*70}\n", flush=True)
    return subprocess.call(cmd, env=env,
                           cwd=str(Path(__file__).resolve().parent.parent))


def maybe_handoff(dataset_name):
    """Claim and run one experiment category, if a marker is waiting. 0 = nothing."""
    if os.environ.get('KIRBY_NO_HANDOFF'):
        return 0
    category, claimed = _claim_handoff(dataset_name)
    if category is None:
        return 0
    try:
        rc = _run_experiments_for(dataset_name, category)
    except BaseException:
        # Put it back so another slot can take it. A job killed at the wall
        # leaves the marker claimed, which is right -- its partial results are
        # on disk and a re-run resumes them.
        try:
            os.rename(str(claimed), str(claimed).split('.claimed.')[0])
        except OSError:
            pass
        raise
    if rc != 0:
        print(f"  HANDOFF: {category} on {dataset_name} exited {rc} — releasing "
              f"the marker so another slot can take it.")
        try:
            os.rename(str(claimed), str(claimed).split('.claimed.')[0])
        except OSError:
            pass
    return rc


def main():
    parser = argparse.ArgumentParser(
        description='Precompute hybrid searches for experiment datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('datasets', nargs='*', help='Dataset names')
    parser.add_argument('--handoff', type=str, default=None,
                        metavar='CATEGORY[,CATEGORY]',
                        help='Write handoff markers and exit, computing '
                             'nothing. Each named dataset gets one marker per '
                             'category (complexity, similarity, noise). A '
                             'queued precompute job on that dataset then runs '
                             'those experiments instead, in the slot it '
                             'already holds -- useful when the precompute has '
                             'been done by hand and the queued jobs would '
                             'otherwise repeat finished work. Markers are '
                             'claimed atomically, so eleven jobs on one '
                             'dataset cannot run the same category twice.')
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
    parser.add_argument('--tune', action='store_true',
                        help='Tune each model per fold, on the feature matrix '
                             'that fold is about to fit. ⛔ A SECOND PRECOMPUTE, '
                             'not a variation on the first: `tune` is an '
                             'IDENTITY column of the result database, so a '
                             'tuned run shares no row with an untuned one and '
                             'has to be run for every condition the untuned '
                             'pass was. ⛔ And it is about 42x the fits — see '
                             'the cost table this prints before it starts.')
    parser.add_argument('--tune-n-iter', type=int, default=20,
                        help='RandomizedSearchCV candidates per model '
                             '(default 20, which is what n15 asks for). It is '
                             'part of the identity token, so a different value '
                             'writes rows nothing asks for.')
    parser.add_argument('--no-tune-knn-k', action='store_true',
                        help="Force the neighbour count back to 5 instead of "
                             "keeping the one tuning chose. Default is to KEEP "
                             "it, matching n15's 'tuned20_k'. Only reaches a "
                             'search whose model list holds k-nearest-'
                             'neighbours: the distance lineage and the knn pass.')
    parser.add_argument('--rep-counts', type=str, default=None,
                        help='Comma-separated representation counts a '
                             "configuration may use; 'all' means the whole "
                             "pool (e.g. '2,3,5,all'). Overrides --speed's "
                             'list (full = 2,3,5,all; fast = 3,all). ALWAYS '
                             'forced to the whole pool alone under a '
                             'non-predictive objective, which computes no '
                             'per-fold allocation scores to rank '
                             'representations by.')
    parser.add_argument('--distance', action='store_true',
                        help='Run distance-evaluator conditions (DISTANCE_EVALUATORS + '
                             'DISTANCE_BUDGETS) for similarity/noise experiments')
    parser.add_argument('--pruning', type=float, default=None,
                        help='Legacy single-stage adaptive pruning threshold '
                             'override (0.15 = the old default). Superseded by '
                             '--pruning-mode, which wins when it names a '
                             'schedule. OFF if omitted.')
    parser.add_argument('--pruning-mode', type=str, default='no_pruning',
                        choices=list(PRUNING_MODES.keys()) + ['calibrated'],
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

    if args.handoff is not None:
        cats = [c.strip() for c in args.handoff.split(',') if c.strip()]
        bad = [c for c in cats if c not in ('complexity', 'similarity', 'noise')]
        if bad:
            parser.error(f"--handoff: {bad} — expected complexity, similarity "
                         f"or noise.")
        d = _handoff_dir()
        d.mkdir(parents=True, exist_ok=True)
        if not args.datasets:
            parser.error("--handoff needs the dataset names to mark.")
        n = 0
        for ds in args.datasets:
            for c in cats:
                (d / f'{ds}__{c}').touch()
                n += 1
        print(f"{n} marker(s) written under {d}")
        print("Each queued precompute job on one of these datasets will take ONE")
        print("of them when it starts, run those experiments for the rest of its")
        print("wall, and leave the others for the next slot. A job that finds no")
        print("marker behaves exactly as it does today.")
        print("\nRemove them with:  rm " + str(d) + "/*")
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

    # Representation counts. 'all' / 'none' / '' all mean the whole pool, which
    # find_best_hybrid spells None. Parsed here rather than in _csv because that
    # helper cannot produce a None entry.
    rep_counts = None
    if args.rep_counts:
        rep_counts = []
        for tok in args.rep_counts.split(','):
            tok = tok.strip().lower()
            if not tok:
                continue
            if tok in ('all', 'none', 'pool'):
                rep_counts.append(None)
                continue
            try:
                n = int(tok)
            except ValueError:
                parser.error(f"--rep-counts: {tok!r} is not a number or 'all'.")
            if n < 1:
                parser.error(f"--rep-counts: {n} is below 1.")
            rep_counts.append(n)
        if not rep_counts:
            parser.error("--rep-counts was given but named no counts.")
    pruning = args.pruning if args.pruning is not None else False
    # Resolve pruning mode + apply individual-knob overrides
    override_checkpoints = None
    if args.pruning_checkpoints:
        override_checkpoints = [int(b) for b in args.pruning_checkpoints.split(',')
                                if b.strip()]
    if args.pruning_mode == 'calibrated':
        # THE SETTLED SCHEDULE, and the only way to reach it from here.
        #
        # None of the three named modes in _shared.PRUNING_MODES gives it. They
        # predate the 2026-09-16/17 calibration (FRAMEWORK_FINALIZATION 7.3) and
        # were never updated, so:
        #   default_pruning     drops the checkpoint drop entirely -- which is
        #                       most of the saving, and the only rule that is a
        #                       floor under every dataset -- and sets the
        #                       allocation threshold to 0.15 instead of 0.20.
        #   aggressive_pruning  drops at budgets 25, 50 AND 100. Only 100 was
        #                       validated. The record is explicit that no axis
        #                       has a safe setting firing after fold 0 alone.
        #   no_pruning          off, which is what the precompute does today.
        #
        # _normalize_pruning_schedule starts from DEFAULT_PRUNING_SCHEDULE and
        # overlays whatever schedule it is handed, so handing it NOTHING and
        # asking for adaptive_pruning=True is what yields the calibrated values
        # unaltered.
        pruning_schedule, model_overrides, min_samples_gate = None, None, 0
        pruning = True
    else:
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
    if args.pruning_mode == 'calibrated':
        from kirby.find_best_hybrid import DEFAULT_PRUNING_SCHEDULE as _DPS
        print(f"  the settled schedule (FRAMEWORK_FINALIZATION 7.3, "
              f"2026-09-16/17): {_DPS}")
        print(f"  SAFE means every (model, budget) condition returned the "
              f"IDENTICAL best 5-fold score as the unpruned grid on all five "
              f"recorded datasets. Saving 13% to 44%, 30% across the five.")
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

    # Say what tuning costs BEFORE the first fit, with the numbers read out of
    # the engine's own grids rather than asserted here. A tuned pass that is
    # going to take forty times as long as the untuned one should say so while
    # there is still time to cancel it.
    tune_knn_k = not args.no_tune_knn_k
    if args.tune:
        from kirby.find_best_hybrid import (
            _HP_SEARCH_SPACES as _HP, _count_grid_size as _gs,
            _tune_identity_token as _tok)
        print(f"TUNING ON — identity token {_tok(True, args.tune_n_iter, tune_knn_k)!r}")
        print(f"  ⛔ These rows are a SEPARATE SET. `tune` is an identity column")
        print(f"     of the result database, so nothing untuned is reused and")
        print(f"     nothing here is read by an experiment that does not ask for")
        print(f"     tuning. Run it for every condition the untuned pass covered.")
        print(f"  fits per configuration, from the engine's grids:")
        _models_here = models or list(DOWNSTREAM_MODELS)
        for _m in _models_here:
            _g = _HP.get(_m)
            if _g is None:
                print(f"     {_m:10s}  1   NO GRID — silently UNTUNED")
            else:
                _n = min(args.tune_n_iter, _gs(_g))
                _note = (f"n_iter clamped to the grid's {_gs(_g)} points"
                         if _n < args.tune_n_iter else "")
                print(f"     {_m:10s} {_n * 3 + 1:3d}   {_n} candidates x 3 inner "
                      f"folds + 1 final fit  {_note}")
        print(f"  A full untuned pass is 8,160 configurations on a 10-"
              f"representation pool. Tuned, the fit count is about 42x that.")
    if rep_counts is not None:
        print(f"Representation counts: "
              f"{['all' if c is None else c for c in rep_counts]}")
        _asked_below_pool = [c for c in rep_counts if c is not None]
        _non_predictive = [o for o in objectives if o != OBJECTIVE_PREDICTIVE]
        if _asked_below_pool and _non_predictive:
            print(f"  NOTE: {_non_predictive} runs over the whole pool only. A "
                  f"count below the pool size ranks representations by that "
                  f"fold's allocation scores, and that objective computes "
                  f"none. The predictive runs still use {rep_counts}.")
    else:
        print(f"Representation counts: from --speed {args.speed}")
    print(f"Datasets: {datasets}")

    t_start = time.time()
    _done = _asked = 0
    _refused = []
    for ds in datasets:
        _counts = precompute_dataset(ds, tiers=tiers, models=models,
                           knn_models=args.with_knn,
                           ridge_models=args.with_ridge,
                           objectives=objectives,
                           condition_filter=args.condition, distance=args.distance,
                           speed=args.speed,
                           rep_counts=rep_counts,
                           tune=args.tune,
                           tune_n_iter=args.tune_n_iter,
                           tune_knn_k=tune_knn_k,
                           allocation_methods=_csv(args.allocation_methods),
                           importance_methods=_csv(args.importance_methods),
                           internal_evaluators=_csv(args.evaluators),
                           adaptive_pruning=pruning,
                           adaptive_pruning_schedule=pruning_schedule,
                           model_pruning_aggressiveness=model_overrides,
                           min_samples_for_overrides=min_samples_gate)
        # (searches written, searches attempted). attempted is None when the
        # dataset never got as far as a search -- it failed to load, or its
        # representation pool was incomplete.
        _ok, _n = _counts if _counts else (0, None)
        if _n is None:
            _refused.append(ds)
        else:
            _done += _ok
            _asked += _n

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"{len(datasets)} datasets in {total:.0f}s: "
          f"{_done}/{_asked} searches written")
    if _refused:
        print(f"REFUSED OR FAILED TO LOAD, no search attempted "
              f"({len(_refused)}): {_refused}")
    print(f"{'='*70}")
    # Exit non-zero when anything was asked for and not written. Before
    # 2026-09-13 every failure was printed and swallowed, so a job in which
    # every dataset failed to load exited 0 and read as a clean run.
    # A queued slot whose precompute is already done spends the rest of its wall
    # on the experiments, if a marker is waiting for this dataset. Runs AFTER
    # the precompute above, so a job that still had work to do does it first.
    _hand_rc = 0
    for ds in datasets:
        _hand_rc = maybe_handoff(ds) or _hand_rc

    if _refused or _done < _asked:
        sys.exit(1)
    if _hand_rc:
        sys.exit(_hand_rc)


if __name__ == '__main__':
    main()
