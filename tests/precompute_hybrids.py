#!/usr/bin/env python3
"""Precompute hybrid searches for all datasets × rep tiers × aug conditions.

For each dataset in DATASET_REPS, runs find_best_hybrid with cache=True for:
  1. general reps + no aug
  2. general reps + uncapped aug
  3. general reps + capped aug
  4. top_tier reps + no aug
  5. top_tier reps + uncapped aug
  6. top_tier reps + capped aug

Results are cached to ~/.cache/kirby/hybrid_search/ (or KIRBY_CACHE_DIR).
Experiments then load these via get_precomputed_search() — instant, no recompute.

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
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from experiments._shared import (
    load_data, SCREENING_BUDGETS, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
)
from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.find_best_hybrid import find_best_hybrid


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
    return 'molecular'


def _get_metric(domain, task_type, labels):
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


def precompute_dataset(dataset_name, speed='fast', tiers=('general', 'top'),
                       models=None, condition_filter=None):
    """Run all precompute conditions for one dataset."""
    ds_config = DATASET_REPS.get(dataset_name)
    if ds_config is None:
        print(f"  {dataset_name}: not in DATASET_REPS, skipping")
        return

    domain = _get_domain(dataset_name)
    print(f"\n{'='*70}")
    print(f"  {dataset_name} (domain={domain})")
    print(f"{'='*70}")

    # Load data
    try:
        data = load_data(dataset_name, rep_preset='all')
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return

    all_reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data.get('task_type', 'regression')
    metric = _get_metric(domain, task_type, labels)

    # Build rep sets
    general_names = ds_config.get('general', [])
    top_names = ds_config.get('top_tier', [])
    aug_names = ds_config.get('augmentation', [])

    general_reps = {k: all_reps[k] for k in general_names if k in all_reps}
    top_reps = {k: all_reps[k] for k in top_names if k in all_reps}
    aug_reps = {k: all_reps[k] for k in aug_names if k in all_reps}

    print(f"  general={len(general_reps)}, top={len(top_reps)}, aug={len(aug_reps)}")
    print(f"  metric={metric}, task={task_type}, n={len(labels)}")

    # NOTE: pass models=None so find_best_hybrid uses speed-tier defaults.
    # This matches how get_precomputed_search() calls find_best_hybrid,
    # ensuring identical cache keys. Do NOT pass DOWNSTREAM_MODELS explicitly.
    budgets = SCREENING_BUDGETS

    # Build conditions
    conditions = []
    if 'general' in tiers and len(general_reps) >= 2:
        conditions.append(('general', general_reps, None, 'none'))
        if aug_reps:
            conditions.append(('general', general_reps, aug_reps, 'uncapped'))
            conditions.append(('general', general_reps, aug_reps, 'capped'))
    elif 'general' in tiers and len(general_reps) == 1 and aug_reps:
        # Single general rep — use augmentation mode
        conditions.append(('general', general_reps, aug_reps, 'uncapped'))

    if 'top' in tiers and len(top_reps) >= 2:
        conditions.append(('top', top_reps, None, 'none'))
        if aug_reps:
            conditions.append(('top', top_reps, aug_reps, 'uncapped'))
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
        }
        want_tier, want_strat = _cond_map[condition_filter]
        conditions = [(t, r, a, s) for t, r, a, s in conditions
                      if t == want_tier and s == want_strat]

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
            result = find_best_hybrid(
                base_reps=reps,
                labels=labels,
                groups=groups,
                budgets=budgets,
                models=models,
                augmentations=augs,
                augmentation_strategy=strat,
                speed=speed,
                metric=metric,
                task_type=task_type,
                cache=True,
                verbose=True,
            )
            elapsed = time.time() - t0
            n_models = len(result.best_by_model)
            n_results = len(result.all_results)
            print(f"  {label}: {n_results} configs, {n_models} models, "
                  f"{elapsed:.0f}s")
            n_ok += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {label}: FAILED after {elapsed:.0f}s — {e}")

    total = time.time() - t_total
    print(f"\n  {dataset_name}: {n_ok}/{len(conditions)} conditions "
          f"precomputed in {total:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Precompute hybrid searches for experiment datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('datasets', nargs='*', help='Dataset names')
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
                                 'top_borderline'],
                        help='Run a single specific condition (for per-job SLURM)')
    parser.add_argument('--speed', type=str, default='fast',
                        choices=['fast', 'full'],
                        help='Search speed tier')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: DOWNSTREAM_MODELS)')

    args = parser.parse_args()

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

    print(f"Precompute: {len(datasets)} datasets, tiers={tiers}, speed={args.speed}")
    if args.condition:
        print(f"Condition: {args.condition}")
    print(f"Datasets: {datasets}")

    t_start = time.time()
    for ds in datasets:
        precompute_dataset(ds, speed=args.speed, tiers=tiers, models=models,
                           condition_filter=args.condition)

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"All done: {len(datasets)} datasets in {total:.0f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
