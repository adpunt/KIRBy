#!/usr/bin/env python3
"""Precompute fine-tuned representations for all (dataset, FT rep, fold) combos.

For each dataset with 'ft_reps' in DATASET_REPS, this script:
  1. Loads the dataset (SMILES or sequences + labels + groups)
  2. Generates fold splits using the same splitter as find_best_hybrid
  3. Calls each FT factory per fold — factory caches embeddings to disk

After precomputation, experiments construct the same factories and hit cache
(no GPU needed). Cache location: ~/.cache/kirby/finetuned_reps/ or
$KIRBY_CACHE_DIR/finetuned_reps/.

Usage:
    python tests/precompute_ft_reps.py logd                      # one dataset
    python tests/precompute_ft_reps.py logd caco2 qm9            # multiple
    python tests/precompute_ft_reps.py --domain molecular         # all molecular FT
    python tests/precompute_ft_reps.py --domain antibody          # all antibody FT
    python tests/precompute_ft_reps.py --all                      # everything
    python tests/precompute_ft_reps.py logd --speed fast          # 3-fold only
    python tests/precompute_ft_reps.py logd --rep lora_molformer  # single rep
    python tests/precompute_ft_reps.py --all --dry-run            # show plan only
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

from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.find_best_hybrid import _make_cv_splitter, SPEED_TIERS


# =============================================================================
# FACTORY DISPATCH — maps factory type strings to constructor calls
# =============================================================================

def _build_factory(dataset_name, rep_name, spec, data):
    """Construct a FT factory callable from a DATASET_REPS ft_reps spec.

    Args:
        dataset_name: Dataset name (used for cache key).
        rep_name: FT rep name (e.g. 'lora_molformer').
        spec: Dict with 'factory' and 'base_model' keys.
        data: Dict with domain-specific data ('smiles' or 'heavy_seqs'/'light_seqs').

    Returns:
        Factory callable: (train_idx, val_idx, train_labels) -> (train_embs, val_embs)
    """
    factory_type = spec['factory']
    base_model = spec['base_model']

    if factory_type == 'molecular_lora':
        from kirby.finetuning import make_molecular_lora_factory
        return make_molecular_lora_factory(
            smiles_list=data['smiles'],
            dataset_name=dataset_name,
            base_model=base_model,
        )
    elif factory_type == 'molecular_contrastive':
        from kirby.finetuning import make_molecular_contrastive_factory
        return make_molecular_contrastive_factory(
            smiles_list=data['smiles'],
            dataset_name=dataset_name,
            base_model=base_model,
        )
    elif factory_type == 'molecular_gnn':
        from kirby.finetuning import make_molecular_gnn_factory
        return make_molecular_gnn_factory(
            smiles_list=data['smiles'],
            dataset_name=dataset_name,
            base_model=base_model,
        )
    elif factory_type == 'antibody_lora':
        from kirby.finetuning import make_antibody_lora_factory
        return make_antibody_lora_factory(
            heavy_seqs=data['heavy_seqs'],
            light_seqs=data['light_seqs'],
            dataset_name=dataset_name,
            base_model=base_model,
        )
    else:
        raise ValueError(f"Unknown factory type: {factory_type}")


# =============================================================================
# DOMAIN DETECTION & DATA LOADING
# =============================================================================

def _get_domain(dataset_name):
    """Detect domain from dataset name (matches precompute_hybrids.py)."""
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


def _load_dataset_minimal(dataset_name):
    """Load just the raw data needed for FT factories (no frozen reps).

    Returns dict with:
        'smiles' (molecular) or 'heavy_seqs'/'light_seqs' (antibody)
        'labels', 'groups'
    """
    domain = _get_domain(dataset_name)

    if domain == 'molecular':
        from find_best_hybrid_molecular import load_dataset, assign_scaffold_groups
        smiles, labels, task_type = load_dataset(dataset_name)
        groups, n_scaffolds = assign_scaffold_groups(smiles)
        print(f"  Loaded: {len(smiles)} molecules, {n_scaffolds} scaffolds")
        return {
            'smiles': smiles,
            'labels': labels,
            'groups': groups,
        }

    elif domain == 'antibody':
        from antibody_baseline_survey import load_dataset as load_ab_dataset
        data = load_ab_dataset(dataset_name)
        print(f"  Loaded: {len(data['labels'])} antibodies, "
              f"{len(np.unique(data['groups']))} groups")
        return {
            'heavy_seqs': data['heavy_seqs'],
            'light_seqs': data['light_seqs'],
            'labels': data['labels'],
            'groups': data['groups'],
        }

    else:
        raise ValueError(f"FT precompute not supported for domain: {domain}")


# =============================================================================
# CORE PRECOMPUTE LOGIC
# =============================================================================

def precompute_one(dataset_name, rep_name, spec, n_folds, seed=42):
    """Precompute FT embeddings for one (dataset, rep, n_folds) combo.

    Returns True on success, False on failure.
    """
    print(f"\n  --- {rep_name} ({n_folds}-fold) ---")

    # Load data
    data = _load_dataset_minimal(dataset_name)
    labels = data['labels']
    groups = data['groups']

    # Build factory
    factory = _build_factory(dataset_name, rep_name, spec, data)

    # Generate fold splits — MUST match find_best_hybrid's _precompute_fold_reps()
    splitter, use_groups = _make_cv_splitter(groups, n_folds, seed)
    dummy_X = np.zeros((len(labels), 1))
    split_args = (dummy_X, labels, groups) if use_groups else (dummy_X, labels)
    folds = list(splitter.split(*split_args))

    print(f"  {len(folds)} folds, seed={seed}")

    success = True
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_labels = labels[train_idx]
        t0 = time.time()
        print(f"    Fold {fold_idx+1}/{len(folds)} "
              f"(train={len(train_idx)}, val={len(val_idx)})...", end=' ', flush=True)
        try:
            train_embs, val_embs = factory(train_idx, val_idx, train_labels)
            elapsed = time.time() - t0
            print(f"done ({train_embs.shape[1]}D, {elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED after {elapsed:.1f}s: {e}")
            success = False

    return success


def precompute_dataset(dataset_name, speeds=('fast', 'full'), seed=42,
                       rep_filter=None):
    """Precompute all FT reps for one dataset at all requested speeds."""
    ds_config = DATASET_REPS.get(dataset_name)
    if ds_config is None:
        print(f"  {dataset_name}: not in DATASET_REPS, skipping")
        return

    ft_reps = ds_config.get('ft_reps', {})
    if not ft_reps:
        print(f"  {dataset_name}: no ft_reps configured, skipping")
        return

    if rep_filter:
        ft_reps = {k: v for k, v in ft_reps.items() if k == rep_filter}
        if not ft_reps:
            print(f"  {dataset_name}: rep '{rep_filter}' not in ft_reps, skipping")
            return

    domain = _get_domain(dataset_name)
    print(f"\n{'='*70}")
    print(f"  {dataset_name} (domain={domain}, {len(ft_reps)} FT reps)")
    print(f"{'='*70}")

    n_ok = 0
    n_total = 0
    t_start = time.time()

    for speed in speeds:
        n_folds = SPEED_TIERS[speed]['n_folds']
        for rep_name, spec in ft_reps.items():
            n_total += 1
            ok = precompute_one(dataset_name, rep_name, spec, n_folds, seed)
            if ok:
                n_ok += 1

    total = time.time() - t_start
    print(f"\n  {dataset_name}: {n_ok}/{n_total} combos in {total:.0f}s")


# =============================================================================
# CLI
# =============================================================================

def _datasets_with_ft(domain=None):
    """Return sorted list of dataset names that have ft_reps configured."""
    result = []
    for ds, config in DATASET_REPS.items():
        if config.get('ft_reps'):
            if domain is None or _get_domain(ds) == domain:
                result.append(ds)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(
        description='Precompute fine-tuned representations for experiment pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('datasets', nargs='*', help='Dataset names')
    parser.add_argument('--domain', type=str, default=None,
                        choices=['molecular', 'antibody'],
                        help='Precompute all FT reps in a domain')
    parser.add_argument('--all', action='store_true',
                        help='Precompute ALL FT reps')
    parser.add_argument('--speed', type=str, default='both',
                        choices=['fast', 'full', 'both'],
                        help='Which speed tiers to precompute folds for')
    parser.add_argument('--rep', type=str, default=None,
                        help='Filter to a single FT rep name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match experiments)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show plan without computing')

    args = parser.parse_args()

    # Resolve dataset list
    if args.all:
        datasets = _datasets_with_ft()
    elif args.domain:
        datasets = _datasets_with_ft(args.domain)
    elif args.datasets:
        datasets = args.datasets
    else:
        parser.print_help()
        sys.exit(1)

    speeds = ('fast', 'full') if args.speed == 'both' else (args.speed,)

    # Summary
    total_combos = 0
    print(f"FT Precompute: {len(datasets)} datasets, speeds={speeds}")
    for ds in datasets:
        ft = DATASET_REPS.get(ds, {}).get('ft_reps', {})
        if args.rep:
            ft = {k: v for k, v in ft.items() if k == args.rep}
        n_folds_total = sum(SPEED_TIERS[s]['n_folds'] for s in speeds)
        n_combos = len(ft) * len(speeds)
        total_combos += n_combos
        n_fold_calls = len(ft) * n_folds_total
        print(f"  {ds}: {len(ft)} reps × {len(speeds)} speeds = "
              f"{n_combos} combos ({n_fold_calls} fold calls)")

    print(f"\nTotal: {total_combos} combos")

    if args.dry_run:
        print("\n[dry-run] No computation performed.")
        return

    # Run
    t_start = time.time()
    for ds in datasets:
        precompute_dataset(ds, speeds=speeds, seed=args.seed,
                           rep_filter=args.rep)

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"All done: {len(datasets)} datasets in {total:.0f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
