#!/usr/bin/env python3
"""
Find Best Hybrid Nucleotide Representation

Thin genomics wrapper around kirby.find_best_hybrid. Handles DNA/RNA sequence
loading, nucleotide representation generation, grouping, and output formatting.

Datasets:
  NT Revised (18 classification tasks):
    nt_promoter_all, nt_promoter_tata, nt_promoter_no_tata,
    nt_enhancers, nt_enhancers_types,
    nt_splice_sites_all, nt_splice_sites_acceptors, nt_splice_sites_donors,
    nt_H2AFZ, nt_H3K27ac, nt_H3K27me3, nt_H3K36me3, nt_H3K4me1,
    nt_H3K4me2, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3, nt_H4K20me1

  Genomic Benchmarks (4 classification tasks):
    gb_human_nontata_promoters, gb_human_enhancers_cohn,
    gb_demo_coding_vs_intergenomic_seqs, gb_demo_human_or_worm

Usage:
    # Quick local test with classical reps
    python find_best_hybrid_nucleotide.py nt_H3K4me3 --reps fast --search fast

    # All classical reps
    python find_best_hybrid_nucleotide.py nt_promoter_tata --reps classical --search medium

    # All reps (requires GPU + model deps)
    python find_best_hybrid_nucleotide.py nt_enhancers --reps all --search long
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import hashlib
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

sys.modules.setdefault('bottleneck', None)

import numpy as np

# Add src and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from kirby.find_best_hybrid import find_best_hybrid, SearchResult

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

_SEARCH_NAME_MAP = {
    'quick': 'fast', 'balanced': 'fast', 'medium': 'medium',
    'precise': 'medium', 'full': 'long', 'exhaustive': 'long',
    'fast': 'fast', 'long': 'long',
}

REP_PRESETS = {
    'fast': ['kmer4', 'kmer6', 'gc_content', 'dinuc_freq',
             'tetranuc_freq', 'complexity'],
    'classical': ['kmer3', 'kmer4', 'kmer5', 'kmer6', 'gc_content',
                  'dinuc_freq', 'tetranuc_freq', 'tnf_canonical',
                  'dna_shape', 'complexity', 'codon_usage'],
    'lm_only': ['dnabert2', 'nt_v2_250m', 'nt_v2_500m', 'hyenadna',
                'caduceus', 'grover_dna', 'ntv3_100m',
                'gena_lm', 'splicebert', 'mistral_dna', 'dnabert_s',
                'gpn', 'rnafm', 'rinalmo'],
    'all': [
        # Foundation models (14)
        'dnabert2', 'nt_v2_250m', 'nt_v2_500m', 'hyenadna',
        'caduceus', 'grover_dna', 'ntv3_100m',
        'gena_lm', 'splicebert', 'mistral_dna', 'dnabert_s',
        'gpn', 'rnafm', 'rinalmo',
        # Classical features
        'kmer3', 'kmer4', 'kmer5', 'kmer6', 'gc_content',
        'dinuc_freq', 'tetranuc_freq', 'tnf_canonical',
        'dna_shape', 'complexity', 'codon_usage',
        # Factory reps
        'onehot_pca50',
    ],
}

FACTORY_REPS = {'onehot_pca50'}


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset(name: str) -> dict:
    """Load nucleotide dataset by name.

    Prefixes:
        nt_  → NT Revised task (strip prefix)
        gb_  → Genomic Benchmark task (strip prefix)

    Returns:
        dict with keys: sequences, labels, groups, n_groups, task_type,
        n_classes, property_name
    """
    from kirby.datasets.nucleotide import (
        load_nt_revised_task, load_genomic_benchmark,
        NT_TASKS, GB_TASKS,
    )

    if name.startswith('nt_'):
        task_name = name[3:]
        return load_nt_revised_task(task_name)
    elif name.startswith('gb_'):
        task_name = name[3:]
        return load_genomic_benchmark(task_name)
    else:
        # Try as-is for NT tasks (e.g., 'H3K4me3')
        if name in NT_TASKS:
            return load_nt_revised_task(name)
        if name in GB_TASKS:
            return load_genomic_benchmark(name)
        raise ValueError(
            f"Unknown nucleotide dataset '{name}'.\n"
            f"NT Revised (prefix nt_): {['nt_' + t for t in NT_TASKS]}\n"
            f"Genomic Benchmarks (prefix gb_): {['gb_' + t for t in GB_TASKS]}"
        )


# =============================================================================
# REPRESENTATION GENERATION
# =============================================================================

def _rep_cache_path(sequences: List[str], rep_name: str) -> Path:
    """Get cache file path keyed by sequence list hash."""
    # Hash uses count + first/middle/last sequences for collision resistance
    n = len(sequences)
    key_parts = [str(n)]
    for idx in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        if 0 <= idx < n:
            key_parts.append(sequences[idx][:200])
    h = hashlib.md5('|'.join(key_parts).encode()).hexdigest()[:12]
    cache_dir = Path.home() / '.cache' / 'kirby' / 'reps' / f'nuc_{h}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{rep_name}.npy'


def get_representations(
    sequences: List[str],
    preset_or_list: str,
    cache: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate nucleotide representations.

    Args:
        sequences: DNA/RNA sequences
        preset_or_list: Preset name or comma-separated rep names
        cache: Cache representations to disk (default True)
    """
    from kirby.representations.nucleotide import NUCLEOTIDE_REPS

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    # Filter out factory reps
    base_rep_names = [r for r in rep_names if r not in FACTORY_REPS]

    print(f"\nGenerating {len(base_rep_names)} base representations "
          f"(+ {len(rep_names) - len(base_rep_names)} per-fold factory reps)...")

    representations = {}
    for rep_name in base_rep_names:
        if rep_name not in NUCLEOTIDE_REPS:
            print(f"  WARNING: Unknown representation '{rep_name}', skipping")
            continue

        # Try loading from cache
        if cache:
            cache_file = _rep_cache_path(sequences, rep_name)
            if cache_file.exists():
                rep = np.load(cache_file)
                print(f"  {rep_name}: loaded from cache ({rep.shape[1]} features)")
                representations[rep_name] = rep
                continue

        try:
            start = time.time()
            print(f"  Generating {rep_name}...", end=' ', flush=True)
            rep = NUCLEOTIDE_REPS[rep_name](sequences)
            elapsed = time.time() - start
            print(f"{rep.shape[1]} features ({elapsed:.1f}s)")

            # Validate
            if rep.shape[0] != len(sequences):
                print(f"  WARNING: {rep_name} shape mismatch "
                      f"({rep.shape[0]} vs {len(sequences)}), skipping")
                continue
            if np.isnan(rep).any():
                n_nan = np.isnan(rep).sum()
                print(f"  WARNING: {rep_name} has {n_nan} NaN values, replacing with 0")
                rep = np.nan_to_num(rep, nan=0.0)

            representations[rep_name] = rep.astype(np.float32)

            # Cache to disk
            if cache:
                cache_file = _rep_cache_path(sequences, rep_name)
                np.save(cache_file, rep)

        except ImportError as e:
            print(f"skipped (not installed: {e})")
        except Exception as e:
            print(f"FAILED: {e}")

    return representations


def get_rep_factories(
    sequences: List[str],
    preset_or_list: str,
) -> Dict[str, callable]:
    """Create per-fold rep factories for data-dependent nucleotide reps."""
    from kirby.representations.nucleotide import make_onehot_pca50_factory

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    factories = {}
    if 'onehot_pca50' in rep_names:
        try:
            print("  Creating onehot_pca50 factory...")
            factories['onehot_pca50'] = make_onehot_pca50_factory(sequences)
        except Exception as e:
            print(f"  WARNING: onehot_pca50 factory failed: {e}")

    return factories


# =============================================================================
# OUTPUT
# =============================================================================

def print_summary(result: SearchResult, task_type: str):
    """Print formatted result summary."""
    print("\n" + "=" * 70)
    print("NUCLEOTIDE HYBRID SEARCH RESULTS")
    print("=" * 70)

    if not result.best_by_model:
        print("No results found.")
        return

    print(f"\nBest by model:")
    for model_name, best in result.best_by_model.items():
        print(f"  [{model_name}]")
        print(f"    Dimension:  {best.get('budget', 'unknown')}")
        print(f"    Allocation: {best.get('allocation_method', 'unknown')}")
        print(f"    MCC:        {best.get('score_mean', float('nan')):.4f}")

    if hasattr(result, 'baselines') and result.baselines:
        print(f"\nBaseline comparison (individual reps):")
        # Baselines: {rep_name: {'dimension': int, 'rf': float, 'xgboost': float, ...}}
        # Find best model score per rep
        model_keys = set()
        for info in result.baselines.values():
            model_keys.update(k for k in info if k not in ('dimension', '_secondary'))
        sorted_baselines = sorted(
            result.baselines.items(),
            key=lambda x: max(
                (x[1].get(m, float('-inf')) for m in model_keys),
                default=float('-inf'),
            ),
            reverse=True,
        )
        for rep_name, info in sorted_baselines[:10]:
            dim = info.get('dimension', '?')
            scores = {m: info[m] for m in model_keys if m in info}
            best_score = max(scores.values()) if scores else float('nan')
            best_model = max(scores, key=scores.get) if scores else '?'
            print(f"  {rep_name:25s}  MCC={best_score:.4f} ({best_model})  dim={dim}")

    print()


def save_results(result: SearchResult, dataset_name: str,
                 search_mode: str, reps_tested: List[str],
                 output_path: Optional[str] = None):
    """Save results to JSON."""
    if output_path is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'results_nucleotide_{dataset_name}_{ts}.json'

    from experiments._shared import convert_numpy

    output = {
        'metadata': {
            'dataset': dataset_name,
            'domain': 'nucleotide',
            'search_mode': search_mode,
            'reps_tested': reps_tested,
            'timestamp': datetime.now().isoformat(),
        },
        'best_by_model': result.best_by_model if hasattr(result, 'best_by_model') else {},
        'baselines': result.baselines if hasattr(result, 'baselines') else {},
    }

    with open(output_path, 'w') as f:
        json.dump(convert_numpy(output), f, indent=2)

    print(f"Results saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_datasets = (
        ['nt_' + t for t in [
            'promoter_all', 'promoter_tata', 'promoter_no_tata',
            'enhancers', 'enhancers_types',
            'splice_sites_all', 'splice_sites_acceptors', 'splice_sites_donors',
            'H2AFZ', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
            'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K9me3', 'H4K20me1',
        ]] +
        ['gb_' + t for t in [
            'human_nontata_promoters', 'human_enhancers_cohn',
            'demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm',
        ]]
    )

    parser = argparse.ArgumentParser(
        description='Find optimal hybrid nucleotide representation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets:\n  " + "\n  ".join(all_datasets),
    )

    parser.add_argument('dataset', type=str,
                        help="Dataset name (e.g., nt_H3K4me3, gb_human_nontata_promoters)")
    parser.add_argument('--reps', type=str, default='fast',
                        help="'fast', 'classical', 'lm_only', 'all', or comma-separated")
    parser.add_argument('--search', type=str, default='fast',
                        choices=list(_SEARCH_NAME_MAP.keys()),
                        help="Search speed")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default=None,
                        help="Override models: comma-separated")
    parser.add_argument('--allocation', type=str, default=None,
                        help="Override allocation methods: comma-separated")
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Override number of CV folds')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable representation caching')

    args = parser.parse_args()
    speed = _SEARCH_NAME_MAP.get(args.search, args.search)

    # 1. Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    sequences = data['sequences']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    n_classes = data['n_classes']
    print(f"  {len(sequences)} sequences, {n_classes} classes, "
          f"task: {task_type}")

    # 2. Generate representations
    base_reps = get_representations(sequences, args.reps, cache=not args.no_cache)

    if not base_reps:
        print("ERROR: No representations were generated. Check dependencies.")
        sys.exit(1)

    # 3. Create per-fold factories
    rep_factories = get_rep_factories(sequences, args.reps)

    # 4. Resolve overrides
    models_arg = None
    if args.model:
        models_arg = [m.strip() for m in args.model.split(',')]

    alloc_arg = None
    if args.allocation:
        alloc_arg = [a.strip() for a in args.allocation.split(',')]

    # 5. Run hybrid search
    print(f"\nRunning hybrid search (speed={speed}, metric=mcc)...")
    result = find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        models=models_arg,
        metric='mcc',
        secondary_metrics=['accuracy', 'f1_weighted'],
        task_type=task_type,
        speed=speed,
        n_folds=args.n_folds,
        allocation_methods=alloc_arg,
        seed=args.seed,
        rep_factories=rep_factories if rep_factories else None,
    )

    # 6. Output
    print_summary(result, task_type)
    save_results(result, args.dataset, speed, list(base_reps.keys()), args.output)


if __name__ == '__main__':
    main()
