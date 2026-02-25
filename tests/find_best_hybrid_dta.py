#!/usr/bin/env python3
"""
Find Best Hybrid Drug-Target Affinity (DTA) Representation

Thin DTA wrapper around kirby.find_best_hybrid. Handles drug-target pair loading,
drug + protein representation generation, cold-target grouping, and output formatting.
The actual search logic lives in the domain-agnostic library.

The framework takes Dict[str, ndarray] where each value is (n_samples, d).
For drug-target pairs, drug reps and protein reps are separate entries — the
framework combines them automatically. No core library changes needed.

Datasets (all regression):
  - davis: DAVIS kinase (30K pairs, 68 drugs × 442 kinases, Kd)
  - kiba:  KIBA integrated scores (~117K pairs)

Usage:
    # Quick local test
    python find_best_hybrid_dta.py davis --reps fast --search fast

    # Medium search with LM reps
    python find_best_hybrid_dta.py davis --reps fast+lm --search medium

    # KIBA dataset
    python find_best_hybrid_dta.py kiba --reps fast --search fast
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

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kirby.find_best_hybrid import find_best_hybrid, SearchResult, Blueprint

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

REP_PRESETS = {
    'fast': [
        'drug_ecfp4', 'drug_maccs', 'drug_pdv',
        'protein_esm2', 'protein_aac',
    ],  # 5 reps
    'fast+lm': [
        'drug_ecfp4', 'drug_maccs', 'drug_pdv',
        'drug_chemberta', 'drug_molformer',
        'protein_esm2', 'protein_aac', 'protein_descriptors',
    ],  # 8 reps
    'full': [
        'drug_ecfp4', 'drug_maccs', 'drug_pdv',
        'drug_chemberta', 'drug_molformer', 'drug_selformer',
        'protein_esm2', 'protein_aac', 'protein_descriptors', 'protein_kmer',
    ],  # 10 reps
}

_SEARCH_NAME_MAP = {
    'quick': 'fast',
    'balanced': 'fast',
    'medium': 'medium',
    'precise': 'medium',
    'full': 'long',
    'exhaustive': 'long',
    'fast': 'fast',
    'long': 'long',
}

# Dispatch: (function_name, module, rep_type)
# rep_type: 'drug' = fn(smiles), 'protein' = fn(sequences)
_REP_DISPATCH = {
    # Drug reps (from kirby.representations.molecular)
    'drug_ecfp4':      ('create_ecfp4',      'molecular', 'drug'),
    'drug_maccs':      ('create_maccs',       'molecular', 'drug'),
    'drug_pdv':        ('create_pdv',         'molecular', 'drug'),
    'drug_chemberta':  ('create_chemberta',   'molecular', 'drug'),
    'drug_molformer':  ('create_molformer',   'molecular', 'drug'),
    'drug_selformer':  ('create_selformer',   'molecular', 'drug'),
    'drug_mol2vec':    ('create_mol2vec',     'molecular', 'drug'),
    # Protein reps (from kirby.representations.protein)
    'protein_esm2':        ('create_esm2_protein_embeddings', 'protein', 'protein'),
    'protein_aac':         ('create_protein_aac',             'protein', 'protein'),
    'protein_descriptors': ('create_protein_descriptors',     'protein', 'protein'),
    'protein_kmer':        ('create_protein_kmer',            'protein', 'protein'),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(name: str):
    """Load DTA dataset. Thin wrapper around kirby.datasets.dta.load_dataset."""
    from kirby.datasets.dta import load_dataset as _load
    return _load(name)


# =============================================================================
# REPRESENTATION GENERATION
# =============================================================================

def _rep_cache_path(drug_smiles: List[str], protein_seqs: List[str],
                    rep_name: str) -> Path:
    """Get cache file path keyed by data hash."""
    data_str = '\n'.join(drug_smiles[:100]) + '\n---\n' + '\n'.join(protein_seqs[:100])
    data_str += f'\n{len(drug_smiles)}'
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
    seagate = Path('/Volumes/seagate/kirby/reps') / f'dta_{data_hash}'
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate / f'{rep_name}.npy'
    cache_dir = Path.home() / '.cache' / 'kirby' / 'reps' / f'dta_{data_hash}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{rep_name}.npy'


def get_representations(
    drug_smiles: List[str],
    protein_seqs: List[str],
    preset_or_list: str,
    cache: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate drug + protein representations.

    Drug reps are computed from SMILES using molecular.py.
    Protein reps are computed from amino acid sequences using protein.py.
    All are returned in a single dict — the framework combines them.

    Args:
        drug_smiles: SMILES strings for each drug-target pair.
        protein_seqs: Amino acid sequences for each pair.
        preset_or_list: Preset name or comma-separated rep names.
        cache: Cache computed representations to disk.
    """
    from kirby.representations import molecular as mol
    from kirby.representations import protein as prot

    module_map = {'molecular': mol, 'protein': prot}

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    # Deduplicate drug SMILES and protein seqs for efficient computation
    # (Many pairs share the same drug or protein)
    unique_drugs = sorted(set(drug_smiles))
    unique_proteins = sorted(set(protein_seqs))
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    protein_to_idx = {s: i for i, s in enumerate(unique_proteins)}

    print(f"\nGenerating {len(rep_names)} representations...")
    print(f"  Unique drugs: {len(unique_drugs)}, unique proteins: {len(unique_proteins)}")
    print(f"  Total pairs: {len(drug_smiles)}")

    representations = {}

    for rep_name in rep_names:
        if rep_name not in _REP_DISPATCH:
            print(f"  WARNING: Unknown representation '{rep_name}', skipping")
            continue

        # Try loading from cache
        if cache:
            cache_file = _rep_cache_path(drug_smiles, protein_seqs, rep_name)
            if cache_file.exists():
                rep = np.load(cache_file)
                print(f"  {rep_name}: loaded from cache ({rep.shape[1]} features)")
                representations[rep_name] = rep
                continue

        func_name, module_name, rep_type = _REP_DISPATCH[rep_name]

        try:
            start = time.time()
            print(f"  Generating {rep_name}...", end=' ', flush=True)

            mod = module_map[module_name]
            func = getattr(mod, func_name)

            if rep_type == 'drug':
                # Compute on unique drugs, then expand to all pairs
                unique_rep = func(unique_drugs)
                pair_indices = np.array([drug_to_idx[s] for s in drug_smiles])
                rep = unique_rep[pair_indices]
            elif rep_type == 'protein':
                # Compute on unique proteins, then expand to all pairs
                unique_rep = func(unique_proteins)
                pair_indices = np.array([protein_to_idx[s] for s in protein_seqs])
                rep = unique_rep[pair_indices]
            else:
                raise ValueError(f"Unknown rep type '{rep_type}' for {rep_name}")

            elapsed = time.time() - start
            print(f"done ({rep.shape[1]} features, {elapsed:.1f}s)")
            representations[rep_name] = rep

            if cache:
                cache_file = _rep_cache_path(drug_smiles, protein_seqs, rep_name)
                np.save(cache_file, rep)

        except Exception as e:
            print(f"FAILED: {e}")

    if not representations:
        raise ValueError("No representations could be generated!")

    return representations


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def _get(result, key):
    """Access result field by key — works with both dicts and objects."""
    if isinstance(result, dict):
        return result[key]
    return getattr(result, key)


_METRIC_DISPLAY = {
    'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE',
    'accuracy': 'Accuracy', 'auc': 'AUC',
}


def print_summary(result: SearchResult, task_type: str):
    """Print formatted summary to console."""
    metric = _METRIC_DISPLAY.get(result.metric_name, result.metric_name)
    lower_is_better = result.lower_is_better
    dimension_tiers = [50, 100, 200, 500]

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    best = result.best_overall
    if best is None:
        print("\nNo results found.")
        return

    bp = _get(best, 'blueprint')
    print(f"\n=== BEST OVERALL ===")
    print(f"Model: {_get(best, 'model')}, Method: {bp.allocation_method}, "
          f"Importance: {bp.importance_method}, Budget: {bp.budget}")
    print(f"{metric}: {_get(best, 'score_mean'):.4f} +/- {_get(best, 'score_std'):.4f}, "
          f"Dimension: {_get(best, 'n_features')}")

    print(f"\n=== BEST BY MODEL ===")
    models_evaluated = list(result.best_by_model.keys())
    print(f"{'Model':<10} {'Dim':<6} {'Method':<20} {'Importance':<18} {metric:<14} {'Budget'}")
    print("-" * 85)
    for model_name, r in result.best_by_model.items():
        bp = _get(r, 'blueprint')
        print(f"{model_name:<10} {_get(r, 'n_features'):<6} {bp.allocation_method:<20} "
              f"{bp.importance_method:<18} "
              f"{_get(r, 'score_mean'):.4f}+/-{_get(r, 'score_std'):.2f}   {bp.budget}")

    print(f"\n=== BEST BY DIMENSION ===")
    print(f"{'Dim':<8} {'Model':<10} {'Method':<20} {'Importance':<18} {metric:<12}")
    print("-" * 75)
    for tier, r in sorted(result.best_by_dimension.items()):
        bp = _get(r, 'blueprint')
        print(f"<={tier:<7} {_get(r, 'model'):<10} {bp.allocation_method:<20} "
              f"{bp.importance_method:<18} "
              f"{_get(r, 'score_mean'):.4f}+/-{_get(r, 'score_std'):.2f}")

    print(f"\n=== PERFORMANCE MATRIX ({metric}) ===")
    tiers = [str(t) for t in dimension_tiers]
    header = f"{'Model':<12}" + "".join(f"{'<='+t:<10}" for t in tiers)
    print(header)
    print("-" * len(header))
    for model_name in models_evaluated:
        if model_name in result.performance_matrix:
            row = f"{model_name:<12}"
            for tier in tiers:
                score = result.performance_matrix[model_name].get(tier, None)
                if score is not None:
                    row += f"{score:<10.4f}"
                else:
                    row += f"{'--':<10}"
            print(row)

    print(f"\n=== BASELINE COMPARISON (full dimension) ===")
    baselines = result.baselines
    header = f"{'Baseline':<15} {'Dim':<8}" + "".join(f"{m:<10}" for m in models_evaluated)
    print(header)
    print("-" * len(header))
    for baseline_name, baseline_scores in baselines.items():
        row = f"{baseline_name:<15} {baseline_scores.get('dimension', '?'):<8}"
        for m in models_evaluated:
            score = baseline_scores.get(m, None)
            if score is not None:
                row += f"{score:<10.4f}"
            else:
                row += f"{'--':<10}"
        print(row)

    print(f"\n=== HYBRID IMPROVEMENT ===")
    for model_name, r in result.best_by_model.items():
        hybrid_score = _get(r, 'score_mean')
        hybrid_dim = _get(r, 'n_features')

        best_baseline_name = None
        best_baseline_score = None
        for baseline_name, baseline_scores in baselines.items():
            if baseline_name == 'all_concat':
                continue
            if model_name not in baseline_scores:
                continue
            score = baseline_scores[model_name]
            if best_baseline_score is None:
                best_baseline_score = score
                best_baseline_name = baseline_name
            elif (lower_is_better and score < best_baseline_score) or \
                 (not lower_is_better and score > best_baseline_score):
                best_baseline_score = score
                best_baseline_name = baseline_name

        if best_baseline_name is not None:
            improvement = hybrid_score - best_baseline_score
            baseline_dim = baselines[best_baseline_name].get('dimension', '?')
            better = improvement < 0 if lower_is_better else improvement > 0
            sign = '' if better else '(!)'
            print(f"  {model_name}: Hybrid({hybrid_dim}d) vs {best_baseline_name}({baseline_dim}d): "
                  f"{hybrid_score:.4f} vs {best_baseline_score:.4f} ({improvement:+.4f}) {sign}")

    print("=" * 80)


def save_results(
    result: SearchResult,
    dataset_name: str,
    search_mode: str,
    reps_tested: List[str],
    output_path: str,
    base_reps: Optional[Dict[str, np.ndarray]] = None,
    labels: Optional[np.ndarray] = None,
):
    """Save results JSON."""
    from kirby.hybrid import create_hybrid

    def result_to_dict(r, include_weights: bool = False):
        bp = _get(r, 'blueprint')
        d = {
            'model': _get(r, 'model'),
            'score_mean': round(_get(r, 'score_mean'), 4),
            'score_std': round(_get(r, 'score_std'), 4),
            'dimension': _get(r, 'n_features'),
            'blueprint': bp.to_dict(),
        }

        if include_weights and base_reps is not None and labels is not None:
            try:
                _, feature_info = create_hybrid(
                    base_reps, labels,
                    allocation_method=bp.allocation_method,
                    budget=bp.budget,
                    importance_method=bp.importance_method,
                )
                weights_by_rep = {}
                for rep_name in feature_info:
                    if rep_name in ('allocation', 'greedy_history', 'greedy_feature_history'):
                        continue
                    if isinstance(feature_info[rep_name], dict) and 'importance_scores' in feature_info[rep_name]:
                        scores = feature_info[rep_name]['importance_scores']
                        indices = feature_info[rep_name]['selected_indices']
                        weights_by_rep[rep_name] = {
                            'indices': indices.tolist() if hasattr(indices, 'tolist') else list(indices),
                            'scores': scores.tolist() if hasattr(scores, 'tolist') else list(scores),
                        }
                d['feature_weights'] = weights_by_rep
            except Exception as e:
                print(f"  Warning: Could not extract weights for {_get(r, 'model')}: {e}")

        return d

    best = result.best_overall
    output = {
        'metadata': {
            'dataset': dataset_name,
            'search_mode': search_mode,
            'reps_available': reps_tested,
            'timestamp': datetime.now().isoformat(),
            'domain': 'dta',
        },
        'best_overall': result_to_dict(best, include_weights=True) if best else None,
        'best_by_dimension': {
            str(tier): result_to_dict(r)
            for tier, r in result.best_by_dimension.items()
        },
        'best_by_model': {
            model: result_to_dict(r, include_weights=True)
            for model, r in result.best_by_model.items()
        },
        'performance_matrix': result.performance_matrix,
        'baselines': result.baselines,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find the optimal hybrid DTA representation for each model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s davis --reps fast --search fast
  %(prog)s davis --reps fast+lm --search medium
  %(prog)s kiba --reps fast --search fast

Output:
  - Best overall: single best model+hybrid combination
  - Best by dimension: optimal at each tier (50, 100, 200, 500)
  - Best by model: optimal hybrid per model type
  - Performance matrix: full Model x Dimension grid
  - Baseline comparison: hybrid vs full-dimension baselines
        """
    )

    parser.add_argument('dataset', type=str,
                       help="'davis' or 'kiba'")
    parser.add_argument('--reps', type=str, default='fast',
                       help="'fast', 'fast+lm', 'full', or comma-separated list")
    parser.add_argument('--search', type=str, default='fast',
                       choices=list(_SEARCH_NAME_MAP.keys()),
                       help="Search speed: 'fast', 'medium', 'long'")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default=None,
                       help="Override models: 'rf', 'xgboost', 'ridge', etc. or comma-separated")
    parser.add_argument('--save-weights', action='store_true',
                       help='Fit best config on full data and save feature weights')
    parser.add_argument('--weight-method', type=str, default='lightgbm_gain',
                       help="Importance method for feature weights (default: lightgbm_gain)")
    parser.add_argument('--test-rep-subsets', action='store_true',
                       help='Also test leave-one-rep-out subsets (when >=3 reps)')
    parser.add_argument('--cache', action='store_true',
                       help='Cache hybrid search results to disk')
    parser.add_argument('--max-pairs', type=int, default=None,
                       help='Subsample to this many pairs (for KIBA which has 117K)')
    parser.add_argument('--adaptive-pruning', action='store_true',
                       help='Enable adaptive model pruning during search')
    parser.add_argument('--n-folds', type=int, default=None,
                       help='Override number of CV folds (default: from speed tier)')

    args = parser.parse_args()

    speed = _SEARCH_NAME_MAP.get(args.search, args.search)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    drug_smiles = data['drug_smiles']
    protein_seqs = data['protein_seqs']
    labels = data['labels']
    task_type = data['task_type']
    metadata = data['metadata']

    # Subsample if requested (KIBA is very large)
    if args.max_pairs and len(labels) > args.max_pairs:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(labels), args.max_pairs, replace=False)
        idx.sort()
        drug_smiles = [drug_smiles[i] for i in idx]
        protein_seqs = [protein_seqs[i] for i in idx]
        labels = labels[idx]
        metadata['drug_ids'] = [metadata['drug_ids'][i] for i in idx]
        metadata['protein_ids'] = [metadata['protein_ids'][i] for i in idx]
        metadata['n_drugs'] = len(set(metadata['drug_ids']))
        metadata['n_proteins'] = len(set(metadata['protein_ids']))
        print(f"  Subsampled to {args.max_pairs} pairs")

    print(f"  {len(labels)} pairs, {metadata['n_drugs']} drugs × {metadata['n_proteins']} proteins")
    print(f"  Task: {task_type}")

    # Cold-target grouping: group by protein for GroupKFold
    from kirby.datasets.dta import assign_cold_target_groups
    protein_ids = metadata['protein_ids']
    groups, n_groups = assign_cold_target_groups(protein_ids)
    print(f"  Cold-target groups: {n_groups} unique proteins")

    # Generate representations
    base_reps = get_representations(
        drug_smiles, protein_seqs, args.reps, cache=args.cache,
    )

    # Resolve model override
    models_arg = None
    if args.model:
        models_arg = [m.strip() for m in args.model.split(',')]

    # Call domain-agnostic search
    result = find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        models=models_arg,
        metric='r2',
        task_type=task_type,
        speed=speed,
        n_folds=args.n_folds,
        return_fitted=args.save_weights,
        weight_method=args.weight_method if args.save_weights else None,
        test_rep_subsets=args.test_rep_subsets,
        seed=args.seed,
        cache=args.cache,
        adaptive_pruning=args.adaptive_pruning,
    )

    # Print summary
    print_summary(result, task_type)

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = f"results_{args.dataset}_{args.search}.json"

    save_results(result, args.dataset, args.search, list(base_reps.keys()), output_path,
                 base_reps=base_reps, labels=labels)

    return 0


if __name__ == '__main__':
    sys.exit(main())
