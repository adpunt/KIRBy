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
        'drug_chemberta', 'drug_molformer', 'drug_clamp',
        'protein_esm2', 'protein_esmc', 'protein_aac', 'protein_descriptors',
    ],  # 10 reps
    'full': [
        'drug_ecfp4', 'drug_maccs', 'drug_pdv', 'drug_pdv_binary_mean',
        'drug_pdv_rank', 'drug_map4', 'drug_mol2vec',
        'drug_mordred', 'drug_avalon',
        'drug_chemberta', 'drug_molformer', 'drug_selformer', 'drug_clamp',
        'drug_rmat', 'drug_unimol_v2',
        'drug_minimol', 'drug_gp_molformer', 'drug_mole',
        'drug_chemberta_pca50', 'drug_molformer_pca50', 'drug_selformer_pca50',
        'drug_clamp_pca50', 'drug_rmat_pca50',
        'drug_mordred_pca50',
        'drug_sns',
        'protein_esm2', 'protein_esmc', 'protein_esm2_midlayers',
        'protein_amplify', 'protein_esmfold', 'protein_alphafold_db',
        'protein_aac', 'protein_descriptors', 'protein_kmer',
        'protein_mint',
        'protein_esm3', 'protein_saprot',
        'protein_aido',
        'protein_xtrimopglm',
        'protein_contact_graph', 'protein_usr_shape',
        'protein_pocket', 'protein_esm2_pocket', 'protein_pocket_composition',
    ],  # 44 reps (25 drug + 19 protein)
}

# Reps that require per-fold fitting (data-dependent transforms).
# These are created as rep_factories instead of base_reps to prevent leakage.
FACTORY_REPS = {
    'drug_pdv_binary_mean', 'drug_pdv_rank',
    'drug_chemberta_pca50', 'drug_molformer_pca50', 'drug_selformer_pca50',
    'drug_clamp_pca50', 'drug_rmat_pca50', 'drug_mordred_pca50',
    'drug_sns',
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
    'drug_pdv_binary_mean': ('create_pdv_binary_mean', 'molecular', 'drug'),
    'drug_pdv_rank':   ('create_pdv_rank',    'molecular', 'drug'),
    'drug_chemberta':  ('create_chemberta',   'molecular', 'drug'),
    'drug_molformer':  ('create_molformer',   'molecular', 'drug'),
    'drug_selformer':  ('create_selformer',   'molecular', 'drug'),
    'drug_mol2vec':    ('create_mol2vec',     'molecular', 'drug'),
    'drug_clamp':      ('create_clamp',       'molecular', 'drug'),
    'drug_rmat':       ('create_rmat',        'molecular', 'drug'),
    'drug_map4':       ('create_map4',        'molecular', 'drug'),
    'drug_unimol_v2':  ('create_unimol_v2',   'molecular', 'drug'),
    'drug_minimol':    ('create_minimol',     'molecular', 'drug'),
    'drug_gp_molformer': ('create_gp_molformer', 'molecular', 'drug'),
    'drug_mole':       ('create_mole',        'molecular', 'drug'),
    'drug_mordred':    ('create_mordred',     'molecular', 'drug'),
    'drug_avalon':     ('create_avalon',      'molecular', 'drug'),
    # PCA-reduced drug LM embeddings (50D) — only for reps already in DTA
    'drug_chemberta_pca50':  ('create_chemberta',  'molecular', 'drug', 'pca50'),
    'drug_molformer_pca50':  ('create_molformer',  'molecular', 'drug', 'pca50'),
    'drug_selformer_pca50':  ('create_selformer',  'molecular', 'drug', 'pca50'),
    'drug_clamp_pca50':      ('create_clamp',      'molecular', 'drug', 'pca50'),
    'drug_rmat_pca50':       ('create_rmat',       'molecular', 'drug', 'pca50'),
    'drug_mordred_pca50':    ('create_mordred',    'molecular', 'drug', 'pca50'),
    # Protein reps (from kirby.representations.protein)
    'protein_esm2':        ('create_esm2_protein_embeddings', 'protein', 'protein'),
    'protein_aac':         ('create_protein_aac',             'protein', 'protein'),
    'protein_descriptors': ('create_protein_descriptors',     'protein', 'protein'),
    'protein_kmer':        ('create_protein_kmer',            'protein', 'protein'),
    'protein_esmc':        ('create_esmc_protein_embeddings', 'protein', 'protein'),
    'protein_esm2_midlayers': ('create_esm2_midlayers_protein_embeddings', 'protein', 'protein'),
    'protein_amplify':     ('create_amplify_protein_embeddings', 'protein', 'protein'),
    'protein_esmfold':     ('create_esmfold_protein_features',   'protein', 'protein'),
    'protein_alphafold_db': ('create_alphafold_db_features',     'protein', 'protein'),
    'protein_mint':    ('create_mint_protein_embeddings',       'protein', 'protein'),
    'protein_esm3':    ('create_esm3_protein_embeddings',       'protein', 'protein'),
    'protein_saprot':  ('create_saprot_protein_embeddings',     'protein', 'protein'),
    'protein_aido':    ('create_aido_protein_embeddings',       'protein', 'protein'),
    'protein_xtrimopglm': ('create_xtrimopglm_protein_embeddings', 'protein', 'protein'),
    # Structural protein reps
    'protein_contact_graph':     ('create_contact_graph_features',       'protein', 'protein'),
    'protein_usr_shape':         ('create_usr_shape_features',           'protein', 'protein'),
    'protein_pocket':            ('create_pocket_features',              'protein', 'protein'),
    'protein_esm2_pocket':       ('create_esm2_pocket_embeddings',       'protein', 'protein'),
    'protein_pocket_composition': ('create_pocket_composition_features', 'protein', 'protein'),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(name: str):
    """Load DTA dataset. Thin wrapper around kirby.datasets.dta/pdbbind."""
    if name.startswith('pdbbind'):
        from kirby.datasets.pdbbind import load_dataset as _load_pdbbind
        return _load_pdbbind(name)
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
    protein_ids: Optional[List[str]] = None,
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
        protein_ids: Optional protein identifiers (UniProt IDs or gene names)
            for AlphaFold DB lookups.
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

    # Map unique protein seqs to their identifiers (for AlphaFold DB)
    unique_protein_ids = None
    if protein_ids is not None:
        seq_to_pid = {}
        for seq, pid in zip(protein_seqs, protein_ids):
            if seq not in seq_to_pid:
                seq_to_pid[seq] = pid
        unique_protein_ids = [seq_to_pid.get(s) for s in unique_proteins]

    # Filter out factory reps — these are handled by get_rep_factories()
    base_rep_names = [r for r in rep_names if r not in FACTORY_REPS]

    print(f"\nGenerating {len(base_rep_names)} representations...")
    print(f"  Unique drugs: {len(unique_drugs)}, unique proteins: {len(unique_proteins)}")
    print(f"  Total pairs: {len(drug_smiles)}")

    representations = {}

    for rep_name in base_rep_names:
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

        dispatch_entry = _REP_DISPATCH[rep_name]
        func_name, module_name, rep_type = dispatch_entry[:3]
        transform = dispatch_entry[3] if len(dispatch_entry) > 3 else None

        try:
            start = time.time()
            print(f"  Generating {rep_name}...", end=' ', flush=True)

            mod = module_map[module_name]
            func = getattr(mod, func_name)

            # pca50 reps are in FACTORY_REPS and handled by
            # get_rep_factories() — they never reach this loop.
            if transform == 'pca50':
                raise RuntimeError(
                    f"{rep_name} has pca50 transform but is not in "
                    f"FACTORY_REPS — add it there to prevent leakage")

            if rep_type == 'drug':
                # Compute on unique drugs, then expand to all pairs
                unique_rep = func(unique_drugs)
                pair_indices = np.array([drug_to_idx[s] for s in drug_smiles])
                rep = unique_rep[pair_indices]
            elif rep_type == 'protein':
                # Compute on unique proteins, then expand to all pairs
                # Reps that benefit from UniProt IDs (structure lookup)
                _needs_uniprot = {
                    'protein_alphafold_db', 'protein_contact_graph',
                    'protein_usr_shape', 'protein_pocket',
                    'protein_esm2_pocket', 'protein_pocket_composition',
                }
                if rep_name in _needs_uniprot and unique_protein_ids is not None:
                    from kirby.representations.protein import _resolve_uniprot_ids
                    resolved_ids = _resolve_uniprot_ids(unique_protein_ids)
                    unique_rep = func(unique_proteins, uniprot_ids=resolved_ids)
                else:
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
# PER-FOLD FACTORIES (leakage-free drug reps)
# =============================================================================

def get_rep_factories(
    drug_smiles: List[str],
    preset_or_list: str,
) -> Dict[str, any]:
    """Create per-fold rep factories for data-dependent drug representations.

    DTA reps are computed on unique drugs then expanded to pair-level indices.
    Factories pre-compute the base embedding at pair level so that train_idx /
    val_idx (which are pair-level) slice correctly.

    Args:
        drug_smiles: SMILES strings for each drug-target pair.
        preset_or_list: Preset name or comma-separated rep names.

    Returns:
        Dict of {rep_name: factory_callable} for reps that need per-fold fitting.
    """
    from kirby.representations import molecular as mol

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    requested = [r for r in rep_names if r in FACTORY_REPS]
    if not requested:
        return {}

    print(f"\nCreating {len(requested)} per-fold drug rep factories...")

    # Pre-compute unique drug embeddings once, expand to pair level
    unique_drugs = sorted(set(drug_smiles))
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    pair_drug_indices = np.array([drug_to_idx[s] for s in drug_smiles])

    factories = {}

    # PDV-derived reps (binary_mean, rank)
    pdv_needed = set(requested) & {'drug_pdv_binary_mean', 'drug_pdv_rank'}
    pair_pdv = None
    if pdv_needed:
        print("  Pre-computing PDV for factory reps...", end=' ', flush=True)
        unique_pdv = mol.create_pdv(unique_drugs)
        pair_pdv = unique_pdv[pair_drug_indices]
        print(f"done ({pair_pdv.shape[1]} features)")

    if 'drug_pdv_binary_mean' in requested and pair_pdv is not None:
        def _make_binary_mean(full_pdv):
            def factory(train_idx, val_idx, train_labels):
                train_pdv = full_pdv[train_idx]
                val_pdv = full_pdv[val_idx]
                col_means = train_pdv.mean(axis=0)
                train_out = (train_pdv > col_means).astype(np.float32)
                val_out = (val_pdv > col_means).astype(np.float32)
                return train_out, val_out
            return factory
        factories['drug_pdv_binary_mean'] = _make_binary_mean(pair_pdv)
        print("  drug_pdv_binary_mean factory ready")

    if 'drug_pdv_rank' in requested and pair_pdv is not None:
        def _make_rank(full_pdv):
            def factory(train_idx, val_idx, train_labels):
                from scipy.stats import rankdata
                train_pdv = full_pdv[train_idx]
                val_pdv = full_pdv[val_idx]
                n_train = train_pdv.shape[0]
                train_ranked = np.zeros_like(train_pdv)
                val_ranked = np.zeros_like(val_pdv)
                for j in range(train_pdv.shape[1]):
                    train_ranked[:, j] = rankdata(train_pdv[:, j], method='average') / n_train
                    sorted_train = np.sort(train_pdv[:, j])
                    val_ranked[:, j] = np.searchsorted(sorted_train, val_pdv[:, j]) / n_train
                return train_ranked.astype(np.float32), val_ranked.astype(np.float32)
            return factory
        factories['drug_pdv_rank'] = _make_rank(pair_pdv)
        print("  drug_pdv_rank factory ready")

    # PCA-reduced LM reps
    pca_base_fns = {
        'drug_chemberta_pca50': 'create_chemberta',
        'drug_molformer_pca50': 'create_molformer',
        'drug_selformer_pca50': 'create_selformer',
        'drug_clamp_pca50': 'create_clamp',
        'drug_rmat_pca50': 'create_rmat',
        'drug_mordred_pca50': 'create_mordred',
    }
    for rep_name, func_name in pca_base_fns.items():
        if rep_name not in requested:
            continue
        try:
            start = time.time()
            print(f"  Pre-computing {func_name} for {rep_name}...", end=' ', flush=True)
            base_fn = getattr(mol, func_name)
            unique_embs = base_fn(unique_drugs)
            pair_embs = unique_embs[pair_drug_indices]

            def _make_pca(full_X, n_components=50):
                def factory(train_idx, val_idx, train_labels):
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    X_train, X_val = full_X[train_idx], full_X[val_idx]
                    nc = min(n_components, X_train.shape[1], X_train.shape[0])
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_val_s = scaler.transform(X_val)
                    pca = PCA(n_components=nc, random_state=42)
                    return (pca.fit_transform(X_train_s).astype(np.float32),
                            pca.transform(X_val_s).astype(np.float32))
                return factory

            factories[rep_name] = _make_pca(pair_embs)
            elapsed = time.time() - start
            print(f"done ({pair_embs.shape[1]}D base, {elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Sort & Slice (SNS) — fitted per fold on unique training drugs
    if 'drug_sns' in requested:
        print("  Setting up drug_sns factory...", end=' ', flush=True)
        start = time.time()

        def _make_sns(pair_smiles, uniq_drugs, pair_idx):
            def factory(train_idx, val_idx, train_labels):
                # Get unique training drug SMILES for fitting
                train_drug_set = set(pair_idx[train_idx])
                unique_train_drugs = [uniq_drugs[i] for i in sorted(train_drug_set)]

                # Fit SNS on unique training drugs only
                _, featurizer = mol.create_sns(
                    unique_train_drugs, return_featurizer=True)

                # Transform ALL unique drugs with fitted featurizer
                all_features = mol.create_sns(
                    uniq_drugs, reference_featurizer=featurizer)

                # Expand to pair level
                train_features = all_features[pair_idx[train_idx]]
                val_features = all_features[pair_idx[val_idx]]

                return train_features, val_features
            return factory

        factories['drug_sns'] = _make_sns(
            drug_smiles, unique_drugs, pair_drug_indices)
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")

    return factories


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
    'spearman': 'Spearman ρ', 'pearson': 'Pearson r',
    'ci': 'CI', 'f1_weighted': 'F1w',
}


def print_summary(result: SearchResult, task_type: str):
    """Print formatted summary to console."""
    metric = _METRIC_DISPLAY.get(result.metric_name, result.metric_name)
    lower_is_better = result.lower_is_better
    dimension_tiers = [50, 100, 200, 500]

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    if not result.best_by_model:
        print("\nNo results found.")
        return

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

    # Secondary metrics for baselines
    sec_keys = set()
    for baseline_scores in baselines.values():
        if '_secondary' in baseline_scores:
            sec_keys.update(baseline_scores['_secondary'].keys())
    if sec_keys:
        print(f"\n=== SECONDARY METRICS (baselines) ===")
        # Group by metric name
        sec_metric_names = sorted(set(k.split('_', 1)[1] for k in sec_keys if '_' in k))
        for sec_metric in sec_metric_names:
            display_name = _METRIC_DISPLAY.get(sec_metric, sec_metric)
            print(f"\n  {display_name}:")
            header = f"  {'Baseline':<15} {'Dim':<8}" + "".join(f"{m:<10}" for m in models_evaluated)
            print(header)
            for baseline_name, baseline_scores in baselines.items():
                sec = baseline_scores.get('_secondary', {})
                row = f"  {baseline_name:<15} {baseline_scores.get('dimension', '?'):<8}"
                for m in models_evaluated:
                    key = f'{m}_{sec_metric}'
                    val = sec.get(key, {}).get('mean')
                    if val is not None:
                        row += f"{val:<10.4f}"
                    else:
                        row += f"{'--':<10}"
                print(row)

    # Top-3 base representations (ranked by RF)
    if baselines:
        metric_name = result.metric_name
        rf_scores = []
        for rep_name, model_scores in baselines.items():
            if rep_name.startswith('_'):
                continue
            rf_score = model_scores.get('rf')
            if rf_score is not None and not np.isnan(rf_score):
                rf_scores.append((rep_name, rf_score))
        rf_scores.sort(key=lambda x: x[1], reverse=not lower_is_better)
        top3 = rf_scores[:3]
        if top3:
            print(f"\n  TOP-3 BASE REPS (by RF {metric}):")
            for i, (rname, score) in enumerate(top3, 1):
                print(f"    {i}. {rname:<25} {score:.4f}")

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
    cold_setting: str = 'target',
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

    output = {
        'metadata': {
            'dataset': dataset_name,
            'search_mode': search_mode,
            'cold_setting': cold_setting,
            'reps_available': reps_tested,
            'timestamp': datetime.now().isoformat(),
            'domain': 'dta',
        },
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

    # Extract top-3 base reps by RF score
    lower_is_better = result.lower_is_better
    rf_scores = []
    for rep_name, model_scores in result.baselines.items():
        if rep_name.startswith('_'):
            continue
        rf_score = model_scores.get('rf')
        if rf_score is not None and not np.isnan(rf_score):
            rf_scores.append((rep_name, rf_score))
    rf_scores.sort(key=lambda x: x[1], reverse=not lower_is_better)
    output['top3_base_reps'] = [
        {'rep': name, 'rf_score': round(score, 4)} for name, score in rf_scores[:3]
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def generate_explanations(
    best_result: dict,
    base_reps: Dict[str, np.ndarray],
    drug_smiles: List[str],
    labels: np.ndarray,
    output_path: str,
):
    """Generate and save feature explanations for the best DTA configuration."""
    from kirby.hybrid import create_hybrid
    from kirby.importance.hybrid import explain_hybrid, format_explanation_summary

    print("\n=== GENERATING EXPLANATIONS ===")

    bp = _get(best_result, 'blueprint')

    print(f"  Recreating hybrid: method={bp.allocation_method}, budget={bp.budget}")
    _, feature_info = create_hybrid(
        base_reps, labels,
        allocation_method=bp.allocation_method,
        budget=bp.budget,
        importance_method=bp.importance_method,
    )

    print("  Mapping features to interpretations...")
    explanations = explain_hybrid(
        feature_info=feature_info,
        smiles_list=drug_smiles[:100],
        top_k=30,
    )

    print("\n" + format_explanation_summary(explanations, max_features=15))

    explain_path = output_path.replace('.json', '_explanations.json')
    Path(explain_path).parent.mkdir(parents=True, exist_ok=True)

    explain_output = {
        'model': _get(best_result, 'model'),
        'blueprint': bp.to_dict(),
        'by_representation': explanations['by_representation'],
        'top_features': [
            {
                'rank': i + 1,
                'rep': f['rep_name'],
                'local_idx': f['local_idx'],
                'importance': round(f['importance'], 6),
                'description': f['interpretation'].get('description', ''),
                'type': f['interpretation'].get('type', ''),
            }
            for i, f in enumerate(explanations['top_features'])
        ],
    }

    with open(explain_path, 'w') as f:
        json.dump(explain_output, f, indent=2)

    print(f"\nExplanations saved to: {explain_path}")
    return explanations


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
    parser.add_argument('--allocation', type=str, default=None,
                       help="Override allocation methods: comma-separated from "
                            "'greedy', 'performance_weighted', 'softmax_T5', "
                            "'dominant_greedy', 'two_stage', 'stability_selection'")
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
    parser.add_argument('--n-folds', type=int, default=None,
                       help='Override number of CV folds (default: from speed tier)')
    parser.add_argument('--finetune', type=str, default=None,
                       help='Comma-separated fine-tuning methods: '
                            'protein_esm2_lora')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                       help='Fine-tuning epochs per fold (default: 5)')
    parser.add_argument('--no-ft-cache', action='store_true',
                       help='Disable fine-tuning cache')
    parser.add_argument('--tune', action='store_true',
                       help='Tune model HPs via RandomizedSearchCV before hybrid search')
    parser.add_argument('--tune-iter', type=int, default=20,
                       help='RandomizedSearchCV iterations per model (default: 20)')
    parser.add_argument('--explain', action='store_true',
                       help='Generate feature explanations for best hybrid config')
    parser.add_argument('--cold-setting', choices=['target', 'protein', 'drug'], default='target',
                       help='Cold split: target/protein (default, 30%% seq identity clusters) '
                            'or drug (unique drug groups)')

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

    # Cold-target or cold-drug grouping for GroupKFold
    # PDBBind: use LP-PDBBind leak-proof grouping (30% seq identity clusters)
    protein_ids = metadata['protein_ids']
    from kirby.datasets.dta import assign_cold_target_groups, assign_cold_drug_groups
    if args.dataset.startswith('pdbbind'):
        from kirby.datasets.pdbbind import assign_lp_groups
        groups, n_groups = assign_lp_groups(protein_seqs)
        print(f"  LP-PDBBind groups: {n_groups} clusters (30% seq identity)")
    elif args.cold_setting == 'drug':
        groups, n_groups = assign_cold_drug_groups(metadata['drug_ids'])
        print(f"  Cold-drug groups: {n_groups} unique drugs")
    else:
        groups, n_groups = assign_cold_target_groups(protein_ids)
        print(f"  Cold-target groups: {n_groups} unique proteins")

    # Generate representations
    base_reps = get_representations(
        drug_smiles, protein_seqs, args.reps, cache=args.cache,
        protein_ids=protein_ids,
    )

    # Create per-fold factories for data-dependent drug reps (prevents leakage)
    leakage_free_factories = get_rep_factories(drug_smiles, args.reps)

    # Resolve model override
    models_arg = None
    if args.model:
        models_arg = [m.strip() for m in args.model.split(',')]

    alloc_arg = None
    if args.allocation:
        alloc_arg = [a.strip() for a in args.allocation.split(',')]

    # Hyperparameter tuning
    if args.tune:
        from kirby.find_best_hybrid import tune_models
        print("\n=== Hyperparameter Tuning ===")
        tuned_models = tune_models(
            base_reps=base_reps, labels=labels, groups=groups,
            task_type=task_type, speed=speed, n_iter=args.tune_iter,
            seed=args.seed,
        )
        models_arg = tuned_models

    # Build fine-tuning factories
    rep_factories = None
    if args.finetune:
        ft_methods = [m.strip() for m in args.finetune.split(',')]
        rep_factories = {}
        ft_cache = not args.no_ft_cache

        for method in ft_methods:
            if method == 'protein_esm2_lora':
                from kirby.finetuning.dta_protein_lora import make_dta_protein_lora_factory
                rep_factories['protein_esm2_lora'] = make_dta_protein_lora_factory(
                    protein_seqs=protein_seqs,
                    drug_smiles=drug_smiles,
                    dataset_name=args.dataset,
                    base_model='esm2',
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            else:
                print(f"  WARNING: Unknown fine-tuning method '{method}', skipping")

        if rep_factories:
            print(f"\nFine-tuning factories: {list(rep_factories.keys())}")
        else:
            rep_factories = None

    # Merge leakage-free factories with fine-tuning factories
    if leakage_free_factories:
        if rep_factories is None:
            rep_factories = {}
        rep_factories.update(leakage_free_factories)

    # Call domain-agnostic search
    result = find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        models=models_arg,
        metric='ci',
        secondary_metrics=['r2', 'spearman', 'rmse'],
        task_type=task_type,
        speed=speed,
        n_folds=args.n_folds,
        allocation_methods=alloc_arg,
        return_fitted=args.save_weights,
        weight_method=args.weight_method if args.save_weights else None,
        test_rep_subsets=args.test_rep_subsets,
        seed=args.seed,
        cache=args.cache,
        dataset_name=args.dataset,
        tier='general',
        rep_factories=rep_factories,
    )

    # Print summary
    print_summary(result, task_type)

    # Save results
    if args.output:
        output_path = args.output
    else:
        tune_suffix = '_tuned' if args.tune else ''
        cold_suffix = f'_cold_{args.cold_setting}' if args.cold_setting != 'target' else ''
        output_path = f"results_{args.dataset}_{args.search}{tune_suffix}{cold_suffix}.json"

    save_results(result, args.dataset, args.search, list(base_reps.keys()), output_path,
                 base_reps=base_reps, labels=labels,
                 cold_setting=args.cold_setting)

    # Explanations (iterate over models)
    if args.explain and result.best_by_model:
        for model_name, best_result in result.best_by_model.items():
            print(f"\n--- Explanations for model: {model_name} ---")
            generate_explanations(best_result, base_reps, drug_smiles, labels,
                                  output_path.replace('.json', f'_{model_name}.json'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
