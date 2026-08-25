#!/usr/bin/env python3
"""
Find Best Hybrid TCGA Representation

Thin TCGA wrapper around kirby.find_best_hybrid. Handles expression data loading,
genomic/WSI representation generation, TSS grouping, and output formatting.
The actual search logic lives in the domain-agnostic library.

Datasets:
  - tcga_luad_vs_lusc:  Binary classification — LUAD vs LUSC (~1016 samples)
  - tcga_brca_pam50:    4-class classification — PAM50 subtypes (~900 samples)
  - tcga_brca_survival: Regression — overall survival days (~400-500 samples)

Usage:
    # Quick local test
    python find_best_hybrid_tcga.py tcga_luad_vs_lusc --reps fast --search fast

    # Medium search with more reps
    python find_best_hybrid_tcga.py tcga_brca_pam50 --reps fast+learned --search medium

    # Full search
    python find_best_hybrid_tcga.py tcga_brca_survival --reps all --search long

    # Genomic-only (skip WSI reps)
    python find_best_hybrid_tcga.py tcga_luad_vs_lusc --reps fast --genomic-only --search fast
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
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kirby.find_best_hybrid import find_best_hybrid, SearchResult, Blueprint

warnings.filterwarnings('ignore')


# =============================================================================
# TSS GROUPING
# =============================================================================

def assign_tss_groups(sample_ids: List[str]) -> Tuple[np.ndarray, int]:
    """Assign tissue source site (TSS) group IDs for GroupKFold.

    TSS from TCGA barcode positions 5-6 (second hyphen-separated field).
    Each TSS represents a contributing institution — grouping prevents
    data leakage from site-specific confounders (Howard et al. 2021).

    Returns:
        (group_ids, n_unique_sites)
    """
    from kirby.datasets.tcga import _extract_tss_groups
    groups = _extract_tss_groups(sample_ids)
    n_unique = len(np.unique(groups))
    return groups, n_unique


# =============================================================================
# CONFIGURATION
# =============================================================================

REP_PRESETS = {
    'fast': [
        'gene_pca100', 'gene_hallmark', 'gene_progeny',
        'wsi_ctranspath', 'wsi_phikon',
    ],  # 5 reps

    'fast+diverse': [  # Maximally diverse paradigms, small footprint
        'gene_pca100', 'gene_hallmark', 'gene_progeny',
        'gene_nmf50', 'gene_stats', 'gene_immune',
        'gene_quantile_rank',
        'wsi_ctranspath', 'wsi_phikon',
    ],  # 9 reps

    'multimodal': [  # Cross-modality showcase
        'gene_pca100', 'gene_hallmark', 'gene_immune',
        'gene_nmf50', 'gene_stats',
        'cnv_gistic', 'rppa', 'mirna_top200', 'methyl_pca50',
        'wsi_ctranspath', 'wsi_phikon',
    ],  # 11 reps

    'fast+learned': [
        'gene_hvg2000', 'gene_pca100', 'gene_hallmark', 'gene_progeny',
        'gene_kegg', 'gene_dorothea', 'gene_combined_pathway',
        'gene_nmf50', 'gene_ica50', 'gene_stats',
        'gene_geneformer', 'gene_geneformer_v2', 'gene_bulkrnabert',
        'gene_scfoundation', 'gene_scgpt', 'gene_uce',
        'wsi_ctranspath', 'wsi_phikon', 'wsi_hoptimus0', 'wsi_moco',
    ],  # 20 reps

    'all': [
        # Expression-derived (18)
        'gene_hvg2000', 'gene_pca100', 'gene_hallmark', 'gene_progeny',
        'gene_kegg', 'gene_dorothea', 'gene_combined_pathway',
        'gene_nmf50', 'gene_ica50', 'gene_stats',
        'gene_reactome', 'gene_oncogenic', 'gene_immune',
        'gene_quantile_rank', 'gene_corr_modules', 'gene_chromosome_arm',
        'gene_autoencoder50', 'gene_sparse_pca50',
        # Foundation model embeddings (6, GPU)
        'gene_geneformer', 'gene_geneformer_v2', 'gene_bulkrnabert',
        'gene_scfoundation', 'gene_scgpt', 'gene_uce',
        # Multi-omic (23)
        'cnv_gistic', 'cnv_pca50', 'rppa', 'mirna_top200',
        'methyl_pca50', 'methyl_hvg',
        'mutation_matrix', 'mut_tmb', 'mut_driver', 'mut_pathway', 'mut_pca50',
        'sig_sbs', 'sig_dbs', 'sig_id',
        'immune_cibersort', 'immune_xcell', 'stemness_mrnasi',
        'purity_consensus', 'heterogeneity_math',
        'msi_status', 'hrd_score', 'cimp_status',
        'clinical',
        # Tier 2 multi-omic (4)
        'lncrna_pca50', 'gsva_hallmark', 'promoter_meth', 'splicing_psi50',
        # WSI (10)
        'wsi_ctranspath', 'wsi_phikon', 'wsi_hoptimus0', 'wsi_gigapath', 'wsi_moco',
        'wsi_titan', 'wsi_uni2h', 'wsi_virchow2', 'wsi_conch', 'wsi_hoptimus1',
        # New omics reps (2026-06-05): learned fusion, consensus subtype,
        # variational + survival-supervised latents, CPTAC proteomics.
        # Factory/graceful-skip — cptac & survival skip cleanly if data/labels absent.
        'omics_fusion_ae50', 'consensus_subtype', 'vae_latent50',
        'survival_latent50', 'cptac_proteomics',
    ],  # 66 reps

    # Genomic-only presets (no WSI)
    'genomic_fast': [
        'gene_pca100', 'gene_hallmark', 'gene_progeny',
    ],  # 3 reps

    'genomic_diverse': [  # Multi-omic without WSI
        'gene_pca100', 'gene_hallmark', 'gene_immune',
        'gene_nmf50', 'gene_stats',
        'cnv_gistic', 'rppa', 'mirna_top200', 'methyl_pca50',
        'mut_tmb', 'purity_consensus',
    ],  # 11 reps

    'genomic_all': [
        'gene_hvg2000', 'gene_pca100', 'gene_hallmark', 'gene_progeny',
        'gene_kegg', 'gene_dorothea', 'gene_combined_pathway',
        'gene_nmf50', 'gene_ica50', 'gene_stats',
        'gene_reactome', 'gene_oncogenic', 'gene_immune',
        'gene_quantile_rank', 'gene_corr_modules', 'gene_chromosome_arm',
        'gene_autoencoder50', 'gene_sparse_pca50',
        'gene_geneformer', 'gene_geneformer_v2', 'gene_bulkrnabert',
        'gene_scfoundation', 'gene_scgpt', 'gene_uce',
        'cnv_gistic', 'cnv_pca50', 'rppa', 'mirna_top200',
        'methyl_pca50', 'methyl_hvg',
        'mutation_matrix', 'mut_tmb', 'mut_driver', 'mut_pathway', 'mut_pca50',
        'sig_sbs', 'sig_dbs', 'sig_id',
        'immune_cibersort', 'immune_xcell', 'stemness_mrnasi',
        'purity_consensus', 'heterogeneity_math',
        'msi_status', 'hrd_score', 'cimp_status',
        'clinical',
        # Tier 2 multi-omic (4)
        'lncrna_pca50', 'gsva_hallmark', 'promoter_meth', 'splicing_psi50',
    ],  # 51 reps
}


# Reps that require per-fold fitting (z-scoring, PCA, etc.).
# These are created as rep_factories instead of base_reps to prevent leakage.
FACTORY_REPS = {
    'gene_hvg2000', 'gene_pca100',
    'gene_hallmark', 'gene_progeny', 'gene_kegg', 'gene_dorothea',
    'gene_combined_pathway', 'gene_reactome', 'gene_oncogenic',
    'gene_immune',
    'gene_nmf50', 'gene_ica50', 'gene_sparse_pca50',
    'gene_quantile_rank', 'gene_corr_modules', 'gene_autoencoder50',
    # Tier 2 factory reps
    'lncrna_pca50', 'gsva_hallmark', 'promoter_meth', 'splicing_psi50',
    # Multi-omics fusion / consensus subtype / VAE / survival-supervised latents
    # (all per-fold, leakage-safe — see make_*_factory in genomic.py)
    'omics_fusion_ae50', 'consensus_subtype', 'vae_latent50',
    'survival_latent50',
}

# Dispatch: (function_name, rep_type)
# rep_type determines how the function is called:
#   'genomic':      fn(expression, gene_names)
#   'genomic_expr': fn(expression) — no gene_names needed
#   'mutation':     fn(sample_ids, cohort)
#   'wsi':          load_wsi_features(sample_ids, wsi_dir, model_key)
_REP_DISPATCH = {
    # Genomic needing gene_names: fn(expression, gene_names) -> np.ndarray
    'gene_hvg2000':          ('create_hvg2000',           'genomic'),
    'gene_hallmark':         ('create_hallmark_scores',   'genomic'),
    'gene_progeny':          ('create_progeny_scores',    'genomic'),
    'gene_kegg':             ('create_kegg_scores',       'genomic'),
    'gene_dorothea':         ('create_dorothea_scores',   'genomic'),
    'gene_combined_pathway': ('create_combined_pathway',  'genomic'),
    'gene_reactome':         ('create_reactome_scores',   'genomic'),
    'gene_oncogenic':        ('create_oncogenic_scores',  'genomic'),
    'gene_immune':           ('create_immune_scores',     'genomic'),
    # Genomic expression-only: fn(expression) -> np.ndarray
    'gene_pca100':           ('create_gene_pca100',       'genomic_expr'),
    'gene_nmf50':            ('create_nmf50',             'genomic_expr'),
    'gene_ica50':            ('create_ica50',             'genomic_expr'),
    'gene_stats':            ('create_gene_stats',        'genomic_expr'),
    'gene_quantile_rank':    ('create_quantile_rank_pca', 'genomic_expr'),
    'gene_corr_modules':     ('create_corr_modules',      'genomic_expr'),
    'gene_autoencoder50':    ('create_autoencoder50',     'genomic_expr'),
    'gene_sparse_pca50':     ('create_sparse_pca50',      'genomic_expr'),
    # Genomic needing gene_names (chromosome arm map): fn(expression, gene_names)
    'gene_chromosome_arm':   ('create_chromosome_arm',    'genomic'),
    # Foundation model embeddings: fn(expression, gene_names) -> np.ndarray (GPU)
    'gene_geneformer':       ('create_geneformer_embeddings', 'genomic'),
    'gene_geneformer_v2':    ('create_geneformer_v2_embeddings', 'genomic'),
    'gene_bulkrnabert':      ('create_bulkrnabert_embeddings', 'genomic'),
    'gene_scfoundation':     ('create_scfoundation_embeddings', 'genomic'),
    'gene_scgpt':            ('create_scgpt_embeddings',        'genomic'),
    'gene_uce':              ('create_uce_embeddings',          'genomic'),
    'gene_cancerfoundation': ('create_cancerfoundation_embeddings', 'genomic'),
    'gene_bmfm_rna':         ('create_bmfm_rna_embeddings', 'genomic'),
    # Mutation-pattern: fn(sample_ids, cohort) -> np.ndarray
    'cnv_gistic':            ('create_cnv_gistic',        'mutation'),
    'cnv_pca50':             ('create_cnv_pca50',         'mutation'),
    'mirna_top200':          ('create_mirna_top200',      'mutation'),
    # DNA methylation: fn(sample_ids, cohort) -> np.ndarray
    'methyl_pca50':          ('create_methylation_pca50', 'mutation'),
    'methyl_hvg':            ('create_methylation_hvg',   'mutation'),
    # Mutation: fn(sample_ids, cohort) -> np.ndarray
    'mutation_matrix':       ('create_mutation_matrix',   'mutation'),
    'mut_tmb':               ('create_mut_tmb',           'mutation'),
    'mut_driver':            ('create_mut_driver',        'mutation'),
    'mut_pathway':           ('create_mut_pathway',       'mutation'),
    'mut_pca50':             ('create_mut_pca50',         'mutation'),
    # Mutational signatures: fn(sample_ids, cohort) -> np.ndarray
    'sig_sbs':               ('create_sig_sbs',           'mutation'),
    'sig_dbs':               ('create_sig_dbs',           'mutation'),
    'sig_id':                ('create_sig_id',            'mutation'),
    # PanCanAtlas pre-computed: fn(sample_ids, cohort) -> np.ndarray
    'immune_cibersort':      ('create_immune_cibersort',  'mutation'),
    'immune_xcell':          ('create_immune_xcell',      'mutation'),
    'stemness_mrnasi':       ('create_stemness_mrnasi',   'mutation'),
    'purity_consensus':      ('create_purity_consensus',  'mutation'),
    'heterogeneity_math':    ('create_heterogeneity_math','mutation'),
    'msi_status':            ('create_msi_status',        'mutation'),
    'hrd_score':             ('create_hrd_score',         'mutation'),
    'cimp_status':           ('create_cimp_status',       'mutation'),
    # Clinical: fn(sample_ids, cohort) -> np.ndarray
    'clinical':              ('create_clinical',          'mutation'),
    # Tier 2 multi-omic
    'lncrna_pca50':          ('create_lncrna_pca50',     'mutation'),
    'gsva_hallmark':         ('create_gsva_hallmark',    'genomic'),
    'promoter_meth':         ('create_promoter_methylation', 'mutation'),
    'splicing_psi50':        ('create_splicing_psi50',   'mutation'),
    # RPPA: fn(sample_ids) -> np.ndarray (pan-cancer, no cohort needed)
    'rppa':                  ('create_rppa',              'rppa'),
    # CPTAC mass-spec proteomics: fn(sample_ids, cohort) -> np.ndarray
    # GRACEFUL-SKIP when local data absent (mirrors BMFM-RNA).
    'cptac_proteomics':      ('create_cptac_proteomics',  'mutation'),
    # Per-fold factory reps (handled in get_rep_factories, not dispatched here;
    # listed so --reps name resolution does not warn "Unknown representation").
    'omics_fusion_ae50':     ('make_omics_fusion_factory',     'factory'),
    'consensus_subtype':     ('make_consensus_subtype_factory','factory'),
    'vae_latent50':          ('make_vae_factory',              'factory'),
    'survival_latent50':     ('make_survival_ae_factory',      'factory'),
    # WSI — slide-level features (GigaSSL: ctranspath/phikon/hoptimus0/gigapath/moco;
    #        TITAN: HF gated; UNI2-h: manual prep;
    #        Virchow2: Paige AI 1280D; CONCH v1.5: Mahmood Lab 768D;
    #        H-optimus-1: Bioptimus 1536D)
    'wsi_ctranspath':        ('load_wsi_features',        'wsi'),
    'wsi_phikon':            ('load_wsi_features',        'wsi'),
    'wsi_hoptimus0':         ('load_wsi_features',        'wsi'),
    'wsi_gigapath':          ('load_wsi_features',        'wsi'),
    'wsi_moco':              ('load_wsi_features',        'wsi'),
    'wsi_titan':             ('load_wsi_features',        'wsi'),
    'wsi_uni2h':             ('load_wsi_features',        'wsi'),
    'wsi_virchow2':          ('load_wsi_features',        'wsi'),
    'wsi_conch':             ('load_wsi_features',        'wsi'),
    'wsi_hoptimus1':         ('load_wsi_features',        'wsi'),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(name: str):
    """Load TCGA dataset. Thin wrapper around kirby.datasets.tcga.load_dataset."""
    from kirby.datasets.tcga import load_dataset as _load
    return _load(name)


# =============================================================================
# REPRESENTATION GENERATION
# =============================================================================

def _rep_cache_path(sample_ids: List[str], rep_name: str) -> Path:
    """Get cache file path keyed by sample hash. Prefers /Volumes/seagate."""
    seq_str = '\n'.join(sample_ids)
    seq_hash = hashlib.sha256(seq_str.encode()).hexdigest()[:16]
    seagate = Path('/Volumes/seagate/kirby/reps') / f'tcga_{seq_hash}'
    if seagate.parent.parent.exists():
        seagate.mkdir(parents=True, exist_ok=True)
        return seagate / f'{rep_name}.npy'
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    cache_dir = cache_root / 'reps' / f'tcga_{seq_hash}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{rep_name}.npy'


def get_representations(
    expression: np.ndarray,
    gene_names: np.ndarray,
    sample_ids: List[str],
    wsi_dir: Optional[Path],
    preset_or_list: str,
    cohort: str = '',
    cache: bool = True,
    genomic_only: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate TCGA representations.

    Handles three rep types transparently:
    - Genomic: computed from expression matrix (PCA, pathway scores, HVG)
    - Mutation: loaded from cached MAF data
    - WSI: loaded from pre-computed GigaSSL features

    Args:
        expression: (n_samples, n_genes) expression matrix
        gene_names: (n_genes,) gene symbols
        sample_ids: TCGA barcodes
        wsi_dir: path to WSI features directory (or None)
        preset_or_list: preset name or comma-separated rep names
        cohort: TCGA cohort name (for mutation loading)
        cache: cache computed representations to disk
        genomic_only: if True, skip WSI reps even if in preset
    """
    from kirby.representations import genomic as gen

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    # Filter WSI reps if genomic_only
    if genomic_only:
        rep_names = [r for r in rep_names if not r.startswith('wsi_')]

    # Filter out factory reps — these are handled by get_rep_factories()
    base_rep_names = [r for r in rep_names if r not in FACTORY_REPS]

    print(f"\nGenerating {len(base_rep_names)} base representations "
          f"(+ {len(rep_names) - len(base_rep_names)} per-fold factory reps)...")

    representations = {}
    _failed_reps = []  # (rep, error) for REQUESTED reps that raised on load
    for rep_name in base_rep_names:
        if rep_name not in _REP_DISPATCH:
            print(f"  WARNING: Unknown representation '{rep_name}', skipping")
            continue

        # Try loading from cache
        if cache:
            cache_file = _rep_cache_path(sample_ids, rep_name)
            if cache_file.exists():
                rep = np.load(cache_file)
                # Validate cached shape matches expected n_samples
                if rep.shape[0] != expression.shape[0]:
                    print(f"  {rep_name}: cache stale ({rep.shape[0]} != {expression.shape[0]} samples), regenerating...")
                    cache_file.unlink()
                else:
                    print(f"  {rep_name}: loaded from cache ({rep.shape[1]} features)")
                    representations[rep_name] = rep
                    continue

        func_name, rep_type = _REP_DISPATCH[rep_name]

        try:
            start = time.time()
            print(f"  Generating {rep_name}...", end=' ', flush=True)

            if rep_type == 'genomic':
                func = getattr(gen, func_name)
                rep = func(expression, gene_names)

            elif rep_type == 'genomic_expr':
                func = getattr(gen, func_name)
                rep = func(expression)

            elif rep_type == 'mutation':
                func = getattr(gen, func_name)
                rep = func(sample_ids, cohort)

            elif rep_type == 'rppa':
                func = getattr(gen, func_name)
                rep = func(sample_ids)

            elif rep_type == 'wsi':
                # Extract model name from rep key (e.g., 'wsi_ctranspath' -> 'ctranspath')
                model_key = rep_name.replace('wsi_', '')
                rep = gen.load_wsi_features(sample_ids, wsi_dir, model_key)

            else:
                raise ValueError(f"Unknown rep type '{rep_type}' for {rep_name}")

            elapsed = time.time() - start

            # Validate n_samples matches expected
            expected_n = expression.shape[0]
            if rep.shape[0] != expected_n:
                raise RuntimeError(
                    f"{rep_name} has {rep.shape[0]} samples, expected {expected_n}. "
                    f"Shape: {rep.shape}"
                )

            print(f"done ({rep.shape[1]} features, {elapsed:.1f}s)")
            representations[rep_name] = rep

            if cache:
                cache_file = _rep_cache_path(sample_ids, rep_name)
                np.save(cache_file, rep)

        except Exception as e:
            print(f"FAILED: {e}")
            _failed_reps.append((rep_name, str(e)))

    # A2: dropped reps -> WARN, do not halt (user preference). A dropped rep
    # changes rep_names and may rebuild this dataset's cache.
    if _failed_reps:
        print(f"  WARNING: DROPPED {len(_failed_reps)} requested TCGA rep(s): "
              f"{[r for r, _ in _failed_reps]} — rep_names changed, may rebuild. "
              f"First error: {_failed_reps[0][1]}")

    if not representations:
        raise ValueError("No representations could be generated!")

    return representations


def get_rep_factories(
    expression: np.ndarray,
    gene_names: np.ndarray,
    preset_or_list: str,
    sample_ids: Optional[List[str]] = None,
    cohort: str = '',
    survival_time: Optional[np.ndarray] = None,
    survival_event: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Create per-fold rep factories for data-dependent TCGA representations.

    These reps require fitting on training data (z-scoring, PCA) and are
    passed as rep_factories to find_best_hybrid() to prevent leakage.

    Args:
        expression: (n_samples, n_genes) expression matrix
        gene_names: (n_genes,) gene symbols
        preset_or_list: preset name or comma-separated rep names
        sample_ids: TCGA barcodes (needed for methylation/lncRNA/splicing factories)
        cohort: TCGA cohort string (needed for methylation/lncRNA/splicing factories)

    Returns:
        Dict of {rep_name: factory_callable} for reps that need per-fold fitting
    """
    from kirby.representations.genomic import (
        make_hvg2000_factory, make_gene_pca_factory,
        make_gene_set_score_factory, make_weighted_score_factory,
        make_combined_pathway_factory,
        make_gene_immune_factory,
        make_nmf_factory, make_ica_factory, make_sparse_pca_factory,
        make_quantile_rank_pca_factory, make_corr_modules_factory,
        make_autoencoder_factory,
        make_lncrna_pca_factory, make_gsva_hallmark_factory,
        make_promoter_methylation_factory, make_splicing_psi_factory,
        make_omics_fusion_factory, make_consensus_subtype_factory,
        make_vae_factory, make_survival_ae_factory,
    )
    from kirby.representations.data import (
        hallmark, progeny_weights, kegg, dorothea_regulons,
        reactome, oncogenic,
    )

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    # Only create factories for requested reps that are in FACTORY_REPS
    requested_factories = [r for r in rep_names if r in FACTORY_REPS]
    if not requested_factories:
        return {}

    print(f"\nCreating {len(requested_factories)} per-fold rep factories...")

    # Gene set loaders for unweighted scoring
    _gene_set_loaders = {
        'gene_hallmark': hallmark,
        'gene_kegg': kegg,
        'gene_reactome': reactome,
        'gene_oncogenic': oncogenic,
    }
    # Weighted network loaders
    _weighted_loaders = {
        'gene_progeny': progeny_weights,
        'gene_dorothea': dorothea_regulons,
    }

    # Tier 2 reps that need sample_ids + cohort
    _tier2_needs_samples = {'lncrna_pca50', 'promoter_meth', 'splicing_psi50'}

    factories = {}
    for rep_name in requested_factories:
        try:
            start = time.time()
            print(f"  Setting up {rep_name} factory...", end=' ', flush=True)

            if rep_name == 'gene_hvg2000':
                factories[rep_name] = make_hvg2000_factory(expression, gene_names)
            elif rep_name == 'gene_pca100':
                factories[rep_name] = make_gene_pca_factory(expression)
            elif rep_name in _gene_set_loaders:
                gs = _gene_set_loaders[rep_name]()
                factories[rep_name] = make_gene_set_score_factory(
                    expression, gene_names, gs
                )
            elif rep_name in _weighted_loaders:
                weights = _weighted_loaders[rep_name]()
                factories[rep_name] = make_weighted_score_factory(
                    expression, gene_names, weights
                )
            elif rep_name == 'gene_combined_pathway':
                factories[rep_name] = make_combined_pathway_factory(
                    expression, gene_names
                )
            elif rep_name == 'gene_immune':
                factories[rep_name] = make_gene_immune_factory(
                    expression, gene_names
                )
            elif rep_name == 'gene_nmf50':
                factories[rep_name] = make_nmf_factory(expression)
            elif rep_name == 'gene_ica50':
                factories[rep_name] = make_ica_factory(expression)
            elif rep_name == 'gene_sparse_pca50':
                factories[rep_name] = make_sparse_pca_factory(expression)
            elif rep_name == 'gene_quantile_rank':
                factories[rep_name] = make_quantile_rank_pca_factory(expression)
            elif rep_name == 'gene_corr_modules':
                factories[rep_name] = make_corr_modules_factory(expression)
            elif rep_name == 'gene_autoencoder50':
                factories[rep_name] = make_autoencoder_factory(expression)
            # --- Multi-omics fusion / consensus / VAE / survival latents ---
            elif rep_name == 'omics_fusion_ae50':
                factories[rep_name] = make_omics_fusion_factory(
                    expression, sample_ids=sample_ids, cohort=cohort
                )
            elif rep_name == 'consensus_subtype':
                factories[rep_name] = make_consensus_subtype_factory(expression)
            elif rep_name == 'vae_latent50':
                factories[rep_name] = make_vae_factory(expression)
            elif rep_name == 'survival_latent50':
                # Graceful-skip when survival labels are absent (raises inside
                # the factory builder; caught by the surrounding try/except).
                factories[rep_name] = make_survival_ae_factory(
                    expression,
                    survival_time=survival_time,
                    survival_event=survival_event,
                )
            # --- Tier 2 factories ---
            elif rep_name == 'gsva_hallmark':
                factories[rep_name] = make_gsva_hallmark_factory(
                    expression, gene_names
                )
            elif rep_name in _tier2_needs_samples:
                if sample_ids is None:
                    print(f"WARNING: {rep_name} factory needs sample_ids — skipping")
                    continue
                if rep_name == 'lncrna_pca50':
                    factories[rep_name] = make_lncrna_pca_factory(
                        sample_ids, cohort
                    )
                elif rep_name == 'promoter_meth':
                    factories[rep_name] = make_promoter_methylation_factory(
                        sample_ids, cohort
                    )
                elif rep_name == 'splicing_psi50':
                    factories[rep_name] = make_splicing_psi_factory(
                        sample_ids, cohort
                    )
            else:
                print(f"WARNING: no factory for '{rep_name}'")
                continue

            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

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
):
    """Save results JSON with blueprint code."""
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
            'reps_available': reps_tested,
            'timestamp': datetime.now().isoformat(),
            'domain': 'tcga',
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

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def generate_explanations(
    best_result: dict,
    base_reps: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_path: str,
):
    """Generate and save feature explanations for the best TCGA configuration."""
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
        top_k=30,
    )

    print("\n" + format_explanation_summary(explanations, max_features=15))

    explain_path = output_path.replace('.json', '_explanations.json')

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

def smoke_test_new_reps() -> int:
    """Synthetic smoke test for the four new multi-omics reps.

    Proves each rep (a) imports, (b) dispatches via get_rep_factories /
    get_representations, and (c) graceful-skips cleanly on absent external data.
    Uses tiny synthetic expression — no network, no GPU required.
    """
    from kirby.representations import genomic as gen

    rng = np.random.RandomState(0)
    n, g = 40, 300
    expression = rng.randn(n, g).astype(np.float32)
    gene_names = np.array([f"GENE{i}" for i in range(g)])
    sample_ids = [f"TCGA-XX-{i:04d}-01" for i in range(n)]
    # A train/val split for exercising the per-fold factories.
    train_idx = np.arange(0, 30)
    val_idx = np.arange(30, n)
    surv_time = rng.randint(50, 3000, size=n).astype(np.float32)
    surv_event = rng.randint(0, 2, size=n).astype(np.int64)

    ok = True

    # 1. omics_fusion_ae50 — learned 50-D fusion. sample_ids omitted here so the
    #    smoke test stays offline (no RPPA/CNV download); with sample_ids the
    #    extra modalities are appended (or warn-and-skip on network miss).
    try:
        fac = gen.make_omics_fusion_factory(expression, sample_ids=None)
        tr, va = fac(train_idx, val_idx, surv_time[train_idx])
        assert tr.shape == (30, 50) and va.shape == (10, 50), (tr.shape, va.shape)
        print(f"  omics_fusion_ae50: OK train{tr.shape} val{va.shape}")
    except Exception as e:
        ok = False
        print(f"  omics_fusion_ae50: FAILED {e}")

    # 2. consensus_subtype — fixed-width one-hot + soft membership.
    try:
        fac = gen.make_consensus_subtype_factory(
            expression, n_clusters=4, n_runs=3, n_top_genes=100)
        tr, va = fac(train_idx, val_idx, surv_time[train_idx])
        assert tr.shape == (30, 8) and va.shape == (10, 8), (tr.shape, va.shape)
        print(f"  consensus_subtype: OK train{tr.shape} val{va.shape}")
    except Exception as e:
        ok = False
        print(f"  consensus_subtype: FAILED {e}")

    # 3. vae_latent50 — variational bottleneck mean.
    try:
        fac = gen.make_vae_factory(expression, epochs=2, n_top_genes=100)
        tr, va = fac(train_idx, val_idx, surv_time[train_idx])
        assert tr.shape == (30, 50) and va.shape == (10, 50), (tr.shape, va.shape)
        print(f"  vae_latent50: OK train{tr.shape} val{va.shape}")
    except Exception as e:
        ok = False
        print(f"  vae_latent50: FAILED {e}")

    # 4a. survival_latent50 WITH labels — Cox-supervised latents.
    try:
        fac = gen.make_survival_ae_factory(
            expression, survival_time=surv_time, survival_event=surv_event,
            epochs=2, n_top_genes=100)
        tr, va = fac(train_idx, val_idx, surv_time[train_idx])
        assert tr.shape == (30, 50) and va.shape == (10, 50), (tr.shape, va.shape)
        print(f"  survival_latent50 (labels): OK train{tr.shape} val{va.shape}")
    except Exception as e:
        ok = False
        print(f"  survival_latent50 (labels): FAILED {e}")

    # 4b. survival_latent50 WITHOUT labels — must graceful-skip (raise).
    try:
        gen.make_survival_ae_factory(expression, survival_time=None,
                                     survival_event=None)
        ok = False
        print("  survival_latent50 (no labels): FAILED — expected a raise")
    except RuntimeError:
        print("  survival_latent50 (no labels): OK graceful-skip (raised)")

    # 5. cptac_proteomics — must graceful-skip on absent data (clear raise).
    try:
        gen.create_cptac_proteomics(sample_ids)
        print("  cptac_proteomics: OK (local data present, loaded)")
    except (FileNotFoundError, RuntimeError) as e:
        msg = str(e)
        assert 'CPTAC' in msg, msg
        print("  cptac_proteomics: OK graceful-skip "
              f"({msg.split('—')[0].strip()[:50]}...)")
    except Exception as e:
        ok = False
        print(f"  cptac_proteomics: FAILED unexpected {type(e).__name__}: {e}")

    # 6. Dispatch wiring: all five names resolve.
    for name in ('omics_fusion_ae50', 'consensus_subtype', 'vae_latent50',
                 'survival_latent50', 'cptac_proteomics'):
        if name not in _REP_DISPATCH:
            ok = False
            print(f"  dispatch '{name}': FAILED — not in _REP_DISPATCH")
    print("  dispatch: all five names present in _REP_DISPATCH")

    print("\nSMOKE TEST: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(
        description='Find the optimal hybrid TCGA representation for each model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s tcga_luad_vs_lusc --reps fast --search fast
  %(prog)s tcga_brca_pam50 --reps fast+learned --search medium
  %(prog)s tcga_brca_survival --reps all --search long
  %(prog)s tcga_luad_vs_lusc --reps fast --genomic-only --search fast

Output:
  - Best overall: single best model+hybrid combination
  - Best by dimension: optimal at each tier (50, 100, 200, 500)
  - Best by model: optimal hybrid per model type
  - Performance matrix: full Model x Dimension grid
  - Baseline comparison: hybrid vs full-dimension baselines
        """
    )

    parser.add_argument('dataset', type=str, nargs='?',
                       help="tcga_luad_vs_lusc, tcga_brca_pam50, or tcga_brca_survival")
    parser.add_argument('--smoke-new-reps', action='store_true',
                       help='Run the synthetic smoke test for the new multi-omics '
                            'reps (omics_fusion_ae50/consensus_subtype/vae_latent50/'
                            'survival_latent50/cptac_proteomics) and exit.')
    parser.add_argument('--reps', type=str, default='fast',
                       help="Preset: 'fast'(5), 'fast+diverse'(8), 'multimodal'(12), "
                            "'fast+learned'(15), 'all'(57), 'genomic_fast'(3), "
                            "'genomic_diverse'(10), 'genomic_all'(18), or comma-separated list")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--genomic-only', action='store_true',
                       help='Skip WSI representations (genomic features only)')
    parser.add_argument('--allocation', type=str, default=None,
                       help="Override allocation methods: comma-separated from "
                            "'greedy', 'performance_weighted', 'fixed', "
                            "'dominant_greedy', 'weighted_global', or any "
                            "NAMED_ALLOCATIONS key (e.g. 'softmax_T5'). "
                            "'two_stage' and 'stability_selection' are NOT "
                            "dispatched and raise ValueError - see the "
                            "TESTED AND CUT blocks in kirby/hybrid.py.")
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
    parser.add_argument('--n-folds', type=int, default=None,
                       help='Override number of CV folds (default: 5)')
    parser.add_argument('--finetune', type=str, default=None,
                       help='Comma-separated fine-tuning methods: '
                            'geneformer, scfoundation, mock_lora')
    parser.add_argument('--finetune-epochs', type=int, default=3,
                       help='Fine-tuning epochs per fold (default: 3)')
    parser.add_argument('--no-ft-cache', action='store_true',
                       help='Disable fine-tuning cache')
    parser.add_argument('--tune', action='store_true',
                       help='Tune model HPs via RandomizedSearchCV before hybrid search')
    parser.add_argument('--tune-iter', type=int, default=20,
                       help='RandomizedSearchCV iterations per model (default: 20)')
    parser.add_argument('--explain', action='store_true',
                       help='Generate feature explanations for best hybrid config')

    args = parser.parse_args()

    if args.smoke_new_reps:
        return smoke_test_new_reps()

    if not args.dataset:
        parser.error("dataset is required unless --smoke-new-reps is given")


    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    expression, labels, task_type, metadata = load_dataset(args.dataset)
    sample_ids = metadata['sample_ids']
    gene_names = metadata['gene_names']
    wsi_dir = metadata.get('wsi_dir')
    cohort = metadata.get('cohort', '')
    class_names = metadata.get('class_names')

    print(f"  {len(sample_ids)} samples, task: {task_type}")
    if class_names:
        print(f"  Classes: {class_names}")

    # TSS grouping
    print("  Assigning TSS groups...")
    groups, n_groups = assign_tss_groups(sample_ids)
    print(f"  {n_groups} TSS sites")

    # Generate representations
    base_reps = get_representations(
        expression, gene_names, sample_ids, wsi_dir,
        args.reps, cohort=cohort, genomic_only=args.genomic_only,
    )

    # Create per-fold factories for data-dependent reps (prevents leakage)
    leakage_free_factories = get_rep_factories(
        expression, gene_names, args.reps,
        sample_ids=sample_ids, cohort=cohort,
        survival_time=metadata.get('survival_time'),
        survival_event=metadata.get('survival_event'),
    )

    # Resolve model override
    models_arg = None
    if args.model:
        models_arg = [m.strip() for m in args.model.split(',')]

    alloc_arg = None
    if args.allocation:
        alloc_arg = [a.strip() for a in args.allocation.split(',')]

    # Select metric
    if task_type == 'classification':
        n_classes = len(np.unique(labels))
        if n_classes == 2:
            metric = 'auc'
            secondary_metrics = ['accuracy', 'f1_weighted']
        else:
            metric = 'f1_weighted'
            secondary_metrics = ['accuracy']
    else:
        metric = 'r2'
        secondary_metrics = ['rmse', 'spearman']

    # Build fine-tuning factories
    rep_factories = None
    if args.finetune:
        ft_methods = [m.strip() for m in args.finetune.split(',')]
        rep_factories = {}
        ft_cache = not args.no_ft_cache

        for method in ft_methods:
            if method == 'mock_lora':
                from kirby.finetuning._base import mock_factory
                rep_factories['mock_lora'] = mock_factory(n_features=64, seed=args.seed)
            elif method == 'geneformer':
                from kirby.finetuning.tcga_geneformer import make_geneformer_factory
                rep_factories['geneformer'] = make_geneformer_factory(
                    expression_data=expression,
                    gene_names=gene_names,
                    sample_ids=sample_ids,
                    dataset_name=args.dataset,
                    task_type=task_type,
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            elif method == 'scfoundation':
                from kirby.finetuning.tcga_scfoundation import make_scfoundation_factory
                rep_factories['scfoundation'] = make_scfoundation_factory(
                    expression_data=expression,
                    gene_names=gene_names,
                    sample_ids=sample_ids,
                    dataset_name=args.dataset,
                    task_type=task_type,
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

    # Hyperparameter tuning
    if args.tune:
        from kirby.find_best_hybrid import tune_models
        print("\n=== Hyperparameter Tuning ===")
        tuned_models = tune_models(
            base_reps=base_reps, labels=labels, groups=groups,
            task_type=task_type, n_iter=args.tune_iter,
            seed=args.seed,
        )
        models_arg = tuned_models

    # Call domain-agnostic search
    result = find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        models=models_arg,
        metric=metric,
        secondary_metrics=secondary_metrics,
        task_type=task_type,
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
        output_path = f"results_{args.dataset}{tune_suffix}.json"

    save_results(result, args.dataset, 'full', list(base_reps.keys()), output_path,
                 base_reps=base_reps, labels=labels)

    # Explanations (iterate over models)
    if args.explain and result.best_by_model:
        for model_name, best_result in result.best_by_model.items():
            print(f"\n--- Explanations for model: {model_name} ---")
            generate_explanations(best_result, base_reps, labels,
                                  output_path.replace('.json', f'_{model_name}.json'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
