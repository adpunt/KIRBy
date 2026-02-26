#!/usr/bin/env python3
"""
Find Best Hybrid Antibody Representation

Thin antibody wrapper around kirby.find_best_hybrid. Handles sequence loading,
antibody representation generation, sequence identity grouping, and output
formatting. The actual search logic lives in the domain-agnostic library.

Datasets (all regression):
  - sabdab_affinity: SKEMPI 2.0 ΔΔG (binding affinity changes)
  - sabdab_structure: SAbDab resolution (X-ray structure quality)
  - sabdab_humanness: germline identity score

Usage:
    # Quick local test
    python find_best_hybrid_antibody.py sabdab_affinity --reps fast --search fast

    # Medium search with language models
    python find_best_hybrid_antibody.py sabdab_affinity --reps fast+lm --search medium

    # Full search on server
    python find_best_hybrid_antibody.py data.csv --heavy-col VH --light-col VL --target-col ddG \\
        --reps all_server --search long
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import hashlib
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Block stale system bottleneck (Anaconda 2022.05, compiled against old numpy)
sys.modules.setdefault('bottleneck', None)

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kirby.find_best_hybrid import find_best_hybrid, SearchResult, Blueprint

warnings.filterwarnings('ignore')


# =============================================================================
# SEQUENCE IDENTITY GROUPING
# =============================================================================

def assign_identity_groups(heavy_sequences: List[str], threshold: float = 0.8,
                           min_groups: int = 0) -> Tuple[Optional[np.ndarray], int]:
    """Group antibodies by VH sequence identity for GroupKFold.

    Simple greedy clustering:
    1. For each sequence, compute normalized Levenshtein similarity to cluster centers
    2. If >= threshold similarity to any center, assign to that cluster
    3. Otherwise, start a new cluster

    If min_groups > 0 and initial clustering yields fewer groups, progressively
    tightens the threshold (0.9, 0.95, 0.98, 0.99). If still insufficient
    (e.g., truly identical intra-family data), returns (None, 0) to signal
    that GroupKFold is not appropriate and callers should use regular KFold.

    Args:
        heavy_sequences: List of heavy chain amino acid sequences.
        threshold: Minimum fractional identity to join a cluster (0-1). Default 0.8.
        min_groups: Minimum number of groups required (e.g., n_folds for CV).
                    If 0, no minimum is enforced.

    Returns:
        (groups, n_groups): group label array and number of unique groups.
        If min_groups > 0 and cannot be satisfied, returns (None, 0).
    """
    def _identity(s1: str, s2: str) -> float:
        """Fractional identity = matches / max(len1, len2)."""
        if not s1 or not s2:
            return 0.0
        n = max(len(s1), len(s2))
        matches = sum(a == b for a, b in zip(s1, s2))
        return matches / n

    def _cluster(seqs, thresh):
        centers = []
        groups = np.zeros(len(seqs), dtype=int)
        for i, seq in enumerate(seqs):
            assigned = False
            for center_idx, (_, center_seq) in enumerate(centers):
                if _identity(seq, center_seq) >= thresh:
                    groups[i] = center_idx
                    assigned = True
                    break
            if not assigned:
                groups[i] = len(centers)
                centers.append((i, seq))
        return groups, len(centers)

    groups, n_groups = _cluster(heavy_sequences, threshold)

    if min_groups > 0 and n_groups < min_groups:
        for fallback_thresh in [0.9, 0.95, 0.98, 0.99]:
            if fallback_thresh <= threshold:
                continue
            groups, n_groups = _cluster(heavy_sequences, fallback_thresh)
            if n_groups >= min_groups:
                print(f"  Identity grouping: raised threshold {threshold}->{fallback_thresh} "
                      f"to get {n_groups} groups (needed {min_groups})")
                return groups, n_groups
        # Still not enough — grouping not meaningful for this data
        print(f"  Identity grouping: only {n_groups} groups even at threshold=0.99 "
              f"(need {min_groups}). Falling back to ungrouped CV.")
        return None, 0

    return groups, n_groups


# =============================================================================
# CONFIGURATION
# =============================================================================

REP_PRESETS = {
    'fast': ['kmer', 'physicochemical', 'aac', 'blosum62'],
    'fast+lm': ['kmer', 'physicochemical', 'aac', 'blosum62',
                'antiberty', 'ablang2', 'igbert'],
    'all_server': [
        'kmer', 'physicochemical', 'aac', 'blosum62',
        'antiberty', 'ablang2', 'igbert', 'esm2', 'prot_t5',
        'ab_roberta', 'balm', 'pigen', 'structural', 'saprot',
    ],
    'all': [
        'kmer', 'physicochemical', 'aac', 'blosum62',
        'antiberty', 'ablang2', 'igbert', 'esm2', 'prot_t5',
        'ab_roberta', 'balm', 'pigen', 'esmc', 'sapiens', 'structural', 'saprot',
    ],
    # Mutation-aware presets (for intra-family FLAb datasets)
    # Mutation-specific features (positional, biophysical_delta, lm_score)
    # are now augmentations, not base reps — see get_mutation_augmentations()
    'mutation_fast': ['kmer', 'aac', 'blosum62', 'physicochemical',
                      'esm2', 'prot_t5', 'varpos_physchem'],
    'mutation_full': ['kmer', 'aac', 'blosum62', 'physicochemical',
                      'antiberty', 'ablang2', 'esm2', 'prot_t5',
                      'varpos_onehot', 'varpos_physchem'],
    # Mutation + structural (requires ImmuneBuilder on server)
    'mutation_structural': ['kmer', 'aac', 'blosum62', 'physicochemical',
                            'antiberty', 'ablang2', 'esm2', 'prot_t5', 'structural'],

    # --- DMS/intra-family: mutation-specific features as BASE reps ---
    # Full-sequence LM embeddings are near-useless for DMS (published ρ = 0.08).
    # Instead use per-model LM scores, positional, and biophysical delta.
    # Variable-position encodings capture what ACTUALLY changes across samples.
    # Region-filtered variants (cdr/cdr_h3/framework) use Chothia boundaries:
    #   varpos_cdr_*     = CDR positions only (H1+H2+H3, L1+L2+L3)
    #   varpos_cdrh3_*   = CDR-H3 only (most variable, binding-critical, lowest dim)
    #   varpos_fw_*      = framework only (stability-relevant)
    # USE 'dms' for intra-family DMS datasets; 'inter_family' for diverse datasets.
    'dms': [
        'lm_score_ablang2', 'lm_score_esm2',
        'positional', 'biophysical_delta',
        'aac', 'blosum62',
        'varpos_physchem',                              # all positions, 6D/pos
        'varpos_cdr_physchem', 'varpos_cdrh3_physchem', # CDR-focused, much lower dim
        'antifold_score',                               # structure-conditioned, 5D
        'sapiens_score',                                # humanness-conditioned, 5D
        'esm1v_score',                                  # ESM-1v variant-effect, 5D
    ],  # 3 base + 9 aug (small reps auto-moved to augmentations)
    'dms_full': [
        'lm_score_ablang2', 'lm_score_esm2', 'lm_score_prot_t5', 'lm_score_antiberty',
        'positional', 'biophysical_delta',
        'esm2_site',
        'aac', 'blosum62', 'developability',
        'varpos_onehot', 'varpos_blosum', 'varpos_physchem',      # all positions
        'varpos_cdr_onehot', 'varpos_cdr_physchem',               # CDR only
        'varpos_cdrh3_physchem', 'varpos_fw_physchem',            # H3 / framework
        'antifold_score',                                          # structure-conditioned, 5D
        'sapiens_score',                                           # humanness-conditioned, 5D
        'esm1v_score',                                             # ESM-1v variant-effect, 5D
    ],  # 8 base + 12 aug (small reps auto-moved to augmentations)

    # --- Inter-family: full-sequence embeddings ARE appropriate ---
    'inter_family': [
        'ablang2', 'esm2', 'prot_t5',
        'physicochemical', 'aac', 'developability',
    ],  # 6 reps
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

# Small-dim reps (<= 20D) that waste search budget as base reps in DMS presets.
# Moved to augmentations: appended after hybrid is built, kept only if they help.
_DMS_AUG_REPS = {
    'dms': {
        'lm_score_ablang2', 'lm_score_esm2',
        'positional', 'biophysical_delta',
        'aac', 'blosum62', 'antifold_score',
        'sapiens_score', 'esm1v_score',
    },
    'dms_full': {
        'lm_score_ablang2', 'lm_score_esm2', 'lm_score_prot_t5', 'lm_score_antiberty',
        'positional', 'biophysical_delta',
        'aac', 'blosum62', 'developability', 'antifold_score',
        'sapiens_score', 'esm1v_score',
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _download_skempi() -> pd.DataFrame:
    """Download or load cached SKEMPI 2.0 data."""
    import requests

    cache_dir = Path.home() / '.cache' / 'kirby'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / 'skempi_v2.csv'

    if not cached.exists():
        url = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
        print(f"  Downloading SKEMPI 2.0 from {url}...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        cached.write_bytes(resp.content)

    return pd.read_csv(cached, sep=';')


def load_dataset(name_or_path: str, heavy_col: str = 'VH',
                 light_col: str = 'VL', target_col: str = 'target'
                 ) -> Tuple[List[str], Optional[List[str]], np.ndarray, str, Dict]:
    """Load antibody dataset.

    Built-in:
      - sabdab_affinity: SKEMPI 2.0 ΔΔG (regression)
      - flab_*: FLAb benchmark datasets (intra/inter-family)
      - abcov_*: Ab-CoV therapeutic datasets (inter-family)

    Custom: CSV with heavy_seq, light_seq (optional), target columns.

    Returns:
        (heavy_seqs, light_seqs_or_None, labels, task_type, metadata)
        metadata keys: wildtype_heavy, wildtype_light, mutations, is_intra_family
    """
    name_lower = name_or_path.lower()

    # FLAb datasets
    if name_lower.startswith('flab_'):
        from kirby.datasets.flab import load_flab_dataset, FLAB_DATASETS
        data = load_flab_dataset(name_lower)
        metadata = {
            'wildtype_heavy': data['wildtype_heavy'],
            'wildtype_light': data['wildtype_light'],
            'mutations': data['mutations'],
            'is_intra_family': data['is_intra_family'],
        }
        return (data['heavy_seqs'], data['light_seqs'], data['labels'],
                data['task_type'], metadata)

    # Ab-CoV datasets
    if name_lower.startswith('abcov_'):
        from kirby.datasets.abcov import load_abcov_dataset
        data = load_abcov_dataset(name_lower)
        metadata = {
            'wildtype_heavy': None,
            'wildtype_light': None,
            'mutations': data['mutations'],
            'is_intra_family': False,
        }
        return (data['heavy_seqs'], data['light_seqs'], data['labels'],
                data['task_type'], metadata)

    # AbAgym datasets
    if name_lower.startswith('abagym_'):
        from kirby.datasets.abagym import load_abagym_dataset
        data = load_abagym_dataset(name_or_path)
        metadata = {
            'wildtype_heavy': data['wildtype_heavy'],
            'wildtype_light': data['wildtype_light'],
            'mutations': data['mutations'],
            'is_intra_family': data['is_intra_family'],
        }
        return (data['heavy_seqs'], data['light_seqs'], data['labels'],
                data['task_type'], metadata)

    # AB-Bind single-point mutations only
    if name_lower == 'abbind_spm':
        from kirby.datasets.abbind import load_abbind_dataset
        data = load_abbind_dataset(single_point_only=True)
        metadata = {
            'wildtype_heavy': data['wildtype_heavy'],
            'wildtype_light': data['wildtype_light'],
            'mutations': data['mutations'],
            'is_intra_family': data['is_intra_family'],
        }
        return (data['heavy_seqs'], data['light_seqs'], data['labels'],
                data['task_type'], metadata)

    # AbBind dataset (all mutations)
    if name_lower == 'abbind':
        from kirby.datasets.abbind import load_abbind_dataset
        data = load_abbind_dataset()
        metadata = {
            'wildtype_heavy': data['wildtype_heavy'],
            'wildtype_light': data['wildtype_light'],
            'mutations': data['mutations'],
            'is_intra_family': data['is_intra_family'],
        }
        return (data['heavy_seqs'], data['light_seqs'], data['labels'],
                data['task_type'], metadata)

    # Combined ΔΔG dataset (SKEMPI 2.0 + AB-Bind, deduplicated)
    if name_lower == 'ddg_combined':
        from kirby.datasets.ddg_combined import load_ddg_combined_dataset
        data = load_ddg_combined_dataset()
        metadata = {
            'wildtype_heavy': data['wildtype_heavy'],
            'wildtype_light': data['wildtype_light'],
            'mutations': data['mutations'],
            'is_intra_family': data['is_intra_family'],
        }
        return (data['heavy_seqs'], data['light_seqs'], data['labels'],
                data['task_type'], metadata)

    if name_lower == 'sabdab_affinity':
        df = _download_skempi()

        # SKEMPI has Protein 1/2 sequences and ΔΔG
        # Extract antibody VH sequences from the Mutation column entries
        # Use a simplified approach: get unique Ab-Ag pairs
        if 'Affinity_mut_parsed' in df.columns:
            target = 'Affinity_mut_parsed'
        elif 'DDG' in df.columns:
            target = 'DDG'
        else:
            # Try to find ΔΔG column
            ddg_cols = [c for c in df.columns if 'ddg' in c.lower() or 'DDG' in c]
            if not ddg_cols:
                raise ValueError(f"Cannot find ΔΔG column. Available: {list(df.columns)}")
            target = ddg_cols[0]

        # Get sequence columns
        seq_cols = [c for c in df.columns if 'Sequence' in c or 'sequence' in c]
        if len(seq_cols) >= 2:
            heavy_seqs_raw = df[seq_cols[0]].tolist()
            light_seqs_raw = df[seq_cols[1]].tolist() if len(seq_cols) > 1 else None
        else:
            # Fall back: use Protein column
            prot_cols = [c for c in df.columns if 'Protein' in c]
            heavy_seqs_raw = df[prot_cols[0]].tolist() if prot_cols else None
            light_seqs_raw = None

        if heavy_seqs_raw is None:
            raise ValueError(f"Cannot find sequence columns. Available: {list(df.columns)}")

        labels_raw = pd.to_numeric(df[target], errors='coerce')

        # Filter valid entries
        valid = labels_raw.notna()
        if heavy_seqs_raw:
            valid &= pd.Series(heavy_seqs_raw).notna() & (pd.Series(heavy_seqs_raw).str.len() > 10)

        heavy_seqs = [s for s, v in zip(heavy_seqs_raw, valid) if v]
        light_seqs = None
        if light_seqs_raw:
            light_seqs = [s for s, v in zip(light_seqs_raw, valid) if v]
            # Remove entries with missing light chains
            has_light = [isinstance(s, str) and len(s) > 10 for s in light_seqs]
            if not all(has_light):
                light_seqs = None  # Fall back to heavy-only

        labels = labels_raw[valid].values.astype(np.float64)

        # Deduplicate by heavy sequence: take median
        dedup = pd.DataFrame({'heavy': heavy_seqs, 'label': labels})
        if light_seqs:
            dedup['light'] = light_seqs
            dedup = dedup.groupby(['heavy', 'light']).agg({'label': 'median'}).reset_index()
            heavy_seqs = dedup['heavy'].tolist()
            light_seqs = dedup['light'].tolist()
        else:
            dedup = dedup.groupby('heavy').agg({'label': 'median'}).reset_index()
            heavy_seqs = dedup['heavy'].tolist()

        labels = dedup['label'].values
        print(f"  Loaded {len(heavy_seqs)} antibodies from SKEMPI 2.0 (ΔΔG)")
        metadata = {'wildtype_heavy': None, 'wildtype_light': None,
                     'mutations': [[] for _ in heavy_seqs], 'is_intra_family': False}
        return heavy_seqs, light_seqs, labels, 'regression', metadata

    elif name_or_path.endswith('.csv') or os.path.exists(name_or_path):
        print(f"Loading custom dataset from {name_or_path}...")
        df = pd.read_csv(name_or_path)

        # Case-insensitive column matching
        col_map = {c.lower(): c for c in df.columns}
        h_col = col_map.get(heavy_col.lower())
        l_col = col_map.get(light_col.lower())
        t_col = col_map.get(target_col.lower())

        if h_col is None:
            raise ValueError(f"Heavy chain column '{heavy_col}' not found. Available: {list(df.columns)}")
        if t_col is None:
            raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

        valid = df[h_col].notna() & df[t_col].notna()
        heavy_seqs = df.loc[valid, h_col].astype(str).tolist()
        labels = pd.to_numeric(df.loc[valid, t_col], errors='coerce').values

        light_seqs = None
        if l_col is not None:
            light_valid = df.loc[valid, l_col].notna()
            if light_valid.sum() > len(heavy_seqs) * 0.5:
                light_seqs = df.loc[valid, l_col].astype(str).tolist()
                light_seqs = [s if isinstance(s, str) and len(s) > 5 else '' for s in light_seqs]

        # Filter NaN labels
        finite = np.isfinite(labels)
        heavy_seqs = [s for s, v in zip(heavy_seqs, finite) if v]
        if light_seqs:
            light_seqs = [s for s, v in zip(light_seqs, finite) if v]
        labels = labels[finite]

        # Detect task type
        unique = np.unique(labels)
        if len(unique) <= 20 and np.allclose(labels, labels.astype(int)):
            task_type = 'classification'
        else:
            task_type = 'regression'

        print(f"  Loaded {len(heavy_seqs)} antibodies, task: {task_type}")
        metadata = {'wildtype_heavy': None, 'wildtype_light': None,
                     'mutations': [[] for _ in heavy_seqs], 'is_intra_family': False}
        return heavy_seqs, light_seqs, labels, task_type, metadata

    else:
        raise ValueError(
            f"Unknown dataset: {name_or_path}. "
            f"Built-in options: sabdab_affinity, flab_*, abcov_*, abagym_*, "
            f"abbind, abbind_spm, ddg_combined. Or provide a CSV path."
        )


# =============================================================================
# REPRESENTATION GENERATION
# =============================================================================

def _rep_cache_path(heavy_seqs: List[str], light_seqs: Optional[List[str]],
                    rep_name: str) -> Path:
    """Get cache file path keyed by sequence hash."""
    seq_str = '\n'.join(heavy_seqs)
    if light_seqs:
        seq_str += '\n---\n' + '\n'.join(light_seqs)
    seq_hash = hashlib.sha256(seq_str.encode()).hexdigest()[:16]
    cache_dir = Path.home() / '.cache' / 'kirby' / 'reps' / seq_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{rep_name}.npy'


# Dispatch: (function, input_mode)
# input_mode: 'single' = fn(sequences), 'paired' = fn(heavy, light),
#             'single_chain' = fn(sequences) called per-chain then concat
_REP_DISPATCH = {
    # Classical: concat sequences, call fn(sequences)
    'kmer': ('create_kmer_frequency', 'concat'),
    'aac': ('compute_aac_features', 'concat'),
    'blosum62': ('compute_blosum62_features', 'concat'),
    # Paired native: fn(heavy, light)
    'physicochemical': ('create_physicochemical_profile', 'paired'),
    'ablang2': ('create_ablang2_embeddings', 'paired'),
    'igbert': ('create_igbert_embeddings', 'paired'),
    'esm2': ('create_esm2_embeddings', 'paired'),
    'prot_t5': ('create_prot_t5_embeddings', 'paired'),
    'pigen': ('create_pigen_embeddings', 'paired'),
    'esmc': ('create_esmc_embeddings', 'paired'),
    'imgt_onehot': ('create_imgt_onehot', 'paired'),
    # Single-chain: fn(sequences) called on heavy + light, concat results
    'antiberty': ('create_antiberty_embeddings', 'single_chain'),
    'ab_roberta': ('create_ab_roberta_embeddings', 'single_chain'),
    'balm': ('create_balm_embeddings', 'single_chain'),
    'sapiens': ('create_sapiens_embeddings', 'single_chain'),
    # Developability: compute_developability_features(heavy, light)
    'developability': ('compute_developability_features', 'paired'),
    # Mutation-aware: require mutations/wildtype metadata
    'positional': ('compute_mutation_positional_features', 'mutation'),
    'biophysical_delta': ('compute_biophysical_delta', 'mutation'),
    'lm_score': ('compute_lm_mutation_scores', 'mutation'),
    # Per-model LM mutation scores (each produces 5D)
    'lm_score_ablang2': ('compute_lm_mutation_scores', 'mutation_lm'),
    'lm_score_esm2': ('compute_lm_mutation_scores', 'mutation_lm'),
    'lm_score_prot_t5': ('compute_lm_mutation_scores', 'mutation_lm'),
    'lm_score_antiberty': ('compute_lm_mutation_scores', 'mutation_lm'),
    # ESM-2 per-residue at mutation site (1280D)
    'esm2_site': ('create_esm2_site_embedding', 'mutation_site'),
    # Variable-position encodings (for intra-family DMS datasets)
    # Full-sequence variants
    'varpos_onehot': ('create_variable_position_encoding', 'varpos'),
    'varpos_blosum': ('create_variable_position_encoding', 'varpos'),
    'varpos_physchem': ('create_variable_position_encoding', 'varpos'),
    # CDR-only variants (lower-dim, focused on binding-relevant regions)
    'varpos_cdr_onehot': ('create_variable_position_encoding', 'varpos'),
    'varpos_cdr_physchem': ('create_variable_position_encoding', 'varpos'),
    # CDR-H3-only (most variable region, critical for binding)
    'varpos_cdrh3_physchem': ('create_variable_position_encoding', 'varpos'),
    # Framework-only (stability-relevant mutations outside CDRs)
    'varpos_fw_physchem': ('create_variable_position_encoding', 'varpos'),
    # Structural: computed separately (not in antibody.py)
    'structural': (None, 'structural'),
    # AntiFold inverse folding score (5D, structure-conditioned)
    'antifold_score': ('compute_antifold_scores', 'antifold'),
    # Sapiens humanness score (5D, humanness-conditioned)
    'sapiens_score': ('compute_sapiens_mutation_scores', 'mutation'),
    # ESM-1v masked marginal mutation scores (5D, variant-effect)
    'esm1v_score': ('compute_esm1v_mutation_scores', 'mutation'),
    # SaProt structure-aware embeddings (1280D paired)
    'saprot': ('create_saprot_embeddings', 'paired'),
}


def get_representations(heavy_seqs: List[str], light_seqs: Optional[List[str]],
                        preset_or_list: str, cache: bool = True,
                        wt_heavy: Optional[str] = None,
                        wt_light: Optional[str] = None,
                        mutations: Optional[List] = None,
                        delta: bool = False) -> Dict[str, np.ndarray]:
    """Generate antibody representations.

    Handles paired vs single-chain transparently:
    - Paired models (ablang2, igbert, pigen, esm2, prot_t5): pass both chains
    - Single-chain models (antiberty, ab_roberta, balm, sapiens):
      encode heavy + light separately, concatenate
    - Classical (kmer, aac, blosum62): concatenate sequences, then encode
    - Paired-native (physicochemical): pass both chains
    - Mutation-aware (positional, biophysical_delta, lm_score): require metadata

    When delta=True, generates delta reps (ΔX = X_mutant - X_wildtype) for
    every base rep, doubling the rep count. Requires wt_heavy.
    """
    from kirby.representations import antibody as ab

    # Check if mutation data is actually present (not just a list of empty lists)
    has_mutations = (mutations is not None and any(len(m) > 0 for m in mutations))

    if preset_or_list in REP_PRESETS:
        rep_names = REP_PRESETS[preset_or_list]
    else:
        rep_names = [r.strip().lower() for r in preset_or_list.split(',')]

    print(f"\nGenerating {len(rep_names)} representations...")

    representations = {}
    for rep_name in rep_names:
        if rep_name not in _REP_DISPATCH:
            print(f"  WARNING: Unknown representation '{rep_name}', skipping")
            continue

        # Try loading from cache
        if cache:
            cache_file = _rep_cache_path(heavy_seqs, light_seqs, rep_name)
            if cache_file.exists():
                rep = np.load(cache_file)
                print(f"  {rep_name}: loaded from cache ({rep.shape[1]} features)")
                representations[rep_name] = rep
                continue

        func_name, mode = _REP_DISPATCH[rep_name]

        try:
            start = time.time()
            print(f"  Generating {rep_name}...", end=' ', flush=True)

            if mode == 'structural':
                from kirby.representations.structural import compute_structural_features
                rep = compute_structural_features(heavy_seqs, light_seqs)

            elif mode == 'antifold':
                from kirby.representations.structural import compute_antifold_scores
                if not has_mutations:
                    print(f"SKIPPED (no mutation data)")
                    continue
                rep = compute_antifold_scores(
                    wt_heavy or '', heavy_seqs, mutations,
                    wt_light=wt_light,
                )

            elif mode == 'concat':
                func = getattr(ab, func_name)
                if light_seqs:
                    concat_seqs = [h + l for h, l in zip(heavy_seqs, light_seqs)]
                else:
                    concat_seqs = heavy_seqs
                rep = func(concat_seqs)

            elif mode == 'paired':
                func = getattr(ab, func_name)
                if light_seqs:
                    rep = func(heavy_seqs, light_seqs)
                else:
                    rep = func(heavy_seqs)

            elif mode == 'single_chain':
                func = getattr(ab, func_name)
                rep_h = func(heavy_seqs)
                if light_seqs:
                    rep_l = func(light_seqs)
                    rep = np.hstack([rep_h, rep_l])
                else:
                    rep = rep_h

            elif mode == 'mutation':
                func = getattr(ab, func_name)
                if not has_mutations:
                    print(f"SKIPPED (no mutation data)")
                    continue
                if rep_name == 'positional':
                    rep = func(mutations, wt_heavy or '', wt_light)
                elif rep_name == 'biophysical_delta':
                    rep = func(mutations)
                elif rep_name == 'lm_score':
                    rep = func(wt_heavy or '', heavy_seqs, mutations)
                elif rep_name == 'sapiens_score':
                    rep = func(wt_heavy or '', heavy_seqs, mutations)
                elif rep_name == 'esm1v_score':
                    rep = func(wt_heavy or '', heavy_seqs, mutations)
                else:
                    raise ValueError(f"Unknown mutation rep: {rep_name}")

            elif mode == 'mutation_lm':
                func = getattr(ab, func_name)
                if not has_mutations:
                    print(f"SKIPPED (no mutation data)")
                    continue
                # Extract model_name from rep_name: lm_score_ablang2 -> ablang2
                lm_model = rep_name.replace('lm_score_', '')
                rep = func(wt_heavy or '', heavy_seqs, mutations, model_name=lm_model)

            elif mode == 'mutation_site':
                func = getattr(ab, func_name)
                if not has_mutations or not wt_heavy:
                    print(f"SKIPPED (no mutation/wildtype data)")
                    continue
                rep = func(wt_heavy, heavy_seqs, mutations)

            elif mode == 'varpos':
                func = getattr(ab, func_name)
                # Parse encoding and region from rep_name
                # e.g. varpos_physchem -> enc=physchem, region=all
                #      varpos_cdr_physchem -> enc=physchem, region=cdr
                #      varpos_cdrh3_physchem -> enc=physchem, region=cdr_h3
                #      varpos_fw_physchem -> enc=physchem, region=framework
                suffix = rep_name.replace('varpos_', '')
                region = 'all'
                if suffix.startswith('cdrh3_'):
                    region = 'cdr_h3'
                    suffix = suffix.replace('cdrh3_', '')
                elif suffix.startswith('cdr_'):
                    region = 'cdr'
                    suffix = suffix.replace('cdr_', '')
                elif suffix.startswith('fw_'):
                    region = 'framework'
                    suffix = suffix.replace('fw_', '')
                enc = suffix
                kwargs = dict(encoding=enc, region=region)
                if light_seqs:
                    rep = func(heavy_seqs, light_seqs, **kwargs)
                else:
                    rep = func(heavy_seqs, **kwargs)

            else:
                raise ValueError(f"Unknown mode '{mode}' for {rep_name}")

            elapsed = time.time() - start
            print(f"done ({rep.shape[1]} features, {elapsed:.1f}s)")
            representations[rep_name] = rep

            if cache:
                cache_file = _rep_cache_path(heavy_seqs, light_seqs, rep_name)
                np.save(cache_file, rep)

        except Exception as e:
            print(f"FAILED: {e}")

    if not representations:
        raise ValueError("No representations could be generated!")

    # --- Delta representations ---
    if delta and wt_heavy:
        print(f"\n  Generating delta representations (ΔX = X_mutant - X_wildtype)...")
        for rep_name, (func_name, mode) in _REP_DISPATCH.items():
            if rep_name not in representations:
                continue
            if mode in ('mutation', 'mutation_lm', 'mutation_site', 'varpos'):
                continue  # mutation/varpos reps are already "delta-like"
            if mode in ('structural', 'antifold'):
                continue  # structure-based reps already encode per-sample differences

            cache_key = f"delta_{rep_name}"
            if cache:
                cache_file = _rep_cache_path(heavy_seqs, light_seqs, cache_key)
                if cache_file.exists():
                    rep = np.load(cache_file)
                    print(f"  delta_{rep_name}: loaded from cache ({rep.shape[1]} features)")
                    representations[cache_key] = rep
                    continue

            try:
                func = getattr(ab, func_name)
                if mode == 'concat':
                    wt_seq = wt_heavy + (wt_light or '')
                    wt_rep = func([wt_seq])
                elif mode == 'paired':
                    if wt_light:
                        wt_rep = func([wt_heavy], [wt_light])
                    else:
                        wt_rep = func([wt_heavy])
                elif mode == 'single_chain':
                    wt_rep = func([wt_heavy])
                    if wt_light:
                        wt_rep_l = func([wt_light])
                        wt_rep = np.hstack([wt_rep, wt_rep_l])
                else:
                    continue

                # ΔX = X_mutant - X_wildtype (broadcast: wt_rep is 1×d)
                delta_rep = representations[rep_name] - wt_rep
                representations[cache_key] = delta_rep
                print(f"  delta_{rep_name}: ({delta_rep.shape[1]} features)")

                if cache:
                    cache_file = _rep_cache_path(heavy_seqs, light_seqs, cache_key)
                    np.save(cache_file, delta_rep)

            except Exception as e:
                print(f"  delta_{rep_name}: FAILED ({e})")

    return representations


def get_augmentations(heavy_seqs: List[str],
                      light_seqs: Optional[List[str]]) -> Dict[str, np.ndarray]:
    """Generate augmented features (94D total)."""
    from kirby.representations.antibody import (
        compute_developability_features,
        compute_humanness_scores,
        compute_cdr_length_features,
        compute_per_cdr_physicochemical,
        compute_extended_liability_features,
        compute_camsol_features,
    )

    print("\nGenerating augmented features...")
    augmentations = {}

    aug_funcs = [
        ('developability', lambda: compute_developability_features(heavy_seqs, light_seqs)),
        ('humanness', lambda: compute_humanness_scores(heavy_seqs)),
        ('cdr_lengths', lambda: compute_cdr_length_features(heavy_seqs, light_seqs)),
        ('per_cdr_physchem', lambda: compute_per_cdr_physicochemical(heavy_seqs, light_seqs)),
        ('extended_liability', lambda: compute_extended_liability_features(heavy_seqs, light_seqs)),
        ('camsol', lambda: compute_camsol_features(heavy_seqs)),
    ]

    for name, func in aug_funcs:
        try:
            start = time.time()
            print(f"  Generating {name}...", end=' ', flush=True)
            feat = func()
            elapsed = time.time() - start
            print(f"done ({feat.shape[1]} features, {elapsed:.1f}s)")
            augmentations[name] = feat
        except Exception as e:
            print(f"FAILED: {e}")

    return augmentations


def get_mutation_augmentations(
    mutations: List,
    wt_heavy: Optional[str],
    wt_light: Optional[str],
    heavy_seqs: List[str],
) -> Dict[str, np.ndarray]:
    """Generate mutation-specific augmentation features (up to 22D).

    These describe what changed (biophysical properties, CDR position, LM score)
    and are always included for mutation datasets. They complement the base
    sequence-level representations.
    """
    from kirby.representations.antibody import (
        compute_biophysical_delta,
        compute_mutation_positional_features,
        compute_lm_mutation_scores,
    )

    print("\nGenerating mutation augmentations...")
    augmentations = {}

    mut_funcs = [
        ('biophysical_delta', lambda: compute_biophysical_delta(mutations)),
        ('positional', lambda: compute_mutation_positional_features(
            mutations, wt_heavy or '', wt_light)),
        ('lm_score', lambda: compute_lm_mutation_scores(
            wt_heavy or '', heavy_seqs, mutations)),
    ]

    for name, func in mut_funcs:
        try:
            start = time.time()
            print(f"  Generating {name}...", end=' ', flush=True)
            feat = func()
            elapsed = time.time() - start
            print(f"done ({feat.shape[1]} features, {elapsed:.1f}s)")
            augmentations[name] = feat
        except Exception as e:
            print(f"FAILED: {e}")

    return augmentations


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

    best = result.best_overall
    output = {
        'metadata': {
            'dataset': dataset_name,
            'search_mode': search_mode,
            'reps_available': reps_tested,
            'timestamp': datetime.now().isoformat(),
            'domain': 'antibody',
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

    blueprint_path = output_path.replace('.json', '_blueprint.py')
    save_blueprint_code(result, reps_tested, blueprint_path)


def save_blueprint_code(result: SearchResult, reps_tested: List[str], output_path: str):
    """Generate importable Python code for reusing the hybrids."""
    lines = [
        '"""',
        'Auto-generated antibody hybrid blueprints.',
        '',
        'Usage:',
        '    from results_blueprint import create_rf_hybrid',
        '    X = create_rf_hybrid(heavy_seqs, light_seqs, labels)',
        '"""',
        '',
        'from kirby.hybrid import create_hybrid, apply_feature_selection',
        'from kirby.representations.antibody import (',
        '    create_kmer_frequency, create_physicochemical_profile,',
        '    create_ablang2_embeddings, create_antiberty_embeddings,',
        '    create_igbert_embeddings, create_esm2_embeddings,',
        '    create_prot_t5_embeddings, create_ab_roberta_embeddings,',
        '    create_balm_embeddings, create_pigen_embeddings,',
        '    create_esmc_embeddings, create_sapiens_embeddings,',
        '    compute_aac_features, compute_blosum62_features,',
        '    create_imgt_onehot,',
        ')',
        '',
        '',
    ]

    for model_name, r in result.best_by_model.items():
        bp = _get(r, 'blueprint')
        n_features = _get(r, 'n_features')
        score_mean = _get(r, 'score_mean')

        lines.extend([
            f"def create_{model_name}_hybrid(heavy_seqs, light_seqs, labels):",
            f'    """',
            f'    Create the optimal antibody hybrid for {model_name.upper()} model.',
            f'    ',
            f'    Blueprint: method={bp.allocation_method}, budget={bp.budget}, importance={bp.importance_method}',
            f'    Expected dimension: ~{n_features}',
            f'    Expected score: {score_mean:.4f}',
            f'    """',
            f'    # Generate required representations (see find_best_hybrid_antibody.py for dispatch)',
            f'    reps = {{}}',
            f'    # TODO: Fill in rep generation calls for: {bp.reps}',
            f'    ',
            f'    X_hybrid, feature_info = create_hybrid(',
            f'        reps, labels,',
            f"        allocation_method='{bp.allocation_method}',",
            f'        budget={bp.budget},',
            f"        importance_method='{bp.importance_method}',",
            f'    )',
            f'    return X_hybrid, feature_info',
            f'',
            f'',
        ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Blueprint code saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find the optimal hybrid antibody representation for each model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sabdab_affinity --reps fast --search fast
  %(prog)s sabdab_affinity --reps fast+lm --search medium
  %(prog)s data.csv --heavy-col VH --light-col VL --target-col ddG --reps all_server

Output:
  - Best overall: single best model+hybrid combination
  - Best by dimension: optimal at each tier (50, 100, 200, 500)
  - Best by model: optimal hybrid per model type
  - Performance matrix: full Model x Dimension grid
  - Baseline comparison: hybrid vs full-dimension baselines
        """
    )

    parser.add_argument('dataset', type=str,
                       help="'sabdab_affinity' or path to CSV")
    parser.add_argument('--heavy-col', type=str, default='VH',
                       help="Heavy chain column name (default: VH)")
    parser.add_argument('--light-col', type=str, default='VL',
                       help="Light chain column name (default: VL)")
    parser.add_argument('--target-col', type=str, default='target',
                       help="Target column name (default: target)")
    parser.add_argument('--reps', type=str, default='fast',
                       help="'fast', 'fast+lm', 'all_server', 'all', or comma-separated list")
    parser.add_argument('--search', type=str, default='fast',
                       choices=list(_SEARCH_NAME_MAP.keys()),
                       help="Search speed: 'fast', 'medium', 'long'")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augment', action='store_true',
                       help='Include augmented features (developability, humanness, etc.)')
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
    parser.add_argument('--identity-threshold', type=float, default=0.8,
                       help="Sequence identity threshold for GroupKFold (default: 0.8)")
    parser.add_argument('--cache', action='store_true',
                       help='Cache hybrid search results to disk')
    parser.add_argument('--delta', action='store_true',
                       help='Generate delta representations (ΔX = X_mut - X_wt). '
                            'Requires intra-family dataset with wildtype info.')
    parser.add_argument('--adaptive-pruning', action='store_true',
                       help='Enable adaptive model pruning during search')
    parser.add_argument('--n-folds', type=int, default=None,
                       help='Override number of CV folds (default: from speed tier)')
    parser.add_argument('--finetune', type=str, default=None,
                       help='Comma-separated fine-tuning methods: '
                            'lora_esm2, lora_esm2_small, lora_antiberty, lora_ablang2, '
                            'freeze_esm2, freeze_esm2_small, freeze_ablang2, '
                            'mlm_esm2, mlm_esm2_small, mlm_antiberty, mock_lora')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                       help='Fine-tuning epochs per fold (default: 5)')
    parser.add_argument('--lora-rank', type=int, default=8,
                       help='LoRA rank (default: 8)')
    parser.add_argument('--no-ft-cache', action='store_true',
                       help='Disable fine-tuning cache')

    args = parser.parse_args()

    speed = _SEARCH_NAME_MAP.get(args.search, args.search)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    heavy_seqs, light_seqs, labels, task_type, metadata = load_dataset(
        args.dataset, args.heavy_col, args.light_col, args.target_col
    )
    print(f"  {len(heavy_seqs)} antibodies, task: {task_type}")
    if light_seqs:
        print(f"  Paired (VH+VL)")
    else:
        print(f"  Heavy chain only")

    wt_heavy = metadata.get('wildtype_heavy')
    wt_light = metadata.get('wildtype_light')
    mutations = metadata.get('mutations')

    if args.delta and not wt_heavy:
        print("  WARNING: --delta requested but no wildtype available. Skipping delta reps.")
        args.delta = False

    # Compute sequence identity groups for CV
    print("  Assigning sequence identity groups...")
    groups, n_groups = assign_identity_groups(heavy_seqs, threshold=args.identity_threshold,
                                               min_groups=3)
    if groups is None:
        print(f"  Ungrouped CV (intra-family data, grouping not meaningful)")
    else:
        print(f"  {n_groups} groups (threshold={args.identity_threshold})")

    # Generate representations
    base_reps = get_representations(
        heavy_seqs, light_seqs, args.reps,
        wt_heavy=wt_heavy, wt_light=wt_light,
        mutations=mutations, delta=args.delta,
    )

    # Move small DMS reps from base to augmentations
    augmentations = None
    aug_names = _DMS_AUG_REPS.get(args.reps, set())
    if aug_names:
        dms_augs = {}
        for name in list(base_reps.keys()):
            if name in aug_names:
                dms_augs[name] = base_reps.pop(name)
        if dms_augs:
            augmentations = dms_augs
            print(f"  Moved {len(dms_augs)} small reps to augmentations: "
                  f"{sorted(dms_augs.keys())}")

    # Generate additional augmentations (--augment flag)
    if args.augment:
        extra_augs = get_augmentations(heavy_seqs, light_seqs)
        if extra_augs:
            if augmentations is None:
                augmentations = extra_augs
            else:
                augmentations.update(extra_augs)

    # Resolve model override
    models_arg = None
    if args.model:
        models_arg = [m.strip() for m in args.model.split(',')]

    alloc_arg = None
    if args.allocation:
        alloc_arg = [a.strip() for a in args.allocation.split(',')]

    # Build fine-tuning factories
    rep_factories = None
    if args.finetune:
        ft_methods = [m.strip() for m in args.finetune.split(',')]
        rep_factories = {}
        ft_cache = not args.no_ft_cache
        dataset_name = Path(args.dataset).stem if os.path.exists(args.dataset) else args.dataset

        for method in ft_methods:
            if method == 'mock_lora':
                from kirby.finetuning._base import mock_factory
                rep_factories['mock_lora'] = mock_factory(n_features=64, seed=args.seed)
            elif method.startswith('lora_'):
                from kirby.finetuning.antibody_lora import make_antibody_lora_factory
                model_key = method.replace('lora_', '')
                rep_factories[method] = make_antibody_lora_factory(
                    heavy_seqs=heavy_seqs,
                    light_seqs=light_seqs,
                    mutations=mutations,
                    dataset_name=dataset_name,
                    base_model=model_key,
                    lora_rank=args.lora_rank,
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            elif method.startswith('freeze_'):
                from kirby.finetuning.antibody_freeze import make_antibody_freeze_factory
                model_key = method.replace('freeze_', '')
                rep_factories[method] = make_antibody_freeze_factory(
                    heavy_seqs=heavy_seqs,
                    light_seqs=light_seqs,
                    mutations=mutations,
                    dataset_name=dataset_name,
                    base_model=model_key,
                    epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            elif method.startswith('mlm_'):
                from kirby.finetuning.antibody_mlm import make_antibody_mlm_factory
                model_key = method.replace('mlm_', '')
                rep_factories[method] = make_antibody_mlm_factory(
                    heavy_seqs=heavy_seqs,
                    mutations=mutations,
                    dataset_name=dataset_name,
                    base_model=model_key,
                    mlm_epochs=args.finetune_epochs,
                    cache=ft_cache,
                )
            else:
                print(f"  WARNING: Unknown fine-tuning method '{method}', skipping")

        if rep_factories:
            print(f"\nFine-tuning factories: {list(rep_factories.keys())}")
        else:
            rep_factories = None

    # Call domain-agnostic search
    result = find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        models=models_arg,
        metric='r2' if task_type == 'regression' else 'accuracy',
        task_type=task_type,
        speed=speed,
        n_folds=args.n_folds,
        allocation_methods=alloc_arg,
        return_fitted=args.save_weights,
        weight_method=args.weight_method if args.save_weights else None,
        test_rep_subsets=args.test_rep_subsets,
        seed=args.seed,
        augmentations=augmentations,
        cache=args.cache,
        adaptive_pruning=args.adaptive_pruning,
        rep_factories=rep_factories,
    )

    # Print summary
    print_summary(result, task_type)

    # Save results
    if args.output:
        output_path = args.output
    else:
        dataset_name = Path(args.dataset).stem if os.path.exists(args.dataset) else args.dataset
        output_path = f"results_{dataset_name}_{args.search}.json"

    save_results(result, args.dataset, args.search, list(base_reps.keys()), output_path,
                 base_reps=base_reps, labels=labels)

    return 0


if __name__ == '__main__':
    sys.exit(main())
