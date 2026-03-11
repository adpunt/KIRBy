#!/usr/bin/env python3
"""
Antibody Region Ablation Experiment

Tests whether finer-grained pooling of LM token embeddings — per-CDR,
framework-only, entropy-guided position selection — improves prediction
on antibody tasks vs the standard full-sequence or CDR-combined pooling.

12 methods tested (all using ESM-2 650M token-level hidden states):
  full_seq         — mean all tokens (baseline)
  cdr_combined     — mean CDR1+2+3 (baseline)
  per_cdr          — CDR1, CDR2, CDR3 separately concatenated
  fw_only          — FR1+2+3+4 only
  cdr_plus_fw      — CDR concat FW
  all_7_regions    — each of 7 regions separate
  var_top10        — top-10 entropy positions
  var_top20        — top-20 entropy positions
  var_top50        — top-50 entropy positions
  region_var_k3    — top-3 per region, 7 regions
  region_var_k5    — top-5 per region, 7 regions
  region_var_k10   — top-10 per region, 7 regions

Usage:
    python tests/experiments/antibody_region_ablation.py --help
    python tests/experiments/antibody_region_ablation.py
    python tests/experiments/antibody_region_ablation.py --datasets flab_binding_phillips_h3
    python tests/experiments/antibody_region_ablation.py --methods full_seq,per_cdr,var_top20
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import gc
import argparse
import warnings
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from kirby.representations.antibody import (
    _IMGT_REGION_RANGES,
    get_region_indices,
    get_imgt_position_map,
)

# =========================================================================
# CONFIG
# =========================================================================

DEFAULT_DATASETS = [
    'flab_binding_phillips_h3',   # 5000, CDR-H3 DMS
    'flab_expression_koenig',     # 4275, expression DMS
    'flab_binding_koenig',        # 4275, binding DMS
    'flab_thermo_tresanco',       # 672, inter-family, heavy-only
    'abagym_HER2',                # 146, small DMS
]

ALL_METHODS = [
    'full_seq', 'cdr_combined', 'per_cdr', 'fw_only', 'cdr_plus_fw',
    'all_7_regions', 'var_top10', 'var_top20', 'var_top50',
    'region_var_k3', 'region_var_k5', 'region_var_k10',
]

REGION_ORDER = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4']
CDR_REGIONS = ['CDR1', 'CDR2', 'CDR3']
FW_REGIONS = ['FR1', 'FR2', 'FR3', 'FR4']

ESM2_DIM = 1280  # per-token hidden dim


# =========================================================================
# TOKEN-LEVEL ESM-2 EXTRACTION
# =========================================================================

def extract_token_embeddings_esm2(
    sequences: List[str],
    batch_size: int = 4,
) -> List[np.ndarray]:
    """Extract per-residue ESM-2 hidden states (no pooling).

    Returns list of (L_i, 1280) arrays, one per sequence.
    Processes in batches, strips BOS/EOS tokens.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'facebook/esm2_t33_650M_UR50D'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    all_embeddings = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state  # (B, L, 1280)

        for i in range(len(batch)):
            seq_len = len(batch[i])
            # ESM-2: token 0 = BOS, tokens 1..seq_len = residues, then EOS+PAD
            # Extract residue tokens only (skip BOS)
            residue_embs = hidden[i, 1:seq_len + 1, :].cpu().numpy()
            all_embeddings.append(residue_embs.astype(np.float32))

    # Free GPU memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_embeddings


# =========================================================================
# ENTROPY COMPUTATION
# =========================================================================

def _compute_entropy_from_maps(
    sequences: List[str],
    pos_maps_list: List[dict],
) -> Dict[int, float]:
    """Compute Shannon entropy from pre-computed IMGT position maps.

    Avoids redundant ANARCI calls by reusing pos_maps_list.
    """
    from collections import defaultdict

    position_aas = defaultdict(list)
    for seq, pos_map in zip(sequences, pos_maps_list):
        for imgt_pos, raw_idx in pos_map.items():
            if raw_idx < len(seq):
                position_aas[imgt_pos].append(seq[raw_idx])

    entropy_map = {}
    for imgt_pos, aas in position_aas.items():
        counts = Counter(aas)
        total = len(aas)
        if total == 0:
            entropy_map[imgt_pos] = 0.0
            continue
        probs = np.array([c / total for c in counts.values()])
        probs = probs[probs > 0]
        entropy_map[imgt_pos] = float(-np.sum(probs * np.log2(probs)))

    return entropy_map


def get_top_k_variable_indices(
    pos_map: dict,
    entropy_map: Dict[int, float],
    k: int,
    seq_len: int,
) -> List[int]:
    """Get 0-based raw indices of top-k highest-entropy IMGT positions.

    Args:
        pos_map: pre-computed {imgt_pos: raw_idx} from get_imgt_position_map()
        entropy_map: {imgt_pos: entropy} from compute_positional_entropy()
        k: number of positions to select
        seq_len: length of the raw sequence (for fallback)
    """
    if not pos_map:
        return list(range(min(k, seq_len)))

    # Score each residue by its IMGT position entropy
    scored = []
    for imgt_pos, raw_idx in pos_map.items():
        ent = entropy_map.get(imgt_pos, 0.0)
        scored.append((ent, raw_idx))

    # Sort by entropy descending, take top k
    scored.sort(key=lambda x: -x[0])
    indices = [raw_idx for _, raw_idx in scored[:k]]
    indices.sort()  # preserve sequence order

    if not indices:
        return list(range(min(k, seq_len)))
    return indices


def get_top_k_per_region_indices(
    pos_map: dict,
    regions: dict,
    entropy_map: Dict[int, float],
    k: int,
    seq_len: int,
) -> Dict[str, List[int]]:
    """Get top-k highest-entropy positions within each of the 7 regions.

    Args:
        pos_map: pre-computed {imgt_pos: raw_idx} from get_imgt_position_map()
        regions: pre-computed region dict from get_region_indices()
        entropy_map: {imgt_pos: entropy} from compute_positional_entropy()
        k: number of positions to select per region
        seq_len: length of the raw sequence (for fallback)

    Returns dict {region_name: [raw_indices]}.
    """
    # If fallback, return uniform selection
    if 'FULL' in regions:
        step = max(1, seq_len // 7)
        return {r: list(range(i * step, min((i + 1) * step, seq_len)))[:k]
                for i, r in enumerate(REGION_ORDER)}

    # Build reverse map once (not per-region)
    raw_to_imgt = {raw_idx: imgt_pos for imgt_pos, raw_idx in pos_map.items()}

    result = {}
    for region in REGION_ORDER:
        region_raw_indices = regions.get(region, [])
        if not region_raw_indices:
            result[region] = []
            continue

        scored = []
        for raw_idx in region_raw_indices:
            imgt_pos = raw_to_imgt.get(raw_idx)
            ent = entropy_map.get(imgt_pos, 0.0) if imgt_pos is not None else 0.0
            scored.append((ent, raw_idx))

        scored.sort(key=lambda x: -x[0])
        selected = sorted([raw_idx for _, raw_idx in scored[:k]])
        result[region] = selected

    return result


# =========================================================================
# POOLING FUNCTIONS
# =========================================================================

def _mean_pool_indices(token_embs: np.ndarray, indices: List[int]) -> np.ndarray:
    """Mean-pool token embeddings at given indices. Returns (D,) vector."""
    if not indices:
        return token_embs.mean(axis=0)
    valid = [i for i in indices if i < token_embs.shape[0]]
    if not valid:
        return token_embs.mean(axis=0)
    return token_embs[valid].mean(axis=0)


def pool_full_seq(token_embs: np.ndarray, **kwargs) -> np.ndarray:
    """Mean over all tokens → (D,)."""
    return token_embs.mean(axis=0)


def pool_cdr_combined(
    token_embs: np.ndarray, regions: dict, **kwargs,
) -> np.ndarray:
    """Mean over CDR1+2+3 tokens combined → (D,)."""
    cdr_indices = []
    for r in CDR_REGIONS:
        cdr_indices.extend(regions.get(r, []))
    return _mean_pool_indices(token_embs, cdr_indices)


def pool_per_cdr(
    token_embs: np.ndarray, regions: dict, **kwargs,
) -> np.ndarray:
    """CDR1, CDR2, CDR3 pooled separately → (3*D,)."""
    parts = []
    for r in CDR_REGIONS:
        parts.append(_mean_pool_indices(token_embs, regions.get(r, [])))
    return np.concatenate(parts)


def pool_fw_only(
    token_embs: np.ndarray, regions: dict, **kwargs,
) -> np.ndarray:
    """FR1+2+3+4 pooled together → (D,)."""
    fw_indices = []
    for r in FW_REGIONS:
        fw_indices.extend(regions.get(r, []))
    return _mean_pool_indices(token_embs, fw_indices)


def pool_cdr_plus_fw(
    token_embs: np.ndarray, regions: dict, **kwargs,
) -> np.ndarray:
    """CDR pooled ++ FW pooled → (2*D,)."""
    cdr = pool_cdr_combined(token_embs, regions)
    fw = pool_fw_only(token_embs, regions)
    return np.concatenate([cdr, fw])


def pool_all_7_regions(
    token_embs: np.ndarray, regions: dict, **kwargs,
) -> np.ndarray:
    """Each of 7 regions pooled separately → (7*D,)."""
    parts = []
    for r in REGION_ORDER:
        parts.append(_mean_pool_indices(token_embs, regions.get(r, [])))
    return np.concatenate(parts)


def pool_top_k_variable(
    token_embs: np.ndarray, pos_map: dict, entropy_map: dict,
    k: int = 20, **kwargs,
) -> np.ndarray:
    """Mean over top-k highest-entropy positions → (D,)."""
    indices = get_top_k_variable_indices(pos_map, entropy_map, k, token_embs.shape[0])
    return _mean_pool_indices(token_embs, indices)


def pool_top_k_per_region(
    token_embs: np.ndarray, pos_map: dict, regions: dict,
    entropy_map: dict, k: int = 5, **kwargs,
) -> np.ndarray:
    """Top-k per region, 7 regions separate → (7*D,)."""
    per_region = get_top_k_per_region_indices(
        pos_map, regions, entropy_map, k, token_embs.shape[0])
    parts = []
    for r in REGION_ORDER:
        parts.append(_mean_pool_indices(token_embs, per_region.get(r, [])))
    return np.concatenate(parts)


# =========================================================================
# METHOD REGISTRY
# =========================================================================

def build_method_features(
    method: str,
    token_embs_list: List[np.ndarray],
    regions_list: List[dict],
    pos_maps_list: List[dict],
    entropy_map: Dict[int, float],
) -> np.ndarray:
    """Build (N, D) feature matrix for a given pooling method.

    Args:
        pos_maps_list: pre-computed position maps (one per sequence)
            from get_imgt_position_map(), avoids redundant ANARCI calls.
    """
    rows = []
    for i in range(len(token_embs_list)):
        te = token_embs_list[i]
        reg = regions_list[i]
        pm = pos_maps_list[i]

        if method == 'full_seq':
            rows.append(pool_full_seq(te))
        elif method == 'cdr_combined':
            rows.append(pool_cdr_combined(te, reg))
        elif method == 'per_cdr':
            rows.append(pool_per_cdr(te, reg))
        elif method == 'fw_only':
            rows.append(pool_fw_only(te, reg))
        elif method == 'cdr_plus_fw':
            rows.append(pool_cdr_plus_fw(te, reg))
        elif method == 'all_7_regions':
            rows.append(pool_all_7_regions(te, reg))
        elif method == 'var_top10':
            rows.append(pool_top_k_variable(te, pm, entropy_map, k=10))
        elif method == 'var_top20':
            rows.append(pool_top_k_variable(te, pm, entropy_map, k=20))
        elif method == 'var_top50':
            rows.append(pool_top_k_variable(te, pm, entropy_map, k=50))
        elif method == 'region_var_k3':
            rows.append(pool_top_k_per_region(te, pm, reg, entropy_map, k=3))
        elif method == 'region_var_k5':
            rows.append(pool_top_k_per_region(te, pm, reg, entropy_map, k=5))
        elif method == 'region_var_k10':
            rows.append(pool_top_k_per_region(te, pm, reg, entropy_map, k=10))
        else:
            raise ValueError(f"Unknown method: {method}")

    return np.vstack(rows).astype(np.float32)


# =========================================================================
# EVALUATION
# =========================================================================

def evaluate_cv(X, labels, groups, n_folds=5):
    """5-fold GroupKFold + StandardScaler + RidgeCV. Returns dict with r2 and spearman."""
    n_unique = len(np.unique(groups)) if groups is not None else 0

    if groups is None or n_unique < 2:
        print("  WARNING: <2 unique groups, skipping dataset")
        return {'r2_mean': float('nan'), 'r2_std': float('nan'),
                'spearman_mean': float('nan'), 'spearman_std': float('nan'),
                'per_fold': []}

    effective_folds = min(n_folds, n_unique)
    cv = GroupKFold(n_splits=effective_folds)
    splits = list(cv.split(X, labels, groups=groups))

    scores = []
    for train_idx, val_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        model = RidgeCV(alphas=np.logspace(-6, 6, 13))
        model.fit(X_tr, labels[train_idx])
        preds = model.predict(X_va)
        r2 = float(r2_score(labels[val_idx], preds))
        rho = float(spearmanr(labels[val_idx], preds).correlation)
        scores.append({'r2': r2, 'spearman': rho})

    r2_vals = [s['r2'] for s in scores]
    rho_vals = [s['spearman'] for s in scores]
    return {'r2_mean': float(np.mean(r2_vals)),
            'r2_std': float(np.std(r2_vals)),
            'spearman_mean': float(np.mean(rho_vals)),
            'spearman_std': float(np.std(rho_vals)),
            'per_fold': scores}


# =========================================================================
# DATASET PROCESSING
# =========================================================================

def process_chain(
    sequences: List[str],
    methods: List[str],
    entropy_map: Optional[Dict[int, float]] = None,
    batch_size: int = 4,
    chain_name: str = 'H',
) -> Tuple[Dict[str, np.ndarray], Dict[int, float]]:
    """Extract tokens for one chain, compute entropy + regions, build all methods.

    Returns (features_dict, entropy_map).
    """
    # Skip empty chains
    if not sequences or all(s == '' for s in sequences):
        return {}, entropy_map or {}

    print(f"    Extracting ESM-2 tokens ({chain_name})...", flush=True)
    token_embs_list = extract_token_embeddings_esm2(sequences, batch_size)
    print(f"      Done. {len(token_embs_list)} sequences, "
          f"shapes: {token_embs_list[0].shape} ... {token_embs_list[-1].shape}")

    # Compute regions and position maps (single ANARCI pass each)
    needs_regions = any(m not in ('full_seq',) for m in methods)
    needs_entropy = any('var' in m for m in methods)

    if needs_regions:
        print(f"    Computing IMGT regions ({chain_name})...", flush=True)
        regions_list = [get_region_indices(s) for s in sequences]
    else:
        regions_list = [{}] * len(sequences)

    if needs_entropy:
        print(f"    Computing IMGT position maps ({chain_name})...", flush=True)
        pos_maps_list = [get_imgt_position_map(s) for s in sequences]
    else:
        pos_maps_list = [{}] * len(sequences)

    # Compute entropy from pre-computed position maps (no extra ANARCI calls)
    if needs_entropy and entropy_map is None:
        print(f"    Computing positional entropy ({chain_name})...", flush=True)
        entropy_map = _compute_entropy_from_maps(sequences, pos_maps_list)
        n_positions = len(entropy_map)
        if n_positions > 0:
            top3 = sorted(entropy_map.items(), key=lambda x: -x[1])[:3]
            print(f"      {n_positions} IMGT positions, "
                  f"top-3 entropy: {[(p, f'{e:.2f}') for p, e in top3]}")
    elif entropy_map is None:
        entropy_map = {}

    # Build features for each method (uses pre-computed maps, no ANARCI)
    features = {}
    for method in methods:
        X = build_method_features(
            method, token_embs_list, regions_list, pos_maps_list, entropy_map)
        features[method] = X
        print(f"      {method}: {X.shape}")

    return features, entropy_map


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Antibody region ablation experiment')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset names')
    parser.add_argument('--methods', type=str, default=None,
                        help=f'Comma-separated methods (default: all). '
                             f'Options: {",".join(ALL_METHODS)}')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4,
                        help='ESM-2 batch size (default: 4)')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)
    methods = (args.methods.split(',') if args.methods
               else ALL_METHODS)

    # Validate methods
    for m in methods:
        if m not in ALL_METHODS:
            parser.error(f"Unknown method: {m}. Choose from: {ALL_METHODS}")

    output_dir = args.output or 'tests/results/comparisons'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ANTIBODY REGION ABLATION")
    print(f"  Datasets: {datasets}")
    print(f"  Methods: {methods}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load dataset
        try:
            from antibody_baseline_survey import load_dataset
            data = load_dataset(ds_name)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        heavy_seqs = data['heavy_seqs']
        light_seqs = data['light_seqs']
        labels = data['labels']
        groups = data['groups']
        n = len(labels)
        has_light = light_seqs is not None and any(s != '' for s in light_seqs)

        print(f"  {n} samples, {len(np.unique(groups))} groups, "
              f"paired={'yes' if has_light else 'no'}")

        ds_results = {
            'n_samples': n,
            'has_light': has_light,
            'methods': {},
            'entropy': {},
        }

        # Process heavy chain
        h_features, h_entropy = process_chain(
            heavy_seqs, methods, batch_size=args.batch_size, chain_name='H')

        # Save entropy map
        ds_results['entropy']['heavy'] = {
            str(k): round(v, 4) for k, v in sorted(h_entropy.items())
        }

        # Process light chain (if available)
        if has_light:
            l_features, l_entropy = process_chain(
                light_seqs, methods, batch_size=args.batch_size, chain_name='L')
            ds_results['entropy']['light'] = {
                str(k): round(v, 4) for k, v in sorted(l_entropy.items())
            }
        else:
            l_features = {}

        # Evaluate each method
        print(f"\n  --- Evaluation ({args.n_folds}-fold CV) ---")
        for method in methods:
            h_X = h_features.get(method)
            if h_X is None:
                print(f"    {method}: SKIPPED (no heavy features)")
                continue

            if has_light and method in l_features:
                l_X = l_features[method]
                X = np.hstack([h_X, l_X])
            else:
                X = h_X

            result = evaluate_cv(X, labels, groups, args.n_folds)

            ds_results['methods'][method] = {
                'r2_mean': round(result['r2_mean'], 4),
                'r2_std': round(result['r2_std'], 4),
                'spearman_mean': round(result['spearman_mean'], 4),
                'spearman_std': round(result['spearman_std'], 4),
                'dim': X.shape[1],
                'per_fold': result['per_fold'],
            }
            print(f"    {method:20s}  R²={result['r2_mean']:.4f} ± {result['r2_std']:.4f}"
                  f"  ρ={result['spearman_mean']:.4f} ± {result['spearman_std']:.4f}"
                  f"  (dim={X.shape[1]})")

        all_results[ds_name] = ds_results

        # Free memory between datasets
        del h_features, l_features
        gc.collect()

    # Summary table — R²
    print(f"\n{'='*70}")
    print("SUMMARY (R²)")
    print(f"{'='*70}")
    header = f"{'Method':20s}"
    for ds in datasets:
        short = ds.split('_', 1)[-1][:12]
        header += f"  {short:>12s}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:20s}"
        for ds in datasets:
            if ds in all_results and method in all_results[ds]['methods']:
                r2 = all_results[ds]['methods'][method]['r2_mean']
                row += f"  {r2:12.4f}"
            else:
                row += f"  {'---':>12s}"
        print(row)

    # Summary table — Spearman ρ
    print(f"\n{'='*70}")
    print("SUMMARY (Spearman ρ)")
    print(f"{'='*70}")
    header = f"{'Method':20s}"
    for ds in datasets:
        short = ds.split('_', 1)[-1][:12]
        header += f"  {short:>12s}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:20s}"
        for ds in datasets:
            if ds in all_results and method in all_results[ds]['methods']:
                rho = all_results[ds]['methods'][method]['spearman_mean']
                row += f"  {rho:12.4f}"
            else:
                row += f"  {'---':>12s}"
        print(row)

    # Delta vs full_seq baseline
    print(f"\nDelta vs full_seq (R²):")
    for method in methods:
        if method == 'full_seq':
            continue
        row = f"  {method:20s}"
        for ds in datasets:
            if (ds in all_results
                    and method in all_results[ds]['methods']
                    and 'full_seq' in all_results[ds]['methods']):
                delta = (all_results[ds]['methods'][method]['r2_mean']
                         - all_results[ds]['methods']['full_seq']['r2_mean'])
                sign = '+' if delta >= 0 else ''
                row += f"  {sign}{delta:11.4f}"
            else:
                row += f"  {'---':>12s}"
        print(row)

    # Save results
    output_path = os.path.join(output_dir, 'antibody_region_ablation.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # Save entropy maps separately (useful for analysis)
    entropy_path = os.path.join(output_dir, 'antibody_region_entropy.json')
    entropy_data = {ds: res.get('entropy', {}) for ds, res in all_results.items()}
    with open(entropy_path, 'w') as f:
        json.dump(entropy_data, f, indent=2)
    print(f"Entropy maps saved: {entropy_path}")


if __name__ == '__main__':
    main()
