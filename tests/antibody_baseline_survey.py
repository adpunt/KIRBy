#!/usr/bin/env python3
"""
Antibody Baseline Survey

Simple LM embeddings → Ridge → R², per the compass document.
No hybrids, no tuning, no complex presets.

Usage:
    python tests/antibody_baseline_survey.py flab_expression_koenig
    python tests/antibody_baseline_survey.py all_dms
    python tests/antibody_baseline_survey.py all
    python tests/antibody_baseline_survey.py flab_expression_koenig --skip-lm
    python tests/antibody_baseline_survey.py all --models ridge,rf,xgboost
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')


# =============================================================================
# DATASET REGISTRY
# =============================================================================

# {name: (loader_module, group_strategy)}
# group_strategy:
#   'position_modulo' — DMS/SSM: group by mutation position % n_folds (ProteinGym standard)
#   'identity' — inter-family: group by unique sequence hash
#   'complex' — DDG: group by PDB complex ID
#   'experiment' — AbAgym combined: group by experiment name
DATASET_REGISTRY = {
    # DMS (intra-family) — position-based splits to prevent same-position leakage
    'flab_expression_koenig': ('flab', 'position_modulo'),
    'flab_binding_koenig':    ('flab', 'position_modulo'),
    'abagym_Ang2':            ('abagym', 'position_modulo'),
    'abagym_EGFR':            ('abagym', 'position_modulo'),
    'abagym_lysozyme':        ('abagym', 'position_modulo'),
    'abagym_VEGF':            ('abagym', 'position_modulo'),
    'abagym_HER2':            ('abagym', 'position_modulo'),
    # Inter-family
    'ddg_combined':           ('ddg', 'complex'),
    # Full SKEMPI (all protein-protein, not just Ab-Ag) — experiment option.
    # Loaded via load_ddg_combined_dataset(ab_ag_only=False).
    # Non-Ab protein sequences will not benefit from antibody-specific LM reps.
    'ddg_combined_full':      ('ddg_full', 'complex'),
    'abagym_all':             ('abagym', 'experiment'),
    'abcov_kd':               ('abcov', 'identity'),
    'flab_thermo_jain':       ('flab', 'identity'),
    'flab_aggregation_jain':  ('flab', 'identity'),
    # FLAb2 expansion — DMS datasets get position splits, inter-family keep identity
    'flab_binding_phillips_h3':   ('flab', 'position_modulo'),
    'flab_binding_adams':         ('flab', 'position_modulo'),
    'flab_expression_adams':      ('flab', 'position_modulo'),
    'flab_thermo_tresanco':       ('flab', 'identity'),
    'flab_thermo_hutchinson':     ('flab', 'identity'),
    'flab_immunogenicity_marks':  ('flab', 'identity'),
    'flab_aggregation_jain_hicrt':('flab', 'identity'),
    'flab_polyreact_jain_bvp':    ('flab', 'identity'),
    'flab_polyreact_jain_cic':    ('flab', 'identity'),
    'flab_pk_jain_fcrnrt':        ('flab', 'identity'),
    'flab_pk_jain_heprt':         ('flab', 'identity'),
    'flab_aggregation_jain_sas':  ('flab', 'identity'),
    # Classification — Trastuzumab-HER2 binding
    'tz_her2_binary':             ('trastuzumab', 'edit_distance'),
    'tz_her2_multiclass':         ('trastuzumab_3class', 'edit_distance'),
    # Classification — Humanness (human vs non-human VH)
    'humanness_species':          ('humanness_species', 'identity'),
    'humanness_therapeutic':      ('humanness_therapeutic', 'identity'),
    # Ginkgo developability — 9 regression properties, pre-computed 5-fold splits
    'ginkgo_hic':                 ('ginkgo', 'pre_split'),
    'ginkgo_tm2':                 ('ginkgo', 'pre_split'),
    'ginkgo_titer':               ('ginkgo', 'pre_split'),
    'ginkgo_pr_cho':              ('ginkgo', 'pre_split'),
    'ginkgo_acsins':              ('ginkgo', 'pre_split'),
    'ginkgo_purity':              ('ginkgo', 'pre_split'),
    'ginkgo_aggregation':         ('ginkgo', 'pre_split'),
    'ginkgo_colloidal':           ('ginkgo', 'pre_split'),
    'ginkgo_heparin':             ('ginkgo', 'pre_split'),
    # Graphinity synthetic DDG — FoldX/Rosetta computed ΔΔG
    'graphinity_foldx':           ('graphinity_foldx', 'complex'),
    'graphinity_rosetta':         ('graphinity_rosetta', 'complex'),
    'graphinity_augmented':       ('graphinity_augmented', 'complex'),
}

DOMAIN_GROUPS = {
    'all_dms': [
        'flab_expression_koenig', 'flab_binding_koenig',
        'flab_binding_phillips_h3', 'flab_binding_adams', 'flab_expression_adams',
        'abagym_Ang2', 'abagym_EGFR', 'abagym_lysozyme',
        'abagym_VEGF', 'abagym_HER2',
    ],
    'all_inter': [
        'ddg_combined', 'abagym_all', 'abcov_kd',
        'flab_thermo_jain', 'flab_aggregation_jain',
        'flab_thermo_tresanco', 'flab_thermo_hutchinson',
        'flab_immunogenicity_marks',
        'flab_aggregation_jain_hicrt', 'flab_aggregation_jain_sas',
        'flab_polyreact_jain_bvp', 'flab_polyreact_jain_cic',
        'flab_pk_jain_fcrnrt', 'flab_pk_jain_heprt',
    ],
    'all_clf': [
        'tz_her2_binary', 'tz_her2_multiclass',
        'humanness_species', 'humanness_therapeutic',
    ],
    'all_ginkgo': [
        'ginkgo_hic', 'ginkgo_tm2', 'ginkgo_titer', 'ginkgo_pr_cho',
        'ginkgo_acsins', 'ginkgo_purity', 'ginkgo_aggregation',
        'ginkgo_colloidal', 'ginkgo_heparin',
    ],
    'all_ddg': [
        'ddg_combined', 'ddg_combined_full',
        'graphinity_foldx', 'graphinity_rosetta', 'graphinity_augmented',
    ],
    # 'all' excludes ddg_combined_full by default (it's an experiment option,
    # not part of the standard survey). Use --include-all-skempi to add it.
    'all': [d for d in DATASET_REGISTRY.keys() if d != 'ddg_combined_full'],
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _abagym_full_name(short_name: str) -> str:
    """Convert short AbAgym name to full experiment name for the loader."""
    mapping = {
        'abagym_Ang2': 'abagym_Ang2_2017_G6',
        'abagym_EGFR': 'abagym_EGFR_2013_Cetuximab',
        'abagym_lysozyme': 'abagym_lysozyme_2019_D441',
        'abagym_VEGF': 'abagym_VEGF_2017b_G6',
        'abagym_HER2': 'abagym_HER2_2021_trastuzumab',
        'abagym_all': 'abagym_all',
    }
    return mapping.get(short_name, short_name)


def load_dataset(name: str) -> dict:
    """Load antibody dataset. Returns dict with heavy_seqs, light_seqs, labels, groups."""
    source, group_strategy = DATASET_REGISTRY[name]

    if source == 'flab':
        from kirby.datasets.flab import load_flab_dataset
        data = load_flab_dataset(name)
    elif source == 'abagym':
        from kirby.datasets.abagym import load_abagym_dataset
        data = load_abagym_dataset(_abagym_full_name(name))
    elif source == 'ddg':
        from kirby.datasets.ddg_combined import load_ddg_combined_dataset
        data = load_ddg_combined_dataset(ab_ag_only=True)
    elif source == 'ddg_full':
        # Full SKEMPI: all protein-protein complexes, not just Ab-Ag.
        # NOTE: non-Ab sequences will not benefit from antibody LM reps.
        from kirby.datasets.ddg_combined import load_ddg_combined_dataset
        print("  [ddg_combined_full] Using ab_ag_only=False — includes all "
              "protein-protein SKEMPI entries (barnase-barstar, T4-lysozyme, etc.)")
        data = load_ddg_combined_dataset(ab_ag_only=False)
    elif source == 'abcov':
        from kirby.datasets.abcov import load_abcov_dataset
        data = load_abcov_dataset(name)
    elif source == 'trastuzumab':
        from kirby.datasets.trastuzumab import load_trastuzumab_dataset
        data = load_trastuzumab_dataset(preset='binary')
    elif source == 'trastuzumab_3class':
        from kirby.datasets.trastuzumab import load_trastuzumab_dataset
        data = load_trastuzumab_dataset(preset='multiclass')
    elif source == 'humanness_species':
        from kirby.datasets.humanness import load_humanness_dataset
        data = load_humanness_dataset(preset='species')
    elif source == 'humanness_therapeutic':
        from kirby.datasets.humanness import load_humanness_dataset
        data = load_humanness_dataset(preset='therapeutic')
    elif source == 'ginkgo':
        from kirby.datasets.ginkgo_dev import load_ginkgo_dataset
        data = load_ginkgo_dataset(name)
    elif source == 'graphinity_foldx':
        from kirby.datasets.graphinity import load_graphinity_dataset
        data = load_graphinity_dataset(preset='foldx')
    elif source == 'graphinity_rosetta':
        from kirby.datasets.graphinity import load_graphinity_dataset
        data = load_graphinity_dataset(preset='rosetta')
    elif source == 'graphinity_augmented':
        from kirby.datasets.graphinity import load_ddg_augmented_dataset
        data = load_ddg_augmented_dataset()
    else:
        raise ValueError(f"Unknown source: {source}")

    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    task_type = data.get('task_type')
    if task_type not in ('classification', 'regression'):
        raise ValueError(
            f"Antibody loader for '{name}' (source={source}) returned "
            f"task_type={task_type!r}. Every loader MUST set 'task_type' "
            f"to 'classification' or 'regression'."
        )

    # Guard: if light_seqs is None (unpaired), use empty strings
    if light_seqs is None:
        light_seqs = [''] * len(heavy_seqs)

    # Assign groups
    groups, n_groups = _assign_groups(
        heavy_seqs, light_seqs, group_strategy, data)

    return {
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'labels': labels,
        'groups': groups,
        'n_groups': n_groups,
        'n_samples': len(labels),
        'mutations': data.get('mutations'),
        'task_type': task_type,
        'edit_distances': data.get('edit_distances'),
        'fold_assignments': data.get('fold_assignments'),
        'property_name': data.get('property_name'),
    }


def _assign_groups(heavy_seqs, light_seqs, strategy, data):
    """Assign CV groups based on strategy."""
    if strategy == 'pre_split':
        # Pre-computed fold assignments (e.g. Ginkgo hierarchical clustering)
        folds = data.get('fold_assignments')
        if folds is not None:
            groups = np.asarray(folds, dtype=int)
            return groups, len(np.unique(groups))
        # Fallback if no fold assignments
        print("    WARNING: pre_split strategy but no fold_assignments, "
              "falling back to identity")
        return _identity_groups(heavy_seqs)
    elif strategy == 'edit_distance':
        # Group by edit distance from wildtype (preserves distribution across folds)
        ed = data.get('edit_distances')
        if ed is not None:
            groups = np.asarray(ed, dtype=int)
            return groups, len(np.unique(groups))
        # Fallback
        return _identity_groups(heavy_seqs)
    elif strategy == 'complex':
        from kirby.datasets.ddg_combined import assign_complex_groups
        pdb_ids = data.get('pdb_ids', [])
        return assign_complex_groups(pdb_ids)
    elif strategy == 'experiment':
        # For abagym_all: group by experiment name from mutation metadata.
        # Each mutation dict has an 'experiment' key set by the loader.
        mutations = data.get('mutations', [])
        if mutations and mutations[0] and 'experiment' in mutations[0][0]:
            exp_labels = [m[0]['experiment'] for m in mutations]
            unique = sorted(set(exp_labels))
            label_to_id = {l: i for i, l in enumerate(unique)}
            groups = np.array([label_to_id[l] for l in exp_labels])
            return groups, len(unique)
        # Fallback: identity groups
        return _identity_groups(heavy_seqs)
    elif strategy == 'position_modulo':
        # DMS/SSM data: group by mutation position (ProteinGym standard).
        # Prevents same-position leakage where mutations at the same residue
        # appear in both train and test.
        mutations = data.get('mutations')
        if mutations and any(len(m) > 0 for m in mutations):
            groups, n_groups = _assign_position_groups_modulo(mutations)
            if n_groups >= 3:
                return groups, n_groups
            # If too few position groups (very short protein), fall back
            print("    WARNING: <3 position groups, falling back to identity")
        else:
            print("    WARNING: no mutation data, falling back to identity")
        return _identity_groups(heavy_seqs)
    else:  # identity
        return _identity_groups(heavy_seqs)


def _rep_cache_path(heavy_seqs: List[str], light_seqs: List[str],
                    rep_name: str) -> Path:
    """Get cache file path for a representation, keyed by sequence hash.

    Respects KIRBY_CACHE_DIR env var (set by setup.sh on the server).
    Falls back to ~/.cache/kirby.
    """
    import hashlib
    content = '\n'.join(heavy_seqs) + '\n\n' + '\n'.join(light_seqs)
    data_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    base = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))
    cache_dir = base / 'reps' / f'ab_{data_hash}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{rep_name}.npy'


def _identity_groups(seqs, min_groups=3):
    """Assign CV groups by sequence NEAR-identity (ISS-29 leakage fix).

    Was exact-byte-identity grouping, which put single-residue analogs into
    DIFFERENT groups → near-duplicate pairs leaked across the GroupKFold
    train/test boundary (measured: 87% of near-dup pairs spanned a fold on
    flab_thermo_tresanco). Now delegates to sequence_identity_clusters, which
    keeps any pair with normalised identity >= 0.95 in the same cluster
    (single-linkage), so near-duplicates stay on one side of every fold. The
    helper keeps the same min_groups hash-bucket fallback for degenerate panels.
    NOTE (follow-up, NEW issue): for paired datasets this still receives the
    heavy chain only; pass heavy+light concatenated to also catch
    light-chain-only near-duplicates.
    """
    from experiments._shared import sequence_identity_clusters
    return sequence_identity_clusters(seqs, identity_threshold=0.95,
                                      min_groups=min_groups)


def _assign_position_groups_modulo(mutations, n_folds=5):
    """ProteinGym-style modulo split for DMS data.

    Assigns each sample to a fold based on (mutation_position % n_folds).
    Ensures mutations at the same position are always in the same fold,
    preventing position-specific effect leakage.

    Reference: ProteinGym (NeurIPS 2023) fold_modulo_5 scheme.
    """
    groups = np.zeros(len(mutations), dtype=int)
    for i, muts in enumerate(mutations):
        if muts:
            # Use first mutation position (SSM data has 1 mutation per variant)
            groups[i] = muts[0]['pos'] % n_folds
        else:
            # Wildtype: assign to fold 0
            groups[i] = 0
    n_groups = len(np.unique(groups))
    return groups, n_groups


def _assign_position_groups_contiguous(mutations, n_folds=5):
    """ProteinGym-style contiguous split for DMS data.

    Splits the protein into n_folds contiguous segments by position.
    Tests whether models can extrapolate to unseen sequence regions.

    Reference: ProteinGym (NeurIPS 2023) fold_contiguous_5 scheme.
    """
    all_pos = [muts[0]['pos'] for muts in mutations if muts]
    if not all_pos:
        return np.zeros(len(mutations), dtype=int), 1

    max_pos = max(all_pos) + 1
    segment_size = max_pos / n_folds

    groups = np.zeros(len(mutations), dtype=int)
    for i, muts in enumerate(mutations):
        if muts:
            groups[i] = min(int(muts[0]['pos'] / segment_size), n_folds - 1)
        else:
            groups[i] = 0
    n_groups = len(np.unique(groups))
    return groups, n_groups


# =============================================================================
# MODELS
# =============================================================================

def get_models(model_names: List[str], n_samples: int = None,
               task_type: str = 'regression') -> Dict[str, object]:
    """Return model instances by name, using centralized DEFAULT_MODELS.

    Forces n_jobs=1 for antibody datasets (small, no benefit from parallelism).
    """
    from kirby.find_best_hybrid import get_model

    catalog = {}
    for name in model_names:
        try:
            m = get_model(name, task_type=task_type, n_samples=n_samples)
        except ValueError:
            print(f"  WARNING: unknown model '{name}', skipping")
            continue
        if m is None:
            continue  # skipped by size guard
        # Force n_jobs=1 for antibody (small datasets, avoid overhead)
        if hasattr(m, 'n_jobs'):
            m.set_params(n_jobs=1)
        catalog[name] = m

    return catalog


# =============================================================================
# EVALUATION
# =============================================================================

def _is_binary_rep(X):
    """Check if feature matrix is binary (0/1) — should NOT be StandardScaled."""
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


def _nan_clf_result():
    return {'accuracy_mean': float('nan'), 'accuracy_std': float('nan'),
            'f1_weighted_mean': float('nan'), 'f1_weighted_std': float('nan'),
            'auroc_mean': float('nan'), 'auroc_std': float('nan'),
            'per_fold': []}


def _nan_reg_result():
    return {'r2_mean': float('nan'), 'r2_std': float('nan'),
            'spearman_mean': float('nan'), 'spearman_std': float('nan'),
            'pearson_mean': float('nan'), 'pearson_std': float('nan'),
            'rmse_mean': float('nan'), 'rmse_std': float('nan'),
            'per_fold': []}


def evaluate_rep(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model,
    n_folds: int = 5,
    task_type: str = 'regression',
) -> Dict[str, float]:
    """Evaluate one (rep, model) pair via GroupKFold.

    Returns dict with metric means/stds and per_fold details.
    For regression: r2, spearman, pearson, rmse.
    For classification: accuracy, f1_weighted, auroc (binary only).
    """
    is_clf = (task_type == 'classification')
    X = np.asarray(X, dtype=np.float64)
    if is_clf:
        labels = np.asarray(labels, dtype=np.int64)
    else:
        labels = np.asarray(labels, dtype=np.float64)

    n_unique = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique)

    if actual_folds < 2:
        return _nan_clf_result() if is_clf else _nan_reg_result()

    # Use StratifiedGroupKFold for classification to preserve class balance
    if is_clf and n_unique >= actual_folds:
        cv = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True,
                                  random_state=42)
        splits = list(cv.split(X, labels, groups=groups))
    elif n_unique >= actual_folds:
        cv = GroupKFold(n_splits=actual_folds)
        splits = list(cv.split(X, labels, groups=groups))
    else:
        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        splits = list(cv.split(X, labels))

    # Metric accumulators
    if is_clf:
        acc_scores, f1_scores, auroc_scores = [], [], []
    else:
        r2_scores, spearman_scores, pearson_scores, rmse_scores = [], [], [], []
    fold_details = []

    # Only z-score continuous features; leave binary reps (onehot, blosum) raw
    is_binary = _is_binary_rep(X)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        if is_binary:
            X_train = X[train_idx].astype(np.float32)
            X_val = X[val_idx].astype(np.float32)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e10, neginf=-1e10)

        try:
            m = clone(model)
            m.fit(X_train, labels[train_idx])
            y_pred = m.predict(X_val)

            if is_clf:
                acc = float(accuracy_score(labels[val_idx], y_pred))
                f1 = float(f1_score(labels[val_idx], y_pred,
                                    average='weighted', zero_division=0))
                acc_scores.append(acc)
                f1_scores.append(f1)
                fold_info = {
                    'fold': fold_i, 'accuracy': round(acc, 4),
                    'f1_weighted': round(f1, 4),
                    'n_train': len(train_idx), 'n_val': len(val_idx),
                }
                # AUROC for binary only, requires predict_proba
                n_classes = len(np.unique(labels))
                if n_classes == 2 and hasattr(m, 'predict_proba'):
                    try:
                        y_prob = m.predict_proba(X_val)[:, 1]
                        auc = float(roc_auc_score(labels[val_idx], y_prob))
                        auroc_scores.append(auc)
                        fold_info['auroc'] = round(auc, 4)
                    except (ValueError, IndexError):
                        pass
                fold_details.append(fold_info)
            else:
                r2 = r2_score(labels[val_idx], y_pred)
                rho = float(spearmanr(labels[val_idx], y_pred)[0])
                pr = float(pearsonr(labels[val_idx], y_pred)[0])
                rmse_val = float(np.sqrt(mean_squared_error(
                    labels[val_idx], y_pred)))
                r2_scores.append(r2)
                spearman_scores.append(rho)
                pearson_scores.append(pr)
                rmse_scores.append(rmse_val)
                fold_details.append({
                    'fold': fold_i,
                    'r2': round(float(r2), 4),
                    'spearman': round(rho, 4),
                    'pearson': round(pr, 4),
                    'rmse': round(rmse_val, 4),
                    'n_train': len(train_idx),
                    'n_val': len(val_idx),
                })
        except Exception as e:
            fold_details.append({'fold': fold_i, 'error': str(e)})

    if is_clf:
        if not acc_scores:
            result = _nan_clf_result()
            result['per_fold'] = fold_details
            return result
        result = {
            'accuracy_mean': round(float(np.mean(acc_scores)), 4),
            'accuracy_std': round(float(np.std(acc_scores)), 4),
            'f1_weighted_mean': round(float(np.mean(f1_scores)), 4),
            'f1_weighted_std': round(float(np.std(f1_scores)), 4),
            'per_fold': fold_details,
        }
        if auroc_scores:
            result['auroc_mean'] = round(float(np.mean(auroc_scores)), 4)
            result['auroc_std'] = round(float(np.std(auroc_scores)), 4)
        else:
            result['auroc_mean'] = float('nan')
            result['auroc_std'] = float('nan')
        return result
    else:
        if not r2_scores:
            result = _nan_reg_result()
            result['per_fold'] = fold_details
            return result
        return {
            'r2_mean': round(float(np.mean(r2_scores)), 4),
            'r2_std': round(float(np.std(r2_scores)), 4),
            'spearman_mean': round(float(np.mean(spearman_scores)), 4),
            'spearman_std': round(float(np.std(spearman_scores)), 4),
            'pearson_mean': round(float(np.mean(pearson_scores)), 4),
            'pearson_std': round(float(np.std(pearson_scores)), 4),
            'rmse_mean': round(float(np.mean(rmse_scores)), 4),
            'rmse_std': round(float(np.std(rmse_scores)), 4),
            'per_fold': fold_details,
        }


# =============================================================================
# SURVEY
# =============================================================================

def evaluate_ft_rep(factory, labels, groups, model, n_folds=5,
                    task_type='regression'):
    """Evaluate a fine-tuning factory via CV. Returns dict with metric means/stds."""
    is_clf = (task_type == 'classification')
    n_unique = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique)

    if actual_folds < 2:
        return _nan_clf_result() if is_clf else _nan_reg_result()

    if is_clf and n_unique >= actual_folds:
        cv = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True,
                                  random_state=42)
        splits = list(cv.split(np.zeros(len(labels)), labels, groups=groups))
    elif n_unique >= actual_folds:
        cv = GroupKFold(n_splits=actual_folds)
        splits = list(cv.split(np.zeros(len(labels)), labels, groups=groups))
    else:
        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        splits = list(cv.split(np.zeros(len(labels)), labels))

    if is_clf:
        acc_scores, f1_scores, auroc_scores = [], [], []
    else:
        r2_scores, spearman_scores, pearson_scores, rmse_scores = [], [], [], []
    fold_details = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        try:
            X_train, X_val = factory(train_idx, val_idx, labels[train_idx])
        except Exception as e:
            fold_details.append({'fold': fold_i, 'error': str(e)})
            continue

        # FT reps are always continuous (LM embeddings) — always scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e10, neginf=-1e10)

        try:
            m = clone(model)
            m.fit(X_train, labels[train_idx])
            y_pred = m.predict(X_val)

            if is_clf:
                acc = float(accuracy_score(labels[val_idx], y_pred))
                f1 = float(f1_score(labels[val_idx], y_pred,
                                    average='weighted', zero_division=0))
                acc_scores.append(acc)
                f1_scores.append(f1)
                fold_info = {
                    'fold': fold_i, 'accuracy': round(acc, 4),
                    'f1_weighted': round(f1, 4),
                    'n_train': len(train_idx), 'n_val': len(val_idx),
                }
                n_classes = len(np.unique(labels))
                if n_classes == 2 and hasattr(m, 'predict_proba'):
                    try:
                        y_prob = m.predict_proba(X_val)[:, 1]
                        auc = float(roc_auc_score(labels[val_idx], y_prob))
                        auroc_scores.append(auc)
                        fold_info['auroc'] = round(auc, 4)
                    except (ValueError, IndexError):
                        pass
                fold_details.append(fold_info)
            else:
                r2 = r2_score(labels[val_idx], y_pred)
                rho = float(spearmanr(labels[val_idx], y_pred)[0])
                pr = float(pearsonr(labels[val_idx], y_pred)[0])
                rmse_val = float(np.sqrt(mean_squared_error(
                    labels[val_idx], y_pred)))
                r2_scores.append(r2)
                spearman_scores.append(rho)
                pearson_scores.append(pr)
                rmse_scores.append(rmse_val)
                fold_details.append({
                    'fold': fold_i, 'r2': round(float(r2), 4),
                    'spearman': round(rho, 4), 'pearson': round(pr, 4),
                    'rmse': round(rmse_val, 4),
                    'n_train': len(train_idx), 'n_val': len(val_idx),
                })
        except Exception as e:
            fold_details.append({'fold': fold_i, 'error': str(e)})

    if is_clf:
        if not acc_scores:
            result = _nan_clf_result()
            result['per_fold'] = fold_details
            return result
        result = {
            'accuracy_mean': round(float(np.mean(acc_scores)), 4),
            'accuracy_std': round(float(np.std(acc_scores)), 4),
            'f1_weighted_mean': round(float(np.mean(f1_scores)), 4),
            'f1_weighted_std': round(float(np.std(f1_scores)), 4),
            'per_fold': fold_details,
        }
        if auroc_scores:
            result['auroc_mean'] = round(float(np.mean(auroc_scores)), 4)
            result['auroc_std'] = round(float(np.std(auroc_scores)), 4)
        else:
            result['auroc_mean'] = float('nan')
            result['auroc_std'] = float('nan')
        return result
    else:
        if not r2_scores:
            result = _nan_reg_result()
            result['per_fold'] = fold_details
            return result
        return {
            'r2_mean': round(float(np.mean(r2_scores)), 4),
            'r2_std': round(float(np.std(r2_scores)), 4),
            'spearman_mean': round(float(np.mean(spearman_scores)), 4),
            'spearman_std': round(float(np.std(spearman_scores)), 4),
            'pearson_mean': round(float(np.mean(pearson_scores)), 4),
            'pearson_std': round(float(np.std(pearson_scores)), 4),
            'rmse_mean': round(float(np.mean(rmse_scores)), 4),
            'rmse_std': round(float(np.std(rmse_scores)), 4),
            'per_fold': fold_details,
        }


def run_survey(
    dataset_name: str,
    rep_names: List[str],
    model_names: List[str],
    n_folds: int = 5,
    augment: bool = False,
    do_finetune: bool = False,
) -> dict:
    """Run baseline survey for one dataset. Returns result dict."""
    print(f"\n{'='*70}")
    print(f"SURVEY: {dataset_name}")
    print(f"{'='*70}")
    t0 = time.time()

    # 1. Load data
    data = load_dataset(dataset_name)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    is_clf = (task_type == 'classification')

    print(f"  {data['n_samples']} samples, {data['n_groups']} groups, "
          f"task={task_type}")

    # 2. Generate representations (with .npy caching)
    # esm2_entropy_top20 needs per-fold fitting (entropy on train only)
    FACTORY_REPS = {'esm2_entropy_top20'}

    from kirby.representations.antibody import ANTIBODY_REPS
    from kirby.representations.mutation import MUTATION_REPS
    reps = {}
    for rep_name in rep_names:
        if rep_name in FACTORY_REPS:
            continue  # handled separately below
        if rep_name not in ANTIBODY_REPS:
            if rep_name not in MUTATION_REPS:
                print(f"  WARNING: unknown rep '{rep_name}', skipping")
            continue

        # Try loading from cache
        cache_file = _rep_cache_path(heavy_seqs, light_seqs, rep_name)
        if cache_file.exists():
            try:
                X = np.load(cache_file, allow_pickle=False)
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 2:
                    print(f"  {rep_name}: loaded from cache ({X.shape[1]}D)")
                    reps[rep_name] = X
                    continue
            except Exception:
                pass  # corrupted cache, recompute

        try:
            print(f"  Generating {rep_name}...", end='', flush=True)
            X = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
            if X.ndim != 2:
                print(f" FAILED: expected 2D array, got {X.ndim}D {X.shape}")
                continue
            reps[rep_name] = X
            print(f" ({X.shape[1]}D)")
            # Save to cache
            np.save(cache_file, X)
        except Exception as e:
            print(f" FAILED: {e}")

    # 2b. Mutation-aware reps (require mutation metadata)
    mutations = data.get('mutations')
    has_mutations = mutations and any(len(m) > 0 for m in mutations)
    if has_mutations:
        for rep_name in rep_names:
            if rep_name in MUTATION_REPS and rep_name not in reps:
                cache_file = _rep_cache_path(heavy_seqs, light_seqs, rep_name)
                if cache_file.exists():
                    try:
                        X = np.load(cache_file, allow_pickle=False)
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 2:
                            print(f"  {rep_name}: loaded from cache ({X.shape[1]}D)")
                            reps[rep_name] = X
                            continue
                    except Exception:
                        pass
                try:
                    print(f"  Generating {rep_name}...", end='', flush=True)
                    X = MUTATION_REPS[rep_name](heavy_seqs, light_seqs, mutations)
                    if X.ndim != 2:
                        print(f" FAILED: expected 2D, got {X.ndim}D")
                        continue
                    reps[rep_name] = X
                    print(f" ({X.shape[1]}D)")
                    np.save(cache_file, X)
                except Exception as e:
                    print(f" FAILED: {e}")

    if not reps:
        print("  WARNING: No new representations generated in this run")
        # Return empty result — main() will merge with existing cached results
        return {
            'dataset': dataset_name,
            'baselines': {},
            'metadata': {'n_samples': data['n_samples'], 'task': task_type},
        }

    # 3. Classify reps by tier
    from experiments._shared import classify_rep_tier
    rep_tiers = {}
    for rep_name, X in reps.items():
        try:
            dim = X.shape[1] if hasattr(X, 'shape') else np.asarray(X).shape[1]
            rep_tiers[rep_name] = classify_rep_tier(dim, rep_name)
        except Exception:
            rep_tiers[rep_name] = 3

    tier_1_reps = sorted(k for k, t in rep_tiers.items() if t == 1)
    tier_2_reps = sorted(k for k, t in rep_tiers.items() if t == 2)
    tier_3_reps = sorted(k for k, t in rep_tiers.items() if t == 3)
    baseline_rep_names = sorted(set(tier_2_reps) | set(tier_3_reps))

    if tier_1_reps:
        print(f"  Tier 1 (augmentation-only): {tier_1_reps}")
    if tier_2_reps:
        print(f"  Tier 2 (standalone + augmentation): {tier_2_reps}")
    print(f"  Tier 3 (standalone only): {len(tier_3_reps)} reps")

    # 4. Get models
    models = get_models(model_names, n_samples=data['n_samples'],
                        task_type=task_type)
    print(f"  Models: {list(models.keys())}")

    # 5. Evaluate each (rep, model) cell — tier 2 + tier 3 only
    baselines = {}
    for rep_name in baseline_rep_names:
        X = reps[rep_name]
        try:
            X = np.asarray(X, dtype=np.float64)
        except Exception:
            print(f"  {rep_name}: SKIPPED (array conversion failed)")
            continue
        dim = X.shape[1]
        print(f"  {rep_name} ({dim}D):", end='', flush=True)

        try:
            entry = {'dimension': dim, 'scores': {}}
            for mname, model in models.items():
                result = evaluate_rep(X, labels, groups, model, n_folds,
                                      task_type=task_type)
                entry['scores'][mname] = result
                if is_clf:
                    acc = result['accuracy_mean']
                    f1 = result['f1_weighted_mean']
                    print(f" {mname}=acc:{acc:.3f}/F1:{f1:.3f}",
                          end='', flush=True)
                else:
                    r2 = result['r2_mean']
                    rho = result['spearman_mean']
                    print(f" {mname}={r2:.3f}/\u03c1={rho:.3f}",
                          end='', flush=True)

            baselines[rep_name] = entry
            print()
        except Exception as e:
            print(f" EVAL FAILED: {e}")

    # 5b. Factory-based reps (per-fold fitting to prevent leakage)
    factory_rep_names = [r for r in rep_names if r in FACTORY_REPS]
    if factory_rep_names:
        from kirby.representations.antibody import make_esm2_entropy_factory

        factory_map = {
            'esm2_entropy_top20': lambda: make_esm2_entropy_factory(
                heavy_seqs, light_seqs),
        }

        for rep_name in factory_rep_names:
            if rep_name not in factory_map:
                continue
            try:
                print(f"  {rep_name} (per-fold):", end='', flush=True)
                factory = factory_map[rep_name]()
                entry = {'dimension': 2560, 'scores': {}}
                for mname, model in models.items():
                    result = evaluate_ft_rep(factory, labels, groups, model,
                                             n_folds, task_type=task_type)
                    entry['scores'][mname] = result
                    if is_clf:
                        acc = result['accuracy_mean']
                        f1 = result['f1_weighted_mean']
                        print(f" {mname}=acc:{acc:.3f}/F1:{f1:.3f}",
                              end='', flush=True)
                    else:
                        r2 = result['r2_mean']
                        rho = result['spearman_mean']
                        print(f" {mname}={r2:.3f}/\u03c1={rho:.3f}",
                              end='', flush=True)
                baselines[rep_name] = entry
                print()
            except Exception as e:
                print(f" FAILED: {e}")

    # 6. Fine-tuned reps (ESM-2 LoRA)
    ft_baselines = {}
    if do_finetune:
        try:
            from kirby.finetuning.antibody_lora import make_antibody_lora_factory
            factory = make_antibody_lora_factory(
                heavy_seqs=heavy_seqs, light_seqs=light_seqs,
                dataset_name=dataset_name, base_model='esm2',
            )
            ft_name = 'esm2_lora'
            print(f"\n--- Fine-tuned reps: [{ft_name}] ---")
            print(f"  {ft_name}:", end='', flush=True)

            entry = {'dimension': 1280, 'scores': {}}
            for mname, model in models.items():
                result = evaluate_ft_rep(factory, labels, groups, model, n_folds,
                                         task_type=task_type)
                entry['scores'][mname] = result
                if is_clf:
                    acc = result['accuracy_mean']
                    f1 = result['f1_weighted_mean']
                    print(f" {mname}=acc:{acc:.3f}/F1:{f1:.3f}",
                          end='', flush=True)
                else:
                    r2 = result['r2_mean']
                    rho = result['spearman_mean']
                    print(f" {mname}={r2:.3f}/\u03c1={rho:.3f}",
                          end='', flush=True)

            ft_baselines[ft_name] = entry
            print()
        except Exception as e:
            print(f"\n  esm2_lora factory failed: {e}")

    # 7. Augmentation test: concat tier-1 + tier-2 reps with top 2 tier-3 base reps
    augmentations = {}
    if augment and baselines:
        primary_key = 'accuracy_mean' if is_clf else 'r2_mean'

        # Candidates for augmentation: tier 1 + tier 2 reps (must have raw matrix)
        aug_candidates = {k: reps[k] for k in sorted(reps.keys())
                          if k in reps and rep_tiers.get(k, 3) in (1, 2)}

        # Find top 2 tier-3 reps by best model score
        tier3_scored = []
        for name, e in baselines.items():
            if name not in reps:
                continue  # factory/per-fold rep — no raw matrix available
            if rep_tiers.get(name, 3) != 3:
                continue
            for mname, s in e['scores'].items():
                val = s.get(primary_key, float('nan'))
                if not np.isnan(val):
                    tier3_scored.append((name, mname, val))

        if tier3_scored and aug_candidates:
            # Sort by score (higher is better for both accuracy and R²)
            tier3_sorted = sorted(tier3_scored, key=lambda x: x[2], reverse=True)
            # Deduplicate: keep only the best model per rep
            seen = set()
            top_bases = []
            for name, mname, score in tier3_sorted:
                if name not in seen:
                    seen.add(name)
                    top_bases.append((name, mname, score))
                if len(top_bases) >= 2:
                    break

            metric_label = 'Acc' if is_clf else 'R²'
            print(f"\n--- Augmentation test: {len(top_bases)} tier-3 base reps × "
                  f"{len(aug_candidates)} tier-1/2 reps ---")

            for base_name, base_model_name, base_score in top_bases:
                X_base = reps[base_name]
                base_model = models[base_model_name]

                print(f"\n  Base: {base_name} ({metric_label}={base_score:.4f}, "
                      f"model={base_model_name})")

                for aug_name in sorted(aug_candidates.keys()):
                    X_aug = np.hstack([X_base, aug_candidates[aug_name]])
                    result = evaluate_rep(X_aug, labels, groups, base_model,
                                          n_folds, task_type=task_type)
                    mean_s = result[primary_key]
                    delta = mean_s - base_score
                    sign = '+' if delta >= 0 else ''
                    print(f"    +{aug_name} ({X_aug.shape[1]}d): "
                          f"{mean_s:.4f} ({sign}{delta:.4f})")
                    augmentations[f"{base_name}+{aug_name}"] = {
                        'base': base_name,
                        'augment': aug_name,
                        'base_dim': int(X_base.shape[1]),
                        'aug_dim': int(aug_candidates[aug_name].shape[1]),
                        'concat_dim': X_aug.shape[1],
                        'model': base_model_name,
                        'base_score': round(base_score, 4),
                        'concat_score': round(float(mean_s), 4),
                        'delta': round(float(delta), 4),
                        'base_tier': 3,
                        'aug_tier': rep_tiers.get(aug_name, 1),
                    }

    elapsed = time.time() - t0

    return {
        'metadata': {
            'dataset': dataset_name,
            'n_samples': data['n_samples'],
            'n_groups': data['n_groups'],
            'n_folds': n_folds,
            'task_type': task_type,
            'reps': sorted(reps.keys()),
            'rep_tiers': {k: v for k, v in rep_tiers.items()},
            'models': list(models.keys()),
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        },
        'baselines': baselines,
        'finetuned_baselines': ft_baselines or None,
        'augmentations': augmentations or None,
    }


# =============================================================================
# OUTPUT
# =============================================================================

def _convert_numpy(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    return obj


def print_summary_table(result):
    """Print compact summary table."""
    meta = result['metadata']
    baselines = result['baselines']
    models = meta.get('models', [])
    is_clf = meta.get('task_type') == 'classification'

    print(f"\n{'='*70}")
    print(f"SUMMARY: {meta['dataset']} ({meta['n_samples']} samples, "
          f"{meta['n_groups']} groups, {meta['n_folds']}-fold, "
          f"task={meta['task_type']})")
    print(f"{'='*70}")

    # Header
    header = f"{'Rep':<25} {'Dim':>5}"
    for m in models:
        if is_clf:
            col_label = f"Acc/F1 ({m})"
        else:
            col_label = f"R\u00b2/\u03c1 ({m})"
        header += f" {col_label:>18}"
    print(header)
    print("-" * len(header))

    for rep_name in sorted(baselines.keys()):
        r = baselines[rep_name]
        row = f"{rep_name:<25} {r['dimension']:>5}"
        for m in models:
            s = r['scores'].get(m, {})
            if is_clf:
                acc = s.get('accuracy_mean', float('nan'))
                f1 = s.get('f1_weighted_mean', float('nan'))
                row += f" {acc:>8.4f}/{f1:.4f}"
            else:
                r2 = s.get('r2_mean', float('nan'))
                rho = s.get('spearman_mean', float('nan'))
                row += f" {r2:>8.4f}/{rho:.4f}"
        print(row)

    # Fine-tuned scores
    ft = result.get('finetuned_baselines')
    if ft:
        print()
        print("FINE-TUNED:")
        print("-" * len(header))
        for ft_name in sorted(ft.keys()):
            r = ft[ft_name]
            dim = r.get('dimension', '?')
            row = f"{ft_name:<25} {dim:>5}"
            for m in models:
                s = r['scores'].get(m, {})
                if is_clf:
                    acc = s.get('accuracy_mean', float('nan'))
                    f1 = s.get('f1_weighted_mean', float('nan'))
                    row += f" {acc:>8.4f}/{f1:.4f}"
                else:
                    r2 = s.get('r2_mean', float('nan'))
                    rho = s.get('spearman_mean', float('nan'))
                    row += f" {r2:>8.4f}/{rho:.4f}"
            print(row)

    print(f"\n  Time: {meta['elapsed_seconds']:.0f}s")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Antibody baseline survey: LM embeddings + Ridge/RF/XGBoost + R2.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  flab_expression_koenig, flab_binding_koenig       (FLAb DMS)
  abagym_Ang2/EGFR/lysozyme/VEGF/HER2              (AbAgym DMS)
  ddg_combined, abagym_all, abcov_kd,               (inter-family)
    flab_thermo_jain, flab_aggregation_jain
  ddg_combined_full                                 (experiment: full SKEMPI PP)
  tz_her2_binary                                    (classification: binder/non-binder)
  tz_her2_multiclass                                (classification: low/mid/high)
  humanness_species                                 (classification: human vs non-human VH)
  humanness_therapeutic                             (classification: human vs non-human therapeutics)
  ginkgo_hic/tm2/titer/pr_cho/acsins               (Ginkgo developability, 5 core)
  ginkgo_purity/aggregation/colloidal/heparin       (Ginkgo developability, 4 additional)
  graphinity_foldx/rosetta/augmented                (Graphinity synthetic DDG)

Shortcuts:
  all_dms       Run all 10 DMS datasets
  all_inter     Run all 14 inter-family datasets
  all_clf       Run all classification datasets
  all_ginkgo    Run all 9 Ginkgo developability properties
  all_ddg       ddg_combined + ddg_combined_full + graphinity
  all           Run everything (excludes ddg_combined_full by default)

DDG experiment flags:
  --include-all-skempi    Add ddg_combined_full to any multi-dataset run.
                          Compares Ab-Ag-only vs all-PP SKEMPI on same reps.

Examples:
  %(prog)s flab_expression_koenig --skip-lm
  %(prog)s all_dms --models ridge,rf
  %(prog)s all_ginkgo --models ridge,rf --skip-lm
  %(prog)s humanness_species --models ridge,rf
  %(prog)s graphinity_foldx --models ridge --skip-lm
  %(prog)s all --include-all-skempi --output tests/results/antibody_survey/
        """
    )

    parser.add_argument('dataset', type=str,
                        help='Dataset name, group shortcut, or "all"')
    parser.add_argument('--include-all-skempi', action='store_true',
                        help='Add ddg_combined_full to the dataset list. '
                             'Uses all SKEMPI protein-protein entries (not just Ab-Ag). '
                             'Equivalent to adding ddg_combined_full to the dataset list. '
                             'Only affects multi-dataset runs (all, all_inter, etc.).')
    parser.add_argument('--skip-lm', action='store_true',
                        help='Skip LM reps (only one-hot, blosum, aac)')
    parser.add_argument('--reps', type=str, default=None,
                        help='Comma-separated rep names to use (overrides --skip-lm)')
    parser.add_argument('--models', type=str,
                        default='ridge,rf,xgboost,lightgbm,mlp',
                        help='Comma-separated model names (default: 5 survey models)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--augment', action='store_true',
                        help='Test augmentation: concat each small rep (<=200D) with top base rep')
    parser.add_argument('--finetune', action='store_true',
                        help='Include fine-tuned reps (ESM-2 LoRA on VH, GPU required)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (JSON file or directory for multi-dataset)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip reps already in the output JSON (incremental mode)')

    args = parser.parse_args()

    # Resolve datasets
    ds_input = args.dataset
    if ds_input in DOMAIN_GROUPS:
        datasets = list(DOMAIN_GROUPS[ds_input])
    elif ds_input in DATASET_REGISTRY:
        datasets = [ds_input]
    else:
        print(f"ERROR: Unknown dataset '{ds_input}'")
        print(f"Available: {sorted(DATASET_REGISTRY.keys())}")
        print(f"Groups: {sorted(DOMAIN_GROUPS.keys())}")
        return 1

    # --include-all-skempi: inject ddg_combined_full if not already present
    if getattr(args, 'include_all_skempi', False):
        if 'ddg_combined_full' not in datasets:
            # Insert right after ddg_combined if present, else append
            if 'ddg_combined' in datasets:
                idx = datasets.index('ddg_combined')
                datasets.insert(idx + 1, 'ddg_combined_full')
            else:
                datasets.append('ddg_combined_full')
            print(f"  [--include-all-skempi] Added ddg_combined_full to dataset list")

    # Resolve reps
    from kirby.representations.antibody import (
        ANTIBODY_REPS, ANTIBODY_LM_REPS, ANTIBODY_FAST_REPS)

    if args.reps and args.reps.strip().lower() != 'all':
        rep_names = [r.strip() for r in args.reps.split(',')]
    elif args.skip_lm:
        rep_names = sorted(ANTIBODY_FAST_REPS)
    else:
        rep_names = sorted(ANTIBODY_REPS.keys())

    model_names = [m.strip() for m in args.models.split(',')]

    print(f"Antibody Baseline Survey")
    print(f"  Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {model_names}")
    print(f"  Folds: {args.n_folds}")

    # Output handling
    output_dir = None
    output_file = None
    if args.output:
        p = Path(args.output)
        if p.suffix == '.json':
            output_file = p
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = p
            output_dir.mkdir(parents=True, exist_ok=True)

    # Run surveys
    all_results = {}
    pass_count = 0
    fail_count = 0

    for ds in datasets:
        # --skip-existing: filter out reps already in the output JSON
        _run_reps = list(rep_names)
        if args.skip_existing and output_dir:
            existing_path = output_dir / f"{ds}.json"
            if existing_path.exists():
                try:
                    with open(existing_path) as f:
                        _existing_data = json.load(f)
                    _existing_reps = set(_existing_data.get('baselines', {}).keys())
                    _run_reps = [r for r in rep_names if r not in _existing_reps]
                    if len(_run_reps) < len(rep_names):
                        print(f"  [{ds}] Skipping {len(rep_names) - len(_run_reps)} existing reps, {len(_run_reps)} remaining")
                    if not _run_reps:
                        print(f"  [{ds}] All reps already done, skipping entirely")
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass

        try:
            result = run_survey(ds, _run_reps, model_names, args.n_folds,
                                augment=args.augment,
                                do_finetune=args.finetune)
            result = _convert_numpy(result)
            print_summary_table(result)
            all_results[ds] = result
            pass_count += 1

            if output_dir:
                out_path = output_dir / f"{ds}.json"
                if out_path.exists():
                    try:
                        with open(out_path) as f:
                            existing = json.load(f)
                        for section in ('baselines', 'finetuned_baselines', 'augmentations'):
                            if section in result and result[section]:
                                if section not in existing or not existing[section]:
                                    existing[section] = {}
                                existing[section].update(result[section])
                        if 'metadata' in result:
                            existing.setdefault('metadata', {}).update(result['metadata'])
                        result = existing
                        print(f"  Merged with existing: {out_path}")
                    except (json.JSONDecodeError, KeyError):
                        pass
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"\n  FAILED: {ds}: {e}")
            traceback.print_exc()
            fail_count += 1

    # Save single-file output
    if output_file and all_results:
        payload = (list(all_results.values())[0]
                   if len(all_results) == 1 else all_results)
        with open(output_file, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved: {output_file}")

    # Cross-dataset summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-DATASET SUMMARY")
        print(f"{'='*70}")
        for ds, r in all_results.items():
            baselines = r.get('baselines', {})
            ds_clf = r.get('metadata', {}).get('task_type') == 'classification'
            if baselines:
                print(f"\n  {ds}:")
                for rep_name in sorted(baselines.keys()):
                    scores = baselines[rep_name]['scores']
                    parts = []
                    for mname, s in scores.items():
                        if ds_clf:
                            acc = s.get('accuracy_mean', float('nan'))
                            f1 = s.get('f1_weighted_mean', float('nan'))
                            parts.append(f"{mname}=acc:{acc:.4f}/F1:{f1:.4f}")
                        else:
                            rho = s.get('spearman_mean', float('nan'))
                            parts.append(
                                f"{mname}={s['r2_mean']:.4f}/\u03c1={rho:.4f}")
                    print(f"    {rep_name:<20} {', '.join(parts)}")
            else:
                print(f"  {ds:<30} FAILED")

    print(f"\nDone: {pass_count} passed, {fail_count} failed")
    return 1 if fail_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
