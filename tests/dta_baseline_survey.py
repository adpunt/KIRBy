#!/usr/bin/env python3
"""
DTA Baseline Survey — Per-Rep R² for Bill of Materials

Evaluates each drug and protein representation individually on Davis and KIBA
using cold-target GroupKFold CV. Outputs the numbers needed for BOM §2D.

For DTA, each rep is tested in two modes:
  - standalone: just that rep's features (drug-only or protein-only)
  - paired: drug rep + protein_aac, or protein rep + drug_ecfp4 (minimal partner)

The "paired" mode is more realistic since you need both drug and protein info
to predict binding affinity.

Usage:
    python tests/dta_baseline_survey.py davis
    python tests/dta_baseline_survey.py kiba --max-pairs 30000
    python tests/dta_baseline_survey.py all
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from copy import deepcopy

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.find_best_hybrid import _concordance_index
warnings.filterwarnings('ignore')


# Which reps to evaluate (ordered: lightweight first, LM last)
DRUG_REPS = [
    'drug_ecfp4', 'drug_maccs', 'drug_pdv', 'drug_map4', 'drug_mol2vec',
    'drug_mordred', 'drug_avalon',
    'drug_chemberta', 'drug_molformer', 'drug_selformer', 'drug_clamp',
    'drug_rmat', 'drug_unimol_v2',
    'drug_minimol', 'drug_gp_molformer', 'drug_mole',
]

# Factory drug reps (need per-fold fitting, can't be pre-computed)
FACTORY_DRUG_REPS = ['drug_sns']

PROTEIN_REPS = [
    'protein_aac', 'protein_descriptors', 'protein_kmer',
    'protein_esm2', 'protein_esmc', 'protein_esm2_midlayers',
    'protein_amplify',
    'protein_mint', 'protein_esmfold', 'protein_alphafold_db',
    'protein_esm3', 'protein_saprot',  # Phase 4B (boltz2 removed: not viable for DTA)
    'protein_aido',
    'protein_xtrimopglm',
    'protein_contact_graph', 'protein_usr_shape',  # Structural
    'protein_pocket', 'protein_esm2_pocket', 'protein_pocket_composition',
]

# Default minimal partners for paired evaluation
DEFAULT_DRUG_PARTNER = 'drug_ecfp4'
DEFAULT_PROTEIN_PARTNER = 'protein_aac'


def _filter_features_by_importance(X: np.ndarray, y: np.ndarray,
                                    n_features: int = 200,
                                    rep_name: str = '') -> np.ndarray:
    """Reduce high-dim features via LightGBM feature importance.

    Fits a quick LightGBM on the full data to rank features, keeps top n_features.
    Slight label leakage but standard practice for tractability.
    """
    if X.shape[1] <= n_features:
        return X

    print(f"  Filtering {rep_name} from {X.shape[1]}D → {n_features}D via feature importance...")
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=50, max_depth=5, random_state=42,
                              n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        print(f"  WARNING: lightgbm not available, using RF for feature selection")
        model = RandomForestRegressor(n_estimators=50, max_depth=5,
                                      random_state=42, n_jobs=-1)

    # Clean input
    X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    model.fit(X_clean, y)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-n_features:]
    top_idx.sort()

    X_filtered = X[:, top_idx]
    print(f"  Kept {n_features} features (importance range: {importances[top_idx[0]]:.1f}–{importances[top_idx[-1]]:.1f})")
    return X_filtered


def load_dataset(name: str):
    if name.startswith('pdbbind'):
        from kirby.datasets.pdbbind import load_dataset as _load_pdbbind
        return _load_pdbbind(name)
    from kirby.datasets.dta import load_dataset as _load
    return _load(name)


def generate_rep(rep_name: str, unique_drugs, unique_proteins,
                 drug_to_idx, protein_to_idx,
                 drug_smiles, protein_seqs,
                 protein_ids=None) -> Optional[np.ndarray]:
    """Generate a single representation expanded to all pairs."""
    from kirby.representations import molecular as mol
    from kirby.representations import protein as prot

    dispatch = {
        'drug_ecfp4':      (mol.create_ecfp4,      'drug'),
        'drug_maccs':      (mol.create_maccs,       'drug'),
        'drug_pdv':        (mol.create_pdv,         'drug'),
        'drug_mol2vec':    (mol.create_mol2vec,     'drug'),
        'protein_aac':     (prot.create_protein_aac,         'protein'),
        'protein_descriptors': (prot.create_protein_descriptors, 'protein'),
        'protein_kmer':    (prot.create_protein_kmer,        'protein'),
    }

    # LM reps — only import if needed
    if rep_name == 'drug_chemberta':
        dispatch['drug_chemberta'] = (mol.create_chemberta, 'drug')
    elif rep_name == 'drug_molformer':
        dispatch['drug_molformer'] = (mol.create_molformer, 'drug')
    elif rep_name == 'drug_selformer':
        dispatch['drug_selformer'] = (mol.create_selformer, 'drug')
    elif rep_name == 'drug_clamp':
        dispatch['drug_clamp'] = (mol.create_clamp, 'drug')
    elif rep_name == 'protein_esm2':
        dispatch['protein_esm2'] = (prot.create_esm2_protein_embeddings, 'protein')
    elif rep_name == 'protein_esmc':
        dispatch['protein_esmc'] = (prot.create_esmc_protein_embeddings, 'protein')
    elif rep_name == 'protein_esm2_midlayers':
        dispatch['protein_esm2_midlayers'] = (prot.create_esm2_midlayers_protein_embeddings, 'protein')
    elif rep_name == 'drug_map4':
        dispatch['drug_map4'] = (mol.create_map4, 'drug')
    elif rep_name == 'drug_rmat':
        dispatch['drug_rmat'] = (mol.create_rmat, 'drug')
    elif rep_name == 'drug_unimol_v2':
        dispatch['drug_unimol_v2'] = (mol.create_unimol_v2, 'drug')
    elif rep_name == 'drug_mordred':
        dispatch['drug_mordred'] = (mol.create_mordred, 'drug')
    elif rep_name == 'drug_avalon':
        dispatch['drug_avalon'] = (mol.create_avalon, 'drug')
    elif rep_name == 'drug_minimol':
        dispatch['drug_minimol'] = (mol.create_minimol, 'drug')
    elif rep_name == 'drug_gp_molformer':
        dispatch['drug_gp_molformer'] = (mol.create_gp_molformer, 'drug')
    elif rep_name == 'drug_mole':
        dispatch['drug_mole'] = (mol.create_mole, 'drug')
    elif rep_name == 'protein_amplify':
        dispatch['protein_amplify'] = (prot.create_amplify_protein_embeddings, 'protein')
    elif rep_name == 'protein_mint':
        dispatch['protein_mint'] = (prot.create_mint_protein_embeddings, 'protein')
    elif rep_name == 'protein_esmfold':
        dispatch['protein_esmfold'] = (prot.create_esmfold_protein_features, 'protein')
    elif rep_name == 'protein_alphafold_db':
        # Resolve gene names → UniProt IDs so AlphaFold DB can fetch structures
        if protein_ids is not None:
            # Build unique protein ID list matching unique_proteins order
            seq_to_pid = {}
            for seq, pid in zip(protein_seqs, protein_ids):
                if seq not in seq_to_pid:
                    seq_to_pid[seq] = pid
            unique_pids = [seq_to_pid.get(seq) for seq in unique_proteins]
            uniprot_ids = prot._resolve_uniprot_ids(unique_pids)
            dispatch['protein_alphafold_db'] = (
                lambda seqs: prot.create_alphafold_db_features(seqs, uniprot_ids=uniprot_ids),
                'protein')
        else:
            dispatch['protein_alphafold_db'] = (prot.create_alphafold_db_features, 'protein')
    elif rep_name == 'protein_esm3':
        dispatch['protein_esm3'] = (prot.create_esm3_protein_embeddings, 'protein')
    elif rep_name == 'protein_saprot':
        dispatch['protein_saprot'] = (prot.create_saprot_protein_embeddings, 'protein')
    elif rep_name == 'protein_aido':
        dispatch['protein_aido'] = (prot.create_aido_protein_embeddings, 'protein')
    elif rep_name == 'protein_xtrimopglm':
        dispatch['protein_xtrimopglm'] = (prot.create_xtrimopglm_protein_embeddings, 'protein')
    elif rep_name == 'protein_contact_graph':
        if protein_ids is not None:
            seq_to_pid = {}
            for seq, pid in zip(protein_seqs, protein_ids):
                if seq not in seq_to_pid:
                    seq_to_pid[seq] = pid
            unique_pids = [seq_to_pid.get(seq) for seq in unique_proteins]
            uniprot_ids = prot._resolve_uniprot_ids(unique_pids)
            dispatch['protein_contact_graph'] = (
                lambda seqs: prot.create_contact_graph_features(seqs, uniprot_ids=uniprot_ids),
                'protein')
        else:
            dispatch['protein_contact_graph'] = (prot.create_contact_graph_features, 'protein')
    elif rep_name == 'protein_usr_shape':
        if protein_ids is not None:
            seq_to_pid = {}
            for seq, pid in zip(protein_seqs, protein_ids):
                if seq not in seq_to_pid:
                    seq_to_pid[seq] = pid
            unique_pids = [seq_to_pid.get(seq) for seq in unique_proteins]
            uniprot_ids = prot._resolve_uniprot_ids(unique_pids)
            dispatch['protein_usr_shape'] = (
                lambda seqs: prot.create_usr_shape_features(seqs, uniprot_ids=uniprot_ids),
                'protein')
        else:
            dispatch['protein_usr_shape'] = (prot.create_usr_shape_features, 'protein')
    elif rep_name == 'protein_pocket':
        if protein_ids is not None:
            seq_to_pid = {}
            for seq, pid in zip(protein_seqs, protein_ids):
                if seq not in seq_to_pid:
                    seq_to_pid[seq] = pid
            unique_pids = [seq_to_pid.get(seq) for seq in unique_proteins]
            uniprot_ids = prot._resolve_uniprot_ids(unique_pids)
            dispatch['protein_pocket'] = (
                lambda seqs: prot.create_pocket_features(seqs, uniprot_ids=uniprot_ids),
                'protein')
        else:
            dispatch['protein_pocket'] = (prot.create_pocket_features, 'protein')
    elif rep_name == 'protein_esm2_pocket':
        if protein_ids is not None:
            seq_to_pid = {}
            for seq, pid in zip(protein_seqs, protein_ids):
                if seq not in seq_to_pid:
                    seq_to_pid[seq] = pid
            unique_pids = [seq_to_pid.get(seq) for seq in unique_proteins]
            uniprot_ids = prot._resolve_uniprot_ids(unique_pids)
            dispatch['protein_esm2_pocket'] = (
                lambda seqs: prot.create_esm2_pocket_embeddings(seqs, uniprot_ids=uniprot_ids),
                'protein')
        else:
            dispatch['protein_esm2_pocket'] = (prot.create_esm2_pocket_embeddings, 'protein')
    elif rep_name == 'protein_pocket_composition':
        if protein_ids is not None:
            seq_to_pid = {}
            for seq, pid in zip(protein_seqs, protein_ids):
                if seq not in seq_to_pid:
                    seq_to_pid[seq] = pid
            unique_pids = [seq_to_pid.get(seq) for seq in unique_proteins]
            uniprot_ids = prot._resolve_uniprot_ids(unique_pids)
            dispatch['protein_pocket_composition'] = (
                lambda seqs: prot.create_pocket_composition_features(seqs, uniprot_ids=uniprot_ids),
                'protein')
        else:
            dispatch['protein_pocket_composition'] = (prot.create_pocket_composition_features, 'protein')

    if rep_name not in dispatch:
        print(f"  WARNING: Unknown rep '{rep_name}', skipping")
        return None

    func, rep_type = dispatch[rep_name]

    try:
        if rep_type == 'drug':
            unique_rep = func(unique_drugs)
            indices = np.array([drug_to_idx[s] for s in drug_smiles])
        else:
            unique_rep = func(unique_proteins)
            indices = np.array([protein_to_idx[s] for s in protein_seqs])
        return unique_rep[indices]
    except Exception as e:
        print(f"  FAILED generating {rep_name}: {e}")
        return None


def evaluate_rep(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 n_folds: int = 5) -> Dict[str, float]:
    """Evaluate a representation with GroupKFold CV. Returns {model: R²}."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    # Handle NaN/Inf/extreme values (some descriptors fail for certain molecules)
    X = X.copy()
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        n_bad = nan_mask.any(axis=0).sum()
        print(f"    (imputing {n_bad} features with NaN/Inf → column median)")
        col_medians = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for col in range(X.shape[1]):
            bad_rows = nan_mask[:, col]
            if bad_rows.any():
                X[bad_rows, col] = col_medians[col]
    # Clip extreme values to prevent overflow in linear algebra
    X = np.clip(X, -1e6, 1e6)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    gkf = GroupKFold(n_splits=n_folds)
    results = {}

    from kirby.find_best_hybrid import get_model as _get_model
    _model_names = ['ridge', 'rf', 'xgboost', 'lightgbm', 'mlp']
    models = {}
    for _mn in _model_names:
        try:
            _m = _get_model(_mn, task_type='regression', n_samples=len(y))
            if _m is not None:
                models[_mn] = lambda _m=_m: deepcopy(_m)
        except Exception:
            pass

    for model_name, model_fn in models.items():
        fold_scores = []
        fold_ci = []
        fold_spearman = []
        fold_rmse = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            # Clean up after scaling: zero-variance cols become NaN
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            # Skip if all features are constant (e.g. all-zero reps)
            if X_train.shape[1] == 0 or np.all(X_train.std(axis=0) == 0):
                continue
            model = model_fn()
            try:
                model.fit(X_train, y[train_idx])
                pred = model.predict(X_test)
            except Exception as e:
                print(f"    WARNING: {model_name} fit/predict failed: {e}")
                continue
            fold_scores.append(r2_score(y[test_idx], pred))
            fold_ci.append(_concordance_index(y[test_idx], pred))
            fold_spearman.append(float(spearmanr(y[test_idx], pred)[0]))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))

        if not fold_scores:
            continue
        mean_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)
        results[model_name] = {
            'mean': round(mean_r2, 4), 'std': round(std_r2, 4),
            'ci_mean': round(float(np.mean(fold_ci)), 4), 'ci_std': round(float(np.std(fold_ci)), 4),
            'spearman_mean': round(float(np.mean(fold_spearman)), 4), 'spearman_std': round(float(np.std(fold_spearman)), 4),
            'rmse_mean': round(float(np.mean(fold_rmse)), 4), 'rmse_std': round(float(np.std(fold_rmse)), 4),
        }

    return results


def make_dta_sns_factory(drug_smiles):
    """Create per-fold SNS factory for DTA (fits on unique drugs, expands to pairs).

    SNS substructure frequencies are computed on unique training drugs only,
    then the fitted featurizer transforms all unique drugs and expands to
    pair level.
    """
    from kirby.representations.molecular import create_sns

    unique_drugs = sorted(set(drug_smiles))
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    pair_drug_indices = np.array([drug_to_idx[s] for s in drug_smiles])

    def factory(train_idx, val_idx, train_labels):
        # Unique training drug SMILES for fitting
        train_drug_set = set(pair_drug_indices[train_idx])
        unique_train_drugs = [unique_drugs[i] for i in sorted(train_drug_set)]

        # Fit SNS on unique training drugs only
        _, featurizer = create_sns(unique_train_drugs, return_featurizer=True)

        # Transform ALL unique drugs with fitted featurizer
        all_features = create_sns(unique_drugs, reference_featurizer=featurizer)

        # Expand to pair level
        train_features = all_features[pair_drug_indices[train_idx]]
        val_features = all_features[pair_drug_indices[val_idx]]

        return train_features, val_features

    return factory


def evaluate_factory_rep(factory, y: np.ndarray, groups: np.ndarray,
                         n_folds: int = 5) -> Dict[str, float]:
    """Evaluate a factory rep with GroupKFold CV. Returns {model: metrics}.

    Like evaluate_rep but calls the factory per fold to get train/val features.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    gkf = GroupKFold(n_splits=n_folds)
    results = {}

    from kirby.find_best_hybrid import get_model as _get_model
    _model_names = ['ridge', 'rf', 'xgboost', 'lightgbm', 'mlp']
    models = {}
    for _mn in _model_names:
        try:
            _m = _get_model(_mn, task_type='regression', n_samples=len(y))
            if _m is not None:
                models[_mn] = lambda _m=_m: deepcopy(_m)
        except Exception:
            pass

    dummy_X = np.zeros((len(y), 1))

    for model_name, model_fn in models.items():
        fold_scores = []
        fold_ci = []
        fold_spearman = []
        fold_rmse = []
        for train_idx, test_idx in gkf.split(dummy_X, y, groups):
            try:
                X_train, X_test = factory(train_idx, test_idx, y[train_idx])
            except Exception as e:
                print(f"    Factory error: {e}")
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

            model = model_fn()
            try:
                model.fit(X_train, y[train_idx])
                pred = model.predict(X_test)
            except Exception as e:
                print(f"    WARNING: {model_name} fit/predict failed: {e}")
                continue
            fold_scores.append(r2_score(y[test_idx], pred))
            fold_ci.append(_concordance_index(y[test_idx], pred))
            fold_spearman.append(float(spearmanr(y[test_idx], pred)[0]))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))

        if not fold_scores:
            continue

        mean_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)
        results[model_name] = {
            'mean': round(mean_r2, 4), 'std': round(std_r2, 4),
            'ci_mean': round(float(np.mean(fold_ci)), 4), 'ci_std': round(float(np.std(fold_ci)), 4),
            'spearman_mean': round(float(np.mean(fold_spearman)), 4), 'spearman_std': round(float(np.std(fold_spearman)), 4),
            'rmse_mean': round(float(np.mean(fold_rmse)), 4), 'rmse_std': round(float(np.std(fold_rmse)), 4),
        }

    return results


def run_survey(dataset_name: str, max_pairs: Optional[int] = None,
               skip_lm: bool = False, skip_kmer: bool = False,
               cold_setting: str = 'target', n_folds: int = 5,
               reps_filter: Optional[List[str]] = None,
               skip_reps: Optional[set] = None):
    """Run full baseline survey on one dataset."""
    print(f"\n{'='*70}")
    print(f"DTA BASELINE SURVEY: {dataset_name.upper()}")
    print(f"{'='*70}")

    # Load data
    data = load_dataset(dataset_name)
    drug_smiles = data['drug_smiles']
    protein_seqs = data['protein_seqs']
    labels = data['labels']
    metadata = data['metadata']

    # Subsample if needed
    if max_pairs and len(labels) > max_pairs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(labels), max_pairs, replace=False)
        idx.sort()
        drug_smiles = [drug_smiles[i] for i in idx]
        protein_seqs = [protein_seqs[i] for i in idx]
        labels = labels[idx]
        metadata['drug_ids'] = [metadata['drug_ids'][i] for i in idx]
        metadata['protein_ids'] = [metadata['protein_ids'][i] for i in idx]
        metadata['n_drugs'] = len(set(drug_smiles))
        metadata['n_proteins'] = len(set(protein_seqs))
        print(f"  Subsampled to {max_pairs} pairs")

    # Cold-target or cold-drug groups
    # PDBBind: use LP-PDBBind leak-proof grouping (30% seq identity clusters)
    from kirby.datasets.dta import assign_cold_target_groups, assign_cold_drug_groups
    if dataset_name.startswith('pdbbind'):
        from kirby.datasets.pdbbind import assign_lp_groups
        groups, n_groups = assign_lp_groups(protein_seqs)
        print(f"  {len(labels)} pairs, {metadata['n_drugs']} drugs × {metadata['n_proteins']} proteins")
        print(f"  LP-PDBBind groups: {n_groups} clusters (30% seq identity)")
    elif cold_setting == 'drug':
        groups, n_groups = assign_cold_drug_groups(metadata['drug_ids'])
        print(f"  {len(labels)} pairs, {metadata['n_drugs']} drugs × {metadata['n_proteins']} proteins")
        print(f"  Cold-drug groups: {n_groups} unique drugs")
    else:
        groups, n_groups = assign_cold_target_groups(metadata['protein_ids'])
        print(f"  {len(labels)} pairs, {metadata['n_drugs']} drugs × {metadata['n_proteins']} proteins")
        print(f"  Cold-target groups: {n_groups} unique proteins")

    # Deduplicate for efficient generation
    unique_drugs = sorted(set(drug_smiles))
    unique_proteins = sorted(set(protein_seqs))
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    protein_to_idx = {s: i for i, s in enumerate(unique_proteins)}

    # Rep-level disk cache (survives job timeouts)
    import hashlib
    _hash_input = ''.join(drug_smiles[:100]) + ''.join(protein_seqs[:100]) + str(len(labels))
    _data_hash = hashlib.sha256(_hash_input.encode()).hexdigest()[:12]
    _cache_dir = Path(os.environ.get('KIRBY_CACHE_DIR',
                      str(Path.home() / '.cache' / 'kirby'))) / 'reps' / f'dta_survey_{dataset_name}_{_data_hash}'
    _cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Rep cache: {_cache_dir}")

    # Decide which reps to run
    drug_reps_to_run = [r for r in DRUG_REPS
                        if not (skip_lm and r in ('drug_chemberta', 'drug_molformer', 'drug_selformer', 'drug_clamp', 'drug_rmat', 'drug_unimol_v2', 'drug_minimol', 'drug_gp_molformer', 'drug_mole'))]
    protein_reps_to_run = [r for r in PROTEIN_REPS
                           if not (skip_lm and r in ('protein_esm2', 'protein_esmc', 'protein_esm2_midlayers', 'protein_amplify', 'protein_mint', 'protein_esmfold', 'protein_alphafold_db', 'protein_esm3', 'protein_saprot', 'protein_aido', 'protein_xtrimopglm', 'protein_esm2_pocket'))
                           and not (skip_kmer and r == 'protein_kmer')]
    factory_drug_reps_to_run = list(FACTORY_DRUG_REPS)

    # Apply --reps filter if provided
    if reps_filter:
        drug_reps_to_run = [r for r in drug_reps_to_run if r in reps_filter]
        protein_reps_to_run = [r for r in protein_reps_to_run if r in reps_filter]
        factory_drug_reps_to_run = [r for r in factory_drug_reps_to_run if r in reps_filter]
    if skip_reps:
        drug_reps_to_run = [r for r in drug_reps_to_run if r not in skip_reps]
        protein_reps_to_run = [r for r in protein_reps_to_run if r not in skip_reps]
        factory_drug_reps_to_run = [r for r in factory_drug_reps_to_run if r not in skip_reps]

    all_reps_to_run = drug_reps_to_run + factory_drug_reps_to_run + protein_reps_to_run

    # Generate all reps first (with disk caching)
    # Skip factory reps here — they're handled separately below
    base_reps_to_generate = [r for r in all_reps_to_run
                             if r not in FACTORY_DRUG_REPS]
    print(f"\nGenerating {len(base_reps_to_generate)} representations...")
    generated = {}
    for rep_name in base_reps_to_generate:
        cache_file = _cache_dir / f'{rep_name}.npy'
        if cache_file.exists():
            rep = np.load(cache_file)
            generated[rep_name] = rep
            print(f"  {rep_name}: {rep.shape} (cached)")
            continue

        t0 = time.time()
        rep = generate_rep(rep_name, unique_drugs, unique_proteins,
                           drug_to_idx, protein_to_idx,
                           drug_smiles, protein_seqs,
                           protein_ids=metadata.get('protein_ids'))
        elapsed = time.time() - t0
        if rep is not None:
            generated[rep_name] = rep
            np.save(cache_file, rep)
            print(f"  {rep_name}: {rep.shape} ({elapsed:.1f}s) [saved]")
        else:
            print(f"  {rep_name}: FAILED ({elapsed:.1f}s)")

    # Pre-filter protein_kmer from 8000D → 200D via LightGBM feature importance
    if 'protein_kmer' in generated:
        generated['protein_kmer'] = _filter_features_by_importance(
            generated['protein_kmer'], labels, n_features=200,
            rep_name='protein_kmer')

    # Factory drug reps (per-fold fitting, e.g. SNS)
    factory_reps = {}
    for rep_name in factory_drug_reps_to_run:
        if rep_name == 'drug_sns':
            print(f"\n  Creating {rep_name} factory...")
            factory_reps[rep_name] = make_dta_sns_factory(drug_smiles)

    # Factory rep dimensions (SNS = 1024 by default)
    _factory_dims = {'drug_sns': 1024}

    # Classify reps by tier
    sys.path.insert(0, str(Path(__file__).parent / 'experiments'))
    from _shared import classify_rep_tier
    rep_tiers = {}
    for rep_name, rep in generated.items():
        rep_tiers[rep_name] = classify_rep_tier(rep.shape[1], rep_name)
    for rep_name in factory_reps:
        dim = _factory_dims.get(rep_name, 1024)
        rep_tiers[rep_name] = classify_rep_tier(dim, rep_name)

    tier_1_names = sorted(k for k, t in rep_tiers.items() if t == 1)
    tier_23_names = sorted(k for k, t in rep_tiers.items() if t >= 2)
    if tier_1_names:
        print(f"\n  Tier 1 (skip standalone): {tier_1_names}")

    # Evaluate: standalone mode (each rep alone) — skip tier 1
    print(f"\n--- STANDALONE EVALUATION (tier 2+3 reps only, "
          f"skipping {len(tier_1_names)} tier-1 reps) ---")
    standalone_results = {}
    for rep_name, rep in generated.items():
        if rep_tiers.get(rep_name, 3) == 1:
            continue  # tier 1: augmentation-only, skip standalone
        t0 = time.time()
        scores = evaluate_rep(rep, labels, groups, n_folds=n_folds)
        elapsed = time.time() - t0
        standalone_results[rep_name] = scores
        best_model = max(scores, key=lambda m: scores[m]['ci_mean'])
        best_ci = scores[best_model]['ci_mean']
        best_r2 = scores[best_model]['mean']
        print(f"  {rep_name:<25} best={best_model} CI={best_ci:.4f} R²={best_r2:.4f} ({elapsed:.1f}s)")

    # Evaluate factory reps (standalone)
    for rep_name, factory in factory_reps.items():
        if rep_tiers.get(rep_name, 3) == 1:
            continue
        t0 = time.time()
        scores = evaluate_factory_rep(factory, labels, groups, n_folds=n_folds)
        elapsed = time.time() - t0
        if scores:
            standalone_results[rep_name] = scores
            best_model = max(scores, key=lambda m: scores[m]['ci_mean'])
            best_ci = scores[best_model]['ci_mean']
            best_r2 = scores[best_model]['mean']
            print(f"  {rep_name:<25} best={best_model} CI={best_ci:.4f} R²={best_r2:.4f} ({elapsed:.1f}s) [factory]")
        else:
            print(f"  {rep_name:<25} FAILED ({elapsed:.1f}s)")

    # Evaluate: paired mode (drug + minimal protein, or protein + minimal drug)
    print(f"\n--- PAIRED EVALUATION (drug+protein_aac or protein+drug_ecfp4) ---")
    paired_results = {}
    drug_partner = generated.get(DEFAULT_DRUG_PARTNER)
    protein_partner = generated.get(DEFAULT_PROTEIN_PARTNER)

    for rep_name, rep in generated.items():
        if rep_name.startswith('drug_') and protein_partner is not None:
            X = np.hstack([rep, protein_partner])
            partner = DEFAULT_PROTEIN_PARTNER
        elif rep_name.startswith('protein_') and drug_partner is not None:
            X = np.hstack([drug_partner, rep])
            partner = DEFAULT_DRUG_PARTNER
        else:
            continue

        t0 = time.time()
        scores = evaluate_rep(X, labels, groups, n_folds=n_folds)
        elapsed = time.time() - t0
        paired_results[rep_name] = scores
        best_model = max(scores, key=lambda m: scores[m]['ci_mean'])
        best_ci = scores[best_model]['ci_mean']
        best_r2 = scores[best_model]['mean']
        print(f"  {rep_name:<25} +{partner:<20} best={best_model} CI={best_ci:.4f} R²={best_r2:.4f} ({elapsed:.1f}s)")

    # Paired evaluation for factory drug reps (concat with protein_aac)
    if protein_partner is not None:
        for rep_name, factory in factory_reps.items():
            partner = DEFAULT_PROTEIN_PARTNER
            # Wrap factory to concat with static protein partner
            def _make_paired_factory(base_factory, static_partner):
                def paired_factory(train_idx, val_idx, train_labels):
                    X_train_drug, X_val_drug = base_factory(
                        train_idx, val_idx, train_labels)
                    X_train = np.hstack([X_train_drug, static_partner[train_idx]])
                    X_val = np.hstack([X_val_drug, static_partner[val_idx]])
                    return X_train, X_val
                return paired_factory

            paired_f = _make_paired_factory(factory, protein_partner)
            t0 = time.time()
            scores = evaluate_factory_rep(paired_f, labels, groups, n_folds=n_folds)
            elapsed = time.time() - t0
            if scores:
                paired_results[rep_name] = scores
                best_model = max(scores, key=lambda m: scores[m]['ci_mean'])
                best_ci = scores[best_model]['ci_mean']
                best_r2 = scores[best_model]['mean']
                print(f"  {rep_name:<25} +{partner:<20} best={best_model} CI={best_ci:.4f} R²={best_r2:.4f} ({elapsed:.1f}s) [factory]")

    # Combined: all drug + all protein
    if drug_partner is not None and protein_partner is not None:
        all_drug_reps = [generated[r] for r in drug_reps_to_run if r in generated]
        all_protein_reps = [generated[r] for r in protein_reps_to_run if r in generated]
        if all_drug_reps and all_protein_reps:
            X_all = np.hstack(all_drug_reps + all_protein_reps)
            print(f"\n--- ALL COMBINED ({X_all.shape[1]}D) ---")
            t0 = time.time()
            all_scores = evaluate_rep(X_all, labels, groups, n_folds=n_folds)
            elapsed = time.time() - t0
            best_model = max(all_scores, key=lambda m: all_scores[m]['ci_mean'])
            best_ci = all_scores[best_model]['ci_mean']
            best_r2 = all_scores[best_model]['mean']
            print(f"  all_concat ({X_all.shape[1]}D): best={best_model} CI={best_ci:.4f} R²={best_r2:.4f} ({elapsed:.1f}s)")
        else:
            all_scores = None
    else:
        all_scores = None

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name.upper()}")
    print(f"{'='*70}")
    models_seen = set()
    for scores in standalone_results.values():
        models_seen.update(scores.keys())
    models_list = sorted(models_seen)

    header = f"{'Rep':<25} {'Dim':<6} " + " ".join(f"{m:<22}" for m in models_list)
    print(header)
    print("-" * len(header))

    for rep_name in all_reps_to_run:
        if rep_name not in standalone_results:
            continue
        dim = (generated[rep_name].shape[1] if rep_name in generated
               else _factory_dims.get(rep_name, '?'))
        row = f"{rep_name:<25} {dim:<6} "
        for m in models_list:
            if m in standalone_results[rep_name]:
                s = standalone_results[rep_name][m]
                row += f"R²={s['mean']:.4f} CI={s['ci_mean']:.4f}  "
            else:
                row += f"{'--':<22} "
        print(row)

    print(f"\nPaired (+ minimal partner):")
    for rep_name in all_reps_to_run:
        if rep_name not in paired_results:
            continue
        dim = (generated[rep_name].shape[1] if rep_name in generated
               else _factory_dims.get(rep_name, '?'))
        row = f"{rep_name:<25} {dim:<6} "
        for m in models_list:
            if m in paired_results[rep_name]:
                s = paired_results[rep_name][m]
                row += f"R²={s['mean']:.4f} CI={s['ci_mean']:.4f}  "
            else:
                row += f"{'--':<22} "
        print(row)

    return {
        'dataset': dataset_name,
        'cold_setting': cold_setting,
        'n_pairs': len(labels),
        'standalone': standalone_results,
        'paired': paired_results,
        'all_combined': all_scores,
        'reps_generated': list(generated.keys()) + list(factory_reps.keys()),
        'reps_failed': [r for r in all_reps_to_run
                        if r not in generated and r not in standalone_results],
    }


def main():
    parser = argparse.ArgumentParser(description='DTA Baseline Survey for BOM')
    parser.add_argument('dataset', choices=[
        'davis', 'kiba', 'bindingdb_kd', 'pdbbind_refined', 'pdbbind_core', 'all'],
                        help='Dataset to evaluate')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Subsample large datasets (default: 30000 for kiba/bindingdb_kd)')
    parser.add_argument('--skip-lm', action='store_true',
                        help='Skip LM reps (ESM-2, ChemBERTa, MolFormer, SelFormer)')
    parser.add_argument('--skip-kmer', action='store_true',
                        help='Skip protein_kmer (8000D→200D via importance filter)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--cold-setting', choices=['target', 'protein', 'drug'], default='target',
                        help='Cold split: target (default) or drug')
    parser.add_argument('--reps', type=str, default=None,
                        help='Comma-separated rep names to evaluate (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results JSON')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip reps already present in output JSON')
    args = parser.parse_args()

    reps_filter = None
    if args.reps:
        reps_filter = [r.strip() for r in args.reps.split(',')]

    datasets = (['davis', 'kiba', 'bindingdb_kd', 'pdbbind_refined', 'pdbbind_core']
                if args.dataset == 'all' else [args.dataset])
    all_results = {}

    for ds in datasets:
        # --skip-existing: find reps already in the output JSON
        skip_reps = set()
        if args.skip_existing and args.output:
            out_path = Path(args.output)
            if out_path.exists():
                try:
                    with open(out_path) as f:
                        _existing = json.load(f)
                    ds_data = _existing.get(ds, _existing)
                    skip_reps = set(ds_data.get('standalone', {}).keys())
                    if skip_reps:
                        print(f"  [{ds}] Skipping {len(skip_reps)} existing reps")
                except (json.JSONDecodeError, KeyError):
                    pass

        max_pairs = args.max_pairs
        if max_pairs is None and ds in ('kiba', 'bindingdb_kd'):
            max_pairs = 30000  # default subsample for large datasets
        result = run_survey(ds, max_pairs=max_pairs,
                            skip_lm=args.skip_lm, skip_kmer=args.skip_kmer,
                            cold_setting=args.cold_setting,
                            n_folds=args.n_folds,
                            reps_filter=reps_filter,
                            skip_reps=skip_reps or None)
        all_results[ds] = result

    if args.output:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Merge with existing results if file exists
        out_path = Path(args.output)
        if out_path.exists():
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                for section in ('standalone', 'paired', 'all_combined'):
                    if section in all_results and all_results[section]:
                        if section not in existing or not existing[section]:
                            existing[section] = {}
                        existing[section].update(all_results[section])
                all_results = existing
                print(f"  Merged with existing: {out_path}")
            except (json.JSONDecodeError, KeyError):
                pass
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
