#!/usr/bin/env python3
"""
Universal Baseline Survey

For every (rep x dataset x model) cell, compute baseline R²/accuracy with
both default and tuned hyperparameters. No hybrids. 3-fold GroupKFold.

Usage:
    python tests/run_baseline_survey.py logd
    python tests/run_baseline_survey.py logd --reps fast
    python tests/run_baseline_survey.py molecular              # all molecular datasets
    python tests/run_baseline_survey.py everything             # all domains
    python tests/run_baseline_survey.py tcga --finetune        # include FT reps (GPU)
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
from typing import Any, Dict, List, Optional, Tuple

# Block stale system bottleneck (Anaconda 2022.05)
sys.modules.setdefault('bottleneck', None)

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error,
                             f1_score, roc_auc_score)
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from experiments._shared import convert_numpy

warnings.filterwarnings('ignore')


# =============================================================================
# DATASET REGISTRY
# =============================================================================

# {dataset_name: (domain, default_rep_preset)}
DATASET_REGISTRY = {
    # Molecular
    'logd':  ('molecular', 'all'),
    'caco2': ('molecular', 'all'),
    'herg':  ('molecular', 'all'),
    'qm9':   ('molecular', 'all'),
    'adme_hlm':        ('molecular', 'all'),
    'adme_rlm':        ('molecular', 'all'),
    'adme_mdr1':       ('molecular', 'all'),
    'adme_solubility': ('molecular', 'all'),
    # Antibody — use antibody_baseline_survey.py instead for antibody datasets
    # TCGA
    'tcga_luad_vs_lusc':  ('tcga', 'all'),
    'tcga_brca_pam50':    ('tcga', 'all'),
    'tcga_brca_survival': ('tcga', 'all'),
    # DTA
    'davis': ('dta', 'full'),
    'kiba':  ('dta', 'full'),
    'bindingdb_kd': ('dta', 'full'),
    'pdbbind_refined': ('dta', 'full'),
    'pdbbind_core':    ('dta', 'full'),
    # Reaction Yield
    'buchwald_hartwig': ('reaction', 'all'),
    'suzuki':           ('reaction', 'all'),
    'bh_doyle_plates':  ('reaction', 'all'),
}

# Build domain -> datasets mapping
DOMAIN_DATASETS = {}
for _ds, (_dom, _) in DATASET_REGISTRY.items():
    DOMAIN_DATASETS.setdefault(_dom, []).append(_ds)

SURVEY_MODELS = ['rf', 'xgboost', 'ridge', 'knn', 'lightgbm',
                  'gp', 'mlp', 'dnn', 'cnn', 'catboost']


def _make_ridge_l2norm(task_type='regression'):
    """Ridge with L2-normalized features (recommended for high-dim embeddings)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.linear_model import RidgeCV
    return make_pipeline(Normalizer(norm='l2'),
                         RidgeCV(alphas=np.logspace(-4, 4, 50)))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_and_reps(dataset_name: str, rep_preset: Optional[str] = None) -> dict:
    """Load dataset + generate representations. Returns standard dict."""
    domain, default_preset = DATASET_REGISTRY[dataset_name]
    preset = rep_preset or default_preset

    if domain == 'dta':
        return _load_dta(dataset_name, preset)
    if domain == 'reaction':
        return _load_reaction(dataset_name)

    # Molecular, antibody, TCGA — handled by _shared.load_data
    from experiments._shared import load_data
    return load_data(dataset_name, rep_preset=preset)


def _load_dta(dataset_name: str, rep_preset: str) -> dict:
    """Load DTA dataset (not handled by _shared.load_data)."""
    from find_best_hybrid_dta import (
        load_dataset as _load_dta_ds,
        get_representations as _get_dta_reps,
    )
    from kirby.datasets.dta import assign_cold_target_groups

    data = _load_dta_ds(dataset_name)
    drug_smiles = data['drug_smiles']
    protein_seqs = data['protein_seqs']
    labels = data['labels']
    task_type = data['task_type']
    metadata = data['metadata']
    protein_ids = metadata['protein_ids']

    # Subsample KIBA (117K pairs too large for baseline survey)
    if len(labels) > 30000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(labels), 30000, replace=False)
        idx.sort()
        drug_smiles = [drug_smiles[i] for i in idx]
        protein_seqs = [protein_seqs[i] for i in idx]
        labels = labels[idx]
        protein_ids = [protein_ids[i] for i in idx]
        print(f"  Subsampled to 30000 pairs")

    reps = _get_dta_reps(drug_smiles, protein_seqs, rep_preset)
    groups, n_groups = assign_cold_target_groups(protein_ids)

    print(f"  {dataset_name}: {len(labels)} pairs, {n_groups} cold-target groups, "
          f"{len(reps)} reps")

    return {
        'labels': labels,
        'reps': reps,
        'groups': groups,
        'n_groups': n_groups,
        'task_type': task_type,
        'domain': 'dta',
        'augmentations': None,
        'drug_smiles': drug_smiles,
        'protein_seqs': protein_seqs,
    }


def _load_reaction(dataset_name: str) -> dict:
    """Load reaction yield dataset with multiple representation types."""
    from kirby.datasets.reaction_yields import load_buchwald_hartwig, load_suzuki
    from kirby.representations.reaction import (
        create_drfp, create_concat_ecfp4, create_concat_maccs,
        create_concat_pdv, create_product_ecfp4, create_reactant_ecfp4,
        create_rxnfp, create_onehot_components, create_reactiont5,
    )

    if dataset_name == 'buchwald_hartwig':
        data = load_buchwald_hartwig()
    elif dataset_name == 'bh_doyle_plates':
        from kirby.datasets.reaction_yields import load_bh_doyle_plates
        data = load_bh_doyle_plates()
    else:
        data = load_suzuki()

    reps = {}

    # Concat fingerprints per component
    reps['concat_ecfp4'] = create_concat_ecfp4(data['components'])
    try:
        reps['concat_maccs'] = create_concat_maccs(data['components'])
    except Exception as e:
        print(f"  WARNING: concat_maccs failed: {e}")
    try:
        reps['concat_pdv'] = create_concat_pdv(data['components'])
    except Exception as e:
        print(f"  WARNING: concat_pdv failed: {e}")

    # Reaction-level fingerprints
    try:
        reps['drfp_2048'] = create_drfp(data['rxn_smiles'])
    except ImportError:
        print("  WARNING: drfp not installed, skipping DRFP rep")

    # Product-only and reactant-only fingerprints
    try:
        reps['product_ecfp4'] = create_product_ecfp4(data['rxn_smiles'])
    except Exception as e:
        print(f"  WARNING: product_ecfp4 failed: {e}")
    try:
        reps['reactant_ecfp4'] = create_reactant_ecfp4(data['rxn_smiles'])
    except Exception as e:
        print(f"  WARNING: reactant_ecfp4 failed: {e}")

    # Learned reaction fingerprints
    try:
        reps['rxnfp'] = create_rxnfp(data['rxn_smiles'])
    except ImportError:
        print("  WARNING: rxnfp not installed, skipping RXNFP rep")
    except Exception as e:
        print(f"  WARNING: rxnfp failed: {e}")

    # ReactionT5 encoder embeddings
    try:
        reps['reactiont5'] = create_reactiont5(data['rxn_smiles'])
    except ImportError:
        print("  WARNING: transformers/torch not installed, skipping ReactionT5 rep")
    except Exception as e:
        print(f"  WARNING: reactiont5 failed: {e}")

    # One-hot component encoding (factorial baseline)
    try:
        reps['onehot_components'] = create_onehot_components(data['components'])
    except Exception as e:
        print(f"  WARNING: onehot_components failed: {e}")

    n = len(data['labels'])
    groups = data.get('groups')
    n_groups = len(np.unique(groups)) if groups is not None else 0
    group_key = data.get('metadata', {}).get('group_key', 'unknown')
    print(f"  {dataset_name}: {n} reactions, {len(reps)} reps, "
          f"{n_groups} groups ({group_key}) "
          f"({', '.join(f'{k}={v.shape[1]}' for k, v in reps.items())})")

    return {
        'labels': data['labels'],
        'reps': reps,
        'groups': groups,
        'n_groups': n_groups,
        'task_type': 'regression',
        'domain': 'reaction',
        'augmentations': None,
        'prebuilt_splits': None,  # use GroupKFold on groups, not random splits
    }


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def get_default_models(task_type: str) -> Dict[str, Any]:
    """Return default model instances for the 5 survey models."""
    from kirby.find_best_hybrid import _resolve_models
    # speed='long' includes all tiers (rf/xgb=fast, ridge/knn=medium, lgbm=long)
    models = _resolve_models(SURVEY_MODELS, task_type, 'long')
    # Set n_jobs=1 for fork safety (avoids SIGSEGV on macOS with LightGBM).
    # Parallelism happens at the SLURM job level, not within models.
    for m in models.values():
        if hasattr(m, 'n_jobs'):
            m.n_jobs = 1
    return models


# =============================================================================
# EVALUATION
# =============================================================================

def _make_cv(groups, n_folds, labels):
    """Create CV splitter and split kwargs."""
    if groups is not None:
        n_unique = len(np.unique(groups))
        actual_folds = min(n_folds, n_unique)
        # Classification: try StratifiedGroupKFold for class balance
        if labels is not None and np.issubdtype(np.asarray(labels).dtype, np.integer):
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                return (
                    StratifiedGroupKFold(
                        n_splits=actual_folds, shuffle=True, random_state=42),
                    {'groups': groups},
                )
            except (ImportError, Exception):
                pass
        return GroupKFold(n_splits=actual_folds), {'groups': groups}
    return KFold(n_splits=n_folds, shuffle=True, random_state=42), {}


def _score(y_true, y_pred, task_type):
    """Compute all domain-appropriate metrics. Returns dict."""
    if task_type == 'classification':
        metrics = {'accuracy': float(accuracy_score(y_true, y_pred))}
        # F1-weighted (works for binary and multiclass)
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred,
                                                average='weighted',
                                                zero_division=0))
        # AUROC — only for binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 2:
            try:
                metrics['auroc'] = float(roc_auc_score(y_true, y_pred))
            except ValueError:
                pass
        return metrics
    # Regression
    metrics = {
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    try:
        rho, _ = spearmanr(y_true, y_pred)
        metrics['spearman'] = float(rho)
    except Exception:
        pass
    try:
        r, _ = pearsonr(y_true, y_pred)
        metrics['pearson'] = float(r)
    except Exception:
        pass
    try:
        from kirby.find_best_hybrid import _concordance_index
        metrics['ci'] = float(_concordance_index(y_true, y_pred))
    except Exception:
        pass
    return metrics


def _get_primary_metric(dataset_name: str, task_type: str) -> tuple:
    """Return (metric_name, higher_is_better) per domain."""
    domain = DATASET_REGISTRY.get(dataset_name, ('unknown', None))[0]
    if task_type == 'classification':
        if domain == 'tcga':
            if dataset_name == 'tcga_luad_vs_lusc':
                return 'auroc', True
            return 'f1_weighted', True
        return 'accuracy', True
    if domain == 'molecular':
        return 'rmse', False
    if domain == 'dta':
        return 'ci', True
    return 'r2', True


def _primary_metric(metrics, task_type, metric_name=None):
    """Extract primary metric value."""
    if metric_name:
        return metrics.get(metric_name, float('nan'))
    if task_type == 'classification':
        return metrics.get('accuracy', float('nan'))
    return metrics.get('r2', float('nan'))


def _is_binary_rep(X):
    """Check if feature matrix is binary (0/1) — should NOT be StandardScaled.

    Binary fingerprints (ECFP, MACCS, DRFP, onehot, blosum) should be used raw.
    Only continuous features (PDV, LM embeddings, descriptors) need z-scoring.
    Matches qsar_qm_models behavior where only continuous_pdv is scaled.
    """
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


def _maybe_scale(X_train, X_val):
    """Scale features only if continuous. Binary features are left raw."""
    if _is_binary_rep(X_train):
        return X_train.astype(np.float32), X_val.astype(np.float32)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=1e10, neginf=-1e10)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=1e10, neginf=-1e10)
    return X_train_s, X_val_s


def evaluate_rep(X, labels, groups, model, n_folds, task_type,
                 prebuilt_splits=None):
    """Evaluate one (rep, model) pair via CV. Returns (mean_metrics, std_metrics)."""
    if prebuilt_splits is not None:
        splits = prebuilt_splits
    else:
        cv, cv_kwargs = _make_cv(groups, n_folds, labels)
        splits = list(cv.split(X, labels, **cv_kwargs))
    fold_metrics = []

    for train_idx, val_idx in splits:
        X_train, X_val = _maybe_scale(X[train_idx], X[val_idx])

        try:
            m = clone(model)
            m.fit(X_train, labels[train_idx])
            fold_metrics.append(
                _score(labels[val_idx], m.predict(X_val), task_type))
        except Exception:
            continue

    if not fold_metrics:
        return {}, {}
    # Aggregate across folds: mean and std for each metric key
    all_keys = fold_metrics[0].keys()
    mean_m = {k: float(np.mean([f[k] for f in fold_metrics if k in f]))
              for k in all_keys}
    std_m = {k: float(np.std([f[k] for f in fold_metrics if k in f]))
             for k in all_keys}
    return mean_m, std_m


def evaluate_ft_rep(factory, labels, groups, model, n_folds, task_type):
    """Evaluate a fine-tuning factory via CV. Returns (mean_metrics, std_metrics)."""
    cv, cv_kwargs = _make_cv(groups, n_folds, labels)
    fold_metrics = []
    dummy_X = np.zeros((len(labels), 1))

    for train_idx, val_idx in cv.split(dummy_X, labels, **cv_kwargs):
        try:
            X_train, X_val = factory(train_idx, val_idx, labels[train_idx])
        except Exception as e:
            print(f" [ft error: {e}]", end='')
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
            fold_metrics.append(
                _score(labels[val_idx], m.predict(X_val), task_type))
        except Exception:
            continue

    if not fold_metrics:
        return {}, {}
    all_keys = fold_metrics[0].keys()
    mean_m = {k: float(np.mean([f[k] for f in fold_metrics if k in f]))
              for k in all_keys}
    std_m = {k: float(np.std([f[k] for f in fold_metrics if k in f]))
             for k in all_keys}
    return mean_m, std_m


# =============================================================================
# FINE-TUNING FACTORY CREATION
# =============================================================================

def make_ft_factories(data, dataset_name, task_type, ft_epochs=3, ft_cache=True):
    """Create fine-tuning factories for the dataset's domain.

    Supported domains:
        - tcga: Geneformer, scFoundation
        - molecular: ChemBERTa LoRA, MolFormer LoRA
        - dta: ESM-2 protein LoRA
    """
    domain = data.get('domain', '')
    factories = {}

    if domain == 'tcga':
        expression = data.get('expression')
        metadata = data.get('metadata', {})
        gene_names = metadata.get('gene_names')
        sample_ids = metadata.get('sample_ids')

        if expression is None or gene_names is None:
            return {}

        try:
            from kirby.finetuning.tcga_geneformer import make_geneformer_factory
            factories['geneformer'] = make_geneformer_factory(
                expression_data=expression, gene_names=gene_names,
                sample_ids=sample_ids, dataset_name=dataset_name,
                task_type=task_type, epochs=ft_epochs, cache=ft_cache,
            )
        except Exception as e:
            print(f"  geneformer factory failed: {e}")

        try:
            from kirby.finetuning.tcga_scfoundation import make_scfoundation_factory
            factories['scfoundation'] = make_scfoundation_factory(
                expression_data=expression, gene_names=gene_names,
                sample_ids=sample_ids, dataset_name=dataset_name,
                task_type=task_type, epochs=ft_epochs, cache=ft_cache,
            )
        except Exception as e:
            print(f"  scfoundation factory failed: {e}")

    elif domain == 'molecular':
        smiles = data.get('smiles')
        if smiles is None:
            return {}

        for base_model in ('chemberta', 'molformer'):
            try:
                from kirby.finetuning.molecular_lora import make_molecular_lora_factory
                factories[f'lora_{base_model}'] = make_molecular_lora_factory(
                    smiles_list=smiles, dataset_name=dataset_name,
                    base_model=base_model, epochs=ft_epochs, cache=ft_cache,
                )
            except Exception as e:
                print(f"  lora_{base_model} factory failed: {e}")

        # 3D GNN fine-tuning (SchNet E2E → 128D embeddings)
        for model_3d in ('schnet',):
            try:
                from kirby.finetuning.molecular_3d import make_molecular_3d_factory
                factories[f'ft_{model_3d}'] = make_molecular_3d_factory(
                    smiles_list=smiles, dataset_name=dataset_name,
                    base_model=model_3d, epochs=100, cache=ft_cache,
                )
            except Exception as e:
                print(f"  ft_{model_3d} factory failed: {e}")

    elif domain == 'dta':
        protein_seqs = data.get('protein_seqs')
        drug_smiles = data.get('drug_smiles')
        if protein_seqs is None or drug_smiles is None:
            return {}

        try:
            from kirby.finetuning.dta_protein_lora import make_dta_protein_lora_factory
            factories['protein_esm2_lora'] = make_dta_protein_lora_factory(
                protein_seqs=protein_seqs, drug_smiles=drug_smiles,
                dataset_name=dataset_name, base_model='esm2',
                epochs=ft_epochs, cache=ft_cache,
            )
        except Exception as e:
            print(f"  protein_esm2_lora factory failed: {e}")

    return factories


# =============================================================================
# BEST-SCORE HELPERS
# =============================================================================



# =============================================================================
# MAIN SURVEY LOOP
# =============================================================================

def run_survey(dataset_name, rep_preset, n_folds, do_finetune, seed,
               tune_trials=0, l2_norm=False, augment=False):
    """Run the full baseline survey for one dataset. Returns result dict."""
    print(f"\n{'='*70}")
    print(f"SURVEY: {dataset_name}")
    print(f"{'='*70}")

    t0 = time.time()

    # 1. Load data + reps
    data = load_dataset_and_reps(dataset_name, rep_preset)
    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    prebuilt_splits = data.get('prebuilt_splits')
    task_type = data.get('task_type', 'regression')
    n_groups = len(np.unique(groups)) if groups is not None else 0
    metric_name, higher_is_better = _get_primary_metric(dataset_name, task_type)

    # 2. Default models
    default_models = get_default_models(task_type)

    # Optional: add L2-normalized Ridge
    if l2_norm:
        default_models['ridge_l2norm'] = _make_ridge_l2norm(task_type)

    model_names = list(default_models.keys())

    # 3. Evaluate each frozen rep
    baselines = {}
    print(f"\n--- Evaluating {len(reps)} reps x {len(model_names)} models ---")

    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        try:
            X = np.asarray(X, dtype=np.float64)
        except Exception:
            print(f"  {rep_name}: SKIPPED (array conversion failed)")
            continue
        dim = X.shape[1]
        print(f"  {rep_name} ({dim}d):", end='', flush=True)

        try:
            entry = {'dimension': dim, 'scores': {}, 'metrics': {}}

            for mname in model_names:
                mean_m, std_m = evaluate_rep(
                    X, labels, groups, default_models[mname], n_folds, task_type,
                    prebuilt_splits=prebuilt_splits)
                primary = _primary_metric(mean_m, task_type, metric_name)
                entry['scores'][mname] = round(primary, 4)
                entry['metrics'][mname] = {
                    k: {'mean': round(v, 4), 'std': round(std_m.get(k, 0), 4)}
                    for k, v in mean_m.items()
                }
                print(f" {mname}={primary:.3f}", end='', flush=True)

            # Optional: Optuna-tuned XGBoost
            if tune_trials > 0:
                try:
                    from experiments._shared import tune_xgboost_optuna
                    if prebuilt_splits is not None:
                        tune_splits = prebuilt_splits
                    else:
                        cv, cv_kwargs = _make_cv(groups, n_folds, labels)
                        tune_splits = list(cv.split(X, labels, **cv_kwargs))
                    tune_fold_metrics = []
                    for train_idx, val_idx in tune_splits:
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X[train_idx])
                        X_va = scaler.transform(X[val_idx])
                        tuned = tune_xgboost_optuna(
                            X_tr, labels[train_idx],
                            n_trials=tune_trials, task_type=task_type, seed=seed,
                        )
                        tune_fold_metrics.append(_score(
                            labels[val_idx], tuned.predict(X_va), task_type))
                    if tune_fold_metrics:
                        t_keys = tune_fold_metrics[0].keys()
                        t_mean = {k: float(np.mean([f[k] for f in tune_fold_metrics if k in f]))
                                  for k in t_keys}
                        t_std = {k: float(np.std([f[k] for f in tune_fold_metrics if k in f]))
                                 for k in t_keys}
                        primary = _primary_metric(t_mean, task_type, metric_name)
                        entry['scores']['xgb_tuned'] = round(primary, 4)
                        entry['metrics']['xgb_tuned'] = {
                            k: {'mean': round(v, 4), 'std': round(t_std.get(k, 0), 4)}
                            for k, v in t_mean.items()
                        }
                        print(f" xgb_tuned={primary:.3f}", end='', flush=True)
                except ImportError:
                    print(" (optuna not installed)", end='', flush=True)

            # Best model for this rep
            valid = [(m, s) for m, s in entry['scores'].items()
                     if not np.isnan(s)]
            if valid:
                best = min(valid, key=lambda x: x[1]) if not higher_is_better else max(valid, key=lambda x: x[1])
                entry['best'] = {'model': best[0], 'score': best[1]}
            else:
                entry['best'] = None

            baselines[rep_name] = entry
            print()
        except Exception as e:
            print(f" EVAL FAILED: {e}")

    # 4. Fine-tuned reps (TCGA only)
    ft_baselines = {}
    if do_finetune:
        ft_factories = make_ft_factories(data, dataset_name, task_type)
        if ft_factories:
            print(f"\n--- Fine-tuned reps: {list(ft_factories.keys())} ---")
            for ft_name, factory in ft_factories.items():
                print(f"  {ft_name}:", end='', flush=True)
                entry = {'dimension': '?', 'scores': {}, 'metrics': {}}

                for mname in model_names:
                    mean_m, std_m = evaluate_ft_rep(
                        factory, labels, groups, default_models[mname],
                        n_folds, task_type)
                    primary = _primary_metric(mean_m, task_type, metric_name)
                    entry['scores'][mname] = round(primary, 4)
                    entry['metrics'][mname] = {
                        k: {'mean': round(v, 4), 'std': round(std_m.get(k, 0), 4)}
                        for k, v in mean_m.items()
                    }
                    print(f" {mname}={primary:.3f}", end='', flush=True)

                valid = [(m, s) for m, s in entry['scores'].items()
                         if not np.isnan(s)]
                if valid:
                    best = min(valid, key=lambda x: x[1]) if not higher_is_better else max(valid, key=lambda x: x[1])
                    entry['best'] = {'model': best[0], 'score': best[1]}
                else:
                    entry['best'] = None

                ft_baselines[ft_name] = entry
                print()

    # 5. Augmentation test: concat small reps with top base, check if it helps
    augmentations = {}
    if augment and baselines:
        # Find the top-scoring base rep (by best model score)
        scored = [(name, e['best']['score']) for name, e in baselines.items()
                  if e.get('best') is not None]
        if scored:
            top_rep, top_score = (min(scored, key=lambda x: x[1]) if not higher_is_better
                                  else max(scored, key=lambda x: x[1]))
            top_model = baselines[top_rep]['best']['model']
            X_top = reps[top_rep]

            # Small reps = ≤200D, excluding the top rep itself
            small_reps = {k: v for k, v in reps.items()
                          if v.shape[1] <= 200 and k != top_rep}

            if small_reps:
                print(f"\n--- Augmentation test: {top_rep} ({metric_name}={top_score:.4f}) "
                      f"+ {len(small_reps)} small reps ---")

                model = default_models[top_model]
                for aug_name in sorted(small_reps.keys()):
                    X_aug = np.hstack([X_top, small_reps[aug_name]])
                    mean_m, _ = evaluate_rep(
                        X_aug, labels, groups, model, n_folds, task_type,
                        prebuilt_splits=prebuilt_splits)
                    mean_s = _primary_metric(mean_m, task_type, metric_name)
                    delta = mean_s - top_score
                    sign = '+' if delta >= 0 else ''
                    print(f"  {top_rep}+{aug_name} ({X_aug.shape[1]}d): "
                          f"{mean_s:.4f} ({sign}{delta:.4f})")
                    augmentations[f"{top_rep}+{aug_name}"] = {
                        'base': top_rep,
                        'augment': aug_name,
                        'base_dim': X_top.shape[1],
                        'aug_dim': small_reps[aug_name].shape[1],
                        'concat_dim': X_aug.shape[1],
                        'model': top_model,
                        'base_score': round(top_score, 4),
                        'concat_score': round(mean_s, 4),
                        'delta': round(delta, 4),
                    }

    elapsed = time.time() - t0

    # 6. Summary
    all_entries = {**baselines, **ft_baselines}
    all_bests = [(name, e['best']) for name, e in all_entries.items()
                 if e['best'] is not None]
    if all_bests:
        overall = min(all_bests, key=lambda x: x[1]['score']) if not higher_is_better else max(all_bests, key=lambda x: x[1]['score'])
        summary = {
            'best_rep': overall[0],
            'best_score': overall[1]['score'],
            'best_model': overall[1]['model'],
        }
    else:
        summary = {'best_rep': None, 'best_score': None, 'best_model': None}

    result = {
        'metadata': {
            'dataset': dataset_name,
            'domain': data.get('domain', ''),
            'n_samples': len(labels),
            'n_groups': n_groups,
            'n_folds': n_folds,
            'n_reps': len(reps),
            'rep_names': sorted(reps.keys()),
            'models': model_names,
            'primary_metric': metric_name,
            'higher_is_better': higher_is_better,
            'all_metrics': (['accuracy', 'f1_weighted'] +
                            (['auroc'] if task_type == 'classification' else [])
                            if task_type == 'classification'
                            else ['r2', 'rmse', 'spearman', 'pearson', 'ci']),
            'task_type': task_type,
            'seed': seed,
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        },
        'baselines': baselines,
        'finetuned_baselines': ft_baselines or None,
        'augmentations': augmentations or None,
        'summary': summary,
    }

    return result


# =============================================================================
# OUTPUT
# =============================================================================

def print_summary_table(result):
    """Print compact summary table."""
    meta = result['metadata']
    baselines = result['baselines']
    summary = result['summary']
    models = meta['models']
    metric = meta.get('primary_metric', meta.get('metric', 'r2'))

    print(f"\n{'='*70}")
    print(f"SUMMARY: {meta['dataset']} ({meta['n_samples']} samples, "
          f"{meta['n_groups']} groups, {metric})")
    print(f"{'='*70}")

    # Header
    header = f"{'Rep':<25} {'Dim':>5}"
    for m in models:
        header += f" {m:>8}"
    header += f"  {'Best':>8}"
    print(header)
    print("-" * len(header))

    for rep_name in sorted(baselines.keys()):
        r = baselines[rep_name]
        row = f"{rep_name:<25} {r['dimension']:>5}"
        for m in models:
            s = r['scores'].get(m, float('nan'))
            row += f" {s:>8.4f}"
        best = r.get('best')
        if best:
            row += f"  {best['score']:>8.4f}"
        print(row)

    # Fine-tuned scores
    ft = result.get('finetuned_baselines')
    if ft:
        print()
        print("FINE-TUNED:")
        print("-" * len(header))
        for ft_name in sorted(ft.keys()):
            r = ft[ft_name]
            row = f"{ft_name:<25} {'?':>5}"
            for m in models:
                s = r['scores'].get(m, float('nan'))
                row += f" {s:>8.4f}"
            best = r.get('best')
            if best:
                row += f"  {best['score']:>8.4f}"
            print(row)

    # Winner
    print()
    if summary.get('best_rep'):
        print(f"WINNER: {summary['best_rep']} + {summary['best_model']} = "
              f"{summary['best_score']:.4f} ({metric})")
    print(f"  Time: {meta['elapsed_seconds']:.0f}s")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Universal baseline survey: every (rep x dataset x model) cell.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  logd, caco2, herg, qm9,                         (molecular)
    adme_hlm, adme_rlm, adme_mdr1, adme_solubility
  tcga_luad_vs_lusc, tcga_brca_pam50,             (TCGA)
    tcga_brca_survival
  davis, kiba, bindingdb_kd,                       (DTA)
    pdbbind_refined, pdbbind_core
  buchwald_hartwig, suzuki, bh_doyle_plates        (reaction yield)

Shortcuts:
  molecular, tcga, dta, reaction                   (run all in domain)
  everything                                       (run all domains)

Note: Antibody datasets moved to tests/antibody_baseline_survey.py

Examples:
  %(prog)s logd --reps fast
  %(prog)s molecular --output tests/results/universal_survey/
  %(prog)s everything
  %(prog)s tcga --finetune
        """
    )

    parser.add_argument('dataset', type=str,
                       help='Dataset name, domain shortcut, or "everything"')
    parser.add_argument('--reps', type=str, default=None,
                       help='Override rep preset (default: per-dataset from registry)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--finetune', action='store_true',
                       help='Include fine-tuned reps (TCGA: geneformer/scfoundation, '
                            'molecular: ChemBERTa/MolFormer LoRA, DTA: ESM-2 protein LoRA)')
    parser.add_argument('--tune', type=int, default=0, metavar='N',
                       help='Optuna-tune XGBoost with N trials (e.g., --tune 50)')
    parser.add_argument('--l2-norm', action='store_true',
                       help='Add ridge_l2norm model (L2-normalize before Ridge)')
    parser.add_argument('--augment', action='store_true',
                       help='Test augmentation: concat each small rep (<=200D) with top base rep')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (JSON file or directory for multi-dataset)')

    args = parser.parse_args()

    # Resolve dataset list
    ds_input = args.dataset
    if ds_input == 'everything':
        datasets = list(DATASET_REGISTRY.keys())
    elif ds_input in DOMAIN_DATASETS:
        datasets = DOMAIN_DATASETS[ds_input]
    elif ds_input in DATASET_REGISTRY:
        datasets = [ds_input]
    else:
        print(f"ERROR: Unknown dataset '{ds_input}'")
        print(f"Available: {sorted(DATASET_REGISTRY.keys())}")
        print(f"Domains: {sorted(DOMAIN_DATASETS.keys())}, everything")
        return 1

    print(f"Universal Baseline Survey")
    print(f"  Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"  Folds: {args.n_folds}, Finetune: {args.finetune}")

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
        try:
            result = run_survey(
                ds, args.reps, args.n_folds,
                args.finetune, args.seed,
                tune_trials=args.tune, l2_norm=args.l2_norm,
                augment=args.augment,
            )
            result = convert_numpy(result)
            print_summary_table(result)
            all_results[ds] = result
            pass_count += 1

            # Save per-dataset JSON
            if output_dir:
                out_path = output_dir / f"{ds}.json"
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
            s = r['summary']
            m = r['metadata']
            if s.get('best_rep'):
                print(f"  {ds:<30} {s['best_rep']:<20} + {s['best_model']:<10} = "
                      f"{s['best_score']:.4f} ({m.get('primary_metric', m.get('metric', 'r2'))})")
            else:
                print(f"  {ds:<30} FAILED")

    print(f"\nDone: {pass_count} passed, {fail_count} failed")
    return 1 if fail_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
