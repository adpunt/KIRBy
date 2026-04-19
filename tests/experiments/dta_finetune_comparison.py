#!/usr/bin/env python3
"""DTA Fine-Tuning Comparison: Does LoRA fine-tuning improve DTA prediction?

Compares frozen vs LoRA-adapted protein/drug language models on drug-target
affinity, using frozen ECFP4 + XGBoost as the downstream model.

Methods:
  1. Frozen ECFP4 + frozen ESM-2 + XGB           (baseline)
  2. Frozen ECFP4 + LoRA ESM-2 + XGB             (protein FT)
  3. Frozen ECFP4 + LoRA ESM-C + XGB             (protein FT)
  4. Frozen ECFP4 + LoRA ESM3 + XGB              (protein FT)
  5. Frozen ECFP4 + LoRA SaProt + XGB            (protein FT)
  6. LoRA ChemBERTa + frozen ESM-2 + XGB         (drug FT)
  7. LoRA MolFormer + frozen ESM-2 + XGB         (drug FT)
  8. LoRA ChemBERTa + LoRA ESM-2 + XGB           (joint FT)

All evaluated with cold-target GroupKFold (5-fold), CI as primary metric.

Usage:
    python tests/experiments/dta_finetune_comparison.py davis
    python tests/experiments/dta_finetune_comparison.py kiba --max-pairs 30000
    python tests/experiments/dta_finetune_comparison.py davis --n-folds 2
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# ── Metrics ──────────────────────────────────────────────────────────────────

def concordance_index(y_true, y_pred):
    """Concordance Index (CI) via Kendall's tau: CI = (tau + 1) / 2."""
    from scipy.stats import kendalltau
    tau, _ = kendalltau(y_true, y_pred)
    return (tau + 1) / 2


def compute_metrics(y_true, y_pred):
    """Compute CI, R-squared, RMSE, and Spearman rho."""
    ci = concordance_index(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rho = float(spearmanr(y_true, y_pred).correlation)
    return {'ci': ci, 'r2': r2, 'rmse': rmse, 'spearman': rho}


# ── Frozen representation helpers ────────────────────────────────────────────

def _extract_frozen_ecfp4(drug_smiles: List[str]) -> np.ndarray:
    """Deduplicated frozen ECFP4 extraction, expanded to per-pair."""
    from kirby.representations.molecular import create_ecfp4

    unique_drugs = sorted(set(drug_smiles))
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    unique_feats = create_ecfp4(unique_drugs)  # (n_unique, 2048)
    pair_indices = np.array([drug_to_idx[s] for s in drug_smiles])
    return unique_feats[pair_indices]


def _extract_frozen_protein(protein_seqs: List[str], model: str) -> np.ndarray:
    """Deduplicated frozen protein LM extraction, expanded to per-pair."""
    from kirby.representations.protein import (
        create_esm2_protein_embeddings,
        create_esmc_protein_embeddings,
        create_esm3_protein_embeddings,
        create_saprot_protein_embeddings,
    )

    _factories = {
        'esm2': create_esm2_protein_embeddings,
        'esmc': create_esmc_protein_embeddings,
        'esm3': create_esm3_protein_embeddings,
        'saprot': create_saprot_protein_embeddings,
    }
    if model not in _factories:
        raise ValueError(f"Unknown protein model '{model}'. "
                         f"Available: {list(_factories.keys())}")

    unique_prots = sorted(set(protein_seqs))
    prot_to_idx = {s: i for i, s in enumerate(unique_prots)}
    unique_feats = _factories[model](unique_prots)
    pair_indices = np.array([prot_to_idx[s] for s in protein_seqs])
    return unique_feats[pair_indices]


# ── Method definitions ───────────────────────────────────────────────────────

# Each method is a dict with:
#   label: display name
#   category: 'baseline' | 'protein_ft' | 'drug_ft' | 'joint_ft'
#   run_fn: callable(fold_data) -> y_pred  (defined later)

METHODS = [
    {'key': 'frozen_ecfp4_esm2',      'label': 'ECFP4 + ESM-2 (frozen)',      'category': 'baseline'},
    {'key': 'ecfp4_lora_esm2',        'label': 'ECFP4 + LoRA-ESM-2',         'category': 'protein_ft'},
    {'key': 'ecfp4_lora_esmc',        'label': 'ECFP4 + LoRA-ESM-C',         'category': 'protein_ft'},
    {'key': 'ecfp4_lora_esm3',        'label': 'ECFP4 + LoRA-ESM3',          'category': 'protein_ft'},
    {'key': 'ecfp4_lora_saprot',      'label': 'ECFP4 + LoRA-SaProt',        'category': 'protein_ft'},
    {'key': 'lora_chemberta_esm2',    'label': 'LoRA-ChemBERTa + ESM-2',     'category': 'drug_ft'},
    {'key': 'lora_molformer_esm2',    'label': 'LoRA-MolFormer + ESM-2',     'category': 'drug_ft'},
    {'key': 'lora_chemberta_lora_esm2', 'label': 'LoRA-ChemBERTa + LoRA-ESM-2', 'category': 'joint_ft'},
]


def _make_xgb():
    """Create XGBoost regressor, falling back to RF if unavailable."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=200, max_depth=6, colsample_bytree=0.5,
            learning_rate=0.1, subsample=0.8,
            n_jobs=-1, random_state=42, verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        print("    [XGBoost unavailable, using RF fallback]")
        return RandomForestRegressor(
            n_estimators=200, max_features='sqrt',
            n_jobs=-1, random_state=42,
        )


def _train_and_predict(X_train, y_train, X_test):
    """Scale, fit XGB, predict."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = _make_xgb()
    model.fit(X_tr, y_train)
    return model.predict(X_te)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('dataset', choices=['davis', 'kiba', 'bindingdb_kd'],
                        help='DTA dataset (davis | kiba | bindingdb_kd)')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Subsample pairs (default: 30000 for KIBA)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='LoRA fine-tuning epochs per fold (default: 5)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable LoRA factory caching')
    parser.add_argument('--output', default='tests/results',
                        help='Output directory (default: tests/results)')
    args = parser.parse_args()

    cache = not args.no_cache

    # ── Load dataset ─────────────────────────────────────────────────────
    from kirby.datasets.dta import load_dataset, assign_cold_target_groups
    from kirby.registry import crosscheck_task_type

    print(f"Loading {args.dataset}...")
    data = load_dataset(args.dataset)
    drug_smiles = data['drug_smiles']
    protein_seqs = data['protein_seqs']
    labels = data['labels']
    metadata = data['metadata']
    # Registry cross-check: DTA datasets are all regression (ci metric).
    crosscheck_task_type(args.dataset, data.get('task_type', 'regression'))

    # Subsample if needed
    max_pairs = args.max_pairs
    if max_pairs is None and args.dataset == 'kiba':
        max_pairs = 30000
    if max_pairs and len(labels) > max_pairs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(labels), max_pairs, replace=False)
        idx.sort()
        drug_smiles = [drug_smiles[i] for i in idx]
        protein_seqs = [protein_seqs[i] for i in idx]
        labels = labels[idx]
        metadata['protein_ids'] = [metadata['protein_ids'][i] for i in idx]
        metadata['drug_ids'] = [metadata['drug_ids'][i] for i in idx]
        print(f"  Subsampled to {max_pairs} pairs")

    groups, n_groups = assign_cold_target_groups(metadata['protein_ids'])
    n = len(labels)
    n_drugs = len(set(drug_smiles))
    n_prots = len(set(protein_seqs))

    print(f"\n{'='*70}")
    print(f"DTA FINE-TUNING COMPARISON: {args.dataset.upper()}")
    print(f"  {n} pairs, {n_drugs} drugs, {n_prots} proteins, {n_groups} target groups")
    print(f"  Folds: {args.n_folds}, FT epochs: {args.epochs}, cache: {cache}")
    print(f"{'='*70}")

    # ── Pre-compute frozen representations (deduplicated) ────────────────
    # These are shared across methods — compute once, index per fold.

    print("\nPre-computing frozen representations...")

    print("  Frozen ECFP4...", end='', flush=True)
    t0 = time.time()
    frozen_ecfp4 = _extract_frozen_ecfp4(drug_smiles)
    print(f" ({frozen_ecfp4.shape[1]}D, {time.time()-t0:.1f}s)")

    print("  Frozen ESM-2...", end='', flush=True)
    t0 = time.time()
    try:
        frozen_esm2 = _extract_frozen_protein(protein_seqs, 'esm2')
        print(f" ({frozen_esm2.shape[1]}D, {time.time()-t0:.1f}s)")
    except Exception as e:
        print(f" FAILED: {e}")
        frozen_esm2 = None

    # ── Create LoRA factories ────────────────────────────────────────────
    print("\nCreating LoRA factories...")

    from kirby.finetuning.dta_protein_lora import make_dta_protein_lora_factory
    from kirby.finetuning.molecular_lora import make_molecular_lora_factory

    # Protein LoRA factories
    prot_lora_factories = {}
    for prot_model in ['esm2', 'esmc', 'esm3', 'saprot']:
        try:
            prot_lora_factories[prot_model] = make_dta_protein_lora_factory(
                protein_seqs, drug_smiles, args.dataset,
                base_model=prot_model, epochs=args.epochs, cache=cache,
            )
            print(f"  LoRA-{prot_model}: ready")
        except Exception as e:
            print(f"  LoRA-{prot_model}: FAILED ({e})")
            prot_lora_factories[prot_model] = None

    # Drug LoRA factories
    drug_lora_factories = {}
    for drug_model in ['chemberta', 'molformer']:
        try:
            drug_lora_factories[drug_model] = make_molecular_lora_factory(
                drug_smiles, args.dataset,
                base_model=drug_model, epochs=args.epochs, cache=cache,
            )
            print(f"  LoRA-{drug_model}: ready")
        except Exception as e:
            print(f"  LoRA-{drug_model}: FAILED ({e})")
            drug_lora_factories[drug_model] = None

    # ── Cross-validation loop ────────────────────────────────────────────

    effective_folds = min(args.n_folds, n_groups)
    if effective_folds < args.n_folds:
        print(f"\n  NOTE: Only {n_groups} target groups, using {effective_folds} folds")

    gkf = GroupKFold(n_splits=effective_folds)
    splits = list(gkf.split(labels, labels, groups))

    results = {m['key']: [] for m in METHODS}

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'─'*60}")
        print(f"Fold {fold+1}/{effective_folds} "
              f"(train={len(train_idx)}, test={len(test_idx)}, "
              f"test proteins={len(set(groups[test_idx]))})")
        print(f"{'─'*60}")

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Cache per-fold frozen slices
        ecfp4_train = frozen_ecfp4[train_idx]
        ecfp4_test = frozen_ecfp4[test_idx]

        if frozen_esm2 is not None:
            esm2_train = frozen_esm2[train_idx]
            esm2_test = frozen_esm2[test_idx]
        else:
            esm2_train = esm2_test = None

        # ── Method 1: Frozen baseline ────────────────────────────────────
        key = 'frozen_ecfp4_esm2'
        print(f"  [{key}]", end='', flush=True)
        t0 = time.time()
        try:
            if esm2_train is None:
                raise RuntimeError("Frozen ESM-2 unavailable")
            X_train = np.hstack([ecfp4_train, esm2_train])
            X_test = np.hstack([ecfp4_test, esm2_test])
            y_pred = _train_and_predict(X_train, y_train, X_test)
            m = compute_metrics(y_test, y_pred)
            print(f" CI={m['ci']:.4f} R2={m['r2']:.4f} rho={m['spearman']:.4f}"
                  f" ({time.time()-t0:.1f}s)")
            results[key].append(m)
        except Exception as e:
            print(f" FAILED: {e}")
            results[key].append(_nan_metrics())

        # ── Methods 2-5: Protein LoRA (frozen ECFP4 + LoRA-protein) ─────
        for prot_model, method_key in [
            ('esm2',   'ecfp4_lora_esm2'),
            ('esmc',   'ecfp4_lora_esmc'),
            ('esm3',   'ecfp4_lora_esm3'),
            ('saprot', 'ecfp4_lora_saprot'),
        ]:
            print(f"  [{method_key}]", end='', flush=True)
            t0 = time.time()
            try:
                factory = prot_lora_factories.get(prot_model)
                if factory is None:
                    raise RuntimeError(f"LoRA-{prot_model} factory unavailable")
                prot_train, prot_test = factory(train_idx, test_idx, y_train)
                X_train = np.hstack([ecfp4_train, prot_train])
                X_test = np.hstack([ecfp4_test, prot_test])
                y_pred = _train_and_predict(X_train, y_train, X_test)
                m = compute_metrics(y_test, y_pred)
                print(f" CI={m['ci']:.4f} R2={m['r2']:.4f} rho={m['spearman']:.4f}"
                      f" ({time.time()-t0:.1f}s)")
                results[method_key].append(m)
            except Exception as e:
                print(f" FAILED: {e}")
                results[method_key].append(_nan_metrics())

        # ── Methods 6-7: Drug LoRA (LoRA-drug + frozen ESM-2) ────────────
        for drug_model, method_key in [
            ('chemberta', 'lora_chemberta_esm2'),
            ('molformer', 'lora_molformer_esm2'),
        ]:
            print(f"  [{method_key}]", end='', flush=True)
            t0 = time.time()
            try:
                factory = drug_lora_factories.get(drug_model)
                if factory is None:
                    raise RuntimeError(f"LoRA-{drug_model} factory unavailable")
                if esm2_train is None:
                    raise RuntimeError("Frozen ESM-2 unavailable")
                drug_train, drug_test = factory(train_idx, test_idx, y_train)
                X_train = np.hstack([drug_train, esm2_train])
                X_test = np.hstack([drug_test, esm2_test])
                y_pred = _train_and_predict(X_train, y_train, X_test)
                m = compute_metrics(y_test, y_pred)
                print(f" CI={m['ci']:.4f} R2={m['r2']:.4f} rho={m['spearman']:.4f}"
                      f" ({time.time()-t0:.1f}s)")
                results[method_key].append(m)
            except Exception as e:
                print(f" FAILED: {e}")
                results[method_key].append(_nan_metrics())

        # ── Method 8: Joint LoRA (LoRA-ChemBERTa + LoRA-ESM-2) ──────────
        key = 'lora_chemberta_lora_esm2'
        print(f"  [{key}]", end='', flush=True)
        t0 = time.time()
        try:
            drug_factory = drug_lora_factories.get('chemberta')
            prot_factory = prot_lora_factories.get('esm2')
            if drug_factory is None:
                raise RuntimeError("LoRA-ChemBERTa factory unavailable")
            if prot_factory is None:
                raise RuntimeError("LoRA-ESM-2 factory unavailable")
            drug_train, drug_test = drug_factory(train_idx, test_idx, y_train)
            prot_train, prot_test = prot_factory(train_idx, test_idx, y_train)
            X_train = np.hstack([drug_train, prot_train])
            X_test = np.hstack([drug_test, prot_test])
            y_pred = _train_and_predict(X_train, y_train, X_test)
            m = compute_metrics(y_test, y_pred)
            print(f" CI={m['ci']:.4f} R2={m['r2']:.4f} rho={m['spearman']:.4f}"
                  f" ({time.time()-t0:.1f}s)")
            results[key].append(m)
        except Exception as e:
            print(f" FAILED: {e}")
            results[key].append(_nan_metrics())

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"DTA FINE-TUNING COMPARISON: {args.dataset.upper()} "
          f"({n} pairs, cold-target, {effective_folds}-fold)")
    print(f"{'='*85}")
    print(f"{'Method':<38} {'Category':<12} {'CI':>14} {'R2':>14} {'Spearman':>10}")
    print(f"{'─'*85}")

    summary = {}
    for method_info in METHODS:
        key = method_info['key']
        label = method_info['label']
        cat = method_info['category']
        fold_scores = results[key]

        valid = [s for s in fold_scores if not np.isnan(s['ci'])]
        if valid:
            ci_vals = [s['ci'] for s in valid]
            r2_vals = [s['r2'] for s in valid]
            rho_vals = [s['spearman'] for s in valid]
            ci_str = f"{np.mean(ci_vals):.4f}+/-{np.std(ci_vals):.4f}"
            r2_str = f"{np.mean(r2_vals):.4f}+/-{np.std(r2_vals):.4f}"
            rho_str = f"{np.mean(rho_vals):.4f}"
            print(f"{label:<38} {cat:<12} {ci_str:>14} {r2_str:>14} {rho_str:>10}")
            summary[key] = {
                'label': label,
                'category': cat,
                'ci_mean': round(float(np.mean(ci_vals)), 4),
                'ci_std': round(float(np.std(ci_vals)), 4),
                'r2_mean': round(float(np.mean(r2_vals)), 4),
                'r2_std': round(float(np.std(r2_vals)), 4),
                'spearman_mean': round(float(np.mean(rho_vals)), 4),
                'spearman_std': round(float(np.std(rho_vals)), 4),
                'rmse_mean': round(float(np.mean([s['rmse'] for s in valid])), 4),
                'n_valid_folds': len(valid),
            }
        else:
            print(f"{label:<38} {cat:<12} {'FAILED':>14} {'FAILED':>14} {'FAILED':>10}")
            summary[key] = {
                'label': label,
                'category': cat,
                'ci_mean': None,
                'n_valid_folds': 0,
            }

    # ── Delta vs baseline ────────────────────────────────────────────────
    baseline_ci = summary.get('frozen_ecfp4_esm2', {}).get('ci_mean')
    if baseline_ci is not None:
        print(f"\n{'─'*85}")
        print(f"Delta CI vs frozen baseline (CI={baseline_ci:.4f}):")
        for method_info in METHODS[1:]:
            key = method_info['key']
            label = method_info['label']
            s = summary.get(key, {})
            if s.get('ci_mean') is not None:
                delta = s['ci_mean'] - baseline_ci
                sign = '+' if delta >= 0 else ''
                print(f"  {label:<38} {sign}{delta:.4f}")

    # ── Save JSON results ────────────────────────────────────────────────
    out_dir = Path(args.output) / 'dta_finetune_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{args.dataset}.json'

    output = {
        'dataset': args.dataset,
        'n_pairs': n,
        'n_drugs': n_drugs,
        'n_proteins': n_prots,
        'n_target_groups': int(n_groups),
        'n_folds': effective_folds,
        'ft_epochs': args.epochs,
        'protocol': 'cold-target GroupKFold',
        'primary_metric': 'CI',
        'summary': summary,
        'per_fold': {k: v for k, v in results.items()},
    }

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(out_file, 'w') as f:
        json.dump(_convert(output), f, indent=2)
    print(f"\n[saved {out_file}]")


def _nan_metrics():
    """Return a metrics dict filled with NaN."""
    return {'ci': float('nan'), 'r2': float('nan'),
            'rmse': float('nan'), 'spearman': float('nan')}


if __name__ == '__main__':
    main()
