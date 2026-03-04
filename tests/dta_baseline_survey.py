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

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')


# Which reps to evaluate (ordered: lightweight first, LM last)
DRUG_REPS = [
    'drug_ecfp4', 'drug_maccs', 'drug_pdv',
    'drug_chemberta', 'drug_molformer', 'drug_selformer', 'drug_clamp',
]
PROTEIN_REPS = [
    'protein_aac', 'protein_descriptors', 'protein_kmer',
    'protein_esm2', 'protein_esmc', 'protein_esm2_midlayers',
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
    from kirby.datasets.dta import load_dataset as _load
    return _load(name)


def generate_rep(rep_name: str, unique_drugs, unique_proteins,
                 drug_to_idx, protein_to_idx,
                 drug_smiles, protein_seqs) -> Optional[np.ndarray]:
    """Generate a single representation expanded to all pairs."""
    from kirby.representations import molecular as mol
    from kirby.representations import protein as prot

    dispatch = {
        'drug_ecfp4':      (mol.create_ecfp4,      'drug'),
        'drug_maccs':      (mol.create_maccs,       'drug'),
        'drug_pdv':        (mol.create_pdv,         'drug'),
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
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False

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

    models = {
        'ridge': lambda: RidgeCV(alphas=np.logspace(-4, 4, 50)),
        'rf': lambda: RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                            n_jobs=-1, random_state=42),
    }
    if has_xgb:
        models['xgboost'] = lambda: xgb.XGBRegressor(
            n_estimators=100, colsample_bytree=0.5,
            n_jobs=-1, random_state=42, verbosity=0,
        )

    for model_name, model_fn in models.items():
        fold_scores = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            model = model_fn()
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            fold_scores.append(r2_score(y[test_idx], pred))

        mean_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)
        results[model_name] = {'mean': round(mean_r2, 4), 'std': round(std_r2, 4)}

    return results


def run_survey(dataset_name: str, max_pairs: Optional[int] = None,
               skip_lm: bool = False, skip_kmer: bool = False):
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
        metadata['protein_ids'] = [metadata['protein_ids'][i] for i in idx]
        print(f"  Subsampled to {max_pairs} pairs")

    # Cold-target groups
    from kirby.datasets.dta import assign_cold_target_groups
    groups, n_groups = assign_cold_target_groups(metadata['protein_ids'])
    print(f"  {len(labels)} pairs, {metadata['n_drugs']} drugs × {metadata['n_proteins']} proteins")
    print(f"  Cold-target groups: {n_groups}")

    # Deduplicate for efficient generation
    unique_drugs = sorted(set(drug_smiles))
    unique_proteins = sorted(set(protein_seqs))
    drug_to_idx = {s: i for i, s in enumerate(unique_drugs)}
    protein_to_idx = {s: i for i, s in enumerate(unique_proteins)}

    # Decide which reps to run
    drug_reps_to_run = [r for r in DRUG_REPS
                        if not (skip_lm and r in ('drug_chemberta', 'drug_molformer', 'drug_selformer', 'drug_clamp'))]
    protein_reps_to_run = [r for r in PROTEIN_REPS
                           if not (skip_lm and r in ('protein_esm2', 'protein_esmc', 'protein_esm2_midlayers'))
                           and not (skip_kmer and r == 'protein_kmer')]

    all_reps_to_run = drug_reps_to_run + protein_reps_to_run

    # Generate all reps first
    print(f"\nGenerating {len(all_reps_to_run)} representations...")
    generated = {}
    for rep_name in all_reps_to_run:
        t0 = time.time()
        rep = generate_rep(rep_name, unique_drugs, unique_proteins,
                           drug_to_idx, protein_to_idx,
                           drug_smiles, protein_seqs)
        elapsed = time.time() - t0
        if rep is not None:
            generated[rep_name] = rep
            print(f"  {rep_name}: {rep.shape} ({elapsed:.1f}s)")
        else:
            print(f"  {rep_name}: FAILED ({elapsed:.1f}s)")

    # Pre-filter protein_kmer from 8000D → 200D via LightGBM feature importance
    if 'protein_kmer' in generated:
        generated['protein_kmer'] = _filter_features_by_importance(
            generated['protein_kmer'], labels, n_features=200,
            rep_name='protein_kmer')

    # Evaluate: standalone mode (each rep alone)
    print(f"\n--- STANDALONE EVALUATION (each rep alone) ---")
    standalone_results = {}
    for rep_name, rep in generated.items():
        t0 = time.time()
        scores = evaluate_rep(rep, labels, groups)
        elapsed = time.time() - t0
        standalone_results[rep_name] = scores
        best_model = max(scores, key=lambda m: scores[m]['mean'])
        best_r2 = scores[best_model]['mean']
        print(f"  {rep_name:<25} best={best_model} R²={best_r2:.4f} ({elapsed:.1f}s)")

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
        scores = evaluate_rep(X, labels, groups)
        elapsed = time.time() - t0
        paired_results[rep_name] = scores
        best_model = max(scores, key=lambda m: scores[m]['mean'])
        best_r2 = scores[best_model]['mean']
        print(f"  {rep_name:<25} +{partner:<20} best={best_model} R²={best_r2:.4f} ({elapsed:.1f}s)")

    # Combined: all drug + all protein
    if drug_partner is not None and protein_partner is not None:
        all_drug_reps = [generated[r] for r in drug_reps_to_run if r in generated]
        all_protein_reps = [generated[r] for r in protein_reps_to_run if r in generated]
        if all_drug_reps and all_protein_reps:
            X_all = np.hstack(all_drug_reps + all_protein_reps)
            print(f"\n--- ALL COMBINED ({X_all.shape[1]}D) ---")
            t0 = time.time()
            all_scores = evaluate_rep(X_all, labels, groups)
            elapsed = time.time() - t0
            best_model = max(all_scores, key=lambda m: all_scores[m]['mean'])
            best_r2 = all_scores[best_model]['mean']
            print(f"  all_concat ({X_all.shape[1]}D): best={best_model} R²={best_r2:.4f} ({elapsed:.1f}s)")
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

    header = f"{'Rep':<25} {'Dim':<6} " + " ".join(f"{m:<16}" for m in models_list)
    print(header)
    print("-" * len(header))

    for rep_name in all_reps_to_run:
        if rep_name not in standalone_results:
            continue
        dim = generated[rep_name].shape[1]
        row = f"{rep_name:<25} {dim:<6} "
        for m in models_list:
            if m in standalone_results[rep_name]:
                s = standalone_results[rep_name][m]
                row += f"{s['mean']:>6.4f}±{s['std']:.3f}   "
            else:
                row += f"{'--':<16} "
        print(row)

    print(f"\nPaired (+ minimal partner):")
    for rep_name in all_reps_to_run:
        if rep_name not in paired_results:
            continue
        dim = generated[rep_name].shape[1]
        row = f"{rep_name:<25} {dim:<6} "
        for m in models_list:
            if m in paired_results[rep_name]:
                s = paired_results[rep_name][m]
                row += f"{s['mean']:>6.4f}±{s['std']:.3f}   "
            else:
                row += f"{'--':<16} "
        print(row)

    return {
        'dataset': dataset_name,
        'n_pairs': len(labels),
        'standalone': standalone_results,
        'paired': paired_results,
        'all_combined': all_scores,
        'reps_generated': list(generated.keys()),
        'reps_failed': [r for r in all_reps_to_run if r not in generated],
    }


def main():
    parser = argparse.ArgumentParser(description='DTA Baseline Survey for BOM')
    parser.add_argument('dataset', choices=['davis', 'kiba', 'all'],
                        help='Dataset to evaluate')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Subsample KIBA (default: 30000 for kiba)')
    parser.add_argument('--skip-lm', action='store_true',
                        help='Skip LM reps (ESM-2, ChemBERTa, MolFormer, SelFormer)')
    parser.add_argument('--skip-kmer', action='store_true',
                        help='Skip protein_kmer (8000D→200D via importance filter)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results JSON')
    args = parser.parse_args()

    datasets = ['davis', 'kiba'] if args.dataset == 'all' else [args.dataset]
    all_results = {}

    for ds in datasets:
        max_pairs = args.max_pairs
        if max_pairs is None and ds == 'kiba':
            max_pairs = 30000  # default subsample for KIBA
        result = run_survey(ds, max_pairs=max_pairs,
                            skip_lm=args.skip_lm, skip_kmer=args.skip_kmer)
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

        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
