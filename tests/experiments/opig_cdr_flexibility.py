#!/usr/bin/env python3
"""
CDR Flexibility Prediction Experiment (OPIG — Spoendlin et al. 2025)

Replicates Spoendlin et al. 2025 "Predicting the conformational flexibility
of antibody and T-cell receptor CDRs" — tests whether sequence-based antibody
reps can predict CDR flexibility, and whether hybrid reps can match the
structure-based ITsFlexible GNN.

Scientific questions:
  1. Can sequence-based reps predict CDR3 flexibility from VH/VL?
  2. Does hybrid augmentation help for flexibility classification?
  3. How far are sequence-based approaches from the structure-based GNN?
  4. Which CDR types are easiest/hardest to classify?

CDR types:
  - CDRH3: heavy chain CDR3 (most variable, hardest to predict)
  - CDRL3: light chain CDR3
  - CDRA3: TCR alpha chain CDR3
  - CDRB3: TCR beta chain CDR3

Paper baselines (Spoendlin et al. 2025, Nature Machine Intelligence):
  ITsFlexible GNN:    ROC AUC = 0.82, PR AUC = 0.60 (overall crystal)
  Outperforms AlphaFold2 pLDDT and ABB2 RMSPE baselines.
  "Only method that consistently achieves high accuracy across all test sets"

Usage:
    # Smoke test (~2 min, CDRH3 only, fast reps)
    python tests/experiments/opig_cdr_flexibility.py \\
        --cdr-types CDRH3 --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation (all antibody CDR types)
    python tests/experiments/opig_cdr_flexibility.py --cdr-types CDRH3,CDRL3

Reference:
    Spoendlin et al. (2025). Nature Machine Intelligence.
    https://github.com/oxpig/ITsFlexible
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, average_precision_score,
    matthews_corrcoef, accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from experiments._shared import convert_numpy, get_model_factories

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CDR_TYPES = ['CDRH3', 'CDRL3']

DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',           # classical
    'ablang2', 'esm2', 'antiberty',      # LMs
    'igt5', 'prot_t5',                    # additional LMs
    'paratope',                            # structural (sequence-based)
]

MODELS_LIST = ['ridge', 'rf']

# Paper baselines
PAPER_BASELINES = {
    'ITsFlexible GNN (crystal)': {'roc_auc': 0.82, 'pr_auc': 0.60},
}


# =============================================================================
# EVALUATION
# =============================================================================

def _is_binary_rep(X):
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


def evaluate_rep_clf(X, labels, groups, model, n_folds):
    """Evaluate one (rep, model) for flexibility classification.

    Returns dict with roc_auc, pr_auc, balanced_accuracy, mcc, accuracy.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if groups is not None and len(np.unique(groups)) >= n_folds:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(cv.split(X, labels, groups=groups))
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(cv.split(X, labels))

    is_binary = _is_binary_rep(X)
    roc_scores, pr_scores, bacc_scores, mcc_scores, acc_scores = [], [], [], [], []

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        if not is_binary:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        try:
            m = clone(model)
            m.fit(X_train, y_train)
        except Exception:
            continue

        preds = m.predict(X_val)
        acc_scores.append(accuracy_score(y_val, preds))
        bacc_scores.append(balanced_accuracy_score(y_val, preds))
        mcc_scores.append(matthews_corrcoef(y_val, preds))

        # Scores for AUROC and PR-AUC
        try:
            if hasattr(m, 'decision_function'):
                scores = m.decision_function(X_val)
            elif hasattr(m, 'predict_proba'):
                scores = m.predict_proba(X_val)[:, 1]
            else:
                scores = preds.astype(float)
            roc_scores.append(roc_auc_score(y_val, scores))
            pr_scores.append(average_precision_score(y_val, scores))
        except Exception:
            pass

    if not acc_scores:
        return None

    result = {
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
        'balanced_accuracy': {'mean': np.mean(bacc_scores), 'std': np.std(bacc_scores)},
        'mcc': {'mean': np.mean(mcc_scores), 'std': np.std(mcc_scores)},
    }
    if roc_scores:
        result['roc_auc'] = {'mean': np.mean(roc_scores), 'std': np.std(roc_scores)}
    if pr_scores:
        result['pr_auc'] = {'mean': np.mean(pr_scores), 'std': np.std(pr_scores)}
    return result


def evaluate_cdr_type(cdr_types, rep_names, n_folds):
    """Load CDR data, generate reps, evaluate all.

    Returns dict with per-rep results.
    """
    from kirby.datasets.itsflexible import load_itsflexible

    print(f"\n{'=' * 60}")
    print(f"CDR types: {cdr_types}")
    print(f"{'=' * 60}")

    data = load_itsflexible(mode='cdr', cdr_types=cdr_types)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    n = len(labels)
    n_flex = int((labels == 1).sum())

    print(f"  {n} sequences, {n_flex} flexible ({100*n_flex/n:.1f}%)")

    # Groups by CDR3 sequence identity (unique sequence → group ID)
    cdr_seqs = data.get('cdr_seqs', heavy_seqs)
    unique_seqs = {}
    group_ids = np.zeros(len(cdr_seqs), dtype=int)
    for i, s in enumerate(cdr_seqs):
        if s not in unique_seqs:
            unique_seqs[s] = len(unique_seqs)
        group_ids[i] = unique_seqs[s]
    groups = group_ids

    # Generate representations
    from kirby.representations.antibody import ANTIBODY_REPS
    reps = {}
    for rep_name in rep_names:
        if rep_name not in ANTIBODY_REPS:
            print(f"  WARNING: unknown rep '{rep_name}', skipping")
            continue
        try:
            X = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 2 and X.shape[0] == n:
                reps[rep_name] = X
                print(f"  {rep_name}: {X.shape[1]}D")
        except Exception as e:
            print(f"  {rep_name}: FAILED ({e})")

    if not reps:
        print("  No reps available, skipping")
        return None

    # Build hybrid
    # No augmentation reps for CDR flexibility classification
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        from kirby.hybrid import apply_feature_selection
        search = find_best_hybrid(
            base_reps=reps,
            labels=labels, groups=groups,
            metric='accuracy',
            task_type='classification',
            augmentation_strategy='uncapped',
            speed='fast', return_fitted=True, verbose=False, cache=True)
        if search.fitted_result is not None:
            fi = {r: {'selected_indices': idx}
                  for r, idx in search.fitted_result.selected_indices.items()}
            reps['kirby_hybrid'] = apply_feature_selection(reps, fi)
            print(f"  kirby_hybrid: {reps['kirby_hybrid'].shape[1]}D (hybrid)")
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    # Evaluate
    models = get_model_factories(MODELS_LIST, 'classification', n_samples=n)
    print(f"\n  Evaluating {len(reps)} reps × {len(models)} models × {n_folds}-fold CV...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            res = evaluate_rep_clf(X, labels, groups, factory(), n_folds)
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                roc = res.get('roc_auc', {}).get('mean', float('nan'))
                pr = res.get('pr_auc', {}).get('mean', float('nan'))
                tag = ' ***' if rep_name == 'kirby_hybrid' else ''
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"ROC={roc:.4f}  PR={pr:.4f}  ({elapsed:.1f}s){tag}")
        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'n_samples': n,
        'n_flexible': n_flex,
        'n_rigid': n - n_flex,
        'cdr_types': cdr_types,
        'results': results,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print comparison table with paper baselines."""
    print(f"\n{'=' * 100}")
    print(f"CDR FLEXIBILITY EXPERIMENT RESULTS (OPIG — Spoendlin et al. 2025)")
    print(f"{'=' * 100}")

    # Paper baselines
    print(f"\n  Paper baselines:")
    for method, metrics in PAPER_BASELINES.items():
        roc = metrics.get('roc_auc', float('nan'))
        pr = metrics.get('pr_auc', float('nan'))
        print(f"    {method}: ROC AUC = {roc:.2f}, PR AUC = {pr:.2f}")

    for key, ds in all_results.items():
        results = ds['results']
        cdr_types = ds['cdr_types']

        print(f"\n  --- CDR types: {cdr_types} ({ds['n_samples']} samples, "
              f"{ds['n_flexible']} flexible) ---")

        print(f"  {'Rep':25s} {'Model':8s} {'ROC AUC':>8s} {'PR AUC':>8s} "
              f"{'MCC':>8s} {'BalAcc':>8s}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        rows = []
        for rep_name, rep_data in results.items():
            for model_name, metrics in rep_data['models'].items():
                roc = metrics.get('roc_auc', {}).get('mean', float('nan'))
                pr = metrics.get('pr_auc', {}).get('mean', float('nan'))
                mcc = metrics.get('mcc', {}).get('mean', float('nan'))
                bacc = metrics.get('balanced_accuracy', {}).get('mean', float('nan'))
                rows.append((rep_name, model_name, roc, pr, mcc, bacc))

        rows.sort(key=lambda r: (r[2] if not np.isnan(r[2]) else 0), reverse=True)

        for rep_name, model_name, roc, pr, mcc, bacc in rows:
            tag = ' ***' if rep_name == 'kirby_hybrid' else ''
            roc_s = f"{roc:.4f}" if not np.isnan(roc) else "—"
            pr_s = f"{pr:.4f}" if not np.isnan(pr) else "—"
            print(f"  {rep_name:25s} {model_name:8s} {roc_s:>8s} {pr_s:>8s} "
                  f"{mcc:8.4f} {bacc:8.4f}{tag}")

        # Gap analysis
        best_roc = max(
            (m.get('roc_auc', {}).get('mean', 0)
             for rd in results.values() for m in rd['models'].values()),
            default=0
        )
        gnn_roc = PAPER_BASELINES['ITsFlexible GNN (crystal)']['roc_auc']
        gap = gnn_roc - best_roc
        print(f"\n  Best KIRBy ROC AUC: {best_roc:.4f}")
        print(f"  ITsFlexible GNN:    {gnn_roc:.2f}")
        print(f"  Gap:                {gap:+.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CDR flexibility experiment (Spoendlin 2025). '
                    'Tests antibody reps for CDR3 flexibility classification.')
    parser.add_argument(
        '--cdr-types', type=str, default=None,
        help='Comma-separated CDR types (default: CDRH3,CDRL3)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 9)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of stratified CV folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_cdr_flexibility.json)')

    args = parser.parse_args()

    cdr_types = (args.cdr_types.split(',') if args.cdr_types
                 else DEFAULT_CDR_TYPES)
    rep_names = (args.reps.split(',') if args.reps else DEFAULT_REPS)

    print("=" * 70)
    print("CDR FLEXIBILITY EXPERIMENT (Spoendlin et al. 2025)")
    print(f"  Paper: Nature Machine Intelligence")
    print(f"  CDR types: {cdr_types}")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}

    # Evaluate each CDR type separately + combined
    for cdr_type in cdr_types:
        try:
            res = evaluate_cdr_type([cdr_type], rep_names, args.n_folds)
            if res is not None:
                all_results[cdr_type] = res
        except Exception as e:
            print(f"  FAILED for {cdr_type}: {e}")
            import traceback
            traceback.print_exc()

    # Also evaluate combined
    if len(cdr_types) > 1:
        try:
            res = evaluate_cdr_type(cdr_types, rep_names, args.n_folds)
            if res is not None:
                all_results['combined'] = res
        except Exception as e:
            print(f"  FAILED for combined: {e}")

    # Analysis
    if all_results:
        print_analysis(all_results)

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_cdr_flexibility.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
