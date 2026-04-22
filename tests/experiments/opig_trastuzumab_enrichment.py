#!/usr/bin/env python3
"""
Trastuzumab-HER2 Binder Enrichment Experiment (OPIG B1)

Extends Chinery et al. 2024 "Baselining the Buzz: Trastuzumab-HER2
Affinity, and Beyond" — the paper evaluates zero-shot LIBRARY DESIGN
methods (BLOSUM, AbLang, ESM, ProteinMPNN as GENERATORS). KIRBy instead
evaluates REPRESENTATION QUALITY for supervised classification. These are
different questions: the paper asks "which generator designs more binders?"
while KIRBy asks "which embedding best separates binders from non-binders?"

Two evaluation modes:
  1. Zero-shot ranking: LLR reps used directly as scores (closer to paper)
  2. Supervised classification: train classifier on reps, rank by score
     (KIRBy's extension — paper did not train classifiers)

Paper baselines (Chinery et al. 2024, Table 2):
  These are % of a designed library predicted as binders by a CNN, NOT
  top-K enrichment on a held-out test set. Numbers are not directly
  comparable to KIRBy's enrichment, but reported for context:
  Random baseline:   ~13% predicted binder rate
  ProteinMPNN:       ~19%
  BLOSUM62:          ~26%
  AbLang:            ~27-29%
  ESM:               ~30%
  DMS (oracle):      ~33%
  Experimental SPR (v2, 720 designs): BLOSUM outperformed all 4 DL methods.

Usage:
    # Smoke test (~5 min)
    python tests/experiments/opig_trastuzumab_enrichment.py \\
        --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation
    python tests/experiments/opig_trastuzumab_enrichment.py

Reference:
    Chinery et al. (2024). bioRxiv/PNAS.
    https://github.com/lewis-chinery/Tz_her2_affinity_and_beyond
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',           # classical (paper used BLOSUM)
    'ablang2', 'esm2', 'antiberty',      # LMs (paper used AbLang, ESM)
    'igt5', 'prot_t5',                    # additional LMs
    'esmc_llr', 'antiberty_llr',          # LLR (paper used ProteinMPNN LLR)
    'abroberta',                           # Ab-RoBERTa
]

ENRICHMENT_FRACTIONS = [0.01, 0.05, 0.10]  # top-1%, top-5%, top-10%

MODELS_LIST = DOWNSTREAM_MODELS

# Paper baselines: CNN-predicted binder enrichment (Chinery et al. 2024)
# % sequences with CNN-HER2-max binding probability >90%
# Note: these are predicted enrichment rates, not raw top-K enrichment.
# KIRBy measures raw top-K enrichment (different metric but comparable in spirit).
PAPER_BASELINES = {
    'Random (paper)':          {'predicted_binder_pct': 0.13},
    'ProteinMPNN (paper)':     {'predicted_binder_pct': 0.19},
    'BLOSUM62 (paper)':        {'predicted_binder_pct': 0.26},
    'AbLang mask-all (paper)': {'predicted_binder_pct': 0.27},
    'AbLang mask-one (paper)': {'predicted_binder_pct': 0.29},
    'ESM mask-one (paper)':    {'predicted_binder_pct': 0.30},
    'DMS oracle (paper)':      {'predicted_binder_pct': 0.33},
}
# Experimental SPR validation (v2, 720 designs):
# BLOSUM outperformed all 4 DL methods, 9 sub-nanomolar binders found


# =============================================================================
# ENRICHMENT METRIC
# =============================================================================

def compute_enrichment(y_true, y_score, fractions=None):
    """Compute binder enrichment at various top-N fractions.

    Enrichment = (% binders in top-N) / (% binders overall).
    Raw enrichment = % binders in top-N (what the paper reports).

    Args:
        y_true: binary labels (1=binder, 0=non-binder)
        y_score: predicted scores (higher = more likely binder)
        fractions: list of fractions to evaluate (e.g., [0.01, 0.05, 0.10])

    Returns:
        dict: {f'enrich_{frac}': raw_enrichment, f'ef_{frac}': enrichment_factor}
    """
    if fractions is None:
        fractions = ENRICHMENT_FRACTIONS

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    base_rate = y_true.mean()

    # Rank by score descending
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    result = {'base_rate': float(base_rate)}
    for frac in fractions:
        k = max(1, int(n * frac))
        top_k = y_sorted[:k]
        raw_enrichment = top_k.mean()
        enrichment_factor = raw_enrichment / base_rate if base_rate > 0 else 0.0
        result[f'enrich_{frac}'] = float(raw_enrichment)
        result[f'ef_{frac}'] = float(enrichment_factor)

    return result


def _is_binary_rep(X):
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep_enrichment(X, labels, groups, model, n_folds):
    """Evaluate one (rep, model) for enrichment via stratified CV.

    For each fold: train → predict scores on hold-out → compute enrichment.
    Returns dict with enrichment metrics (mean/std across folds) + AUROC.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if groups is None or len(np.unique(groups)) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds}-fold group split: "
            f"groups={'None' if groups is None else len(np.unique(groups))} unique")
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(cv.split(X, labels, groups=groups))

    is_binary = _is_binary_rep(X)
    fold_enrichments = []
    auroc_scores = []
    acc_scores = []

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

        # Get prediction scores for ranking
        if hasattr(m, 'decision_function'):
            scores = m.decision_function(X_val)
        elif hasattr(m, 'predict_proba'):
            scores = m.predict_proba(X_val)[:, 1]
        else:
            scores = m.predict(X_val).astype(float)

        preds = m.predict(X_val)
        acc_scores.append(accuracy_score(y_val, preds))

        try:
            auroc_scores.append(roc_auc_score(y_val, scores))
        except Exception:
            pass

        enrichment = compute_enrichment(y_val, scores)
        fold_enrichments.append(enrichment)

    if not fold_enrichments:
        return None

    # Aggregate across folds
    result = {
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
    }
    if auroc_scores:
        result['auroc'] = {'mean': np.mean(auroc_scores), 'std': np.std(auroc_scores)}

    # Enrichment metrics
    for frac in ENRICHMENT_FRACTIONS:
        key = f'enrich_{frac}'
        vals = [fe[key] for fe in fold_enrichments if key in fe]
        if vals:
            result[key] = {'mean': np.mean(vals), 'std': np.std(vals)}

        ef_key = f'ef_{frac}'
        ef_vals = [fe[ef_key] for fe in fold_enrichments if ef_key in fe]
        if ef_vals:
            result[ef_key] = {'mean': np.mean(ef_vals), 'std': np.std(ef_vals)}

    result['base_rate'] = np.mean([fe['base_rate'] for fe in fold_enrichments])
    return result


def evaluate_dataset(rep_names, n_folds):
    """Evaluate all reps × models on tz_her2_binary for enrichment."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: tz_her2_binary (Trastuzumab-HER2 binding)")
    print(f"{'=' * 60}")

    data = load_dataset('tz_her2_binary')
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    n = len(labels)
    n_binders = int((labels == 1).sum())
    print(f"  {n} samples, {n_binders} binders ({100*n_binders/n:.1f}%), "
          f"{n - n_binders} non-binders ({100*(n-n_binders)/n:.1f}%)")

    # Groups by unique sequence identity
    unique_seqs = {}
    group_ids = np.zeros(n, dtype=int)
    for i, s in enumerate(heavy_seqs):
        if s not in unique_seqs:
            unique_seqs[s] = len(unique_seqs)
        group_ids[i] = unique_seqs[s]
    groups = group_ids

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag='tz_her2_binary',
    )

    if not reps:
        print("  No reps, aborting")
        return None

    # Build hybrid
    # Augmentation config matches tz_her2_binary in DATASET_REPS:
    # augmentation = [] (no augmentation reps for antibody classification)
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        from kirby.hybrid import apply_feature_selection
        search = find_best_hybrid(
            base_reps=reps,
            labels=labels, groups=groups,
            metric='auc',
            task_type='classification',
            budgets=FULL_BUDGETS,
            augmentation_strategy='none',
            internal_evaluators=INTERNAL_EVALUATORS,
            speed='full', return_fitted=True, verbose=False, cache=True,
            dataset_name='tz_her2_binary', tier='general',
            adaptive_pruning=0.25)
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        fitted = search.fitted_results.get('rf')
        if fitted is not None:
            fi = {r: {'selected_indices': idx}
                  for r, idx in fitted.selected_indices.items()}
            reps['kirby_hybrid'] = apply_feature_selection(reps, fi)
            print(f"  kirby_hybrid: {reps['kirby_hybrid'].shape[1]}D (hybrid)")
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    # Zero-shot evaluation: LLR reps used directly as ranking scores
    # (matches paper's protocol — no classifier training, just rank by score)
    LLR_REPS = {'esmc_llr', 'antiberty_llr'}
    print(f"\n  Zero-shot ranking (paper's protocol)...")
    zero_shot_results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        if rep_name in LLR_REPS and X.shape[1] <= 3:
            # LLR reps produce low-dim scores — use mean as ranking score
            scores = X.mean(axis=1)
            enrichment = compute_enrichment(labels, scores)
            try:
                auroc_val = roc_auc_score(labels, scores)
            except Exception:
                auroc_val = float('nan')
            zero_shot_results[rep_name] = {
                'enrichment': enrichment,
                'auroc': float(auroc_val),
            }
            e1 = enrichment.get('enrich_0.01', float('nan'))
            print(f"    {rep_name:25s} zero-shot  "
                  f"E@1%={e1:.4f}  AUROC={auroc_val:.4f}")

    # Supervised evaluation (KIRBy's extension beyond paper)
    models = get_model_factories(MODELS_LIST, 'classification', n_samples=n)
    print(f"\n  Supervised evaluation ({len(reps)} reps × {len(models)} models × {n_folds}-fold CV)...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                res = evaluate_rep_enrichment(X, labels, groups, factory(), n_folds)
            except ValueError as e:
                print(f"    {rep_name:25s} {model_name:8s}  SKIPPED: {e}")
                continue
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                e1 = res.get('enrich_0.01', {}).get('mean', float('nan'))
                auroc = res.get('auroc', {}).get('mean', float('nan'))
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"E@1%={e1:.4f}  AUROC={auroc:.4f}  ({elapsed:.1f}s)")
        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'n_samples': n,
        'n_binders': n_binders,
        'base_rate': n_binders / n,
        'zero_shot': zero_shot_results,  # paper's protocol (no training)
        'supervised': results,            # KIRBy's extension (trained classifiers)
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(ds_results):
    """Print enrichment comparison table with paper baselines."""
    if ds_results is None:
        return

    results = ds_results['supervised']
    base_rate = ds_results['base_rate']

    print(f"\n{'=' * 100}")
    print(f"TRASTUZUMAB ENRICHMENT COMPARISON (OPIG B1)")
    print(f"  {ds_results['n_samples']} samples, base binder rate = {base_rate:.3f}")
    print(f"{'=' * 100}")

    # Header
    header = f"  {'Method':30s} {'E@1%':>8s} {'E@5%':>8s} {'E@10%':>8s} {'AUROC':>8s} {'Acc':>8s}"
    print(header)
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    # Random baseline
    print(f"  {'Random baseline':30s} {base_rate:8.4f} {base_rate:8.4f} "
          f"{base_rate:8.4f} {'0.5000':>8s} {'—':>8s}")

    # Paper baselines (predicted binder %, different metric but reference point)
    print(f"  {'— Paper (pred. binder %) —':30s}")
    for method, metrics in PAPER_BASELINES.items():
        pct = metrics.get('predicted_binder_pct', float('nan'))
        pct_s = f"{pct:.2%}" if not np.isnan(pct) else "—"
        print(f"  {method:30s} {pct_s:>8s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s}")

    print(f"  {'. . .':30s}")

    # KIRBy results: best model per rep, sort by E@1%
    rows = []
    for rep_name, rep_data in results.items():
        for model_name, metrics in rep_data['models'].items():
            e1 = metrics.get('enrich_0.01', {}).get('mean', float('nan'))
            e5 = metrics.get('enrich_0.05', {}).get('mean', float('nan'))
            e10 = metrics.get('enrich_0.10', {}).get('mean', float('nan'))
            auroc = metrics.get('auroc', {}).get('mean', float('nan'))
            acc = metrics.get('accuracy', {}).get('mean', float('nan'))
            rows.append((rep_name, model_name, e1, e5, e10, auroc, acc))

    rows.sort(key=lambda r: (r[2] if not np.isnan(r[2]) else 0), reverse=True)

    for rep_name, model_name, e1, e5, e10, auroc, acc in rows:
        label = f"{rep_name} ({model_name})"
        tag = ' ***' if rep_name == 'kirby_hybrid' else ''
        e1s = f"{e1:.4f}" if not np.isnan(e1) else "—"
        e5s = f"{e5:.4f}" if not np.isnan(e5) else "—"
        e10s = f"{e10:.4f}" if not np.isnan(e10) else "—"
        aurocs = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        accs = f"{acc:.4f}" if not np.isnan(acc) else "—"
        print(f"  {label:30s} {e1s:>8s} {e5s:>8s} {e10s:>8s} {aurocs:>8s} {accs:>8s}{tag}")

    # Summary
    best_indiv = [r for r in rows if r[0] != 'kirby_hybrid']
    best_hybrid = [r for r in rows if r[0] == 'kirby_hybrid']

    print(f"\n  Summary (Enrichment @ top-1%):")
    if best_indiv:
        bi = best_indiv[0]
        print(f"    Best individual: {bi[0]} ({bi[1]}) E@1% = {bi[2]:.4f}")
    if best_hybrid:
        bh = best_hybrid[0]
        print(f"    Hybrid:          {bh[0]} ({bh[1]}) E@1% = {bh[2]:.4f}")
        if best_indiv:
            delta = bh[2] - bi[2]
            print(f"    Delta:           {delta:+.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Trastuzumab binder enrichment experiment (OPIG B1). '
                    'Replicates Chinery et al. 2024 enrichment analysis.')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 11)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of stratified CV folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_trastuzumab_enrichment.json)')

    args = parser.parse_args()
    rep_names = args.reps.split(',') if args.reps else DEFAULT_REPS

    print("=" * 70)
    print("TRASTUZUMAB BINDER ENRICHMENT EXPERIMENT (OPIG B1)")
    print(f"  Paper: Chinery et al. 2024")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Enrichment fractions: {ENRICHMENT_FRACTIONS}")
    print("=" * 70)

    ds_results = evaluate_dataset(rep_names, args.n_folds)
    if ds_results is not None:
        print_analysis(ds_results)

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_trastuzumab_enrichment.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(ds_results or {}), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
