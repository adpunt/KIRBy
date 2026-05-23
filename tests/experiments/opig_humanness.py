#!/usr/bin/env python3
"""
Humanness Classification Experiment (OPIG B3)

Extends Prihoda et al. 2022 "BioPhi" (mAbs) — a score discrimination study
comparing 8 humanness scoring methods on IMGT/mAb-DB therapeutic antibodies.
BioPhi does NOT train classifiers; it computes ROC AUC via threshold sweep
on the full dataset (no train/test split, no CV). KIRBy instead trains
supervised classifiers (Ridge, RF) on high-dimensional reps with proper CV.

NOTE: Previous versions cited Marks et al. 2021 (Hu-mAb) as primary reference.
Marks 2021 is one of the 8 methods BioPhi evaluates, not the data source.
Data source = BioPhi's IMGT/mAb-DB curation (Prihoda et al. 2022).

Scientific questions:
  1. Which antibody rep is best for humanness classification?
  2. Does hybrid augmentation improve over the best individual rep?
  3. How do sequence-only LM reps compare to classical encodings (BLOSUM, onehot)?

Datasets:
  - species:     Human VH vs mouse+rhesus+camelid (OAS, 10K balanced)
  - therapeutic: Human vs non-human real therapeutics (BioPhi IMGT, ~553)

Paper baselines (Prihoda et al. 2022, BioPhi):
  - 8 methods scored on IMGT/mAb-DB therapeutics (VH+VL averaged):
    Hu-mAb:    AUROC=97.7%  Accuracy=93.3%  (best overall)
    OASis:     AUROC=97.2%  Accuracy=94.4%  (relaxed threshold, their method)
    Germline:  AUROC=96.8%  Accuracy=92.8%
  - No train/test split — full-dataset threshold sweep, NOT comparable to CV.
  - KIRBy uses VH-only; BioPhi averages VH+VL scores.

Usage:
    # Smoke test (~3 min)
    python tests/experiments/opig_humanness.py --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation
    python tests/experiments/opig_humanness.py --datasets species,therapeutic

Reference:
    Prihoda et al. (2022). BioPhi: A platform for antibody design,
    humanization and humanness evaluation. mAbs 14(1):2020203.
    Marks et al. (2021). J Mol Biol 433(11):166928. (Hu-mAb, one of 8 methods)
    Ramon et al. (2024). Bioinformatics 40(6):btae332. (AbNatiV, species data)
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
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
)

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

DEFAULT_DATASETS = ['humanness_species', 'humanness_therapeutic']

# Hand-picked reps: classical + LMs + LLR + CDR variants
DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',           # classical
    'ablang2', 'esm2', 'antiberty',      # LMs
    'igt5', 'prot_t5',                    # additional LMs
    'esmc_llr', 'antiberty_llr',          # log-likelihood ratios
    'abroberta',                           # Ab-RoBERTa
    'paratope',                            # structural (sequence-based)
]

MODELS_LIST = DOWNSTREAM_MODELS

# Published reference baselines
# Prihoda et al. 2022 (BioPhi): threshold-sweep on full therapeutic dataset,
# NO train/test split. Numbers from notebook 03_oasis_humanness.ipynb Cell 49.
# Note: BioPhi uses VH+VL averaged. KIRBy uses VH-only.
PAPER_BASELINES = {
    # No 'species' entry — Marks 2021 reports near-perfect AUROC on OAS
    # species-level but doesn't publish numbers, and BioPhi doesn't either.
    # Comparison block is skipped when this key is missing.
    'therapeutic': {
        # Human vs non-human therapeutics (BioPhi Cell 49, VH+VL averaged)
        'Hu-mAb (Marks)':      {'auroc': 0.977, 'accuracy': 0.933},
        'OASis relaxed':       {'auroc': 0.972, 'accuracy': 0.944},
        'Germline content':    {'auroc': 0.968, 'accuracy': 0.928},
        'IgReconstruct':       {'auroc': 0.967, 'accuracy': 0.926},
        'MG Score':            {'auroc': 0.959, 'accuracy': 0.918},
        'AbLSTM':              {'auroc': 0.937, 'accuracy': 0.878},
        'T20':                 {'auroc': 0.896, 'accuracy': 0.839},
        'Z-score':             {'auroc': 0.837, 'accuracy': 0.765},
    },
}


# =============================================================================
# EVALUATION
# =============================================================================

def _is_binary_rep(X):
    """Check if feature matrix is binary (0/1)."""
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


def evaluate_rep_clf(X, labels, groups, model, n_folds, rep_name=''):
    """Evaluate one (rep, model) pair for classification via stratified CV.

    Returns dict with accuracy, f1_weighted, auroc, balanced_accuracy (mean/std).
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
    acc_scores, f1_scores, auroc_scores, bacc_scores = [], [], [], []

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
        f1_scores.append(f1_score(y_val, preds, average='weighted'))
        bacc_scores.append(balanced_accuracy_score(y_val, preds))

        # AUROC (binary only, need decision function or probabilities)
        try:
            if hasattr(m, 'decision_function'):
                scores = m.decision_function(X_val)
            elif hasattr(m, 'predict_proba'):
                scores = m.predict_proba(X_val)[:, 1]
            else:
                scores = preds.astype(float)
            auroc_scores.append(roc_auc_score(y_val, scores))
        except Exception:
            pass

    if not acc_scores:
        return None

    result = {
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
        'f1_weighted': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
        'balanced_accuracy': {'mean': np.mean(bacc_scores), 'std': np.std(bacc_scores)},
    }
    if auroc_scores:
        result['auroc'] = {'mean': np.mean(auroc_scores), 'std': np.std(auroc_scores)}
    return result


def evaluate_dataset(dataset_name, rep_names, n_folds, pruning_mode='default_pruning'):
    """Evaluate all reps × models on one humanness dataset.

    Returns dict of {rep_name: {model_name: {metric: {mean, std}}}}.
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    data = load_dataset(dataset_name)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    n = len(labels)
    n_pos = int((labels == 1).sum())
    print(f"  {n} samples, {n_pos} human ({100*n_pos/n:.1f}%), "
          f"{n - n_pos} non-human ({100*(n-n_pos)/n:.1f}%)")

    # Generate identity groups (unique sequence → group ID)
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
        dataset_tag=dataset_name,
        mutations=data.get('mutations'),
    )

    if not reps:
        print("  No reps available, skipping dataset")
        return None

    # Build hybrid and add as rep
    # Augmentation config matches humanness_therapeutic in DATASET_REPS:
    # augmentation = [] (no augmentation reps for antibody classification)
    try:
        from kirby.hybrid import apply_feature_selection
        from experiments._shared import (resolve_pruning_kwargs,
                                          get_best_search_for_downstream)
        _pkw = resolve_pruning_kwargs(pruning_mode, len(labels),
                                       dataset_label=dataset_name)
        search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=None,
            # Explicit 'none' avoids cache-key collision with real
            # uncapped-augmentation runs (project_cache_collision_fix).
            augmentation_strategy='none',
            speed='full',
            metric='auc',
            task_type='classification',
            dataset_name=dataset_name,
            return_fitted=True,
            adaptive_pruning=0.25,
            **_pkw,
        )
        if search is not None:
            top3 = extract_top3_base_reps(search)
            if top3:
                print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        fitted = search.fitted_results.get('rf') if search is not None else None
        if fitted is not None:
            fi = {r: {'selected_indices': idx}
                  for r, idx in fitted.selected_indices.items()}
            reps['kirby_hybrid'] = apply_feature_selection(reps, fi)
            print(f"  kirby_hybrid: {reps['kirby_hybrid'].shape[1]}D (hybrid)")
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    # Evaluate all reps × models
    models = get_model_factories(MODELS_LIST, 'classification', n_samples=n)
    print(f"\n  Evaluating {len(reps)} reps × {len(models)} models × {n_folds}-fold CV...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                res = evaluate_rep_clf(X, labels, groups, factory(), n_folds, rep_name)
            except ValueError as e:
                print(f"    {rep_name:25s} {model_name:8s}  SKIPPED: {e}")
                continue
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                acc = res['accuracy']['mean']
                auroc = res.get('auroc', {}).get('mean', float('nan'))
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"acc={acc:.4f}  auroc={auroc:.4f}  ({elapsed:.1f}s)")
        if rep_results:
            results[rep_name] = {
                'dim': X.shape[1],
                'models': rep_results,
            }

    return {
        'n_samples': n,
        'n_human': n_pos,
        'n_nonhuman': n - n_pos,
        'results': results,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(dataset_name, ds_results):
    """Print comparison table with paper baselines."""
    if ds_results is None:
        return

    results = ds_results['results']
    baselines = PAPER_BASELINES.get(dataset_name.replace('humanness_', ''), {})

    print(f"\n{'=' * 90}")
    print(f"HUMANNESS EXPERIMENT RESULTS: {dataset_name}")
    print(f"  {ds_results['n_samples']} samples "
          f"({ds_results['n_human']} human, {ds_results['n_nonhuman']} non-human)")
    print(f"{'=' * 90}")

    # Header
    print(f"\n  {'Rep':25s} {'Model':8s} {'Acc':>10s} {'AUROC':>10s} {'F1w':>10s} {'BalAcc':>10s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Paper baselines first
    for method, metrics in baselines.items():
        acc = metrics.get('accuracy', float('nan'))
        auroc = metrics.get('auroc', float('nan'))
        acc_s = f"{acc:.4f}" if not np.isnan(acc) else "—"
        auroc_s = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        print(f"  {method:25s} {'paper':8s} {acc_s:>10s} {auroc_s:>10s} {'—':>10s} {'—':>10s}")

    if baselines:
        print(f"  {'. . .':25s}")

    # KIRBy results: find best model per rep, sort by AUROC then accuracy
    rows = []
    for rep_name, rep_data in results.items():
        for model_name, metrics in rep_data['models'].items():
            acc = metrics['accuracy']['mean']
            auroc = metrics.get('auroc', {}).get('mean', float('nan'))
            f1w = metrics['f1_weighted']['mean']
            bacc = metrics['balanced_accuracy']['mean']
            rows.append((rep_name, model_name, acc, auroc, f1w, bacc))

    # Sort by AUROC descending (fallback to accuracy)
    rows.sort(key=lambda r: (r[3] if not np.isnan(r[3]) else 0, r[2]), reverse=True)

    for rep_name, model_name, acc, auroc, f1w, bacc in rows:
        auroc_s = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        is_hybrid = '***' if rep_name == 'kirby_hybrid' else ''
        print(f"  {rep_name:25s} {model_name:8s} {acc:10.4f} {auroc_s:>10s} "
              f"{f1w:10.4f} {bacc:10.4f} {is_hybrid}")

    # Summary: best individual rep vs hybrid
    best_individual = None
    best_hybrid = None
    for rep_name, model_name, acc, auroc, f1w, bacc in rows:
        metric_val = auroc if not np.isnan(auroc) else acc
        if rep_name == 'kirby_hybrid':
            if best_hybrid is None or metric_val > best_hybrid[2]:
                best_hybrid = (rep_name, model_name, metric_val)
        else:
            if best_individual is None or metric_val > best_individual[2]:
                best_individual = (rep_name, model_name, metric_val)

    print(f"\n  Summary:")
    if best_individual:
        print(f"    Best individual rep: {best_individual[0]} ({best_individual[1]}) "
              f"= {best_individual[2]:.4f}")
    if best_hybrid:
        print(f"    Hybrid:             {best_hybrid[0]} ({best_hybrid[1]}) "
              f"= {best_hybrid[2]:.4f}")
        if best_individual:
            delta = best_hybrid[2] - best_individual[2]
            print(f"    Delta:              {delta:+.4f} "
                  f"({'hybrid wins' if delta > 0 else 'individual wins'})")

    for method, metrics in baselines.items():
        ref_val = metrics.get('auroc', metrics.get('accuracy', float('nan')))
        if not np.isnan(ref_val) and best_individual:
            delta = best_individual[2] - ref_val
            print(f"    vs {method}: {delta:+.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Humanness classification experiment (OPIG B3). '
                    'Extends Prihoda et al. 2022 (BioPhi) with KIRBy antibody reps.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated dataset names '
             '(default: humanness_species,humanness_therapeutic)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 12)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of stratified CV folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_humanness.json)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default: 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    args = parser.parse_args()

    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)
    rep_names = (args.reps.split(',') if args.reps
                 else DEFAULT_REPS)

    # Validate datasets
    for ds in datasets:
        if ds not in DATASET_REGISTRY:
            print(f"ERROR: Unknown dataset '{ds}'")
            return 1

    print("=" * 70)
    print("HUMANNESS CLASSIFICATION EXPERIMENT (OPIG B3)")
    print(f"  Paper: Prihoda et al. 2022, mAbs 14(1):2020203 (BioPhi)")
    print(f"  Datasets: {datasets}")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}
    for ds_name in datasets:
        try:
            ds_results = evaluate_dataset(ds_name, rep_names, args.n_folds,
                                           pruning_mode=args.pruning_mode)
            if ds_results is not None:
                all_results[ds_name] = ds_results
                print_analysis(ds_name, ds_results)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_humanness.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
