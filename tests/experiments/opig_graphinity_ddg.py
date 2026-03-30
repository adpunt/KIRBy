#!/usr/bin/env python3
"""
Graphinity DDG Prediction Experiment (OPIG B2)

Replicates Hummer et al. 2025 "Investigating the Volume and Diversity of
Data Needed for Generalizable Antibody-Antigen DDG Prediction" —
tests whether sequence-based reps + simple ML can approach the
structure-based Graphinity EGNN for DDG prediction.

Scientific questions:
  1. Can sequence-based reps predict synthetic DDG (within-distribution)?
  2. How does transfer from synthetic → experimental compare?
  3. Does hybrid augmentation help for DDG prediction?
  4. How far are simple rep→model approaches from the structure-based EGNN?

Conditions:
  - within_foldx:   Train/eval on FoldX synthetic DDG (GroupKFold by complex)
  - within_rosetta:  Train/eval on Rosetta Flex ddG (GroupKFold by complex)
  - transfer_foldx:  Train on ALL FoldX synthetic, eval on ALL experimental SKEMPI
  - experimental:    Train/eval on experimental SKEMPI only (baseline)

Paper baselines (Hummer et al. 2025, Nature Computational Science):
  Graphinity EGNN (structure-based):  Pearson r ~0.87 on experimental test
  Key finding: orders of magnitude more experimental data needed for
  generalizable DDG prediction.

Usage:
    # Smoke test (~3 min, rosetta only which is smaller)
    python tests/experiments/opig_graphinity_ddg.py \\
        --conditions within_rosetta --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation
    python tests/experiments/opig_graphinity_ddg.py

Reference:
    Hummer et al. (2025). Nature Computational Science.
    Zenodo: https://doi.org/10.5281/zenodo.15384945
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from run_baseline_survey import evaluate_rep as _evaluate_rep
from experiments._shared import convert_numpy, get_model_factories

# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_CONDITIONS = ['within_foldx', 'within_rosetta', 'experimental', 'transfer_foldx']

DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',           # classical
    'esm2', 'esm2_concat', 'antiberty',  # LMs
    'ablang2', 'prot_t5',                 # additional LMs
]

MODELS_LIST = ['ridge', 'rf']

# Condition → dataset name mapping
CONDITION_DATASETS = {
    'within_foldx':   'graphinity_foldx',
    'within_rosetta': 'graphinity_rosetta',
    'experimental':   'ddg_combined',
    'transfer_foldx': None,  # handled specially: train on synthetic, eval on experimental
}

# Paper baselines (Hummer et al. 2025, Nature Computational Science)
# IMPORTANT: The paper's core finding is that random splits inflate performance.
# Stringent CDR identity cutoffs reveal much lower generalization.
# Numbers below are from the repo's results notebooks (Figure-2b, Figure-2c).
PAPER_BASELINES = {
    'Graphinity EGNN synthetic FoldX (random)':     {'pearson': 0.896},
    'Graphinity EGNN synthetic FoldX (cutoff_100)':  {'pearson': 0.894},
    'Graphinity EGNN synthetic FoldX (cutoff_90)':   {'pearson': 0.893},
    'Graphinity EGNN synthetic FoldX (cutoff_70)':   {'pearson': 0.894},
    # Experimental: -RevMuts -NonBinders (closest to ddg_combined)
    'Graphinity EGNN experimental (random)':         {'pearson': 0.525},
    'Graphinity EGNN experimental (cutoff_100)':     {'pearson': 0.271},
    'Graphinity EGNN experimental (cutoff_90)':      {'pearson': 0.291},
    'Graphinity EGNN experimental (cutoff_70)':      {'pearson': 0.216},
    # Experimental: +NonBinders variant (main Figure-2c plot)
    'Graphinity EGNN exp+nonbinders (random)':       {'pearson': 0.382},
    'Graphinity EGNN exp+nonbinders (cutoff_70)':    {'pearson': 0.040},
    # Reference: FoldX computed vs experimental correlation
    'FoldX vs experimental':                         {'pearson': 0.46},
}
# Note: KIRBy groups by PDB complex (≈cutoff_100). Synthetic FoldX is stable
# across cutoffs (~0.89), but experimental collapses from 0.53 → 0.22 at cutoff_70.
# The "0.87" number previously cited is from the +RevMuts variant with random splits.


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep_ddg(X, labels, groups, model, n_folds):
    """Evaluate one (rep, model) for DDG regression via GroupKFold.

    Delegates to run_baseline_survey.evaluate_rep for per-fold metrics.
    Also computes global-concatenation Pearson r (paper's method: concatenate
    all held-out predictions, compute single Pearson r on full array).

    Returns dict with pearson, spearman, r2, rmse as {metric: {mean, std}},
    plus pearson_global (single value, comparable to paper's reported numbers).
    """
    from scipy.stats import pearsonr
    from sklearn.base import clone
    from run_baseline_survey import _make_cv, _maybe_scale

    mean_m, std_m = _evaluate_rep(X, labels, groups, model, n_folds, 'regression')
    if not mean_m:
        return None
    result = {k: {'mean': mean_m[k], 'std': std_m.get(k, 0.0)} for k in mean_m}

    # Global-concatenation Pearson r (matches Graphinity's epoch_metrics)
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    cv, cv_kwargs = _make_cv(groups, n_folds, labels)
    all_preds = np.full_like(labels, np.nan)
    for train_idx, val_idx in cv.split(X, labels, **cv_kwargs):
        X_tr, X_va = _maybe_scale(X[train_idx], X[val_idx])
        try:
            m = clone(model)
            m.fit(X_tr, labels[train_idx])
            all_preds[val_idx] = m.predict(X_va)
        except Exception:
            pass
    valid = ~np.isnan(all_preds)
    if valid.sum() > 2:
        r_global, _ = pearsonr(labels[valid], all_preds[valid])
        result['pearson_global'] = float(r_global)
    else:
        result['pearson_global'] = float('nan')

    return result


def evaluate_transfer(rep_names, n_folds):
    """Transfer learning: train on FoldX synthetic, evaluate on experimental.

    No cross-validation — train on ALL synthetic data, test on ALL experimental.
    Groups experimental by PDB complex for reporting variance.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.base import clone
    from run_baseline_survey import _maybe_scale

    print(f"\n{'=' * 60}")
    print(f"Condition: transfer_foldx (train=synthetic, test=experimental)")
    print(f"{'=' * 60}")

    train_data = load_dataset('graphinity_foldx')
    test_data = load_dataset('ddg_combined')

    n_train = len(train_data['labels'])
    n_test = len(test_data['labels'])
    print(f"  Train: {n_train} synthetic FoldX samples")
    print(f"  Test: {n_test} experimental SKEMPI samples")

    # Generate representations for both sets
    from kirby.representations.antibody import ANTIBODY_REPS
    from kirby.representations.mutation import MUTATION_REPS

    reps_train, reps_test = {}, {}
    for rep_name in rep_names:
        rep_fn = ANTIBODY_REPS.get(rep_name) or MUTATION_REPS.get(rep_name)
        if rep_fn is None:
            continue
        try:
            if rep_name in MUTATION_REPS:
                X_tr = rep_fn(train_data['heavy_seqs'],
                              train_data['light_seqs'],
                              train_data.get('mutations', []))
                X_te = rep_fn(test_data['heavy_seqs'],
                              test_data['light_seqs'],
                              test_data.get('mutations', []))
            else:
                X_tr = ANTIBODY_REPS[rep_name](train_data['heavy_seqs'],
                                               train_data['light_seqs'])
                X_te = ANTIBODY_REPS[rep_name](test_data['heavy_seqs'],
                                               test_data['light_seqs'])
            X_tr = np.asarray(X_tr, dtype=np.float64)
            X_te = np.asarray(X_te, dtype=np.float64)
            if X_tr.ndim == 2 and X_tr.shape[0] == n_train and X_te.shape[0] == n_test:
                reps_train[rep_name] = X_tr
                reps_test[rep_name] = X_te
                print(f"  {rep_name}: {X_tr.shape[1]}D")
        except Exception as e:
            print(f"  {rep_name}: FAILED ({e})")

    if not reps_train:
        print("  No reps available, skipping condition")
        return None

    y_train = train_data['labels']
    y_test = test_data['labels']
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n_train)

    print(f"\n  Evaluating {len(reps_train)} reps × {len(models)} models "
          f"(train-on-synthetic, test-on-experimental)...")
    results = {}
    for rep_name in sorted(reps_train.keys()):
        X_tr = reps_train[rep_name]
        X_te = reps_test[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            try:
                X_tr_s, X_te_s = _maybe_scale(X_tr, X_te)
                m = factory()
                m.fit(X_tr_s, y_train)
                preds = m.predict(X_te_s)
                r_p, _ = pearsonr(y_test, preds)
                r_s, _ = spearmanr(y_test, preds)
                rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
                rep_results[model_name] = {
                    'pearson': {'mean': float(r_p), 'std': 0.0},
                    'spearman': {'mean': float(r_s), 'std': 0.0},
                    'rmse': {'mean': rmse, 'std': 0.0},
                    'pearson_global': float(r_p),
                }
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"r={r_p:.4f}  rho={r_s:.4f}  rmse={rmse:.4f}")
            except Exception as e:
                print(f"    {rep_name:25s} {model_name:8s}  FAILED ({e})")
        if rep_results:
            results[rep_name] = {'dim': X_tr.shape[1], 'models': rep_results}

    return {
        'n_samples': n_test,
        'n_train': n_train,
        'condition': 'transfer_foldx',
        'dataset': 'graphinity_foldx -> ddg_combined',
        'results': results,
    }


def evaluate_condition(condition, rep_names, n_folds):
    """Evaluate one condition: load data, generate reps, evaluate all.

    Returns dict with per-rep results.
    """
    if condition == 'transfer_foldx':
        return evaluate_transfer(rep_names, n_folds)

    ds_name = CONDITION_DATASETS[condition]

    print(f"\n{'=' * 60}")
    print(f"Condition: {condition} (dataset: {ds_name})")
    print(f"{'=' * 60}")

    data = load_dataset(ds_name)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    groups = data.get('groups')
    n = len(labels)

    print(f"  {n} samples, DDG range [{labels.min():.2f}, {labels.max():.2f}]")
    if groups is not None:
        print(f"  {len(np.unique(groups))} groups (PDB complexes)")

    # Generate representations
    from kirby.representations.antibody import ANTIBODY_REPS
    from kirby.representations.mutation import MUTATION_REPS
    reps = {}
    for rep_name in rep_names:
        rep_fn = ANTIBODY_REPS.get(rep_name) or MUTATION_REPS.get(rep_name)
        if rep_fn is None:
            continue
        try:
            if rep_name in MUTATION_REPS:
                mutations = data.get('mutations', [])
                X = rep_fn(heavy_seqs, light_seqs, mutations)
            else:
                X = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 2 and X.shape[0] == n:
                reps[rep_name] = X
                print(f"  {rep_name}: {X.shape[1]}D")
        except Exception as e:
            print(f"  {rep_name}: FAILED ({e})")

    if not reps:
        print("  No reps available, skipping condition")
        return None

    # Build hybrid with aac as augmentation rep (matches DDG domain config)
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        from kirby.hybrid import apply_feature_selection
        grp = groups if groups is not None else np.arange(n)
        AUG_REPS = {'aac'}
        base = {k: v for k, v in reps.items() if k not in AUG_REPS}
        aug_dict = {k: v for k, v in reps.items() if k in AUG_REPS}
        search = find_best_hybrid(
            base_reps=base if base else reps,
            labels=labels, groups=grp,
            metric='r2',
            task_type='regression',
            augmentations=aug_dict if aug_dict else None,
            augmentation_strategy='uncapped',
            speed='fast', return_fitted=True, verbose=False, cache=True)
        if search.fitted_result is not None:
            all_reps = {**reps, **(aug_dict or {})}
            fi = {r: {'selected_indices': idx}
                  for r, idx in search.fitted_result.selected_indices.items()}
            reps['kirby_hybrid'] = apply_feature_selection(all_reps, fi)
            print(f"  kirby_hybrid: {reps['kirby_hybrid'].shape[1]}D (hybrid)")
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    # Evaluate
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n)
    print(f"\n  Evaluating {len(reps)} reps × {len(models)} models × {n_folds}-fold CV...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            res = evaluate_rep_ddg(X, labels, groups, factory(), n_folds)
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                r_p = res['pearson']['mean']
                r_g = res.get('pearson_global', float('nan'))
                rho = res['spearman']['mean']
                tag = ' ***' if rep_name == 'kirby_hybrid' else ''
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"r={r_p:.4f}  r_g={r_g:.4f}  rho={rho:.4f}  "
                      f"({elapsed:.1f}s){tag}")
        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'n_samples': n,
        'condition': condition,
        'dataset': ds_name,
        'results': results,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print cross-condition comparison table."""
    print(f"\n{'=' * 100}")
    print(f"GRAPHINITY DDG COMPARISON (OPIG B2)")
    print(f"{'=' * 100}")

    # Paper baselines
    print(f"\n  Paper baselines:")
    for method, metrics in PAPER_BASELINES.items():
        r_p = metrics.get('pearson', float('nan'))
        print(f"    {method}: Pearson r = {r_p:.2f}")

    # Per-condition best results
    print(f"\n  {'Condition':20s} {'N':>7s} {'Best rep':>25s} {'Model':>8s} "
          f"{'Pearson r':>10s} {'Spearman':>10s} {'Hybrid r':>10s} {'Delta':>8s}")
    print(f"  {'-'*20} {'-'*7} {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for cond in ALL_CONDITIONS:
        if cond not in all_results:
            continue

        cr = all_results[cond]
        n = cr['n_samples']
        results = cr['results']

        best_indiv_name, best_model_name = None, None
        best_pearson = float('-inf')
        hybrid_pearson = float('nan')

        for rep_name, rep_data in results.items():
            for model_name, metrics in rep_data['models'].items():
                r_p = metrics['pearson']['mean']
                if rep_name == 'kirby_hybrid':
                    if np.isnan(hybrid_pearson) or r_p > hybrid_pearson:
                        hybrid_pearson = r_p
                else:
                    if r_p > best_pearson:
                        best_pearson = r_p
                        best_indiv_name = rep_name
                        best_model_name = model_name

        # Also get spearman for best individual
        best_spearman = float('nan')
        if best_indiv_name and best_model_name:
            best_spearman = results[best_indiv_name]['models'][best_model_name]['spearman']['mean']

        delta = hybrid_pearson - best_pearson if not np.isnan(hybrid_pearson) else float('nan')
        h_s = f"{hybrid_pearson:.4f}" if not np.isnan(hybrid_pearson) else "—"
        d_s = f"{delta:+.4f}" if not np.isnan(delta) else "—"
        rep_label = best_indiv_name or "—"

        print(f"  {cond:20s} {n:7d} {rep_label:>25s} {best_model_name or '—':>8s} "
              f"{best_pearson:10.4f} {best_spearman:10.4f} {h_s:>10s} {d_s:>8s}")

    # Gap analysis — compare to EGNN at cutoff_100 (closest to KIRBy's GroupKFold by PDB)
    egnn_r = PAPER_BASELINES['Graphinity EGNN experimental (cutoff_100)']['pearson']
    print(f"\n  Structure-based EGNN reference (cutoff_100): Pearson r = {egnn_r:.3f}")
    for cond in ALL_CONDITIONS:
        if cond not in all_results:
            continue
        results = all_results[cond]['results']
        best_r = max(
            (m['pearson']['mean']
             for rd in results.values() for m in rd['models'].values()),
            default=0
        )
        gap = egnn_r - best_r
        print(f"    {cond}: best Pearson r = {best_r:.4f}, "
              f"gap to EGNN = {gap:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Graphinity DDG experiment (OPIG B2). '
                    'Tests sequence-based reps for DDG prediction.')
    parser.add_argument(
        '--conditions', type=str, default=None,
        help='Comma-separated conditions '
             f'(default: {",".join(ALL_CONDITIONS)})')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 8)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of GroupKFold folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_graphinity_ddg.json)')

    args = parser.parse_args()

    conditions = (args.conditions.split(',') if args.conditions
                  else ALL_CONDITIONS)
    rep_names = (args.reps.split(',') if args.reps else DEFAULT_REPS)

    print("=" * 70)
    print("GRAPHINITY DDG EXPERIMENT (OPIG B2)")
    print(f"  Paper: Hummer et al. 2025, Nature Computational Science")
    print(f"  Conditions: {conditions}")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}
    for cond in conditions:
        if cond not in CONDITION_DATASETS:
            print(f"WARNING: Unknown condition '{cond}', skipping")
            continue
        ds_name = CONDITION_DATASETS[cond]
        # transfer_foldx has ds_name=None (handled specially in evaluate_condition)
        if ds_name is not None and ds_name not in DATASET_REGISTRY:
            print(f"WARNING: Dataset '{ds_name}' not in registry, skipping")
            continue
        try:
            cond_results = evaluate_condition(cond, rep_names, args.n_folds)
            if cond_results is not None:
                all_results[cond] = cond_results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Cross-condition analysis
    if all_results:
        print_analysis(all_results)

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_graphinity_ddg.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
