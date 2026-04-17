#!/usr/bin/env python3
"""
Ginkgo Antibody Developability Experiment (OPIG B5)

Replicates van Niekerk et al. 2026 "2025 Ginkgo Datapoints Antibody
Developability Competition" — evaluates KIRBy reps across all 9 biophysical
properties using the competition's pre-computed hierarchical cluster splits.

Scientific questions:
  1. Which antibody reps best predict each developability property?
  2. Does hybrid augmentation help on these small (n=94-242) datasets?
  3. Across all 9 properties, how often does hybrid win?
  4. Which properties are predictable and which are noise?

Properties (GDPa1 v1.2, 242 antibodies):
  HIC (retention), Tm2 (thermal stability), Titer (expression),
  PR_CHO (protein recovery), AC-SINS (self-interaction),
  Purity, SEC %Monomer (aggregation), SMAC (colloidal), HAC (heparin)

Paper baselines (van Niekerk et al. 2026):
  Competition had 100+ team submissions. Best reported Spearman rho:
    HIC ~0.65-0.70, Tm2 ~0.30-0.45, Titer ~0.35-0.50,
    AC-SINS ~0.55-0.65, Heparin ~0.60-0.75 (varies by property)
  Pre-computed 5-fold hierarchical cluster splits (IgG-stratified).

Usage:
    # Smoke test (1 property, fast reps, ~2 min)
    python tests/experiments/opig_ginkgo_developability.py \\
        --properties ginkgo_hic --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation (all 9 properties)
    python tests/experiments/opig_ginkgo_developability.py

Reference:
    van Niekerk et al. (2026). mAbs.
    https://github.com/ginkgobioworks/abdev-benchmark
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from run_baseline_survey import evaluate_rep as _evaluate_rep
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_PROPERTIES = [
    'ginkgo_hic', 'ginkgo_tm2', 'ginkgo_titer', 'ginkgo_pr_cho',
    'ginkgo_acsins', 'ginkgo_purity', 'ginkgo_aggregation',
    'ginkgo_colloidal', 'ginkgo_heparin',
]

DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',           # classical
    'ablang2', 'esm2', 'antiberty',      # LMs
    'igt5', 'prot_t5',                    # additional LMs
    'esm2_concat',                         # ESM2 concat-chain
    'paratope',                            # structural
]

MODELS_LIST = [m for m in DOWNSTREAM_MODELS if m != 'gp']

# Competition baselines (van Niekerk et al. 2026, 113 teams)
# test_top = best team Spearman rho on held-out test set (80 antibodies)
# cv_top = best team cross-validation Spearman rho (246 training antibodies)
# Note: competition only evaluated 5 of 9 properties
PAPER_BASELINES = {
    'ginkgo_hic':         {'test_top': 0.708, 'cv_top': 0.758},
    'ginkgo_tm2':         {'test_top': 0.392, 'cv_top': 0.140},
    'ginkgo_titer':       {'test_top': 0.310, 'cv_top': 0.640},
    'ginkgo_pr_cho':      {'test_top': 0.337, 'cv_top': 0.559},
    'ginkgo_acsins':      {'test_top': 0.356, 'cv_top': 0.653},
    # These 4 were NOT in the competition — no published baselines
    'ginkgo_purity':      {},
    'ginkgo_aggregation': {},
    'ginkgo_colloidal':   {},
    'ginkgo_heparin':     {},
}

# Published method baselines (CV Spearman rho from ginkgobioworks/abdev-benchmark)
PAPER_METHOD_BASELINES = {
    'ablang2_elastic_net': {
        'ginkgo_acsins': 0.509, 'ginkgo_hic': 0.461, 'ginkgo_pr_cho': 0.362,
        'ginkgo_titer': 0.356, 'ginkgo_tm2': 0.101,
    },
    'esm2_tap_ridge': {
        'ginkgo_acsins': 0.480, 'ginkgo_hic': 0.420, 'ginkgo_pr_cho': 0.413,
        'ginkgo_titer': 0.221, 'ginkgo_tm2': 0.265,
    },
    'moe_baseline': {
        'ginkgo_acsins': 0.424, 'ginkgo_hic': 0.656, 'ginkgo_pr_cho': 0.353,
        'ginkgo_titer': 0.184, 'ginkgo_tm2': 0.107,
    },
}


# =============================================================================
# EVALUATION
# =============================================================================

def _fold_assignments_to_splits(fold_assignments, n_folds):
    """Convert Ginkgo's pre-computed fold IDs to (train_idx, val_idx) list."""
    unique_folds = sorted(set(fold_assignments))
    actual_folds = min(n_folds, len(unique_folds))
    if actual_folds < n_folds:
        print(f"  WARNING: requested {n_folds} folds but only "
              f"{len(unique_folds)} pre-computed folds available, "
              f"using {actual_folds}")
    splits = []
    fa = np.asarray(fold_assignments)
    for fold_id in unique_folds[:actual_folds]:
        val_idx = np.where(fa == fold_id)[0]
        train_idx = np.where(fa != fold_id)[0]
        splits.append((train_idx, val_idx))
    return splits


class _PrecomputedFoldSplitter:
    """Sklearn-compatible CV splitter wrapping pre-computed fold assignments.

    Enables find_best_hybrid to use the same hierarchical cluster splits
    that Ginkgo datasets provide, via the cv_splitter parameter.
    """

    def __init__(self, fold_assignments, n_folds):
        self._splits = _fold_assignments_to_splits(fold_assignments, n_folds)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._splits)

    def split(self, X=None, y=None, groups=None):
        yield from self._splits


def _recall_at_k(y_true, y_pred, frac=0.1):
    """Top-K% recall: overlap between true top-K and predicted top-K.

    Matches ginkgobioworks/abdev-benchmark evaluation_metrics.recall_at_k.
    """
    top_k = max(1, int(len(y_true) * frac))
    true_top = set(np.argsort(y_true)[-top_k:].tolist())
    pred_top = set(np.argsort(y_pred)[-top_k:].tolist())
    return len(true_top & pred_top) / top_k


# Properties where LOWER values are better (sign-flip for top-K recall).
# From abdev-benchmark constants.py ASSAY_HIGHER_IS_BETTER:
#   HIC=False, PR_CHO=False, AC-SINS=False (i.e. lower is better for these)
LOWER_IS_BETTER = {'ginkgo_hic', 'ginkgo_pr_cho', 'ginkgo_acsins'}


def evaluate_rep_reg(X, labels, groups, fold_assignments, model, n_folds,
                     prop_name=''):
    """Evaluate one (rep, model) for regression.

    Uses pre-computed fold assignments if available, else GroupKFold.
    Delegates to run_baseline_survey.evaluate_rep for CV, scaling, and metrics.
    Also computes Top-10% recall (benchmark's secondary metric).
    Returns dict with r2, spearman, pearson, rmse, top10_recall as {metric: {mean, std}}.
    """
    from sklearn.base import clone
    from run_baseline_survey import _make_cv, _maybe_scale

    prebuilt = (_fold_assignments_to_splits(fold_assignments, n_folds)
                if fold_assignments is not None else None)
    mean_m, std_m = _evaluate_rep(
        X, labels, groups, model, n_folds, 'regression',
        prebuilt_splits=prebuilt)
    if not mean_m:
        return None
    result = {k: {'mean': mean_m[k], 'std': std_m.get(k, 0.0)} for k in mean_m}

    # Top-10% recall per fold (benchmark's secondary metric)
    X_arr = np.asarray(X, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    if prebuilt:
        splits = prebuilt
    else:
        cv, cv_kw = _make_cv(groups, n_folds, labels_arr)
        splits = list(cv.split(X_arr, labels_arr, **cv_kw))
    recall_scores = []
    flip = prop_name in LOWER_IS_BETTER
    for train_idx, val_idx in splits:
        X_tr, X_va = _maybe_scale(X_arr[train_idx], X_arr[val_idx])
        try:
            m = clone(model)
            m.fit(X_tr, labels_arr[train_idx])
            preds = m.predict(X_va)
            y_t = -labels_arr[val_idx] if flip else labels_arr[val_idx]
            y_p = -preds if flip else preds
            recall_scores.append(_recall_at_k(y_t, y_p, frac=0.1))
        except Exception:
            pass
    if recall_scores:
        result['top10_recall'] = {
            'mean': float(np.mean(recall_scores)),
            'std': float(np.std(recall_scores)),
        }

    return result


def evaluate_property(prop_name, rep_names, n_folds):
    """Evaluate all reps × models on one Ginkgo property.

    Returns dict with per-rep results.
    """
    print(f"\n  --- {prop_name} ---")

    data = load_dataset(prop_name)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    groups = data.get('groups')
    fold_assignments = data.get('fold_assignments')
    n = len(labels)
    print(f"  {n} samples, target={data.get('property_name', prop_name)}")

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag=prop_name,
    )

    if not reps:
        print(f"  No reps available, skipping")
        return None

    # Build hybrid
    # Augmentation config matches ginkgo_* in DATASET_REPS:
    # augmentation = ['paratope', 'structural']
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        from kirby.hybrid import apply_feature_selection
        grp = groups if groups is not None else np.arange(n)
        AUG_REPS = {'paratope', 'structural'}
        base = {k: v for k, v in reps.items() if k not in AUG_REPS}
        aug_dict = {k: v for k, v in reps.items() if k in AUG_REPS}

        # Pass pre-computed hierarchical cluster folds to find_best_hybrid
        # so hybrid search uses the same splits as downstream evaluation.
        # The splitter identity lives in cv_splitter_id so cache metadata
        # captures it — different fold assignments produce different ids.
        cv_kw = {}
        if fold_assignments is not None:
            cv_kw['cv_splitter'] = _PrecomputedFoldSplitter(
                fold_assignments, n_folds)
            cv_kw['cv_splitter_id'] = (
                f'ginkgo_hierarchical_{n_folds}fold_'
                f'h{abs(hash(fold_assignments.tobytes())) % (10**12):012d}')

        search = find_best_hybrid(
            base_reps=base if base else reps,
            labels=labels, groups=grp,
            metric='spearman',
            task_type='regression',
            budgets=FULL_BUDGETS,
            augmentations=aug_dict if aug_dict else None,
            augmentation_strategy='uncapped',
            internal_evaluators=INTERNAL_EVALUATORS,
            speed='full', return_fitted=True, verbose=False, cache=True,
            dataset_name=f"ginkgo_{prop_name}", tier='general',
            adaptive_pruning=0.25, **cv_kw)
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        fitted = search.fitted_results.get('rf')
        if fitted is not None:
            all_reps = {**reps, **(aug_dict or {})}
            fi = {r: {'selected_indices': idx}
                  for r, idx in fitted.selected_indices.items()}
            reps['kirby_hybrid'] = apply_feature_selection(all_reps, fi)
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    # Evaluate
    models = get_model_factories(MODELS_LIST, 'regression', n_samples=n)
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            res = evaluate_rep_reg(X, labels, groups, fold_assignments,
                                   factory(), n_folds, prop_name=prop_name)
            if res is not None:
                rep_results[model_name] = res

        if rep_results:
            # Find best model by Spearman
            best_model = max(rep_results.items(),
                             key=lambda kv: kv[1]['spearman']['mean'])
            best_rho = best_model[1]['spearman']['mean']
            best_r2 = best_model[1]['r2']['mean']
            tag = ' ***' if rep_name == 'kirby_hybrid' else ''
            print(f"    {rep_name:25s} best={best_model[0]:6s}  "
                  f"rho={best_rho:.4f}  R2={best_r2:.4f}{tag}")
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'n_samples': n,
        'property': data.get('property_name', prop_name),
        'results': results,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print cross-property summary table."""
    print(f"\n{'=' * 100}")
    print(f"GINKGO DEVELOPABILITY CROSS-PROPERTY SUMMARY (OPIG B5)")
    print(f"{'=' * 100}")

    # Header
    print(f"\n  {'Property':20s} {'N':>5s} {'Comp.test':>10s} {'Comp.CV':>10s} "
          f"{'Best indiv':>25s} {'rho':>8s} {'Hybrid rho':>10s} {'Delta':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*10} {'-'*25} {'-'*8} {'-'*10} {'-'*8}")

    hybrid_wins = 0
    total = 0

    for prop_name in ALL_PROPERTIES:
        if prop_name not in all_results:
            continue

        ds = all_results[prop_name]
        n = ds['n_samples']
        results = ds['results']

        # Best individual rep (by Spearman)
        best_indiv_name = None
        best_indiv_rho = float('-inf')
        hybrid_rho = float('nan')

        for rep_name, rep_data in results.items():
            best_model_res = max(rep_data['models'].values(),
                                  key=lambda v: v['spearman']['mean'])
            rho = best_model_res['spearman']['mean']
            if rep_name == 'kirby_hybrid':
                hybrid_rho = rho
            elif rho > best_indiv_rho:
                best_indiv_rho = rho
                best_indiv_name = rep_name

        delta = hybrid_rho - best_indiv_rho if not np.isnan(hybrid_rho) else float('nan')
        if not np.isnan(delta) and delta > 0:
            hybrid_wins += 1
        total += 1

        baselines = PAPER_BASELINES.get(prop_name, {})
        test_s = f"{baselines['test_top']:.3f}" if 'test_top' in baselines else "—"
        cv_s = f"{baselines['cv_top']:.3f}" if 'cv_top' in baselines else "—"
        hybrid_s = f"{hybrid_rho:.4f}" if not np.isnan(hybrid_rho) else "—"
        delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "—"
        indiv_label = f"{best_indiv_name}" if best_indiv_name else "—"

        print(f"  {prop_name:20s} {n:5d} {test_s:>10s} {cv_s:>10s} "
              f"{indiv_label:>25s} {best_indiv_rho:8.4f} {hybrid_s:>10s} {delta_s:>8s}")

    print(f"\n  Hybrid wins: {hybrid_wins}/{total} properties")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ginkgo developability experiment (OPIG B5). '
                    'Evaluates all 9 properties with KIRBy antibody reps.')
    parser.add_argument(
        '--properties', type=str, default=None,
        help='Comma-separated property names (default: all 9)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 10)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of CV folds (default: 5, uses pre-computed splits)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_ginkgo_developability.json)')

    args = parser.parse_args()

    properties = (args.properties.split(',') if args.properties
                  else ALL_PROPERTIES)
    rep_names = (args.reps.split(',') if args.reps else DEFAULT_REPS)

    print("=" * 70)
    print("GINKGO DEVELOPABILITY EXPERIMENT (OPIG B5)")
    print(f"  Paper: van Niekerk et al. 2026, mAbs")
    print(f"  Properties: {len(properties)}")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}
    for prop_name in properties:
        if prop_name not in DATASET_REGISTRY:
            print(f"WARNING: Unknown property '{prop_name}', skipping")
            continue
        try:
            prop_results = evaluate_property(prop_name, rep_names, args.n_folds)
            if prop_results is not None:
                all_results[prop_name] = prop_results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Cross-property analysis
    if all_results:
        print_analysis(all_results)

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_ginkgo_developability.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
