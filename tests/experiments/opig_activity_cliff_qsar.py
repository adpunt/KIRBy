#!/usr/bin/env python3
"""
Activity Cliff QSAR Experiment (OPIG — Dablander et al. 2023)

Replicates Dablander et al. 2023 "Exploring QSAR models for activity-cliff
prediction" — tests KIRBy molecular reps on 3 ChEMBL targets and measures
both QSAR performance and activity cliff prediction accuracy.

Scientific questions:
  1. Which molecular rep gives the best QSAR on AC-rich targets?
  2. Does hybrid augmentation reduce activity cliff prediction failures?
  3. How does rep choice affect AC sensitivity specifically?

Targets:
  - D2:   Dopamine D2 receptor (6333 compounds)
  - FXa:  Factor Xa (3605 compounds)
  - Mpro: SARS-CoV-2 main protease (1924 compounds)

Paper baselines (Dablander et al. 2023, J Cheminformatics):
  - 9 models: 3 reps (ECFP, physicochemical descriptors, GIN) x 3 models
    (RF, KNN, MLP)
  - ECFPs + MLP consistently best for QSAR-MAE
  - AC-MCC ~0.5 (known activity) / ~0.3 (unknown) across methods
  - Key finding: "QSAR models frequently fail to predict activity cliffs"

Usage:
    # Smoke test (~3 min, 1 target)
    python tests/experiments/opig_activity_cliff_qsar.py \\
        --targets d2 --n-folds 2 --reps ecfp4,maccs

    # Full evaluation
    python tests/experiments/opig_activity_cliff_qsar.py

Reference:
    Dablander et al. (2023). J Cheminformatics 15:47.
    https://github.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments
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
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import (
    get_representations,
    assign_scaffold_groups,
)
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_TARGETS = ['d2', 'fxa', 'mpro']

DEFAULT_REPS = [
    'ecfp4', 'maccs', 'pdv',             # fingerprints (paper used ECFP)
    'sns',                                 # Sort & Slice (OPIG A1)
    'mordred', 'mol2vec',                  # descriptors
    'avalon',                              # additional FP
]

MODELS_LIST = DOWNSTREAM_MODELS

# Paper baselines (qualitative — exact numbers only in paper figures)
# Dablander et al. 2023 Table 1
PAPER_DATASET_STATS = {
    'd2':   {'compounds': 6333, 'mmps': 35484, 'acs': 461},
    'fxa':  {'compounds': 3605, 'mmps': 21292, 'acs': 1896},
    'mpro': {'compounds': 1924, 'mmps': 12594, 'acs': 521},
}

# Paper findings:
# - ECFP+MLP: best QSAR-MAE across all 3 targets
# - AC-MCC ~0.5 with known activity, ~0.3 without
# - ECFPs > GIN > physicochemical descriptors for QSAR


# =============================================================================
# EVALUATION
# =============================================================================

def _is_binary_rep(X):
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


def evaluate_qsar(X, labels, groups, model, n_folds):
    """Evaluate QSAR regression via scaffold GroupKFold.

    Returns dict with rmse, mae, r2, spearman (mean/std) + per-fold predictions.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    if groups is None or len(np.unique(groups)) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds}-fold group split: "
            f"groups={'None' if groups is None else len(np.unique(groups))} unique")
    cv = GroupKFold(n_splits=n_folds)
    splits = list(cv.split(X, labels, groups=groups))

    is_binary = _is_binary_rep(X)
    rmse_scores, mae_scores, r2_scores, spearman_scores = [], [], [], []
    all_preds = np.full_like(labels, np.nan)

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
        all_preds[val_idx] = preds

        ss_res = np.sum((y_val - preds) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        rmse_scores.append(np.sqrt(np.mean((y_val - preds) ** 2)))
        mae_scores.append(mean_absolute_error(y_val, preds))

        if len(y_val) > 2:
            rho, _ = spearmanr(y_val, preds)
            spearman_scores.append(rho if not np.isnan(rho) else 0.0)

    if not rmse_scores:
        return None

    return {
        'rmse': {'mean': np.mean(rmse_scores), 'std': np.std(rmse_scores)},
        'mae': {'mean': np.mean(mae_scores), 'std': np.std(mae_scores)},
        'r2': {'mean': np.mean(r2_scores), 'std': np.std(r2_scores)},
        'spearman': {'mean': np.mean(spearman_scores), 'std': np.std(spearman_scores)},
        'predictions': all_preds,
    }


def _ac_clf_metrics(y_true, y_pred_prob):
    """Compute AC binary classification metrics from true labels and predicted probs.

    Matches Dablander et al. 2023 scoring.py: MCC as primary, plus AUROC, etc.
    """
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    y_pred = (np.asarray(y_pred_prob) > 0.5).astype(int)
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    if n == 0:
        return None
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auroc = float(roc_auc_score(y_true, y_pred_prob))
    except ValueError:
        auroc = float('nan')
    return {
        'mcc': mcc, 'accuracy': accuracy, 'auroc': auroc,
        'sensitivity': sensitivity, 'specificity': specificity,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'n': n,
    }


def evaluate_ac_performance(labels, predictions, mmp_pairs, ac_threshold=2.0):
    """Evaluate activity cliff prediction from QSAR predictions.

    Matches Dablander et al. 2023:
      - AC = |ΔpActivity| >= 2.0 (label 1)
      - Non-AC = |ΔpActivity| < 1.0 (label 0)
      - Half-cliffs (1.0 <= |Δ| < 2.0) are EXCLUDED from evaluation
      - Predicted AC probability = 2*arctan(|Δpred|/1.5)/π  (soft score)

    Returns dict with pooled + per-partition AC metrics.
    """
    if not mmp_pairs:
        return None

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    AC_SCALE = 1.5  # arctan scaling parameter (Dablander code default)

    # Build arrays: true binary AC label, predicted AC probability
    # Exclude half-cliffs and pairs with missing predictions
    true_labels = []
    pred_probs = []
    for i, j, _ in mmp_pairs:
        if np.isnan(predictions[i]) or np.isnan(predictions[j]):
            continue
        true_delta = abs(labels[i] - labels[j])

        # Exclude half-cliffs: 1.0 <= |Δ| < 2.0
        if 1.0 <= true_delta < ac_threshold:
            continue

        is_ac = int(true_delta >= ac_threshold)
        pred_delta = abs(predictions[i] - predictions[j])
        # Dablander arctan soft scoring
        pred_prob = 2.0 * np.arctan(pred_delta / AC_SCALE) / np.pi

        true_labels.append(is_ac)
        pred_probs.append(pred_prob)

    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    if len(true_labels) == 0:
        return None

    result = _ac_clf_metrics(true_labels, pred_probs)
    if result is None:
        return None

    n_excluded_half = sum(
        1 for i, j, _ in mmp_pairs
        if not (np.isnan(predictions[i]) or np.isnan(predictions[j]))
        and 1.0 <= abs(labels[i] - labels[j]) < ac_threshold
    )
    result['n_half_cliffs_excluded'] = n_excluded_half
    result['ac_threshold'] = ac_threshold
    result['ac_scale'] = AC_SCALE

    # Rename primary metric key for backward compat
    result['ac_mcc'] = result['mcc']
    return result


def evaluate_target(target, rep_preset, n_folds):
    """Evaluate all reps × models on one target.

    Returns dict with QSAR metrics + AC metrics per rep × model.
    """
    from kirby.datasets.chembl_qsar import load_chembl_qsar

    print(f"\n{'=' * 60}")
    print(f"Target: {target}")
    print(f"{'=' * 60}")

    data = load_chembl_qsar(target)
    smiles = data['smiles']
    labels = data['labels']
    mmp_pairs = data['mmp_pairs']
    n = len(smiles)

    # Scaffold groups
    groups, n_groups = assign_scaffold_groups(smiles)
    print(f"  {n} compounds, {n_groups} scaffolds, "
          f"{data['n_activity_cliffs']} ACs (threshold={data['ac_threshold']})")

    # Generate representations
    reps = get_representations(smiles, rep_preset)
    print(f"  {len(reps)} reps loaded")

    if not reps:
        print("  No reps available, skipping target")
        return None

    # Build hybrid
    # Augmentation config matches molecular datasets (caco2, logd) in DATASET_REPS:
    # augmentation = ['maccs', 'pharmacophore3d']
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        from kirby.hybrid import apply_feature_selection
        AUG_REPS = {'maccs', 'pharmacophore3d'}
        base = {k: v for k, v in reps.items() if k not in AUG_REPS}
        aug_dict = {k: v for k, v in reps.items() if k in AUG_REPS}
        search = find_best_hybrid(
            base_reps=base if base else reps,
            labels=labels, groups=groups,
            metric='mae',
            task_type='regression',
            budgets=FULL_BUDGETS,
            augmentations=aug_dict if aug_dict else None,
            augmentation_strategy='uncapped',
            internal_evaluators=INTERNAL_EVALUATORS,
            speed='full', return_fitted=True, verbose=False, cache=True,
            dataset_name=f"chembl_qsar_{target}", tier='general',
            adaptive_pruning=0.25)
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        fitted = search.fitted_results.get('rf')
        if fitted is not None:
            all_reps = {**reps, **(aug_dict or {})}
            fi = {r: {'selected_indices': idx}
                  for r, idx in fitted.selected_indices.items()}
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
        try:
            X = np.asarray(X, dtype=np.float64)
        except Exception:
            continue
        if X.ndim != 2 or X.shape[0] != n:
            continue

        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                qsar_res = evaluate_qsar(X, labels, groups, factory(), n_folds)
            except ValueError as e:
                print(f"    {rep_name:25s} {model_name:8s}  SKIPPED: {e}")
                continue
            elapsed = time.time() - t0

            if qsar_res is None:
                continue

            # AC analysis using out-of-fold predictions
            predictions = qsar_res.pop('predictions')
            ac_res = evaluate_ac_performance(labels, predictions, mmp_pairs,
                                              data['ac_threshold'])

            model_result = {'qsar': qsar_res}
            if ac_res is not None:
                model_result['ac'] = ac_res

            rep_results[model_name] = model_result

            mae = qsar_res['mae']['mean']
            mcc = ac_res['ac_mcc'] if ac_res else float('nan')
            tag = ' ***' if rep_name == 'kirby_hybrid' else ''
            print(f"    {rep_name:25s} {model_name:8s}  "
                  f"MAE={mae:.4f}  AC-MCC={mcc:.4f}  ({elapsed:.1f}s){tag}")

        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'n_compounds': n,
        'n_scaffolds': n_groups,
        'n_mmps': len(mmp_pairs),
        'n_acs': data['n_activity_cliffs'],
        'target_name': data['target_name'],
        'results': results,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print cross-target QSAR + AC comparison."""
    print(f"\n{'=' * 100}")
    print(f"ACTIVITY CLIFF QSAR COMPARISON (Dablander et al. 2023)")
    print(f"{'=' * 100}")

    for target in ALL_TARGETS:
        if target not in all_results:
            continue

        tr = all_results[target]
        results = tr['results']
        stats = PAPER_DATASET_STATS.get(target, {})

        print(f"\n  --- {tr['target_name']} ({tr['n_compounds']} compounds, "
              f"{tr['n_acs']} ACs) ---")

        print(f"  {'Rep':25s} {'Model':8s} {'MAE':>8s} {'RMSE':>8s} "
              f"{'R2':>8s} {'AC-MCC':>8s} {'AC-Sens':>8s}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        rows = []
        for rep_name, rep_data in results.items():
            for model_name, model_res in rep_data['models'].items():
                qsar = model_res['qsar']
                ac = model_res.get('ac', {})
                rows.append((
                    rep_name, model_name,
                    qsar['mae']['mean'], qsar['rmse']['mean'],
                    qsar['r2']['mean'],
                    ac.get('ac_mcc', float('nan')),
                    ac.get('sensitivity', float('nan')),
                ))

        # Sort by MAE (lower is better)
        rows.sort(key=lambda r: r[2])

        for rep_name, model_name, mae, rmse, r2, mcc, sens in rows:
            tag = ' ***' if rep_name == 'kirby_hybrid' else ''
            mcc_s = f"{mcc:.4f}" if not np.isnan(mcc) else "—"
            sens_s = f"{sens:.4f}" if not np.isnan(sens) else "—"
            print(f"  {rep_name:25s} {model_name:8s} {mae:8.4f} {rmse:8.4f} "
                  f"{r2:8.4f} {mcc_s:>8s} {sens_s:>8s}{tag}")

    # Cross-target summary
    print(f"\n  Cross-target summary:")
    print(f"  {'Target':10s} {'Best rep':25s} {'MAE':>8s} {'AC-MCC':>8s} {'Hybrid MAE':>11s} {'Delta':>8s}")
    print(f"  {'-'*10} {'-'*25} {'-'*8} {'-'*8} {'-'*11} {'-'*8}")

    for target in ALL_TARGETS:
        if target not in all_results:
            continue

        tr = all_results[target]
        results = tr['results']

        best_indiv = None
        best_mae = float('inf')
        hybrid_mae = float('nan')

        for rep_name, rep_data in results.items():
            for model_name, model_res in rep_data['models'].items():
                mae = model_res['qsar']['mae']['mean']
                if rep_name == 'kirby_hybrid':
                    if np.isnan(hybrid_mae) or mae < hybrid_mae:
                        hybrid_mae = mae
                else:
                    if mae < best_mae:
                        best_mae = mae
                        best_indiv = rep_name

        # Best AC-MCC for any method
        best_mcc = max(
            (m.get('ac', {}).get('ac_mcc', 0)
             for rd in results.values() for m in rd['models'].values()),
            default=0
        )

        delta = hybrid_mae - best_mae if not np.isnan(hybrid_mae) else float('nan')
        h_s = f"{hybrid_mae:.4f}" if not np.isnan(hybrid_mae) else "—"
        d_s = f"{delta:+.4f}" if not np.isnan(delta) else "—"

        print(f"  {target:10s} {best_indiv or '—':25s} {best_mae:8.4f} "
              f"{best_mcc:8.4f} {h_s:>11s} {d_s:>8s}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Activity cliff QSAR experiment (Dablander 2023). '
                    'Tests molecular reps on 3 ChEMBL targets.')
    parser.add_argument(
        '--targets', type=str, default=None,
        help='Comma-separated target names (default: d2,fxa,mpro)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names or preset '
             '(default: ecfp4,maccs,pdv,sns,mordred,mol2vec,avalon)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of scaffold GroupKFold folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_activity_cliff_qsar.json)')

    args = parser.parse_args()

    targets = (args.targets.split(',') if args.targets else ALL_TARGETS)
    rep_preset = args.reps or ','.join(DEFAULT_REPS)

    print("=" * 70)
    print("ACTIVITY CLIFF QSAR EXPERIMENT (Dablander et al. 2023)")
    print(f"  Paper: J Cheminformatics 15:47")
    print(f"  Targets: {targets}")
    print(f"  Reps: {rep_preset}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}
    for target in targets:
        try:
            target_results = evaluate_target(target, rep_preset, args.n_folds)
            if target_results is not None:
                all_results[target] = target_results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Cross-target analysis
    if all_results:
        print_analysis(all_results)

    # Save results (remove predictions arrays for JSON)
    save_results = {}
    for target, tr in all_results.items():
        save_results[target] = {
            k: v for k, v in tr.items() if k != 'results'
        }
        save_results[target]['results'] = {}
        for rep_name, rep_data in tr['results'].items():
            save_results[target]['results'][rep_name] = rep_data

    output_path = (args.output or
                   'tests/results/comparisons/opig_activity_cliff_qsar.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(save_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
