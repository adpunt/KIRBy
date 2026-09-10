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

# Load xgboost/lightgbm native OpenMP runtimes BEFORE rdkit (imported during
# rep generation) to avoid a duplicate-libomp segfault on macOS when these
# models and threaded sklearn estimators run in sequence. Optional deps —
# guard so environments without them are unaffected.
try:
    import xgboost as _xgboost  # noqa: F401
except Exception:
    pass
try:
    import lightgbm as _lightgbm  # noqa: F401
except Exception:
    pass

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
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
    honest_rescore, build_honest_rescore_input,
    hr_exclude_folds,
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
        except Exception as e:
            # Log fold-fit failures (OPIG-18) instead of silently dropping —
            # a missing fold biases the per-fold metrics and pooled preds.
            print(f"      [fold fit failed: {type(model).__name__} "
                  f"{type(e).__name__}: {e}]", flush=True)
            continue

        preds = m.predict(X_val)
        all_preds[val_idx] = preds

        ss_res = np.sum((y_val - preds) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
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
        'r2': {'mean': np.nanmean(r2_scores) if r2_scores else float('nan'),
               'std': np.nanstd(r2_scores) if r2_scores else float('nan')},
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


def evaluate_target(target, rep_preset, n_folds, pruning_mode='default_pruning'):
    """Evaluate all reps × models on one target.

    Returns dict with QSAR metrics + AC metrics per rep × model.
    """
    from kirby.datasets.chembl_qsar import load_chembl_qsar

    print(f"\n{'=' * 60}")
    print(f"Target: {target}")
    print(f"{'=' * 60}")

    # Test-only row cap (smoke tests). No-op unless KIRBY_SMOKE_N is set.
    _smoke_n = os.environ.get('KIRBY_SMOKE_N', '').strip()
    _max = int(_smoke_n) if _smoke_n.isdigit() and int(_smoke_n) > 0 else None
    data = load_chembl_qsar(target, max_samples=_max)
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
    # honest_block: the per-model, HELD-OUT re-score arm (contract json_convention).
    # Built purely from the SearchResult's per-fold scores — NO leaky full-data
    # feature selection, NO cross-model MAX. Stays None if the search is unavailable.
    honest_block = None
    try:
        from experiments._shared import (
            resolve_pruning_kwargs, get_best_search_for_downstream)
        AUG_REPS = {'maccs', 'pharmacophore3d'}
        base = {k: v for k, v in reps.items() if k not in AUG_REPS}
        aug_dict = {k: v for k, v in reps.items() if k in AUG_REPS}
        _pkw = resolve_pruning_kwargs(pruning_mode, len(labels),
                                       dataset_label=f'chembl_qsar/{target}')
        # Test-only: shrink the hybrid search under KIRBY_SMOKE_N (tiny budget,
        # rf-only evaluator, cheap lightgbm_gain importance) so the on-demand
        # search finishes on tiny local subsets. No-op when the env var is unset,
        # so server runs use FULL_BUDGETS + INTERNAL_EVALUATORS unchanged.
        _is_smoke = bool(os.environ.get('KIRBY_SMOKE_N', '').strip())
        _smoke_search = ({'budgets': [25], 'importance_methods': ['lightgbm_gain']}
                         if _is_smoke else {})
        _eval_list = ['rf'] if _is_smoke else INTERNAL_EVALUATORS
        # In smoke mode skip the augmentation pool: passing augmentations adds
        # the 'uncapped'/'capped' pool conditions, each running a full
        # augmentation search — far too heavy to finish locally. The base hybrid
        # search still exercises the pipeline. Server runs keep augmentations.
        _smoke_augs = None if _is_smoke else (aug_dict if aug_dict else None)
        search, _meta = get_best_search_for_downstream(
            base if base else reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=_smoke_augs,
            task_type='regression',
            dataset_name='chembl_qsar',
            return_fitted=True,
            internal_evaluators=_eval_list,
            condition_suffix=target,
            adaptive_pruning=0.25,
            **_smoke_search,
            **_pkw,
        )
        # Honest per-model held-out comparison (hybrid candidates vs baseline
        # reps), built straight from the search's per-fold scores. The reader
        # feeds this block to honest_rescore(): for each available model it
        # re-selects that model's own best hybrid recipe and best baseline rep
        # per held-out fold, so neither arm is chosen on the fold it's scored on.
        # This SUPERSEDES the same-fold argmax comparison in print_analysis for
        # the honest hybrid-vs-single verdict; the legacy per-rep table below is
        # retained unchanged for backward compatibility.
        if search is not None:
            honest_block = build_honest_rescore_input(search)

        top3 = extract_top3_base_reps(search) if search is not None else None
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        # No 'kirby_hybrid' representation is added to `reps`. It used to be
        # built by refitting the search's winning recipe on every compound, so
        # each fold's own labels had already helped choose the feature indices
        # it was then scored on. Its MAE, RMSE, R2 and AC-MCC were measured on
        # data that built it. The honest per-model held-out comparison below
        # (honest_rescore over the search's per-fold scores) is the only hybrid
        # number this experiment reports. Removed 2026-09-04, split audit.
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
            print(f"    {rep_name:25s} {model_name:8s}  "
                  f"MAE={mae:.4f}  AC-MCC={mcc:.4f}  ({elapsed:.1f}s)")

        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    target_out = {
        'n_compounds': n,
        'n_scaffolds': n_groups,
        'n_mmps': len(mmp_pairs),
        'n_acs': data['n_activity_cliffs'],
        'target_name': data['target_name'],
        'results': results,
    }
    # Attach the honest re-score block under the exact contract key name so the
    # reader can load result[target]['honest_rescore'] and call honest_rescore().
    if honest_block is not None:
        target_out['honest_rescore'] = honest_block
    return target_out


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
            mcc_s = f"{mcc:.4f}" if not np.isnan(mcc) else "—"
            sens_s = f"{sens:.4f}" if not np.isnan(sens) else "—"
            print(f"  {rep_name:25s} {model_name:8s} {mae:8.4f} {rmse:8.4f} "
                  f"{r2:8.4f} {mcc_s:>8s} {sens_s:>8s}")

    # Cross-target summary. The Hybrid MAE and Delta columns were removed on
    # 2026-09-04 (split audit): both came from a hybrid whose features were
    # refit on every compound's label, so the number was measured on rows that
    # chose it. The honest hybrid-vs-best-single table below replaces them.
    print(f"\n  Cross-target summary:")
    print(f"  {'Target':10s} {'Best rep':25s} {'MAE':>8s} {'AC-MCC':>8s}")
    print(f"  {'-'*10} {'-'*25} {'-'*8} {'-'*8}")

    for target in ALL_TARGETS:
        if target not in all_results:
            continue

        tr = all_results[target]
        results = tr['results']

        best_indiv = None
        best_mae = float('inf')

        for rep_name, rep_data in results.items():
            for model_name, model_res in rep_data['models'].items():
                mae = model_res['qsar']['mae']['mean']
                if mae < best_mae:
                    best_mae = mae
                    best_indiv = rep_name

        # Best AC-MCC for any method (MCC range [-1, 1]; ignore NaN entries)
        _mccs = [
            m.get('ac', {}).get('ac_mcc', float('nan'))
            for rd in results.values() for m in rd['models'].values()
        ]
        _mccs = [v for v in _mccs if not (isinstance(v, float) and np.isnan(v))]
        best_mcc = max(_mccs) if _mccs else float('nan')

        print(f"  {target:10s} {best_indiv or '—':25s} {best_mae:8.4f} "
              f"{best_mcc:8.4f}")

    # Honest per-model, held-out QSAR-MAE verdict (selection-optimism corrected).
    # Each model's own hybrid recipe and own best baseline rep are re-selected per
    # held-out fold, so the delta is not inflated by same-fold argmax. Only shown
    # where the search populated the honest_rescore block (fresh runs / warmed
    # per-fold baseline cache); MAE is lower-is-better so +delta = hybrid better.
    have_honest = any('honest_rescore' in all_results.get(t, {}) for t in ALL_TARGETS)
    if have_honest:
        print(f"\n  Honest per-model held-out (LOFO) QSAR-MAE — hybrid vs best single rep:")
        print(f"  {'Target':10s} {'Model':8s} {'Hybrid':>9s} {'BestSingle':>11s} "
              f"{'Delta':>9s} {'Folds':>6s}")
        print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*11} {'-'*9} {'-'*6}")
        for target in ALL_TARGETS:
            blk = all_results.get(target, {}).get('honest_rescore')
            if not blk:
                continue
            res = honest_rescore(blk['candidates_by_model'],
                                 blk['baselines_by_model'],
                                 blk['lower_is_better'],
                                 exclude_folds=hr_exclude_folds(blk))
            for model_name in sorted(res):
                r = res[model_name]
                h_v = r['hybrid_honest']
                b_v = r['best_single_honest']
                d_v = r['delta']
                h_str = f"{h_v:.4f}" if not np.isnan(h_v) else "—"
                b_str = f"{b_v:.4f}" if not np.isnan(b_v) else "—"
                d_str = f"{d_v:+.4f}" if not np.isnan(d_v) else "—"
                flag = " (single-fold; NOT honest)" if r['single_fold'] else ""
                print(f"  {target:10s} {model_name:8s} {h_str:>9s} {b_str:>11s} "
                      f"{d_str:>9s} {r['n_folds_used']:>6d}{flag}")


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
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()


    apply_distance_metric_arg(args)
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
            target_results = evaluate_target(target, rep_preset, args.n_folds,
                                              pruning_mode=args.pruning_mode)
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

    # Replace non-finite floats (NaN/inf) with null before serialising. These
    # arise from genuinely undefined metrics at tiny n (e.g. activity-cliff
    # AUROC when a subset contains 0 cliffs), and NaN is not valid JSON. null
    # is the honest representation of "undefined", and keeps the file NaN-free.
    def _nan_to_none(o):
        if isinstance(o, float):
            return None if (o != o or o in (float('inf'), float('-inf'))) else o
        if isinstance(o, dict):
            return {k: _nan_to_none(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_nan_to_none(v) for v in o]
        return o

    with open(p, 'w') as f:
        json.dump(_nan_to_none(convert_numpy(save_results)), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
