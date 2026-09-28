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
    fold_reselection, build_fold_reselection_input,
    hr_exclude_folds,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    resolve_comparison_models,
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
    # Which fold each compound was HELD OUT in. Dablander splits the matched
    # pairs by whether a compound's activity was available to the model, so the
    # pair-level metrics need this. -1 means no fold produced a prediction.
    held_out_fold = np.full(len(labels), -1, dtype=int)

    for _fold_i, (train_idx, val_idx) in enumerate(splits):
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
        held_out_fold[val_idx] = _fold_i

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
        'held_out_fold': held_out_fold,
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
    # Dablander p.7: "we sometimes encountered the edge case where there were
    # no positive predictions; we then set MCC=0 and ignored ill-defined
    # precision measurements". He names SARS-CoV-2 main protease as where this
    # happens, which is the smallest of the three sets. Without the floor a
    # model that predicts no cliffs at all lands at a small negative MCC and
    # reads as worse than chance rather than as silent.
    n_positive_predictions = tp + fp
    mcc_floored = (n_positive_predictions == 0)
    if mcc_floored:
        mcc = 0.0
    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Precision is ill-defined with no positive predictions; Dablander drops it
    # from the average rather than calling it 0.
    precision = (tp / n_positive_predictions
                 if n_positive_predictions > 0 else float('nan'))
    try:
        auroc = float(roc_auc_score(y_true, y_pred_prob))
    except ValueError:
        auroc = float('nan')
    return {
        'mcc': mcc, 'accuracy': accuracy, 'auroc': auroc,
        'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision, 'mcc_floored_no_positive_preds': mcc_floored,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'n': n,
    }


def evaluate_ac_performance(labels, predictions, mmp_pairs, ac_threshold=2.0,
                            held_out_fold=None):
    """Evaluate activity cliff prediction from QSAR predictions.

    Matches Dablander et al. 2023:
      - AC = |ΔpActivity| >= 2.0 (label 1)
      - Non-AC = |ΔpActivity| <= 1.0 (label 0)
      - Half-cliffs (1.0 < |Δ| < 2.0) are EXCLUDED from evaluation.
        Strict at 1.0, as in his notebook (`1 < abs(...) < 2`): a pair exactly
        ten-fold apart is a non-cliff, not a half-cliff. Until 2026-09-28 this
        file dropped those pairs, 32 of d2's and 130 of fxa's.
      - Predicted AC probability = 2*arctan(|Δpred|/1.5)/π  (soft score)

    That soft score crosses 0.5 exactly when |Δpred| > 1.5, which is
    Dablander's d_crit, so the classification is his; the arctan only makes it
    a graded score. He reports MCC, sensitivity and precision and no AUROC.

    THE PAIR PARTITION (added 2026-09-23). Dablander does not pool the matched
    pairs. He splits them by whether each compound's measured activity was
    available to the model:
      M_inter - one compound held out, the other in training, so ONE MEASURED
                ACTIVITY IS KNOWN. He classifies with |a(known) - f(other)|.
                This is his "activity of one compound is given" arm, MCC ~0.5.
      M_test  - both compounds held out together, both activities unknown,
                classified with |f(s) - f(s~)|. His MCC ~0.3 arm.
    Under k-fold cross-validation each compound is held out exactly once, so a
    pair lands in M_test when its two compounds share a held-out fold and in
    M_inter when they do not. Pooling the two mixes an easier task into a
    harder one and the pooled number matches neither of his figures.

    `held_out_fold` is the per-compound fold index from evaluate_qsar. Without
    it only the pooled block is computed, which is what this function did
    before.

    HOW THE PARTITION IS BUILT (changed 2026-09-28, read against Dablander's
    modules/scoring.py and modules/data_splitting.py). He builds M_inter and
    M_test separately inside every split. In a split, a pair with one compound
    held out is an M_inter case scored |a(training compound) - f(held-out)|.
    Under k-fold, a pair whose compounds sit in different folds is therefore
    an M_inter case twice, once in each compound's fold. This file used to
    average the two distances into one case; it now keeps both, as he does.
    He also computes MCC per split and reports the mean over splits
    (summarise_scores_from_cubic_scores_array, nanmean). That mean is stored
    as 'mcc_fold_mean' beside the pooled 'ac_mcc', with the per-fold values.

    Returns the pooled metrics, plus 'by_partition' with 'M_inter' and
    'M_test' when held_out_fold is given.
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

        # Exclude half-cliffs: 1.0 < |Δ| < 2.0 (strict, as Dablander)
        if 1.0 < true_delta < ac_threshold:
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
        and 1.0 < abs(labels[i] - labels[j]) < ac_threshold
    )
    result['n_half_cliffs_excluded'] = n_excluded_half
    result['ac_threshold'] = ac_threshold
    result['ac_scale'] = AC_SCALE

    # Rename primary metric key for backward compat
    result['ac_mcc'] = result['mcc']

    # ---- Dablander's pair partition ------------------------------------
    # Same cliff definition, same threshold, same soft score; the only
    # difference is WHICH pairs go in and, for M_inter, that the known
    # compound contributes its MEASURED activity rather than a prediction.
    if held_out_fold is not None:
        held_out_fold = np.asarray(held_out_fold)
        # Non-half-cliff pairs and their true labels, once.
        kept = []
        for i, j, _ in mmp_pairs:
            true_delta = abs(labels[i] - labels[j])
            if 1.0 < true_delta < ac_threshold:
                continue
            kept.append((i, j, int(true_delta >= ac_threshold)))
        parts = {'M_inter': ([], []), 'M_test': ([], [])}
        per_fold = {'M_inter': [], 'M_test': []}
        for f in sorted(int(v) for v in np.unique(held_out_fold) if v >= 0):
            fold_parts = {'M_inter': ([], []), 'M_test': ([], [])}
            for i, j, is_ac in kept:
                in_i = held_out_fold[i] == f
                in_j = held_out_fold[j] == f
                if in_i and in_j:
                    # both held out in this fold -> both activities unknown
                    if np.isnan(predictions[i]) or np.isnan(predictions[j]):
                        continue
                    pred_delta = abs(predictions[i] - predictions[j])
                    key = 'M_test'
                elif in_i or in_j:
                    # one held out in this fold; under k-fold the other was a
                    # training compound here, so its measured activity is known
                    h, t = (i, j) if in_i else (j, i)
                    if np.isnan(predictions[h]):
                        continue
                    pred_delta = abs(labels[t] - predictions[h])
                    key = 'M_inter'
                else:
                    continue
                prob = 2.0 * np.arctan(pred_delta / AC_SCALE) / np.pi
                fold_parts[key][0].append(is_ac)
                fold_parts[key][1].append(prob)
                parts[key][0].append(is_ac)
                parts[key][1].append(prob)
            for key, (yt, yp) in fold_parts.items():
                fm = (_ac_clf_metrics(np.array(yt), np.array(yp))
                      if yt else None)
                per_fold[key].append({
                    'fold': f, 'n_pairs': len(yt), 'n_acs': int(sum(yt)),
                    'mcc': fm['mcc'] if fm else float('nan'),
                    'sensitivity': fm['sensitivity'] if fm else float('nan'),
                    'precision': fm['precision'] if fm else float('nan'),
                })
        by_part = {}
        for key, (yt, yp) in parts.items():
            if not yt:
                by_part[key] = {'n_pairs': 0}
                continue
            m = _ac_clf_metrics(np.array(yt), np.array(yp))
            if m is None:
                by_part[key] = {'n_pairs': len(yt)}
                continue
            m['n_pairs'] = len(yt)
            m['n_acs'] = int(sum(yt))
            m['ac_mcc'] = m['mcc']
            _fold_mccs = [r['mcc'] for r in per_fold[key]
                          if not np.isnan(r['mcc'])]
            m['mcc_fold_mean'] = (float(np.mean(_fold_mccs))
                                  if _fold_mccs else float('nan'))
            m['per_fold'] = per_fold[key]
            by_part[key] = m
        result['by_partition'] = by_part
    return result


def evaluate_hybrid_per_fold(base, aug_dict, labels, groups, n_folds, models,
                             target, search_kwargs):
    """Build a hybrid inside every scaffold fold and predict that fold's held-out rows.

    The recipe for a fold comes from a search over that fold's TRAINING rows
    only. The whole-compound search cannot be used here: its feature indices
    were chosen with the held-out labels in hand, so any cliff number it
    produced would be measured on the rows that built it.

    Returns {model_name: {'qsar': ..., 'predictions': ..., 'per_fold_blueprint':
    ..., 'per_fold_n_features': ...}}. 'predictions' is one out-of-fold value
    per compound, indexed like `labels`, nan where a fold produced nothing —
    the same array evaluate_ac_performance takes for a representation.
    """
    from experiments._shared import (
        build_fold_hybrid, get_best_for_downstream,
        get_best_search_for_downstream)

    labels = np.asarray(labels, dtype=np.float64)
    n = len(labels)
    # Row slicing below needs arrays; asarray is a no-op on one already.
    base = {k: np.asarray(v) for k, v in base.items()}
    aug_dict = ({k: np.asarray(v) for k, v in aug_dict.items()}
                if aug_dict else None)
    # GroupKFold assigns folds from `groups` and the row count alone, so these
    # are the folds every representation was scored on in evaluate_qsar.
    splits = list(GroupKFold(n_splits=n_folds).split(
        np.zeros((n, 1)), labels, groups=groups))

    preds = {mn: np.full(n, np.nan) for mn in models}
    # Which fold each compound was held out in, so the AC arm can split the
    # matched pairs into Dablander's M_inter and M_test. Filled per model
    # because a model whose search found no recipe on a fold has no prediction
    # for that fold's compounds.
    held_out = {mn: np.full(n, -1, dtype=int) for mn in models}
    folds = {mn: {'rmse': [], 'mae': [], 'r2': [], 'spearman': [],
                  'blueprint': [], 'n_features': []} for mn in models}

    # {downstream model: [fold indices where its search scored no configuration]}
    _no_recipe = {}
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        tr = {k: v[train_idx] for k, v in base.items()}
        va = {k: v[val_idx] for k, v in base.items()}
        aug_tr = {k: v[train_idx] for k, v in aug_dict.items()} if aug_dict else None
        aug_va = {k: v[val_idx] for k, v in aug_dict.items()} if aug_dict else None
        y_tr, y_va = labels[train_idx], labels[val_idx]
        g_tr = groups[train_idx] if groups is not None else None

        try:
            fold_search, _fmeta = get_best_search_for_downstream(
                tr, y_tr, g_tr,
                downstream_model='rf', budget=None,
                augmentations=aug_tr,
                task_type='regression',
                dataset_name='chembl_qsar',
                condition_suffix=f'{target}_heldout_fold{fold_i}',
                adaptive_pruning=0.25,
                **search_kwargs,
            )
        except Exception as e:
            print(f"    fold {fold_i} hybrid search failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            fold_search = None
        if fold_search is None:
            print(f"    fold {fold_i}: no hybrid recipe from this fold's "
                  f"training rows", flush=True)
            continue

        for model_name, factory in models.items():
            best = get_best_for_downstream(fold_search, model_name)
            if best is None:
                # ⛔ SAID OUT LOUD SINCE 2026-09-18. This was a bare `continue`,
                # so a downstream model the fold's search never scored simply
                # had no hybrid row, and the result file did not say which model
                # or how often. It is not hypothetical: svm's effective row
                # ceiling drops to 5,000 training molecules under the 'uncapped'
                # augmentation strategy (_STRATEGY_MAX_SAMPLES at
                # find_best_hybrid.py:2862), and d2 carries 6,333 compounds, so
                # on a server run — which passes augmentations, unlike a smoke
                # run — svm can be scored in the per-representation table below
                # and absent from the hybrid arm. A reader comparing the two
                # then reads svm's missing hybrid as a loss rather than as an
                # absence.
                _no_recipe.setdefault(model_name, []).append(fold_i)
                continue
            bp = best['blueprint']
            try:
                X_tr_h, X_va_h, _fi = build_fold_hybrid(
                    tr, va, y_tr, bp,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=g_tr, model=factory(),
                    metric_fn=mean_absolute_error, lower_is_better=True)
                scaler = StandardScaler()
                X_tr_h = scaler.fit_transform(X_tr_h)
                X_va_h = scaler.transform(X_va_h)
                m = factory()
                m.fit(X_tr_h, y_tr)
                p = m.predict(X_va_h)
            except Exception as e:
                print(f"    fold {fold_i} {model_name} hybrid failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue

            preds[model_name][val_idx] = p
            held_out[model_name][val_idx] = fold_i
            fm = folds[model_name]
            ss_res = np.sum((y_va - p) ** 2)
            ss_tot = np.sum((y_va - y_va.mean()) ** 2)
            fm['r2'].append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
            fm['rmse'].append(np.sqrt(np.mean((y_va - p) ** 2)))
            fm['mae'].append(mean_absolute_error(y_va, p))
            if len(y_va) > 2:
                rho, _ = spearmanr(y_va, p)
                fm['spearman'].append(rho if not np.isnan(rho) else 0.0)
            fm['n_features'].append(int(X_tr_h.shape[1]))
            fm['blueprint'].append({
                'budget': bp.budget,
                'allocation': bp.allocation_method,
                'importance': bp.importance_method,
                'evaluator': bp.internal_evaluator,
            })

    if _no_recipe:
        print(f"    ⛔ no hybrid recipe for: " + ", ".join(
            f"{m} on {len(f)} of {len(splits)} folds" for m, f in sorted(_no_recipe.items())))
        print(f"       Those models are scored in the per-representation table "
              f"and absent from the hybrid arm. Absence, not a loss.")

    out = {}
    for model_name, fm in folds.items():
        if not fm['rmse']:
            continue
        out[model_name] = {
            'qsar': {
                'rmse': {'mean': np.mean(fm['rmse']), 'std': np.std(fm['rmse'])},
                'mae': {'mean': np.mean(fm['mae']), 'std': np.std(fm['mae'])},
                'r2': {'mean': np.nanmean(fm['r2']),
                       'std': np.nanstd(fm['r2'])},
                'spearman': {
                    'mean': np.mean(fm['spearman']) if fm['spearman'] else float('nan'),
                    'std': np.std(fm['spearman']) if fm['spearman'] else float('nan')},
                # How many of the folds produced a hybrid for this model. A
                # mean over fewer folds than the representations got is not the
                # same number, so it travels with the score.
                'n_folds': len(fm['rmse']),
            },
            'predictions': preds[model_name],
            'held_out_fold': held_out[model_name],
            'per_fold_blueprint': fm['blueprint'],
            'per_fold_n_features': fm['n_features'],
            # Folds where this model's search scored no configuration, so it
            # got no hybrid. Empty for a model the search covered on every
            # fold. Travels with the score so a short mean is traceable.
            'folds_without_a_recipe': sorted(_no_recipe.get(model_name, [])),
        }
    return out


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
    # reselection_block: the per-model, HELD-OUT re-score arm (contract json_convention).
    # Built purely from the SearchResult's per-fold scores — NO leaky full-data
    # feature selection, NO cross-model MAX. Stays None if the search is unavailable.
    reselection_block = None
    hybrid_inputs = None
    # The top-2 rows, added 2026-09-18 as an extra beside everything already
    # reported. Filled inside the try below, where the SearchResult exists;
    # stays None when the search raises, so target_out never carries a half
    # block. Nothing here changes the per-representation table: that still
    # scores every representation get_representations returned against all 5
    # downstream models in MODELS_LIST.
    top2_block = None
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
        from experiments._shared import smoke_budgets
        _smoke_search = ({'budgets': smoke_budgets(), 'importance_methods': ['lightgbm_gain']}
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
            # return_fitted=True was dropped 2026-09-12: nothing in this file
            # read search.fitted_results, and the hybrid arm below fits its own
            # model inside each fold.
            internal_evaluators=_eval_list,
            condition_suffix=target,
            adaptive_pruning=0.25,
            **_smoke_search,
            **_pkw,
        )
        # Honest per-model held-out comparison (hybrid candidates vs baseline
        # reps), built straight from the search's per-fold scores. The reader
        # feeds this block to fold_reselection(): for each available model it
        # re-selects that model's own best hybrid recipe and best baseline rep
        # per held-out fold, so neither arm is chosen on the fold it's scored on.
        # This SUPERSEDES the same-fold argmax comparison in print_analysis for
        # the honest hybrid-vs-single verdict; the legacy per-rep table below is
        # retained unchanged for backward compatibility.
        if search is not None:
            reselection_block = build_fold_reselection_input(search)

        top3 = extract_top3_base_reps(search) if search is not None else None
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        # No 'kirby_hybrid' entry is added to `reps`. It used to be built by
        # refitting the search's winning recipe on every compound, so each
        # fold's own labels had already helped choose the feature indices it was
        # then scored on. Its MAE, RMSE, R2 and AC-MCC were measured on data
        # that built it. Removed 2026-09-04, split audit. The hybrid arm below
        # builds one hybrid per fold from that fold's training rows instead, so
        # its cliff numbers come from held-out predictions.
        #
        # What that arm needs: the same representations, the same augmentation
        # pool and the same search settings as the search above. Pruning is
        # resolved once from the whole compound count and reused per fold; a
        # fold is within a fifth of that count.
        hybrid_inputs = {
            'base': base if base else reps,
            'aug_dict': _smoke_augs,
            'search_kwargs': dict(internal_evaluators=_eval_list,
                                  **_smoke_search, **_pkw),
        }

        # ── Top-2 rows, read off the search above. 0 new model fits. ───────
        # Every call below reads rows the search already stored, so this adds
        # no fit to the ~13,500,000 the three targets already run
        # (EXPERIMENT_COMPUTE_AUDIT_2026_09_18.md:90 — a dated claim, not
        # re-measured here).
        #
        # Additive. `results` below still holds one entry per representation
        # and one model entry per downstream model, and print_analysis still
        # ranks every one of those rows. These keys sit on target_out beside
        # 'results', never inside it: print_analysis reads rep_data['models']
        # for every value of results, so a key there with no 'models' raises
        # KeyError and takes the whole cross-target table with it.
        #
        # require_full=False everywhere: a downstream model this search never
        # scored comes back as a short list carrying its own shortfall_reason
        # instead of raising. svm is the live case — its row ceiling falls to
        # 5,000 training molecules under the 'uncapped' augmentation strategy
        # (find_best_hybrid.py:2862) and d2 carries 6,333 compounds.
        if search is not None:
            _sel_ctx = f'opig_activity_cliff_qsar/{target}'
            top2_block = {}
            try:
                # 'rf' is the model this search was asked for
                # (downstream_model='rf' above) and the model the Top-3 line
                # printed a few lines up already ranks by, so the two rankings
                # are the same quantity ranked to different depths.
                top2_block['top2_base_reps'] = top_base_reps(
                    search, 'rf', 2, require_full=False, context=_sel_ctx)
            except Exception as _e:
                print(f"  top2_base_reps not added: {type(_e).__name__}: {_e}")
                top2_block['top2_base_reps_absent_reason'] = \
                    f"rf: {type(_e).__name__}: {_e}"
            try:
                # budget=None: the best two configurations across every budget
                # in the search, not within one budget in features. This file
                # searches all budgets at once (budget=None above), so a
                # per-budget cut here would name a pair the file never used.
                top2_block['top2_hybrids'] = top_hybrids(
                    search, 'rf', 2, require_full=False, context=_sel_ctx)
            except Exception as _e:
                print(f"  top2_hybrids not added: {type(_e).__name__}: {_e}")
                top2_block['top2_hybrids_absent_reason'] = \
                    f"rf: {type(_e).__name__}: {_e}"
            try:
                # 5 downstream models in MODELS_LIST * (2 representations +
                # 2 hybrids) = 20 rows. models=MODELS_LIST rather than the
                # default, so the models axis is reported at the width this
                # file scores at and not at _shared.py's default width.
                top2_block['top2_pairs_by_model'] = select_model_rep_pairs(
                    search, models=MODELS_LIST, n_base=2, n_hybrids=2,
                    require_full=False, context=_sel_ctx)
            except Exception as _e:
                print(f"  top2_pairs_by_model not added: "
                      f"{type(_e).__name__}: {_e}")
                top2_block['top2_pairs_by_model_absent_reason'] = \
                    f"{type(_e).__name__}: {_e}"

            # Two reference numbers for the rank-1 hybrid, per downstream
            # model, not one:
            #   delta_vs_source_mean — against the MEAN of the base
            #     representations the budget split actually gave features to,
            #     read off Blueprint.allocation_per_fold, so a representation
            #     given 0 features on every fold is out of that mean.
            #   delta_vs_best_rep — against the single best base
            #     representation for that same downstream model.
            # Positive means the hybrid is better, on either metric direction.
            #
            # hybrid= is passed rather than left to default. hybrid_deltas with
            # hybrid=None reads _picked['selected'][0]['row'] at
            # _shared.py:7432, and the entry top_hybrids builds
            # (_shared.py:7043-7066) carries no 'row' key — read in this
            # session, both ends. The rank-1 entry handed in below carries
            # 'score_mean' and 'blueprint', the only two keys hybrid_deltas
            # reads off it.
            #
            # One try per downstream model, so a model with no stored hybrid
            # does not cost the other four their entry. Keys are downstream
            # model names; this dict is new and nothing iterates it.
            #
            # The names to loop come off the search, not from MODELS_LIST.
            # MODELS_LIST is a fixed list written in this file; the search scores
            # whatever list it was given, which can be shorter -- a dataset's
            # skip policy drops svm, and svm drops itself above 10,000
            # molecules. A name the search never scored is named here and left
            # out; no other model's rows are used in its place.
            #
            # The models compared and the models left out are now SAVED as well
            # as printed (2026-09-28). Until now every reason below went to the
            # terminal only, so a saved block holding four models read the same
            # as a run configured for four.
            _deltas = {}
            _delta_models, _delta_dropped = resolve_comparison_models(
                search, MODELS_LIST)
            top2_block['top2_models_compared'] = list(_delta_models)
            top2_block['top2_models_dropped'] = {}
            for _name, _why in _delta_dropped:
                print(f"  hybrid_deltas for downstream model {_name} not "
                      f"added: {_why}")
                top2_block['top2_models_dropped'][_name] = _why
            for _m in _delta_models:
                try:
                    _top1 = top_hybrids(search, _m, 1, require_full=False,
                                        context=f'{_sel_ctx}/deltas/{_m}')
                    if not _top1['selected']:
                        print(f"  hybrid_deltas for downstream model {_m} not "
                              f"added: {_top1['shortfall_reason']}")
                        top2_block['top2_models_dropped'][_m] = (
                            'no hybrid delta: %s' % _top1['shortfall_reason'])
                        continue
                    _deltas[_m] = hybrid_deltas(
                        search, _m, hybrid=_top1['selected'][0],
                        context=f'{_sel_ctx}/deltas/{_m}')
                except Exception as _e:
                    print(f"  hybrid_deltas for downstream model {_m} not "
                          f"added: {type(_e).__name__}: {_e}")
                    top2_block['top2_models_dropped'][_m] = (
                        'no hybrid delta: %s: %s' % (type(_e).__name__, _e))
            if _deltas:
                top2_block['hybrid_deltas_by_model'] = _deltas
            if not top2_block:
                top2_block = None
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
            ac_res = evaluate_ac_performance(
                labels, predictions, mmp_pairs, data['ac_threshold'],
                held_out_fold=qsar_res.pop('held_out_fold', None))

            model_result = {'qsar': qsar_res}
            if ac_res is not None:
                model_result['ac'] = ac_res

            rep_results[model_name] = model_result

            mae = qsar_res['mae']['mean']
            _bp = (ac_res or {}).get('by_partition', {})
            m_int = _bp.get('M_inter', {}).get('mcc_fold_mean', float('nan'))
            m_test = _bp.get('M_test', {}).get('mcc_fold_mean', float('nan'))
            print(f"    {rep_name:25s} {model_name:8s}  "
                  f"MAE={mae:.4f}  MCC-inter={m_int:.4f}  "
                  f"MCC-test={m_test:.4f}  ({elapsed:.1f}s)")

        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    # Hybrid arm. One hybrid per fold, built on that fold's training rows and
    # predicting its held-out rows; the 5 held-out prediction vectors pool into
    # one value per compound, which is what evaluate_ac_performance scores. That
    # is the same path a representation takes, so the hybrid's MCC, sensitivity,
    # specificity and AUROC sit in the per-target table beside them.
    if hybrid_inputs is not None:
        print(f"\n  Per-fold hybrid × {len(models)} models × {n_folds}-fold CV...")
        try:
            hyb = evaluate_hybrid_per_fold(
                hybrid_inputs['base'], hybrid_inputs['aug_dict'],
                labels, groups, n_folds, models, target,
                hybrid_inputs['search_kwargs'])
        except Exception as e:
            print(f"  Per-fold hybrid arm failed: {type(e).__name__}: {e}")
            hyb = {}

        hybrid_results = {}
        for model_name, hres in hyb.items():
            predictions = hres.pop('predictions')
            ac_res = evaluate_ac_performance(
                labels, predictions, mmp_pairs, data['ac_threshold'],
                held_out_fold=hres.pop('held_out_fold', None))
            model_result = {'qsar': hres['qsar'],
                            'per_fold_blueprint': hres['per_fold_blueprint'],
                            'per_fold_n_features': hres['per_fold_n_features']}
            if ac_res is not None:
                model_result['ac'] = ac_res
            hybrid_results[model_name] = model_result

            mae = hres['qsar']['mae']['mean']
            _bp = (ac_res or {}).get('by_partition', {})
            m_int = _bp.get('M_inter', {}).get('mcc_fold_mean', float('nan'))
            m_test = _bp.get('M_test', {}).get('mcc_fold_mean', float('nan'))
            print(f"    {'kirby_hybrid':25s} {model_name:8s}  "
                  f"MAE={mae:.4f}  MCC-inter={m_int:.4f}  "
                  f"MCC-test={m_test:.4f}  "
                  f"({hres['qsar']['n_folds']}/{n_folds} folds)")

        # ⛔ WHICH DOWNSTREAM MODELS HAVE NO HYBRID ROW, AND WHY. Recorded
        # 2026-09-18. A model whose search scored no configuration was dropped
        # in silence, so the per-representation table above could carry svm
        # while the hybrid arm did not, and a reader comparing the two read the
        # gap as a loss. svm is the real case: its effective row ceiling falls
        # to 5,000 training molecules under the 'uncapped' augmentation
        # strategy (find_best_hybrid.py:2862) and d2 carries 6,333 compounds.
        _absent = sorted(m for m in models if m not in hybrid_results)
        if _absent:
            print(f"    ⛔ no hybrid row at all for: {', '.join(_absent)} "
                  f"— absent from the search, not beaten by it")

        if hybrid_results:
            # No single feature count: each fold and each model has its own
            # budget. The counts are stored per model as per_fold_n_features.
            results['kirby_hybrid'] = {
                'dim': None,
                'models': hybrid_results,
                'models_without_any_hybrid': _absent,
                'models_asked_for': sorted(models),
            }

    target_out = {
        'n_compounds': n,
        'n_scaffolds': n_groups,
        'n_mmps': len(mmp_pairs),
        'n_acs': data['n_activity_cliffs'],
        'target_name': data['target_name'],
        'results': results,
    }
    # Attach the per-fold reselection block under the exact contract key name so the
    # reader can load result[target]['fold_reselection'] and call fold_reselection().
    if reselection_block is not None:
        target_out['fold_reselection'] = reselection_block
    # Beside 'results', not inside it. main() copies every top-level key of
    # target_out except 'results' straight into the saved JSON, and
    # print_analysis reads only 'results', 'target_name', 'n_compounds',
    # 'n_acs' and 'fold_reselection' — so this key is saved and printed by
    # nothing, which is what an extra is.
    if top2_block is not None:
        target_out['top2_selection'] = top2_block
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

        # Three cliff MCCs per row. MCC-inter and MCC-test are Dablander's two
        # figures (one activity known / both unknown), each the mean over folds
        # as he reports it. MCC-pool scores every pair prediction against
        # prediction and matches neither of his figures; it is kept because
        # earlier result files carry only it.
        print(f"  {'Rep':25s} {'Model':8s} {'MAE':>8s} {'RMSE':>8s} "
              f"{'R2':>8s} {'MCC-int':>8s} {'MCC-test':>8s} {'MCC-pool':>8s}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        rows = []
        for rep_name, rep_data in results.items():
            for model_name, model_res in rep_data['models'].items():
                qsar = model_res['qsar']
                ac = model_res.get('ac', {}) or {}
                bp = ac.get('by_partition', {}) or {}
                rows.append((
                    rep_name, model_name,
                    qsar['mae']['mean'], qsar['rmse']['mean'],
                    qsar['r2']['mean'],
                    bp.get('M_inter', {}).get('mcc_fold_mean', float('nan')),
                    bp.get('M_test', {}).get('mcc_fold_mean', float('nan')),
                    ac.get('ac_mcc', float('nan')),
                ))

        # Sort by MAE (lower is better)
        rows.sort(key=lambda r: r[2])

        def _f(v):
            return f"{v:.4f}" if v is not None and not np.isnan(v) else "—"
        for rep_name, model_name, mae, rmse, r2, m_int, m_test, m_pool in rows:
            print(f"  {rep_name:25s} {model_name:8s} {mae:8.4f} {rmse:8.4f} "
                  f"{r2:8.4f} {_f(m_int):>8s} {_f(m_test):>8s} {_f(m_pool):>8s}")

    # Cross-target summary. The Hybrid MAE and Delta columns were removed on
    # 2026-09-04 (split audit): both came from a hybrid whose features were
    # refit on every compound's label, so the number was measured on rows that
    # chose it. The honest hybrid-vs-best-single table below replaces them; the
    # per-fold hybrid is a row of the per-target table above.
    #
    # Both numbers here are a best-of-N over (representation, model) rows, so
    # each is the lowest or the highest of N draws and is paired with nothing.
    # nMAE and nMCC are how many rows each was taken over. kirby_hybrid is not a
    # single representation, so it is out of the MAE minimum; its MAE is read
    # off the honest per-model table below.
    print(f"\n  Cross-target summary:")
    print(f"  {'Target':10s} {'Best rep':25s} {'MAE':>8s} {'nMAE':>6s} "
          f"{'MCC-test':>8s} {'nMCC':>6s}")
    print(f"  {'-'*10} {'-'*25} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")

    for target in ALL_TARGETS:
        if target not in all_results:
            continue

        tr = all_results[target]
        results = tr['results']

        best_indiv = None
        best_mae = float('inf')
        n_mae_rows = 0

        for rep_name, rep_data in results.items():
            if rep_name == 'kirby_hybrid':
                continue
            for model_name, model_res in rep_data['models'].items():
                mae = model_res['qsar']['mae']['mean']
                n_mae_rows += 1
                if mae < best_mae:
                    best_mae = mae
                    best_indiv = rep_name

        # Best MCC-test (both activities unknown, Dablander's harder figure)
        # for any method, the hybrid included. Higher is better; NaN ignored.
        _mccs = [
            ((m.get('ac') or {}).get('by_partition') or {}).get(
                'M_test', {}).get('mcc_fold_mean', float('nan'))
            for rd in results.values() for m in rd['models'].values()
        ]
        _mccs = [v for v in _mccs if not (isinstance(v, float) and np.isnan(v))]
        best_mcc = max(_mccs) if _mccs else float('nan')

        print(f"  {target:10s} {best_indiv or '—':25s} {best_mae:8.4f} "
              f"{n_mae_rows:6d} {best_mcc:8.4f} {len(_mccs):6d}")

    # Honest per-model, held-out QSAR-MAE verdict (selection-optimism corrected).
    # Each model's own hybrid recipe and own best baseline rep are re-selected per
    # held-out fold, so the delta is not inflated by same-fold argmax. Only shown
    # where the search populated the fold_reselection block (fresh runs / warmed
    # per-fold baseline cache); MAE is lower-is-better so +delta = hybrid better.
    have_reselection = any('fold_reselection' in all_results.get(t, {}) for t in ALL_TARGETS)
    if have_reselection:
        print(f"\n  Honest per-model held-out (LOFO) QSAR-MAE — hybrid vs best single rep:")
        print(f"  {'Target':10s} {'Model':8s} {'Hybrid':>9s} {'BestSingle':>11s} "
              f"{'Delta':>9s} {'Folds':>6s}")
        print(f"  Delta is MAE of the best single representation minus MAE of "
              f"the hybrid, averaged over paired folds; positive means the "
              f"hybrid has the lower error.")
        print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*11} {'-'*9} {'-'*6}")
        for target in ALL_TARGETS:
            blk = all_results.get(target, {}).get('fold_reselection')
            if not blk:
                continue
            res = fold_reselection(blk['candidates_by_model'],
                                 blk['baselines_by_model'],
                                 blk['lower_is_better'],
                                 exclude_folds=hr_exclude_folds(blk))
            for model_name in sorted(res):
                r = res[model_name]
                h_v = r['hybrid_reselected']
                b_v = r['best_single_reselected']
                # The paired mean of per-fold differences, which is what
                # fold_reselection says to report; r['delta'] is a difference
                # of two means and drifts when either side drops a fold.
                d_v = r.get('delta_mean_of_folds', r['delta'])
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

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard targets finished by an earlier attempt '
             '(<output>.partial.jsonl) and recompute every one.')

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

    output_path = (args.output or
                   'tests/results/comparisons/opig_activity_cliff_qsar.json')

    # Each target's result is written to <output>.partial.jsonl as soon as it
    # finishes and read back on the next attempt (2026-09-25;
    # experiments/_checkpoint.py). Before, a failure or a kill lost every
    # target, because the one JSON below was written after the last. A target
    # that raised or returned {'error'} is not stored, is retried next time,
    # and makes the script exit 1 after the JSON is written.
    from experiments._checkpoint import (
        script_checkpoint, is_failed_item, exit_if_failed)
    _ck = script_checkpoint('opig_activity_cliff_qsar', output_path, args,
                            exclude=('targets', 'output'),
                            rep_preset=rep_preset)
    _failed = []

    all_results = {}
    for target in targets:
        try:
            if _ck.done(target=target):
                target_results = _ck.get(target=target)
                print(f"  {target}: finished in an earlier attempt, read back")
            else:
                target_results = evaluate_target(
                    target, rep_preset, args.n_folds,
                    pruning_mode=args.pruning_mode)
                _ck.put(target_results, target=target)
            if is_failed_item(target_results):
                _failed.append(target)
            if target_results is not None:
                all_results[target] = target_results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            _failed.append(target)

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
    exit_if_failed(_failed, 'target')


if __name__ == '__main__':
    main()
