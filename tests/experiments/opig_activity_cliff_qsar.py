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
    # Smoke test (1 target, 2 of the 10 representations, 2 outer folds)
    python tests/experiments/opig_activity_cliff_qsar.py \\
        --targets d2 --n-folds 2 --reps ecfp4,pdv

    # Full evaluation (outer fold count from get_config's n_splits)
    python tests/experiments/opig_activity_cliff_qsar.py

Aligned 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.3 answers 1, 5, 6, 17,
18 and 2.4): representations are DATASET_REPS['chembl_qsar']['general'] (10);
models are chembl_qsar's two decided ones (lightgbm, dnn); per model per outer
fold the 2 best base representations and the 2 best general hybrids of a search
on that fold's training rows, no distance hybrid; pruning off; 0/1 features
unscaled; every finished fold written to <output>.partial.jsonl.

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
from experiments._shared import convert_numpy, get_model_factories
# 0/1 features pass through unscaled, every other feature is standardised on
# the training rows (Adelaide, 2026-10-05, handoff 2.3 answer 17). Replaces
# StandardScaler here, which scaled every feature of a hybrid and skipped a
# representation only when its WHOLE matrix was 0/1.
from kirby.scaling import FeatureScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

# The dataset name every search and every DECISIONS.md lookup uses. The three
# targets below are three loads of this one dataset (dataset_policy.py:342).
DATASET = 'chembl_qsar'

ALL_TARGETS = ['d2', 'fxa', 'mpro']

# Seed of the outer scaffold folds. nested_cv_outer_folds and the per-fold
# search cache namespace both take it, so the folds scored here are the folds
# the searches were run on.
SEED = 42

# The hybrids each model gets per outer fold: the 2 best of that fold's general
# search. The 'distance1' slot (best hybrid of a distance search) was dropped
# 2026-10-05: distance hybrids only where the experiment measures something
# with neighbours or distances (handoff 2.3 answer 1), and this one fits
# lightgbm and dnn on features.
SLOTS = ('general1', 'general2')

# Part of the checkpoint identity. Bumped 2026-10-05 so no partial file from
# the 7-representation, 5-model, whole-target run is read back: a stored unit
# is now one fold of one target, not one target.
CHECKPOINT_VERSION = ('2026-10-05: one unit per fold; DATASET_REPS 10; '
                      'decided models; 2 base + 2 general hybrids, no '
                      'distance; pruning off; FeatureScaler')

# Until 2026-10-05 the default was this file's own seven (ecfp4, maccs, pdv,
# sns, mordred, mol2vec, avalon). sns is a per-fold factory representation that
# get_representations never builds, so the dirty run had six. The ten below are
# all plain builders in find_best_hybrid_molecular.get_representations (none
# is in FACTORY_REPS); one that cannot be built (grover is not installed
# locally) is dropped with a warning and listed under 'reps_missing' in the
# result file.


def approved_reps():
    """chembl_qsar's pruned representations, DATASET_REPS[...]['general'] (10)."""
    from experiments.complexity.complexity_helpers import DATASET_REPS
    return list(DATASET_REPS[DATASET]['general'])


def augmentation_reps():
    """The added-feature sources, only where the decided grid keeps a condition.

    decided_aug_conditions('chembl_qsar', 'general') is ['none'] (2026-10-05),
    so this is empty and no augmentation matrix is built or handed to a search.
    Until 2026-10-05 maccs and pharmacophore3d were always handed to every fold
    search as augmentations and taken out of the base representations.
    """
    from experiments.decided_grid import decided_aug_conditions
    from experiments.complexity.complexity_helpers import DATASET_REPS
    kept = [c for c in decided_aug_conditions(DATASET, 'general') if c != 'none']
    if not kept:
        return []
    return list(DATASET_REPS[DATASET].get('augmentation', []))


def decided_model_names():
    """The dataset's two downstream models (DECISIONS.md Step 1).

    Every representation and every hybrid is scored with these and no others.
    chembl_qsar is decided as lightgbm + dnn, settings copied from herg_ki
    (handoff 2.3.1, 2026-10-05). Until 2026-10-04 this file scored five models
    (DOWNSTREAM_MODELS) and the distance searches also scored knn.
    """
    from experiments.decided_grid import decided_models
    return [m for m in decided_models(DATASET, 'general') if m != 'knn']


def default_n_folds(distance_metric=None):
    """The outer fold count when --n-folds is not given: get_config's n_splits.

    One setting for every experiment (handoff 2.4 step 14, 2026-10-05):
    DEFAULT_N_FOLDS, or KIRBY_LOCAL_FOLDS when set. Was a literal 5.
    distance_metric is passed through because get_config sets it process-wide.
    """
    from experiments._shared import get_config
    return int(get_config(argparse.Namespace(
        mode='full', distance_metric=distance_metric))['n_splits'])


def metric_triplet(labels):
    """(metric name, metric function, lower_is_better) for chembl_qsar.

    From the dataset registry ('mae', lower is better), so the hybrid rebuild
    and the result file use the metric the searches ranked on.
    """
    from experiments._shared import get_domain_metric
    return get_domain_metric({'dataset_name': DATASET,
                              'task_type': 'regression',
                              'domain': 'molecular', 'labels': labels})


# What each reported score's direction is. Written into the result file.
METRIC_DIRECTIONS = {'mae': 'lower is better', 'rmse': 'lower is better',
                     'r2': 'higher is better', 'spearman': 'higher is better',
                     'ac_mcc': 'higher is better'}


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

def _scores(y, p):
    """mae, rmse, r2 and spearman of one prediction vector against its labels."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    out = {'mae': float(mean_absolute_error(y, p)),
           'rmse': float(np.sqrt(np.mean((y - p) ** 2))),
           'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
           'spearman': float('nan')}
    if len(y) > 2:
        rho, _ = spearmanr(y, p)
        out['spearman'] = float(rho) if not np.isnan(rho) else 0.0
    return out


def _fit_scaled(model, X_tr, y_tr, X_va):
    """Fit a clone of `model` on the training rows, predict both sides.

    FeatureScaler is fitted on the training rows only: 0/1 features pass
    through, the rest are standardised (handoff 2.3 answer 17, 2026-10-05).
    Returns (training-row predictions, held-out predictions).
    """
    scaler = FeatureScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    m = clone(model)
    m.fit(X_tr, y_tr)
    return m.predict(X_tr), m.predict(X_va)


def _mean_spread(values):
    """Mean and spread of one score over the outer folds.

    One fold has no spread: 'std' is None and 'spread' says so (handoff 2.4
    step 14, 2026-10-05), rather than a 0.0 that reads as a measured spread.
    """
    v = [float(x) for x in values
         if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return {'mean': float('nan'), 'std': None, 'n_folds': 0}
    if len(v) == 1:
        return {'mean': v[0], 'std': None, 'n_folds': 1,
                'spread': 'one fold, no spread'}
    return {'mean': float(np.mean(v)), 'std': float(np.std(v)),
            'n_folds': len(v)}


def _summarise_folds(per_fold):
    """Held-out and training-row scores over folds, from per-fold records.

    per_fold: [{'fold', 'train': _scores, 'held_out': _scores, ...}].
    The held-out block keeps the 'qsar' layout print_analysis reads
    (qsar[metric]['mean']).
    """
    keys = ('mae', 'rmse', 'r2', 'spearman')
    qsar = {k: _mean_spread([f['held_out'][k] for f in per_fold]) for k in keys}
    qsar['n_folds'] = len(per_fold)
    train = {k: _mean_spread([f['train'][k] for f in per_fold]) for k in keys}
    return qsar, train


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
                            held_out_fold=None, single_split=False):
    """Evaluate activity cliff prediction from QSAR predictions.

    Matches Dablander et al. 2023:
      - AC = |ΔpActivity| >= 2.0 (label 1)
      - Non-AC = |ΔpActivity| < 1.0 (label 0)
      - Half-cliffs (1.0 <= |Δ| < 2.0) are EXCLUDED from evaluation
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

    single_split (added 2026-10-05): with one outer fold (a single grouped
    held-out split) most compounds are never held out, so their fold index
    stays -1 and they were in training. A pair with one held-out compound and
    one such compound is then M_inter (the training compound's measured
    activity is known). Under k folds a -1 means a fold produced no prediction,
    and the pair is left out as before.

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

    # ---- Dablander's pair partition ------------------------------------
    # Same cliff definition, same threshold, same soft score; the only
    # difference is WHICH pairs go in and, for M_inter, that the known
    # compound contributes its MEASURED activity rather than a prediction.
    if held_out_fold is not None:
        held_out_fold = np.asarray(held_out_fold)
        parts = {'M_inter': ([], []), 'M_test': ([], [])}
        for i, j, _ in mmp_pairs:
            fi, fj = held_out_fold[i], held_out_fold[j]
            if fi < 0 and fj < 0:
                continue
            if (fi < 0 or fj < 0) and not single_split:
                continue
            true_delta = abs(labels[i] - labels[j])
            if 1.0 <= true_delta < ac_threshold:
                continue
            is_ac = int(true_delta >= ac_threshold)
            if fi < 0 or fj < 0:
                # One split: the compound never held out was in training, so
                # its measured activity was known; the other was predicted.
                held, known = (i, j) if fj < 0 else (j, i)
                if np.isnan(predictions[held]):
                    continue
                pred_delta = abs(labels[known] - predictions[held])
                key = 'M_inter'
            elif fi == fj:
                # both held out together -> both activities unknown
                if np.isnan(predictions[i]) or np.isnan(predictions[j]):
                    continue
                pred_delta = abs(predictions[i] - predictions[j])
                key = 'M_test'
            else:
                # held out in different folds: in the fold that scored i, j was
                # in training, so j's measured activity was known. Score the
                # pair the way Dablander does, measured against predicted, and
                # take the compound whose prediction exists.
                if np.isnan(predictions[i]) and np.isnan(predictions[j]):
                    continue
                if np.isnan(predictions[i]):
                    pred_delta = abs(labels[i] - predictions[j])
                elif np.isnan(predictions[j]):
                    pred_delta = abs(labels[j] - predictions[i])
                else:
                    # both predictions exist (each from its own fold); average
                    # the two one-known-one-predicted distances so neither
                    # compound is privileged.
                    pred_delta = 0.5 * (abs(labels[i] - predictions[j])
                                        + abs(labels[j] - predictions[i]))
                key = 'M_inter'
            parts[key][0].append(is_ac)
            parts[key][1].append(2.0 * np.arctan(pred_delta / AC_SCALE) / np.pi)
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
            by_part[key] = m
        result['by_partition'] = by_part
    return result


def _hybrid_summary(entry):
    """The JSON-safe record of one chosen hybrid (no Blueprint object).

    Allocation, importance, budget in features, internal evaluator,
    representation count and augmentation condition, all read off the per-fold
    search's own row, plus its score there (mean over that search's inner
    folds of the outer fold's training rows).
    """
    return {
        'aug_condition': entry.get('aug_condition'),
        'allocation': entry.get('allocation_method'),
        'importance': entry.get('importance_method'),
        'budget': entry.get('budget'),
        'evaluator': entry.get('internal_evaluator'),
        'rep_count': entry.get('rep_count'),
        'n_features': entry.get('n_features'),
        'inner_score_mean': entry.get('score_mean'),
        'inner_fold_scores': entry.get('fold_scores'),
        'selection_scope': 'fold_train_rows',
    }


def _missing(target, fold_i, model_name, arm, error):
    """One arm a fold could not fill, with dataset, target, fold, model, why.

    A structured record since 2026-10-05 (R2/R6): the arm is skipped, never
    filled with a fallback recipe, and the reason is saved in the result file.
    """
    return {'dataset': DATASET, 'target': target, 'fold': int(fold_i),
            'model': model_name, 'arm': arm, 'error': str(error)}


def evaluate_fold(target, fold_i, train_idx, val_idx, *, reps, aug_dict,
                  labels, groups, n_outer, models, metric):
    """Everything this experiment measures on ONE outer scaffold fold.

    1. Every representation x every decided model: fitted on the fold's
       training rows, scored on its training rows and its held-out rows.
    2. One general search on the fold's training rows (fold_standard_selection
       -> search_on_fold, chembl_qsar's decided grid, condition_suffix=<target>
       so d2, fxa and mpro keep separate stores). Per model it gives the 2 best
       base representations and the 2 best hybrids with distinct selected
       features. No whole-dataset search is read anywhere in this file: its
       feature indices were chosen with the held-out labels in hand.
    3. Each chosen hybrid rebuilt on the fold's training rows from the fold
       search's own blueprint (its own representation scores), fitted, and
       scored on training and held-out rows.

    No pruning arguments are passed (handoff 2.3 answer 6, 2026-10-05); until
    then every fold search ran adaptive_pruning=0.25 plus a pruning schedule.

    Returns a JSON-safe payload, stored as one checkpoint unit. Held-out
    predictions are kept per arm (aligned to val_idx) so the activity-cliff
    metrics can be computed over every compound once all folds are in.
    """
    from experiments._shared import (build_fold_hybrid,
                                     fold_hybrid_scoring_kwargs,
                                     fold_searches,
                                     fold_standard_selection)
    from experiments.decided_grid import UndecidedSearch
    from kirby.result_cache import StaleResultDB

    metric_name, metric_fn, lower_is_better = metric
    y_tr, y_va = labels[train_idx], labels[val_idx]
    g_tr = groups[train_idx] if groups is not None else None
    tr = {k: v[train_idx] for k, v in reps.items()}
    va = {k: v[val_idx] for k, v in reps.items()}
    aug_tr = {k: v[train_idx] for k, v in aug_dict.items()} if aug_dict else None
    aug_va = {k: v[val_idx] for k, v in aug_dict.items()} if aug_dict else None

    payload = {'fold': int(fold_i), 'n_train': int(len(train_idx)),
               'n_held_out': int(len(val_idx)),
               'selection_scope': 'fold_train_rows',
               'reps': {}, 'hybrids': {s: {} for s in SLOTS},
               'selection': {}, 'rule_missing': [], 'search_failed': False}

    # ---- 1. every representation, every model ---------------------------
    for rep_name in sorted(reps):
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                p_tr, p_va = _fit_scaled(factory(), tr[rep_name], y_tr,
                                         va[rep_name])
            except Exception as e:
                # Logged and saved (OPIG-18): a missing fold changes the
                # per-fold means and the pooled predictions.
                print(f"      fold {fold_i} {rep_name} {model_name} fit "
                      f"failed: {type(e).__name__}: {e}", flush=True)
                payload['rule_missing'].append(_missing(
                    target, fold_i, model_name, rep_name,
                    f'fit raised {type(e).__name__}: {e}'))
                continue
            rec = {'fold': int(fold_i), 'train': _scores(y_tr, p_tr),
                   'held_out': _scores(y_va, p_va),
                   'held_out_pred': np.asarray(p_va, dtype=np.float64),
                   'seconds': round(time.time() - t0, 2)}
            payload['reps'].setdefault(rep_name, {})[model_name] = rec
            print(f"    fold {fold_i} {rep_name:20s} {model_name:8s} "
                  f"{metric_name}={rec['held_out'][metric_name]:.4f} "
                  f"(training rows {rec['train'][metric_name]:.4f}, "
                  f"{rec['seconds']:.1f}s)", flush=True)

    # ---- 2. the fold's general search -----------------------------------
    sel_kw = dict(train_idx=train_idx, fold_i=fold_i, n_outer_folds=n_outer,
                  dataset_name=DATASET, augmentations=aug_dict,
                  task_type='regression', metric=metric_name, seed=SEED,
                  # The target keeps d2, fxa and mpro, all filed as
                  # chembl_qsar, in separate stored searches; the fold is
                  # already in the cache namespace.
                  condition_suffix=target)
    search_error = None
    try:
        sel = fold_standard_selection(reps, labels, groups,
                                      models=list(models), **sel_kw)
    except (UndecidedSearch, StaleResultDB):
        # A search DECISIONS.md does not cover is not run, and a stale result
        # store is not read around: the target stops here either way.
        raise
    except Exception as e:
        print(f"    fold {fold_i} general search failed: "
              f"{type(e).__name__}: {e}", flush=True)
        search_error = f'search raised {type(e).__name__}: {e}'
        sel = {m: None for m in models}
    if search_error is None and all(v is None for v in sel.values()):
        # fold_standard_selection prints a failed search's error and returns
        # None per model. fold_searches is memoised on the same arguments, so
        # this reads the failure back without running the search again.
        try:
            srch = fold_searches(reps, labels, groups, tier='general', **sel_kw)
            fails = getattr(srch, 'failures', None) or []
            search_error = ('; '.join(str(f.get('error') or f.get('error_type'))
                                      for f in fails)
                            or 'the search returned no result and no error')
        except Exception as e:
            search_error = (f'search returned no result; reading its error '
                            f'raised {type(e).__name__}: {e}')
    payload['search_failed'] = search_error is not None

    # ---- 3. per model: 2 base representations, 2 hybrids ----------------
    for model_name, factory in models.items():
        s = sel.get(model_name)
        if s is None:
            payload['selection'][model_name] = None
            payload['rule_missing'].append(_missing(
                target, fold_i, model_name, 'general search',
                search_error or 'no selection returned for this model'))
            continue
        base_detail = s.get('base_detail') or {}
        top2_base = list(s['base'])
        payload['selection'][model_name] = {
            'top2_base_reps': top2_base,
            # Their scores in the fold search (mean over its inner folds of
            # this outer fold's training rows).
            'top2_base_detail': [
                {'rep': e.get('rep'), 'score_mean': e.get('score_mean')}
                for e in base_detail.get('selected', [])],
            'base_shortfall_reason': base_detail.get('shortfall_reason'),
            'hybrids': {f'general{i}': _hybrid_summary(e)
                        for i, e in enumerate(s['hybrids'], start=1)},
            'n_hybrids_skipped_identical_features':
                len(s.get('hybrids_skipped_identical_features') or []),
        }
        for slot_i, slot in enumerate(SLOTS):
            entry = s['hybrids'][slot_i] if slot_i < len(s['hybrids']) else None
            if entry is None:
                # A model with no recipe on a fold is an absence, not a loss
                # (said out loud since 2026-09-18).
                payload['rule_missing'].append(_missing(
                    target, fold_i, model_name, slot,
                    'the fold search has no configuration for this slot'))
                continue
            bp = entry['blueprint']
            _use_aug = entry.get('aug_condition') not in (None, 'none')
            try:
                # The blueprint is this fold's own search's, so a
                # representation count is resolved with this fold's
                # allocation scores (blueprint.rep_scores), never the whole
                # dataset's (dirty-run bug 1).
                X_tr_h, X_va_h, _fi = build_fold_hybrid(
                    tr, va, y_tr, bp,
                    augmentations_train=aug_tr if _use_aug else None,
                    augmentations_val=aug_va if _use_aug else None,
                    groups=g_tr,
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=factory(), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                p_tr, p_va = _fit_scaled(factory(), X_tr_h, y_tr, X_va_h)
            except Exception as e:
                print(f"    fold {fold_i} {model_name} {slot} hybrid "
                      f"failed: {type(e).__name__}: {e}", flush=True)
                payload['rule_missing'].append(_missing(
                    target, fold_i, model_name, slot,
                    f'build or fit raised {type(e).__name__}: {e}'))
                continue
            rec = {'train': _scores(y_tr, p_tr),
                   'held_out': _scores(y_va, p_va),
                   'held_out_pred': np.asarray(p_va, dtype=np.float64),
                   'n_features': int(X_tr_h.shape[1]),
                   'blueprint': dict(_hybrid_summary(entry), fold=int(fold_i),
                                     slot=slot,
                                     # The model's top 2 single representations
                                     # on this fold's training rows.
                                     top2_base_reps=top2_base)}
            payload['hybrids'][slot][model_name] = rec
            print(f"    fold {fold_i} {slot:20s} {model_name:8s} "
                  f"{metric_name}={rec['held_out'][metric_name]:.4f} "
                  f"(training rows {rec['train'][metric_name]:.4f}, "
                  f"{rec['n_features']} features)", flush=True)
    return payload


def _pool_predictions(fold_payloads, splits, n, arm_of):
    """One held-out prediction per compound, and the fold it was held out in.

    arm_of(payload) -> the arm's record in that fold, or None. nan / -1 where
    no fold produced a prediction for the compound.
    """
    preds = np.full(n, np.nan)
    held_out = np.full(n, -1, dtype=int)
    per_fold = []
    for payload, (_tr, val_idx) in zip(fold_payloads, splits):
        rec = arm_of(payload)
        if rec is None:
            continue
        preds[val_idx] = rec['held_out_pred']
        held_out[val_idx] = payload['fold']
        per_fold.append(rec)
    return preds, held_out, per_fold


def evaluate_target(target, rep_names, n_folds, checkpoint):
    """Evaluate every representation and the per-fold hybrids on one target.

    Each outer fold is one checkpoint unit (target, fold), written as soon as
    the fold finishes (dirty-run bug 12, 2026-10-05); until then a target was
    stored only after all its folds. A fold whose search failed is not stored,
    so the next attempt retries it; its failure is in 'rule_missing'.

    Returns dict with QSAR metrics + AC metrics per representation x model and
    per hybrid slot x model.
    """
    from kirby.datasets.chembl_qsar import load_chembl_qsar
    from kirby.find_best_hybrid import nested_cv_outer_folds
    from experiments._shared import as_rep_dtype

    print(f"\n{'=' * 60}")
    print(f"Target: {target}")
    print(f"{'=' * 60}")

    # The two decided downstream models (lightgbm, dnn; handoff 2.3.1).
    model_names = decided_model_names()

    # Test-only row cap (smoke tests). No-op unless KIRBY_SMOKE_N is set.
    _smoke_n = os.environ.get('KIRBY_SMOKE_N', '').strip()
    _max = int(_smoke_n) if _smoke_n.isdigit() and int(_smoke_n) > 0 else None
    data = load_chembl_qsar(target, max_samples=_max)
    smiles = data['smiles']
    labels = np.asarray(data['labels'], dtype=np.float64)
    mmp_pairs = data['mmp_pairs']
    n = len(smiles)

    # Scaffold groups
    groups, n_groups = assign_scaffold_groups(smiles)
    print(f"  {n} compounds, {n_groups} scaffolds, "
          f"{data['n_activity_cliffs']} ACs (threshold={data['ac_threshold']})")

    # Representations, float32 on load (handoff 2.3 answer 18). Until
    # 2026-10-05 this file converted every matrix to float64 itself.
    loaded = get_representations(smiles, ','.join(rep_names))
    reps = {}
    for k, v in loaded.items():
        X = as_rep_dtype(v)
        if X.ndim == 2 and X.shape[0] == n:
            reps[k] = X
    reps_missing = [r for r in rep_names if r not in reps]
    print(f"  {len(reps)} of {len(rep_names)} representations built"
          + (f"; missing: {', '.join(reps_missing)}" if reps_missing else ''))
    if not reps:
        print("  No reps available, skipping target")
        return None

    # Added-feature sources, only where the decided grid keeps an
    # augmentation condition (empty for chembl_qsar, 2026-10-05).
    aug_names = augmentation_reps()
    aug_dict = None
    if aug_names and not os.environ.get('KIRBY_SMOKE_N', '').strip():
        _aug = get_representations(smiles, ','.join(aug_names))
        aug_dict = {k: as_rep_dtype(v) for k, v in _aug.items()} or None

    metric = metric_triplet(labels)
    metric_name, _metric_fn, lower_is_better = metric

    # The outer scaffold folds, from the same function the per-fold searches'
    # cache namespace is tied to. One fold = a single grouped held-out split.
    splits = nested_cv_outer_folds(reps, labels, groups, n_folds, SEED,
                                   'regression')
    if len(splits) != n_folds:
        raise ValueError(
            f"{target}: asked for {n_folds} scaffold folds, the splitter gave "
            f"{len(splits)} ({n_groups} scaffolds)")
    n_outer = len(splits)
    single_split = (n_outer == 1)

    models = get_model_factories(model_names, 'regression', n_samples=n)
    print(f"\n  {len(reps)} reps x {len(models)} models x {n_outer} outer "
          f"fold(s), plus {len(SLOTS)} per-fold hybrids per model"
          + (" (one held-out split, no spread over folds)"
             if single_split else ''))

    fold_payloads = []
    n_failed_search = 0
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        if checkpoint.done(target=target, fold=fold_i):
            payload = checkpoint.get(target=target, fold=fold_i)
            print(f"  fold {fold_i}: finished in an earlier attempt, read back")
        else:
            payload = evaluate_fold(
                target, fold_i, train_idx, val_idx, reps=reps,
                aug_dict=aug_dict, labels=labels, groups=groups,
                n_outer=n_outer, models=models, metric=metric)
            if payload['search_failed']:
                print(f"  fold {fold_i}: search failed; not stored, retried "
                      f"on the next attempt", flush=True)
            else:
                checkpoint.put(payload, target=target, fold=fold_i)
        if payload['search_failed']:
            n_failed_search += 1
        fold_payloads.append(payload)

    # ---- representations: pooled out-of-fold predictions -> AC metrics --
    results = {}
    for rep_name in sorted(reps):
        rep_results = {}
        for model_name in models:
            preds, held_out, per_fold = _pool_predictions(
                fold_payloads, splits, n,
                lambda p: p['reps'].get(rep_name, {}).get(model_name))
            if not per_fold:
                continue
            qsar, train = _summarise_folds(per_fold)
            ac_res = evaluate_ac_performance(
                labels, preds, mmp_pairs, data['ac_threshold'],
                held_out_fold=held_out, single_split=single_split)
            model_result = {
                'qsar': qsar, 'train_rows': train,
                'per_fold': [{'fold': r['fold'], 'train': r['train'],
                              'held_out': r['held_out']} for r in per_fold]}
            if ac_res is not None:
                model_result['ac'] = ac_res
            rep_results[model_name] = model_result
            mcc = ac_res['ac_mcc'] if ac_res else float('nan')
            print(f"    {rep_name:25s} {model_name:8s}  "
                  f"MAE={qsar['mae']['mean']:.4f}  AC-MCC={mcc:.4f}")
        if rep_results:
            results[rep_name] = {'dim': int(reps[rep_name].shape[1]),
                                 'models': rep_results}

    # ---- hybrids: one row per slot -------------------------------------
    # general1 keeps the 'kirby_hybrid' name readers already use. The
    # 'kirby_hybrid__distance1' row is gone (2026-10-05, answer 1).
    _row_name = {'general1': 'kirby_hybrid',
                 'general2': 'kirby_hybrid__general2'}
    for slot in SLOTS:
        hybrid_results = {}
        for model_name in models:
            preds, held_out, per_fold = _pool_predictions(
                fold_payloads, splits, n,
                lambda p: p['hybrids'].get(slot, {}).get(model_name))
            folds_without = [p['fold'] for p in fold_payloads
                             if p['hybrids'].get(slot, {}).get(model_name)
                             is None]
            if not per_fold:
                continue
            qsar, train = _summarise_folds(per_fold)
            ac_res = evaluate_ac_performance(
                labels, preds, mmp_pairs, data['ac_threshold'],
                held_out_fold=held_out, single_split=single_split)
            model_result = {
                'qsar': qsar, 'train_rows': train,
                'per_fold_blueprint': [r['blueprint'] for r in per_fold],
                'per_fold_n_features': [r['n_features'] for r in per_fold],
                'per_fold': [{'fold': r['blueprint']['fold'],
                              'train': r['train'], 'held_out': r['held_out'],
                              'n_features': r['n_features']}
                             for r in per_fold],
                # Folds where this slot had no hybrid for this model. A mean
                # over fewer folds than the representations got is not the
                # same number, so it travels with the score.
                'folds_without_a_recipe': folds_without}
            if ac_res is not None:
                model_result['ac'] = ac_res
            hybrid_results[model_name] = model_result
            mcc = ac_res['ac_mcc'] if ac_res else float('nan')
            print(f"    {_row_name[slot]:25s} {model_name:8s}  "
                  f"MAE={qsar['mae']['mean']:.4f}  AC-MCC={mcc:.4f}  "
                  f"({len(per_fold)}/{n_outer} folds)")

        # ⛔ WHICH DOWNSTREAM MODELS HAVE NO HYBRID ROW, AND WHY. Recorded
        # 2026-09-18. A model whose search scored no configuration was dropped
        # in silence, so a reader read the gap as a loss.
        _absent = sorted(m for m in models if m not in hybrid_results)
        if _absent:
            print(f"    ⛔ no {slot} hybrid row at all for: "
                  f"{', '.join(_absent)} — absent from the search, not "
                  f"beaten by it")
        if hybrid_results:
            # No single feature count: each fold and each model has its own
            # budget. The counts are stored per model as per_fold_n_features.
            results[_row_name[slot]] = {
                'dim': None,
                'slot': slot,
                'lineage': 'general',
                'models': hybrid_results,
                'models_without_any_hybrid': _absent,
                'models_asked_for': sorted(models),
            }

    rule_missing = [m for p in fold_payloads for m in p['rule_missing']]
    target_out = {
        'dataset_name': DATASET,
        'condition_suffix': target,
        'n_compounds': n,
        'n_scaffolds': n_groups,
        'n_mmps': len(mmp_pairs),
        'n_acs': data['n_activity_cliffs'],
        'target_name': data['target_name'],
        'models': list(model_names),
        'reps_requested': list(rep_names),
        'reps_built': sorted(reps),
        # Requested but not built (a builder failed or is not installed).
        'reps_missing': reps_missing,
        'augmentation_reps': sorted(aug_dict) if aug_dict else [],
        'n_outer_folds': n_outer,
        'single_split': single_split,
        'seed': SEED,
        # The metric the searches ranked on and its direction; the other
        # reported scores' directions beside it.
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'metric_directions': METRIC_DIRECTIONS,
        # Every hybrid and every top-2 base representation below was chosen by
        # a search on the held-out fold's training rows only.
        'selection_scope': 'fold_train_rows',
        'results': results,
        # Per fold and model: the 2 best base representations and the chosen
        # hybrids (allocation, importance, budget, evaluator, representation
        # count, augmentation condition).
        'per_fold_selection': [
            {'fold': p['fold'], 'n_train': p['n_train'],
             'n_held_out': p['n_held_out'],
             'selection_scope': p['selection_scope'],
             'models': p['selection']} for p in fold_payloads],
        # Every arm a fold could not fill, and why. Never silently dropped.
        'rule_missing': rule_missing,
        'n_folds_with_failed_search': n_failed_search,
    }
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
    # chose it. The per-fold hybrid rows of the per-target table above replace
    # them. The whole-dataset 'honest per-model' table that followed was
    # removed 2026-10-04: it re-picked per fold from a search that had seen
    # every row.
    #
    # Both numbers here are a best-of-N over (representation, model) rows, so
    # each is the lowest or the highest of N draws and is paired with nothing.
    # nMAE and nMCC are how many rows each was taken over. kirby_hybrid is not a
    # single representation, so it is out of the MAE minimum; its MAE is read
    # off its own rows in the per-target table above.
    print(f"\n  Cross-target summary:")
    print(f"  {'Target':10s} {'Best rep':25s} {'MAE':>8s} {'nMAE':>6s} "
          f"{'AC-MCC':>8s} {'nMCC':>6s}")
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
            if rep_name.startswith('kirby_hybrid'):
                continue
            for model_name, model_res in rep_data['models'].items():
                mae = model_res['qsar']['mae']['mean']
                n_mae_rows += 1
                if mae < best_mae:
                    best_mae = mae
                    best_indiv = rep_name

        # Best AC-MCC for any method, the hybrid included (MCC range [-1, 1];
        # ignore NaN entries)
        _mccs = [
            m.get('ac', {}).get('ac_mcc', float('nan'))
            for rd in results.values() for m in rd['models'].values()
        ]
        _mccs = [v for v in _mccs if not (isinstance(v, float) and np.isnan(v))]
        best_mcc = max(_mccs) if _mccs else float('nan')

        print(f"  {target:10s} {best_indiv or '—':25s} {best_mae:8.4f} "
              f"{n_mae_rows:6d} {best_mcc:8.4f} {len(_mccs):6d}")


# =============================================================================
# MAIN
# =============================================================================

def _write_results(all_results, output_path):
    """Write the JSON of every target finished so far.

    Called after each target and once at the end, so a kill loses at most the
    target in progress; its finished folds are in <output>.partial.jsonl.
    """
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
        json.dump(_nan_to_none(convert_numpy(all_results)), f, indent=2)
    return p


def main():
    parser = argparse.ArgumentParser(
        description='Activity cliff QSAR experiment (Dablander 2023). '
                    'Tests molecular reps on 3 ChEMBL targets.')
    parser.add_argument(
        '--targets', type=str, default=None,
        help='Comma-separated target names (default: d2,fxa,mpro)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help="Comma-separated subset of DATASET_REPS['chembl_qsar']['general'] "
             "(default: all 10). A name outside the 10 is refused.")
    parser.add_argument(
        '--n-folds', type=int, default=None,
        help="Outer scaffold folds (nested_cv_outer_folds). Default: "
             "get_config's n_splits. 1 = one grouped held-out split.")
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_activity_cliff_qsar.json)')
    # --pruning-mode removed 2026-10-05: pruning is off in every per-fold
    # search (handoff 2.3 answer 6).

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard folds finished by an earlier attempt '
             '(<output>.partial.jsonl) and recompute every one.')

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()

    apply_distance_metric_arg(args)
    targets = (args.targets.split(',') if args.targets else ALL_TARGETS)
    approved = approved_reps()
    if args.reps:
        rep_names = [r.strip().lower() for r in args.reps.split(',') if r.strip()]
        outside = [r for r in rep_names if r not in approved]
        if outside:
            parser.error(f"--reps {outside}: not in DATASET_REPS['{DATASET}']"
                         f"['general'] {approved}")
    else:
        rep_names = approved
    if args.n_folds is None:
        args.n_folds = default_n_folds(getattr(args, 'distance_metric', None))
    if args.n_folds < 1:
        parser.error('--n-folds must be 1 or more')

    print("=" * 70)
    print("ACTIVITY CLIFF QSAR EXPERIMENT (Dablander et al. 2023)")
    print(f"  Paper: J Cheminformatics 15:47")
    print(f"  Targets: {targets}")
    print(f"  Reps: {','.join(rep_names)}")
    print(f"  Models: the two decided for {DATASET}: "
          f"{', '.join(decided_model_names())}")
    print(f"  Outer folds: {args.n_folds}"
          + (" (one grouped held-out split, no spread over folds)"
             if args.n_folds == 1 else ''))
    print("=" * 70)

    output_path = (args.output or
                   'tests/results/comparisons/opig_activity_cliff_qsar.json')

    # Each FOLD of each target is written to <output>.partial.jsonl as soon as
    # it finishes and read back on the next attempt (per target since
    # 2026-09-25; per fold since 2026-10-05, dirty-run bug 12). The identity
    # carries CHECKPOINT_VERSION, so no file from the earlier 7-representation
    # run is read back. A target that raised, or had a fold whose search
    # failed, makes the script exit 1 after the JSON is written.
    from experiments._checkpoint import script_checkpoint, exit_if_failed
    _ck = script_checkpoint('opig_activity_cliff_qsar', output_path, args,
                            exclude=('targets', 'output'),
                            rep_names=list(rep_names),
                            checkpoint_version=CHECKPOINT_VERSION)
    _failed = []

    all_results = {}
    for target in targets:
        try:
            target_results = evaluate_target(target, rep_names, args.n_folds,
                                             _ck)
            if target_results is not None:
                all_results[target] = target_results
                if target_results.get('n_folds_with_failed_search'):
                    _failed.append(target)
            _write_results(all_results, output_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            _failed.append(target)

    # Cross-target analysis
    if all_results:
        print_analysis(all_results)

    p = _write_results(all_results, output_path)
    print(f"\nResults saved: {p}")
    exit_if_failed(_failed, 'target')


if __name__ == '__main__':
    main()
