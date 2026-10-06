#!/usr/bin/env python3
"""C11: Model-level fusion (ensemble) vs feature-level fusion (hybrid).

Paper: Complexity
Core question: an ensemble of one model per representation aggregates K sets of
predictions and needs all K featurisers + K models at inference forever. The
hybrid pays a ONE-OFF search, then deploys a SINGLE model on a fused feature
vector. Does the hybrid recover the ensemble's accuracy at a fraction of the
standing cost?

-----------------------------------------------------------------------------
THIS IS A PROCESS COMPARISON. EVERYTHING HAPPENS PER FOLD.
-----------------------------------------------------------------------------
Each arm is a PROCESS: a procedure you could actually run on a training set to
produce a deployable predictor. Every process is replicated independently on
each CV fold, and the ONLY thing it is ever allowed to see is that fold's
TRAINING split. It is then evaluated once, on that fold's held-out split.

Concretely, for one outer fold, the ensemble process is:

    1. split the fold's TRAINING data with an INNER group-aware CV
    2. score every (model, rep) on that inner CV          <- selection signal
    3. rank reps / pick top-K / choose weights / pick the aggregator,
       using the inner scores ONLY
    4. refit the chosen members on the FULL training split
    5. aggregate their predictions on the held-out split  <- the only time the
                                                             held-out split is
                                                             touched
Repeat, independently, for every fold. Average the held-out scores.

Nothing crosses a fold boundary. In particular the member ranking must NOT come
from the other folds' scores: a model scored on fold g was TRAINED on data that
includes fold f, so using fold g's score to choose members for fold f leaks
fold f's labels into the selection. That is a real leak and it is exactly what
an earlier version of this file did.

The hybrid process (aligned 2026-10-05, handoff 2.4): for each outer fold, one
search on that fold's training rows over the dataset's decided grid
(DECISIONS.md; fold_standard_selection -> fold_searches -> search_on_fold,
pruning off). Per decided downstream model it names the 2 best base
representations and the 2 best hybrids, both ranked on that fold's training
rows; a second hybrid whose selected features equal the first's is skipped for
the next. Each hybrid is rebuilt on the fold's training rows with
build_fold_hybrid, using the representation scores that same per-fold search
measured, and scored once on the held-out rows. No whole-dataset search is read
anywhere, and no choice is made from another fold's held-out score.

-----------------------------------------------------------------------------
ARMS
-----------------------------------------------------------------------------
  base_top1/2    the 2 best base representations of this fold's search
                 (ranked on the fold's training rows), per model
  hybrid_top1/2  the 2 best hybrids of this fold's search, over every
                 budget, per model (the standard selection). DECIDED
                 2026-10-06 (Adelaide delegated): the extra 2 hybrids at each
                 decided budget ('hybrid', 'hybrid_general2') are dropped.
  best_single    one model per base rep; report the rep with the best mean
                 over the held-out folds (picked on those folds, 2026-10-05)
  pca_matched    concat all reps, PCA to the budget dim, one model
  ensemble_s1    STYLE 1 (per model): fix one downstream model, train it on
                 every rep in a rep set, aggregate the PREDICTIONS
  ensemble_s2    STYLE 2 (optimal): let each rep pick its OWN best model
                 (per fold, on inner CV), then aggregate those predictions
  *_auto         the self-contained variants: the process ALSO picks its rep
                 set and aggregator on the inner CV. These are the true
                 like-for-like comparators for the hybrid, because the hybrid
                 likewise selects its own recipe from training data.
  oracle_single  best (model x rep) pair picked separately on each
                 held-out fold, by that fold's own score. NOT a fair arm.
                 It is the best any single-representation arm could do on
                 that fold, NOT an upper bound on the ensembles or hybrids:
                 they can beat it (ensemble 0.672 vs this 0.659 MCC on
                 nt_H3K4me3, server results 2026-10-01). Corrected 2026-10-05.

Rep sets:     top (top-3 by inner CV)  |  all (every representation of
              DATASET_REPS[ds]['general'], 2026-10-05)
Aggregators:  mean, median, trimmed20, iqr_filtered, mad_filtered, weighted,
              topk_mean, majority_vote (hard-label classification only)

-----------------------------------------------------------------------------
HISTORY / DIRECTIVE RECONCILIATION
-----------------------------------------------------------------------------
The prediction-averaging arm was removed on 2026-05-12 (commit 5d72d3c) under
the "no averaged baselines" directive. It is reinstated here because that
directive and this experiment are about different objects:

  * BANNED, and still banned: averaging rep SCORES into a composite "mean
    baseline" number. It hides which reps are good. Nothing here does that -
    baselines are still reported per rep, individually.
  * REINSTATED: averaging model PREDICTIONS. That is an ensemble - a real
    method and the strongest competitor the hybrid has. It is scored as a
    METHOD arm, never as a baseline.

2026-10-05 alignment (handoff 2.4), what changed:
  * models: the dataset's two decided models (decided_models(ds, 'general')),
    not the five DOWNSTREAM_MODELS.
  * the 'whole_data' recipe search is gone (no whole-dataset search anywhere);
    'per_fold' is the only setting.
  * the 2026-10-02 set of 2 general + 1 distance hybrid is replaced by the
    standard 2 base + 2 hybrids per model per fold. No distance hybrid: c11
    measures nothing with neighbours (handoff 2.3 item 1).
  * the budget and the best single representation are no longer re-picked
    from the other folds' held-out scores (fold_reselection), and the
    "re-picked ensemble" (best ensemble process by the other folds' held-out
    scores) is gone. The head-to-head reads choices made on training rows
    only: hybrid_top1, base_top1, and the ensemble chosen on the inner CV
    (ensemble_s1_auto).
  * representations: the single-representation and ensemble arms use every
    representation of DATASET_REPS[ds]['general']. Before, they left out
    every representation under 50 features (onehot_components on
    buchwald_hartwig) while the hybrid drew from it.
  * feature scaling done here (single representations, PCA input, hybrid
    input) is FeatureScaler: 0/1 features are left unscaled, every other
    feature is standardised (handoff 2.3 item 17).
  * outer folds from nested_cv_outer_folds with config['n_splits'], the same
    split the per-fold searches are filed under. One fold works: a spread over
    folds is then written as None with 'one fold, no spread'.

-----------------------------------------------------------------------------
COST ACCOUNTING
-----------------------------------------------------------------------------
Per Dehghani et al., "The Efficiency Misnomer" (ICLR 2022), no single cost
number is trustworthy; several are reported per arm:

  n_featurisers            distinct base reps that must be COMPUTED at inference
  featuriser_dim           summed width of those reps (upstream encoder cost)
  n_models_deployed        models to store and run
  model_input_dim          width of the downstream model's input
  model_bytes              pickled size of the deployed model(s)
  storage_bytes_per_sample bytes that must be PERSISTED per sample
  selection_fits           model fits the PROCESS itself burns to make its
                           choices (0 for a plain mean over all reps - that
                           ensemble needs no search at all, and is therefore
                           the real competitor to beat)
  fit_time_s / predict_time_s  measured wall-clock, summed over members

The honest caveat, stated up front because it is the first thing a reviewer
attacks: the hybrid does NOT escape the upstream featurisers. It still has to
run every rep its blueprint draws a column from. Its saving is real only insofar
as the blueprint touches FEWER reps than the ensemble does - so `n_featurisers`
is measured for the hybrid from `feature_info['allocation']`, not assumed.
PCA is the worst offender and is measured, not excused: it needs ALL reps at
inference plus a dense (budget x total_dim) projection matrix.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import pickle

import numpy as np
from scipy import stats as sp_stats
from sklearn.decomposition import PCA

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
    get_model_factories, cv_split,
    get_domain_metric,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_searches, fold_standard_selection,
)
from experiments.decided_grid import (
    decided_models, decided_search, decided_aug_conditions,
)
from kirby.find_best_hybrid import (
    nested_cv_outer_folds, nested_cv_cache_namespace, DEFAULT_N_FOLDS,
)
# 0/1 features pass through unscaled, every other feature is standardised
# (handoff 2.3 item 17, 2026-10-05).
from kirby.scaling import FeatureScaler
# A stale result database must stop the run, not degrade it (search_on_fold
# re-raises it on purpose; result_cache.py), 2026-10-05.
from kirby.result_cache import StaleResultDB

TOP_K = 3        # size of the "top" rep set (C11b, kept at 3 members, 2026-10-05)
AGG_TOP_K = 3    # members kept by the 'topk_mean' aggregator
INNER_FOLDS = 3  # inner CV used by every process to make its own choices
SEED = 42        # outer split and per-fold search namespace (ocv<n>f<i>s42)

# Stamped into the result file. Bumped 2026-10-05 for the per-fold standard
# selection: a file without it, or with another value, was written by the
# older design and is not this one.
# Bumped 2026-10-06: the 2 hybrids per decided budget are dropped.
RESULT_SCHEMA = 'c11_per_fold_standard_selection_2026_10_06'
ONE_FOLD_NOTE = 'one fold, no spread'


# =========================================================================
# FIT / PREDICT
# =========================================================================

def _fit_predict(model_factory, X_tr, y_tr, X_va, task_type):
    """Fit on a training split and predict a held-out split.

    Returns (model, agg_pred, hard_pred, fit_s, predict_s).

    `agg_pred` is what the aggregators average - it must live in a CONTINUOUS
    space or averaging is meaningless:
      regression      -> the predicted value
      classification  -> the predict_proba matrix (n, n_classes)
    `hard_pred` is the raw predict() output, kept for majority voting.
    """
    model = model_factory()
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    t0 = time.time()
    hard = model.predict(X_va)
    pred_s = time.time() - t0

    agg = hard
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        try:
            agg = model.predict_proba(X_va)
        except Exception:
            agg = hard
    return model, agg, hard, fit_s, pred_s


def _prep(X, tr_idx, va_idx):
    """Scale on train, apply to val. Scaler is fit on TRAIN ONLY.

    No PCA rotation. cap_dimensions used to rotate any representation wider
    than the training split through a PCA here, and the hybrid arm applies a
    StandardScaler and nothing else — so every single-representation baseline
    and every ensemble member was rotated and the arm they are compared
    against was not. The rotation comes off this side rather than being added
    to the hybrid: it is a second difference between the arms, and the
    comparison is meant to carry one.

    2026-10-05: FeatureScaler, not StandardScaler (handoff 2.3 item 17). A
    feature whose training values are all 0 or 1 passes through unscaled;
    every other feature is standardised. The hybrid arm uses the same scaler.
    """
    scaler = FeatureScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_va = scaler.transform(X[va_idx])
    return X_tr, X_va


def _to_metric_space(agg, task_type, metric_needs_proba, classes):
    """Map an aggregated prediction back into the space the metric expects.

    Averaging happens in probability space; the metric may want scores (AUC) or
    hard labels (accuracy/f1/mcc). Getting this wrong silently scores garbage,
    e.g. handing 0.4 to accuracy_score.
    """
    agg = np.asarray(agg)
    if task_type != 'classification':
        return agg

    if agg.ndim == 1:
        # Members had no predict_proba, so these are hard labels. A mean of
        # hard labels is a fractional non-class (0/1 -> 0.4) and would be
        # silently garbage to accuracy/f1/mcc: snap it back to the nearest real
        # class. A majority vote is already a class, so this is a no-op there.
        # AUC, by contrast, wants the continuous score.
        if metric_needs_proba:
            return agg
        cls = np.asarray(classes)
        return cls[np.argmin(np.abs(cls[None, :] - agg[:, None]), axis=1)]

    if metric_needs_proba:
        return agg[:, 1] if agg.shape[1] == 2 else agg
    return np.asarray(classes)[np.argmax(agg, axis=1)]


def _model_bytes(model):
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0


# =========================================================================
# AGGREGATORS
# =========================================================================
# Each takes a stack P of member predictions with members on axis 0:
#   regression      P.shape == (M, n)
#   classification  P.shape == (M, n, n_classes)
# plus per-member weights w (M,), and returns the aggregate without axis 0.
# Every one is a fixed rule; none of them ever sees the held-out split.

def _agg_mean(P, w):
    return P.mean(axis=0)


def _agg_median(P, w):
    return np.median(P, axis=0)


def _agg_trimmed(P, w):
    """20% trimmed mean - drops the most extreme member at each tail."""
    if P.shape[0] < 5:
        return np.median(P, axis=0)
    return sp_stats.trim_mean(P, 0.2, axis=0)


def _mask_mean(P, keep):
    """Mean over axis 0 under a boolean keep-mask; median where all rejected."""
    kf = keep.astype(float)
    while kf.ndim < P.ndim:
        kf = kf[..., None]
    denom = kf.sum(axis=0)
    out = (P * kf).sum(axis=0) / np.where(denom == 0, 1.0, denom)
    return np.where(denom == 0, np.median(P, axis=0), out)


def _agg_iqr_filtered(P, w):
    """Per-sample Tukey fence: drop members outside [Q1-1.5*IQR, Q3+1.5*IQR]."""
    if P.shape[0] < 4:
        return P.mean(axis=0)
    q1 = np.percentile(P, 25, axis=0)
    q3 = np.percentile(P, 75, axis=0)
    iqr = q3 - q1
    return _mask_mean(P, (P >= (q1 - 1.5 * iqr)) & (P <= (q3 + 1.5 * iqr)))


def _agg_mad_filtered(P, w):
    """Per-sample MAD fence: drop members >3 scaled-MAD from the median."""
    if P.shape[0] < 4:
        return P.mean(axis=0)
    med = np.median(P, axis=0)
    mad = np.median(np.abs(P - med), axis=0) * 1.4826
    mad = np.where(mad == 0, 1e-12, mad)
    return _mask_mean(P, np.abs(P - med) <= (3.0 * mad))


def _agg_weighted(P, w):
    """Mean weighted by each member's INNER-CV score on this fold's train split."""
    if w is None or not np.isfinite(w).any() or np.sum(w) <= 0:
        return P.mean(axis=0)
    ww = np.asarray(w, dtype=float)
    while ww.ndim < P.ndim:
        ww = ww[..., None]
    return (P * ww).sum(axis=0) / ww.sum(axis=0)


def _agg_topk(P, w):
    """Mean of the AGG_TOP_K members that scored best on the inner CV."""
    if w is None or P.shape[0] <= AGG_TOP_K:
        return P.mean(axis=0)
    idx = np.argsort(-np.asarray(w, dtype=float))[:AGG_TOP_K]
    return P[idx].mean(axis=0)


def _agg_majority(P, w):
    """Hard majority vote. Classification only; P is (M, n) hard labels."""
    return np.asarray(sp_stats.mode(P, axis=0, keepdims=False).mode)


CONTINUOUS_AGGREGATORS = {
    'mean': _agg_mean,
    'median': _agg_median,
    'trimmed20': _agg_trimmed,
    'iqr_filtered': _agg_iqr_filtered,
    'mad_filtered': _agg_mad_filtered,
    'weighted': _agg_weighted,
    f'top{AGG_TOP_K}_mean': _agg_topk,
}

# Which processes actually need the inner CV to make a choice. A plain mean over
# ALL reps needs no selection signal at all - it is the zero-search ensemble, and
# therefore the honest competitor to beat.
_NEEDS_SELECTION_AGG = {'weighted', f'top{AGG_TOP_K}_mean'}


def _needs_selection(rep_set, agg_name):
    return rep_set == 'top' or agg_name in _NEEDS_SELECTION_AGG


def _rank_weights(scores, lower_is_better):
    """Turn inner-CV scores into non-negative weights.

    Rank-based, so one blown-up member (a rep whose model diverged) cannot
    dominate the weighting the way raw-score weights would.
    """
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    if not finite.any():
        return None
    filler = np.nanmax(s[finite]) if lower_is_better else np.nanmin(s[finite])
    s = np.where(finite, s, filler)
    order = np.argsort(s) if lower_is_better else np.argsort(-s)
    w = np.empty(len(s), dtype=float)
    w[order] = np.arange(len(s), 0, -1)  # best member gets the largest weight
    return w


# =========================================================================
# COST MODEL
# =========================================================================

def _cost_row(n_featurisers, featuriser_dim, n_models, model_input_dim,
              model_bytes, fit_s, predict_s, storage_dim, selection_fits=0):
    """One arm's cost. Several indicators, never one."""
    return {
        'n_featurisers': int(n_featurisers),
        'featuriser_dim': int(featuriser_dim),
        'n_models_deployed': int(n_models),
        'model_input_dim': int(model_input_dim),
        'model_bytes': int(model_bytes),
        'storage_bytes_per_sample': int(4 * storage_dim),  # float32
        'selection_fits': int(selection_fits),
        'total_fit_time_s': float(fit_s),
        'total_predict_time_s': float(predict_s),
    }

def _summary(fold_scores):
    """Mean and spread over the folds that scored.

    Returns (mean, std, note). With one scored fold the spread is None and the
    note says 'one fold, no spread' (2026-10-05: a fold count of 1 is a single
    held-out split, not a spread of zero).
    """
    valid = [float(s) for s in fold_scores
             if s is not None and np.isfinite(s)]
    if not valid:
        return float('nan'), None, 'no fold scored'
    if len(valid) == 1:
        return valid[0], None, ONE_FOLD_NOTE
    return float(np.mean(valid)), float(np.std(valid)), None


def _entry_fields(e):
    """The recipe of one top_hybrids entry, scalars only."""
    return {
        'label': e.get('label'),
        'allocation_method': e.get('allocation_method'),
        'importance_method': e.get('importance_method'),
        'budget': e.get('budget'),
        'internal_evaluator': e.get('internal_evaluator'),
        'rep_count': e.get('rep_count'),
        'step_size': e.get('step_size'),
        'aug_condition': e.get('aug_condition') or 'none',
        'n_features_in_search': e.get('n_features'),
        # Mean over the search's inner folds, inside this outer fold's
        # training rows.
        'training_rows_score': e.get('score_mean'),
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """Ensemble (model-level fusion) vs hybrid (feature-level fusion)."""
    print("\n" + "=" * 78)
    print("EXPERIMENT C11: ENSEMBLE (model-level) VS HYBRID (feature-level)")
    print("  Process comparison. Every process is replicated per fold and makes")
    print("  ALL of its choices on that fold's training split only.")
    print("=" * 78)

    # 2026-10-05: 'whole_data' is gone. It took the hybrid's recipe from one
    # search over every row, which the alignment removes everywhere (handoff
    # 2.4 step 4). 'per_fold' is the only setting left.
    recipe_search = kwargs.get('recipe_search', 'per_fold')
    if recipe_search != 'per_fold':
        raise ValueError(
            f"recipe_search={recipe_search!r}: only 'per_fold' exists since "
            f"2026-10-05 (no whole-dataset search anywhere).")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']

    from kirby.find_best_hybrid import BUILTIN_METRICS
    metric_needs_proba = (metric_fn is BUILTIN_METRICS.get('auc', (None,))[0])

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # ---- Models: the dataset's two decided models (DECISIONS.md Step 1). ----
    # knn is a model of the distance search only, and c11 runs none.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    models = get_model_factories(decided, task_type=task_type,
                                 n_samples=len(labels))
    models_dropped = [
        {'downstream_model': m,
         'reason': ('get_model_factories gave no factory (above the model\'s '
                    'sample limit, or unknown name)')}
        for m in decided if m not in models]
    if not models:
        print(f"  No decided model can be fitted on {dataset_name}: "
              f"{models_dropped}")
        return {'error': 'no decided model can be fitted',
                'decided_models': decided, 'models_dropped': models_dropped,
                'result_schema': RESULT_SCHEMA}

    # ---- Representations ---------------------------------------------------
    # The hybrid builder gets data['reps'] as handed over (never filtered
    # before a hybrid builder, handoff 2.4 step 9). The per-representation
    # loops -- single representations, ensemble members, PCA input -- use
    # DATASET_REPS[ds]['general'] (representation pruning). Under run_all the
    # two are the same list. 2026-10-05: the tier filter that left every
    # representation under 50 features out of the singles and the ensembles
    # (onehot_components on buchwald_hartwig, while the hybrid drew from it)
    # is gone.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    rep_names = sorted(reps.keys())
    _pruned = (DATASET_REPS.get(dataset_name) or {}).get('general')
    if _pruned:
        single_reps = [r for r in rep_names if r in set(_pruned)]
    else:
        single_reps = list(rep_names)
    hybrid_pool_only = [r for r in rep_names if r not in set(single_reps)]
    if hybrid_pool_only:
        print(f"  NOTE: {hybrid_pool_only} are loaded but not in "
              f"DATASET_REPS['{dataset_name}']['general']; they are left out "
              f"of the single-representation and ensemble arms.")
    rep_dims = {k: int(v.shape[1]) for k, v in reps.items()}
    n_reps = len(single_reps)
    total_dim = sum(rep_dims[r] for r in rep_names)
    single_dim = sum(rep_dims[r] for r in single_reps)

    classes = np.unique(labels) if task_type == 'classification' else None

    # ---- Outer folds: the split the per-fold searches are filed under. -----
    # The count comes from config['n_splits'] only. nested_cv_outer_folds is
    # the single source of the split, so fold i here is fold i of the search
    # namespace ocv<n>f<i>s42.
    n_splits = int(config['n_splits'])
    folds = list(nested_cv_outer_folds(reps, labels, groups, n_splits, SEED,
                                       task_type))
    n_out = len(folds)
    sample_X = next(iter(reps.values()))

    print(f"\n  {n_reps} representations, {total_dim} features in the hybrid "
          f"pool, decided models {list(models)}, {n_out} outer fold(s)")
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    # Each fold's INNER folds, for the ensemble process's own choices. Indices
    # INTO the outer training split, group-aware. INNER_FOLDS whatever the
    # outer count (2026-10-05). Before, one outer fold meant one inner split:
    # 'top' and 'weighted' still chose members on that one split and scored,
    # but the *_auto processes need 2 or more inner splits, so
    # ensemble_s1_auto, ensemble_s2_auto and the head-to-head's ensemble side
    # were NaN.
    inner_folds = []
    for tr_idx, _ in folds:
        try:
            inner = list(cv_split(groups[tr_idx], INNER_FOLDS, sample_X[tr_idx],
                                  labels[tr_idx], mode='full',
                                  task_type=task_type))
        except Exception as e:
            print(f"    inner split failed ({type(e).__name__}: {e})")
            inner = []
        inner_folds.append(inner)
    n_in = [len(x) for x in inner_folds]
    print(f"  Inner CV per fold (process's own selection signal): {n_in}")
    if not any(n_in):
        print("  WARNING: no inner splits available - selection-based processes "
              "(top / weighted / topk / *_auto) will be skipped.")

    # ---- Augmentation: Step 8's conditions only. ---------------------------
    # The general search runs 'none' on every dataset (Step 8 keeps added
    # features on the distance search of ginkgo_colloidal and ginkgo_hic
    # only), so no c11 hybrid carries added features. search_on_fold applies
    # this itself; the line below says so in the log and the result file.
    augmentations = data.get('augmentations') or None
    aug_conditions = list(decided_aug_conditions(dataset_name, 'general'))
    augmentation_note = None
    if augmentations and not [c for c in aug_conditions if c != 'none']:
        augmentation_note = (
            f"augmentation comparison skipped on {dataset_name}: Step 8 cut "
            f"added features from the general search (conditions "
            f"{aug_conditions}); hybrids are built without them")
        print(f"  {augmentation_note}")

    # ---- Budgets (features): the decided general budgets. ------------------
    # Used by the PCA-matched control only (one row per budget). DECIDED
    # 2026-10-06 (Adelaide delegated): no per-budget hybrid rows.
    g_grid = decided_search(dataset_name, 'general')
    budgets = sorted(b for b in g_grid['budgets'] if b <= total_dim)
    budgets_above_pool = sorted(b for b in g_grid['budgets'] if b > total_dim)
    if budgets_above_pool:
        print(f"  Budgets above the pool's {total_dim} features get no "
              f"PCA-matched row: {budgets_above_pool}")

    # =====================================================================
    # 0. PER-FOLD SEARCH AND STANDARD SELECTION
    #    One search per outer fold, on that fold's training rows, decided
    #    grid, pruning off. Per model: 2 base representations + 2 hybrids
    #    over every budget (the standard 2 per model only).
    # =====================================================================
    print(f"\n  Per-fold searches: {n_out} outer fold(s), decided grid, "
          f"augmentation conditions {aug_conditions}")
    sel_all = []
    fold_search_errors = []
    search_fits_per_fold = []
    for f, (tr_idx, _va) in enumerate(folds):
        common = dict(train_idx=tr_idx, fold_i=f, n_outer_folds=n_out,
                      dataset_name=dataset_name, augmentations=augmentations,
                      task_type=task_type, metric=metric_name, seed=SEED)
        # Same arguments fold_standard_selection passes, so the call below is
        # served from the in-process memo; this one only reads the failures.
        try:
            srch = fold_searches(reps, labels, groups, tier='general', **common)
        except StaleResultDB:
            raise
        except Exception as e:
            srch = None
            fold_search_errors.append({
                'fold': int(f), 'tier': 'general', 'augmentation_condition': None,
                'error': f'{type(e).__name__}: {e}'})
            print(f"    fold {f}: search raised {type(e).__name__}: {e}")
        for rec in (getattr(srch, 'failures', None) or []):
            fold_search_errors.append({
                'fold': int(f), 'tier': 'general',
                'augmentation_condition': rec.get('aug_condition'),
                'error': (f"{rec.get('error_type')}: {rec.get('error')}")})
        n_fits = 0
        for _cond, s_f in (srch.items() if srch is not None else []):
            if s_f is None:
                continue
            rows = getattr(s_f, 'all_results', None) or []
            n_inner_search = next((len(r.get('fold_scores') or [])
                                   for r in rows if r.get('fold_scores')),
                                  DEFAULT_N_FOLDS)
            n_fits += len(rows) * n_inner_search
        search_fits_per_fold.append(int(n_fits))

        def _standard_sel(budget=None):
            try:
                return fold_standard_selection(
                    reps, labels, groups, models=list(models), budget=budget,
                    **common)
            except StaleResultDB:
                raise
            except Exception as e:
                fold_search_errors.append({
                    'fold': int(f), 'tier': 'general',
                    'augmentation_condition': None, 'budget': budget,
                    'error': f'selection {type(e).__name__}: {e}'})
                print(f"    fold {f} budget {budget}: selection raised "
                      f"{type(e).__name__}: {e}")
                return {m: None for m in models}
        sel_all.append(_standard_sel(None))
        _ok = sorted(m for m, s in sel_all[-1].items() if s is not None)
        print(f"    fold {f}: selection for {_ok}; "
              f"{n_fits} search fits")

    # Fits the hybrid's search burns, summed over the outer folds (one search
    # per fold, each over its own inner folds). Same quantity the file
    # reported before 2026-10-05 as search_cost_model_fits.
    search_fits = int(sum(search_fits_per_fold))

    # Per-fold training/held-out slices of every representation, built once.
    _slices = {}

    def _fold_slices(f):
        if f not in _slices:
            tr, va = folds[f]
            augs = augmentations or {}
            _slices[f] = (
                {k: v[tr] for k, v in reps.items()},
                {k: v[va] for k, v in reps.items()},
                ({k: v[tr] for k, v in augs.items()} if augs else None),
                ({k: v[va] for k, v in augs.items()} if augs else None))
        return _slices[f]

    # One build + fit per (fold, model, augmentation condition, recipe label):
    # the per-fold record below reads the scores the hybrid_top rows made.
    _hyb_cache = {}

    def _score_hybrid(f, model_name, entry):
        cond = entry.get('aug_condition') or 'none'
        key = (f, model_name, cond, entry.get('label'))
        if key in _hyb_cache:
            return _hyb_cache[key]
        tr_idx, va_idx = folds[f]
        tr_reps, va_reps, tr_aug, va_aug = _fold_slices(f)
        if cond == 'none':
            tr_aug = va_aug = None
        bp = entry['blueprint']
        rec = dict(_entry_fields(entry), fold=int(f), heldout_score=float('nan'),
                   n_features=None, reps_touched=[], fit_s=float('nan'),
                   predict_s=float('nan'), model_bytes=0, error=None)
        try:
            t0 = time.time()
            X_tr_h, X_va_h, info = build_fold_hybrid(
                tr_reps, va_reps, labels[tr_idx], bp,
                augmentations_train=tr_aug, augmentations_val=va_aug,
                groups=(groups[tr_idx] if groups is not None else None),
                # This fold's search measured these on this fold's training
                # rows (dirty-run bug 1): the representations a recipe with a
                # representation count keeps are the ones that search ranked.
                precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                **fold_hybrid_scoring_kwargs(
                    bp, model=models[model_name](), metric_fn=metric_fn,
                    lower_is_better=lower_is_better))
            build_s = time.time() - t0
        except Exception as e:
            rec['error'] = f'build {type(e).__name__}: {e}'
            print(f"    fold {f} {model_name} {entry.get('label')}: hybrid "
                  f"build failed ({e})")
            _hyb_cache[key] = rec
            return rec
        # THE reviewer's question: how many featurisers does the hybrid still
        # need? Measured from the build, never assumed.
        alloc = info.get('allocation') or {}
        touched = {r for r, n in alloc.items()
                   if isinstance(n, (int, float)) and n > 0}
        aug_info = info.get('augmentation_info') or {}
        if isinstance(aug_info, dict):
            touched |= {k for k in aug_info if k in reps}
        rec['reps_touched'] = sorted(touched)
        rec['n_features'] = int(np.asarray(X_tr_h).shape[1])
        try:
            sc = FeatureScaler()
            X_tr_s = sc.fit_transform(X_tr_h)
            X_va_s = sc.transform(X_va_h)
            model, agg, _h, a, b = _fit_predict(
                models[model_name], X_tr_s, labels[tr_idx], X_va_s, task_type)
            scored = _to_metric_space(agg, task_type, metric_needs_proba,
                                      classes)
            rec['heldout_score'] = float(metric_fn(labels[va_idx], scored))
            rec['fit_s'] = float(a + build_s)
            rec['predict_s'] = float(b)
            rec['model_bytes'] = _model_bytes(model)
        except Exception as e:
            rec['error'] = f'fit {type(e).__name__}: {e}'
            print(f"    fold {f} {model_name} {entry.get('label')}: hybrid fit "
                  f"failed ({e})")
        _hyb_cache[key] = rec
        return rec

    results = []
    cost_rows = []
    per_fold = [{'fold': int(f), 'n_train': int(len(tr)), 'n_val': int(len(va)),
                 'cache_namespace': nested_cv_cache_namespace(f, n_out, SEED),
                 'selection_scope': 'fold_train_rows',
                 'metric': metric_name,
                 'lower_is_better': bool(lower_is_better),
                 'search_errors': [e for e in fold_search_errors
                                   if e['fold'] == f],
                 'models': {}}
                for f, (tr, va) in enumerate(folds)]
    shortfall = []

    # ---- The two prediction stores that make everything else free. ----------
    # outer[model][rep][f]        : predictions on fold f's HELD-OUT split,
    #                               from a model fit on fold f's TRAIN split.
    # inner[model][rep][f][j]     : predictions on inner-val j, from a model fit
    #                               on inner-train j. Lives entirely inside fold
    #                               f's TRAIN split -> this is the ONLY legal
    #                               selection signal for fold f.
    outer_agg, outer_hard, outer_score = {}, {}, {}
    inner_agg, inner_hard, inner_score, inner_fold_score = {}, {}, {}, {}
    member_cost = {}

    y_out = [labels[v] for _, v in folds]
    y_in = [[labels[tr[iv]] for _, iv in inner_folds[f]]
            for f, (tr, _) in enumerate(folds)]

    for model_name, model_factory in models.items():
        print(f"\n  --- {model_name} ---")

        for d in (outer_agg, outer_hard, outer_score, inner_agg, inner_hard,
                  inner_score, inner_fold_score, member_cost):
            d[model_name] = {}

        # =================================================================
        # 1. Every single representation, every fold.
        #    This loop ALSO trains every ensemble member (outer) and every
        #    selection probe (inner). Nothing downstream refits anything.
        # =================================================================
        best_score = np.inf if lower_is_better else -np.inf
        best_rep = None
        single_results = {}

        for rep_name in single_reps:
            X = reps[rep_name]
            o_scores, o_agg, o_hard = [], [], []
            i_agg_f, i_hard_f, i_score_f = [], [], []
            ft = pt = 0.0
            mbytes = 0

            for f, (tr_idx, va_idx) in enumerate(folds):
                # ---- OUTER: fit on TRAIN, predict HELD-OUT --------------
                try:
                    X_tr, X_va = _prep(X, tr_idx, va_idx)
                    model, agg, hard, a, b = _fit_predict(
                        model_factory, X_tr, labels[tr_idx], X_va, task_type)
                    scored = _to_metric_space(agg, task_type,
                                              metric_needs_proba, classes)
                    o_scores.append(float(metric_fn(labels[va_idx], scored)))
                    o_agg.append(np.asarray(agg))
                    o_hard.append(np.asarray(hard))
                    ft += a
                    pt += b
                    mbytes = max(mbytes, _model_bytes(model))
                except Exception:
                    o_scores.append(float('nan'))
                    o_agg.append(None)
                    o_hard.append(None)

                # ---- INNER: entirely within fold f's TRAIN split --------
                # This is the process's selection signal. It never touches
                # va_idx, so nothing it decides can leak the held-out labels.
                ia, ih, isc = [], [], []
                for itr, iva in inner_folds[f]:
                    a_idx, b_idx = tr_idx[itr], tr_idx[iva]
                    try:
                        Xi_tr, Xi_va = _prep(X, a_idx, b_idx)
                        _m, agg_i, hard_i, _, _ = _fit_predict(
                            model_factory, Xi_tr, labels[a_idx], Xi_va, task_type)
                        s_i = _to_metric_space(agg_i, task_type,
                                               metric_needs_proba, classes)
                        isc.append(float(metric_fn(labels[b_idx], s_i)))
                        ia.append(np.asarray(agg_i))
                        ih.append(np.asarray(hard_i))
                    except Exception:
                        isc.append(float('nan'))
                        ia.append(None)
                        ih.append(None)
                i_agg_f.append(ia)
                i_hard_f.append(ih)
                i_score_f.append(isc)

            o_scores = [float(s) for s in o_scores]
            outer_agg[model_name][rep_name] = o_agg
            outer_hard[model_name][rep_name] = o_hard
            outer_score[model_name][rep_name] = o_scores
            inner_agg[model_name][rep_name] = i_agg_f
            inner_hard[model_name][rep_name] = i_hard_f
            inner_fold_score[model_name][rep_name] = i_score_f
            # Mean inner score per outer fold -> the per-fold ranking signal.
            inner_score[model_name][rep_name] = [
                float(np.nanmean(s)) if any(np.isfinite(x) for x in s)
                else float('nan')
                for s in i_score_f
            ]
            member_cost[model_name][rep_name] = (ft / max(n_out, 1),
                                                 pt / max(n_out, 1), mbytes)

            mean_s, std_s, note = _summary(o_scores)
            single_results[rep_name] = {
                'score_mean': mean_s,
                'score_std': std_s,
                'spread_note': note,
                'dim': rep_dims[rep_name],
                'fit_time_s': ft / max(n_out, 1),
                'predict_time_s': pt / max(n_out, 1),
                'model_bytes': mbytes,
                'fold_scores': o_scores,
            }
            if np.isfinite(mean_s) and (
                    (lower_is_better and mean_s < best_score) or
                    (not lower_is_better and mean_s > best_score)):
                best_score, best_rep = mean_s, rep_name

        if best_rep is None:
            print("    Best single: NONE (all reps failed)")
        else:
            bs = single_results[best_rep]
            print(f"    Best single (picked on the held-out folds): {best_rep} "
                  f"{metric_name}={bs['score_mean']:.4f} dim={bs['dim']}")
            results.append({
                'model': model_name, 'method': 'best_single', 'augmented': False,
                'selection_scope': 'heldout_fold_mean',
                'score_mean': bs['score_mean'], 'score_std': bs['score_std'],
                'spread_note': bs['spread_note'],
                'total_dim': bs['dim'], 'n_models_trained': 1,
                'total_fit_time_s': bs['fit_time_s'],
                'total_predict_time_s': bs['predict_time_s'],
                'fold_scores': bs['fold_scores'],
                'detail': {'best_rep': best_rep, 'all_reps': single_results},
                'cost': _cost_row(1, bs['dim'], 1, bs['dim'], bs['model_bytes'],
                                  bs['fit_time_s'], bs['predict_time_s'],
                                  bs['dim']),
            })
            cost_rows.append((model_name, 'best_single', results[-1]['cost'],
                              bs['score_mean']))

        # =================================================================
        # 1b. Standard selection, base half: the 2 best base representations
        #     of each fold's search (ranked on that fold's training rows).
        #     The held-out score is the fit above: same model, same scaler,
        #     same rows. 0 new fits.
        # =================================================================
        for i in range(2):
            slot = f'base_top{i + 1}'
            f_s, reps_f, train_f = [], [], []
            for f in range(n_out):
                s_m = sel_all[f].get(model_name)
                sel = ((s_m or {}).get('base_detail') or {}).get('selected') or []
                if len(sel) <= i:
                    f_s.append(float('nan'))
                    reps_f.append(None)
                    train_f.append(None)
                    continue
                rep = sel[i]['rep']
                reps_f.append(rep)
                train_f.append(sel[i].get('score'))
                f_s.append(outer_score[model_name].get(
                    rep, [float('nan')] * n_out)[f])
            if not any(r is not None for r in reps_f):
                continue
            mean_s, std_s, note = _summary(f_s)
            used = sorted({r for r in reps_f if r})
            results.append({
                'model': model_name, 'method': slot, 'slot': slot,
                'augmented': False, 'selection_scope': 'fold_train_rows',
                'score_mean': mean_s, 'score_std': std_s, 'spread_note': note,
                'fold_scores': [float(s) for s in f_s],
                'rep_per_fold': reps_f,
                'training_rows_score_per_fold': train_f,
                'n_models_trained': 1,
                'total_dim': max((rep_dims[r] for r in used), default=0),
                'cost': _cost_row(
                    len(used), sum(rep_dims[r] for r in used), 1,
                    max((rep_dims[r] for r in used), default=0),
                    max((member_cost[model_name].get(r, (0, 0, 0))[2]
                         for r in used), default=0),
                    float(np.mean([member_cost[model_name].get(r, (0, 0, 0))[0]
                                   for r in used])),
                    float(np.mean([member_cost[model_name].get(r, (0, 0, 0))[1]
                                   for r in used])),
                    max((rep_dims[r] for r in used), default=0)),
            })
            cost_rows.append((model_name, slot, results[-1]['cost'], mean_s))
            print(f"    {slot}: {metric_name}={mean_s:.4f} reps per fold "
                  f"{reps_f}")

        # =================================================================
        # 2. PCA-matched control. Fit on TRAIN, transform HELD-OUT.
        # =================================================================
        for budget in budgets:
            f_s, f_fit, f_pred = [], [], []
            n_comp = 0
            pca_bytes = 0
            # 2026-10-05: PCA can give at most (training rows - 1) components,
            # so every budget above that builds the same projection and its
            # row copies the one below. The number of components actually
            # used is stored per scored fold, beside the budget.
            n_comp_folds = []
            for tr_idx, va_idx in folds:
                y_tr = labels[tr_idx]
                X_tr_c = np.hstack([reps[r][tr_idx] for r in single_reps])
                X_va_c = np.hstack([reps[r][va_idx] for r in single_reps])
                n_comp = min(budget, X_tr_c.shape[1], len(y_tr) - 1)
                if n_comp < 2:
                    continue
                try:
                    # FeatureScaler: 0/1 features unscaled (2026-10-05).
                    sc = FeatureScaler()
                    X_tr_s = sc.fit_transform(X_tr_c)
                    X_va_s = sc.transform(X_va_c)
                    pca = PCA(n_components=n_comp, random_state=42)
                    X_tr_p = pca.fit_transform(X_tr_s)
                    X_va_p = pca.transform(X_va_s)
                    model, agg, _h, a, b = _fit_predict(
                        model_factory, X_tr_p, y_tr, X_va_p, task_type)
                except Exception:
                    continue
                scored = _to_metric_space(agg, task_type, metric_needs_proba,
                                          classes)
                f_s.append(float(metric_fn(labels[va_idx], scored)))
                f_fit.append(a)
                f_pred.append(b)
                n_comp_folds.append(int(n_comp))
                pca_bytes = max(pca_bytes,
                                _model_bytes(model) + pca.components_.nbytes)

            if not f_s:
                continue
            mean_s, std_s, note = _summary(f_s)
            results.append({
                'model': model_name, 'method': 'pca_matched', 'budget': budget,
                'score_mean': mean_s, 'score_std': std_s, 'spread_note': note,
                'total_dim': n_comp, 'n_models_trained': 1,
                # Components actually used, one per scored fold (same order
                # as fold_scores). Below the budget when the training rows
                # cap it; 'total_dim' is the last fold's value only.
                'n_components_per_fold': n_comp_folds,
                'capped_by_training_rows': bool(
                    any(c < min(budget, single_dim) for c in n_comp_folds)),
                'total_fit_time_s': float(np.mean(f_fit)),
                'total_predict_time_s': float(np.mean(f_pred)),
                'augmented': False, 'fold_scores': [float(s) for s in f_s],
                # PCA needs EVERY rep at inference plus a dense projection.
                'cost': _cost_row(n_reps, single_dim, 1, n_comp, pca_bytes,
                                  float(np.mean(f_fit)), float(np.mean(f_pred)),
                                  single_dim),
            })
            cost_rows.append((model_name, f'pca_b{budget}', results[-1]['cost'],
                              results[-1]['score_mean']))

        # =================================================================
        # 3. Hybrids, every one from its own fold's search.
        #    hybrid_top1/2: the standard 2, over every budget. The 2 best
        #    at each decided budget are no longer scored (DECIDED 2026-10-06,
        #    Adelaide delegated).
        #    Each is rebuilt on the fold's training rows and scored once on
        #    the held-out rows.
        # =================================================================
        def _hybrid_row(method, slot, entries_per_fold, budget=None):
            """One results row from one recipe per fold (None = no recipe)."""
            recs = []
            for f, e in enumerate(entries_per_fold):
                if e is None or e.get('blueprint') is None:
                    recs.append(None)
                    continue
                recs.append(_score_hybrid(f, model_name, e))
            if not any(r is not None for r in recs):
                return None
            f_s = [float(r['heldout_score']) if r is not None else float('nan')
                   for r in recs]
            mean_s, std_s, note = _summary(f_s)
            built = [r for r in recs if r is not None and r['error'] is None
                     and np.isfinite(r['heldout_score'])]
            touched = set()
            for r in built:
                touched |= set(r['reps_touched'])
            n_touch = len(touched) if touched else n_reps
            touch_dim = sum(rep_dims.get(r, 0) for r in touched) or total_dim
            widths = [r['n_features'] if r is not None else None for r in recs]
            n_feat = next((w for w in reversed(widths) if w), 0)
            fit_m = (float(np.mean([r['fit_s'] for r in built]))
                     if built else float('nan'))
            pred_m = (float(np.mean([r['predict_s'] for r in built]))
                      if built else float('nan'))
            row = {
                'model': model_name, 'method': method, 'slot': slot,
                'augmented': False, 'lineage': 'general', 'tier': 'general',
                'objective': 'predictive',
                'selection_scope': 'fold_train_rows',
                'recipe_search': 'per_fold',
                'budget': budget,
                'score_mean': mean_s, 'score_std': std_s, 'spread_note': note,
                'fold_scores': f_s,
                'n_folds_scored': len(built),
                'total_dim': n_feat,
                'width_built_per_fold': widths,
                'n_models_trained': 1,
                'total_fit_time_s': fit_m,
                'total_predict_time_s': pred_m,
                'reps_touched': sorted(touched),
                # The recipe each fold picked on its own training rows, with
                # its training-row score and its held-out score.
                'recipes_per_fold': [
                    None if r is None else
                    {k: r[k] for k in (
                        'fold', 'label', 'allocation_method',
                        'importance_method', 'budget', 'internal_evaluator',
                        'rep_count', 'step_size', 'aug_condition',
                        'n_features', 'n_features_in_search',
                        'training_rows_score', 'heldout_score', 'error')}
                    for r in recs],
            }
            if built:
                row['cost'] = _cost_row(
                    n_touch, touch_dim, 1, n_feat,
                    max(r['model_bytes'] for r in built), fit_m, pred_m,
                    n_feat, selection_fits=search_fits)
            return row

        for i in range(2):
            slot = f'hybrid_top{i + 1}'
            entries = []
            for f in range(n_out):
                s_m = sel_all[f].get(model_name)
                hs = (s_m or {}).get('hybrids') or []
                entries.append(hs[i] if len(hs) > i else None)
            row = _hybrid_row(slot, slot, entries)
            if row is None:
                continue
            results.append(row)
            if 'cost' in row:
                cost_rows.append((model_name, slot, row['cost'],
                                  row['score_mean']))
            print(f"    {slot}: {metric_name}={row['score_mean']:.4f} "
                  f"({row['n_folds_scored']}/{n_out} folds scored) budgets "
                  f"{[e.get('budget') if e else None for e in entries]}")

        # Per-fold record: what the fold's search chose for this model, and
        # what each choice scored on the held-out rows.
        for f in range(n_out):
            s_m = sel_all[f].get(model_name)
            if s_m is None:
                per_fold[f]['models'][model_name] = None
                shortfall.append({'fold': int(f), 'model': model_name,
                                  'reason': 'the per-fold search failed'})
                continue
            bsel = (s_m.get('base_detail') or {}).get('selected') or []
            hsel = s_m.get('hybrids') or []
            if len(bsel) < 2 or len(hsel) < 2:
                shortfall.append({
                    'fold': int(f), 'model': model_name,
                    'n_base': len(bsel), 'n_hybrids': len(hsel),
                    'reason': (s_m.get('base_detail') or {}).get(
                        'shortfall_reason')})
            per_fold[f]['models'][model_name] = {
                'base': [{'slot': f'base_top{i + 1}', 'rep': e['rep'],
                          'training_rows_score': e.get('score'),
                          'heldout_score': outer_score[model_name].get(
                              e['rep'], [float('nan')] * n_out)[f]}
                         for i, e in enumerate(bsel)],
                'hybrids': [
                    dict({k: v for k, v in _score_hybrid(f, model_name, e).items()
                          if k not in ('reps_touched',)},
                         slot=f'hybrid_top{i + 1}')
                    for i, e in enumerate(hsel)],
                'hybrids_skipped_identical_features': [
                    x.get('label') for x in
                    (s_m.get('hybrids_skipped_identical_features') or [])],
            }

    # =====================================================================
    # 4/5. ENSEMBLE PROCESSES
    #
    # No model is fitted below this line. Every member (outer) and every
    # selection probe (inner) was already trained above, so a process is just:
    #   per fold -> read the inner scores -> choose -> aggregate the outer
    #   predictions -> score once on the held-out split.
    # =====================================================================
    print("\n" + "=" * 78)
    print("  ENSEMBLE PROCESSES (per-fold; choices made on the inner CV only)")
    print("=" * 78)

    all_models = [m for m in models if m in outer_score and outer_score[m]]

    def _select(f, model_key, rep_set):
        """Run the SELECTION step of a process on fold f's training split.

        Returns (members, weights) where members is [(model, rep), ...].
        Ranking comes from `inner_score`, which was computed purely inside
        fold f's TRAIN split. The held-out split is not consulted.
        """
        scored = []
        if model_key == 'optimal':
            # STYLE 2: every rep picks its own best model, on the inner CV.
            for rep in single_reps:
                best_m, best_s = None, (np.inf if lower_is_better else -np.inf)
                for m in all_models:
                    s = inner_score.get(m, {}).get(rep, [np.nan] * n_out)[f]
                    if not np.isfinite(s):
                        continue
                    if (lower_is_better and s < best_s) or \
                       (not lower_is_better and s > best_s):
                        best_m, best_s = m, s
                if best_m is not None:
                    scored.append(((best_m, rep), best_s))
        else:
            # STYLE 1: one fixed downstream model, one member per rep.
            for rep in single_reps:
                s = inner_score.get(model_key, {}).get(rep, [np.nan] * n_out)[f]
                if np.isfinite(s):
                    scored.append(((model_key, rep), s))

        scored.sort(key=lambda x: x[1], reverse=not lower_is_better)
        if rep_set == 'top':
            scored = scored[:TOP_K]
        members = [mr for mr, _ in scored]
        w = _rank_weights([s for _, s in scored], lower_is_better)
        return members, w

    def _stack(members, store, f, j=None):
        """Stack member predictions, dropping any member that failed to fit."""
        rows, kept = [], []
        for (m, rep) in members:
            arr = store[m][rep][f]
            p = arr if j is None else arr[j]
            if p is None:
                continue
            rows.append(p)
            kept.append((m, rep))
        if len(rows) < 2:
            return None, []
        try:
            return np.stack(rows, axis=0), kept
        except ValueError:
            return None, []

    def _aggregate(members, w, store, y_true, f, j=None, agg_fn=None):
        """Aggregate members and score once against y_true."""
        P, kept = _stack(members, store, f, j)
        if P is None:
            return float('nan')
        w_kept = None
        if w is not None and len(w) == len(members):
            wmap = dict(zip(members, w))
            w_kept = np.array([wmap[k] for k in kept], dtype=float)
        try:
            agg = agg_fn(P, w_kept)
            scored = _to_metric_space(agg, task_type, metric_needs_proba, classes)
            return float(metric_fn(y_true, scored))
        except Exception:
            return float('nan')

    def _run_process(model_key, rep_set, agg_name, agg_fn, hard=False):
        """Replicate one process across every fold. Returns per-fold scores."""
        store = outer_hard if hard else outer_agg
        out = []
        for f in range(n_out):
            if _needs_selection(rep_set, agg_name) and not inner_folds[f]:
                out.append(float('nan'))       # no inner CV -> cannot select
                continue
            members, w = _select(f, model_key, rep_set)
            if len(members) < 2:
                out.append(float('nan'))
                continue
            out.append(_aggregate(members, w, store, y_out[f], f, agg_fn=agg_fn))
        return out

    def _inner_score_of(model_key, rep_set, agg_name, agg_fn, f, hard=False):
        """Score a process on fold f's INNER CV. Used by the *_auto processes.

        Membership for inner fold j is ranked leave-one-inner-fold-out, so the
        aggregator is chosen without any inner fold scoring itself. All of this
        still lives inside fold f's TRAIN split.
        """
        store = inner_hard if hard else inner_agg
        J = len(inner_folds[f])
        if J < 2:
            return float('nan')
        vals = []
        for j in range(J):
            scored = []
            src = all_models if model_key == 'optimal' else [model_key]
            for rep in single_reps:
                best_m, best_s = None, (np.inf if lower_is_better else -np.inf)
                for m in src:
                    fs = inner_fold_score.get(m, {}).get(rep)
                    if not fs:
                        continue
                    other = [fs[f][k] for k in range(J)
                             if k != j and np.isfinite(fs[f][k])]
                    if not other:
                        continue
                    s = float(np.mean(other))
                    if (lower_is_better and s < best_s) or \
                       (not lower_is_better and s > best_s):
                        best_m, best_s = m, s
                if best_m is not None:
                    scored.append(((best_m, rep), best_s))
            scored.sort(key=lambda x: x[1], reverse=not lower_is_better)
            if rep_set == 'top':
                scored = scored[:TOP_K]
            members = [mr for mr, _ in scored]
            if len(members) < 2:
                continue
            w = _rank_weights([s for _, s in scored], lower_is_better)
            vals.append(_aggregate(members, w, store, y_in[f][j], f, j=j,
                                   agg_fn=agg_fn))
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float('nan')

    rep_sets = ['top', 'all']

    for model_key, method in ([(m, 'ensemble_s1') for m in all_models] +
                              [('optimal', 'ensemble_s2')]):
        if model_key == 'optimal' and not all_models:
            continue
        print(f"\n  {model_key} ({method}):")

        aggs = list(CONTINUOUS_AGGREGATORS.items())
        if task_type == 'classification' and not metric_needs_proba:
            # Majority vote only makes sense for hard-label metrics; under AUC a
            # 0/1 vote discards the ranking the metric is built on.
            aggs.append(('majority_vote', _agg_majority))

        for rep_set in rep_sets:
            for agg_name, agg_fn in aggs:
                hard = (agg_name == 'majority_vote')
                fs = _run_process(model_key, rep_set, agg_name, agg_fn, hard=hard)
                if not any(np.isfinite(s) for s in fs):
                    continue
                _emit(results, cost_rows, model_key,
                      method, rep_set, agg_name, fs, _select, member_cost,
                      rep_dims, metric_name, n_out, inner_folds, single_reps,
                      all_models)

        # ---- The self-contained process: it ALSO picks its own rep set and
        # ---- aggregator, on the inner CV. This is the true like-for-like
        # ---- comparator for the hybrid, which likewise selects its own recipe
        # ---- from training data.
        auto_fs, auto_choice = [], []
        for f in range(n_out):
            if len(inner_folds[f]) < 2:
                auto_fs.append(float('nan'))
                auto_choice.append(None)
                continue
            best_cfg, best_v = None, (np.inf if lower_is_better else -np.inf)
            for rep_set in rep_sets:
                for agg_name, agg_fn in aggs:
                    hard = (agg_name == 'majority_vote')
                    v = _inner_score_of(model_key, rep_set, agg_name, agg_fn, f,
                                        hard=hard)
                    if not np.isfinite(v):
                        continue
                    if (lower_is_better and v < best_v) or \
                       (not lower_is_better and v > best_v):
                        best_cfg, best_v = (rep_set, agg_name, agg_fn, hard), v
            if best_cfg is None:
                auto_fs.append(float('nan'))
                auto_choice.append(None)
                continue
            rep_set, agg_name, agg_fn, hard = best_cfg
            members, w = _select(f, model_key, rep_set)
            store = outer_hard if hard else outer_agg
            auto_fs.append(_aggregate(members, w, store, y_out[f], f,
                                      agg_fn=agg_fn))
            auto_choice.append(f'{rep_set}|{agg_name}')

        if any(np.isfinite(s) for s in auto_fs):
            _emit(results, cost_rows, model_key,
                  f'{method}_auto', 'auto', 'auto', auto_fs, _select,
                  member_cost, rep_dims, metric_name, n_out, inner_folds,
                  single_reps, all_models, auto_choice=auto_choice)
            print(f"    per-fold choice: {auto_choice}")

    # ---- ORACLE: best (model x rep) read off the held-out fold itself. -----
    # Deliberately optimistic, NOT a fair arm: on each held-out fold, the
    # single (model, representation) pair that scored best on THAT fold.
    # It bounds single-representation arms only. An ensemble or hybrid can
    # score above it, so it is not an upper bound on them (2026-10-05).
    if all_models:
        oracle = []
        for f in range(n_out):
            best = np.nan
            for m in all_models:
                for rep in single_reps:
                    s = outer_score.get(m, {}).get(rep, [np.nan] * n_out)[f]
                    if not np.isfinite(s):
                        continue
                    if not np.isfinite(best) or \
                       (lower_is_better and s < best) or \
                       (not lower_is_better and s > best):
                        best = s
            oracle.append(float(best))
        if any(np.isfinite(s) for s in oracle):
            o_mean, o_std, o_note = _summary(oracle)
            results.append({
                'model': 'optimal', 'method': 'oracle_single', 'augmented': False,
                'rep_set': 'all', 'aggregator': 'none', 'is_oracle': True,
                'selection_scope': 'heldout_fold',
                'score_mean': o_mean, 'score_std': o_std, 'spread_note': o_note,
                'fold_scores': oracle,
                'n_models_trained': 1, 'total_dim': 0,
                'total_fit_time_s': 0.0, 'total_predict_time_s': 0.0,
                'description': (
                    "best single (model, representation) pair picked on each "
                    "held-out fold by that fold's score; bounds the "
                    "single-representation arms only, not the ensembles or "
                    "hybrids"),
            })
            print(f"\n  oracle_single (best single pair picked on each held-out "
                  f"fold; not a bound on ensembles/hybrids): "
                  f"{metric_name}={o_mean:.4f}")

    # =====================================================================
    # HEAD-TO-HEAD. Every side chose on training rows only:
    #   hybrid_top1      rank-1 hybrid of the fold's search
    #   base_top1        rank-1 base representation of the fold's search
    #   ensemble_auto    ensemble_s1_auto, rep set + aggregator on inner CV
    # 2026-10-05: the budget and the best single representation are no longer
    # re-picked from the other folds' held-out scores (fold_reselection), and
    # the 're-picked ensemble' (best ensemble process by the other folds'
    # held-out scores) is gone with them.
    # =====================================================================
    def _delta(better, worse):
        """Direction-corrected: positive means `better` really is better."""
        if not (np.isfinite(better) and np.isfinite(worse)):
            return float('nan')
        return (worse - better) if lower_is_better else (better - worse)

    def _paired_delta(better_folds, worse_folds):
        """Mean over folds of the per-fold difference, and how many folds.

        `_delta` above subtracts two means, which is only the same number when
        both arms scored the same folds. Both lists are fold-INDEX-aligned, so
        position f is the same held-out fold on both sides; a fold missing
        from either side contributes no term.
        """
        a = list(better_folds or [])
        b = list(worse_folds or [])
        terms = []
        for f in range(max(len(a), len(b))):
            av = a[f] if f < len(a) else None
            bv = b[f] if f < len(b) else None
            if av is None or bv is None:
                continue
            av, bv = float(av), float(bv)
            if not (np.isfinite(av) and np.isfinite(bv)):
                continue
            terms.append((bv - av) if lower_is_better else (av - bv))
        if not terms:
            return float('nan'), 0
        return float(np.mean(terms)), len(terms)

    def _row(model, method):
        return next((r for r in results
                     if r['model'] == model and r['method'] == method), None)

    head_to_head = {}
    for m in all_models:
        h1 = _row(m, 'hybrid_top1') or {}
        h2 = _row(m, 'hybrid_top2') or {}
        b1 = _row(m, 'base_top1') or {}
        au = _row(m, 'ensemble_s1_auto') or {}
        bs = _row(m, 'best_single') or {}
        h_folds = h1.get('fold_scores') or []
        h2_folds = h2.get('fold_scores') or []
        b_folds = b1.get('fold_scores') or []
        a_folds = au.get('fold_scores') or []
        h = h1.get('score_mean', float('nan'))
        a = au.get('score_mean', float('nan'))
        b = b1.get('score_mean', float('nan'))
        auto_paired, auto_nf = _paired_delta(a_folds, h_folds)
        auto2_paired, auto2_nf = _paired_delta(a_folds, h2_folds)
        single_paired, single_nf = _paired_delta(h_folds, b_folds)
        head_to_head[m] = {
            'selection_scope': 'fold_train_rows',
            'hybrid_recipe_search': 'per_fold',
            'hybrid_top1': h,
            'hybrid_top2': h2.get('score_mean', float('nan')),
            'ensemble_auto': a,            # self-contained ensemble process
            'base_top1': b,
            # Labelled, not a training-row choice: the representation with
            # the best mean over the held-out folds (C11e, 2026-10-05).
            'best_single_picked_on_heldout': bs.get('score_mean',
                                                    float('nan')),
            'auto_minus_hybrid': _delta(a, h),
            'hybrid_minus_single': _delta(h, b),
            # The paired form: the mean over folds of the per-fold
            # difference, not the difference of two means. They separate
            # whenever the two arms cover different folds; this is the one to
            # report.
            'auto_minus_hybrid_mean_of_folds': auto_paired,
            'auto_minus_hybrid_top2_mean_of_folds': auto2_paired,
            'hybrid_minus_single_mean_of_folds': single_paired,
            'n_paired_folds': {
                'auto_minus_hybrid': auto_nf,
                'auto_minus_hybrid_top2': auto2_nf,
                'hybrid_minus_single': single_nf,
            },
            'spread_note': (ONE_FOLD_NOTE if n_out == 1 else None),
            'hybrid_fold_scores': h_folds,
            'hybrid_top2_fold_scores': h2_folds,
            'ensemble_auto_fold_scores': [float(x) for x in a_folds],
            'base_top1_fold_scores': b_folds,
            'ensemble_auto_choice_per_fold': au.get('auto_choice_per_fold', []),
            'hybrid_recipe_per_fold': [
                None if r is None else r.get('label')
                for r in (h1.get('recipes_per_fold') or [])],
            'base_top1_rep_per_fold': b1.get('rep_per_fold', []),
        }
    au2 = _row('optimal', 'ensemble_s2_auto')
    if au2 is not None:
        # Style 2 lets each representation pick its own model, so it belongs
        # to no single model's hybrid. Its own entry, as before 2026-10-05.
        head_to_head['optimal'] = {
            'selection_scope': 'fold_train_rows',
            'ensemble_auto': au2['score_mean'],
            'ensemble_auto_fold_scores': au2['fold_scores'],
            'ensemble_auto_choice_per_fold': au2.get('auto_choice_per_fold', []),
        }

    print("\n" + "=" * 78)
    print("  HEAD-TO-HEAD (every side chosen on training rows only)")
    print("=" * 78)
    print(f"    {'Model':<9} {'Hybrid1':>9} {'Hybrid2':>9} {'EnsAuto':>9} "
          f"{'Base1':>9} {'Auto-Hyb':>9} {'paired':>9}")
    print("    " + "-" * 72)
    for m in all_models:
        hh = head_to_head[m]
        print(f"    {m:<9} {hh['hybrid_top1']:>9.4f} {hh['hybrid_top2']:>9.4f} "
              f"{hh['ensemble_auto']:>9.4f} {hh['base_top1']:>9.4f} "
              f"{hh['auto_minus_hybrid']:>+9.4f} "
              f"{hh['auto_minus_hybrid_mean_of_folds']:>+9.4f}")
    if 'optimal' in head_to_head:
        print(f"    {'optimal':<9} {'':>9} {'':>9} "
              f"{head_to_head['optimal']['ensemble_auto']:>9.4f}")
    print("    Auto-Hyb is the difference of the two means; paired is the mean")
    print("    over folds of the per-fold difference. They differ only when the")
    print("    two arms scored different folds. Auto-Hyb > 0 => the ensemble")
    print(f"    scored better in {metric_name}.")
    if n_out == 1:
        print(f"    {ONE_FOLD_NOTE}.")
    if shortfall:
        print(f"\n  {len(shortfall)} (fold, model) pair(s) short of 2 base "
              f"representations or 2 hybrids; reasons in 'shortfall'.")

    # =====================================================================
    # COST + BREAK-EVEN
    # =====================================================================
    print("\n" + "=" * 78)
    print("  STANDING COST PER ARM (after any one-off search)")
    print("=" * 78)
    print(f"    {'Model':<9} {'Arm':<26} {'Score':>8} {'Feat':>5} {'FeatDim':>8} "
          f"{'Mdls':>5} {'InDim':>7} {'MB':>7} {'SelFits':>8}")
    print("    " + "-" * 92)
    for mdl, arm, c, score in cost_rows:
        sc = score if np.isfinite(score) else float('nan')
        print(f"    {mdl:<9} {arm:<26} {sc:>8.4f} {c['n_featurisers']:>5} "
              f"{c['featuriser_dim']:>8} {c['n_models_deployed']:>5} "
              f"{c['model_input_dim']:>7} {c['model_bytes'] / 1e6:>7.2f} "
              f"{c['selection_fits']:>8}")

    # The pair compared here is the head-to-head's: hybrid_top1 against the
    # self-contained ensemble (or, where no inner CV existed, the zero-search
    # ensemble: every representation, plain mean). Before 2026-10-05 both
    # sides were the arm with the best mean over the held-out folds.
    break_even = {}
    for model_name in all_models:
        h = _row(model_name, 'hybrid_top1')
        e = _row(model_name, 'ensemble_s1_auto')
        if e is None:
            e = next((r for r in results if r['model'] == model_name
                      and r['method'] == 'ensemble_s1'
                      and r.get('rep_set') == 'all'
                      and r.get('aggregator') == 'mean'), None)
        if h is None or e is None or 'cost' not in h or 'cost' not in e:
            continue
        n_saved = e['cost']['n_models_deployed'] - h['cost']['n_models_deployed']

        break_even[model_name] = {
            'hybrid_arm': 'hybrid_top1',
            'ensemble_arm': f"{e['method']}|{e.get('rep_set')}|"
                            f"{e.get('aggregator')}",
            'search_cost_model_fits': int(search_fits),
            'ensemble_selection_fits': e['cost']['selection_fits'],
            'ensemble_models_deployed': e['cost']['n_models_deployed'],
            'hybrid_models_deployed': h['cost']['n_models_deployed'],
            'ensemble_featurisers': e['cost']['n_featurisers'],
            'hybrid_featurisers': h['cost']['n_featurisers'],
            'featuriser_reduction': (
                e['cost']['n_featurisers'] / h['cost']['n_featurisers']
                if h['cost']['n_featurisers'] else float('nan')),
            'storage_reduction': (
                e['cost']['storage_bytes_per_sample'] /
                h['cost']['storage_bytes_per_sample']
                if h['cost']['storage_bytes_per_sample'] else float('nan')),
            'model_fits_saved_per_retrain': int(n_saved),
            'break_even_retrains': (float(search_fits) / n_saved if n_saved > 0
                                    else float('inf')),
            'hybrid_recipe_search': 'per_fold',
            'accuracy_delta_ensemble_minus_hybrid': _delta(
                e['score_mean'], h['score_mean']),
            # ★ THE load-bearing number. Xu et al. (Cost-Sensitive Tree of
            # Classifiers, ICML 2013): test-time cost is DOMINATED by feature
            # extraction, and a featuriser is paid for the first time ANY of its
            # columns is used. So if the blueprint draws even one column from
            # every rep, the hybrid runs every encoder the ensemble runs and the
            # compute saving collapses to the downstream forward pass - a
            # rounding error next to a protein language model.
            # The featurisers counted here are the union over the folds' own
            # recipes, so a recipe that moves between folds shows up as a
            # larger count -- which is what deployment would actually pay.
            'hybrid_rep_sparsity': (h['cost']['n_featurisers'] / n_reps
                                    if n_reps else float('nan')),
            'featurisation_saving_exists': bool(
                h['cost']['n_featurisers'] < e['cost']['n_featurisers']),
        }

    if break_even:
        print("\n" + "=" * 78)
        print("  BREAK-EVEN: what the hybrid's one-off search buys")
        print("=" * 78)
        for m, be in break_even.items():
            d = be['accuracy_delta_ensemble_minus_hybrid']
            print(f"    {m} ({be['hybrid_arm']} against {be['ensemble_arm']}):")
            print(f"      featurisers  {be['ensemble_featurisers']:>3} (ens) -> "
                  f"{be['hybrid_featurisers']:>3} (hyb)   "
                  f"{be['featuriser_reduction']:.1f}x fewer")
            print(f"      models       {be['ensemble_models_deployed']:>3} (ens) -> "
                  f"{be['hybrid_models_deployed']:>3} (hyb)")
            print(f"      storage/sample                     "
                  f"{be['storage_reduction']:.1f}x smaller")
            print(f"      ensemble - hybrid, {metric_name} "
                  f"(positive = ensemble scored better): {d:+.4f}")
            print(f"      one-off search                     "
                  f"{be['search_cost_model_fits']} fits, repaid after "
                  f"{be['break_even_retrains']:.1f} retrains")
            print(f"      ensemble's OWN selection cost      "
                  f"{be['ensemble_selection_fits']} fits per fold")
            if not be['featurisation_saving_exists']:
                print("      The hybrid's recipes draw from every representation "
                      "the ensemble needs:")
                print("      featurisation cost is the same; the only saving is "
                      "the downstream forward pass.")

        print("\n    Deployment regime matters and the two answers differ:")
        print("      online - featurisers run per query -> the win is compute,")
        print("               real only where rep_sparsity < 1.")
        print("      batch  - reps precomputed and cached -> featurisation is paid")
        print("               once per sample, so the honest axis is STORAGE.")
        print("    This pipeline precomputes reps, i.e. the BATCH regime.")

    return {
        'result_schema': RESULT_SCHEMA,
        'results': results,
        'per_fold': per_fold,
        'selection_scope': 'fold_train_rows',
        'selection_rule': ('per model, per outer fold: the 2 best base '
                           'representations and the 2 best hybrids of a '
                           "search on that fold's training rows (decided "
                           'grid, pruning off)'),
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'n_outer_folds': int(n_out),
        'n_splits_requested': int(n_splits),
        'spread_note': (ONE_FOLD_NOTE if n_out == 1 else None),
        'models': list(models),
        'decided_models': decided,
        'models_dropped': models_dropped,
        'n_reps': n_reps,
        'total_dim': total_dim,
        'rep_names': rep_names,
        'single_reps': single_reps,
        'reps_in_hybrid_pool_only': hybrid_pool_only,
        'rep_dims': rep_dims,
        'n_inner_folds': n_in,
        'budgets': budgets,
        'budgets_above_pool': budgets_above_pool,
        'aug_conditions': aug_conditions,
        'augmentation_note': augmentation_note,
        'head_to_head': head_to_head,
        'break_even': break_even,
        'search_cost_model_fits': int(search_fits),
        'search_fits_per_fold': search_fits_per_fold,
        'hybrid_recipe_search': 'per_fold',
        'hybrid_recipe_selected_on': "each outer fold's training rows",
        'shortfall': shortfall,
        # Every per-fold search or selection that raised, with its outer fold,
        # augmentation condition and the error text. A hybrid arm missing from
        # 'results' has its cause here.
        'fold_search_errors': fold_search_errors,
    }


def _emit(results, cost_rows, model_key, method,
          rep_set, agg_name, fold_scores, select_fn, member_cost, rep_dims,
          metric_name, n_out, inner_folds, single_reps, all_models,
          auto_choice=None):
    """Record one ensemble process: score and standing cost."""
    valid = [s for s in fold_scores if np.isfinite(s)]
    if not valid:
        return
    score, score_std, spread_note = _summary(fold_scores)

    # Membership is re-selected INDEPENDENTLY on every fold, from that fold's
    # training split alone. Record it per fold: it is the audit trail proving no
    # held-out information entered the choice, and it shows how much the process
    # actually changes its mind from fold to fold.
    #
    # For the *_auto arms the rep set is not fixed: fold f picked its own on
    # that fold's inner CV, and auto_choice[f] records it as 'rep_set|agg'.
    per_fold_members = []
    for f in range(n_out):
        rep_set_f = rep_set
        if rep_set == 'auto':
            choice = auto_choice[f] if (auto_choice and f < len(auto_choice)) else None
            if choice is None:
                per_fold_members.append([])   # this fold made no choice
                continue
            rep_set_f = choice.split('|', 1)[0]
        try:
            members, _ = select_fn(f, model_key, rep_set_f)
            per_fold_members.append(sorted(f'{m}:{r}' for m, r in members))
        except Exception:
            per_fold_members.append([])

    # Cost is what this process DEPLOYS. Since membership varies by fold, take
    # the union: you must ship every featuriser and model the process can ask for.
    union = set()
    for f in range(n_out):
        for s in per_fold_members[f]:
            m, r = s.split(':', 1)
            union.add((m, r))
    if not union:
        return

    reps_needed = {rep for _, rep in union}
    fit_s = pred_s = 0.0
    mbytes = 0
    for (m, rep) in union:
        c = member_cost.get(m, {}).get(rep)
        if not c:
            continue
        fit_s += c[0]
        pred_s += c[1]
        mbytes += c[2]
    feat_dim = sum(rep_dims.get(r, 0) for r in reps_needed)

    # What the PROCESS itself burns to make its choices, per fold. A plain mean
    # over all reps needs no selection signal at all -> 0 fits. That zero-search
    # ensemble is the honest competitor to beat.
    n_probe_models = len(all_models) if model_key == 'optimal' else 1
    J = int(np.mean([len(x) for x in inner_folds])) if inner_folds else 0
    sel_fits = (J * len(single_reps) * n_probe_models
                if (rep_set == 'auto' or _needs_selection(rep_set, agg_name))
                else 0)

    row = {
        'model': model_key, 'method': method, 'rep_set': rep_set,
        'aggregator': agg_name, 'augmented': False,
        # A fixed rule: membership on the inner CV where it needs one, none
        # where it does not. Nothing reads a held-out score.
        'selection_scope': 'fold_train_rows',
        'score_mean': score, 'score_std': score_std,
        'spread_note': spread_note,
        'fold_scores': [float(s) for s in fold_scores],
        'n_models_trained': len(union), 'total_dim': feat_dim,
        'members': sorted(f'{m}:{r}' for m, r in union),
        'members_per_fold': per_fold_members,
        'total_fit_time_s': fit_s, 'total_predict_time_s': pred_s,
        # An ensemble must run EVERY member's featuriser and EVERY member's
        # model, at every inference, forever.
        'cost': _cost_row(len(reps_needed), feat_dim, len(union), feat_dim,
                          mbytes, fit_s, pred_s, feat_dim,
                          selection_fits=sel_fits),
    }
    if auto_choice is not None:
        row['auto_choice_per_fold'] = auto_choice
    results.append(row)
    cost_rows.append((model_key, f'{method.replace("ensemble_", "")}|{rep_set}'
                                 f'|{agg_name}', row['cost'], score))
    print(f"    {method}/{rep_set:<4}/{agg_name:<14} {metric_name}={score:>8.4f}  "
          f"members={len(union):>2} featurisers={len(reps_needed):>2} "
          f"selfits={sel_fits}")


if __name__ == '__main__':
    parser = create_experiment_parser('C11: Ensemble vs Hybrid',
                                      has_delta=True, has_augment=True)
    # 2026-10-05: 'whole_data' removed (no whole-dataset search anywhere).
    # The flag stays so existing command lines still parse.
    parser.add_argument(
        '--recipe-search', dest='recipe_search',
        choices=['per_fold'], default='per_fold',
        help="Where the hybrid's recipe is chosen: per_fold, the only "
             "setting. Each outer fold searches its own on its own training "
             "rows, matching what the ensemble arm already does.")
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Recipe search: {args.recipe_search}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  recipe_search=args.recipe_search)
    save_experiment_results(results, args.output, 'c11_ensemble_vs_hybrid',
                            args.dataset)
