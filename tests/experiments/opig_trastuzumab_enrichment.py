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
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
    honest_rescore, build_honest_rescore_input, hr_exclude_folds,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Representations come from DATASET_REPS['tz_her2_binary'] — the pool the
# rep-pruning step approved and precompute builds. The hand-written list this
# replaced asked for onehot, aac, igt5 and abroberta, none of which are in that
# pool and none of which this experiment does anything special with; it also
# skipped 6 of the 10 that are. The paper's own methods survive the switch:
# BLOSUM and AbLang2 are both in the pool already.
#
# Three representations are added back, each for a reason:
#   esmc_llr, antiberty_llr  the zero-shot ranking arm scores these two and
#                            nothing else (LLR_REPS, below). The paper's
#                            protocol is zero-shot ranking; drop these and that
#                            arm produces no rows at all.
#   esm2                     the paper evaluates ESM as one of its four deep
#                            -learning methods, and its headline result is that
#                            BLOSUM beat all four. Without ESM that comparison
#                            loses one of the four.
REP_EXCEPTIONS = ['esmc_llr', 'antiberty_llr', 'esm2']


def _default_reps():
    from experiments._shared import approved_rep_pool
    return approved_rep_pool('tz_her2_binary', extra=REP_EXCEPTIONS)

ENRICHMENT_FRACTIONS = [0.01, 0.05, 0.10]  # top-1%, top-5%, top-10%
# Note: at n_val ~= 1000 (5-fold on ~5K), top-1% = ~10 samples — high
# variance. Reported but interpret with care; E@5% and E@10% are more stable.

MODELS_LIST = DOWNSTREAM_MODELS


def _smoke_search_kwargs():
    """Smoke-only hybrid-search shrink, gated on KIRBY_SMOKE_N.

    Caps the find_best_hybrid pool to one small budget + fast tier for local
    smoke runs (default is minutes of compute).
    No-op when KIRBY_SMOKE_N is unset.
    """
    if not os.environ.get('KIRBY_SMOKE_N', '').strip():
        return {}
    # speed is now accepted but ignored (the former tier dict was removed; defaults
    # live in the DEFAULT_* module constants). Smoke shrink is via tiny budgets
    # only. internal_evaluators=['rf'] cuts the search ~5x (DEFAULT_INTERNAL_EVALUATORS
    # is match/rf/svm/lightgbm/mlp); rf alone still exercises the hybrid pipeline.
    # Also drop importance to lightgbm_gain (cheap): DEFAULT_IMPORTANCE_METHODS runs
    # deeplift+treeshap, which compute SHAP over the (here 5600-dim) features
    # per fold and dominate the smoke wall time. lightgbm_gain still produces a
    # real per-rep importance ranking so the hybrid pipeline is exercised.
    return {'speed': 'full', 'budgets': [25], 'internal_evaluators': ['rf'],
            'importance_methods': ['lightgbm_gain']}


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

    # Rank by score descending. Use kind='stable' so that ties are broken
    # deterministically by input order (not by quicksort's data-dependent
    # tie-breaking, which can leak label information when scores are coarse
    # — e.g. binary reps or LLR sums collapsing to a handful of values).
    order = np.argsort(-y_score, kind='stable')
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

def _fold_splits(labels, groups, n_folds):
    """The one fold partition every arm of this file scores on.

    Built once and handed to both the per-representation loop and the per-fold
    hybrid arm, so the two can be subtracted fold by fold. StratifiedGroupKFold
    reads only the number of rows off X, so passing a different representation
    would give the same partition anyway; passing the same list removes the
    question.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if groups is None or len(np.unique(groups)) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds}-fold group split: "
            f"groups={'None' if groups is None else len(np.unique(groups))} unique")
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    return list(cv.split(np.zeros((len(labels), 1)), labels, groups=groups))


def evaluate_rep_enrichment(X, labels, groups, model, n_folds,
                            fold_splits=None):
    """Evaluate one (rep, model) for enrichment via stratified CV.

    For each fold: train → predict scores on hold-out → compute enrichment.
    Returns dict with enrichment metrics (mean/std across folds) + AUROC, plus
    a fold-index-aligned 'enrich_<frac>_per_fold' list per fraction. A fold
    whose fit raised holds None in that list rather than being dropped, which is
    what lets the per-fold hybrid arm be subtracted from this representation on
    the SAME held-out fold.

    `fold_splits` is the shared partition from _fold_splits; when omitted the
    same partition is rebuilt here.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    splits = (list(fold_splits) if fold_splits is not None
              else _fold_splits(labels, groups, n_folds))

    is_binary = _is_binary_rep(X)
    fold_enrichments = []
    auroc_scores = []
    acc_scores = []
    n_folds_attempted = len(splits)
    n_folds_completed = 0

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
        except Exception as _fit_err:
            print(
                f"[WARN] Fold fit failed (no score for this fold): {_fit_err!r}",
                flush=True,
            )
            # Hold the fold's place so the per-fold lists below stay aligned
            # with the fold partition. Dropping it here used to shift every
            # later fold's score up one slot.
            fold_enrichments.append(None)
            continue
        n_folds_completed += 1

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

    if not any(fe is not None for fe in fold_enrichments):
        return None

    if n_folds_completed < n_folds_attempted:
        print(
            f"[WARN] Only {n_folds_completed}/{n_folds_attempted} folds completed "
            f"(fit failures dropped silently before P1-fix); results may be biased.",
            flush=True,
        )

    # Aggregate across folds
    result = {
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
        'n_folds_completed': int(n_folds_completed),
        'n_folds_attempted': int(n_folds_attempted),
    }
    if auroc_scores:
        result['auroc'] = {'mean': np.mean(auroc_scores), 'std': np.std(auroc_scores)}

    # Enrichment metrics
    for frac in ENRICHMENT_FRACTIONS:
        key = f'enrich_{frac}'
        vals = [fe[key] for fe in fold_enrichments if fe is not None and key in fe]
        if vals:
            result[key] = {'mean': np.mean(vals), 'std': np.std(vals)}
        # Fold-index-aligned held-out enrichment, one entry per fold of the
        # shared partition. None = this fold produced no score.
        result[f'{key}_per_fold'] = [
            (float(fe[key]) if fe is not None and key in fe else None)
            for fe in fold_enrichments
        ]

        ef_key = f'ef_{frac}'
        ef_vals = [fe[ef_key] for fe in fold_enrichments
                   if fe is not None and ef_key in fe]
        if ef_vals:
            result[ef_key] = {'mean': np.mean(ef_vals), 'std': np.std(ef_vals)}

    result['base_rate'] = np.mean([fe['base_rate'] for fe in fold_enrichments
                                   if fe is not None])
    return result


def _model_scores(m, X_val):
    """Ranking scores for the held-out rows, highest = most likely binder."""
    if hasattr(m, 'decision_function'):
        return m.decision_function(X_val)
    if hasattr(m, 'predict_proba'):
        return m.predict_proba(X_val)[:, 1]
    return m.predict(X_val).astype(float)


def _evaluate_hybrid_enrichment_per_fold(
    reps, labels, groups, fold_splits, models,
    pruning_mode='default_pruning',
):
    """Build the hybrid PER FOLD and count binders in its top-ranked rows.

    This is question 3 of the file — does a hybrid beat the best single
    representation at binder enrichment. The AUROC comparison beside it does
    not answer it: AUROC ranks the whole held-out fold, enrichment counts what
    lands in the top 1, 5 and 10 percent of it.

    Each fold:
      1. find_best_hybrid on the training rows only, metric='auc'.
      2. Take that fold's selected features for the downstream model (its own
         if the search scored it, rf's otherwise).
      3. apply_feature_selection on the fold's training and held-out blocks.
      4. Fit the model on the training block, score the held-out block, and run
         compute_enrichment on those scores — the same count the single
         representations get.

    Returns (rep_results_by_model, dim_estimate, top3_first_fold), shaped like
    evaluate_rep_enrichment's output so it drops into the same table. Returns
    (None, 0, None) when every fold failed.
    """
    from kirby.find_best_hybrid import find_best_hybrid
    from kirby.hybrid import apply_feature_selection
    from experiments._shared import resolve_pruning_kwargs

    labels = np.asarray(labels, dtype=np.int64)
    per_model = {m: {'enrich': [], 'auroc': [], 'acc': []} for m in models}
    dims = []
    top3_first_fold = None

    _smoke = _smoke_search_kwargs()
    _budgets = _smoke.get('budgets', FULL_BUDGETS)
    _evals = _smoke.get('internal_evaluators', INTERNAL_EVALUATORS)
    _smoke_imp = ({'importance_methods': _smoke['importance_methods']}
                  if 'importance_methods' in _smoke else {})

    def _pad_fold():
        """Record a missing held-out score for every model on this fold.

        Without it the per-model lists lose fold alignment and the paired
        hybrid-minus-representation subtraction would compare two different
        held-out folds.
        """
        for _s in per_model.values():
            _s['enrich'].append(None)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_reps = {r: arr[train_idx] for r, arr in reps.items()}
        train_labels = labels[train_idx]
        train_groups = groups[train_idx]

        try:
            _pkw = resolve_pruning_kwargs(
                pruning_mode, len(train_labels),
                dataset_label=f'tz_her2_binary/fold{fold_idx}')
            # Per-fold training subsets do not align with the full-data
            # precompute cache, so each fold gets its own 'general'-tier
            # namespace via condition_suffix (same template as
            # opig_kingsmith_component_splits and opig_ginkgo_developability).
            fold_search = find_best_hybrid(
                train_reps, train_labels,
                groups=train_groups,
                augmentations=None,
                augmentation_strategy='none',
                budgets=_budgets,
                metric='auc',
                task_type='classification',
                internal_evaluators=_evals,
                return_fitted=True,
                cache=True,
                dataset_name='tz_her2_binary',
                tier='general',
                condition_suffix=f'enrich_fold{fold_idx}',
                adaptive_pruning=0.25,
                **_smoke_imp,
                **_pkw,
            )
        except Exception as e:
            print(f"    fold {fold_idx}: find_best_hybrid failed: {e}")
            _pad_fold()
            continue

        if fold_idx == 0 and fold_search is not None:
            top3_first_fold = extract_top3_base_reps(fold_search)

        fitted_all = (getattr(fold_search, 'fitted_results', None) or {}
                      if fold_search is not None else {})
        if not fitted_all:
            print(f"    fold {fold_idx}: no fitted result; skipping fold")
            _pad_fold()
            continue

        val_reps = {r: arr[val_idx] for r, arr in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]
        fold_dim_recorded = False

        for model_name in models:
            # Each downstream model uses the recipe the search picked for it.
            # A model the search did not score falls back to rf's recipe, and
            # to whatever single recipe exists if rf is not among them.
            fitted = (fitted_all.get(model_name) or fitted_all.get('rf')
                      or fitted_all[sorted(fitted_all)[0]])
            try:
                fi = {r: {'selected_indices': idx}
                      for r, idx in fitted.selected_indices.items()}
                train_hybrid = np.asarray(
                    apply_feature_selection(train_reps, fi), dtype=np.float64)
                val_hybrid = np.asarray(
                    apply_feature_selection(val_reps, fi), dtype=np.float64)
            except Exception as e:
                print(f"    fold {fold_idx} {model_name}: hybrid assembly "
                      f"failed: {e}")
                per_model[model_name]['enrich'].append(None)
                continue

            if not fold_dim_recorded:
                dims.append(train_hybrid.shape[1])
                fold_dim_recorded = True

            if not _is_binary_rep(train_hybrid):
                scaler = StandardScaler()
                train_hybrid_s = scaler.fit_transform(train_hybrid)
                val_hybrid_s = scaler.transform(val_hybrid)
            else:
                train_hybrid_s, val_hybrid_s = train_hybrid, val_hybrid

            try:
                m = clone(models[model_name]())
                m.fit(train_hybrid_s, y_train)
                scores = _model_scores(m, val_hybrid_s)
                preds = m.predict(val_hybrid_s)
            except Exception as e:
                print(f"    fold {fold_idx} {model_name}: fit failed: {e}")
                per_model[model_name]['enrich'].append(None)
                continue

            per_model[model_name]['enrich'].append(
                compute_enrichment(y_val, scores))
            per_model[model_name]['acc'].append(accuracy_score(y_val, preds))
            try:
                per_model[model_name]['auroc'].append(
                    roc_auc_score(y_val, scores))
            except Exception:
                pass

    if not dims:
        return None, 0, top3_first_fold

    rep_results = {}
    for model_name, s in per_model.items():
        folds = s['enrich']
        if not any(fe is not None for fe in folds):
            continue
        res = {
            'n_folds_completed': int(sum(fe is not None for fe in folds)),
            'n_folds_attempted': int(len(fold_splits)),
        }
        if s['acc']:
            res['accuracy'] = {'mean': np.mean(s['acc']),
                               'std': np.std(s['acc'])}
        if s['auroc']:
            res['auroc'] = {'mean': np.mean(s['auroc']),
                            'std': np.std(s['auroc'])}
        for frac in ENRICHMENT_FRACTIONS:
            key = f'enrich_{frac}'
            vals = [fe[key] for fe in folds if fe is not None and key in fe]
            if vals:
                res[key] = {'mean': np.mean(vals), 'std': np.std(vals)}
            res[f'{key}_per_fold'] = [
                (float(fe[key]) if fe is not None and key in fe else None)
                for fe in folds
            ]
            ef_key = f'ef_{frac}'
            ef_vals = [fe[ef_key] for fe in folds
                       if fe is not None and ef_key in fe]
            if ef_vals:
                res[ef_key] = {'mean': np.mean(ef_vals),
                               'std': np.std(ef_vals)}
        res['base_rate'] = np.mean([fe['base_rate'] for fe in folds
                                    if fe is not None])
        rep_results[model_name] = res

    return rep_results, int(np.mean(dims)), top3_first_fold


def _paired_enrichment_delta(results):
    """Hybrid minus best single representation, on the SAME held-out fold.

    One block per enrichment fraction, keyed by downstream model. Both arms are
    re-picked on the other folds, so neither is a maximum taken on the fold it
    is scored on. Empty when the hybrid arm produced nothing.

    Enrichment is a fraction of the top-ranked held-out rows that are binders,
    so higher is better; it is not a BUILTIN_METRICS entry, and metric_name is
    left off for that reason.
    """
    hybrid = results.get('kirby_hybrid', {}).get('models', {})
    if not hybrid:
        return {}

    out = {}
    for frac in ENRICHMENT_FRACTIONS:
        key = f'{frac}'
        fold_key = f'enrich_{frac}_per_fold'
        candidates_by_model = {}
        baselines_by_model = {}
        for model_name, summary in hybrid.items():
            folds = summary.get(fold_key)
            if not folds:
                continue
            candidates_by_model[model_name] = [
                {'name': 'kirby_hybrid', 'fold_scores': list(folds)}]
        for rep_name, rep_data in results.items():
            if rep_name == 'kirby_hybrid':
                continue
            for model_name, summary in rep_data.get('models', {}).items():
                folds = summary.get(fold_key)
                if not folds:
                    continue
                baselines_by_model.setdefault(
                    model_name, {})[rep_name] = list(folds)
        if not candidates_by_model or not baselines_by_model:
            continue
        out[f'enrich_{key}'] = honest_rescore(
            candidates_by_model, baselines_by_model, lower_is_better=False)
    return out


def evaluate_dataset(rep_names, n_folds, pruning_mode='default_pruning'):
    """Evaluate all reps × models on tz_her2_binary for enrichment."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: tz_her2_binary (Trastuzumab-HER2 binding)")
    print(f"{'=' * 60}")

    data = load_dataset('tz_her2_binary')
    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    n = len(labels)
    n_binders = int((labels == 1).sum())
    print(f"  {n} samples, {n_binders} binders ({100*n_binders/n:.1f}%), "
          f"{n - n_binders} non-binders ({100*(n-n_binders)/n:.1f}%)")

    # Use registry-provided groups (edit-distance buckets per
    # DATASET_REGISTRY['tz_her2_binary']). Earlier code rebuilt groups by
    # sequence identity, which collapsed Trastuzumab CDRH3 variants into
    # ~N singletons and degenerated StratifiedGroupKFold to stratified
    # random (P0-2 fix).
    groups = data.get('groups')
    if groups is None:
        # Fall back to identity groups only when registry didn't supply them.
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
        mutations=data.get('mutations'),
    )

    if not reps:
        print("  No reps, aborting")
        return None

    # Hybrid search. No `kirby_hybrid` matrix is materialised for the
    # supervised loop: the earlier flow ran the search on the full label
    # vector and re-evaluated it under the same StratifiedGroupKFold, which
    # carries the per-representation `selected_indices` across folds (P0 fix).
    #
    # The number reported for the hybrid is the leave-one-fold-out re-score
    # (honest_block below), not the search's own best mean AUC. That best mean
    # was the maximum over every allocation method, importance method,
    # evaluator and budget, taken on the same folds it was measured on, while
    # every single representation beside it was a mean over held-out folds.
    # Changed 2026-09-04, split audit.
    # Augmentation config matches tz_her2_binary in DATASET_REPS:
    # augmentation = [] (no augmentation reps for antibody classification)
    hybrid_search_meta = None
    honest_block = None
    honest_result = None
    try:
        from experiments._shared import (resolve_pruning_kwargs,
                                          get_best_search_for_downstream)
        _pkw = resolve_pruning_kwargs(pruning_mode, len(labels),
                                       dataset_label='tz_her2_binary')
        _smoke = _smoke_search_kwargs()
        search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=None,
            metric='auc',
            task_type='classification',
            dataset_name='tz_her2_binary',
            return_fitted=True,
            adaptive_pruning=0.25,
            **{k: v for k, v in _smoke.items() if k != 'speed'},
            **_pkw,
        )
        if search is not None:
            top3 = extract_top3_base_reps(search)
            if top3:
                print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            # Per-fold scores for every searched recipe and every baseline
            # representation, then the leave-one-fold-out re-score: on each
            # held-out fold both arms pick their winner from the other folds
            # only, so the hybrid AUROC is measured the same way as the
            # per-representation AUROCs it sits beside.
            honest_block = build_honest_rescore_input(search)
            # Fold 0 is dropped from BOTH arms. The search filtered its own
            # recipe list using fold 0's scores, so fold 0's hybrid score is
            # read off recipes fold 0 helped keep, while every single
            # representation passed through no such filter. hr_exclude_folds
            # reads which folds that was off the block.
            _excluded = hr_exclude_folds(honest_block)
            honest_result = honest_rescore(
                honest_block['candidates_by_model'],
                honest_block['baselines_by_model'],
                honest_block['lower_is_better'],
                metric_name=honest_block.get('metric_name'),
                exclude_folds=_excluded,
            )
            if _excluded:
                print(f"  honest re-score drops fold(s) {_excluded} from both "
                      f"arms (the search filtered its recipe list on them)")
            for mn, r in sorted(honest_result.items()):
                flag = ' [single-fold: NOT honest]' if r['single_fold'] else ''
                # delta_mean_of_folds is the paired per-fold difference. r['delta']
                # is the difference of the two arms' means and is not printed
                # here; the two diverge whenever either arm drops a fold.
                print(f"  honest × {mn:8s}  hybrid={r['hybrid_honest']:.4f}  "
                      f"best_single={r['best_single_honest']:.4f}  "
                      f"delta_mean_of_folds={r['delta_mean_of_folds']:+.4f} "
                      f"over {r['n_paired_folds']} paired folds{flag}")
        if _meta is not None and _meta.get('score_mean') is not None:
            hybrid_search_meta = {
                # The search's own winning mean, kept ONLY as a search
                # diagnostic. It is a maximum over recipes taken on the folds
                # that scored them; do not print it beside a held-out AUROC.
                'search_best_cv_auc': float(_meta['score_mean']),
                'tier': _meta.get('tier'),
                'aug_condition': _meta.get('aug_condition'),
                'note': (
                    'search_best_cv_auc is the best recipe mean over the '
                    'search folds, not a held-out score — it is the maximum '
                    'over every allocation method, importance method, '
                    'evaluator and budget, measured on the folds that chose '
                    'it. The hybrid number to report is in honest_comparison, '
                    'a leave-one-fold-out re-score against the same baseline '
                    'representations (2026-09-04, split audit). The key was '
                    'renamed from rf_cv_auc, so a reader that asks for the old '
                    'name gets nothing rather than this number.'
                ),
            }
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
            # LLR reps return [sum_llr, mean_llr, min_llr] per sequence
            # (see src/kirby/representations/mutation.py, around lines
            # 551-553 / 581 for ESM-C / AntiBERTy respectively). Averaging
            # across the 3 columns was a P0 bug: sum_llr scales with the
            # number of mutations, dominates the mean, and effectively
            # ranks sequences by mutation count rather than per-residue
            # binder plausibility. Use column 1 (mean_llr) — length-
            # invariant and matches the paper's per-residue protocol.
            scores = X[:, 1]
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

    # Supervised evaluation (KIRBy's extension beyond paper).
    # NOTE: this is a second pass over `reps` after the zero-shot loop
    # above. The two passes are kept separate because the zero-shot pass
    # has no model factory and the supervised pass runs |models| inner
    # iterations per rep; merging them is mechanical but not minimal-diff.
    models = get_model_factories(MODELS_LIST, 'classification', n_samples=n)
    # One fold partition, shared by the per-representation loop and the per-fold
    # hybrid arm, so their held-out enrichments can be subtracted fold by fold.
    try:
        fold_splits = _fold_splits(labels, groups, n_folds)
    except ValueError as e:
        print(f"  Cannot split: {e}")
        return None
    print(f"\n  Supervised evaluation ({len(reps)} reps × {len(models)} models × {n_folds}-fold CV)...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                res = evaluate_rep_enrichment(X, labels, groups, factory(),
                                              n_folds, fold_splits=fold_splits)
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

    # ===== Per-fold hybrid, counted the same way as the representations =====
    # The whole-row search above answers the AUROC question. This arm answers
    # the enrichment one: a fresh search on each fold's training rows, its
    # features applied to that fold's held-out rows, and the same top-1/5/10
    # percent binder count.
    print(f"\n  Per-fold hybrid enrichment ({n_folds} folds, train-only search)...")
    try:
        hybrid_enrich, hybrid_dim, hybrid_top3 = (
            _evaluate_hybrid_enrichment_per_fold(
                reps, labels, groups, fold_splits, models,
                pruning_mode=pruning_mode))
    except Exception as e:
        print(f"  Per-fold hybrid enrichment failed: {e}")
        hybrid_enrich, hybrid_dim, hybrid_top3 = None, 0, None

    if hybrid_top3:
        print(f"  Top-3 base reps (fold 0): "
              f"{', '.join(f'{nm}={sc:.4f}' for nm, sc in hybrid_top3)}")
    if hybrid_enrich:
        results['kirby_hybrid'] = {'dim': hybrid_dim, 'models': hybrid_enrich}
        for model_name, res in sorted(hybrid_enrich.items()):
            e1 = res.get('enrich_0.01', {}).get('mean', float('nan'))
            auroc = res.get('auroc', {}).get('mean', float('nan'))
            print(f"    {'kirby_hybrid':25s} {model_name:8s}  "
                  f"E@1%={e1:.4f}  AUROC={auroc:.4f}  ***")

    return {
        'n_samples': n,
        'n_binders': n_binders,
        'base_rate': n_binders / n,
        'zero_shot': zero_shot_results,  # paper's protocol (no training)
        'supervised': results,            # KIRBy's extension (trained classifiers)
        'hybrid_search': hybrid_search_meta,  # search diagnostic; NOT held-out
        # The held-out hybrid-vs-best-single comparison, per downstream model,
        # plus the raw per-fold scores it was computed from.
        'honest_rescore': honest_block,
        'honest_comparison': honest_result,
        # Hybrid minus best single representation on the same held-out fold, in
        # binder enrichment, one block per top fraction. This is the file's
        # question 3; honest_comparison above is in AUROC and does not answer it.
        'enrichment_comparison': _paired_enrichment_delta(results),
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
            # Stored key is f'enrich_{0.10}' -> 'enrich_0.1' (Python drops the
            # trailing zero), so look up 'enrich_0.1' not 'enrich_0.10'.
            e10 = metrics.get('enrich_0.1', {}).get('mean', float('nan'))
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

    print(f"\n  Summary (Enrichment @ top-1%):")
    if best_indiv:
        bi = best_indiv[0]
        print(f"    Best individual: {bi[0]} ({bi[1]}) E@1% = {bi[2]:.4f}")

    # Hybrid number: the leave-one-fold-out re-score. On each held-out fold the
    # hybrid recipe and the best single representation are both picked from the
    # other folds, so the two arms pay the same tax. AUROC, so it is not
    # rankable against the E@1% column above; printed separately.
    honest = ds_results.get('honest_comparison') or {}
    if honest:
        _dropped = sorted({f for r in honest.values()
                           for f in (r.get('excluded_folds') or [])})
        drop_note = (f", fold(s) {_dropped} dropped from both arms"
                     if _dropped else "")
        print(f"\n  Hybrid vs best single representation "
              f"(leave-one-fold-out, AUROC{drop_note}):")
        # Delta is delta_mean_of_folds: the per-fold difference, averaged over
        # the folds where both arms scored. Folds is that count.
        print(f"    {'Model':10s} {'Hybrid':>10s} {'BestSingle':>12s} "
              f"{'Delta':>10s} {'Folds':>6s}")
        print(f"    {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*6}")
        for model_name in sorted(honest):
            r = honest[model_name]
            h, b = r['hybrid_honest'], r['best_single_honest']
            d = r['delta_mean_of_folds']
            h_s = f"{h:.4f}" if not np.isnan(h) else "—"
            b_s = f"{b:.4f}" if not np.isnan(b) else "—"
            d_s = f"{d:+.4f}" if not np.isnan(d) else "—"
            flag = ' (single-fold: NOT honest)' if r['single_fold'] else ''
            print(f"    {model_name:10s} {h_s:>10s} {b_s:>12s} {d_s:>10s} "
                  f"{r['n_paired_folds']:>6d}{flag}")

    # Question 3: the same subtraction in binder enrichment rather than AUROC.
    enrich_cmp = ds_results.get('enrichment_comparison') or {}
    if enrich_cmp:
        print(f"\n  Hybrid vs best single representation "
              f"(leave-one-fold-out, binder enrichment):")
        print(f"    Hybrid and BestSingle are the fraction of the top-ranked "
              f"held-out rows that are binders. Delta is the per-fold "
              f"difference, averaged over the folds where both arms scored.")
        print(f"    {'Top':>6s} {'Model':10s} {'Hybrid':>10s} "
              f"{'BestSingle':>12s} {'Delta':>10s} {'Folds':>6s}")
        print(f"    {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*6}")
        for frac in ENRICHMENT_FRACTIONS:
            block = enrich_cmp.get(f'enrich_{frac}') or {}
            top_s = f"{frac * 100:g}%"
            for model_name in sorted(block):
                r = block[model_name]
                h, b = r['hybrid_honest'], r['best_single_honest']
                d = r['delta_mean_of_folds']
                h_s = f"{h:.4f}" if not np.isnan(h) else "—"
                b_s = f"{b:.4f}" if not np.isnan(b) else "—"
                d_s = f"{d:+.4f}" if not np.isnan(d) else "—"
                print(f"    {top_s:>6s} {model_name:10s} {h_s:>10s} "
                      f"{b_s:>12s} {d_s:>10s} {r['n_paired_folds']:>6d}")


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
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()

    apply_distance_metric_arg(args)
    rep_names = args.reps.split(',') if args.reps else _default_reps()
    # Validate rep names against the registries so a typo doesn't silently
    # drop a rep (P1-8 fix).
    from kirby.representations.antibody import ANTIBODY_REPS
    from kirby.representations.mutation import MUTATION_REPS
    _known = set(ANTIBODY_REPS) | set(MUTATION_REPS)
    _unknown = [r for r in rep_names if r not in _known]
    if _unknown:
        print(f"  WARNING: unknown rep name(s) {sorted(_unknown)} — "
              f"will be silently skipped by compute_or_load_reps")

    print("=" * 70)
    print("TRASTUZUMAB BINDER ENRICHMENT EXPERIMENT (OPIG B1)")
    print(f"  Paper: Chinery et al. 2024")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Enrichment fractions: {ENRICHMENT_FRACTIONS}")
    print("=" * 70)

    ds_results = evaluate_dataset(rep_names, args.n_folds,
                                    pruning_mode=args.pruning_mode)
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
