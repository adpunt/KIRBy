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
        --n-folds 2 --reps blosum,ablang2,antiberty

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
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories,
    fold_reselection, build_fold_hybrid,
    fold_hybrid_scoring_kwargs, fold_standard_selection,
    STANDARD_N_BASE, STANDARD_N_HYBRIDS,
)
from experiments.decided_grid import decided_models, decided_aug_conditions
# 2026-10-05: 0/1 features pass through, continuous features are standardised,
# in every place this file scales for itself (answer 17). It used to use
# StandardScaler on a matrix unless EVERY value in it was 0 or 1, so a hybrid
# holding ecfp4 bits beside pdv numbers standardised the bits too.
from kirby.scaling import FeatureScaler

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

# DECIDED 2026-10-06 (Adelaide delegated): these three are scored on their own
# -- in the zero-shot ranking arm and in the table of single representations --
# and are NOT in the pool the per-fold hybrid search picks features from. The
# search runs on the dataset's approved pool, the same 10 representations every
# other experiment on tz_her2_binary searches. A consequence: none of the three
# can be the best single representation of a fold in the per-fold AUROC table
# either, because that representation is the top-ranked one of the same search.
SEARCH_EXCLUDED_REPS = tuple(REP_EXCEPTIONS)


def _default_reps():
    from experiments._shared import approved_rep_pool
    return approved_rep_pool('tz_her2_binary', extra=REP_EXCEPTIONS)

ENRICHMENT_FRACTIONS = [0.01, 0.05, 0.10]  # top-1%, top-5%, top-10%
# Note: at n_val ~= 1000 (5-fold on ~5K), top-1% = ~10 samples — high
# variance. Reported but interpret with care; E@5% and E@10% are more stable.

DATASET = 'tz_her2_binary'

# The two downstream models DECISIONS.md Step 1 decided for tz_her2_binary
# (2026-10-03). Read from the record, not written down here, so the file cannot
# drift from it. This used to be DOWNSTREAM_MODELS -- rf, xgboost, svm, dnn,
# lightgbm -- and three of those five are not approved for this dataset.
#
# knn is NOT in this list. Step 9 runs knn inside the distance SEARCH, which the
# per-fold distance search below asks the record for; no knn is fitted on the
# held-out rows.
MODELS_LIST = decided_models(DATASET, 'general')

# The hybrids scored per downstream model per outer fold, 2026-10-05: the 2
# best of that fold's own general search. There is no distance hybrid any more.
#
# ⛔ What this replaces, and why. Until 2026-10-05 the file kept three hybrids
# per model -- the top 2 of the general search plus the top 1 of a second
# search run with distance-tier settings (the owner's set of 2026-10-02).
# Adelaide's answer 1 of 2026-10-05: a distance hybrid is built only where the
# experiment measures something with neighbours or distances. This file counts
# how many of the top-ranked held-out molecules a trained classifier gets right;
# no neighbour list and no distance is computed anywhere in it. So the distance
# search and its slot are gone, and the two general hybrids are what is scored
# (answer in handoff 2.5, group C).
#
# 'general1' keeps the existing 'kirby_hybrid' result key.
RULE_SLOTS = ['general1', 'general2']

# Both counts come from the shared selection rule (2 base representations and 2
# hybrids per model per fold), not from numbers written down here.
N_BASE_REPS = STANDARD_N_BASE
N_HYBRIDS = STANDARD_N_HYBRIDS


def _slot_key(slot):
    """Result key for one hybrid slot. general1 keeps the old name."""
    return 'kirby_hybrid' if slot == 'general1' else f'kirby_hybrid__{slot}'


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


# 2026-10-05: the whole-matrix 0/1 test that was here (_is_binary_rep) is gone.
# It decided whether to scale a representation by looking at every value in it
# at once, so one continuous feature made the 0/1 features standardise with it.
# FeatureScaler decides per feature instead (answer 17).


# =============================================================================
# EVALUATION
# =============================================================================

def _fold_splits(reps, labels, groups, n_folds, seed=42):
    """The one fold partition every arm of this file scores on.

    Built once and handed to both the per-representation loop and the per-fold
    hybrid arm, so the two can be subtracted fold by fold.

    2026-10-05: this is now kirby.find_best_hybrid.nested_cv_outer_folds, the
    same function the per-fold searches are keyed on, so a search filed under
    this fold really ran on the training rows scored here. It used to be its own
    StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42) built
    in this file. Two consequences: a fold count of 1 is a single grouped
    held-out split rather than an error, and the fold numbering cannot drift
    from the per-fold search namespace (ocv{n}f{i}s{seed}).
    """
    from kirby.find_best_hybrid import nested_cv_outer_folds
    labels = np.asarray(labels, dtype=np.int64)
    if groups is None or len(np.unique(groups)) < max(2, n_folds):
        raise ValueError(
            f"Cannot create {n_folds}-fold group split: "
            f"groups={'None' if groups is None else len(np.unique(groups))} unique")
    return nested_cv_outer_folds(reps, labels, groups, n_folds, seed,
                                 'classification')


def _spread(vals, n_folds):
    """The spread of one number over the outer folds, or the no-spread note.

    2026-10-05: with a single outer fold there is nothing to take a spread
    over, and np.std of one number is 0.0, which reads as agreement between
    folds that were never measured. One fold returns the string instead
    (handoff 2.4 step 14).
    """
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if n_folds == 1 or len(vals) == 1:
        return 'one fold, no spread'
    return float(np.std(vals))


def evaluate_rep_enrichment(X, labels, groups, model, n_folds,
                            fold_splits):
    """Evaluate one (rep, model) for enrichment via stratified CV.

    For each fold: train → predict scores on hold-out → compute enrichment.
    Returns dict with enrichment metrics (mean/std across folds) + AUROC, plus
    a fold-index-aligned 'enrich_<frac>_per_fold' list per fraction. A fold
    whose fit raised holds None in that list rather than being dropped, which is
    what lets the per-fold hybrid arm be subtracted from this representation on
    the SAME held-out fold.

    `fold_splits` is the shared partition from _fold_splits.
    """
    # No dtype conversion: a representation arrives as float32 and stays
    # float32 (answer 18, 2026-10-05). It used to be cast to float64 here,
    # which gives different tree scores from the search's own (dirty-run bug 5).
    X = np.asarray(X)
    labels = np.asarray(labels, dtype=np.int64)

    splits = list(fold_splits)

    fold_enrichments = []
    # Held-out AUROC per fold, one entry per fold of the shared partition, None
    # where the fit failed or AUROC was undefined. The per-fold AUROC table
    # (Adelaide, answer 7, 2026-10-05) reads these: it needs fold k's number,
    # not the mean, so that the hybrid can be set against the single
    # representation on the SAME held-out fold.
    auroc_per_fold = []
    auroc_scores = []
    acc_scores = []
    n_folds_attempted = len(splits)
    n_folds_completed = 0

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Every feature whose training values are all 0 or 1 passes through;
        # every other feature is standardised (answer 17, 2026-10-05). The test
        # used to be on the whole matrix at once: one continuous feature in a
        # 0/1 representation standardised the 0/1 features with it.
        scaler = FeatureScaler()
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
            auroc_per_fold.append(None)
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
            _a = float(roc_auc_score(y_val, scores))
        except Exception:
            _a = None
        auroc_per_fold.append(_a)
        if _a is not None:
            auroc_scores.append(_a)

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
        'accuracy': {'mean': np.mean(acc_scores),
                     'std': _spread(acc_scores, n_folds_attempted)},
        'n_folds_completed': int(n_folds_completed),
        'n_folds_attempted': int(n_folds_attempted),
        # The metric these numbers are in, and which way is better. AUROC and
        # the binder fraction are both higher-is-better.
        'metric_name': 'auroc',
        'lower_is_better': False,
    }
    if auroc_scores:
        result['auroc'] = {'mean': np.mean(auroc_scores),
                           'std': _spread(auroc_scores, n_folds_attempted)}
    result['auroc_per_fold'] = list(auroc_per_fold)

    # Enrichment metrics
    for frac in ENRICHMENT_FRACTIONS:
        key = f'enrich_{frac}'
        vals = [fe[key] for fe in fold_enrichments if fe is not None and key in fe]
        if vals:
            result[key] = {'mean': np.mean(vals),
                           'std': _spread(vals, n_folds_attempted)}
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
            result[ef_key] = {'mean': np.mean(ef_vals),
                              'std': _spread(ef_vals, n_folds_attempted)}

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


def _fit_and_count(train_X, val_X, y_train, y_val, factory, label):
    """Fit on the training block, count binders in the held-out block's top
    fractions, and return (one fold's scores, error text).

    2026-10-05: the error text is returned as well. It used to return None
    alone, and the caller stored that None as the slot's score, where the
    summary read it as "the search found no recipe for this model on this
    fold". A recipe existed and was built; the fit on the held-out rows is what
    raised, and the text only reached stdout.

    0/1 features pass through the scaler, continuous features are standardised
    (answer 17, 2026-10-05). No dtype conversion: the features arrive float32
    from the representations and stay float32 (answer 18).
    """
    train_X = np.asarray(train_X)
    val_X = np.asarray(val_X)
    scaler = FeatureScaler()
    train_s = scaler.fit_transform(train_X)
    val_s = scaler.transform(val_X)
    try:
        m = clone(factory())
        m.fit(train_s, y_train)
        scores = _model_scores(m, val_s)
        preds = m.predict(val_s)
    except Exception as e:
        print(f"    {label}: fit failed: {e}")
        return None, f'{type(e).__name__}: {e}'
    try:
        auroc = float(roc_auc_score(y_val, scores))
    except Exception:
        auroc = None
    return {'enrich': compute_enrichment(y_val, scores),
            'acc': float(accuracy_score(y_val, preds)),
            'auroc': auroc,
            'n_features': int(train_X.shape[1])}, None


def _hybrid_fold_block(fold_idx, train_idx, val_idx, reps, labels, groups,
                       n_folds, models):
    """One outer fold: search its training rows, build the 2 hybrids per model,
    and count binders in the top-ranked held-out rows.

    `reps` is the search pool the caller passes: since 2026-10-06 the approved
    pool without esmc_llr, antiberty_llr and esm2, which are scored alone only
    (SEARCH_EXCLUDED_REPS).

    This is question 3 of the file — does a hybrid beat the best single
    representation at binder enrichment. The AUROC comparison beside it does
    not answer it: AUROC ranks the whole held-out fold, enrichment counts what
    lands in the top 1, 5 and 10 percent of it.

    2026-10-05, what changed and why:
      * The two searches this function used to run itself (one general, one with
        distance-tier settings, both through find_best_hybrid directly) are now
        one call to fold_standard_selection, which runs the dataset's decided
        grid on this fold's training rows and hands back the 2 best base
        representations and the 2 best hybrids per decided model. The distance
        search is gone: no neighbour list and no distance is computed in this
        file (answer 1).
      * Pruning is passed nowhere. The two old calls passed adaptive_pruning
        0.25 plus a schedule from --pruning-mode, and the schedule's per-axis
        votes deleted members of the decided importance, evaluator,
        representation-count and allocation axes after folds 0 and 1 of each
        search, so which part of the decided grid survived was settled inside
        each run.
      * A second hybrid whose selected features are identical to the first's is
        skipped inside fold_standard_selection (distinct_features=True) and
        named in 'hybrids_skipped_identical_features'.

    Returns one JSON-safe dict for this fold, so it is stored on its own as
    soon as the fold finishes (dirty-run bug 12: the file used to store every
    fold's scores in one unit and lose all of them when a run was killed).
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    labels = np.asarray(labels, dtype=np.int64)
    # metric='auc' below, so this is the function the search itself scored with
    # (BUILTIN_METRICS['auc'] = (_safe_roc_auc, False)). Passed to
    # build_fold_hybrid so a recipe whose internal evaluator is 'match'
    # resolves to the downstream model it was ranked for. Without it
    # create_hybrid's no-model branch resolves 'match' to lightgbm whatever the
    # recipe's model.
    _metric_fn, _metric_lib = BUILTIN_METRICS['auc']

    # 'fit_failures' (2026-10-05): one row per (slot, model) whose hybrid was
    # built and whose fit on the held-out rows raised, with the error. These
    # rows are kept apart from the folds where the search ranked no recipe, so
    # the summary cannot report a failed fit as a missing recipe.
    out = {'fold': int(fold_idx), 'scores': {}, 'recipes': [],
           'base_reps': [], 'missing': [], 'fit_failures': [],
           'hybrids_skipped_identical_features': []}

    train_reps = {r: arr[train_idx] for r, arr in reps.items()}
    val_reps = {r: arr[val_idx] for r, arr in reps.items()}
    train_groups = np.asarray(groups)[train_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    try:
        # Per-fold training subsets do not align with the whole-dataset
        # precompute cache, so each fold's search is filed under this fold's
        # own nested-CV namespace (ocv{n}f{i}s{seed}), which the shared helper
        # names from fold_i and n_outer_folds. The fold partition comes from
        # nested_cv_outer_folds, so the namespace and the rows agree.
        #
        # Every axis comes from the record: models, budgets, importance
        # methods, allocation methods, internal evaluators, representation
        # counts and the augmentation condition (DECISIONS.md, via the decided
        # grid). None is passed for all of them so the decided lists apply.
        sel = fold_standard_selection(
            reps, labels, groups,
            train_idx=train_idx, fold_i=fold_idx, n_outer_folds=n_folds,
            dataset_name=DATASET, models=list(models), distance=False,
            augmentations=None, task_type='classification', metric='auc',
            n_base=N_BASE_REPS, n_hybrids=N_HYBRIDS, distinct_features=True)
    except Exception as e:
        # A failed per-fold search is recorded with the dataset, the fold and
        # the error, and this fold's arms are skipped. Nothing is filled in
        # from another fold or another model.
        print(f"    fold {fold_idx}: per-fold selection failed: {e}")
        out['missing'].append({'dataset': DATASET, 'fold': int(fold_idx),
                               'model': None,
                               'reason': f'the per-fold search raised '
                                         f'{type(e).__name__}: {e}'})
        out['scores'] = {s: {m: None for m in models} for s in RULE_SLOTS}
        return out

    for slot in RULE_SLOTS:
        out['scores'][slot] = {}

    for model_name in models:
        chosen = sel.get(model_name)
        if chosen is None:
            for slot in RULE_SLOTS:
                out['scores'][slot][model_name] = None
            out['missing'].append({
                'dataset': DATASET, 'fold': int(fold_idx),
                'model': model_name,
                'reason': 'the per-fold search scored no configuration for '
                          'this model'})
            continue

        # The 2 base representations this fold's search ranked highest for this
        # model, ranked on the training rows. Recorded with their training-row
        # score; the per-fold AUROC table below reads the first of them as the
        # best single representation for this fold, so the comparison never
        # picks a representation on the held-out numbers it is compared with.
        _bd = chosen.get('base_detail') or {}
        out['base_reps'].append({
            'fold': int(fold_idx), 'model': model_name,
            'base_reps': list(chosen.get('base') or []),
            'score_train_rows': [b.get('score')
                                 for b in (_bd.get('selected') or [])],
            'metric_name': 'auc', 'lower_is_better': False,
            'selection_scope': 'fold_train_rows'})
        if not chosen.get('base'):
            out['missing'].append({
                'dataset': DATASET, 'fold': int(fold_idx),
                'model': model_name,
                'reason': f'no base representation ranked: '
                          f'{_bd.get("shortfall_reason")}'})

        for _sk in (chosen.get('hybrids_skipped_identical_features') or []):
            out['hybrids_skipped_identical_features'].append({
                'fold': int(fold_idx), 'model': model_name,
                'recipe': _sk.get('label') if isinstance(_sk, dict) else str(_sk),
                'reason': 'its selected features are identical to a hybrid '
                          'already chosen on this fold'})

        hybrids = list(chosen.get('hybrids') or [])
        for slot_i, slot in enumerate(RULE_SLOTS):
            if slot_i >= len(hybrids):
                out['scores'][slot][model_name] = None
                out['missing'].append({
                    'dataset': DATASET, 'fold': int(fold_idx),
                    'model': model_name, 'slot': slot,
                    'reason': f'the per-fold search ranked {len(hybrids)} '
                              f'hybrid(s) for this model, so this slot has no '
                              f'recipe'})
                continue
            entry = hybrids[slot_i]
            bp = entry.get('blueprint')
            try:
                # model= and metric_fn= are what make a recipe whose internal
                # evaluator is 'match' resolve to THIS downstream model. Called
                # without them, create_hybrid's no-model branch resolved
                # 'match' to lightgbm for every model, so a recipe ranked under
                # one evaluator was rebuilt under another and filed under the
                # ranked recipe's name. 'match' is one of the four internal
                # evaluators the record keeps for this dataset.
                #
                # The recipe's representation count is resolved from
                # blueprint.rep_scores, and this blueprint comes from THIS
                # fold's own search, so those scores were measured on this
                # fold's training rows (dirty-run bug 1).
                _trh, _vah, _ = build_fold_hybrid(
                    train_reps, val_reps, y_train, bp,
                    groups=train_groups,
                    **fold_hybrid_scoring_kwargs(
                        bp, model=clone(models[model_name]()),
                        metric_fn=_metric_fn, lower_is_better=_metric_lib))
            except Exception as e:
                print(f"    fold {fold_idx} {model_name} {slot}: build "
                      f"failed: {e}")
                out['scores'][slot][model_name] = None
                out['missing'].append({
                    'dataset': DATASET, 'fold': int(fold_idx),
                    'model': model_name, 'slot': slot,
                    'reason': f'build raised {type(e).__name__}: {e}'})
                continue
            scored, fit_error = _fit_and_count(
                _trh, _vah, y_train, y_val, models[model_name],
                f'fold {fold_idx} {model_name} {slot}')
            out['scores'][slot][model_name] = scored
            if scored is None:
                # The recipe was ranked and the features were built; the fit on
                # this fold's held-out rows raised. Recorded with the dataset,
                # the fold, the model, the slot and the error (2026-10-05).
                out['fit_failures'].append({
                    'dataset': DATASET, 'fold': int(fold_idx),
                    'model': model_name, 'slot': slot,
                    'reason': f'the hybrid was built and the fit on the '
                              f'held-out rows raised {fit_error}'})
            # Every part of the recipe, named: the four axes the record decides
            # plus the representation count and the augmentation condition.
            # 'label' alone was a slash-joined string nothing could be read out
            # of.
            out['recipes'].append({
                'fold': int(fold_idx), 'model': model_name, 'slot': slot,
                'tier': 'general', 'recipe': entry.get('label'),
                'allocation_method': entry.get('allocation_method'),
                'importance_method': entry.get('importance_method'),
                'budget': entry.get('budget'),
                'internal_evaluator': entry.get('internal_evaluator'),
                'rep_count': entry.get('rep_count'),
                'aug_condition': entry.get('aug_condition'),
                'n_features': (scored or {}).get('n_features'),
                'score_train_rows': entry.get('score_mean'),
                'score_held_out': ((scored or {}).get('auroc')),
                'metric_name': 'auc', 'lower_is_better': False,
                'selection_scope': 'fold_train_rows'})
    return out


def _aggregate_hybrid_folds(fold_blocks, n_folds, model_names):
    """Turn one dict per fold into the per-slot summaries the table reads.

    2026-10-05: every per-fold list is indexed by the fold number each block
    carries ('fold'), and is n_folds long whether that fold produced a block or
    not. It used to append one entry per block that reached this function, in
    the order the blocks arrived. A fold whose whole block raised is dropped
    before this point, so with fold 0 gone the hybrid list started at fold 1
    while the single representation's list (evaluate_rep_enrichment) kept its
    None placeholder and stayed n_folds long: every printed row and every
    stored delta then subtracted fold k+1's hybrid from fold k's single
    representation under the label 'fold k'.

    A (slot, model) with no held-out score on a fold holds None at that fold's
    index, and which of the three reasons it was is kept apart:
      'no_recipe'      that fold's search ranked no recipe for this slot
      'fit_failed'     the hybrid was built and the held-out fit raised
      'fold_not_run'   the whole fold block raised before any arm was scored
    """
    by_fold = {int(blk['fold']): blk for blk in fold_blocks}
    fit_failed = {(int(r['fold']), r['model'], r['slot'])
                  for blk in fold_blocks
                  for r in (blk.get('fit_failures') or [])}

    per_slot = {}
    for slot in RULE_SLOTS:
        per_model = {m: {'enrich': [None] * n_folds,
                         'auroc': [None] * n_folds,
                         'acc': [], 'no_recipe': [], 'fit_failed': [],
                         'fold_not_run': []}
                     for m in model_names}
        for fold_idx in range(n_folds):
            blk = by_fold.get(fold_idx)
            got = ((blk['scores'].get(slot) or {}) if blk is not None else {})
            for m in model_names:
                s = per_model[m]
                row = got.get(m)
                if row is None:
                    if blk is None:
                        s['fold_not_run'].append(fold_idx)
                    elif (fold_idx, m, slot) in fit_failed:
                        s['fit_failed'].append(fold_idx)
                    else:
                        s['no_recipe'].append(fold_idx)
                    continue
                s['enrich'][fold_idx] = row['enrich']
                s['auroc'][fold_idx] = row.get('auroc')
                if row.get('acc') is not None:
                    s['acc'].append(row['acc'])
        per_slot[slot] = per_model

    dims = [r.get('n_features') for blk in fold_blocks for r in blk['recipes']
            if r.get('slot') == 'general1' and r.get('n_features')]
    return per_slot, (int(np.mean(dims)) if dims else 0)


def _summarize_enrichment(per_model, n_folds, verbose=False):
    """Per-model enrichment summary, shaped like evaluate_rep_enrichment's.

    `n_folds` is the number of outer folds attempted, which is what the spread
    over folds is reported against: with one fold there is no spread
    (2026-10-05).
    """
    rep_results = {}
    for model_name, s in per_model.items():
        folds = s['enrich']
        if not any(fe is not None for fe in folds):
            # 2026-10-05: the three reasons are printed apart. This line used
            # to say the search stored no recipe on any fold, whatever had
            # happened, including every fold's held-out fit raising on a recipe
            # that was found and built.
            if verbose:
                print(f"    {model_name}: no enrichment row -- "
                      f"{len(s['no_recipe'])} of {n_folds} folds ranked no "
                      f"recipe for it, the held-out fit raised on "
                      f"{len(s['fit_failed'])}, and "
                      f"{len(s['fold_not_run'])} fold(s) produced no block at "
                      f"all; the errors are in 'rule_missing'")
            continue
        res = {
            'n_folds_completed': int(sum(fe is not None for fe in folds)),
            'n_folds_attempted': int(n_folds),
            # Which folds this model has no recipe of its own on. Those folds
            # contribute no score rather than another model's features scored
            # under this model's name.
            'folds_without_a_recipe_for_this_model': list(s['no_recipe']),
            # Folds where a recipe was ranked and built and the fit on the
            # held-out rows raised, and folds whose whole block raised. Kept
            # apart from the line above (2026-10-05); the error text of each is
            # in 'rule_missing'.
            'folds_where_the_held_out_fit_raised': list(s['fit_failed']),
            'folds_the_hybrid_arm_did_not_run': list(s['fold_not_run']),
            'metric_name': 'auc',
            'lower_is_better': False,
            'selection_scope': 'fold_train_rows',
        }
        if s['acc']:
            res['accuracy'] = {'mean': np.mean(s['acc']),
                               'std': _spread(s['acc'], n_folds)}
        _au = [a for a in s['auroc'] if a is not None]
        if _au:
            res['auroc'] = {'mean': np.mean(_au),
                            'std': _spread(_au, n_folds)}
        # Held-out AUROC fold by fold, aligned with the fold partition, for the
        # per-fold AUROC table (answer 7, 2026-10-05).
        res['auroc_per_fold'] = list(s['auroc'])
        for frac in ENRICHMENT_FRACTIONS:
            key = f'enrich_{frac}'
            vals = [fe[key] for fe in folds if fe is not None and key in fe]
            if vals:
                res[key] = {'mean': np.mean(vals), 'std': _spread(vals, n_folds)}
            res[f'{key}_per_fold'] = [
                (float(fe[key]) if fe is not None and key in fe else None)
                for fe in folds
            ]
            ef_key = f'ef_{frac}'
            ef_vals = [fe[ef_key] for fe in folds
                       if fe is not None and ef_key in fe]
            if ef_vals:
                res[ef_key] = {'mean': np.mean(ef_vals),
                               'std': _spread(ef_vals, n_folds)}
        res['base_rate'] = np.mean([fe['base_rate'] for fe in folds
                                    if fe is not None])
        rep_results[model_name] = res
    return rep_results


def _per_fold_auroc_table(results, base_reps_per_fold, n_folds):
    """Hybrid against the best single representation, in held-out AUROC, one
    row per downstream model per outer fold (Adelaide, answer 7, 2026-10-05).

    The best single representation for a fold is the top-ranked base
    representation of THAT fold's own search, ranked on its training rows
    (base_reps_per_fold, selection_scope 'fold_train_rows'). It is never the
    representation with the best held-out AUROC, which would be picked on the
    numbers it is being compared with.

    Both AUROC numbers are the held-out rows of the same fold. The whole-dataset
    AUROC comparison this replaces read one search over all ~5,000 rows.
    """
    rows = []
    best_by_fold_model = {}
    for r in (base_reps_per_fold or []):
        reps_ranked = r.get('base_reps') or []
        if reps_ranked:
            best_by_fold_model[(int(r['fold']), r['model'])] = reps_ranked[0]

    for slot in RULE_SLOTS:
        hyb = (results.get(_slot_key(slot)) or {}).get('models') or {}
        for model_name, summary in sorted(hyb.items()):
            hyb_folds = summary.get('auroc_per_fold') or []
            for fold_idx in range(n_folds):
                h = (hyb_folds[fold_idx]
                     if fold_idx < len(hyb_folds) else None)
                rep_name = best_by_fold_model.get((fold_idx, model_name))
                b = None
                if rep_name is not None:
                    _rep_folds = ((results.get(rep_name) or {})
                                  .get('models', {}).get(model_name, {})
                                  .get('auroc_per_fold') or [])
                    b = (_rep_folds[fold_idx]
                         if fold_idx < len(_rep_folds) else None)
                rows.append({
                    'fold': int(fold_idx), 'model': model_name, 'slot': slot,
                    'hybrid_auroc_held_out': h,
                    'best_single_rep': rep_name,
                    'best_single_auroc_held_out': b,
                    'delta_hybrid_minus_single': (
                        (h - b) if (h is not None and b is not None) else None),
                    'metric_name': 'auroc', 'lower_is_better': False,
                    'best_single_chosen_on': 'fold_train_rows'})
    return rows


def _paired_enrichment_delta(results, hybrid_key='kirby_hybrid'):
    """Hybrid minus best single representation, on the SAME held-out fold.

    One block per enrichment fraction, keyed by downstream model. Both arms are
    re-picked on the other folds, so neither is a maximum taken on the fold it
    is scored on. Empty when the hybrid arm produced nothing.

    Enrichment is a fraction of the top-ranked held-out rows that are binders,
    so higher is better; it is not a BUILTIN_METRICS entry, and metric_name is
    left off for that reason.
    """
    hybrid = results.get(hybrid_key, {}).get('models', {})
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
                {'name': hybrid_key, 'fold_scores': list(folds)}]
        for rep_name, rep_data in results.items():
            # Every slot of the owner's set is a hybrid, never a single
            # representation.
            if rep_name.startswith('kirby_hybrid'):
                continue
            for model_name, summary in rep_data.get('models', {}).items():
                folds = summary.get(fold_key)
                if not folds:
                    continue
                baselines_by_model.setdefault(
                    model_name, {})[rep_name] = list(folds)
        if not candidates_by_model or not baselines_by_model:
            continue
        # No exclude_folds here, unlike evaluate_dataset's call below. There is
        # exactly one candidate per model — the hybrid rebuilt inside each fold —
        # so there is no recipe list for fold 0 to have filtered, and both arms
        # pay the same held-out tax on every fold.
        out[f'enrich_{key}'] = fold_reselection(
            candidates_by_model, baselines_by_model, lower_is_better=False)
    return out


# Set by main() (2026-09-25). evaluate_dataset writes each (representation,
# model) result and the per-fold hybrid block to <output>.partial.jsonl as they
# finish and reads them back on the next attempt (experiments/_checkpoint.py).
# None when the function is called from elsewhere: nothing is stored then.
_CK = None
# What failed inside evaluate_dataset without raising; main() exits 1 on it
# after the JSON is written.
_FAILED = []


def _ck_cached(compute, store_if=lambda v: True, **key):
    if _CK is None:
        return compute()
    if _CK.done(**key):
        return _CK.get(**key)
    val = compute()
    if store_if(val):
        _CK.put(val, **key)
    return val


def evaluate_dataset(rep_names, n_folds):
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

    # No whole-dataset hybrid search runs here any more.
    #
    # ⛔ WHAT WAS HERE, and why it went (R2, 2026-10-03). One search over all
    # ~5,000 rows, read through get_best_search_for_downstream, which returns it
    # with selection_scope 'whole_dataset'. Four things were then read off it:
    # the leave-one-fold-out AUROC comparison (build_fold_reselection_input +
    # fold_reselection), the owner's hybrid set scored per held-out fold
    # (get_rule_hybrids), the top 2 representations and top 2 configurations per
    # model (select_model_rep_pairs), and each model's hybrid-minus-baseline
    # deltas (hybrid_deltas). Leaving fold k's score out of a ranking does not
    # leave fold k's rows out of the models that produced every score, so every
    # one of those numbers was chosen with the held-out rows in view. It also
    # asked for rf, which is not one of this dataset's two decided models.
    #
    # What remains is the per-fold arm below: one search per outer fold on that
    # fold's training rows only. The AUROC comparison the deleted block produced
    # now has a per-fold replacement: 'per_fold_auroc' below sets each hybrid
    # against the best single representation of the same fold, with that
    # representation chosen on the fold's training rows (answer 7, 2026-10-05).

    models = get_model_factories(MODELS_LIST, 'classification', n_samples=n)
    # One fold partition, shared by the zero-shot sign check, the
    # per-representation loop and the per-fold hybrid arm, so their held-out
    # numbers can be subtracted fold by fold. The count comes from n_folds,
    # which main() resolves from one setting (get_config's n_splits).
    try:
        fold_splits = _fold_splits(reps, labels, groups, n_folds)
    except ValueError as e:
        print(f"  Cannot split: {e}")
        return None

    # Zero-shot evaluation: LLR reps used directly as ranking scores
    # (matches paper's protocol — no classifier training, just rank by score).
    # These two representations and esm2 are the one exception to the
    # representation rule, kept because this arm is the paper's own protocol
    # (Adelaide, answer 2, 2026-10-05). They are scored ALONE here, and since
    # 2026-10-06 they are kept out of the pool the per-fold hybrid search picks
    # features from (SEARCH_EXCLUDED_REPS).
    # The pool the per-fold hybrid search picks features from: the approved
    # pool, without the three representations that are scored alone only
    # (SEARCH_EXCLUDED_REPS, decided 2026-10-06).
    search_reps = {k: v for k, v in reps.items()
                   if k not in SEARCH_EXCLUDED_REPS}
    print(f"  Hybrid search pool ({len(search_reps)}): "
          f"{', '.join(sorted(search_reps))}")
    print(f"  Scored alone only, kept out of the search pool: "
          f"{', '.join(sorted(r for r in reps if r in SEARCH_EXCLUDED_REPS))}")
    if not search_reps:
        print("  No representation left for the hybrid search pool")

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
            # 2026-10-05, the dirty run's finding on this file: the DIRECTION of
            # this score was never checked. The ranking assumes a higher mean
            # log-likelihood ratio means "more likely a binder"; if the sign
            # runs the other way, every enrichment number here is the count of
            # binders among the molecules the score likes least.
            #
            # Checked on TRAINING ROWS ONLY, fold by fold: the AUROC of the
            # score against the binder labels of that fold's training rows. An
            # AUROC below 0.5 there means the score ranks non-binders first.
            # Nothing is flipped, and the held-out rows are not looked at to
            # decide it -- flipping a sign on held-out rows is choosing the
            # direction on the numbers being reported.
            _train_auroc = []
            for _tr, _ in fold_splits:
                try:
                    _train_auroc.append(
                        float(roc_auc_score(labels[_tr], scores[_tr])))
                except Exception:
                    _train_auroc.append(None)
            _ok = [a for a in _train_auroc if a is not None]
            _below = [a for a in _ok if a < 0.5]
            if not _ok:
                _verdict = ('no training-row AUROC could be computed, so the '
                            'direction of the score is unchecked')
            elif not _below:
                _verdict = ('higher score ranks binders first on every '
                            'training-row fold checked, as the ranking assumes')
            elif len(_below) == len(_ok):
                _verdict = (f'higher score ranks NON-binders first on all '
                            f'{len(_ok)} training-row folds: the score runs '
                            f'the other way and the enrichment below is the '
                            f'count at the wrong end')
            else:
                _verdict = (f'higher score ranks non-binders first on '
                            f'{len(_below)} of {len(_ok)} training-row folds: '
                            f'the direction is not consistent between folds')
            sign_check = {
                'train_row_auroc_per_fold': _train_auroc,
                'n_folds_checked': len(_ok),
                'n_folds_auroc_below_half': len(_below),
                'score_was_flipped': False,
                'checked_on': 'fold_train_rows',
                'verdict': _verdict,
            }
            # Both numbers below rank every molecule in the dataset: no model is
            # fitted in this arm, so there are no training rows to hold out.
            enrichment = compute_enrichment(labels, scores)
            try:
                auroc_val = roc_auc_score(labels, scores)
            except Exception:
                auroc_val = float('nan')
            zero_shot_results[rep_name] = {
                'enrichment': enrichment,
                'auroc': float(auroc_val),
                'scored': 'alone, all rows, no model fitted',
                'sign_check': sign_check,
            }
            e1 = enrichment.get('enrich_0.01', float('nan'))
            print(f"    {rep_name:25s} zero-shot  "
                  f"E@1%={e1:.4f}  AUROC={auroc_val:.4f}")
            print(f"      sign check (training rows only): {_verdict}")

    # Supervised evaluation (KIRBy's extension beyond paper).
    # NOTE: this is a second pass over `reps` after the zero-shot loop
    # above. The two passes are kept separate because the zero-shot pass
    # has no model factory and the supervised pass runs |models| inner
    # iterations per rep; merging them is mechanical but not minimal-diff.
    print(f"\n  Supervised evaluation ({len(reps)} reps × {len(models)} models × {n_folds}-fold CV)...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                res = _ck_cached(
                    lambda: evaluate_rep_enrichment(
                        X, labels, groups, factory(), n_folds,
                        fold_splits=fold_splits),
                    rep=rep_name, model=model_name, n_folds=n_folds,
                    # 2026-10-05: the per-fold held-out AUROC list is new and
                    # the scaler changed (0/1 features pass through), so an
                    # earlier attempt's stored value is a different
                    # measurement and must not be read back.
                    rep_scoring='featurescaler__auroc_per_fold')
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
    # A fresh search on each fold's training rows, its features applied to that
    # fold's held-out rows, and the same top-1/5/10 percent binder count. This
    # is the only hybrid arm in the file.
    #
    # One fold at a time, each stored as it finishes (dirty-run bug 12). The
    # whole arm used to be one stored unit, so a run killed in fold 4 lost
    # folds 0 to 3 as well.
    print(f"\n  Per-fold hybrid enrichment ({n_folds} folds, train-only search)...")
    fold_blocks = []
    # Folds whose whole block raised. Recorded with the fold and the error
    # (2026-10-05); they used to leave only a stdout line and a _FAILED string,
    # and the per-fold lists below silently closed the gap.
    fold_arm_failures = []
    for _fold_idx, (_train_idx, _val_idx) in enumerate(fold_splits):
        try:
            blk = _ck_cached(
                lambda: _hybrid_fold_block(
                    _fold_idx, _train_idx, _val_idx, search_reps, labels,
                    groups, n_folds, models),
                hybrid_fold=_fold_idx, n_folds=n_folds,
                reps=sorted(search_reps),
                # Bumped 2026-10-05: the hybrids are now the 2 best of the
                # fold's own decided-grid search with no distance hybrid, the
                # scaler leaves 0/1 features alone, and the stored shape is one
                # dict per fold. Every earlier partial file is a different
                # measurement and is not read back.
                # Bumped again 2026-10-05: a block now carries 'fit_failures',
                # and the per-fold lists built from it are fold-indexed, so a
                # block stored by an earlier attempt is a different
                # measurement and is not read back.
                # Bumped again 2026-10-06: the search pool no longer holds
                # esmc_llr, antiberty_llr or esm2, so a block stored by an
                # earlier attempt searched a different pool.
                hybrid_set='top2_general_per_fold__decided_grid__no_distance'
                           '__fold_indexed__pool_without_llr_esm2')
        except Exception as e:
            print(f"    fold {_fold_idx}: per-fold hybrid enrichment failed: "
                  f"{e}")
            _FAILED.append(f'per-fold hybrid enrichment, fold {_fold_idx}')
            fold_arm_failures.append({
                'dataset': DATASET, 'fold': int(_fold_idx), 'model': None,
                'reason': f'the whole fold block raised '
                          f'{type(e).__name__}: {e}, so no arm of this fold '
                          f'was scored'})
            continue
        fold_blocks.append(blk)
        for _row in blk['base_reps']:
            print(f"  fold {_row['fold']} {_row['model']:8s} top "
                  f"{N_BASE_REPS} base representations: "
                  f"{', '.join(_row['base_reps'])}")

    per_slot, hybrid_dim = (
        _aggregate_hybrid_folds(fold_blocks, n_folds, list(models))
        if fold_blocks else ({}, 0))
    for _slot in RULE_SLOTS:
        _res = _summarize_enrichment(per_slot.get(_slot) or {}, n_folds,
                                     verbose=(_slot == 'general1'))
        if not _res:
            continue
        results[_slot_key(_slot)] = {
            'dim': hybrid_dim if _slot == 'general1' else None,
            'models': _res, 'slot': _slot}
        for model_name, res in sorted(_res.items()):
            e1 = res.get('enrich_0.01', {}).get('mean', float('nan'))
            auroc = res.get('auroc', {}).get('mean', float('nan'))
            print(f"    {_slot_key(_slot):25s} {model_name:8s}  "
                  f"E@1%={e1:.4f}  AUROC={auroc:.4f}  ***")

    # The second hybrid, filed beside 'kirby_hybrid' with the slot in the name,
    # and paired against the best single representation the same way.
    enrichment_by_slot = {}
    for _slot in RULE_SLOTS[1:]:
        if _slot_key(_slot) in results:
            enrichment_by_slot[_slot] = _paired_enrichment_delta(
                results, hybrid_key=_slot_key(_slot))

    recipes = [r for blk in fold_blocks for r in blk['recipes']]
    base_reps_per_fold = [r for blk in fold_blocks for r in blk['base_reps']]
    # Every arm left empty, in one list: a search that ranked nothing, a
    # held-out fit that raised on a built hybrid, and a fold whose block raised
    # (2026-10-05).
    missing = ([r for blk in fold_blocks for r in blk['missing']]
               + [r for blk in fold_blocks
                  for r in (blk.get('fit_failures') or [])]
               + list(fold_arm_failures))
    skipped_identical = [r for blk in fold_blocks
                         for r in blk['hybrids_skipped_identical_features']]

    return {
        'n_samples': n,
        'n_binders': n_binders,
        'base_rate': n_binders / n,
        # The two downstream models DECISIONS.md Step 1 decided for this
        # dataset. Written out so a result file says which models produced it.
        # The record's distance models are NOT reported any more: this file runs
        # no distance search and builds no distance hybrid (answer 1,
        # 2026-10-05), so naming them here said it did.
        'decided_models': list(MODELS_LIST),
        # The representations the per-fold hybrid search picked features from,
        # and the ones scored on their own only (decided 2026-10-06).
        'hybrid_search_pool': sorted(search_reps),
        'reps_scored_alone_only': sorted(r for r in reps
                                         if r in SEARCH_EXCLUDED_REPS),
        'n_outer_folds': int(n_folds),
        'aug_conditions': list(decided_aug_conditions(DATASET, 'general')),
        'n_base_reps_per_model_per_fold': int(N_BASE_REPS),
        'n_hybrids_per_model_per_fold': int(N_HYBRIDS),
        # Every search in this file ran on one outer fold's training rows.
        'selection_scope': 'fold_train_rows',
        'metric_name': 'auc',
        'lower_is_better': False,
        'zero_shot': zero_shot_results,  # paper's protocol (no training)
        'supervised': results,            # KIRBy's extension (trained classifiers)
        # Hybrid minus best single representation on the same held-out fold, in
        # binder enrichment, one block per top fraction. 'enrichment_comparison'
        # is general1; 'enrichment_comparison_by_slot' is general2.
        'enrichment_comparison': _paired_enrichment_delta(results),
        'enrichment_comparison_by_slot': enrichment_by_slot,
        # One row per (fold, downstream model, slot): the hybrid's held-out
        # AUROC against the held-out AUROC of the best single representation of
        # the same fold, that representation chosen on the fold's training rows
        # (answer 7, 2026-10-05).
        'per_fold_auroc': _per_fold_auroc_table(results, base_reps_per_fold,
                                                n_folds),
        # One row per (fold, downstream model, slot): the allocation method,
        # importance method, budget in features, internal evaluator,
        # representation count and augmentation condition of the hybrid that was
        # built, its training-row score and its held-out AUROC, and that it was
        # chosen on that fold's training rows.
        'rule_recipes_per_fold': recipes,
        # One row per (fold, downstream model): the base representations that
        # fold's own search ranked highest, with their training-row scores.
        'base_reps_per_fold': base_reps_per_fold,
        # Every arm left empty, with the dataset, fold, model and error.
        'rule_missing': missing,
        # A hybrid dropped because its selected features were identical to one
        # already chosen on that fold.
        'hybrids_skipped_identical_features': skipped_identical,
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
        tag = ' ***' if rep_name.startswith('kirby_hybrid') else ''
        e1s = f"{e1:.4f}" if not np.isnan(e1) else "—"
        e5s = f"{e5:.4f}" if not np.isnan(e5) else "—"
        e10s = f"{e10:.4f}" if not np.isnan(e10) else "—"
        aurocs = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        accs = f"{acc:.4f}" if not np.isnan(acc) else "—"
        print(f"  {label:30s} {e1s:>8s} {e5s:>8s} {e10s:>8s} {aurocs:>8s} {accs:>8s}{tag}")

    # Summary
    # Both hybrid slots are hybrids, not individual representations.
    best_indiv = [r for r in rows if not r[0].startswith('kirby_hybrid')]

    print(f"\n  Summary (Enrichment @ top-1%):")
    if best_indiv:
        bi = best_indiv[0]
        print(f"    Best individual: {bi[0]} ({bi[1]}) E@1% = {bi[2]:.4f}")

    # Hybrid against the best single representation, in held-out AUROC, one row
    # per fold (answer 7, 2026-10-05). This replaces the whole-dataset AUROC
    # table that was removed on 2026-10-03: there, one search over all ~5,000
    # rows chose the recipe and the representation. Here the choice is made on
    # each fold's training rows and the two numbers are that fold's held-out
    # rows.
    auroc_rows = ds_results.get('per_fold_auroc') or []
    if auroc_rows:
        print(f"\n  Per-fold AUROC: hybrid against the best single "
              f"representation")
        print(f"    One row per outer fold per downstream model per hybrid "
              f"slot. BestSingle is the representation that fold's own search "
              f"ranked first on its TRAINING rows; both AUROC numbers are that "
              f"fold's held-out rows. Higher is better.")
        print(f"    {'Fold':>5s} {'Model':10s} {'Slot':10s} {'Hybrid':>9s} "
              f"{'BestSingle':>24s} {'Single':>9s} {'Delta':>9s}")
        print(f"    {'-'*5} {'-'*10} {'-'*10} {'-'*9} {'-'*24} {'-'*9} "
              f"{'-'*9}")
        for r in auroc_rows:
            def _f(v, sign=False):
                if v is None:
                    return '—'
                return f"{v:+.4f}" if sign else f"{v:.4f}"
            print(f"    {r['fold']:>5d} {r['model']:10s} {r['slot']:10s} "
                  f"{_f(r['hybrid_auroc_held_out']):>9s} "
                  f"{(r['best_single_rep'] or '—'):>24s} "
                  f"{_f(r['best_single_auroc_held_out']):>9s} "
                  f"{_f(r['delta_hybrid_minus_single'], True):>9s}")

    # Hybrid minus best single representation, in binder enrichment. On each
    # held-out fold both arms are re-picked on the other folds.
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
                h, b = r['hybrid_reselected'], r['best_single_reselected']
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
        help="Comma-separated representation names. The default is this "
             "dataset's approved pool plus the three stated exceptions "
             "(esmc_llr, antiberty_llr, esm2).")
    parser.add_argument(
        '--n-folds', type=int, default=None,
        help='Number of outer CV folds. The default comes from one setting, '
             "get_config's n_splits, not from a number written in this file; "
             'a count of 1 is a single grouped held-out split.')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_trastuzumab_enrichment.json)')
    # Accepted and IGNORED, with a printed line (2026-10-05). Pruning is off in
    # every per-fold search: the schedule a mode supplies deleted members of the
    # decided importance, internal evaluator, representation count and
    # allocation axes after folds 0 and 1 of each search, so which part of the
    # decided grid was searched was settled inside each run. The two launchers
    # that start this file pass --pruning-mode, so the argument stays and does
    # nothing (tests/slurm_scripts/submit_opig_experiments.sh:37).
    parser.add_argument(
        '--pruning-mode', type=str, default=None,
        help='Accepted for the launchers and ignored: pruning is off in every '
             'per-fold search.')

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard what an earlier attempt finished '
             '(<output>.partial.jsonl) and recompute every item.')

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()

    apply_distance_metric_arg(args)
    if args.pruning_mode:
        print(f"  --pruning-mode {args.pruning_mode}: ignored. No per-fold "
              f"search in this file prunes; the decided grid is searched whole.")
    # The outer fold count, from one setting. get_config's 'n_splits' is
    # DEFAULT_N_FOLDS under --mode full, and KIRBY_LOCAL_FOLDS overrides it
    # there; --n-folds on the command line still wins over both. No literal 5
    # anywhere in this file (handoff 2.4 step 14).
    if args.n_folds is None:
        from argparse import Namespace as _NS
        from experiments._shared import get_config
        args.n_folds = int(get_config(
            _NS(mode='full', distance_metric=args.distance_metric))['n_splits'])
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
    print(f"  Downstream models (DECISIONS.md Step 1): {MODELS_LIST}")
    print(f"  Hybrids per model per fold: {N_HYBRIDS} from that fold's own "
          f"general search; no distance hybrid (answer 1, 2026-10-05)")
    print(f"  Base representations per model per fold: {N_BASE_REPS}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Enrichment fractions: {ENRICHMENT_FRACTIONS}")
    print("=" * 70)

    output_path = (args.output or
                   'tests/results/comparisons/opig_trastuzumab_enrichment.json')
    global _CK
    from experiments._checkpoint import script_checkpoint, exit_if_failed
    _CK = script_checkpoint('opig_trastuzumab_enrichment', output_path, args,
                            exclude=('output',), rep_names=rep_names)

    ds_results = evaluate_dataset(rep_names, args.n_folds)
    if ds_results is not None:
        print_analysis(ds_results)
    else:
        _FAILED.append('dataset (evaluate_dataset returned nothing)')

    # Save results
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(ds_results or {}), f, indent=2)
    print(f"\nResults saved: {p}")
    exit_if_failed(_FAILED, 'part')


if __name__ == '__main__':
    main()
