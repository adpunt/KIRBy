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

The hybrid arm has the same property by construction - `build_fold_hybrid`
refits the blueprint's feature selection on each fold's training split.

-----------------------------------------------------------------------------
ARMS
-----------------------------------------------------------------------------
  best_single    one model per base rep; report the best rep
  pca_matched    concat all reps, PCA to the budget dim, one model
  hybrid         one model on the searched fused representation
  ensemble_s1    STYLE 1 (per model): fix one downstream model, train it on
                 every rep in a rep set, aggregate the PREDICTIONS
  ensemble_s2    STYLE 2 (optimal): let each rep pick its OWN best model
                 (per fold, on inner CV), then aggregate those predictions
  *_auto         the self-contained variants: the process ALSO picks its rep
                 set and aggregator on the inner CV. These are the true
                 like-for-like comparators for the hybrid, because the hybrid
                 likewise selects its own recipe from training data.
  oracle_single  best (model x rep) read off the held-out fold itself.
                 NOT a fair arm - an upper bound, reported so the gap is
                 visible rather than implied.

Rep sets:     top (top-3 by inner CV)  |  all (every tier>=2 rep)
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
    get_model_factories, cv_split,
    get_domain_metric,
    build_fold_hybrid, extract_top3_base_reps,
    get_best_search_for_downstream,
    fold_reselection, classify_rep_tier,
    INTERNAL_EVALUATORS,
)
from kirby.find_best_hybrid import (
    find_best_hybrid, nested_cv_cache_namespace, DEFAULT_N_FOLDS,
)

TOP_K = 3        # size of the "top" rep set
AGG_TOP_K = 3    # members kept by the 'topk_mean' aggregator
INNER_FOLDS = 3  # inner CV used by every process to make its own choices


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
    """
    scaler = StandardScaler()
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


def run(data, config, output_dir, dataset_name, **kwargs):
    """Ensemble (model-level fusion) vs hybrid (feature-level fusion)."""
    print("\n" + "=" * 78)
    print("EXPERIMENT C11: ENSEMBLE (model-level) VS HYBRID (feature-level)")
    print("  Process comparison. Every process is replicated per fold and makes")
    print("  ALL of its choices on that fold's training split (inner CV) only.")
    print("=" * 78)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']

    from kirby.find_best_hybrid import BUILTIN_METRICS
    metric_needs_proba = (metric_fn is BUILTIN_METRICS.get('auc', (None,))[0])

    models = get_model_factories(config['models'], task_type=task_type,
                                 n_samples=len(data['labels']))
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    rep_names = sorted(reps.keys())
    rep_dims = {k: int(v.shape[1]) for k, v in reps.items()}
    n_reps = len(rep_names)
    total_dim = sum(rep_dims.values())

    # "General reps" = every rep legal as a STANDALONE baseline. Tier 1 (<50D)
    # is augmentation-only, so it can never be an ensemble member either.
    general_reps = [r for r in rep_names if classify_rep_tier(rep_dims[r], r) >= 2]
    if not general_reps:
        general_reps = list(rep_names)

    classes = np.unique(labels) if task_type == 'classification' else None

    print(f"\n  {n_reps} base reps ({len(general_reps)} tier>=2 / ensemble-eligible), "
          f"total dim={total_dim}, {len(models)} models, {n_folds} folds")
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    sample_X = next(iter(reps.values()))
    # The `b <= n_samples` half of this filter is GONE since 2026-09-10:
    # find_best_hybrid's allow_budget_exceeds_n now defaults to True, so
    # precompute searches and stores every budget asked for, including ones
    # above the row count. Keeping the clause made this file ask for a shorter
    # list than the pool holds — on the small sets, three budgets out of eight —
    # which is the divergence that makes a blueprint lookup return None and
    # silently drop a column.
    from experiments._shared import FULL_BUDGETS
    budgets = [b for b in FULL_BUDGETS if b <= total_dim]

    # Materialise the OUTER folds once. Every arm must see byte-identical splits
    # or the ensemble cannot aggregate across reps and nothing is paired.
    # task_type is passed, not inferred: left out, cv_split falls back to its
    # integer-label heuristic, which misses a classification dataset whose
    # labels are not integer-typed and splits it with a plain GroupKFold.
    folds = list(cv_split(groups, n_folds, sample_X, labels, mode=mode,
                          task_type=task_type))
    n_out = len(folds)

    # Materialise each fold's INNER folds once. These are indices INTO the outer
    # training split, so the inner CV never sees a single held-out sample. Groups
    # are carried through, so the inner split stays scaffold/group-aware too.
    n_inner = 1 if n_folds == 1 else INNER_FOLDS
    inner_folds = []
    for tr_idx, _ in folds:
        try:
            inner = list(cv_split(groups[tr_idx], n_inner, sample_X[tr_idx],
                                  labels[tr_idx],
                                  mode='intermediary' if n_inner == 1 else 'full',
                                  task_type=task_type))
        except Exception:
            inner = []
        inner_folds.append(inner)
    n_in = [len(x) for x in inner_folds]
    print(f"  Inner CV per fold (process's own selection signal): {n_in}")
    if not any(n_in):
        print("  WARNING: no inner splits available - selection-based processes "
              "(top / weighted / topk / *_auto) will be skipped.")

    augmentations = data.get('augmentations')
    last_search = None
    search_fits = 0   # the hybrid's ONE-OFF search, in model fits

    # Where the hybrid's recipe — allocation method, importance method,
    # internal evaluator, budget, augmentation condition — is chosen.
    #
    #   'per_fold'   each outer fold runs its own search on its own training
    #                rows. The ensemble arm already makes every one of its
    #                choices that way, so this is what puts the two arms on
    #                the same footing. Costs one pool of searches per fold.
    #   'whole_data' one search over every row, shared with precompute. Free,
    #                but the hybrid arm then had rows the ensemble arm never
    #                saw, so the head-to-head numbers are written under names
    #                that say so and `auto_minus_hybrid` is not produced.
    recipe_search = kwargs.get('recipe_search', 'per_fold')
    if recipe_search not in ('per_fold', 'whole_data'):
        raise ValueError(
            f"recipe_search={recipe_search!r}; use 'per_fold' or 'whole_data'.")
    print(f"  Hybrid recipe chosen: {recipe_search}")

    # The augmentation condition is part of the recipe, so a per-fold search
    # sweeps the same conditions get_best_search_for_downstream picks among:
    # 'none' alone when the dataset has no augmentation representations, and
    # 'none' / 'uncapped' / 'capped' when it has. Nothing is dropped for cost.
    _aug_conditions = ([('none', None), ('uncapped', augmentations),
                        ('capped', augmentations)] if augmentations
                       else [('none', None)])

    def _search_one_fold(fold_i, tr_idx):
        """Run the recipe pool on ONE outer fold's training rows.

        Returns [(aug_condition, SearchResult)]. Each search is cached under
        its own namespace, one per outer fold, so the folds do not evict each
        other and a re-run reuses them.
        """
        tr_reps = {k: v[tr_idx] for k, v in reps.items()}
        y_tr = labels[tr_idx]
        g_tr = groups[tr_idx] if groups is not None else None
        out = []
        for cond, aug in _aug_conditions:
            aug_tr = ({k: v[tr_idx] for k, v in aug.items()} if aug else None)
            try:
                s_f = find_best_hybrid(
                    base_reps=tr_reps, labels=y_tr, groups=g_tr,
                    models={n: f() for n, f in models.items()},
                    budgets=budgets,
                    augmentations=aug_tr,
                    augmentation_strategy=cond,
                    internal_evaluators=INTERNAL_EVALUATORS,
                    metric=metric_name,
                    task_type=task_type,
                    cache=True,
                    verbose=False,
                    dataset_name=dataset_name,
                    cache_namespace=nested_cv_cache_namespace(fold_i, n_out, 42),
                    tier='general',
                )
                out.append((cond, s_f))
            except Exception as e:
                print(f"    fold {fold_i} recipe search ({cond}) failed: {e}")
        return out

    fold_searches = []
    if recipe_search == 'per_fold':
        print(f"  Searching the recipe inside each of {n_out} outer training "
              f"folds, {len(_aug_conditions)} augmentation condition(s) each.")
        for f_i, (tr_idx, _va) in enumerate(folds):
            fold_searches.append(_search_one_fold(f_i, tr_idx))
            n_found = len(fold_searches[-1])
            print(f"    fold {f_i}: {n_found}/{len(_aug_conditions)} "
                  f"search(es) available")

    def _blueprints_for(model_name):
        """Hybrid recipe per budget for this downstream model.

        Under 'per_fold' the return is {fold_idx: {budget: blueprint}} — each
        fold's own winner, picked on that fold's training rows across the same
        augmentation conditions the whole-dataset pool compares.

        Under 'whole_data' it is {budget: blueprint}, one recipe for every
        fold, chosen with every row's label.
        """
        nonlocal last_search, search_fits
        if recipe_search == 'per_fold':
            per_fold = {}
            for f_i, pool in enumerate(fold_searches):
                bps_f = {}
                for budget in budgets:
                    best_row, best_search = None, None
                    for _cond, s_f in pool:
                        row = (s_f.best_by_model_budget.get(model_name, {})
                               .get(budget))
                        if not row or row.get('score_mean') is None:
                            continue
                        sc = row['score_mean']
                        if not np.isfinite(sc):
                            continue
                        if best_row is None or (
                                sc < best_row['score_mean'] if lower_is_better
                                else sc > best_row['score_mean']):
                            best_row, best_search = row, s_f
                    if best_row is not None:
                        bps_f[budget] = best_row['blueprint']
                        last_search = best_search
                        n_cfg = len(getattr(best_search, 'all_results', None) or [])
                        # One search per outer fold, each over its OWN folds:
                        # the fits the process itself burns. The second factor
                        # is the search's fold count, read off the winning
                        # row's per-fold scores — not n_out. The two are both
                        # 5 by default and coincide, but on --mode intermediary
                        # n_out is 1 while the search still runs
                        # DEFAULT_N_FOLDS, and n_out squared is then 5x low.
                        n_search_folds = (len(best_row.get('fold_scores') or [])
                                          or DEFAULT_N_FOLDS)
                        search_fits = max(search_fits,
                                          n_cfg * n_out * n_search_folds)
                per_fold[f_i] = bps_f
            return per_fold

        bps = {}
        for budget in budgets:
            try:
                search, meta = get_best_search_for_downstream(
                    reps, labels, groups,
                    downstream_model=model_name, budget=budget,
                    augmentations=augmentations,
                    dataset_name=dataset_name, task_type=task_type, data=data)
            except Exception as e:
                print(f"    Blueprint lookup failed ({model_name}, b={budget}): {e}")
                continue
            if meta:
                bps[budget] = meta['best_config']['blueprint']
                last_search = search
                n_cfg = len(getattr(search, 'all_results', None) or [])
                search_fits = max(search_fits, n_cfg * n_out)
        return bps

    results = []
    reselection_candidates = {}      # hybrid recipes -> fold_reselection
    reselected_baselines = {}       # single reps    -> fold_reselection
    reselection_ens_candidates = {}  # ensemble procs -> fold_reselection
    cost_rows = []

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
        blueprints = _blueprints_for(model_name)

        def _bp_at(fold_i, budget):
            """This fold's recipe at this budget, or None.

            Under 'per_fold' each fold has its own; under 'whole_data' every
            fold is handed the same one.
            """
            if recipe_search == 'per_fold':
                return blueprints.get(fold_i, {}).get(budget)
            return blueprints.get(budget)

        _have_bp = (any(bps for bps in blueprints.values())
                    if recipe_search == 'per_fold' else bool(blueprints))
        if not _have_bp:
            print(f"    No recipe for {model_name} - hybrid arm empty.")

        for d in (outer_agg, outer_hard, outer_score, inner_agg, inner_hard,
                  inner_score, inner_fold_score, member_cost):
            d[model_name] = {}

        # =================================================================
        # 1. Best single base rep.
        #    This loop ALSO trains every ensemble member (outer) and every
        #    selection probe (inner). Nothing downstream refits anything.
        # =================================================================
        best_score = np.inf if lower_is_better else -np.inf
        best_rep = None
        single_results = {}
        baseline_folds = {}
        hybrid_cands = []

        for rep_name in rep_names:
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
            baseline_folds[rep_name] = o_scores
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

            valid = [s for s in o_scores if np.isfinite(s)]
            mean_s = float(np.mean(valid)) if valid else float('nan')
            single_results[rep_name] = {
                'score_mean': mean_s,
                'score_std': float(np.std(valid)) if valid else float('nan'),
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

        reselected_baselines[model_name] = baseline_folds
        if best_rep is None:
            print("    Best single: NONE (all reps failed)")
            continue

        bs = single_results[best_rep]
        print(f"    Best single: {best_rep} {metric_name}={bs['score_mean']:.4f} "
              f"dim={bs['dim']}")
        results.append({
            'model': model_name, 'method': 'best_single', 'augmented': False,
            'score_mean': bs['score_mean'], 'score_std': bs['score_std'],
            'total_dim': bs['dim'], 'n_models_trained': 1,
            'total_fit_time_s': bs['fit_time_s'],
            'total_predict_time_s': bs['predict_time_s'],
            'fold_scores': bs['fold_scores'],
            'detail': {'best_rep': best_rep, 'all_reps': single_results},
            'cost': _cost_row(1, bs['dim'], 1, bs['dim'], bs['model_bytes'],
                              bs['fit_time_s'], bs['predict_time_s'], bs['dim']),
        })
        cost_rows.append((model_name, 'best_single', results[-1]['cost'],
                          bs['score_mean']))

        # =================================================================
        # 2. PCA-matched control. Fit on TRAIN, transform HELD-OUT.
        # =================================================================
        for budget in budgets:
            f_s, f_fit, f_pred = [], [], []
            n_comp = 0
            pca_bytes = 0
            for tr_idx, va_idx in folds:
                y_tr = labels[tr_idx]
                X_tr_c = np.hstack([reps[r][tr_idx] for r in rep_names])
                X_va_c = np.hstack([reps[r][va_idx] for r in rep_names])
                n_comp = min(budget, X_tr_c.shape[1], len(y_tr) - 1)
                if n_comp < 2:
                    continue
                try:
                    sc = StandardScaler()
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
                pca_bytes = max(pca_bytes,
                                _model_bytes(model) + pca.components_.nbytes)

            if not f_s:
                continue
            results.append({
                'model': model_name, 'method': 'pca_matched', 'budget': budget,
                'score_mean': float(np.mean(f_s)), 'score_std': float(np.std(f_s)),
                'total_dim': n_comp, 'n_models_trained': 1,
                'total_fit_time_s': float(np.mean(f_fit)),
                'total_predict_time_s': float(np.mean(f_pred)),
                'augmented': False, 'fold_scores': [float(s) for s in f_s],
                # PCA needs EVERY rep at inference plus a dense projection.
                'cost': _cost_row(n_reps, total_dim, 1, n_comp, pca_bytes,
                                  float(np.mean(f_fit)), float(np.mean(f_pred)),
                                  total_dim),
            })
            cost_rows.append((model_name, f'pca_b{budget}', results[-1]['cost'],
                              results[-1]['score_mean']))

        # =================================================================
        # 3. Hybrid. build_fold_hybrid refits the selection on TRAIN only.
        # =================================================================
        for budget in budgets:
            if not any(_bp_at(f_i, budget) is not None
                       for f_i in range(len(folds))):
                continue
            # One entry per OUTER FOLD, always, in fold order. A fold this arm
            # cannot score records NaN and keeps its slot: the baseline arm
            # above does the same, and the per-fold reselection pairs the two arms
            # by fold INDEX. Dropping the entry instead shifted every later
            # fold's hybrid score up one position and paired it against the
            # wrong fold of the single-representation arm.
            f_s = [float('nan')] * len(folds)
            f_fit = [float('nan')] * len(folds)
            f_pred = [float('nan')] * len(folds)
            n_feat = 0
            hyb_bytes = 0
            touched = set()
            recipes_this_budget = []

            for f_i, (tr_idx, va_idx) in enumerate(folds):
                bp_f = _bp_at(f_i, budget)
                if bp_f is None:
                    print(f"    fold {f_i} b={budget}: no recipe - fold recorded "
                          f"as missing.")
                    continue
                recipes_this_budget.append({
                    'fold': f_i,
                    'allocation': bp_f.allocation_method,
                    'importance': bp_f.importance_method,
                    'internal_evaluator': bp_f.internal_evaluator,
                    'augmentation_strategy': bp_f.augmentation_strategy,
                })
                tr_reps = {k: v[tr_idx] for k, v in reps.items()}
                va_reps = {k: v[va_idx] for k, v in reps.items()}
                y_tr = labels[tr_idx]
                augs = augmentations or {}
                tr_aug = {k: v[tr_idx] for k, v in augs.items()} if augs else None
                va_aug = {k: v[va_idx] for k, v in augs.items()} if augs else None
                try:
                    t0 = time.time()
                    X_tr_h, X_va_h, info = build_fold_hybrid(
                        tr_reps, va_reps, y_tr, bp_f,
                        augmentations_train=tr_aug, augmentations_val=va_aug,
                        groups=(groups[tr_idx] if groups is not None else None),
                        model=model_factory(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better)
                    build_s = time.time() - t0
                except Exception as e:
                    print(f"    fold {f_i} b={budget}: hybrid build failed "
                          f"({e}) - fold recorded as missing.")
                    continue

                # THE reviewer's question: how many featurisers does the hybrid
                # still need? Measured from the blueprint, never assumed.
                alloc = info.get('allocation') or {}
                touched |= {r for r, n in alloc.items()
                            if isinstance(n, (int, float)) and n > 0}
                aug_info = info.get('augmentation_info') or {}
                if isinstance(aug_info, dict):
                    touched |= {k for k in aug_info if k in reps}

                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr_h)
                X_va_s = sc.transform(X_va_h)
                n_feat = X_tr_h.shape[1]
                try:
                    model, agg, _h, a, b = _fit_predict(
                        model_factory, X_tr_s, y_tr, X_va_s, task_type)
                except Exception as e:
                    print(f"    fold {f_i} b={budget}: hybrid fit failed "
                          f"({e}) - fold recorded as missing.")
                    continue
                scored = _to_metric_space(agg, task_type, metric_needs_proba,
                                          classes)
                f_s[f_i] = float(metric_fn(labels[va_idx], scored))
                f_fit[f_i] = a + build_s
                f_pred[f_i] = b
                hyb_bytes = max(hyb_bytes, _model_bytes(model))

            # NaNs are dropped here, at scoring time, not at collection time.
            valid_s = [s for s in f_s if np.isfinite(s)]
            if not valid_s:
                continue
            valid_fit = [t for t in f_fit if np.isfinite(t)]
            valid_pred = [t for t in f_pred if np.isfinite(t)]
            b_lab = 'full' if budget == total_dim else str(budget)
            score = float(np.mean(valid_s))
            n_touch = len(touched) if touched else n_reps
            touch_dim = sum(rep_dims.get(r, 0) for r in touched) or total_dim
            _missing = len(f_s) - len(valid_s)
            print(f"    Hybrid b={b_lab:<5}: {metric_name}={score:.4f} "
                  f"dim={n_feat} featurisers={n_touch}/{n_reps}"
                  + (f" ({len(valid_s)}/{len(f_s)} folds scored)"
                     if _missing else ""))

            f_s_f = [float(s) for s in f_s]
            hybrid_cands.append({'name': f'hybrid_b{b_lab}', 'fold_scores': f_s_f})
            results.append({
                'model': model_name, 'method': 'hybrid', 'augmented': False,
                'budget': budget, 'score_mean': score,
                'score_std': float(np.std(valid_s)), 'total_dim': n_feat,
                'n_models_trained': 1,
                'n_folds_scored': len(valid_s),
                'total_fit_time_s': float(np.mean(valid_fit)),
                'total_predict_time_s': float(np.mean(valid_pred)),
                'fold_scores': f_s_f, 'reps_touched': sorted(touched),
                # Where this budget's recipe came from, and — under
                # 'per_fold' — which recipe each fold actually picked. When
                # they differ between folds, one recipe is not the answer.
                'recipe_search': recipe_search,
                'recipes_per_fold': recipes_this_budget,
                'cost': _cost_row(n_touch, touch_dim, 1, n_feat, hyb_bytes,
                                  float(np.mean(valid_fit)),
                                  float(np.mean(valid_pred)),
                                  n_feat, selection_fits=search_fits),
            })
            cost_rows.append((model_name, f'hybrid_b{b_lab}',
                              results[-1]['cost'], score))

        reselection_candidates[model_name] = hybrid_cands

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
            for rep in general_reps:
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
            for rep in general_reps:
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
            for rep in general_reps:
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
        reselection_ens_candidates.setdefault(model_key, [])

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
                _emit(results, cost_rows, reselection_ens_candidates, model_key,
                      method, rep_set, agg_name, fs, _select, member_cost,
                      rep_dims, metric_name, n_out, inner_folds, general_reps,
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
            _emit(results, cost_rows, reselection_ens_candidates, model_key,
                  f'{method}_auto', 'auto', 'auto', auto_fs, _select,
                  member_cost, rep_dims, metric_name, n_out, inner_folds,
                  general_reps, all_models, auto_choice=auto_choice)
            print(f"    per-fold choice: {auto_choice}")

    # ---- ORACLE: best (model x rep) read off the held-out fold itself. -----
    # Deliberately optimistic, NOT a fair arm. Reported so the gap between it
    # and the honest processes is visible rather than implied.
    if all_models:
        oracle = []
        for f in range(n_out):
            best = np.nan
            for m in all_models:
                for rep in general_reps:
                    s = outer_score.get(m, {}).get(rep, [np.nan] * n_out)[f]
                    if not np.isfinite(s):
                        continue
                    if not np.isfinite(best) or \
                       (lower_is_better and s < best) or \
                       (not lower_is_better and s > best):
                        best = s
            oracle.append(float(best))
        if any(np.isfinite(s) for s in oracle):
            results.append({
                'model': 'optimal', 'method': 'oracle_single', 'augmented': False,
                'rep_set': 'all', 'aggregator': 'none', 'is_oracle': True,
                'score_mean': float(np.nanmean(oracle)),
                'score_std': float(np.nanstd(oracle)), 'fold_scores': oracle,
                'n_models_trained': 1, 'total_dim': 0,
                'total_fit_time_s': 0.0, 'total_predict_time_s': 0.0,
            })
            print(f"\n  oracle_single (UPPER BOUND - peeks at the held-out fold): "
                  f"{metric_name}={np.nanmean(oracle):.4f}")

    # =====================================================================
    # PER-FOLD RESELECTION + HEAD-TO-HEAD
    # =====================================================================
    reselected_hybrid = fold_reselection(reselection_candidates, reselected_baselines,
                                   lower_is_better)
    reselected_ens = fold_reselection(reselection_ens_candidates, reselected_baselines,
                                lower_is_better)

    def _delta(better, worse):
        """Direction-corrected: positive means `better` really is better."""
        if not (np.isfinite(better) and np.isfinite(worse)):
            return float('nan')
        return (worse - better) if lower_is_better else (better - worse)

    def _paired_delta(better_folds, worse_folds):
        """Mean over folds of the per-fold difference, and how many folds.

        `_delta` above subtracts two means, which is only the same number when
        both arms scored the same folds — a hybrid budget that lost a fold, or
        an ensemble process that had no inner split on one fold, makes them
        diverge. Both lists are fold-INDEX-aligned, so position f is the same
        held-out fold on both sides; a fold missing from either side
        contributes no term.
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

    # The *_auto process is self-contained, so its plain fold-mean is already
    # honest - no leave-one-fold-out reselection is needed or wanted.
    auto_by_key = {}
    auto_folds_by_key = {}
    for r in results:
        if r['method'].endswith('_auto'):
            auto_by_key[r['model']] = r['score_mean']
            auto_folds_by_key[r['model']] = r.get('fold_scores') or []

    # Under 'per_fold' the hybrid's recipe was searched inside each outer
    # training fold, so it paid the same tax the ensemble process pays and the
    # head-to-head deltas below are a like-for-like comparison.
    #
    # Under 'whole_data' the recipe came from a search over every row, while
    # the ensemble made every one of its choices inside the fold. The hybrid
    # side then saw rows the ensemble side never did, so the difference is not
    # a comparison. The hybrid number and the deltas are written under names
    # carrying that, and the names a reader would take for held-out numbers —
    # hybrid_reselected, auto_minus_hybrid, ensemble_minus_hybrid,
    # hybrid_minus_single — are not written at all.
    _reselected_recipe = (recipe_search == 'per_fold')
    _h_key = ('hybrid_reselected' if _reselected_recipe
              else 'hybrid_recipe_selected_on_all_rows')
    _suffix = '' if _reselected_recipe else '_recipe_all_rows'

    head_to_head = {}
    for m in sorted(set(reselected_hybrid) | set(reselected_ens)):
        h = reselected_hybrid.get(m, {}).get('hybrid_reselected', float('nan'))
        e = reselected_ens.get(m, {}).get('hybrid_reselected', float('nan'))
        b = reselected_hybrid.get(m, {}).get('best_single_reselected', float('nan'))
        a = auto_by_key.get(m, float('nan'))
        # Fold-index-aligned held-out scores behind each of the three means
        # above. fold_reselection returns them already paired; the *_auto arm
        # writes its own per-fold scores into `results`.
        h_folds = reselected_hybrid.get(m, {}).get('hybrid_fold_scores', [])
        e_folds = reselected_ens.get(m, {}).get('hybrid_fold_scores', [])
        b_folds = reselected_hybrid.get(m, {}).get('baseline_fold_scores', [])
        a_folds = auto_folds_by_key.get(m, [])
        ens_paired, ens_nf = _paired_delta(e_folds, h_folds)
        auto_paired, auto_nf = _paired_delta(a_folds, h_folds)
        single_paired, single_nf = _paired_delta(h_folds, b_folds)
        head_to_head[m] = {
            _h_key: h,
            'hybrid_recipe_search': recipe_search,
            'ensemble_reselected': e,          # best process, LOFO-reselected
            'ensemble_auto': a,            # self-contained process (preferred)
            'best_single_reselected': b,
            f'ensemble_minus_hybrid{_suffix}': _delta(e, h),
            f'auto_minus_hybrid{_suffix}': _delta(a, h),
            f'hybrid_minus_single{_suffix}': _delta(h, b),
            # The paired form of the same three differences: the mean over
            # folds of the per-fold difference, not the difference of two
            # means. They separate whenever the two arms cover different
            # folds, and this is the one to report.
            f'ensemble_minus_hybrid_mean_of_folds{_suffix}': ens_paired,
            f'auto_minus_hybrid_mean_of_folds{_suffix}': auto_paired,
            f'hybrid_minus_single_mean_of_folds{_suffix}': single_paired,
            'n_paired_folds': {
                'ensemble_minus_hybrid': ens_nf,
                'auto_minus_hybrid': auto_nf,
                'hybrid_minus_single': single_nf,
            },
            'hybrid_fold_scores': h_folds,
            'ensemble_fold_scores': e_folds,
            'ensemble_auto_fold_scores': [float(x) for x in a_folds],
            'best_single_fold_scores': b_folds,
            'ensemble_rule_per_fold':
                reselected_ens.get(m, {}).get('hybrid_winners_per_fold', []),
            'hybrid_recipe_per_fold':
                reselected_hybrid.get(m, {}).get('hybrid_winners_per_fold', []),
            'single_fold': reselected_hybrid.get(m, {}).get(
                'single_fold', reselected_ens.get(m, {}).get('single_fold', False)),
        }

    print("\n" + "=" * 78)
    print("  HEAD-TO-HEAD" if _reselected_recipe
          else "  HEAD-TO-HEAD (hybrid recipe from every row - NOT comparable)")
    print("=" * 78)
    print(f"    {'Model':<9} {'Hybrid':>9} {'EnsAuto':>9} {'EnsBest':>9} "
          f"{'Single':>9} {'Auto-Hyb':>9} {'paired':>9}")
    print("    " + "-" * 72)
    for m, hh in head_to_head.items():
        print(f"    {m:<9} {hh[_h_key]:>9.4f} {hh['ensemble_auto']:>9.4f} "
              f"{hh['ensemble_reselected']:>9.4f} {hh['best_single_reselected']:>9.4f} "
              f"{hh[f'auto_minus_hybrid{_suffix}']:>+9.4f} "
              f"{hh[f'auto_minus_hybrid_mean_of_folds{_suffix}']:>+9.4f}")
    print("    Auto-Hyb is the difference of the two means; paired is the mean")
    print("    over folds of the per-fold difference. They differ only when the")
    print("    two arms scored different folds.")
    print("\n    EnsAuto is the fair comparator: a self-contained process that")
    print("    picks its own rep set + aggregator per fold, on that fold's train")
    print("    split - exactly as the hybrid picks its own recipe.")
    print("    Auto-Hyb > 0 => the ensemble is more accurate. The cost table says")
    print("    what that accuracy costs, forever, per inference.")
    if not _reselected_recipe:
        print("\n    The hybrid column's recipe was chosen over every row, "
              "including")
        print("    each fold's held-out rows; the ensemble columns chose "
              "everything")
        print("    inside the fold. Auto-Hyb is written as "
              "auto_minus_hybrid_recipe_all_rows")
        print("    and is not a like-for-like difference. Re-run with "
              "--recipe-search per_fold")
        print("    for the number that is.")

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

    break_even = {}
    for model_name in all_models:
        hybs = [r for r in results if r['model'] == model_name
                and r['method'] == 'hybrid' and 'cost' in r]
        enss = [r for r in results if r['model'] == model_name
                and r['method'] == 'ensemble_s1_auto' and 'cost' in r]
        if not enss:
            enss = [r for r in results if r['model'] == model_name
                    and r['method'] == 'ensemble_s1'
                    and r.get('rep_set') == 'all' and 'cost' in r]
        if not hybs or not enss:
            continue
        _best = min if lower_is_better else max
        h = _best(hybs, key=lambda x: x['score_mean'])
        e = _best(enss, key=lambda x: x['score_mean'])
        n_saved = e['cost']['n_models_deployed'] - h['cost']['n_models_deployed']

        break_even[model_name] = {
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
            'hybrid_recipe_search': recipe_search,
            # Under 'whole_data' this difference is written under a name that
            # says the hybrid side's recipe came from every row, so it is not
            # read as an ensemble-versus-hybrid result.
            f'accuracy_delta_ensemble_minus_hybrid{_suffix}': _delta(
                e['score_mean'], h['score_mean']),
            # ★ THE load-bearing number. Xu et al. (Cost-Sensitive Tree of
            # Classifiers, ICML 2013): test-time cost is DOMINATED by feature
            # extraction, and a featuriser is paid for the first time ANY of its
            # columns is used. So if the blueprint draws even one column from
            # every rep, the hybrid runs every encoder the ensemble runs and the
            # compute saving collapses to the downstream forward pass - a
            # rounding error next to a protein language model.
            # Under 'per_fold' the featurisers counted here are the union
            # over the folds' own recipes, so a recipe that moves between
            # folds shows up as a larger count — which is what deployment
            # would actually pay.
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
            d = be[f'accuracy_delta_ensemble_minus_hybrid{_suffix}']
            verdict = ('ensemble is better' if d > 0 else
                       'HYBRID is better' if d < 0 else 'tied')
            print(f"    {m}:")
            print(f"      featurisers  {be['ensemble_featurisers']:>3} (ens) -> "
                  f"{be['hybrid_featurisers']:>3} (hyb)   "
                  f"{be['featuriser_reduction']:.1f}x fewer")
            print(f"      models       {be['ensemble_models_deployed']:>3} (ens) -> "
                  f"{be['hybrid_models_deployed']:>3} (hyb)")
            print(f"      storage/sample                     "
                  f"{be['storage_reduction']:.1f}x smaller")
            print(f"      accuracy: ensemble - hybrid        "
                  f"{d:+.4f} {metric_name}  ({verdict})")
            print(f"      one-off search                     "
                  f"{be['search_cost_model_fits']} fits, repaid after "
                  f"{be['break_even_retrains']:.1f} retrains")
            print(f"      ensemble's OWN selection cost      "
                  f"{be['ensemble_selection_fits']} fits per fold")
            if not be['featurisation_saving_exists']:
                print("      *** WARNING: the blueprint touches EVERY rep the "
                      "ensemble needs.")
                print("      *** Featurisation cost is IDENTICAL; the only saving "
                      "is the downstream")
                print("      *** forward pass. Do NOT claim a compute win here.")

        print("\n    Deployment regime matters and the two answers differ:")
        print("      online - featurisers run per query -> the win is compute,")
        print("               real only where rep_sparsity < 1.")
        print("      batch  - reps precomputed and cached -> featurisation is paid")
        print("               once per sample, so the honest axis is STORAGE.")
        print("    This pipeline precomputes reps, i.e. the BATCH regime.")

    top3 = extract_top3_base_reps(last_search) if last_search is not None else []
    return {
        'results': results,
        'n_reps': n_reps,
        'total_dim': total_dim,
        'rep_names': rep_names,
        'general_reps': general_reps,
        'rep_dims': rep_dims,
        'metric_name': metric_name,
        'n_inner_folds': n_in,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # Existing contract - hybrid vs best single rep. UNCHANGED.
        'fold_reselection': {
            'lower_is_better': bool(lower_is_better),
            # Every fold counts. The candidates here are the arms this file built itself,
            # not a search's recipe list, so no fold filtered them.
            'exclude_folds': [],
            'candidates_by_model': reselection_candidates,
            'baselines_by_model': reselected_baselines,
        },
        # Same shape, ensemble PROCESSES as the candidates. NOTE: the key
        # 'hybrid_reselected' returned by fold_reselection() means "best ENSEMBLE
        # process, LOFO-reselected" when fed this block. Prefer `ensemble_auto`
        # in head_to_head - it is self-contained and needs no reselection.
        'fold_reselection_ensemble': {
            'lower_is_better': bool(lower_is_better),
            # Every fold counts. The candidates here are the arms this file built itself,
            # not a search's recipe list, so no fold filtered them.
            'exclude_folds': [],
            'candidates_by_model': reselection_ens_candidates,
            'baselines_by_model': reselected_baselines,
        },
        'head_to_head': head_to_head,
        'break_even': break_even,
        'search_cost_model_fits': int(search_fits),
        # Where the hybrid's recipe came from. 'per_fold' means each outer
        # fold searched its own on its own training rows, which is what the
        # ensemble arm already does — only then is the head-to-head a
        # comparison. 'whole_data' means one recipe from every row.
        'hybrid_recipe_search': recipe_search,
        'hybrid_recipe_selected_on': ("each outer fold's training rows"
                                      if recipe_search == 'per_fold'
                                      else 'all rows'),
        'hybrid_recipe_aug_conditions': [c for c, _a in _aug_conditions],
    }


def _emit(results, cost_rows, reselection_ens_candidates, model_key, method,
          rep_set, agg_name, fold_scores, select_fn, member_cost, rep_dims,
          metric_name, n_out, inner_folds, general_reps, all_models,
          auto_choice=None):
    """Record one ensemble process: score, honest candidate, standing cost."""
    valid = [s for s in fold_scores if np.isfinite(s)]
    if not valid:
        return
    score = float(np.mean(valid))

    # Membership is re-selected INDEPENDENTLY on every fold, from that fold's
    # training split alone. Record it per fold: it is the audit trail proving no
    # held-out information entered the choice, and it shows how much the process
    # actually changes its mind from fold to fold.
    #
    # For the *_auto arms the rep set is not fixed: fold f picked its own on
    # that fold's inner CV, and auto_choice[f] records it as 'rep_set|agg'.
    # Re-running the selection with 'all' instead described a process this arm
    # never ran, so the membership, the model count and the whole cost row
    # below belonged to the all-representation arm.
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
    sel_fits = (J * len(general_reps) * n_probe_models
                if (rep_set == 'auto' or _needs_selection(rep_set, agg_name))
                else 0)

    reselection_ens_candidates.setdefault(model_key, []).append(
        {'name': f'{method}|{rep_set}|{agg_name}',
         'fold_scores': [float(s) for s in fold_scores]})

    row = {
        'model': model_key, 'method': method, 'rep_set': rep_set,
        'aggregator': agg_name, 'augmented': False,
        'score_mean': score, 'score_std': float(np.std(valid)),
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
    parser.add_argument(
        '--recipe-search', dest='recipe_search',
        choices=['per_fold', 'whole_data'], default='per_fold',
        help="Where the hybrid's recipe is chosen. per_fold (default): each "
             "outer fold searches its own on its own training rows, matching "
             "what the ensemble arm already does, so the head-to-head is a "
             "comparison. Costs one pool of searches per outer fold. "
             "whole_data: one search over every row, free, but the hybrid "
             "then saw rows the ensemble never did — the head-to-head fields "
             "are renamed to say so and auto_minus_hybrid is not written.")
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {config['models']}")
    print(f"Recipe search: {args.recipe_search}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  recipe_search=args.recipe_search)
    save_experiment_results(results, args.output, 'c11_ensemble_vs_hybrid',
                            args.dataset)
