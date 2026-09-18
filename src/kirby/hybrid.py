"""
Core hybrid representation creation functions - ENHANCED with Greedy Allocation

Supports multiple feature importance methods and allocation strategies:
- Allocation methods: 'performance_weighted', 'fixed', 'weighted_global', plus the
  named performance_weighted variants — see NAMED_ALLOCATIONS for the full list.
  The whole GREEDY family is CUT and must not be used; the names still resolve so
  old configs do not crash. See the ⛔ block above NAMED_ALLOCATIONS.
- Importance methods (8 retained after benchmark correlation analysis):
  Fast: treeshap, xgboost_gain, lightgbm_gain, deeplift
  Slower: permutation, kernelshap
  Long search only: integrated_gradients, gradientshap
- Augmentation strategies: 'all', 'none', 'greedy_ablation'

Importance-sensitive vs insensitive allocations:
  'greedy' and 'dominant_greedy' use internal RF feature_importances_ for
  feature selection, so they produce identical results regardless of the
  importance_method parameter. All other allocation methods (performance_weighted,
  and all named performance_weighted variants) use the specified importance_method
  and will produce different results per method.
  find_best_hybrid accounts for this — insensitive allocations are only evaluated
  once per (budget, model) rather than N times for N importance methods.

THE BUDGET IS A MAXIMUM, NOT A TARGET. It is the most features a hybrid may
use; a hybrid that scores better on fewer features is a better answer and comes
back narrower. A hybrid must never come back WIDER than its budget, and it is
compared against a base representation cut to the SAME budget, not against that
representation at its full native width. See the block "WHAT THE BUDGET IS"
below for the three rules that follow from it.

Default configuration:
- allocation_method='performance_weighted'   (set 2026-09-10; was 'top_3')
- importance_method='lightgbm_gain'
- internal_evaluator='rf'   (set 2026-09-16; was 'lightgbm', a blanket
  ridge->lightgbm sweep on 2026-07-16 that was never a comparison. f29's
  per-representation fits, the nearest real numbers, show rf 2x-20x cheaper
  than lightgbm on every one of 5 datasets, winning on 1, tying on 1, and
  losing on 3 (worst: logd rmse 0.6126 vs 0.5066). No experiment has scored
  internal_evaluator choice directly -- see FRAMEWORK_FINALIZATION.md f31.)
- budget=100
"""

import os
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import StandardScaler

from kirby._warn import KirbyWarning, warn_once


def lgbm_n_jobs() -> int:
    """Thread count for every LightGBM fit. -1 (all cores) unless overridden.

    On macOS LightGBM and torch link DIFFERENT OpenMP runtimes — LightGBM loads
    libomp.dylib, torch loads libiomp5.dylib (from functorch/.dylibs). Both load
    without complaint, and a single-threaded LightGBM fit works fine alongside
    torch. But a MULTI-threaded LightGBM fit after torch has started its own
    thread pool segfaults the process outright: no Python traceback, faulthandler
    prints "Fatal Python error:" and dies before it can write the stack.

    Every search hits this, because importance_methods defaults to
    ['deeplift', 'lightgbm_gain', 'treeshap'] and 'deeplift' runs torch + captum
    (hybrid.py compute_feature_importance) before the allocation pre-compute
    reaches its first LightGBM evaluator.

    Measured on this machine: without deeplift, n_jobs=-1 is fine; after
    deeplift, n_jobs=-1 crashes and n_jobs=1 does not.

    Set KIRBY_LGBM_THREADS=1 to make a local run survivable. Leave it unset on
    the server, where there is one OpenMP runtime and nothing to work around —
    the default returned here is the same -1 that was hardcoded before.
    """
    try:
        return int(os.environ.get('KIRBY_LGBM_THREADS', '-1'))
    except ValueError:
        return -1


# ============================================================================
# SEARCH OBJECTIVE MODES
# ============================================================================
# Mirror of the constants in find_best_hybrid (kept here too so hybrid.py has no
# import dependency on find_best_hybrid). 'predictive' = allocators maximize
# downstream model performance. 'similarity' = allocators maximize held-out
# ACTIVITY READ-ACROSS (predict each held-out molecule from its five nearest
# neighbours, Spearman against truth), fitting no model. Scaffold groups for the
# held-out rows are required; create_hybrid slices and passes them down.
#
# ⛔ SAC IS CUT AS AN OBJECTIVE. CLOSED 2026-08-26 by the owner. It is also no
# longer the score that splits the budget: until this change every geometry
# objective, read-across included, ranked representations by SAC when dividing
# the budget. The name still resolves so old configs raise a readable error.
# The numbers on both sides are recorded in find_best_hybrid, above
# OBJECTIVE_PREDICTIVE. Do not re-derive that decision.
OBJECTIVE_PREDICTIVE = 'predictive'
OBJECTIVE_SIMILARITY_SAC = 'similarity_sac'   # CUT — see the block above
OBJECTIVE_SIMILARITY = 'similarity'
# Added 2026-08-27. MIRROR of find_best_hybrid.OBJECTIVE_STRUCT_FID. This file
# keeps its own copy so it need not import find_best_hybrid; a name added in
# one and missed in the other makes every fold fail with "All folds failed".
OBJECTIVE_STRUCT_FID = 'struct_fid'
# Added 2026-09-03. MIRROR of find_best_hybrid.OBJECTIVE_PREDICTIVE_CORR.
# Prediction scored as a CORRELATION rather than as the dataset's own metric --
# the same form prediction takes inside a blend. It exists so the prediction
# corner of an objective-weight grid sits on the SAME scale as every mixture
# containing prediction. Without it, 'predict 1.00' is measured in rmse while
# 'predict 0.75 + neighbour 0.25' is measured in Spearman, and the two cannot
# be read as ends of one edge. Scored here through the blend machinery as
# {OBJECTIVE_PREDICTIVE: 1.0}, so it is the exact limit of those mixtures.
OBJECTIVE_PREDICTIVE_CORR = 'predictive_corr'

# Added 2026-09-03. MIRROR of find_best_hybrid.KNN_TUNED_K_ATTR, same
# no-import rule as the objective names above. find_best_hybrid stamps this
# attribute on a KNN estimator whose n_neighbors was chosen by tune_models and
# must reach the fit; the 'knn' branch of _evaluate_allocation reads it. If the
# two strings ever disagree, that branch silently falls back to k=5 and a tuned
# run reads as an untuned one.
KNN_TUNED_K_ATTR = '_kirby_tuned_k'

_GEOMETRY_OBJECTIVES = (OBJECTIVE_SIMILARITY, OBJECTIVE_STRUCT_FID)


# Every objective NAME this file knows how to score. An objective outside this
# set used to fall straight through _is_geometry_objective into the predictive
# branch, so a typo — or the cut 'similarity_sac' — silently ran a prediction
# search under the wrong name and reported it as the objective that was asked
# for. Checked by _check_objective_name, called from create_hybrid.
_KNOWN_OBJECTIVES = ((OBJECTIVE_PREDICTIVE, OBJECTIVE_PREDICTIVE_CORR)
                     + _GEOMETRY_OBJECTIVES)


def _check_objective_name(obj) -> None:
    """Raise on an objective name this file cannot score. Blends are checked by
    normalise_objective, which has its own message."""
    if isinstance(obj, dict):
        return
    if obj in _KNOWN_OBJECTIVES:
        return
    if obj == OBJECTIVE_SIMILARITY_SAC:
        raise ValueError(
            "objective='similarity_sac' was CUT on 2026-08-26 and no longer "
            "scores anything. Use 'similarity' (each molecule's property "
            "predicted from its five nearest neighbours) or 'struct_fid' (MCS "
            "structural fidelity).")
    raise ValueError(
        f"unknown objective {obj!r}. Known names are "
        f"{list(_KNOWN_OBJECTIVES)}, or a dict blending "
        f"{list(_BLENDABLE_COMPONENTS)}. An unknown name used to be scored as "
        f"'predictive' in silence.")


def _is_geometry_objective(obj) -> bool:
    """True for objectives scored on rep-space GEOMETRY (no downstream model).

    'similarity' (each molecule's property predicted from its five nearest
    neighbours) and 'struct_fid' (MCS) qualify, plus a dict blending them.
    Mirror of find_best_hybrid._is_geometry_objective."""
    if isinstance(obj, dict):
        return bool(obj) and all(k in _GEOMETRY_OBJECTIVES for k in obj)
    return obj in _GEOMETRY_OBJECTIVES


# ============================================================================
# BLENDED OBJECTIVES
# ============================================================================
# An objective may be one name, or a dict of names to weights:
#
#     objective={'similarity': 0.75, 'similarity_sac': 0.25}
#
# Blends are only accepted by the allocators that score EVERY representation
# before choosing (performance_weighted and its named variants, and
# weighted_global). A blend scores the representations under each component
# and blends the SCORES -- every component comes back on one scale, bounded in
# [-1, 1] and higher-is-better, so a weighted mean of them is a score in the
# same units (owner's decision 2026-09-10; it used to blend RANKS). Scoring
# needs the whole set of representations at once.
#
# The greedy family cannot do that -- it keeps or drops one representation at a
# time against a running best. The greedy family is CUT (see the block above
# NAMED_ALLOCATIONS), so nothing is lost by refusing blends there. If it is ever
# revived, it needs its own rule and must not silently fall back.
#
# TWO COMPONENTS ARE BLENDABLE. This machinery was built on 2026-08-26 to blend
# SAC with the nearest-neighbour score; SAC was cut hours later, leaving one
# component. MCS structural fidelity became selectable on 2026-08-27 and is the
# second. A one-entry dict is accepted and behaves exactly like the plain name.
#
# Scoring MCS needs two things the nearest-neighbour score does not:
# `structural_pairs` ({'I','J','mcs'}, row numbers into the WHOLE dataset) and
# `row_index_val` (the whole-dataset row number of each held-out row). Without
# both, every representation scores NaN and the split is arbitrary — so the
# component raises rather than returning a tie.
_BLENDABLE_COMPONENTS = (OBJECTIVE_PREDICTIVE, OBJECTIVE_SIMILARITY,
                         OBJECTIVE_STRUCT_FID)


def normalise_objective(objective):
    """Return (components, is_blend) with components as {name: weight}.

    A plain string comes back as a single-entry dict weighted 1.0, so callers
    have one shape to handle. Raises on an unknown name, a non-positive weight,
    or an empty dict -- a blend that silently drops a component would report a
    number nobody asked for.
    """
    if objective == OBJECTIVE_PREDICTIVE_CORR:
        # Not a plain name for scoring purposes: it is the one-component blend
        # {predictive: 1.0}, and it must take the blend path so the budget is
        # split by the prediction CORRELATION -- Spearman on a regression set,
        # 2*(AUROC-0.5) on a classification one -- rather than by the dataset's
        # own metric. That is what every mixture containing prediction does, so
        # the corner is the limit of the edge rather than a different
        # calculation that happens to sit at its end.
        return {OBJECTIVE_PREDICTIVE: 1.0}, True
    if not isinstance(objective, dict):
        return {objective: 1.0}, False
    if not objective:
        raise ValueError(
            "objective={} is empty. Give at least one of "
            f"{_BLENDABLE_COMPONENTS}, or pass a plain objective name.")
    bad = [k for k in objective if k not in _BLENDABLE_COMPONENTS]
    if bad:
        raise ValueError(
            f"objective blend contains {bad}, which cannot be blended. "
            f"Blendable components are {_BLENDABLE_COMPONENTS}. "
            f"'{OBJECTIVE_PREDICTIVE}' fits a downstream model and is not one "
            f"of them.")
    bad_w = {k: v for k, v in objective.items()
             if not isinstance(v, (int, float)) or v <= 0}
    if bad_w:
        raise ValueError(
            f"objective blend has non-positive weights {bad_w}. A component "
            f"weighted 0 should be left out, not passed as 0.")
    total = float(sum(objective.values()))
    return {k: float(v) / total for k, v in objective.items()}, True


def _score_component(component, X_val_scaled, y_val, groups_val, is_clf,
                     row_index_val=None, structural_pairs=None,
                     X_train_raw=None, X_val_raw=None, y_train=None,
                     internal_evaluator='rf'):
    """One blend component on one already-standardised held-out block.

    Every component comes back on the SAME scale, bounded in [-1, 1] and
    higher-is-better, because the caller takes a weighted mean of them.
    """
    if component == OBJECTIVE_PREDICTIVE:
        # Prediction as a CORRELATION, not as the dataset's own metric: an RMSE
        # cannot be averaged with two Spearman correlations. The fit is handed
        # straight back to _evaluate_allocation on its plain predictive path, so
        # every internal-evaluator branch and every import fallback is reused
        # rather than copied. It scales internally, hence the RAW blocks.
        if X_train_raw is None or y_train is None or X_val_raw is None:
            raise ValueError(
                "blending %r needs the representation's raw TRAINING and "
                "held-out blocks and the training labels, so a model can be "
                "fitted before its predictions are correlated with truth. "
                "allocate_performance_weighted and allocate_weighted_global "
                "pass them; a caller that does not cannot blend prediction."
                % OBJECTIVE_PREDICTIVE)
        try:
            _, y_pred = _evaluate_allocation(
                X_train_raw, X_val_raw, y_train, y_val,
                internal_evaluator=internal_evaluator,
                objective=OBJECTIVE_PREDICTIVE, is_clf=is_clf,
                return_predictions=True,
                # AUROC below needs a RANKING, not thresholded class labels.
                # Read off hard labels it collapses to (sensitivity +
                # specificity) / 2 and carries about half the signal, while the
                # two correlations this is blended with keep all of theirs.
                # Added 2026-09-10 (metric-direction audit).
                graded_predictions_for_clf=True)
            if y_pred is None:
                return -np.inf
            yv = np.asarray(y_val)
            if is_clf:
                from sklearn.metrics import roc_auc_score
                if len(np.unique(yv)) < 2:
                    return -np.inf
                # 2*(AUROC - 0.5): 0 at chance, 1 at perfect, so it sits on the
                # same scale and zero point as the two correlations.
                s_ = 2.0 * (float(roc_auc_score(yv, np.asarray(y_pred))) - 0.5)
            else:
                from scipy.stats import spearmanr as _sp
                r, _p = _sp(yv, np.asarray(y_pred))
                s_ = float(r)
        except Exception:
            return -np.inf
        return s_ if np.isfinite(s_) else -np.inf
    if component == OBJECTIVE_SIMILARITY:
        from kirby.utils.noise_mitigation_methods import activity_readacross
        # missing_groups='raise' on purpose. Without scaffold groups this
        # returned 0.0 for every representation, which is not a failure the
        # caller can see -- it is a tie, and a tie makes the split arbitrary.
        #
        # degenerate='nan' for the same reason one layer down: under 10 held-out
        # rows, one scaffold group, or constant activity, activity_readacross
        # returns 0.0 -- which is also what a real correlation of zero looks
        # like. Every representation then scored 0.0, every weight became 0, and
        # the budget was split EQUALLY under the requested method's name. NaN
        # here becomes -inf below, which the allocator drops instead.
        # find_best_hybrid._readacross_objective_score has returned NaN for
        # these since 2026-08-27; this is the same rule on the budget-split side.
        # is_clf must be forwarded. Without it a classification run split the
        # budget by Spearman while find_best_hybrid._readacross_objective_score
        # scored the configuration by 2*(AUROC-0.5) -- the two agree in ORDER on
        # a binary label, but not in scale, and the prediction branch above is
        # already on the AUROC scale, so a blend would have mixed two zero
        # points. See that function for the measurement behind the choice.
        s = activity_readacross(X_val_scaled, np.asarray(y_val), groups_val,
                                degenerate='nan', is_clf=is_clf)
        return s if np.isfinite(s) else -np.inf
    if component == OBJECTIVE_STRUCT_FID:
        if structural_pairs is None or row_index_val is None:
            raise ValueError(
                "objective 'struct_fid' splits the budget by MCS structural "
                "fidelity, which needs structural_pairs ({'I','J','mcs'}) and "
                "row_index_val (whole-dataset row number of each held-out "
                "row). "
                f"structural_pairs="
                f"{'given' if structural_pairs is not None else 'None'}, "
                f"row_index_val="
                f"{'given' if row_index_val is not None else 'None'}. "
                "Without them every representation scores NaN, which reads as "
                "a tie rather than a failure. create_hybrid slices "
                "row_index_val from its own `row_index`; pass it through.")
        from kirby.find_best_hybrid import _structural_fidelity_score
        s = _structural_fidelity_score(X_val_scaled, row_index_val,
                                       structural_pairs)
        return s if np.isfinite(s) else -np.inf
    raise ValueError(f"unknown geometry component: {component!r}")


def blended_rep_scores(rep_dict_train, rep_dict_val, val_labels, groups_val,
                       components, is_clf=False, row_index_val=None,
                       structural_pairs=None, train_labels=None,
                       internal_evaluator='rf'):
    """Score every representation under a blended geometry objective.

    Each representation is standardised on its own training rows and scored on
    the held-out rows under every component. The component SCORES are then
    combined with the component weights -- _score_component returns all three on
    one scale, bounded in [-1, 1] and higher-is-better, so a weighted mean of
    them is a score in the same units. Returns {rep_name: blended_score}, higher
    is better, which is the same shape allocate_performance_weighted already
    feeds to its weighting scheme -- so every named weighting variant keeps
    working unchanged.

    BLENDS ON SCORES, NOT RANKS -- owner's decision, 2026-09-10
    (FRAMEWORK_FINALIZATION.md 4). It used to rank each component across
    representations first (_rank01) and blend the ranks. Ranking throws away how
    far apart two representations are, and it made the corner of the weight
    triangle -- all the weight on one component -- differ from that component's
    own plain objective, which splits the budget by the raw score. It now agrees.
    Every stored blend result predates this and has to be re-run.
    """
    names = sorted(rep_dict_train)
    per_component = {c: [] for c in components}
    for name in names:
        X_tr = np.nan_to_num(np.clip(rep_dict_train[name], -1e10, 1e10),
                             nan=0.0, posinf=1e10, neginf=-1e10)
        X_va = np.nan_to_num(np.clip(rep_dict_val[name], -1e10, 1e10),
                             nan=0.0, posinf=1e10, neginf=-1e10)
        scaler = StandardScaler().fit(X_tr)
        X_va_s = scaler.transform(X_va)
        for c in components:
            per_component[c].append(
                _score_component(c, X_va_s, val_labels, groups_val, is_clf,
                                 row_index_val=row_index_val,
                                 structural_pairs=structural_pairs,
                                 X_train_raw=X_tr, X_val_raw=X_va,
                                 y_train=train_labels,
                                 internal_evaluator=internal_evaluator))
    # A component that came back non-finite for EVERY representation carries no
    # information about any of them, and adding it would make every blended
    # score -inf -- the blend would silently become nothing at all while still
    # reporting itself under the blend's name.
    for c in components:
        if not any(np.isfinite(v) for v in per_component[c]):
            raise ValueError(
                "blend component %r scored non-finite for all %d "
                "representations on this held-out block, so it separates none "
                "of them. Blending it anyway drops every representation to the "
                "same score and removes the component from the blend without "
                "saying so. The usual cause is a degenerate held-out block -- "
                "one group, under ten rows, or constant activity."
                % (c, len(names)))
    # A single representation that failed ONE component keeps -inf there, which
    # carries through the weighted sum to -inf overall: the allocator drops it
    # rather than scoring it on the components that did work. Same rule as the
    # plain objectives, where a non-finite score is dropped.
    blended = np.zeros(len(names), dtype=np.float64)
    for c, w in components.items():
        blended += w * np.asarray(per_component[c], dtype=np.float64)
    return ({n: float(s) for n, s in zip(names, blended)},
            {c: {n: float(v) for n, v in zip(names, per_component[c])}
             for c in components})


# ============================================================================
# WHAT THE BUDGET IS
# ============================================================================
#
# THE BUDGET IS A MAXIMUM, NOT A TARGET. It is the most features a hybrid may
# use. A hybrid that scores better on fewer features is a BETTER answer, not a
# short one, and the search is expected to return it.
#
# Owner's framing, stated again on 2026-09-10 because it keeps getting lost:
#
#     "I care about finding the best set of features (best hybrid) for a given
#      dataset, model, and budget. That is the goal. A good way to compare
#      against a base rep is with the same budget. [...] If I find a
#      lower-budget alternative for the hybrid that's even better, think of
#      budget as the max."
#
# THREE THINGS FOLLOW, and each one is a rule the code has to keep:
#
# 1. A hybrid may come back NARROWER than its budget. It does whenever a
#    representation's importance method cannot justify its whole quota — see
#    concatenate_features, which stops at the last feature scoring above zero
#    rather than padding the rest out by column order. The real width is
#    recorded per representation as `n_features`, against `n_requested`.
#
# 2. A hybrid must NEVER come back wider than its budget. Anything appended
#    outside the budget breaks the comparison the budget exists to make.
#
# 3. THE COMPARISON IS AT THE SAME BUDGET. A hybrid of N features is measured
#    against a base representation cut to N features, chosen by the same
#    importance method on the same fold — not against that representation at
#    its full native width. A delta between N features and 2048 features mixes
#    the method with the width and cannot be read as either.
#
# The budget LIST is a separate matter: the search reports a winner per budget,
# so a lower budget winning is a result the grid can express, and
# `best_by_model` takes the argmax across all of them.
# ============================================================================

# ============================================================================
# NAMED ALLOCATION REGISTRY
# ============================================================================

# ---------------------------------------------------------------------------
# ⛔ THE GREEDY FAMILY IS CUT. CLOSED 2026-04-09, RE-CONFIRMED 2026-08-25/26.
#
# This covers EVERY greedy name below — 'greedy', 'greedy_forced_s*',
# 'greedy_fill_s*', 'dominant_greedy', 'dom_greedy_f*' — and the three
# undispatched routines further down (allocate_greedy_feature,
# allocate_two_stage, allocate_stability_selection).
#
# DO NOT re-add any of them to DEFAULT_ALLOCATION_METHODS, do not propose them
# for a new study, and do not re-derive this decision. It has been re-derived
# repeatedly and the answer has not changed. Owner's words, 2026-08-25:
# "they were dropped for a reason - especially greedy. Subpar results and
# absurd computation. I think I left them in so I would know not to attempt
# them again" and "I don't want greedy."
#
# They stay in the file as the record of a measured negative result. They are
# still reachable by name so old configs do not crash; nothing selects them.
#
# Two facts recorded here so nobody reports them as a new discovery:
#   - dominant_greedy was never named in a cut commit and won herg_ki and
#     ginkgo_hic outright in f01 tier-1. It is cut anyway, by the decision
#     above. This is not new evidence.
#   - The step_size/patience entries in the greedy rows below are DEAD. They
#     land in **kwargs while the greedy branch reads the create_hybrid
#     parameters of the same name, so all four "forced"/base names run one
#     configuration and all three "fill" names run another. Not fixed,
#     deliberately: fixing it would resurrect cut arms.
# ---------------------------------------------------------------------------

NAMED_ALLOCATIONS = {
    # --- Base methods ---
    'greedy': ('greedy', {}),                    # CUT — see the block above
    'performance_weighted': ('performance_weighted', {}),
    'fixed': ('fixed', {}),

    # --- Softmax temperature variants ---
    'softmax_T0.5': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 0.5}}),
    'softmax_T1': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 1.0}}),
    'softmax_T2': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 2.0}}),
    'softmax_T5': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 5.0}}),
    'softmax_T10': ('performance_weighted', {'weighting_scheme': 'softmax', 'weighting_params': {'temperature': 10.0}}),

    # --- Power law variants ---
    'power_2': ('performance_weighted', {'weighting_scheme': 'power_law', 'weighting_params': {'power': 2.0}}),
    'power_3': ('performance_weighted', {'weighting_scheme': 'power_law', 'weighting_params': {'power': 3.0}}),

    # --- Thresholded variants ---
    'thresh_0.3': ('performance_weighted', {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.3}}),
    'thresh_0.5': ('performance_weighted', {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.5}}),
    'thresh_0.7': ('performance_weighted', {'weighting_scheme': 'thresholded', 'weighting_params': {'alpha': 0.7}}),

    # --- Min-floor variant ---
    'min_floor': ('performance_weighted', {'weighting_scheme': 'min_floor'}),

    # --- Top-k variants ---
    'top_2': ('performance_weighted', {'weighting_scheme': 'top_k', 'weighting_params': {'k': 2}}),
    'top_3': ('performance_weighted', {'weighting_scheme': 'top_k', 'weighting_params': {'k': 3}}),
    'top_5': ('performance_weighted', {'weighting_scheme': 'top_k', 'weighting_params': {'k': 5}}),

    # --- Rank-based variants ---
    'rank_linear': ('performance_weighted', {'weighting_scheme': 'rank_linear'}),
    'rank_exp_d0.1': ('performance_weighted', {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.1}}),
    'rank_exp_d0.3': ('performance_weighted', {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.3}}),
    'rank_exp_d0.5': ('performance_weighted', {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.5}}),

    # --- Score gap variants ---
    'score_gap_0.2': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.2}}),
    'score_gap_0.3': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.3}}),
    'score_gap_0.5': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.5}}),
    'score_gap_0.7': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.7}}),

    # --- Entropy variants ---
    'entropy_0.5': ('performance_weighted', {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 0.5}}),
    'entropy_1.0': ('performance_weighted', {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 1.0}}),
    'entropy_2.0': ('performance_weighted', {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 2.0}}),

    # --- Pairwise-distance redundancy penalty ---
    # Same penalty shape as entropy_*, but redundancy is the Spearman
    # correlation between two representations' z-scored pairwise DISTANCE
    # vectors (the quantity a concatenated distance sums over) instead of the
    # correlation between their per-sample row norms. The penalty multiplies
    # the representation's RANK, so the shift `_sanitize_baseline_scores`
    # applies cannot flatten it. See the 'pairdist_redundancy' branch in
    # _compute_allocation_weights for why both of those differ from entropy_*.
    'pairdist_0.5': ('performance_weighted', {'weighting_scheme': 'pairdist_redundancy', 'weighting_params': {'diversity_penalty': 0.5}}),
    'pairdist_1.0': ('performance_weighted', {'weighting_scheme': 'pairdist_redundancy', 'weighting_params': {'diversity_penalty': 1.0}}),
    'pairdist_2.0': ('performance_weighted', {'weighting_scheme': 'pairdist_redundancy', 'weighting_params': {'diversity_penalty': 2.0}}),
    # Score base instead of rank base. Same redundancy quantity as the three
    # above, same base quantity as entropy_*. Registered so an experiment can
    # tell a rank-base effect apart from a pairwise-distance-redundancy effect
    # rather than reading one difference for the other.
    'pairdist_score_0.5': ('performance_weighted', {'weighting_scheme': 'pairdist_redundancy', 'weighting_params': {'diversity_penalty': 0.5, 'base': 'score'}}),
    'pairdist_score_1.0': ('performance_weighted', {'weighting_scheme': 'pairdist_redundancy', 'weighting_params': {'diversity_penalty': 1.0, 'base': 'score'}}),
    'pairdist_score_2.0': ('performance_weighted', {'weighting_scheme': 'pairdist_redundancy', 'weighting_params': {'diversity_penalty': 2.0, 'base': 'score'}}),

    # --- CUT: greedy with early stopping. See the ⛔ block at the top. ---
    'greedy_forced_s5': ('greedy', {'step_size': 5, 'patience': None}),
    'greedy_forced_s10': ('greedy', {'step_size': 10, 'patience': None}),
    'greedy_forced_s25': ('greedy', {'step_size': 25, 'patience': None}),

    # --- CUT: greedy with budget fill. See the ⛔ block at the top. ---
    'greedy_fill_s5': ('greedy', {'step_size': 5, 'budget_fill': True}),
    'greedy_fill_s10': ('greedy', {'step_size': 10, 'budget_fill': True}),
    'greedy_fill_s25': ('greedy', {'step_size': 25, 'budget_fill': True}),

    # --- CUT: dominant greedy. See the ⛔ block at the top. ---
    'dominant_greedy': ('dominant_greedy', {}),
    'dom_greedy_f0.7': ('dominant_greedy', {'dominant_fraction': 0.7}),
    'dom_greedy_f0.9': ('dominant_greedy', {'dominant_fraction': 0.9}),

    # --- Weighted global-pooling (ADDITIVE; see allocate_weighted_global) ---
    # MEASURED default = 'linear' (performance_weighted family) under the full
    # roster. pool_total_mult default 2. Scheme/pool_total_mult variants ride in
    # via the _extra dict exactly like the softmax_/power_ entries.
    'weighted_global': ('weighted_global', {'weighting_scheme': 'linear', 'pool_total_mult': 2}),
    'wglobal_power2': ('weighted_global', {'weighting_scheme': 'power_law', 'weighting_params': {'power': 2.0}, 'pool_total_mult': 2}),
    'wglobal_pt4': ('weighted_global', {'weighting_scheme': 'linear', 'pool_total_mult': 4}),

    # Legacy aliases for backward compatibility with old experiment configs
    'rank_exp': ('performance_weighted', {'weighting_scheme': 'rank_exp', 'weighting_params': {'decay': 0.3}}),
    'score_gap': ('performance_weighted', {'weighting_scheme': 'score_gap', 'weighting_params': {'gap': 0.3}}),
    'entropy': ('performance_weighted', {'weighting_scheme': 'entropy', 'weighting_params': {'diversity_penalty': 1.0}}),
}


# ============================================================================
# GREEDY ALLOCATION
# ============================================================================

def allocate_greedy_forward(rep_dict_train, rep_dict_val, train_labels, val_labels,
                           total_budget=100, step_size=10, patience=3,
                           internal_evaluator='match',
                           model=None, metric_fn=None, lower_is_better=False,
                           greedy_tolerance=0.0,
                           precomputed_importance=None,
                           objective=OBJECTIVE_PREDICTIVE, is_clf=False):
    """
    Greedy forward feature allocation with plateau detection.

    Iteratively adds features from the representation that improves validation
    performance most, stopping when performance plateaus or budget is exhausted.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        step_size: Features to add per iteration (default: 10)
        patience: Early stopping patience, None to disable (default: 3)
        model: Optional sklearn estimator for metric-aware evaluation.
        metric_fn: Optional callable(y_true, y_pred) -> float.
        lower_is_better: Whether lower metric values are better (default: False).
        greedy_tolerance: Absolute tolerance for patience tracking (default: 0.0).
            When > 0, iterations that score within tolerance of the best are
            treated as non-degrading and reset the patience counter.

    Returns:
        tuple: (best_allocation, history, final_feature_info)
            - best_allocation: Dict of {rep_name: n_features} for best-performing iteration
            - history: List of iteration details
            - final_feature_info: Feature selection info matching best_allocation
    """
    allocation = {rep: 0 for rep in rep_dict_train.keys()}
    remaining_budget = total_budget

    best_val_r2 = -np.inf
    best_allocation = None
    best_feature_info = {}
    steps_without_improvement = 0
    history = []

    # Cache RF importance scores per rep — computed once, reused every iteration.
    # Importance depends only on (rep features, labels) which are constant across
    # the greedy loop. The cache is local to this call and discarded on return.
    _importance_cache = {}

    iteration = 0
    while remaining_budget > 0:
        iteration += 1
        iter_best_gain = -np.inf
        iter_best_rep = None
        iter_best_feat_info = None
        iter_best_trial_allocation = None

        # Try adding step_size features from each rep
        for rep_name in rep_dict_train.keys():
            # Check if rep is already maxed out
            max_features = rep_dict_train[rep_name].shape[1]
            if allocation[rep_name] >= max_features:
                continue  # Already using all features from this rep

            trial_allocation = allocation.copy()
            # Add step_size, but cap at max_features AND remaining_budget
            features_to_add = min(step_size, max_features - allocation[rep_name], remaining_budget)
            if features_to_add <= 0:
                continue
            trial_allocation[rep_name] = allocation[rep_name] + features_to_add

            # Create hybrid on train and apply SAME features to val
            X_train_trial, feat_info = _create_hybrid_with_allocation(
                rep_dict_train, train_labels, trial_allocation,
                importance_cache=_importance_cache,
                precomputed_importance=precomputed_importance,
            )
            X_val_trial = apply_feature_selection(rep_dict_val, feat_info)
            
            # Evaluate on validation
            r2 = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels,
                                     model=model, metric_fn=metric_fn,
                                     lower_is_better=lower_is_better,
                                     internal_evaluator=internal_evaluator,
                                     objective=objective, is_clf=is_clf)

            if r2 > iter_best_gain:
                iter_best_gain = r2
                iter_best_rep = rep_name
                iter_best_feat_info = feat_info
                iter_best_trial_allocation = trial_allocation
        
        # Stopping conditions
        if iter_best_rep is None:
            break
        
        # Commit this iteration's best choice
        features_added = sum(iter_best_trial_allocation.values()) - sum(allocation.values())
        allocation = iter_best_trial_allocation
        remaining_budget -= features_added
        
        # Track global best (for patience-based early stopping)
        if iter_best_gain > best_val_r2:
            best_val_r2 = iter_best_gain
            best_allocation = allocation.copy()
            best_feature_info = iter_best_feat_info
            steps_without_improvement = 0
        elif greedy_tolerance > 0 and iter_best_gain >= best_val_r2 - greedy_tolerance:
            # Within tolerance of best — increment at half rate to allow
            # plateau exploration without preventing convergence
            steps_without_improvement += 0.5
        else:
            steps_without_improvement += 1
        
        history.append({
            'iteration': iteration,
            'rep': iter_best_rep,
            'allocation': allocation.copy(),
            'val_r2': iter_best_gain
        })
        
        # Patience-based early stopping
        if patience is not None and steps_without_improvement >= patience:
            break
    
    # If we never found anything, raise an error
    if best_allocation is None:
        raise ValueError(
            "Greedy allocation found no valid feature combinations. "
            "Check that representations have features and data is valid."
        )
    
    return best_allocation, history, best_feature_info


# ============================================================================
# TESTED AND CUT — DELIBERATELY NOT DISPATCHED. DO NOT WIRE BACK IN.
#
# allocate_greedy_feature is kept as a record of a measured negative result, not
# as a candidate. create_hybrid's allocation_method branch does not route to it,
# and adding a branch would undo a decision the numbers below already settled.
# Dispatch was removed 2026-03-29..04-03; see PIPELINE_READINESS_TODO.md
# ("Dead strategies removed ... dispatch removed, defs kept").
#
# MEASURED (best score per strategy, read out of the result JSONs 2026-08-25):
#   f01 tests/results/framework/f01_allocation_strategies/
#       herg 21/21 · tcga_luad_vs_lusc 21/21 · tcga_brca_pam50 17/17
#       heprt 20/21 · abagym_EGFR 19/21 · tresanco 17/21 · fcrnrt 11/17
#       ginkgo_hic 2/21  <- the single exception across both experiments
#   f12 tests/results/framework/f12_allocation_v2/  (34 strategies)
#       herg 34/34 · flab_pk_jain_fcrnrt 34/34 · caco2 33/34
#       abagym_EGFR 32/34 · buchwald_hartwig 29/34
#
# Caveat on the above, so nobody over-reads it: f12 ran at max_folds=1 and f01
# tier-1 is a single fold at one budget and one model. These are screening
# ranks, not nested_cv_evaluate deltas.
#
# On cost: in f01 its wall time is roughly EQUAL to performance_weighted. The
# case against it is the score, not the compute. (The 9-11x cost blowup that
# gets attributed to "the greedy family" belongs to allocate_stability_selection
# below, not to this function.)
# ============================================================================
def allocate_greedy_feature(rep_dict_train, rep_dict_val, train_labels, val_labels,
                            total_budget=100, patience=5, importance_method='treeshap',
                            internal_evaluator='lightgbm',
                            model=None, metric_fn=None, lower_is_better=False):
    """
    Feature-level greedy allocation. TESTED AND CUT — see the block above.

    Unlike rep-based greedy which adds chunks of features from the best rep,
    this adds one feature at a time from whichever rep has the best next feature.
    More precise but slower.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        patience: Early stopping patience (default: 5)
        importance_method: Method for ranking features ('treeshap', 'permutation', etc.)

    Returns:
        tuple: (allocation, history, feature_info)
    """
    # Use configurable importance method for ranking
    all_importance = compute_feature_importance(
        rep_dict_train, train_labels, method=importance_method
    )

    # Get importance ranking for each rep
    rep_importance = {}
    for rep_name, importances in all_importance.items():
        # Indices sorted by importance (descending)
        ranked_indices = np.argsort(importances)[::-1]
        rep_importance[rep_name] = {
            'ranked_indices': ranked_indices,
            'scores': importances,
            'next_idx': 0,  # Pointer to next feature to consider
        }

    # Track selected features per rep
    selected = {rep: [] for rep in rep_dict_train.keys()}

    best_val_score = -np.inf
    best_selected = None
    best_feature_info = None
    steps_without_improvement = 0
    history = []

    for iteration in range(total_budget):
        iter_best_score = -np.inf
        iter_best_rep = None
        iter_best_feature_idx = None

        # Try adding the next best feature from each rep
        for rep_name in rep_dict_train.keys():
            info = rep_importance[rep_name]

            # Skip if we've used all features from this rep
            if info['next_idx'] >= len(info['ranked_indices']):
                continue

            # Get next feature to try
            feature_idx = info['ranked_indices'][info['next_idx']]

            # Build trial feature set
            trial_selected = {r: list(s) for r, s in selected.items()}
            trial_selected[rep_name].append(feature_idx)

            # Create hybrid with trial selection
            X_train_trial, X_val_trial = _build_from_selection(
                rep_dict_train, rep_dict_val, trial_selected
            )

            if X_train_trial.shape[1] == 0:
                continue

            # Evaluate
            score = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels,
                                         model=model, metric_fn=metric_fn,
                                         lower_is_better=lower_is_better,
                                         internal_evaluator=internal_evaluator)

            if score > iter_best_score:
                iter_best_score = score
                iter_best_rep = rep_name
                iter_best_feature_idx = feature_idx

        # No valid feature found
        if iter_best_rep is None:
            break

        # Commit the best feature
        selected[iter_best_rep].append(iter_best_feature_idx)
        rep_importance[iter_best_rep]['next_idx'] += 1

        # Track best overall
        if iter_best_score > best_val_score:
            best_val_score = iter_best_score
            best_selected = {r: list(s) for r, s in selected.items()}
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        history.append({
            'iteration': iteration + 1,
            'rep': iter_best_rep,
            'feature_idx': iter_best_feature_idx,
            'val_score': iter_best_score,
            'allocation': {r: len(s) for r, s in selected.items()},
        })

        # Early stopping
        if patience and steps_without_improvement >= patience:
            break

    if best_selected is None:
        raise ValueError("Greedy feature allocation found no valid features.")

    # Build final feature_info from best_selected
    feature_info = {}
    for rep_name, indices in best_selected.items():
        if len(indices) == 0:
            continue
        indices_arr = np.array(indices)
        scores = rep_importance[rep_name]['scores'][indices_arr]
        feature_info[rep_name] = {
            'selected_indices': indices_arr,
            'importance_scores': scores,
            'n_features': len(indices),
        }

    allocation = {r: len(s) for r, s in best_selected.items()}
    feature_info['allocation'] = allocation

    return allocation, history, feature_info


def _build_from_selection(rep_dict_train, rep_dict_val, selected):
    """Helper: Build train/val arrays from explicit feature selection."""
    train_parts = []
    val_parts = []

    for rep_name, indices in selected.items():
        if len(indices) == 0:
            continue
        train_parts.append(rep_dict_train[rep_name][:, indices])
        val_parts.append(rep_dict_val[rep_name][:, indices])

    if not train_parts:
        return np.array([]).reshape(rep_dict_train[list(rep_dict_train.keys())[0]].shape[0], 0), \
               np.array([]).reshape(rep_dict_val[list(rep_dict_val.keys())[0]].shape[0], 0)

    return np.hstack(train_parts), np.hstack(val_parts)


# ============================================================================
# BUDGET-SPLIT ORIGIN — one normalisation rule per metric
#
# FRAMEWORK_FINALIZATION.md §3.5, decided 2026-09-10 (Shape B), specified in
# BUDGET_SPLIT_NORMALIZATION_HANDOFF_2026_09_09.md. Built 2026-09-10.
#
# A representation's share of the budget is proportional to its score. That only
# means something if zero on the score means "contributes nothing". Spearman and
# MCC have that property. Negative RMSE does not — its zero is a perfect
# prediction. AUROC does not — chance is 0.5.
#
# What ran before: _sanitize_baseline_scores shifted the vector up by its own
# minimum, but only when the minimum was negative. So on an rmse dataset every
# score is a negated error, the shift always fired, and the WORST representation
# landed exactly on zero and got no features; on a spearman or auc dataset
# nothing shifted and the worst representation kept a share proportional to its
# raw score. Measured on four representations under `linear`: rmse split
# 43/35/22/0 features per 100, spearman split 27/26/25/23.
#
# What runs now: a skill score against a reference model, keyed on the metric.
# 1 means perfect, 0 means no better than the reference, and everything is on
# one scale whatever the dataset is scored by.
# ============================================================================

# §3.7 (a) — a negative spearman or mcc. 'clip' is what §3.5's rule says and
# what f32 measured; 'shift' is what ran before 2026-09-10. One line to flip.
ORIGIN_NEGATIVE_CORRELATION = 'clip'

# §3.7 (b) — a null model with zero error, which makes the skill score
# undefined. 'warn' keeps the run going on the old shift for that vector and
# says which dataset and fold took it; 'raise' costs the fold the way
# _require_scored_reps does. One line to flip.
ORIGIN_NULL_MODEL_PERFECT = 'warn'

# Metrics whose zero already means "contributes nothing".
_ORIGIN_ZERO_IS_NOTHING = ('spearman', 'pearson', 'mcc', 'r2',
                           'sac', 'readacross', 'struct_fidelity')
# Metrics whose zero is chance at 0.5.
_ORIGIN_CHANCE_AT_HALF = ('auc', 'ci')
# Error metrics: skill score against a reference model's error.
_ORIGIN_ERROR = ('rmse', 'mae')
# Classification rates whose reference is always predicting the majority
# training class (FRAMEWORK_FINALIZATION.md §3.5 (b)).
_ORIGIN_MAJORITY_CLASS = ('f1_weighted', 'accuracy')


def metric_name_for_fn(metric_fn):
    """Name of `metric_fn` in BUILTIN_METRICS, or None when it is not in it.

    Identity, not a string guess — the same lookup _scorer_for_metric_fn uses.
    """
    if metric_fn is None:
        return None
    try:
        from kirby.find_best_hybrid import BUILTIN_METRICS
    except Exception:
        return None
    for _name, (_fn, _lower) in BUILTIN_METRICS.items():
        if _fn is metric_fn:
            return _name
    return None


def origin_null_reference(metric_name, y_train, y_val):
    """The reference model's score for `metric_name`, or None when none is needed.

    rmse: the RMSE of predicting the training mean on the held-out labels.
    mae: the MAE of predicting the training median.
    f1_weighted / accuracy: the score of always predicting the majority training
    class. One number from the training labels, no random draws — §3.5 (b).
    """
    if metric_name not in _ORIGIN_ERROR + _ORIGIN_MAJORITY_CLASS:
        return None
    y_tr = np.asarray(y_train, dtype=float) if metric_name in _ORIGIN_ERROR \
        else np.asarray(y_train)
    y_va = np.asarray(y_val, dtype=float) if metric_name in _ORIGIN_ERROR \
        else np.asarray(y_val)
    if y_tr.size == 0 or y_va.size == 0:
        return None
    try:
        if metric_name == 'rmse':
            return float(np.sqrt(np.mean((y_va - float(np.mean(y_tr))) ** 2)))
        if metric_name == 'mae':
            return float(np.mean(np.abs(y_va - float(np.median(y_tr)))))
        vals, counts = np.unique(y_tr, return_counts=True)
        majority = vals[int(np.argmax(counts))]
        y_null = np.full(y_va.shape, majority)
        if metric_name == 'accuracy':
            from sklearn.metrics import accuracy_score
            return float(accuracy_score(y_va, y_null))
        from sklearn.metrics import f1_score
        return float(f1_score(y_va, y_null, average='weighted'))
    except Exception:
        return None


def origin_normalise_scores(scores_higher_is_better, metric_name,
                            null_reference=None, where=''):
    """Put per-representation scores on a scale where 0 means "contributes nothing".

    Input is the higher-is-better form every allocator already receives — a raw
    correlation, or minus the error for rmse/mae. Output is non-negative, so the
    shift in _sanitize_baseline_scores becomes a no-op on it.

    Returns None when the metric is not one the rule covers, or when the
    reference model is perfect and ORIGIN_NULL_MODEL_PERFECT is 'warn'. None
    means "use what ran before", so an unrecognised metric is never silently
    reshaped.
    """
    if not scores_higher_is_better or metric_name is None:
        return None

    def _keep_non_finite(out):
        # A representation nobody could score stays non-finite, so the guard and
        # the sanitiser see a failure rather than a zero.
        return {rn: (out[rn] if np.isfinite(scores_higher_is_better[rn])
                     else scores_higher_is_better[rn])
                for rn in scores_higher_is_better}

    if metric_name in _ORIGIN_ZERO_IS_NOTHING:
        if ORIGIN_NEGATIVE_CORRELATION != 'clip':
            return None
        return _keep_non_finite({rn: max(0.0, float(s)) if np.isfinite(s) else s
                                 for rn, s in scores_higher_is_better.items()})

    if metric_name in _ORIGIN_CHANCE_AT_HALF:
        return _keep_non_finite(
            {rn: (max(0.0, 2.0 * (float(s) - 0.5)) if np.isfinite(s) else s)
             for rn, s in scores_higher_is_better.items()})

    if metric_name in _ORIGIN_ERROR:
        if null_reference is None or not np.isfinite(null_reference):
            return None
        if null_reference <= 1e-12:
            if ORIGIN_NULL_MODEL_PERFECT == 'raise':
                raise ValueError(
                    "%s: the null model for %r has zero error on this held-out "
                    "block, so a skill score is undefined and every "
                    "representation would weigh zero. Set "
                    "ORIGIN_NULL_MODEL_PERFECT='warn' to fall back to the shift."
                    % (where or 'origin_normalise_scores', metric_name))
            warn_once(
                "%s: the null model for %r has zero error on this held-out "
                "block (every held-out label equals the training mean or "
                "median), so the budget split falls back to the pre-2026-09-10 "
                "shift for this vector." % (where or 'budget split', metric_name))
            return None
        # score is MINUS the error. 1 - error/null_error: 1 perfect, 0 no better
        # than the reference.
        return _keep_non_finite(
            {rn: (max(0.0, 1.0 - (-float(s)) / float(null_reference))
                  if np.isfinite(s) else s)
             for rn, s in scores_higher_is_better.items()})

    if metric_name in _ORIGIN_MAJORITY_CLASS:
        if null_reference is None or not np.isfinite(null_reference):
            return None
        if null_reference >= 1.0 - 1e-12:
            if ORIGIN_NULL_MODEL_PERFECT == 'raise':
                raise ValueError(
                    "%s: always predicting the majority training class already "
                    "scores %r perfectly on this held-out block, so a skill "
                    "score is undefined." % (where or 'origin', metric_name))
            warn_once(
                "%s: always predicting the majority training class already "
                "scores %r perfectly on this held-out block, so the budget "
                "split falls back to the pre-2026-09-10 shift for this vector."
                % (where or 'budget split', metric_name))
            return None
        denom = 1.0 - float(null_reference)
        return _keep_non_finite(
            {rn: (max(0.0, (float(s) - float(null_reference)) / denom)
                  if np.isfinite(s) else s)
             for rn, s in scores_higher_is_better.items()})

    return None


def _apply_origin_rule(baseline_scores, metric_fn, train_labels, val_labels,
                       where=''):
    """Origin-normalise in place if the metric is one the rule covers."""
    _mn = metric_name_for_fn(metric_fn)
    if _mn is None:
        return baseline_scores
    _null = origin_null_reference(_mn, train_labels, val_labels)
    _out = origin_normalise_scores(baseline_scores, _mn, _null, where=where)
    return baseline_scores if _out is None else _out


def _sanitize_baseline_scores(baseline_scores):
    """Shift scores to non-negative, replacing non-finite scores with 0.

    Non-finite scores arise when `_evaluate_allocation` returns -inf on a
    numerical failure (e.g. a model raising LinAlgError on a degenerate
    feature matrix). If such a score
    reaches the `s - min_val` shift downstream, it poisons the weights
    (-inf - -inf = NaN), which then crashes the proportional allocation
    when `total_w` becomes NaN. Drop infeasible reps from the min
    calculation and assign them weight 0 afterwards.
    """
    if not baseline_scores:
        return baseline_scores
    finite = {rn: s for rn, s in baseline_scores.items() if np.isfinite(s)}
    if not finite:
        # All reps infeasible — equal weight so allocation doesn't blow up.
        return {rn: 1.0 for rn in baseline_scores}
    min_val = min(finite.values())
    shift = -min_val if min_val < 0 else 0.0
    return {rn: (s + shift if np.isfinite(s) else 0.0)
            for rn, s in baseline_scores.items()}


# Which internal evaluators have already announced that they could not fit and
# were substituted. One line per (asked, used) pair per process: the substitution
# repeats on every representation of every fold, and a line each would bury the
# run's own output.
_EVALUATOR_FALLBACKS_SEEN = set()


def _announce_evaluator_fallback(asked, used, err):
    """Say once that an internal evaluator was substituted, and for what.

    An evaluator that cannot fit -- no torch on the node is the usual reason --
    used to be replaced in silence, while every stored result went on recording
    the evaluator that was ASKED for. A run could therefore split every budget
    with a random forest and report itself as dnn throughout. f31 found this on
    2026-09-06 and left it unfixed; said out loud since 2026-09-10.
    """
    key = (asked, used)
    if key in _EVALUATOR_FALLBACKS_SEEN:
        return
    _EVALUATOR_FALLBACKS_SEEN.add(key)
    warn_once(
        f"internal_evaluator={asked!r} could not fit, so every budget split "
        f"asking for it is being computed with {used!r} instead "
        f"({type(err).__name__}: {err}). Results still RECORD {asked!r}. "
        f"Said once per process.")


class EvaluatorLibraryMissing(ImportError):
    """The library an internal evaluator is named for is not installed here."""


def _refuse_missing_evaluator_library(asked, module, err):
    """Stop the run when the library an evaluator is named for is absent.

    These branches used to substitute: ask for lightgbm on a node without it
    and the budget split came back from xgboost, or from a random forest, and
    was written to a file named for lightgbm with the requested name in its
    metadata. A later run on a node that HAS the library matched that file on
    every key and read the substitute's numbers as its own. Nothing on disk
    recorded which library actually fitted, so it could not be detected
    afterwards -- not from the filename, not from the metadata, not from the
    stored row.

    Owner decision 2026-09-10: refuse. A node with the wrong environment costs
    a resubmit; a file named lightgbm that does not hold lightgbm costs a
    result nobody can trust. This fires ONLY on a missing library. A fit that
    fails on one fold's rows is a different thing and still falls back, out
    loud, through _announce_evaluator_fallback.
    """
    raise EvaluatorLibraryMissing(
        f"internal_evaluator={asked!r} needs {module!r}, which cannot be "
        f"imported on this machine ({type(err).__name__}: {err}). This used to "
        f"fall back to another library and store the score under the name you "
        f"asked for, so the substitution was undetectable afterwards. Either "
        f"install {module!r} here, submit to a partition that has it, or ask "
        f"for an evaluator you can actually run.")


def _require_scored_reps(baseline_scores, objective, where):
    """Raise when no representation could be scored on the held-out block.

    Every representation coming back non-finite is not a tie, it is a slice on
    which the quantity cannot be formed: under ten held-out rows, one scaffold
    group, or constant activity. The scores then all shift to the same number,
    every weight becomes zero, and the budget is split EQUALLY -- or the
    remainder is handed out in roster order -- under the name of whichever
    allocation method was asked for. On tz_her2_binary the held-out block
    always holds one CDRH3 edit-distance group, so every similarity run there
    took that path and no allocation method ever competed.
    """
    if not baseline_scores:
        return
    if any(np.isfinite(s) for s in baseline_scores.values()):
        return
    raise ValueError(
        "%s: not one of the %d representations could be scored under "
        "objective=%r on this held-out block, so there is nothing to split the "
        "budget by, and this configuration is refused. "
        "GET BETTER GROUPS. This fold's held-out block holds one group, or "
        "under ten rows, or one activity value, so it cannot say anything "
        "about any representation. That is how the dataset is grouped, not a "
        "property of the recipe: the grouping is too coarse, or one group is "
        "big enough to take a whole fold. Splitting equally instead would be "
        "filed under the requested allocation method's name and would be "
        "indistinguishable from a method that chose an equal split."
        % (where, len(baseline_scores), objective))


def _abs_pearson(a, b):
    """|Pearson r| between two vectors; 0.0 when either is constant."""
    r = np.corrcoef(a, b)[0, 1]
    return abs(r) if not np.isnan(r) else 0.0


def _abs_spearman(a, b):
    """|Spearman rho| between two vectors; 0.0 when either is constant."""
    from scipy.stats import spearmanr
    r, _p = spearmanr(a, b)
    if r is None or not np.isfinite(r):
        return 0.0
    return abs(float(r))


def _rownorm_vectors(rep_dict_train, names):
    """Per-sample row norm of each representation, standardised.

    The redundancy quantity of the 'entropy' weighting scheme: one
    n_samples-length vector per representation, whatever its feature count.
    On molecular features it is close to a molecule-size proxy, which is the
    reason the 'pairdist_redundancy' scheme exists.
    """
    out = {}
    for name in names:
        X = rep_dict_train[name]
        X_clean = np.nan_to_num(np.clip(X, -1e10, 1e10), nan=0.0)
        v = np.linalg.norm(X_clean, axis=1)
        std = np.std(v)
        if std > 0:
            v = (v - v.mean()) / std
        out[name] = v
    return out


def _pairdist_vectors(rep_dict_train, names, n_pairs=5000, seed=42):
    """Pairwise Euclidean distance of each representation over shared row pairs.

    The redundancy quantity of the 'pairdist_redundancy' weighting scheme.
    Every representation is z-scored per feature, then one distance is computed
    for each of `n_pairs` off-diagonal row pairs. The pairs are drawn once and
    reused for every representation, so two distance vectors are comparable
    element by element.

    `_sample_offdiag_pairs` is the sampler the geometry scorer already uses
    (kirby.utils.noise_mitigation_methods) -- same function, same seed, so
    redundancy is measured on the same pairs the objective is scored on.

    Returns {rep_name: distance_vector}, or None when there are too few rows
    to sample any pair.
    """
    from kirby.utils.noise_mitigation_methods import _sample_offdiag_pairs

    rows = {rep_dict_train[n].shape[0] for n in names}
    if len(rows) != 1:
        raise ValueError(
            f"_pairdist_vectors: representations have different row counts "
            f"{sorted(rows)}. The sampled pairs index rows, so they must align.")
    n_rows = rows.pop()
    if n_rows < 3:
        return None
    pi, pj = _sample_offdiag_pairs(n_rows, n_pairs, seed)
    if len(pi) == 0:
        return None

    out = {}
    for name in names:
        X = np.asarray(rep_dict_train[name], dtype=np.float64)
        X_clean = np.nan_to_num(np.clip(X, -1e10, 1e10), nan=0.0)
        # Standardise BEFORE the distance. A constant feature gets scale 1 and
        # contributes 0 to every pair, which is what StandardScaler does.
        Xs = StandardScaler().fit_transform(X_clean)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        out[name] = np.linalg.norm(Xs[pi] - Xs[pj], axis=1)
    return out


def _penalise_against_accepted(sorted_items, base_val, vectors,
                               diversity_penalty, corr_fn):
    """Sorted-by-score, penalise-against-earlier-accepted weighting.

    Walks representations in descending score order. Each one's redundancy is
    the largest |correlation| between its own vector and the vector of any
    representation already walked past, and its weight is

        base_val[name] * max(0, 1 - diversity_penalty * redundancy)

    Shared by 'entropy' and 'pairdist_redundancy' so the only things that
    differ between the two schemes are `vectors` (what redundancy is measured
    on), `corr_fn` (how two of those vectors are compared) and `base_val`
    (what the penalty multiplies).

    Returns (weights, redundancy), both {rep_name: float}, in walk order.
    """
    accepted = []
    weights = {}
    redundancy = {}
    for name, _score in sorted_items:
        v = vectors[name]
        if not accepted:
            max_corr = 0.0
        else:
            max_corr = max(corr_fn(v, vectors[prev]) for prev in accepted)
        penalty = max(0.0, 1.0 - diversity_penalty * max_corr)
        weights[name] = base_val[name] * penalty
        redundancy[name] = float(max_corr)
        accepted.append(name)
    return weights, redundancy


def redundancy_between_reps(rep_dict_train, measure='pairdist',
                            n_pairs=5000, seed=42):
    """Full redundancy matrix between representations under one measure.

    Read-only. Nothing in the allocation path calls this -- it exists so an
    experiment can report the redundancy numbers themselves rather than only
    the weights they produce.

    measure='rownorm'  -> |Pearson r| between standardised per-sample row
                          norms. The quantity the 'entropy' scheme uses.
    measure='pairdist' -> |Spearman rho| between z-scored pairwise distance
                          vectors over shared row pairs. The quantity the
                          'pairdist_redundancy' scheme uses.

    Returns {'names': [...], 'matrix': ndarray (n_reps, n_reps), symmetric,
    1.0 on the diagonal}, or None when pairdist has too few rows to sample.
    """
    names = sorted(rep_dict_train)
    if measure == 'rownorm':
        vectors, corr_fn = _rownorm_vectors(rep_dict_train, names), _abs_pearson
    elif measure == 'pairdist':
        vectors = _pairdist_vectors(rep_dict_train, names, n_pairs=n_pairs,
                                    seed=seed)
        corr_fn = _abs_spearman
        if vectors is None:
            return None
    else:
        raise ValueError(
            f"redundancy_between_reps: measure={measure!r}. "
            f"Use 'rownorm' or 'pairdist'.")
    n = len(names)
    M = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr_fn(vectors[names[i]], vectors[names[j]]))
            M[i, j] = M[j, i] = r
    return {'names': names, 'matrix': M}


def _compute_allocation_weights(baseline_scores, scheme='linear', params=None,
                                rep_dict_train=None):
    """Compute allocation weights from baseline scores using various schemes.

    Args:
        baseline_scores: Dict of {rep_name: score} (already clipped to >= 0).
        scheme: Weighting scheme name.
        params: Dict of scheme-specific parameters.
        rep_dict_train: Optional dict of {rep_name: train_features}. Required
            only for schemes that need the raw features ('entropy' and
            'pairdist_redundancy').

    Returns:
        Dict of {rep_name: weight} (non-negative, not necessarily normalised).
    """
    params = params or {}

    if scheme == 'linear':
        return dict(baseline_scores)

    elif scheme == 'softmax':
        temperature = params.get('temperature', 1.0)
        scores = np.array(list(baseline_scores.values()))
        # Subtract max for numerical stability
        exp_scores = np.exp((scores - scores.max()) / max(temperature, 1e-10))
        return {k: float(w) for k, w in zip(baseline_scores.keys(), exp_scores)}

    elif scheme == 'power_law':
        power = params.get('power', 2.0)
        return {k: float(v ** power) for k, v in baseline_scores.items()}

    elif scheme == 'thresholded':
        alpha = params.get('alpha', 0.5)
        max_score = max(baseline_scores.values()) if baseline_scores else 0
        threshold = alpha * max_score
        return {k: (v if v >= threshold else 0.0) for k, v in baseline_scores.items()}

    elif scheme == 'top_k':
        k = params.get('k', 3)
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True)
        top_names = {name for name, _ in sorted_reps[:k]}
        return {name: (score if name in top_names else 0.0)
                for name, score in baseline_scores.items()}

    elif scheme == 'min_floor':
        # Special: handled in allocate_performance_weighted directly
        return dict(baseline_scores)

    elif scheme == 'rank_linear':
        # Weights by rank position: best rep gets n, worst gets 1.
        # Score magnitudes are ignored — prevents clusters of similar scores
        # from diluting budget across mediocre reps.
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1])
        return {name: float(rank) for rank, (name, _)
                in enumerate(sorted_reps, start=1)}

    elif scheme == 'rank_exp':
        # Exponential rank decay: exp(-rank * decay).  Concentrates budget on
        # top few reps with smooth falloff — no hard cutoff like top_k but
        # more aggressive than rank_linear.  decay controls steepness.
        decay = params.get('decay', 0.3)
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1],
                             reverse=True)
        return {name: float(np.exp(-rank * decay))
                for rank, (name, _) in enumerate(sorted_reps)}

    elif scheme == 'score_gap':
        # Zero out any rep whose score is more than `gap` below the best.
        # gap is a fraction of best_score (e.g. 0.3 = within 30%).
        gap_frac = params.get('gap', 0.3)
        best_score = max(baseline_scores.values()) if baseline_scores else 0
        gap = gap_frac * best_score if best_score > 0 else 0
        return {k: (v if best_score - v <= gap else 0.0)
                for k, v in baseline_scores.items()}

    elif scheme == 'entropy':
        # Penalize reps that are redundant with higher-scoring reps.
        # Walk reps in descending score order. Each rep's weight is its
        # baseline score multiplied by (1 - max_correlation_with_any_
        # already-accepted_rep).  This down-weights reps that duplicate
        # information already captured by a better rep.
        #
        # Redundancy is measured via correlation of sample-wise L2 norms
        # (always n_samples-length, regardless of feature dimensionality).
        #
        # HISTORICAL ARM, UNCHANGED ON PURPOSE. Its three known defects are
        # written out under 'pairdist_redundancy' below and are fixed ONLY
        # there, so a study can separate "penalising redundancy is a bad idea"
        # from "the row-norm way of measuring redundancy is a bad measurement".
        # Every number this scheme has ever produced (f01, f12) still
        # reproduces.
        if rep_dict_train is None:
            return dict(baseline_scores)
        diversity_penalty = params.get('diversity_penalty', 1.0)
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1],
                             reverse=True)
        vectors = _rownorm_vectors(rep_dict_train, [n for n, _ in sorted_reps])
        weights, _redundancy = _penalise_against_accepted(
            sorted_reps, dict(baseline_scores), vectors,
            diversity_penalty, _abs_pearson)
        return weights

    elif scheme == 'pairdist_redundancy':
        # Same sorted-by-score, penalise-against-earlier-accepted structure as
        # 'entropy'. Two things differ, both deliberate.
        #
        # 1. WHAT REDUNDANCY IS MEASURED ON. 'entropy' correlates each
        #    representation's per-sample row norm ||x_i||. On molecular
        #    features the row norm is close to a molecule-size proxy (heavy
        #    atom count), so nearly every pair of representations correlates
        #    strongly whether or not they carry the same information, and the
        #    penalty lands almost uniformly. Here each representation is
        #    z-scored per feature, one Euclidean distance is computed for each
        #    of a fixed set of sampled row pairs (the SAME pairs for every
        #    representation), and redundancy is the Spearman correlation
        #    between two representations' distance vectors. That is the
        #    quantity a concatenated distance actually sums over, which is the
        #    thing the penalty is supposed to protect.
        #
        #    The z-scoring is not optional. Without it the features with the
        #    largest raw scale set the distance on their own: measured on an
        #    RDKit descriptor block in this repo, the unstandardised distance
        #    correlates 0.976 with a single-descriptor distance and 0.205 with
        #    the standardised answer.
        #
        # 2. WHAT THE PENALTY MULTIPLIES. Default base is RANK, not score.
        #    `_sanitize_baseline_scores` has already subtracted the minimum
        #    from every score by the time this function runs. Under a geometry
        #    objective the scores are Spearman correlations near zero and
        #    several are negative, so after that shift the worst
        #    representation sits at exactly 0 -- multiplying it by any penalty
        #    changes nothing -- and the surviving spread is a shifted quantity
        #    whose ratios no longer mean "how good this representation is".
        #    Rank survives the shift because the shift is monotone. Rank was
        #    chosen over "penalise the score before the shift" because a
        #    pre-shift score can be negative, and multiplying a negative score
        #    by a penalty in [0, 1] makes it LARGER, which inverts the penalty.
        #    Pass base='score' to get the old score-based behaviour; that arm
        #    exists so an experiment can separate the base change from the
        #    redundancy change.
        #
        # params: diversity_penalty (default 1.0), base ('rank' | 'score',
        # default 'rank'), n_pairs (default 5000), seed (default 42).
        if rep_dict_train is None:
            return dict(baseline_scores)
        diversity_penalty = params.get('diversity_penalty', 1.0)
        base = params.get('base', 'rank')
        n_pairs = int(params.get('n_pairs', 5000))
        seed = int(params.get('seed', 42))
        sorted_reps = sorted(baseline_scores.items(), key=lambda x: x[1],
                             reverse=True)
        n_reps = len(sorted_reps)
        if base == 'rank':
            # Best rep gets n_reps, worst gets 1 -- identical to 'rank_linear'.
            base_val = {name: float(n_reps - i)
                        for i, (name, _s) in enumerate(sorted_reps)}
        elif base == 'score':
            base_val = dict(baseline_scores)
        else:
            raise ValueError(
                f"pairdist_redundancy: base={base!r}. Use 'rank' or 'score'.")

        vectors = _pairdist_vectors(rep_dict_train, [n for n, _ in sorted_reps],
                                    n_pairs=n_pairs, seed=seed)
        if vectors is None:
            # Too few rows to sample pairs from. No redundancy is measurable,
            # so fall through to the unpenalised base quantity rather than
            # inventing a penalty.
            return base_val
        weights, _redundancy = _penalise_against_accepted(
            sorted_reps, base_val, vectors, diversity_penalty, _abs_spearman)
        return weights

    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")


def allocate_equal_split(rep_widths, total_budget):
    """Split `total_budget` features evenly across representations.

    The neutral reference point for the allocation axis: every representation
    gets the same number of features, so any other allocator's result can be
    read as a gain or loss against an even split at the SAME total budget.

    `rep_widths` is {rep_name: n_available_features}. Each representation gets
    total_budget // n_reps, the remainder goes one each to the first names in
    sorted order, and a representation narrower than its share is capped at its
    own width with the shortfall redistributed to representations that still
    have capacity. The returned allocation therefore sums to exactly
    total_budget whenever the representations are wide enough in total.

    Returns: {rep_name: n_features}
    """
    names = sorted(rep_widths)
    n_reps = len(names)
    if n_reps == 0:
        return {}
    base = total_budget // n_reps
    remainder = total_budget - base * n_reps
    allocation = {}
    for i, name in enumerate(names):
        share = base + (1 if i < remainder else 0)
        allocation[name] = min(share, int(rep_widths[name]))

    # Redistribute whatever the width caps took away, narrowest-first so the
    # spare goes to representations that can actually absorb it.
    shortfall = total_budget - sum(allocation.values())
    while shortfall > 0:
        has_room = [n for n in names if allocation[n] < int(rep_widths[n])]
        if not has_room:
            break
        for name in sorted(has_room, key=lambda n: int(rep_widths[n])):
            if shortfall <= 0:
                break
            allocation[name] += 1
            shortfall -= 1

    return allocation


def allocate_performance_weighted(rep_dict_train, rep_dict_val, train_labels, val_labels,
                                  total_budget=100, weighting_scheme='linear',
                                  weighting_params=None, internal_evaluator='rf',
                                  precomputed_scores=None,
                                  model=None, metric_fn=None, lower_is_better=False,
                                  objective=OBJECTIVE_PREDICTIVE, is_clf=False,
                                  groups_val=None, row_index_val=None,
                                  structural_pairs=None, score_sink=None):
    """
    Allocate features proportional to each representation's baseline performance.

    Tests each representation individually on validation set, then allocates budget
    proportional to their R² scores (regression) or accuracy (classification).
    Better performing reps get more features.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        weighting_scheme: How to transform baseline scores into weights.
            'linear' (default, identical to original), 'softmax', 'power_law',
            'thresholded', 'top_k', 'min_floor', 'rank_linear', 'exponential',
            'score_gap', 'entropy', 'pairdist_redundancy'.
        weighting_params: Dict of scheme-specific parameters.
        precomputed_scores: Pre-computed {rep_name: score} dict (optional).
            If provided, skips per-rep model fitting entirely.

    Returns:
        dict: Allocation of {rep_name: n_features}
    """
    # A blend arrives as {component: weight}; a plain name normalises to a
    # single-entry dict so both take one path below.
    _objective_components, _objective_is_blend = normalise_objective(objective)
    if _objective_is_blend and groups_val is None and \
            OBJECTIVE_SIMILARITY in _objective_components:
        raise ValueError(
            "a blend containing %r needs scaffold groups for the held-out rows "
            "(groups_val). Without them read-across scores every "
            "representation the same, which looks like a tie rather than a "
            "failure." % OBJECTIVE_SIMILARITY)
    if precomputed_scores is not None and _objective_is_blend:
        # A blend scores the representations under EVERY component and
        # combines those scores. Precomputed scores are one number per
        # representation under one quantity, so using them would split the
        # budget by that quantity alone while the run reports itself under the
        # blend's name. Dropped rather than used. find_best_hybrid also stops
        # passing them for a blend; this is the second lock, because the
        # failure is silent.
        precomputed_scores = None
    if precomputed_scores is not None:
        missing_reps = [rn for rn in rep_dict_train if rn not in precomputed_scores]
        if missing_reps:
            warn_once(
                f"precomputed_scores is missing {len(missing_reps)} of "
                f"{len(rep_dict_train)} representations {missing_reps[:5]} — they "
                f"are treated as UNSCORED, not as scoring zero. A zero is a real "
                f"score and would have earned them a share of the budget.")
        # NaN, not 0.0, for a representation the caller could not score. 0.0 is a
        # real number: on a correlation it means "no signal", and after the shift
        # below it is simply the smallest entry, which still earns a share. NaN is
        # non-finite, which the sanitiser drops and _require_scored_reps counts.
        baseline_scores = {rn: precomputed_scores.get(rn, float('nan'))
                           for rn in rep_dict_train}
    elif _objective_is_blend:
        # A blended geometry objective scores every representation under each
        # component and combines those scores, so every representation must be
        # scored before any of them can be weighted. Handled here, before the
        # per-representation loop, because _evaluate_allocation only ever sees
        # one representation and cannot compare.
        baseline_scores, _component_scores = blended_rep_scores(
            rep_dict_train, rep_dict_val, val_labels, groups_val,
            _objective_components, is_clf=is_clf,
            row_index_val=row_index_val, structural_pairs=structural_pairs,
            train_labels=train_labels, internal_evaluator=internal_evaluator)
        # BEFORE the sanitiser, which replaces every non-finite score with the
        # same number and so erases the difference between "scored badly" and
        # "could not be scored". blended_rep_scores returns -inf for a
        # representation that failed ANY one component, and its own guard only
        # asks that each component separated SOMETHING -- so with one
        # representation failing one component and another failing a different
        # one, every blended score can be -inf while no component looks dead.
        # The sanitiser then hands back 1.0 for all of them and the budget is
        # split EQUALLY under the requested allocation method's name. Same rule
        # the plain branch below has had since 2026-09-03.
    else:
        baseline_scores = {}

        # Test each representation individually using _evaluate_allocation
        # so that monkey-patching the evaluator affects performance_weighted
        # uniformly.
        for rep_name, X_train in rep_dict_train.items():
            X_val = rep_dict_val[rep_name]

            # Clean
            X_train_clean = np.clip(X_train, -1e10, 1e10)
            X_train_clean = np.nan_to_num(X_train_clean, nan=0.0, posinf=1e10, neginf=-1e10)
            X_val_clean = np.clip(X_val, -1e10, 1e10)
            X_val_clean = np.nan_to_num(X_val_clean, nan=0.0, posinf=1e10, neginf=-1e10)

            score = _score_rep_for_budget(X_train_clean, X_val_clean,
                                          train_labels, val_labels,
                                          internal_evaluator=internal_evaluator,
                                          metric_fn=metric_fn,
                                          lower_is_better=lower_is_better,
                                          objective=objective, is_clf=is_clf,
                                          groups_val=groups_val,
                                          row_index_val=row_index_val,
                                          structural_pairs=structural_pairs)

            baseline_scores[rep_name] = score

        # Hand the RAW scores back to the caller, before the shift below.
        #
        # Scoring these ten representations is one model fit each and is the
        # single most expensive thing in building a hybrid -- ~240s of the 336s
        # a call takes on flab_thermo_tresanco. It does not depend on the
        # allocation method, the augmentation settings or the budget, so a
        # caller sweeping those recomputes an identical number every time.
        #
        # RAW, not shifted: feeding these back as `precomputed_scores` runs them
        # through _sanitize_baseline_scores on that path too, so raw in gives an
        # identical result and shifted in would shift twice.
        #
        # These are scored on the INTERNAL train/validation carve of the fold's
        # training rows, which is the only split that reproduces this path. An
        # attempt on 2026-08-27 to recompute them in an experiment used the
        # outer fold's validation rows instead and silently changed which
        # representations got budget. That is why they are returned from here
        # rather than recomputed anywhere else.
        if score_sink is not None:
            score_sink.update(baseline_scores)

    # OUTSIDE the branches since 2026-09-10, so all three paths — precomputed
    # scores, a blend, and scoring the representations here — are guarded and
    # shifted the same way. The guard used to sit inside the last of the three,
    # so a search, which always passes precomputed scores, could never reach it.
    #
    # Order matters: the guard reads non-finite scores, and the sanitiser
    # replaces them. Guard first.
    _require_scored_reps(baseline_scores, objective,
                         'allocate_performance_weighted')
    # Shift scores to non-negative (weighting schemes require >= 0).
    # Preserves relative ordering.  Negated lower-is-better scores
    # (e.g. -RMSE) are all negative, so a simple max(,0) would clip
    # them all to 0 — shift by min instead.
    # Reps that returned -inf or NaN (a representation that could not be
    # scored) would poison the shift, so they are dropped from the min
    # calculation and given weight 0 afterwards.
    # Which representations could not be scored at all. Read BEFORE the
    # sanitiser, which replaces every non-finite score with 0.0. That 0.0 is
    # weight zero only for the schemes whose weight IS the score: under softmax,
    # rank_linear and rank_exp it is a real entry in the ordering, and measured
    # on two scored representations at 0.70 and 0.68 plus one unscored, the
    # unscored one took 20, 17 and 24 features per 100 respectively. Zeroed
    # explicitly below so every scheme agrees with the docstring.
    # Added 2026-09-10 (metric-direction audit).
    _unscored = {rn for rn, sc in baseline_scores.items() if not np.isfinite(sc)}
    # One normalisation rule per metric, so zero on the score means "contributes
    # nothing" whatever the dataset is scored by. FRAMEWORK_FINALIZATION.md §3.5,
    # built 2026-09-10 — see the block above _sanitize_baseline_scores. Returns
    # the input unchanged for a metric the rule does not cover, and for a
    # geometry or blend objective, whose scores are already correlations.
    if objective == OBJECTIVE_PREDICTIVE:
        baseline_scores = _apply_origin_rule(
            baseline_scores, metric_fn, train_labels, val_labels,
            where='allocate_performance_weighted')
    baseline_scores = _sanitize_baseline_scores(baseline_scores)

    # Apply weighting scheme to transform baseline scores into weights
    weights = _compute_allocation_weights(baseline_scores, weighting_scheme,
                                          weighting_params,
                                          rep_dict_train=rep_dict_train)
    for _rn in _unscored:
        weights[_rn] = 0.0

    # Handle min_floor scheme: guarantee minimum allocation per rep
    if weighting_scheme == 'min_floor':
        floor = (weighting_params or {}).get('floor', 0.05)
        # A representation nobody could score gets no floor either — the floor is
        # a guarantee for representations that competed. Added 2026-09-10.
        _floored = [rn for rn in weights if rn not in _unscored]
        n_reps = len(_floored)
        floor_per_rep = max(1, int(floor * total_budget)) if n_reps else 0
        # floor_per_rep * n_reps could exceed the budget outright (floor=0.05
        # with 25 representations asks for 1.25x it). The allocation was then
        # built at floor_per_rep each and the trim loop below could not take any
        # of it back, because it only ever reduces DOWN TO floor_per_rep — so
        # min_floor spent more than the budget it was given, and its width was
        # not comparable with any other allocation method's. Capped so the
        # floors always fit; the cap can only bite when they would not.
        if n_reps and floor_per_rep * n_reps > total_budget:
            floor_per_rep = max(1, int(total_budget // n_reps))
        floor_total = floor_per_rep * n_reps
        remainder_budget = max(0, total_budget - floor_total)

        total_w = sum(weights.values())
        allocation = {}
        for rep_name, w in weights.items():
            max_features = rep_dict_train[rep_name].shape[1]
            prop = int(round((w / total_w) * remainder_budget)) if total_w > 0 else 0
            _fl = 0 if rep_name in _unscored else floor_per_rep
            allocation[rep_name] = min(_fl + prop, max_features)

        # Adjust to hit budget
        current_total = sum(allocation.values())
        if current_total < total_budget:
            sorted_reps = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            remaining = total_budget - current_total
            for rep_name, _ in sorted_reps:
                if remaining <= 0:
                    break
                can_add = rep_dict_train[rep_name].shape[1] - allocation[rep_name]
                add_amount = min(can_add, remaining)
                allocation[rep_name] += add_amount
                remaining -= add_amount
        elif current_total > total_budget:
            sorted_reps = sorted(weights.items(), key=lambda x: x[1])
            excess = current_total - total_budget
            for rep_name, _ in sorted_reps:
                if excess <= 0:
                    break
                _fl = 0 if rep_name in _unscored else floor_per_rep
                reduce_by = min(allocation[rep_name] - _fl, excess)
                if reduce_by > 0:
                    allocation[rep_name] -= reduce_by
                    excess -= reduce_by
        return allocation

    # Standard path: proportional allocation from weights
    total_w = sum(weights.values())

    if not np.isfinite(total_w) or total_w == 0:
        # All weights zero or non-finite — fall back to equal allocation.
        # Non-finite total_w can happen if a weighting scheme (e.g. power_law)
        # amplifies a pathological score; sanitisation upstream should catch
        # most cases, this is defensive insurance.
        #
        # ALWAYS SAID OUT LOUD. Every representation scoring the same number
        # (the usual cause: a geometry score that could not be formed on this
        # slice, which used to come back as 0.0 for everything) shifts to zero
        # and lands here — so the run divided the budget equally while still
        # reporting itself under the requested allocation method's name. That is
        # indistinguishable in the results from a method that genuinely chose an
        # equal split.
        _msg = (f"allocation {weighting_scheme!r}: every representation "
                f"weighed {'non-finitely' if not np.isfinite(total_w) else 'zero'}"
                f" ({len(weights)} representations, scores "
                f"{sorted(set(round(float(v), 6) for v in baseline_scores.values()))[:5]}"
                f"), so the budget was split EQUALLY. The result is an equal "
                f"split recorded under the name {weighting_scheme!r}.")
        warn_once(_msg)
        return allocate_equal_split(
            {rn: rep_dict_train[rn].shape[1] for rn in weights}, total_budget)

    # ── Proportional shares, rounded by LARGEST REMAINDER ──────────────────
    #
    # REWRITTEN 2026-09-10. What stood here rounded each share independently and
    # then repaired the total with a rule that had nothing to do with the
    # weights: any shortfall went to the single BEST-scoring representation,
    # taking as much as it could hold before the next one saw any; any overshoot
    # came off the WORST-scoring ones with no floor, so a representation the
    # weights had sized at one or two features came back with none. Measured on
    # 400 random draws of ten representations at budget 100: 96 overshot on
    # rounding, and 39 of the 400 zeroed a representation that had a non-zero
    # share. A separate draw with two narrow representations gave the
    # third-best 43 features where its proportional share was 31, and nothing
    # extra to the seven others that had room.
    #
    # Largest remainder is the standard way to round a proportional division and
    # it is what the weighting scheme means: floor every share, then hand the
    # leftover features out one at a time, in order of the fractional part each
    # representation lost to the floor. The total is exact, the order the knob
    # asked for is preserved, and nothing is taken off anybody.
    #
    # THE ONE FLOOR: a representation the weights gave a non-zero share to keeps
    # at least one feature. Zero features means "this representation is not in
    # the hybrid", which is a statement about the representation; arriving there
    # by rounding is an accident of the budget.
    _cap = {rn: int(rep_dict_train[rn].shape[1]) for rn in weights}
    _exact = {rn: (w / total_w) * total_budget for rn, w in weights.items()}
    allocation = {rn: min(int(np.floor(v)), _cap[rn]) for rn, v in _exact.items()}

    # A non-zero weight keeps one feature, taken from the largest share that can
    # spare it. Done before the leftovers so the floor cannot be undone.
    for rn, w in weights.items():
        if w > 0 and allocation[rn] == 0 and _cap[rn] > 0:
            _donor = max((d for d in allocation
                          if allocation[d] > 1 and d != rn),
                         key=lambda d: allocation[d], default=None)
            if _donor is None:
                break
            allocation[_donor] -= 1
            allocation[rn] = 1

    # Hand out what the flooring left over, one feature at a time, largest
    # fractional part first, skipping anything already at its full width.
    _left = total_budget - sum(allocation.values())
    if _left > 0:
        _order = sorted(_exact, key=lambda rn: (_exact[rn] - np.floor(_exact[rn]),
                                                _exact[rn]), reverse=True)
        _i = 0
        while _left > 0 and any(allocation[rn] < _cap[rn] for rn in allocation):
            rn = _order[_i % len(_order)]
            if allocation[rn] < _cap[rn]:
                allocation[rn] += 1
                _left -= 1
            _i += 1
    elif _left < 0:
        # Only reachable through the one-feature floor above, and only by the
        # number of representations it rescued. Taken off the largest shares
        # first, and never below one feature.
        _over = -_left
        for rn in sorted(allocation, key=lambda k: allocation[k], reverse=True):
            if _over <= 0:
                break
            _take = min(allocation[rn] - 1, _over)
            if _take > 0:
                allocation[rn] -= _take
                _over -= _take

    return allocation


def allocate_dominant_greedy(rep_dict_train, rep_dict_val, train_labels, val_labels,
                             total_budget=100, step_size=10, patience=3,
                             dominant_fraction=0.7, internal_evaluator='match',
                             model=None, metric_fn=None, lower_is_better=False,
                             greedy_tolerance=0.0,
                             precomputed_importance=None,
                             objective=OBJECTIVE_PREDICTIVE, is_clf=False):
    """
    Dominant-greedy allocation: give the best rep a large share, greedy-fill the rest.

    1. Evaluate each rep individually (reuses _evaluate_allocation helper).
    2. Give the best rep int(total_budget * dominant_fraction) features.
    3. Greedy-fill remaining budget: iteratively try adding step_size features from
       each other rep, keep whichever improves validation most.
    4. Patience-based early stopping on greedy phase.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        step_size: Features to add per greedy iteration (default: 10)
        patience: Early stopping patience, None to disable (default: 3)
        dominant_fraction: Fraction of budget for best rep (default: 0.7)

    Returns:
        tuple: (best_allocation, history, final_feature_info)
    """
    # --- Phase 1: evaluate each rep individually ---
    rep_scores = {}
    for rep_name in rep_dict_train:
        X_tr = rep_dict_train[rep_name]
        X_va = rep_dict_val[rep_name]
        score = _evaluate_allocation(X_tr, X_va, train_labels, val_labels,
                                     model=model, metric_fn=metric_fn,
                                     lower_is_better=lower_is_better,
                                     internal_evaluator=internal_evaluator,
                                     objective=objective, is_clf=is_clf)
        rep_scores[rep_name] = score

    best_rep = max(rep_scores, key=rep_scores.get)
    dominant_budget = int(total_budget * dominant_fraction)
    max_feats_best = rep_dict_train[best_rep].shape[1]
    dominant_budget = min(dominant_budget, max_feats_best)

    # --- Phase 2: initialise allocation with dominant rep ---
    allocation = {rep: 0 for rep in rep_dict_train}
    allocation[best_rep] = dominant_budget
    remaining_budget = total_budget - dominant_budget

    # Cache RF importance scores — local to this call, discarded on return.
    _importance_cache = {}

    # Build initial hybrid and score
    X_train_init, init_feat_info = _create_hybrid_with_allocation(
        rep_dict_train, train_labels, allocation,
        importance_cache=_importance_cache,
        precomputed_importance=precomputed_importance,
    )
    X_val_init = apply_feature_selection(rep_dict_val, init_feat_info)
    best_val_score = _evaluate_allocation(X_train_init, X_val_init, train_labels, val_labels,
                                         model=model, metric_fn=metric_fn,
                                         lower_is_better=lower_is_better,
                                         internal_evaluator=internal_evaluator,
                                         objective=objective, is_clf=is_clf)

    best_allocation = allocation.copy()
    best_feature_info = init_feat_info
    steps_without_improvement = 0
    history = [{
        'iteration': 0, 'phase': 'dominant', 'rep': best_rep,
        'allocation': allocation.copy(), 'val_score': best_val_score,
    }]

    # --- Phase 3: greedy fill remaining budget ---
    iteration = 0
    while remaining_budget > 0:
        iteration += 1
        iter_best_gain = -np.inf
        iter_best_rep = None
        iter_best_feat_info = None
        iter_best_trial_allocation = None

        for rep_name in rep_dict_train:
            max_features = rep_dict_train[rep_name].shape[1]
            if allocation[rep_name] >= max_features:
                continue

            trial_allocation = allocation.copy()
            features_to_add = min(step_size, max_features - allocation[rep_name],
                                  remaining_budget)
            if features_to_add <= 0:
                continue
            trial_allocation[rep_name] = allocation[rep_name] + features_to_add

            X_train_trial, feat_info = _create_hybrid_with_allocation(
                rep_dict_train, train_labels, trial_allocation,
                importance_cache=_importance_cache,
                precomputed_importance=precomputed_importance,
            )
            X_val_trial = apply_feature_selection(rep_dict_val, feat_info)
            score = _evaluate_allocation(X_train_trial, X_val_trial, train_labels, val_labels,
                                         model=model, metric_fn=metric_fn,
                                         lower_is_better=lower_is_better,
                                         internal_evaluator=internal_evaluator,
                                         objective=objective, is_clf=is_clf)

            if score > iter_best_gain:
                iter_best_gain = score
                iter_best_rep = rep_name
                iter_best_feat_info = feat_info
                iter_best_trial_allocation = trial_allocation

        if iter_best_rep is None:
            break

        features_added = sum(iter_best_trial_allocation.values()) - sum(allocation.values())
        allocation = iter_best_trial_allocation
        remaining_budget -= features_added

        if iter_best_gain > best_val_score:
            best_val_score = iter_best_gain
            best_allocation = allocation.copy()
            best_feature_info = iter_best_feat_info
            steps_without_improvement = 0
        elif greedy_tolerance > 0 and iter_best_gain >= best_val_score - greedy_tolerance:
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        history.append({
            'iteration': iteration, 'phase': 'greedy', 'rep': iter_best_rep,
            'allocation': allocation.copy(), 'val_score': iter_best_gain,
        })

        if patience is not None and steps_without_improvement >= patience:
            break

    return best_allocation, history, best_feature_info


def _concat_rep_dicts(rep_dict):
    """Concatenate rep dicts into flat array with offset tracking.

    Returns:
        (X_concat, rep_names, rep_offsets, rep_sizes)
    """
    rep_names = sorted(rep_dict.keys())
    rep_offsets = {}
    rep_sizes = {}
    offset = 0
    for name in rep_names:
        rep_offsets[name] = offset
        rep_sizes[name] = rep_dict[name].shape[1]
        offset += rep_sizes[name]
    X_concat = np.hstack([rep_dict[name] for name in rep_names])
    return X_concat, rep_names, rep_offsets, rep_sizes


def _global_to_per_rep_feature_info(selected_global, rep_names, rep_offsets,
                                     rep_sizes, importance_scores):
    """Convert global selected indices to per-rep feature_info dict."""
    feature_info = {}
    for name in rep_names:
        offset = rep_offsets[name]
        size = rep_sizes[name]
        mask = (selected_global >= offset) & (selected_global < offset + size)
        local_indices = selected_global[mask] - offset
        if len(local_indices) > 0:
            scores = importance_scores[selected_global[mask]]
            feature_info[name] = {
                'selected_indices': local_indices,
                'importance_scores': scores,
                'n_features': len(local_indices),
            }
    return feature_info


# ============================================================================
# NOT DISPATCHED — but READ THE NUMBERS BEFORE ASSUMING IT WAS CUT ON MERIT.
#
# create_hybrid does not route to allocate_two_stage, and it has been described
# elsewhere in this repo as a cut method alongside allocate_greedy_feature and
# allocate_stability_selection. Its measured record does NOT match that framing,
# so this block records what was actually observed rather than repeating the
# label.
#
# MEASURED (best score per strategy, read out of the result JSONs 2026-08-25):
#   f01 tests/results/framework/f01_allocation_strategies/
#       tcga_luad_vs_lusc  1/21   <- outright winner, acc 0.9920
#       abagym_EGFR        3/21
#       fcrnrt 9/17 · heprt 10/21 · tresanco 14/21 · ginkgo_hic 15/21
#       tcga_brca_pam50 15/17 · herg 19/21
#   slurm-11604966.out (f01 tier-1 screening, logd, 5028 samples):
#       two_stage rmse=0.6326 in 180.7s
#       performance_weighted rmse=0.6360 in 446.3s
#       -> better score AND 2.5x faster than performance_weighted on that run.
#
# So: mid-pack to poor on the molecular and antibody sets, best-in-class on the
# TCGA sets, at roughly performance_weighted cost or below. It is not an
# "absurd computation" method and it is not uniformly subpar.
#
# EXPERIMENT_DESIGN_DISCUSSION.md justifies dropping it with "F04 showed worse
# on 9/10 datasets". f04_two_phase_search.py compares two-phase SEARCH (coarse
# grid then step-size refinement) against a full grid; it is not established
# that F04 ever measured THIS allocator. Do not cite that line as the reason.
#
# Leaving it undispatched is a live decision, not a settled one.
# ============================================================================
def allocate_two_stage(rep_dict_train, rep_dict_val, train_labels, val_labels,
                       total_budget=100, stage1_method='treeshap',
                       stage2_method='permutation', stage1_multiplier=3):
    """
    Two-stage allocation: fast screening then precise selection.
    NOT DISPATCHED — see the block above; its record is mixed, not negative.

    Stage 1: Use fast method to reduce to stage1_multiplier * budget candidates.
    Stage 2: Use slower method on candidates for final selection.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        stage1_method: Fast importance method for screening (default: 'treeshap')
        stage2_method: Precise importance method for final selection (default: 'permutation')
        stage1_multiplier: Candidates = multiplier * budget (default: 3)

    Returns:
        tuple: (allocation, feature_info)
    """
    X_train, rep_names, rep_offsets, rep_sizes = _concat_rep_dicts(rep_dict_train)
    n_features = X_train.shape[1]
    stage1_budget = min(stage1_multiplier * total_budget, n_features)

    # Stage 1: Fast screening
    stage1_scores = compute_feature_importance(
        {'features': X_train}, train_labels, method=stage1_method
    )['features']
    stage1_candidates = np.argsort(stage1_scores)[-stage1_budget:]

    # Stage 2: Precise selection on candidates
    X_train_subset = X_train[:, stage1_candidates]
    try:
        stage2_scores = compute_feature_importance(
            {'features': X_train_subset}, train_labels, method=stage2_method
        )['features']
        selected_local = np.argsort(stage2_scores)[-total_budget:]
    except Exception:
        selected_local = np.argsort(stage1_scores[stage1_candidates])[-total_budget:]

    selected_global = stage1_candidates[selected_local]

    feature_info = _global_to_per_rep_feature_info(
        selected_global, rep_names, rep_offsets, rep_sizes, stage1_scores
    )
    allocation = {name: feature_info[name]['n_features']
                  for name in rep_names if name in feature_info}
    for name in rep_names:
        if name not in allocation:
            allocation[name] = 0

    return allocation, feature_info


def allocate_weighted_global(rep_dict_train, rep_dict_val, train_labels, val_labels,
                             total_budget=100, weighting_scheme='linear',
                             weighting_params=None, pool_total_mult=2,
                             importance_method='lightgbm_gain',
                             internal_evaluator='rf',
                             precomputed_scores=None, precomputed_importance=None,
                             model=None, metric_fn=None, lower_is_better=False,
                             objective=OBJECTIVE_PREDICTIVE, is_clf=False,
                             groups_val=None, row_index_val=None,
                             structural_pairs=None):
    """Weighted global-pooling allocation (engine port of c31).

    Instead of giving each rep a fixed per-rep QUOTA (which starves the best
    rep), this sizes each rep's SLICE of a shared candidate pool by its
    performance weight, pools those candidate features into one matrix, then
    runs ONE global importance ranking over the whole pool to pick the final
    ``total_budget`` features.

    Pipeline (mirrors c31 build_fold_global / topB_from_global):
      1. Per-rep standalone higher-is-better scores (pool-sizing source).
         Uses ``precomputed_scores`` if given, else scores each rep via
         ``_evaluate_allocation`` — identical to allocate_performance_weighted.
      2. ``_sanitize_baseline_scores`` -> ``_compute_allocation_weights`` with the
         SAME (scheme, params, rep_dict_train) call the allocation strategies
         use, so pool-sizing weighting is byte-identical to them.
      3. slots_i = round(w_i / sum(w) * pool_total_mult * total_budget), capped
         at rep width.
      4. Take each rep's top ``slots_i`` candidate columns by ``importance_method``
         (reusing ``precomputed_importance`` when it matches the rep width).
      5. Glue the candidate columns into one matrix (tracking an origins list
         [(rep_name, true_local_col), ...]), run ONE global ``importance_method``
         ranking, keep the top ``total_budget`` pool positions.
      6. Map kept pool positions back to per-rep TRUE local column indices.

    Args mirror allocate_performance_weighted verbatim, plus the two
    globals-only extras ``pool_total_mult`` and ``importance_method``.

    Returns:
        tuple(allocation, feature_info)
          allocation: {rep_name: n_features_selected} (0-filled for reps that
            contribute nothing, like allocate_two_stage).
          feature_info: {rep_name: {'selected_indices': np.ndarray (LOCAL true
            column ids into that rep's own matrix), 'importance_scores':
            np.ndarray (global-rank scores of the kept cols), 'n_features': int}}.
          create_hybrid attaches feature_info['allocation'] itself (mirrors the
          performance_weighted flow), so it is NOT set here.
    """
    rep_names = sorted(rep_dict_train.keys())

    # -- 1. per-rep standalone higher-is-better scores (pool-sizing source) --
    # A blend arrives as {component: weight}; a plain name normalises to a
    # single-entry dict so both take one path below.
    _objective_components, _objective_is_blend = normalise_objective(objective)
    if _objective_is_blend and groups_val is None and \
            OBJECTIVE_SIMILARITY in _objective_components:
        raise ValueError(
            "a blend containing %r needs scaffold groups for the held-out rows "
            "(groups_val). Without them read-across scores every "
            "representation the same, which looks like a tie rather than a "
            "failure." % OBJECTIVE_SIMILARITY)
    if precomputed_scores is not None and _objective_is_blend:
        # A blend scores the representations under EVERY component and
        # combines those scores. Precomputed scores are one number per
        # representation under one quantity, so using them would split the
        # budget by that quantity alone while the run reports itself under the
        # blend's name. Dropped rather than used. find_best_hybrid also stops
        # passing them for a blend; this is the second lock, because the
        # failure is silent.
        precomputed_scores = None
    if precomputed_scores is not None:
        missing_reps = [rn for rn in rep_names if rn not in precomputed_scores]
        if missing_reps:
            warn_once(
                f"precomputed_scores is missing {len(missing_reps)} of "
                f"{len(rep_names)} representations {missing_reps[:5]} — they "
                f"are treated as UNSCORED, not as scoring zero. A zero is a real "
                f"score and would have earned them a share of the budget.")
        # NaN, not 0.0. Every real score arriving here has been sign-flipped to
        # higher-is-better by _score_rep_for_budget, so on an rmse dataset every
        # real score is NEGATIVE and an invented 0.0 is the LARGEST of them --
        # the one representation nobody could score would have taken the biggest
        # slice of the budget. allocate_performance_weighted was corrected to NaN
        # and this twin was not; fixed 2026-09-10 (metric-direction audit).
        baseline_scores = {rn: precomputed_scores.get(rn, float('nan'))
                           for rn in rep_names}
    elif _objective_is_blend:
        # A blended geometry objective scores every representation under each
        # component and combines those scores, so every representation must be
        # scored before any of them can be weighted. Handled here, before the
        # per-representation loop, because _evaluate_allocation only ever sees
        # one representation and cannot compare.
        baseline_scores, _component_scores = blended_rep_scores(
            rep_dict_train, rep_dict_val, val_labels, groups_val,
            _objective_components, is_clf=is_clf,
            row_index_val=row_index_val, structural_pairs=structural_pairs,
            train_labels=train_labels, internal_evaluator=internal_evaluator)
    else:
        baseline_scores = {}
        for rn in rep_names:
            X_tr = rep_dict_train[rn]
            X_vl = rep_dict_val[rn]
            X_tr_c = np.nan_to_num(np.clip(X_tr, -1e10, 1e10),
                                   nan=0.0, posinf=1e10, neginf=-1e10)
            X_vl_c = np.nan_to_num(np.clip(X_vl, -1e10, 1e10),
                                   nan=0.0, posinf=1e10, neginf=-1e10)
            baseline_scores[rn] = _score_rep_for_budget(
                X_tr_c, X_vl_c, train_labels, val_labels,
                internal_evaluator=internal_evaluator,
                metric_fn=metric_fn, lower_is_better=lower_is_better,
                objective=objective, is_clf=is_clf, groups_val=groups_val,
                row_index_val=row_index_val, structural_pairs=structural_pairs)
    # OUTSIDE the else since 2026-09-10, so it covers the blend branch above as
    # well as the per-representation loop. A blend returns -inf for a
    # representation that failed any one component, and the sanitiser on the next
    # line replaces every non-finite score with the same number -- after which an
    # all-failed vector is indistinguishable from a tie and the budget is split
    # equally under the requested method's name.
    _require_scored_reps(baseline_scores, objective,
                         'allocate_weighted_global')
    # Read the unscored set BEFORE the sanitiser turns every non-finite score
    # into 0.0 — see the note in allocate_performance_weighted. Added 2026-09-10.
    _unscored = {rn for rn, sc in baseline_scores.items() if not np.isfinite(sc)}
    # Same per-metric origin rule as allocate_performance_weighted. §3.5.
    if objective == OBJECTIVE_PREDICTIVE:
        baseline_scores = _apply_origin_rule(
            baseline_scores, metric_fn, train_labels, val_labels,
            where='allocate_weighted_global')
    # Shift to non-negative, non-finite -> 0 (identical to the allocators).
    baseline_scores = _sanitize_baseline_scores(baseline_scores)

    # -- 2. pool-sizing weights (byte-identical to allocation strategies) --
    weights = _compute_allocation_weights(baseline_scores, scheme=weighting_scheme,
                                          params=weighting_params,
                                          rep_dict_train=rep_dict_train)
    for _rn in _unscored:
        weights[_rn] = 0.0
    total_w = sum(weights.values())

    # -- 3. per-rep pool slot counts --
    pool_total = int(round(pool_total_mult * total_budget))
    slots = {}
    # When every weight is zero or non-finite there is nothing to be
    # proportional to. The per-representation allocators fall back to an equal
    # split and say so out loud; this path used to give every representation
    # ZERO slots instead, so the candidate pool came out empty, feature_info
    # came back with no entries, and the caller died on
    # "No features to select. feature_info contains no valid representation
    # entries." — a message that names neither the allocation method nor the
    # all-zero scores behind it.
    #
    # Measured 2026-09-12: on fed05's three antibody silos every allocation
    # score came back 0.0, and the ten wglobal_power2 recipes in the search
    # were the only ones to die. The per-representation methods on the same
    # scores warned and split equally.
    #
    # Same fallback here, and the same warning, so the two paths behave alike
    # and an equal split is never silently recorded under a weighted method's
    # name.
    _degenerate_w = not (np.isfinite(total_w) and total_w > 0)
    if _degenerate_w:
        warn_once(
            f"allocation {weighting_scheme!r} (weighted_global): every "
            f"representation weighed "
            f"{'non-finitely' if not np.isfinite(total_w) else 'zero'} "
            f"({len(weights)} representations, scores "
            f"{sorted(set(round(float(v), 6) for v in baseline_scores.values()))[:5]}"
            f"), so the candidate pool was split EQUALLY. The result is an "
            f"equal split recorded under the name {weighting_scheme!r}.")
    _n_rep = max(1, len(rep_names))
    for rn in rep_names:
        width = rep_dict_train[rn].shape[1]
        if _degenerate_w:
            s = int(round(pool_total / _n_rep))
        else:
            s = int(round((weights[rn] / total_w) * pool_total))
        slots[rn] = min(max(s, 0), width)

    # -- 4. per-rep candidate importance (top slots_i by importance_method) --
    def _rep_importance(rn):
        if precomputed_importance is not None and rn in precomputed_importance:
            imp = np.asarray(precomputed_importance[rn])
            if len(imp) == rep_dict_train[rn].shape[1]:
                return imp
        return compute_feature_importance(
            {rn: rep_dict_train[rn]}, train_labels,
            method=importance_method)[rn]

    origins = []   # parallel to the glued pool: (rep_name, true_local_col)
    mats = []
    for rn in rep_names:
        k = slots[rn]
        if k <= 0:
            continue
        imp = _rep_importance(rn)
        k = min(k, len(imp))
        if k <= 0:
            continue
        # Stop at the last feature with a POSITIVE importance score, the same
        # rule concatenate_features applies (see the note there). Importance
        # methods produce large blocks of exact zeros — lightgbm_gain writes 0.0
        # for every feature no tree split on — and once the slot count passes the
        # number that scored above zero, the rest are filled from a block of
        # exact ties that np.argsort breaks by column index. The rule was added
        # at one of the two selection points and not the other.
        # Fixed 2026-09-10 (metric-direction audit).
        _pos = int(np.sum(np.asarray(imp) > 0))
        if _pos > 0:
            k = min(k, _pos)
        top_local = np.argsort(imp)[-k:][::-1]
        X_tr = rep_dict_train[rn][:, top_local]
        X_tr_c = np.nan_to_num(np.clip(X_tr, -1e10, 1e10),
                               nan=0.0, posinf=1e10, neginf=-1e10)
        mats.append(X_tr_c)
        origins.extend([(rn, int(c)) for c in top_local])

    # -- 5. glue + ONE global importance rank + top-B --
    feature_info = {}
    if mats:
        glued = np.hstack(mats)
        gimp = compute_feature_importance(
            {'__pool__': glued}, train_labels,
            method=importance_method)['__pool__']
        B = min(int(total_budget), glued.shape[1])
        # Same positive-importance cut as the per-representation take above and
        # as concatenate_features: the budget is a ceiling, not a quota to fill
        # from a block of exact zeros. Fixed 2026-09-10 (metric-direction audit).
        _pos_g = int(np.sum(np.asarray(gimp) > 0))
        if _pos_g > 0:
            B = min(B, _pos_g)
        if B > 0:
            top_global = np.argsort(gimp)[-B:][::-1]
            per_rep = {}
            for t in top_global:
                rn, c = origins[t]
                d = per_rep.setdefault(rn, {'idx': [], 'scores': []})
                d['idx'].append(c)
                d['scores'].append(float(gimp[t]))
            for rn, d in per_rep.items():
                sel = np.array(d['idx'], dtype=int)
                feature_info[rn] = {
                    'selected_indices': sel,
                    'importance_scores': np.array(d['scores']),
                    'n_features': len(sel),
                }

    # -- 6. allocation dict (0-filled, like allocate_two_stage) --
    allocation = {rn: (feature_info[rn]['n_features'] if rn in feature_info else 0)
                  for rn in rep_names}

    return allocation, feature_info


# ============================================================================
# TESTED AND CUT ON COST — DELIBERATELY NOT DISPATCHED. DO NOT WIRE BACK IN.
#
# This is the function the "absurd computation" objection is actually about.
# create_hybrid does not route to it. Dispatch was removed 2026-03-29..04-03.
#
# MEASURED (best score per strategy, read out of the result JSONs 2026-08-25):
#   f01 tests/results/framework/f01_allocation_strategies/
#       abagym_EGFR 21/21 · heprt 21/21 · fcrnrt 17/17
#       tresanco 20/21 · ginkgo_hic 20/21 · herg 12/21
#       tcga_luad_vs_lusc 6/21
#       tcga_brca_pam50    1/17   <- outright winner, acc 0.8993
#   slurm-11604966.out (f01 tier-1 screening, logd, 5028 samples):
#       stability_selection rmse=0.6256 in 2680.8s   <- best score on that run
#       performance_weighted rmse=0.6360 in  446.3s
#       -> ~6x the wall time to win by 0.010 rmse. f01 medians put it at
#          9-11x performance_weighted on five of six datasets.
#
# Same caveat as the other two: f01 tier-1 is a single fold at one budget and
# one model, so these are screening ranks, not nested_cv_evaluate deltas.
#
# The bootstrap RNG was unseeded and is now seeded (random_state=42, local
# RandomState) — see REPRODUCIBILITY_LEDGER.md item Y2. That ledger still says
# this method is "live if a caller overrides allocation_methods"; that is no
# longer true, overriding raises ValueError from create_hybrid.
#
# The pattern worth noting rather than explaining: it and allocate_two_stage
# both place first on a TCGA set and near-last on the molecular/antibody sets.
# ============================================================================
def allocate_stability_selection(rep_dict_train, rep_dict_val, train_labels, val_labels,
                                  total_budget=100, n_bootstrap=20,
                                  importance_method='treeshap',
                                  selection_threshold=0.6, random_state=42):
    """
    Stability selection via bootstrapping. TESTED AND CUT — see the block above.

    Runs importance on B bootstrap samples, keeps features that appear
    in top-K for >= threshold fraction of samples.

    Args:
        rep_dict_train: Dict of {rep_name: train_features}
        rep_dict_val: Dict of {rep_name: val_features}
        train_labels: Training labels
        val_labels: Validation labels
        total_budget: Maximum total features to select (default: 100)
        n_bootstrap: Number of bootstrap samples (default: 20)
        importance_method: Importance method to use (default: 'treeshap')
        selection_threshold: Minimum selection frequency to keep (default: 0.6)

    Returns:
        tuple: (allocation, feature_info)
    """
    X_train, rep_names, rep_offsets, rep_sizes = _concat_rep_dicts(rep_dict_train)
    n_samples, n_features = X_train.shape
    top_k = min(total_budget * 2, n_features)

    selection_counts = np.zeros(n_features)

    # Seed a local RNG for reproducible, concurrency-safe bootstrap.
    _rng = np.random.RandomState(random_state)
    for b in range(n_bootstrap):
        boot_idx = _rng.choice(n_samples, n_samples, replace=True)
        X_boot = X_train[boot_idx]
        y_boot = train_labels[boot_idx]

        try:
            scores = compute_feature_importance(
                {'features': X_boot}, y_boot, method=importance_method
            )['features']
            top_indices = np.argsort(scores)[-top_k:]
            selection_counts[top_indices] += 1
        except Exception:
            continue

    selection_prob = selection_counts / n_bootstrap
    above_threshold = np.where(selection_prob >= selection_threshold)[0]

    if len(above_threshold) > total_budget:
        prob_ranking = np.argsort(selection_prob[above_threshold])[::-1]
        selected = above_threshold[prob_ranking[:total_budget]]
    elif len(above_threshold) > 0:
        selected = above_threshold
    else:
        selected = np.argsort(selection_prob)[-total_budget:]

    # Use selection probability as importance scores
    feature_info = _global_to_per_rep_feature_info(
        selected, rep_names, rep_offsets, rep_sizes, selection_prob
    )
    allocation = {name: feature_info[name]['n_features']
                  for name in rep_names if name in feature_info}
    for name in rep_names:
        if name not in allocation:
            allocation[name] = 0

    return allocation, feature_info


def _is_classification_task(labels):
    """Determine if this is a classification or regression task.

    Heuristic: classification if labels are a small set of non-negative integers.
    Tighter than just checking dtype to avoid misclassifying count/age data.
    """
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n_unique = len(unique)
    # Quick reject: too many unique values for classification
    if n_unique >= 10:
        return False
    # Integer dtype with few unique values: classification
    if np.issubdtype(labels.dtype, np.integer):
        return True
    # Float labels that are actually integers (e.g., 0.0, 1.0, 2.0)
    if np.issubdtype(labels.dtype, np.floating):
        return np.allclose(labels, labels.astype(int))
    return False


def _create_hybrid_with_allocation(rep_dict, labels, allocation,
                                    importance_cache=None,
                                    precomputed_importance=None):
    """Helper: Create hybrid with a pre-computed allocation dict {rep_name: n_features}.

    Uses precomputed_importance scores if provided (from compute_feature_importance),
    falls back to importance_cache (per-rep cached scores), then to RF as last resort.

    Args:
        importance_cache: Optional dict {rep_name: importance_scores}. Mutable —
            new scores are written back so iterative callers (greedy) pay the
            compute cost only once per rep.
        precomputed_importance: Optional dict {rep_name: importance_scores} from
            the pre-computed importance cache in find_best_hybrid. Takes priority
            over importance_cache and RF fallback.
    """
    selected_features = []
    feature_info = {}

    is_classification = _is_classification_task(labels)

    for name, X in rep_dict.items():
        n_select = allocation[name]

        if n_select == 0:
            continue

        # Clean data
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

        if n_select >= X_clipped.shape[1]:
            selected_features.append(X_clipped)
            feature_info[name] = {
                'selected_indices': np.arange(X_clipped.shape[1]),
                'importance_scores': np.ones(X_clipped.shape[1]),
                'n_features': X_clipped.shape[1]
            }
            continue

        # Priority: precomputed_importance > importance_cache > RF fallback
        importances = None
        if precomputed_importance is not None and name in precomputed_importance:
            importances = precomputed_importance[name]
        elif importance_cache is not None and name in importance_cache:
            importances = importance_cache[name]

        if importances is None:
            raise ValueError(
                f"No importance scores for rep '{name}'. "
                f"precomputed_importance and importance_cache are both missing this rep. "
                f"This likely means the importance pre-computation in find_best_hybrid "
                f"failed or the scores weren't passed through."
            )

        n_select = min(n_select, X_clipped.shape[1])
        top_idx = np.argsort(importances)[-n_select:][::-1]

        X_selected = X_clipped[:, top_idx]
        selected_features.append(X_selected)

        feature_info[name] = {
            'selected_indices': top_idx,
            'importance_scores': importances[top_idx],
            'n_features': n_select
        }

    if len(selected_features) == 0:
        n_samples = len(labels)
        return np.zeros((n_samples, 0), dtype=np.float32), feature_info

    return np.hstack(selected_features).astype(np.float32), feature_info


# Distance-scorer fit cap.
# The internal_evaluator scorers (svm/mlp/gp) rank allocations during the greedy
# loop. The svm scorer fits an O(n^2)-O(n^3) RBF-SVM and the mlp scorer a
# 200-epoch net, so on large data they fit on a seeded row-subsample while still
# predicting on the full X_val. This keeps cache keys neutral (they key on the
# evaluator NAME, not the fit-sample-count) and routes svm/mlp/gp through one
# constant so they behave consistently.
_SCORER_FIT_MAX_SAMPLES = 500


def _subsample_for_scorer(X, y, max_n=_SCORER_FIT_MAX_SAMPLES, seed=42):
    """Seeded row-subsample of (X, y) for fitting an internal_evaluator scorer.
    Returns (X, y) unchanged when n <= max_n."""
    if X.shape[0] > max_n:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_n, replace=False)
        return X[idx], y[idx]
    return X, y


# Metrics that read a RANKING rather than a decision. On a classification
# dataset these have to be scored from GRADED predictions (predict_proba's
# positive column, else decision_function); read off hard 0/1 labels an AUROC
# collapses to (sensitivity + specificity) / 2 and throws away the ranking it
# exists to measure. Every other entry of BUILTIN_METRICS that a classification
# dataset uses -- accuracy, f1_weighted, mcc -- needs the hard classes and
# raises on a graded vector.
RANKING_METRICS = ('auc',)


def _graded_scores_for_clf(fitted_model, X_val_scaled, y_pred_hard):
    """Ranking scores for a fitted classifier, for an AUROC that means something.

    predict_proba's positive column when there is one, else decision_function,
    else the hard labels the caller already has. Binary only — a multiclass
    predict_proba has no single positive column, and the one place this is used
    (the prediction term of a blended budget split) refuses a label with fewer
    than two classes and scores AUROC, which is binary anyway.
    """
    try:
        if hasattr(fitted_model, 'predict_proba'):
            p = np.asarray(fitted_model.predict_proba(X_val_scaled))
            if p.ndim == 2 and p.shape[1] == 2:
                return p[:, 1]
        if hasattr(fitted_model, 'decision_function'):
            d = np.asarray(fitted_model.decision_function(X_val_scaled))
            if d.ndim == 1:
                return d
    except Exception:
        pass
    return y_pred_hard


def _evaluate_allocation(X_train, X_val, y_train, y_val,
                         model=None, metric_fn=None, lower_is_better=False,
                         internal_evaluator='rf',
                         return_predictions=False,
                         objective=OBJECTIVE_PREDICTIVE, is_clf=False,
                         groups_val=None, row_index_val=None,
                         structural_pairs=None,
                         graded_predictions_for_clf=False,
                         graded_out=None):
    """Helper: Evaluate allocation quality on a train/val split.

    If *objective* is 'similarity' the allocation is scored by ACTIVITY
    READ-ACROSS on the held-out (X_val) geometry: each held-out molecule's
    property is predicted from its five nearest neighbours and Spearman'd
    against truth, higher-is-better. No model is fit; *model* /
    *internal_evaluator* / *metric_fn* are ignored. `groups_val` are the
    scaffold groups of the held-out rows and are REQUIRED here — read-across
    without them scores every representation the same, which reads as a tie
    rather than as a failure.

    Until 2026-08-26 this returned SAC under every geometry objective, so a
    read-across run split its budget by a quantity it was not selecting on.
    SAC is cut; the split now uses the objective it was given.

    Otherwise (the 'predictive' default):
    If *model* is provided it is deep-copied, fitted on (X_train, y_train),
    and scored with *metric_fn(y_val, y_pred)*.  The returned value is
    negated when *lower_is_better* so that "higher return = better" always.

    If *model* is None (default), uses the *internal_evaluator* ('lightgbm',
    'rf', or 'xgboost') with R²/accuracy as the score.

    If *return_predictions* is True, returns (score, y_pred) so callers
    can compute additional metrics from the same fitted model.
    """
    if X_train.shape[1] == 0:
        return (-np.inf, None) if return_predictions else -np.inf

    # Defensive clip + nan_to_num (consistent with concatenate_features)
    X_train = np.clip(X_train, -1e10, 1e10)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    X_val = np.clip(X_val, -1e10, 1e10)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e10, neginf=-1e10)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if objective == OBJECTIVE_PREDICTIVE_CORR:
        # Same reason as the blend below: this objective compares the
        # representations against each other and this helper only ever sees
        # one. Refused rather than allowed to fall through to the model path,
        # which would score the dataset's own metric under this name.
        raise ValueError(
            "_evaluate_allocation cannot score %r: it compares "
            "representations against each other and this helper only ever sees "
            "one. Use an allocator that scores every representation first "
            "(performance_weighted or weighted_global)."
            % (OBJECTIVE_PREDICTIVE_CORR,))

    if isinstance(objective, dict) and len(objective) > 1:
        # A blend cannot be scored one allocation at a time -- its components
        # are compared ACROSS representations and this helper only ever sees
        # one.
        # Checked before the geometry branch so a blend containing 'predictive'
        # (not a geometry objective) cannot fall through to the model path and
        # be scored on prediction alone, silently, under the blend's name.
        raise ValueError(
            "_evaluate_allocation cannot score the blend %r: a blend "
            "combines scores measured on every representation and this helper "
            "only ever sees one. Use an allocator that scores every "
            "representation first (performance_weighted or weighted_global)."
            % (objective,))

    if _is_geometry_objective(objective):
        # Geometry objective: score the held-out geometry to rank allocations,
        # no model fit. This is the number that decides how the budget is split
        # across representations, so it has to be the quantity being selected
        # on — see the note in the docstring about what it used to be.
        if groups_val is None:
            raise ValueError(
                "_evaluate_allocation: objective=%r needs groups_val, the "
                "scaffold groups of the held-out rows. Read-across without "
                "them scores every representation identically, which looks "
                "like a tie rather than a failure. create_hybrid slices them "
                "from its own `groups` argument; pass them through."
                % (objective,))
        components, _ = normalise_objective(objective)
        if len(components) == 1:
            score = _score_component(next(iter(components)), X_val_scaled,
                                     y_val, groups_val, is_clf,
                                     row_index_val=row_index_val,
                                     structural_pairs=structural_pairs)
        else:
            # A blend cannot be scored one allocation at a time — it combines
            # scores measured on every representation. blended_rep_scores does
            # that.
            raise ValueError(
                "_evaluate_allocation cannot score the blend %r: a blend ranks "
                "representations against each other and this helper only ever "
                "sees one. Use an allocator that scores every representation "
                "first (performance_weighted or weighted_global)."
                % (objective,))
        if not np.isfinite(score):
            score = -np.inf
        return (score, None) if return_predictions else score

    if model is not None:
        if metric_fn is None:
            # metric_fn(y_val, y_pred) raises TypeError on None, and the except
            # below counts TypeError as an infeasible allocation — so EVERY
            # representation scored -inf, every weight became 0, and the budget
            # split fell through to an equal split filed under whatever
            # allocation name the run was launched with. Silent, and it looked
            # like a result. apply_blueprint and _shared.build_fold_hybrid have
            # refused this pair since 2026-08; this is the same refusal at the
            # place the damage happens.
            raise ValueError(
                "_evaluate_allocation was given a model but no metric_fn. The "
                "model branch scores with metric_fn(y_val, y_pred), so without "
                "one every representation is marked infeasible and the budget "
                "is split equally under the requested allocation's name. Pass "
                "both, or neither (the internal evaluator scores on "
                "r2/accuracy).")
        m = deepcopy(model)
        try:
            m.fit(X_train_scaled, y_train)
            y_pred = m.predict(X_val_scaled)
            score = metric_fn(y_val, y_pred)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            # Numerical failure (e.g. LinAlgError on a pathological
            # feature matrix) or sklearn-internal TypeError from a broken model
            # factory. Treat this allocation as infeasible.
            return (-np.inf, None) if return_predictions else -np.inf
        score = -score if lower_is_better else score
        return (score, y_pred) if return_predictions else score

    # Internal evaluator — used during greedy allocation loops.
    # Each branch creates and fits a model m. The common ending at the
    # bottom computes score and optionally returns predictions.
    is_classification = _is_classification_task(y_train)
    m = None  # set by each branch

    if internal_evaluator == 'rf':
        if is_classification:
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=50, max_depth=10,
                                       random_state=42, n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=50, max_depth=10,
                                      random_state=42, n_jobs=-1)
        try:
            m.fit(X_train_scaled, y_train)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator == 'xgboost':
        try:
            from xgboost import XGBRegressor, XGBClassifier
        except ImportError as _e:
            _refuse_missing_evaluator_library('xgboost', 'xgboost', _e)
        if is_classification:
            m = XGBClassifier(n_estimators=50, max_depth=4,
                              random_state=42, verbosity=0, n_jobs=-1)
        else:
            m = XGBRegressor(n_estimators=50, max_depth=4,
                             random_state=42, verbosity=0, n_jobs=-1)
        try:
            m.fit(X_train_scaled, y_train)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except ImportError as _e:
            _refuse_missing_evaluator_library('lightgbm', 'lightgbm', _e)
        if is_classification:
            m = LGBMClassifier(n_estimators=50, max_depth=4,
                               random_state=42, verbosity=-1, n_jobs=lgbm_n_jobs())
        else:
            m = LGBMRegressor(n_estimators=50, max_depth=4,
                              random_state=42, verbosity=-1, n_jobs=lgbm_n_jobs())
        try:
            m.fit(X_train_scaled, y_train)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator == 'mlp':
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        if is_classification:
            m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                              random_state=42, early_stopping=True)
        else:
            m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                             random_state=42, early_stopping=True)
        # Distance-scorer fit cap: subsample fit, predict full X_val.
        X_fit, y_fit = _subsample_for_scorer(X_train_scaled, y_train)
        try:
            m.fit(X_fit, y_fit)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator == 'svm':
        from sklearn.svm import SVR, SVC
        if is_classification:
            m = SVC(kernel='rbf', C=1.0, random_state=42)
        else:
            m = SVR(kernel='rbf', C=1.0)
        # Distance-scorer fit cap: svm is O(n^2)-O(n^3); subsample the fit,
        # predict full X_val.
        X_fit, y_fit = _subsample_for_scorer(X_train_scaled, y_train)
        try:
            m.fit(X_fit, y_fit)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator == 'gp':
        from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        if is_classification:
            m = GaussianProcessClassifier(kernel=kernel, random_state=42,
                                          max_iter_predict=50, n_restarts_optimizer=0)
        else:
            m = GaussianProcessRegressor(kernel=kernel, random_state=42,
                                         n_restarts_optimizer=0, alpha=1e-2)
        # Distance-scorer fit cap: same helper/constant as svm/mlp.
        X_fit, y_fit = _subsample_for_scorer(X_train_scaled, y_train)
        try:
            m.fit(X_fit, y_fit)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator == 'knn':
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        # This branch builds its OWN estimator and ignores `model`, so a tuned
        # n_neighbors would be discarded here even though it survived
        # everywhere else. Read it off `model` when find_best_hybrid stamped
        # it (see KNN_TUNED_K_ATTR in find_best_hybrid). Unstamped models and
        # non-KNN models keep the historical constant, so nothing moves unless
        # tune_knn_k was asked for.
        if getattr(model, KNN_TUNED_K_ATTR, False) and hasattr(model, 'n_neighbors'):
            k = int(getattr(model, 'n_neighbors', 5) or 5)
        else:
            k = min(5, max(1, int(np.sqrt(X_train_scaled.shape[0]))))
        k = min(k, X_train_scaled.shape[0] - 1)
        if k < 1:
            return (0.0, None) if return_predictions else 0.0
        if is_classification:
            m = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        else:
            m = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        try:
            m.fit(X_train_scaled, y_train)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator == 'catboost':
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
        except ImportError as _e:
            _refuse_missing_evaluator_library('catboost', 'catboost', _e)
        if is_classification:
            m = CatBoostClassifier(iterations=50, depth=4, verbose=0,
                                   random_seed=42)
        else:
            m = CatBoostRegressor(iterations=50, depth=4, verbose=0,
                                  random_seed=42)
        try:
            m.fit(X_train_scaled, y_train)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator in ('dnn_small', 'dnn_wide'):
        # THE SAME NETWORK AT TWO OTHER SIZES. Added 2026-09-18 so the neural
        # capacity question c23 asks can be asked in dnn alone. It used to be
        # asked as mlp against dnn — sklearn's MLPRegressor against this one —
        # which confounded the size with the implementation: a different
        # optimiser, a different early-stopping rule and a different
        # initialisation, all moving at once with the width. mlp was cut from the
        # suite on 2026-09-18, and these two are what replaced it.
        #
        #   dnn_small  (64, 32)             smaller than the shipped evaluator
        #   dnn        (256, 128, 64)       the shipped one, the branch below
        #   dnn_wide   (512, 256, 128)      larger
        #
        # Everything but the width is dnn's own default: dropout 0.2, lr 1e-3,
        # 200 epochs, batch 256, patience 20.
        #
        # ⛔ PURELY ADDITIVE. Neither name is in any default evaluator list, so
        # no search selects one and no cache key on disk changes. An experiment
        # asks for them by name. The widths match f31's dnn arms, so an f31
        # result and a c23 result are talking about the same two networks.
        _sizes = {'dnn_small': (64, 32), 'dnn_wide': (512, 256, 128)}
        _hidden = _sizes[internal_evaluator]
        try:
            from kirby.models.dnn import DNNRegressor, DNNClassifier
        except ImportError as _e:
            _refuse_missing_evaluator_library(internal_evaluator, 'torch', _e)
        try:
            if is_classification:
                m = DNNClassifier(hidden_sizes=_hidden)
            else:
                m = DNNRegressor(hidden_sizes=_hidden)
            m.fit(X_train_scaled, y_train)
        except Exception as _dnn_err:
            # Same rule as the shipped dnn branch below: a failed fit on THESE
            # rows falls back to rf and says so.
            _announce_evaluator_fallback(internal_evaluator, 'rf', _dnn_err)
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='rf',
                                        return_predictions=return_predictions,
                                        graded_predictions_for_clf=graded_predictions_for_clf,
                                        graded_out=graded_out)

    elif internal_evaluator == 'dnn':
        try:
            # The import is separated from the fit on purpose. A missing torch
            # and a fit that blows up on one fold's rows used to share one
            # `except Exception`, so both substituted a random forest and both
            # were recorded as 'dnn'. A missing library is an environment
            # mistake and now stops the run; a failed fit is still a fallback,
            # said out loud, because it is a property of those rows.
            from kirby.models.dnn import DNNRegressor, DNNClassifier
        except ImportError as _e:
            _refuse_missing_evaluator_library('dnn', 'torch', _e)
        try:
            # No overrides: the scorer trains on the SAME settings as the
            # downstream DNN — (256,128,64), dropout 0.2, lr 1e-3, 200 epochs,
            # batch 256, patience 20, Adam + ReduceLROnPlateau, 80/20 internal
            # holdout, targets z-normalised. Owner decision 2026-09-04,
            # replacing an epochs=50/patience=10 cheapening. One definition, in
            # kirby/models/dnn.py — do not re-shorten it here.
            if is_classification:
                m = DNNClassifier()
            else:
                m = DNNRegressor()
            m.fit(X_train_scaled, y_train)
        except Exception as _dnn_err:
            # Falls back to rf, not mlp: mlp left DISTANCE_EVALUATORS on
            # 2026-09-04, so a dnn that fails to fit (no torch on the node, say)
            # would otherwise score its allocations with an evaluator the study
            # was moving away from.
            #
            # SAID OUT LOUD since 2026-09-10. It used to substitute in silence,
            # so a whole run could split every budget with a random forest while
            # every result recorded 'dnn' — f31 found this and left it unfixed.
            # The missing-torch case no longer reaches here: the import above
            # refuses it, because that substitution was total rather than
            # occasional and nothing on disk recorded it. What is left here is a
            # fit that failed on THESE rows, which is a property of the fold.
            _announce_evaluator_fallback('dnn', 'rf', _dnn_err)
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='rf',
                                        return_predictions=return_predictions,
                                        graded_predictions_for_clf=graded_predictions_for_clf,
                                        graded_out=graded_out)

    elif internal_evaluator == 'cnn':
        try:
            from kirby.models.cnn import CNNRegressor, CNNClassifier
        except ImportError as _e:
            _refuse_missing_evaluator_library('cnn', 'torch', _e)
        try:
            if is_classification:
                m = CNNClassifier(epochs=50, patience=10)
            else:
                m = CNNRegressor(epochs=50, patience=10)
            m.fit(X_train_scaled, y_train)
        except Exception as _cnn_err:
            _announce_evaluator_fallback('cnn', 'dnn', _cnn_err)
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='dnn',
                                        return_predictions=return_predictions,
                                        graded_predictions_for_clf=graded_predictions_for_clf,
                                        graded_out=graded_out)

    elif internal_evaluator == 'ngboost':
        try:
            from ngboost import NGBRegressor, NGBClassifier
            from ngboost.distns import Normal, Bernoulli
        except ImportError as _e:
            _refuse_missing_evaluator_library('ngboost', 'ngboost', _e)
        if is_classification:
            m = NGBClassifier(Dist=Bernoulli, n_estimators=50,
                              learning_rate=0.05, random_state=42, verbose=False)
        else:
            m = NGBRegressor(Dist=Normal, n_estimators=50,
                             learning_rate=0.05, random_state=42, verbose=False)
        try:
            m.fit(X_train_scaled, y_train)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    elif internal_evaluator == 'bnn':
        try:
            from kirby.models.bnn import BNNRegressor, BNNClassifier
        except ImportError as _e:
            _refuse_missing_evaluator_library('bnn', 'torch', _e)
        try:
            if is_classification:
                m = BNNClassifier(epochs=50, patience=10, mc_samples=5)
            else:
                m = BNNRegressor(epochs=50, patience=10, mc_samples=5)
            m.fit(X_train_scaled, y_train)
        except Exception as _bnn_err:
            _announce_evaluator_fallback('bnn', 'dnn', _bnn_err)
            return _evaluate_allocation(X_train, X_val, y_train, y_val,
                                        internal_evaluator='dnn',
                                        return_predictions=return_predictions,
                                        graded_predictions_for_clf=graded_predictions_for_clf,
                                        graded_out=graded_out)

    elif callable(internal_evaluator):
        m = internal_evaluator()
        try:
            m.fit(X_train_scaled, y_train)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
            return (-np.inf, None) if return_predictions else -np.inf

    else:
        raise ValueError(f"Unknown internal_evaluator: {internal_evaluator}. "
                         f"Use 'rf', 'xgboost', 'lightgbm', 'svm', 'gp', "
                         f"'knn', 'catboost', 'dnn', 'dnn_small', 'dnn_wide', "
                         f"'cnn', 'ngboost', 'bnn', 'mlp' (cut from the suite "
                         f"2026-09-18, still built when named), or a callable "
                         f"that returns an sklearn-compatible model.")

    # Common ending: compute predictions and score from the fitted model
    try:
        y_pred = m.predict(X_val_scaled)
        if is_classification:
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_val, y_pred)
        else:
            score = r2_score(y_val, y_pred)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
        return (-np.inf, None) if return_predictions else -np.inf
    # The SCORE always stays on hard class labels — accuracy needs them, and so
    # do the accuracy/f1/mcc metric functions find_best_hybrid runs over this
    # same y_pred. Only a caller that asks for graded predictions gets them, and
    # only for classification: an AUROC read off thresholded labels is
    # (sensitivity + specificity) / 2, which throws away the ranking it is meant
    # to measure. Added 2026-09-10 (metric-direction audit).
    # A caller that scores SEVERAL metrics off this one fitted model needs BOTH
    # vectors, not one: accuracy, f1_weighted and mcc need the hard classes and
    # raise on a graded vector, while an AUROC read off the hard classes is
    # (sensitivity + specificity) / 2. Pass a dict as `graded_out` and the
    # graded vector is left in it under 'y'; the return is unchanged, so every
    # existing caller is untouched. Added 2026-09-13 -- before it, the
    # per-representation scores that SPLIT THE BUDGET on an auc dataset were
    # read off hard classes and stored under the name 'auc'.
    if return_predictions and graded_out is not None and is_classification:
        graded_out['y'] = _graded_scores_for_clf(m, X_val_scaled, y_pred)
    if return_predictions and graded_predictions_for_clf and is_classification:
        y_out = _graded_scores_for_clf(m, X_val_scaled, y_pred)
        return (score, y_out)
    return (score, y_pred) if return_predictions else score


# ============================================================================
# AUGMENTATION HANDLING
# ============================================================================

def _score_rep_for_budget(X_train, X_val, y_train, y_val,
                          internal_evaluator='rf',
                          metric_fn=None, lower_is_better=False,
                          objective=OBJECTIVE_PREDICTIVE, is_clf=False,
                          groups_val=None, row_index_val=None,
                          structural_pairs=None):
    """One representation's score for sizing its share of the budget.

    Fitted with `internal_evaluator`, NOT with the downstream model, and read
    off with `metric_fn` when there is one.

    Why the downstream model is not used here. Inside a search these numbers
    come from find_best_hybrid's precompute, which fits the internal evaluator
    and reads the dataset's metric off its predictions -- one score per (fold,
    evaluator, metric), shared by every model in the sweep. A create_hybrid call
    made outside a search took the other branch of _evaluate_allocation and
    fitted the downstream model instead, so the same recipe on the same rows
    split its budget two different ways depending on who asked. One device now,
    and it is the evaluator, because that is the axis the search varies.

    The downstream model still scores greedy allocation and every augmentation
    keep-or-drop decision; only the budget split moved.
    """
    if _is_geometry_objective(objective) or isinstance(objective, dict):
        # No model is fitted on this path at all -- the score is read off the
        # held-out geometry -- so there is nothing to choose between.
        return _evaluate_allocation(
            X_train, X_val, y_train, y_val,
            internal_evaluator=internal_evaluator,
            objective=objective, is_clf=is_clf, groups_val=groups_val,
            row_index_val=row_index_val, structural_pairs=structural_pairs)
    _graded = {}
    score, y_pred = _evaluate_allocation(
        X_train, X_val, y_train, y_val,
        internal_evaluator=internal_evaluator,
        objective=objective, is_clf=is_clf,
        return_predictions=True, graded_out=_graded)
    if metric_fn is None or y_pred is None:
        return score
    # auc is scored from the graded vector, everything else from the hard
    # classes. Before 2026-09-13 this whole function scored auc off the hard
    # classes, so on a classification dataset whose metric is auc the number
    # that sized each representation's share of the budget was
    # (sensitivity + specificity) / 2.
    if (metric_name_for_fn(metric_fn) in RANKING_METRICS
            and _graded.get('y') is not None):
        y_pred = _graded['y']
    try:
        raw = float(metric_fn(np.asarray(y_val), np.asarray(y_pred)))
    except Exception:
        # No metric on these predictions: keep the evaluator's own r2/accuracy
        # rather than marking the representation infeasible.
        return score
    if not np.isfinite(raw):
        return -np.inf
    return -raw if lower_is_better else raw


def _select_augmentation_features(X_aug, labels, budget, is_classification):
    """
    Select top features from an augmentation array based on RF importance.
    
    Args:
        X_aug: Augmentation features (n_samples, n_features)
        labels: Target labels
        budget: Max features to select
        is_classification: Whether this is a classification task
        
    Returns:
        tuple: (selected_features, selected_indices, importance_scores)
    """
    # Clean data
    X_clean = np.clip(X_aug, -1e10, 1e10)
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1e10, neginf=-1e10)
    
    n_features = X_clean.shape[1]
    n_select = min(budget, n_features)
    
    if n_select >= n_features:
        # Use all features
        return X_clean, np.arange(n_features), np.ones(n_features)
    
    # Compute importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    if is_classification:
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    model.fit(X_scaled, labels)
    importances = np.nan_to_num(model.feature_importances_, nan=0.0)
    
    # Select top features
    top_idx = np.argsort(importances)[-n_select:][::-1]
    
    return X_clean[:, top_idx], top_idx, importances[top_idx]


def _evaluate_augmentation_addition(
    X_base_train, X_base_val,
    X_aug_train, X_aug_val,
    y_train, y_val,
    baseline_score,
    model=None, metric_fn=None, lower_is_better=False,
    objective=OBJECTIVE_PREDICTIVE, is_clf=False,
    groups_val=None,
):
    """
    Evaluate whether adding an augmentation improves performance.

    Args:
        X_base_train, X_base_val: Base hybrid features
        X_aug_train, X_aug_val: Augmentation features to test
        y_train, y_val: Labels
        baseline_score: Current validation score without augmentation
        objective/is_clf: forwarded to _evaluate_allocation. Under a geometry
            objective the accept/reject test scores held-out GEOMETRY, not a
            fitted model — without this the augmentation path silently reverted
            to the predictive test while the rest of the search was geometric.

    Returns:
        tuple: (new_score, improvement)
    """
    # Combine base + augmentation
    X_train_combined = np.hstack([X_base_train, X_aug_train])
    X_val_combined = np.hstack([X_base_val, X_aug_val])

    new_score = _evaluate_allocation(X_train_combined, X_val_combined, y_train, y_val,
                                     model=model, metric_fn=metric_fn,
                                     lower_is_better=lower_is_better,
                                     objective=objective, is_clf=is_clf,
                                     groups_val=groups_val)
    improvement = new_score - baseline_score

    return new_score, improvement


# ⛔ 'greedy_ablation' IS CUT. CLOSED 2026-08-26 by the owner, in the same
# breath as the rest of the greedy family: "No greedy at all."
#
# It is greedy accept/reject as a feature-admission rule, which the standing
# greedy decision already covers. The one-feature-at-a-time sibling of this was
# deleted on 2026-08-24 and that deletion was correct; this is the whole-block
# version and it walks the donors in dictionary order, keeping each that
# improves against a running best. Order-dependent, base-blind, and it
# ratchets.
#
# Asked on 2026-08-26 whether a very small budget would rescue it. It would
# not, and the reason is in this function rather than in any argument about
# greedy: the cost is one forest fit per donor wider than augmentation_budget
# plus one scoring fit per donor plus one baseline. That count is set by the
# NUMBER OF DONORS and does not fall when the budget falls. A smaller budget
# makes each fit cheaper, not fewer of them -- and running this method at a
# budget the other four do not use would make it incomparable with them, which
# is the thing the experiment exists to do.
#
# The function stays so old configs and saved recipes do not crash, and
# create_hybrid still dispatches the name. Nothing selects it: it is out of
# f24's method list. Do not put it back.
def greedy_augmentation_ablation(
    X_base_train, X_base_val,
    augmentations_train, augmentations_val,
    y_train, y_val,
    augmentation_budget=20,
    threshold=None,
    return_all_scores=False,
    model=None, metric_fn=None, lower_is_better=False,
    objective=OBJECTIVE_PREDICTIVE, is_clf=False,
    groups_val=None,
):
    """
    Greedily select augmentations that improve validation performance.
    
    For each augmentation:
    1. Select top `augmentation_budget` features by importance
    2. Test if adding improves validation score
    3. Keep if improvement exceeds threshold (or all if threshold=None)
    
    Args:
        X_base_train, X_base_val: Base hybrid features
        augmentations_train: Dict of {aug_name: train_features}
        augmentations_val: Dict of {aug_name: val_features}
        y_train, y_val: Labels
        augmentation_budget: Max features per augmentation (default: 20)
        threshold: Minimum improvement required to keep (default: None = keep all improving)
        return_all_scores: Return scores for all augmentations, not just kept ones
        
    Returns:
        dict: {
            'kept_augmentations': List of augmentation names that were kept,
            'augmentation_info': Dict of {aug_name: {'indices': ..., 'improvement': ...}},
            'all_scores': Dict of {aug_name: improvement} (if return_all_scores=True),
            'final_score': Final validation score with all kept augmentations
        }
    """
    is_classification = _is_classification_task(y_train)
    
    # Get baseline score (base hybrid only)
    baseline_score = _evaluate_allocation(X_base_train, X_base_val, y_train, y_val,
                                          model=model, metric_fn=metric_fn,
                                          lower_is_better=lower_is_better,
                                          objective=objective, is_clf=is_clf,
                                          groups_val=groups_val)

    kept_augmentations = []
    augmentation_info = {}
    all_scores = {}
    
    # Current best features (accumulates as we add augmentations)
    X_current_train = X_base_train.copy()
    X_current_val = X_base_val.copy()
    current_score = baseline_score
    
    # Test each augmentation
    for aug_name, X_aug_train in augmentations_train.items():
        X_aug_val = augmentations_val[aug_name]
        
        # Select top features from this augmentation
        X_aug_train_selected, indices, importances = _select_augmentation_features(
            X_aug_train, y_train, augmentation_budget, is_classification
        )
        X_aug_val_selected = X_aug_val[:, indices]
        X_aug_val_selected = np.clip(X_aug_val_selected, -1e10, 1e10)
        X_aug_val_selected = np.nan_to_num(X_aug_val_selected, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Test improvement
        new_score, improvement = _evaluate_augmentation_addition(
            X_current_train, X_current_val,
            X_aug_train_selected, X_aug_val_selected,
            y_train, y_val,
            current_score,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            objective=objective, is_clf=is_clf, groups_val=groups_val,
        )
        
        all_scores[aug_name] = {
            'improvement': improvement,
            'new_score': new_score,
            'n_features': len(indices)
        }
        
        # Decide whether to keep
        keep = False
        if threshold is None:
            # Keep if any improvement
            keep = improvement > 0
        else:
            # Keep if improvement exceeds threshold
            keep = improvement >= threshold
        
        if keep:
            kept_augmentations.append(aug_name)
            augmentation_info[aug_name] = {
                'selected_indices': indices,
                'importance_scores': importances,
                'n_features': len(indices),
                'improvement': improvement
            }
            # Update current features
            X_current_train = np.hstack([X_current_train, X_aug_train_selected])
            X_current_val = np.hstack([X_current_val, X_aug_val_selected])
            current_score = new_score
    
    result = {
        'kept_augmentations': kept_augmentations,
        'augmentation_info': augmentation_info,
        'baseline_score': baseline_score,
        'final_score': current_score,
        'total_improvement': current_score - baseline_score
    }
    
    if return_all_scores:
        result['all_scores'] = all_scores
    
    return result


# ============================================================================
# AUGMENTATION — capped (displaces base features) and uncapped (appends).
#
# Provenance note, 2026-08-25: these were once labelled "from f08 experiments".
# That sweep never ran this code — it carries its own private copy — it had no
# candidates at all on one of its six datasets, and it read its before/after
# score off the same slice its selection optimised. It is not validation for
# either path. Both were rebuilt on measured behaviour instead; see the
# docstrings of augment_capped / augment_uncapped.
# ============================================================================

CANDIDATE_RANKINGS = (
    # base-blind: ranked against the labels, the base representation unseen
    'rf_importance',        # random forest split importance
    'lightgbm_gain',        # LightGBM gain, the scorer the hybrid budget uses
    'mutual_info',          # mutual information, catches non-monotone signal
    'deeplift',             # DeepLIFT attribution through a small MLP
    # base-aware: ranked against what the base representation cannot explain
    'residual_correlation',  # |correlation| with out-of-fold base residuals
    'residual_deeplift',     # DeepLIFT attribution with those residuals as target
)


def _oof_base_residuals(X_base_train, y_train, is_classification, groups_train):
    """Out-of-fold residuals of a cheap forest on the base representation.

    What the base representation CANNOT explain. In-sample residuals of a strong
    model shrink toward zero and are dominated by fit noise, which made this
    useless, so they are always out-of-fold where the row count allows it.

    Three degradations, in order, each inside its own guard because this runs
    inside a search loop where raising aborts the whole create_hybrid call:
    out-of-fold, then in-sample, then centred labels (which is the base-blind
    behaviour and a worse ranking, but a usable one).
    """
    from sklearn.model_selection import (
        KFold, StratifiedKFold, cross_val_predict,
    )
    scaler_b = StandardScaler()
    X_base_s = scaler_b.fit_transform(X_base_train)
    if is_classification:
        est = RandomForestClassifier(n_estimators=50, max_depth=10,
                                     random_state=42, n_jobs=-1)
    else:
        est = RandomForestRegressor(n_estimators=50, max_depth=10,
                                    random_state=42, n_jobs=-1)
    residuals = None
    n_tr = X_base_s.shape[0]
    if n_tr >= 10:
        # 3 folds, not 5, to keep the cost of this ranking device near the
        # single fit it replaced. shuffle=True rather than cv=<int>: an integer
        # cv is KFold(shuffle=False), which validates on contiguous blocks in
        # dataset order, so any row ordering correlated with y (a source file
        # sorted by activity — and GroupKFold hands back ascending indices,
        # which preserves that ordering) biases the out-of-fold predictions
        # until the residuals are close to y itself, i.e. back to the
        # base-blind ranking this exists to replace.
        n_splits = int(min(3, max(2, n_tr // 5)))
        try:
            # Group-aware when the caller passed groups: a molecule's
            # scaffold-mates must not sit in the rows its own residual is
            # predicted from, or the residual measures fit rather than error.
            _g = (np.asarray(groups_train).ravel()
                  if groups_train is not None else None)
            if _g is not None and len(_g) == n_tr and \
                    len(np.unique(_g)) >= n_splits:
                from sklearn.model_selection import (
                    GroupKFold, StratifiedGroupKFold,
                )
                cv = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                           random_state=42)
                      if is_classification
                      else GroupKFold(n_splits=n_splits))
                _cv_groups = _g
            else:
                cv = (StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=42) if is_classification
                      else KFold(n_splits=n_splits, shuffle=True,
                                 random_state=42))
                _cv_groups = None
            oof = None
            if is_classification:
                p = np.asarray(cross_val_predict(
                    est, X_base_s, y_train, groups=_cv_groups,
                    cv=cv, method='predict_proba'))
                if p.ndim == 2 and p.shape[1] == 2:
                    oof = p[:, 1]
            if oof is None:
                oof = cross_val_predict(est, X_base_s, y_train,
                                        groups=_cv_groups, cv=cv)
            residuals = (np.asarray(y_train, dtype=float)
                         - np.asarray(oof, dtype=float))
            _state = 'out_of_fold'
        except Exception as _e_oof:
            residuals, _state = None, None
            _why = f'{type(_e_oof).__name__}: {_e_oof}'
    else:
        _why = f'only {n_tr} training rows, fewer than the 10 needed'
    if residuals is None:
        try:
            est.fit(X_base_s, y_train)
            residuals = (np.asarray(y_train, dtype=float)
                         - np.asarray(est.predict(X_base_s), dtype=float))
            _state = 'in_sample'
            warn_once(
                f"residual ranking fell back to IN-SAMPLE residuals ({_why}). "
                f"A model's error on the rows it was fitted on understates it, "
                f"so the candidate features are ranked against a residual that "
                f"is smaller than the real one.")
        except Exception as _e_fit:
            residuals, _state = None, None
            _why = f'{_why}; refit also failed: {type(_e_fit).__name__}'
    if residuals is None:
        try:
            _yv = np.asarray(y_train, dtype=float)
            residuals = _yv - _yv.mean()
            _state = 'centred_labels'
            warn_once(
                f"residual ranking fell back to CENTRED LABELS ({_why}). That "
                f"is not a residual at all — the ranking is base-BLIND on this "
                f"fold, which is the thing residual_correlation exists to "
                f"avoid, and it is indistinguishable in the result from a fold "
                f"where it worked.")
        except Exception:
            residuals = np.zeros(n_tr, dtype=float)
            _state = 'zeros'
            warn_once("residual ranking produced ZEROS — every candidate "
                      "feature ties and the ranking is arbitrary.")
    # WHICH of the four it used. Returned so the caller can record it: the three
    # fallbacks used to be silent, so a fold that degraded to centred labels was
    # indistinguishable in the result from one that computed real out-of-fold
    # residuals, and a comparison of candidate rankings could not say whether
    # residual_correlation had ever run as itself. Recorded 2026-09-12.
    _last_residual_state.append(_state)
    return residuals


# Which of the four residual states the last _oof_base_residuals call used.
# A list, not a scalar, so a caller can read the whole fold's history; the
# caller is expected to clear it. Values: 'out_of_fold' (what the ranking is
# meant to use), 'in_sample', 'centred_labels', 'zeros'.
_last_residual_state: list = []


def _deeplift_column_scores(X_scaled, target, is_classification,
                            hidden_dims=(64, 32), max_epochs=200, lr=0.001,
                            weight_decay=1e-4, patience=10, val_frac=0.2,
                            batch_size=128, random_state=42):
    """Mean absolute DeepLIFT attribution per column of X_scaled.

    The only ranking device here that is not a tree. rf_importance,
    lightgbm_gain and a TreeSHAP variant all read where a tree chose to split,
    so they cannot disagree with each other about a feature no tree splits on.
    This trains a small network and reads its gradients instead.

    THE NETWORK IS EARLY-STOPPED, AND THAT IS NOT A DETAIL. Measured on herg_ki
    2026-09-07, curated donors, cap 64, rf, 3 folds: trained the original way --
    100 epochs of full-batch Adam, no held-out rows, no weight decay -- it kept
    14.0 features per fold and 9.9 of them came from the planted pure-noise
    donor. Every other ranking rule on the identical candidate pool took under
    1.1. With ~940 training rows against ~340 candidate columns the network
    memorises, every input ends up with some weight, and the attribution that
    comes back is close to arbitrary. Rows are now held out, mini-batched, and
    training stops when the held-out loss stops improving.

    Returns None rather than raising if torch or captum is missing, so a caller
    without them degrades instead of aborting the search.

    LOCAL HAZARD: torch and LightGBM each load an OpenMP runtime and the second
    one to load segfaults the process. Import torch FIRST. Measured: lightgbm
    then torch exits 139 with no traceback; torch then lightgbm is fine, and
    KIRBY_LGBM_THREADS=1 alone does not prevent it.
    """
    try:
        import torch
        import torch.nn as nn
        from captum.attr import DeepLift
    except ImportError:
        return None

    torch.manual_seed(random_state)
    X_t = torch.tensor(np.asarray(X_scaled, dtype=np.float32))
    y_arr = np.asarray(target)
    n_rows = X_t.shape[0]
    n_out = 1
    if is_classification:
        classes = np.unique(y_arr)
        n_out = max(2, len(classes))
        y_t = torch.tensor(np.searchsorted(classes, y_arr).astype(np.int64))
        loss_fn = nn.CrossEntropyLoss()
    else:
        y_t = torch.tensor(y_arr.astype(np.float32)).view(-1, 1)
        loss_fn = nn.MSELoss()

    layers, prev = [], X_t.shape[1]
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    net = nn.Sequential(*layers)

    # Held-out rows for the stopping rule. Below 25 rows there is nothing
    # meaningful to hold out, so it falls back to a fixed short schedule --
    # short, not the original 100, because that is the schedule that memorised.
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_rows)
    n_val = int(round(val_frac * n_rows))
    use_val = n_rows >= 25 and n_val >= 5
    if use_val:
        va_idx = torch.tensor(perm[:n_val].copy())
        tr_idx = torch.tensor(perm[n_val:].copy())
    else:
        tr_idx = torch.arange(n_rows)
        va_idx = None
        max_epochs = min(max_epochs, 30)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, waited = float('inf'), None, 0
    n_tr = int(tr_idx.shape[0])
    bs = max(8, min(batch_size, n_tr))
    for _ in range(max_epochs):
        net.train()
        order = torch.randperm(n_tr)
        for start in range(0, n_tr, bs):
            b = tr_idx[order[start:start + bs]]
            opt.zero_grad()
            loss_fn(net(X_t[b]), y_t[b]).backward()
            opt.step()
        if va_idx is None:
            continue
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(X_t[va_idx]), y_t[va_idx]))
        if vl < best_loss - 1e-6:
            best_loss, waited = vl, 0
            best_state = {k: v.detach().clone()
                          for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    # ATTRIBUTE IN ROW CHUNKS. What comes back is the mean of |attribution| over
    # ROWS, and DeepLIFT attributes each row against its own baseline through a
    # Linear/ReLU stack with no cross-row layer, so a chunked mean is the same
    # number as a whole-matrix one -- only the peak memory differs.
    #
    # Why it has to: captum hooks every layer and holds the input, the output
    # and the multipliers for the WHOLE batch, and it doubles the batch because
    # the baseline goes through with the input. On grover (4800 features) over
    # a 1380-row training fold that is several gigabytes, and a fast-tier search
    # calls this 10 representations x 5 folds. Measured 2026-09-10: the run was
    # killed for memory inside this call on a 16 GB machine.
    _dl = DeepLift(net)
    _tgt = 0 if is_classification else None
    _n = int(X_t.shape[0])
    _width = int(X_t.shape[1])
    # Cap the doubled batch at roughly 4M floats of input, and always take at
    # least one row.
    _chunk = max(1, min(_n, int(4_000_000 // max(1, _width))))
    _acc = np.zeros(_width, dtype=np.float64)
    try:
        for _start in range(0, _n, _chunk):
            _blk = X_t[_start:_start + _chunk]
            _a = _dl.attribute(_blk, baselines=torch.zeros_like(_blk),
                               target=_tgt)
            _acc += np.abs(_a.detach().numpy()).sum(axis=0)
            del _a, _blk
    except Exception:
        return None
    return np.nan_to_num(_acc / max(1, _n), nan=0.0)


def _pool_and_rank_aug_features(augmentations_train, augmentations_val,
                                y_train, X_base_train, model,
                                candidate_ranking='residual_correlation',
                                is_classification=False,
                                max_candidates=200,
                                per_source_budget=None,
                                groups_train=None):
    """Pool all augmentation features, rank them, return ranked arrays + provenance.

    Parameters
    ----------
    candidate_ranking : str
        One of CANDIDATE_RANKINGS, in two families.

        BASE-AWARE — ranked against what the base representation cannot
        explain, using its out-of-fold residuals:
          'residual_correlation' (default)  |correlation| with the residuals
          'residual_deeplift'               DeepLIFT attribution, residuals as
                                            the training target

        BASE-BLIND — ranked against the labels, the base unseen. These promote
        whatever correlates with the training labels, including columns the base
        already covers:
          'rf_importance'    random forest split importance
          'lightgbm_gain'    LightGBM gain, the scorer the hybrid budget uses
          'mutual_info'      mutual information, catches non-monotone signal
          'deeplift'         DeepLIFT attribution through a small MLP

        The first three base-blind rules all read where a tree chose to split,
        so they cannot disagree about a feature no tree splits on. 'mutual_info'
        and the two deeplift rules are the ones that can.
    model : estimator or None
        NOT used by either ranker — both use the same cheap internal forest, so
        that this ranking device does not drag the downstream model (a GP, an
        MLP) into the search loop. Kept in the signature because callers pass it
        positionally.
    per_source_budget : int or dict, optional
        When set, each augmentation source is first reduced to its top-N columns
        by RF importance BEFORE the sources are pooled together. Without this,
        pooling a 2560-column embedding next to a 40-column engineered feature
        set lets the wide source dominate the single global ranking purely by
        column count, and the ranking itself is fit on one fold's training rows.
        Capping per source keeps each source's contribution comparable and the
        candidate set small. None = pool every column (original behaviour).

        Pass a dict {aug_name: N} to cap only some sources — a source absent
        from the dict (or mapped to a falsy value) is left uncapped. That is how
        a caller caps wide embeddings down to their best few columns while
        letting a small engineered feature set contribute all of its columns.
    groups_train : array-like, optional
        Scaffold (or sequence-identity / complex / site) group of each training
        row. When given, the out-of-fold residuals under 'residual_correlation'
        are computed with GroupKFold instead of a shuffled KFold, so a
        molecule's group-mates cannot sit in the rows its own residual is
        predicted from. Without it those residuals shrink toward the fit and
        the ranking is weaker than it reads. None keeps the shuffled split.

    Returns
    -------
    ranked_train, ranked_val : ndarray
        Columns reordered by rank (best first), trimmed to max_candidates.
    provenance : list of (aug_name, col_idx)
        Original identity of each ranked column. Column indices always refer to
        the ORIGINAL source array, so apply_augmentation_selection stays valid
        even when per_source_budget pre-trimmed the source.
    """
    prov = []
    parts_tr, parts_vl = [], []

    def _budget_for(name):
        if not per_source_budget:
            return None
        if isinstance(per_source_budget, dict):
            return per_source_budget.get(name)
        return per_source_budget

    for aug_name, X_tr in augmentations_train.items():
        X_vl = augmentations_val[aug_name]
        X_tr = np.nan_to_num(np.clip(X_tr, -1e10, 1e10), nan=0.0)
        X_vl = np.nan_to_num(np.clip(X_vl, -1e10, 1e10), nan=0.0)
        cols = np.arange(X_tr.shape[1], dtype=int)
        _bud = _budget_for(aug_name)
        if _bud and X_tr.shape[1] > _bud:
            # Fit on y_train only — this runs inside a fold, so no leakage.
            _, sel_idx, _ = _select_augmentation_features(
                X_tr, y_train, _bud, is_classification)
            cols = np.asarray(sel_idx, dtype=int)
            X_tr = X_tr[:, cols]
            X_vl = X_vl[:, cols]
        for col in cols:
            prov.append((aug_name, int(col)))
        parts_tr.append(X_tr)
        parts_vl.append(X_vl)

    if not parts_tr:
        return None, None, []

    X_pool_tr = np.hstack(parts_tr)
    X_pool_vl = np.hstack(parts_vl)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pool_tr)

    if candidate_ranking == 'rf_importance':
        if is_classification:
            rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                        random_state=42, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=50, max_depth=10,
                                       random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y_train)
        scores = np.nan_to_num(rf.feature_importances_, nan=0.0)
    elif candidate_ranking == 'lightgbm_gain':
        # The same scorer the hybrid uses to fill its budget, applied to the
        # candidates. Base-blind, like rf_importance, and a second tree opinion.
        import lightgbm as lgb
        n_jobs = int(os.environ.get('KIRBY_LGBM_THREADS', '-1'))
        Est = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
        gbm = Est(n_estimators=100, importance_type='gain', random_state=42,
                  n_jobs=n_jobs, verbose=-1)
        gbm.fit(X_scaled, y_train)
        scores = np.nan_to_num(
            np.asarray(gbm.feature_importances_, dtype=float), nan=0.0)
    elif candidate_ranking == 'mutual_info':
        # Base-blind and not a tree: it scores a column against the labels
        # directly, so it can rank a feature no tree ever splits on.
        from sklearn.feature_selection import (
            mutual_info_classif, mutual_info_regression,
        )
        fn = mutual_info_classif if is_classification else mutual_info_regression
        scores = np.nan_to_num(
            np.asarray(fn(X_scaled, y_train, random_state=42), dtype=float),
            nan=0.0)
    elif candidate_ranking == 'deeplift':
        scores = _deeplift_column_scores(X_scaled, y_train, is_classification)
        if scores is None:
            raise ImportError(
                "candidate_ranking='deeplift' needs torch and captum. "
                "Install them, or use 'rf_importance' for the other base-blind "
                "ranking.")
    elif candidate_ranking == 'residual_deeplift':
        # deeplift's base-aware twin: the same network, trained on what the base
        # representation cannot explain instead of on the labels. Always a
        # regression target — the residuals are continuous even when the task
        # is classification.
        residuals = _oof_base_residuals(X_base_train, y_train,
                                        is_classification, groups_train)
        scores = _deeplift_column_scores(X_scaled, residuals,
                                         is_classification=False)
        if scores is None:
            raise ImportError(
                "candidate_ranking='residual_deeplift' needs torch and captum. "
                "Install them, or use 'residual_correlation' for the other "
                "base-aware ranking.")
    elif candidate_ranking == 'residual_correlation':
        # Rank candidates by how well they explain what the base representation
        # CANNOT explain. The residual model is deliberately the SAME cheap
        # forest that 'rf_importance' uses, NOT the caller's downstream model:
        # this is a ranking device that runs once per create_hybrid call inside
        # a search loop, so fitting a GP / MLP / deep model n_splits times here
        # would be a silent multiplier on every augmentation-enabled search.
        residuals = _oof_base_residuals(X_base_train, y_train,
                                        is_classification, groups_train)
        scores = np.zeros(X_pool_tr.shape[1])
        for j in range(X_pool_tr.shape[1]):
            col = X_pool_tr[:, j]
            if np.std(col) > 1e-12:
                scores[j] = abs(np.corrcoef(col, residuals)[0, 1])
        scores = np.nan_to_num(scores, nan=0.0)
    else:
        raise ValueError(
            f"Unknown candidate_ranking: {candidate_ranking}. "
            f"Valid: {', '.join(CANDIDATE_RANKINGS)}.")

    order = np.argsort(scores)[::-1][:max_candidates]
    ranked_tr = X_pool_tr[:, order]
    ranked_vl = X_pool_vl[:, order]
    ranked_prov = [prov[i] for i in order]

    return ranked_tr, ranked_vl, ranked_prov


def augment_capped(
    X_base_train, X_base_val,
    base_feature_info,
    augmentations_train, augmentations_val,
    y_train, y_val,
    model,
    metric_fn,
    lower_is_better,
    total_budget,
    candidate_ranking='residual_correlation',
    patience=10,
    max_candidates=50,
    per_source_budget=None,
    margin=0.0,
    fraction=None,
    objective=OBJECTIVE_PREDICTIVE, is_clf=False,
    groups_val=None,
    groups_train=None,
    precomputed_ranking=None,
):
    """Capped augmentation: extras must DISPLACE base features to earn a slot.

    The finished feature count never exceeds ``total_budget``. Candidates are
    ranked once, the base hybrid's own columns are ranked once, and admission
    is a single choice of k from a small ladder: keep the base's best
    ``total_budget - k`` columns plus the top k candidates. k=0 leaves the base
    untouched and wins ties, so "no extras" is the default outcome.

    This is the only strategy where a supplementary feature has to be worth
    more than a base feature. `augment_uncapped` appends outside the budget, so
    there an "augmentation off" arm and an "augmentation on" arm at the same
    nominal budget are not the same width.

    REBUILT 2026-08-25. What it replaced, and why each part went:

    * **The ratchet.** Phase A walked the ranked candidates one at a time and
      kept each that beat a RUNNING MAXIMUM, so the bar only ever rose and a
      few lucky early accepts priced out everything after them. The identical
      loop in the sibling append path was measured claiming +0.021 on its own
      holdout while delivering -0.004 on an untouched fold, and admitting pure
      Gaussian noise. Here each k is scored once against the same split,
      independently, and 5 whole-block choices replace ~50 sequential accepts.
    * **Fill that never ran.** Phase A only had slots to fill when the
      allocation left the budget unspent. Under every shipping allocation it is
      spent, so in practice this strategy never filled and only evicted.
      Displacement is now the mechanism rather than a fallback, and spare
      budget (if any) is simply used before displacement starts.
    * **Per-candidate refits.** Phase B picked the base column to evict with
      permutation importance, refitting a model for EVERY candidate. That is
      why this strategy's own precompute note records it "has only ever
      completed for n<=242". One ranking fit now replaces all of it.
    * **Broken swap bookkeeping.** Phase B wrote the replacement into the base
      slot but decremented the base count, so an already-admitted extra could
      be evicted again as if it were a base column, while a genuine base column
      at the tail became unreachable. There is no in-place swapping left.
    * **The dropped per-source cap.** ``per_source_budget`` was never forwarded
      here, so a 2560-wide donor competed against a 40-wide one on column count
      alone -- while the cap still changed the stored result's identity, filing
      two runs that behaved identically as different results. It is forwarded.

    Parameters
    ----------
    X_base_train, X_base_val : ndarray
        Base hybrid features (already selected by the allocation).
    base_feature_info : dict
        Feature info from base hybrid creation. MUTATED in place: the entries
        for displaced base columns are removed, so that rebuilding the hybrid
        from this dict on other rows reproduces the same feature set.
    augmentations_train, augmentations_val : dict
        {aug_name: ndarray} of candidate augmentation features.
    y_train, y_val : ndarray
    model : estimator
        Downstream model used to score each k.
    metric_fn : callable
        metric_fn(y_true, y_pred) -> float.
    lower_is_better : bool
    total_budget : int
        Max features in the finished hybrid.
    candidate_ranking : str
        One of CANDIDATE_RANKINGS. Base-aware: 'residual_correlation'
        (default), 'residual_deeplift'. Base-blind: 'rf_importance',
        'lightgbm_gain', 'mutual_info', 'deeplift'.
    patience : int
        UNUSED. Kept so existing callers do not break -- there is no sequential
        accept loop left to be patient about.
    max_candidates : int
        Pre-ranking cutoff on the pooled candidate set.
    per_source_budget : int or dict, optional
        Top-N columns taken from EACH source before pooling. None = every
        column.
    margin : float
        How much better than keeping NOTHING a block of extras must score to
        be accepted. 0.0 (default) means any improvement wins, however small,
        so a near-tie is settled by luck. Swept as `augmentation_margin`.

    Returns
    -------
    dict with augmentation_info, kept_augmentations, replacements,
    phase_a_added, phase_b_replaced, baseline_score, final_score,
    total_improvement, n_base_dropped, notes.
    ``baseline_score`` is the k=0 score and ``final_score`` the chosen-k score
    on the SAME split, so ``total_improvement`` is exactly 0.0 when nothing is
    kept. ``replacements`` is always empty now: nothing is swapped in place, so
    there is no one-for-one correspondence to report.
    """
    is_clf_task = _is_classification_task(y_train)
    n_base = X_base_train.shape[1]

    empty = {
        'augmentation_info': {}, 'kept_augmentations': [],
        'replacements': [], 'phase_a_added': 0, 'phase_b_replaced': 0,
        'n_added': 0, 'n_base_dropped': 0,
        'baseline_score': float('nan'), 'final_score': float('nan'),
        'total_improvement': 0.0, 'notes': 'no candidates',
    }

    # The ranking depends on the donors, the per-donor cap, the ranking rule,
    # the base features and the labels -- and on NOTHING else. It does not
    # depend on the downstream model: `model` is passed positionally below but
    # appears nowhere in _pool_and_rank_aug_features' body, and the ranked
    # order was measured identical under a random forest, gradient boosting,
    # ridge and no model at all (2026-08-26). It does not depend on the size
    # rule, the margin, or on which of capped/uncapped is asking -- those two
    # make byte-identical calls.
    #
    # So a caller sweeping those settings recomputes one ranking many times.
    # On herg_ki that is 57% of the whole grid's model fits. Passing the
    # ranking in lets the caller compute it once per fold and reuse it.
    # (ranked_train, ranked_val, provenance), exactly what the function
    # returns. Leave it None and nothing changes.
    if precomputed_ranking is not None:
        ranked_tr, ranked_vl, prov = precomputed_ranking
    else:
        ranked_tr, ranked_vl, prov = _pool_and_rank_aug_features(
            augmentations_train, augmentations_val,
            y_train, X_base_train, model,
            candidate_ranking, is_clf_task, max_candidates,
            per_source_budget=per_source_budget,
            groups_train=groups_train,
        )
    if ranked_tr is None or n_base == 0:
        return empty

    # ---- rank the base hybrid's own columns, once ----
    # Needed because displacement has to drop the WEAKEST base columns, and the
    # per-rep importance scores recorded by the allocation are not comparable
    # across representations. One fit here replaces the per-candidate
    # permutation-importance pass that made this strategy unaffordable.
    try:
        _sc_b = StandardScaler()
        _Xb = _sc_b.fit_transform(
            np.nan_to_num(np.clip(X_base_train, -1e10, 1e10), nan=0.0))
        _rf_cls = (RandomForestClassifier if is_clf_task
                   else RandomForestRegressor)
        _rf_b = _rf_cls(n_estimators=50, max_depth=10, random_state=42,
                        n_jobs=-1)
        _rf_b.fit(_Xb, y_train)
        base_imp = np.nan_to_num(_rf_b.feature_importances_, nan=0.0)
        base_order = np.argsort(base_imp)[::-1]
        del _rf_b
    except Exception:
        # A pathological base matrix must not abort the whole search; fall back
        # to the existing column order, which makes displacement drop the tail.
        base_order = np.arange(n_base)

    # ---- map global base column positions back to (rep, local index) ----
    # Built BEFORE anything is chosen so a mismatch disables displacement
    # rather than corrupting the recipe. Each rep's columns occupy a
    # contiguous span of the concatenated base hybrid, in feature_info order.
    spans, cursor = [], 0
    for rep_name, info in base_feature_info.items():
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        n_rep = len(np.asarray(info['selected_indices']))
        spans.append((rep_name, cursor, n_rep))
        cursor += n_rep
    can_displace = (cursor == n_base)

    n_avail = ranked_tr.shape[1]
    spare = max(0, total_budget - n_base)
    if can_displace:
        k_cap = min(n_avail, max(0, total_budget - 1))
    else:
        # Cannot safely drop base columns, so extras may only use spare budget.
        k_cap = min(n_avail, spare)
    if fraction is None:
        k_ladder = sorted({min(k, k_cap) for k in (0, 5, 10, 20, 40)})
    else:
        # Fraction of the FINISHED width, which is what total_budget is here.
        # 5% of a 500 budget is 25 extras; 5% of a 100 budget is 5. The fixed
        # ladder gave both the same 5.
        k_ladder = [min(int(round(fraction * total_budget)), k_cap)]

    def _selection_for(k):
        """{aug_name: {selected_indices, n_features}} for the top k candidates.

        Grouped by donor in first-appearance order and sorted within a donor,
        which is exactly the order apply_augmentation_selection will rebuild
        them in. Scoring a different column order than the rebuild produces is
        not free: an otherwise identical block, merely permuted, moved a fold
        score across a 0.032 range in this project's own notes.
        """
        picked = {}
        for i in range(k):
            aug_name, col_idx = prov[i]
            picked.setdefault(aug_name, []).append(int(col_idx))
        return {
            name: {'selected_indices': np.array(sorted(set(cols)), dtype=int),
                   'n_features': len(set(cols))}
            for name, cols in picked.items()
        }

    def _cand_block(sel_info, source):
        parts = []
        for name, info in sel_info.items():
            X = source[name][:, info['selected_indices']]
            parts.append(np.nan_to_num(np.clip(X, -1e10, 1e10), nan=0.0))
        return np.hstack(parts) if parts else None

    def _keep_positions(k):
        """Base column positions retained alongside k extras, in file order."""
        m = min(n_base, total_budget - k)
        if m >= n_base:
            return np.arange(n_base)
        return np.sort(base_order[:max(1, m)])

    def _score_for(k):
        pos = _keep_positions(k)
        Xtr, Xvl = X_base_train[:, pos], X_base_val[:, pos]
        if k:
            sel = _selection_for(k)
            ctr = _cand_block(sel, augmentations_train)
            cvl = _cand_block(sel, augmentations_val)
            if ctr is not None:
                Xtr = np.hstack([Xtr, ctr])
                Xvl = np.hstack([Xvl, cvl])
        s = _evaluate_allocation(
            Xtr, Xvl, y_train, y_val,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            objective=objective, is_clf=is_clf, groups_val=groups_val,
        )
        # _evaluate_allocation is higher-is-better. Non-finite means
        # infeasible: -inf so it can never beat k=0 rather than dropping out
        # of the comparison.
        return float(s) if np.isfinite(s) else -np.inf

    # k=0 first and a strict '>' below, so ties go to the SMALLEST k: keeping
    # the base intact is the default outcome, not a fallback.
    baseline_score = _score_for(0)
    best_k, best_score = 0, baseline_score
    # Every rung's score, kept so a caller sweeping the margin does not have to
    # re-fit them. The margin only decides WHICH rung wins; it never changes
    # what any rung scores. Measured 2026-08-26: six margin values gave the
    # identical k=0 score and only two distinct outcomes, so four of the six
    # runs re-fit five models to reach an answer already computed. This is
    # DATA, not a key setting -- recording it changes no stored result.
    rung_scores = {0: baseline_score}
    if fraction is not None:
        # "Rank once, cut at the cap, no inner holdout" (2026-08-20). The grid
        # already fixed the size, so there is nothing to choose here and no
        # comparison to be fooled by. margin is not consulted; the two settings
        # are alternatives, not a stack. fraction=0.0 is the no-extras arm.
        best_k = k_ladder[0]
        best_score = _score_for(best_k) if best_k else baseline_score
        rung_scores[best_k] = best_score
        k_ladder = []
    for k in k_ladder:
        if k == 0:
            continue
        s = _score_for(k)
        rung_scores[k] = s
        # Two conditions. Clear the margin over keeping NOTHING, and be the
        # best of the rungs that do. Comparing only against the running best
        # would let k=5 clear the margin, then k=10 beat k=5 by a hair without
        # clearing the margin itself.
        if s > baseline_score + margin and s > best_score:
            best_k, best_score = k, s

    if best_k == 0:
        return {
            **empty,
            'baseline_score': baseline_score,
            'final_score': baseline_score,
            'rung_scores': rung_scores,
            'notes': (f'top-k displacement over k={k_ladder}; k=0 won, base '
                      f'left intact'),
        }

    augmentation_info = _selection_for(best_k)
    keep_pos = _keep_positions(best_k)
    dropped = set(range(n_base)) - set(int(p) for p in keep_pos)

    # Apply the displacement to the recipe itself, so a rebuild on other rows
    # selects the same base columns. Uses the spans computed above; the
    # cursor advances by each rep's ORIGINAL width whether or not it lost
    # columns.
    if dropped and can_displace:
        for rep_name, start, n_rep in spans:
            local = [p - start for p in dropped if start <= p < start + n_rep]
            if not local:
                continue
            info = base_feature_info[rep_name]
            sel = np.asarray(info['selected_indices'])
            mask = np.ones(n_rep, dtype=bool)
            mask[local] = False
            info['selected_indices'] = sel[mask]
            info['n_features'] = int(mask.sum())
            imp = info.get('importance_scores')
            if imp is not None and len(np.asarray(imp)) == n_rep:
                info['importance_scores'] = np.asarray(imp)[mask]

    # Honest split of the count: extras that fit in unspent budget did not
    # displace anything; the rest did.
    filled = min(best_k, spare)
    return {
        'augmentation_info': augmentation_info,
        'kept_augmentations': list(augmentation_info.keys()),
        'replacements': [],
        'phase_a_added': int(filled),
        'phase_b_replaced': int(best_k - filled),
        'n_added': int(best_k),
        'n_base_dropped': int(len(dropped)),
        'baseline_score': baseline_score,
        'final_score': best_score,
        'rung_scores': rung_scores,
        'total_improvement': best_score - baseline_score,
        'notes': (f'top-k displacement over k={k_ladder}, scored on the given '
                  f'split; {len(dropped)} base columns displaced'),
    }


def augment_uncapped(
    X_base_train, X_base_val,
    augmentations_train, augmentations_val,
    y_train, y_val,
    model,
    metric_fn,
    lower_is_better,
    candidate_ranking='residual_correlation',
    patience=10,
    max_candidates=50,
    per_source_budget=None,
    margin=0.0,
    fraction=None,
    objective=OBJECTIVE_PREDICTIVE, is_clf=False,
    groups_val=None,
    groups_train=None,
    precomputed_ranking=None,
):
    """Uncapped augmentation: keep a validated BLOCK of extra features, or none.

    Candidates are ranked once (by out-of-fold residual complementarity by
    default), then admission is a single top-k block choice: k is picked from a
    small ladder (0, 5, 10, 20, 40, clipped to the pool size), each k scored
    ONCE on the split passed in, and the best k wins. k=0 (keep nothing) is
    always on the ladder and wins ties, so "no extras" is the default outcome
    whenever the extras do not clearly help.

    This replaces the old one-at-a-time greedy loop, which accepted a column on
    a single split whenever it beat a running maximum. That running maximum is
    the max of many noisy scores, so it rose even on pure noise: the reported
    internal gain was a selection artefact and did not transfer to a held-out
    fold. There is no ratchet here — every k is scored against the same split,
    independently, and 5 whole-block scores are compared instead of ~50
    sequential accept decisions.

    Parameters
    ----------
    X_base_train, X_base_val : ndarray
        Base hybrid features.
    augmentations_train, augmentations_val : dict
        {aug_name: ndarray} of candidate augmentation features.
    y_train, y_val : ndarray
    model : estimator
        Downstream model used to score each k.
    metric_fn : callable
    lower_is_better : bool
    candidate_ranking : str
        One of CANDIDATE_RANKINGS; 'residual_correlation' by default.
    patience : int
        UNUSED. Kept only so existing callers do not break — there is no
        sequential accept loop left to be patient about.
    max_candidates : int
        Pre-ranking cutoff, applied to the pooled candidate set. This is a hard
        truncation of the ranked pool; the k-ladder is clipped to whatever
        survives it, so it can never exceed the pool size.
    per_source_budget : int, optional
        Top-N columns to take from EACH augmentation source before pooling.
        None = take every column (original behaviour).
    margin : float
        How much better than keeping NOTHING a block of extras must score to
        be accepted. 0.0 (default) means any improvement wins, however small,
        so a near-tie is settled by luck. Swept as `augmentation_margin`.

    Returns
    -------
    dict with augmentation_info, kept_augmentations, n_added,
    baseline_score, final_score, total_improvement, notes.
    baseline_score is the k=0 score and final_score the chosen-k score on the
    SAME split, so total_improvement is exactly 0.0 when nothing is kept.
    """
    # The GUESS from the label values, kept in its own name. It used to be
    # assigned to `is_clf`, overwriting the task type the caller passed in, so
    # the same dataset under 'uncapped' and under 'capped' could be scored as
    # two different task types -- augment_capped keeps the argument and names
    # its guess separately (is_clf_task, line 3907). Renamed 2026-09-13.
    is_clf_task = _is_classification_task(y_train)

    # The ranking depends on the donors, the per-donor cap, the ranking rule,
    # the base features and the labels -- and on NOTHING else. It does not
    # depend on the downstream model: `model` is passed positionally below but
    # appears nowhere in _pool_and_rank_aug_features' body, and the ranked
    # order was measured identical under a random forest, gradient boosting,
    # ridge and no model at all (2026-08-26). It does not depend on the size
    # rule, the margin, or on which of capped/uncapped is asking -- those two
    # make byte-identical calls.
    #
    # So a caller sweeping those settings recomputes one ranking many times.
    # On herg_ki that is 57% of the whole grid's model fits. Passing the
    # ranking in lets the caller compute it once per fold and reuse it.
    # (ranked_train, ranked_val, provenance), exactly what the function
    # returns. Leave it None and nothing changes.
    if precomputed_ranking is not None:
        ranked_tr, ranked_vl, prov = precomputed_ranking
    else:
        ranked_tr, ranked_vl, prov = _pool_and_rank_aug_features(
            augmentations_train, augmentations_val,
            y_train, X_base_train, model,
            # The GUESS here, matching augment_capped line 3938 -- the ranker
            # builds a classifier or a regressor from it and the two passes must
            # rank identically. The SCORE below uses the caller's is_clf.
            candidate_ranking, is_clf_task, max_candidates,
            per_source_budget=per_source_budget,
            groups_train=groups_train,
        )

    if ranked_tr is None:
        return {
            'augmentation_info': {}, 'kept_augmentations': [],
            'n_added': 0,
            'baseline_score': float('nan'), 'final_score': float('nan'),
        }

    n_avail = ranked_tr.shape[1]
    if fraction is None:
        k_ladder = sorted({min(k, n_avail) for k in (0, 5, 10, 20, 40)})
    else:
        # Extras are appended OUTSIDE the budget here, so the reference width
        # is the base hybrid's own width rather than a finished total.
        k_ladder = [min(int(round(fraction * X_base_train.shape[1])), n_avail)]

    # Every k is scored on the split we were handed, and on nothing else.
    # Extra shuffled sub-splits of the TRAINING rows were tried here and
    # REMOVED: the ranking above is fit on every training row, so such a
    # sub-split validates on rows the ranking already saw. Those votes are
    # selection-contaminated, they push toward larger k (measured: on pure
    # noise the clean split scored k=0 above k=5 while both sub-splits scored
    # k=5 above k=0), and two of them outvoted the one clean score. They were
    # also plain random splits, which this codebase does not do.
    notes = f'top-k block choice over k={k_ladder}, scored on the given split'

    def _selection_for_k(k):
        """{aug_name: sorted unique columns} for the top k candidates.

        Grouped by donor in first-appearance order and sorted within a donor —
        exactly the order apply_augmentation_selection rebuilds them in, and the
        same shape augment_capped's _selection_for produces.
        """
        picked = {}
        for i in range(k):
            aug_name, col_idx = prov[i]
            picked.setdefault(aug_name, []).append(int(col_idx))
        return {name: np.array(sorted(set(cols)), dtype=int)
                for name, cols in picked.items()}

    def _cand_block_k(sel, source):
        parts = []
        for name, cols in sel.items():
            X = source[name][:, cols]
            parts.append(np.nan_to_num(np.clip(X, -1e10, 1e10), nan=0.0))
        return np.hstack(parts) if parts else None

    def _score_for_k(k):
        # Score the block in the order the REBUILD produces, not in pooled rank
        # order. Ranking interleaves donors; apply_augmentation_selection
        # concatenates whole donors, and augmentation_info below records whole
        # donors too. augment_capped closes this gap with _selection_for /
        # _cand_block; this function scored a permutation of the block it then
        # recorded. An otherwise identical block, merely permuted, moved a fold
        # score across a 0.032 range in this project's own notes.
        # Fixed 2026-09-10 (metric-direction audit).
        _sel = _selection_for_k(k) if k else {}
        cand_tr = _cand_block_k(_sel, augmentations_train) if k else None
        cand_vl = _cand_block_k(_sel, augmentations_val) if k else None
        X_tr = np.hstack([X_base_train, cand_tr]) if cand_tr is not None else X_base_train
        X_vl = np.hstack([X_base_val, cand_vl]) if cand_vl is not None else X_base_val
        s = _evaluate_allocation(
            X_tr, X_vl, y_train, y_val,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            objective=objective, is_clf=is_clf, groups_val=groups_val,
        )
        # _evaluate_allocation is higher-is-better (it negates the metric when
        # lower_is_better). A non-finite score means infeasible — -inf so it can
        # never beat k=0, instead of being dropped from the comparison.
        return float(s) if np.isfinite(s) else -np.inf

    # k=0 first, and strict '>' below, so ties go to the SMALLEST k — keeping
    # nothing is the default outcome, not a fallback.
    baseline_score = _score_for_k(0)
    best_k, best_score = 0, baseline_score
    # See augment_capped: every rung's score, so a caller sweeping the margin
    # reads them instead of re-fitting them. DATA, not a key setting.
    rung_scores = {0: baseline_score}
    if fraction is not None:
        # See augment_capped: the grid fixed the size, so nothing is chosen
        # here and margin is not consulted.
        best_k = k_ladder[0]
        best_score = _score_for_k(best_k) if best_k else baseline_score
        rung_scores[best_k] = best_score
        k_ladder = []
    for k in k_ladder:
        if k == 0:
            continue
        k_score = _score_for_k(k)
        rung_scores[k] = k_score
        # Clear the margin over keeping NOTHING, and be the best rung that
        # does. Same rule as augment_capped.
        if k_score > baseline_score + margin and k_score > best_score:
            best_k, best_score = k, k_score

    aug_selected = {}
    for i in range(best_k):
        aug_name, col_idx = prov[i]
        aug_selected.setdefault(aug_name, []).append(col_idx)

    augmentation_info = {}
    for aug_name, cols in aug_selected.items():
        col_arr = np.array(sorted(set(cols)), dtype=int)
        augmentation_info[aug_name] = {
            'selected_indices': col_arr,
            'n_features': len(col_arr),
        }

    return {
        'augmentation_info': augmentation_info,
        'kept_augmentations': list(aug_selected.keys()),
        'n_added': best_k,
        'baseline_score': baseline_score,
        'final_score': best_score,
        'rung_scores': rung_scores,
        # Exactly 0.0 when nothing is kept — and -inf minus -inf (every k
        # infeasible) must not become a nan improvement.
        'total_improvement': (best_score - baseline_score) if best_k else 0.0,
        'notes': notes,
    }


# ============================================================================
# FEATURE IMPORTANCE COMPUTATION
# ============================================================================

def compute_feature_importance(base_reps, labels, method='treeshap', model=None, **kwargs):
    """
    Compute feature importance scores for each representation.

    Can either train a model internally (for feature selection) or explain a
    pre-trained model (for model interpretation).

    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values (n_samples,) - used for training if model=None
        method: Method for computing importance:
            Fast methods:
            - 'treeshap' (default): TreeSHAP values (MLP-stable, handles correlations)
            - 'xgboost_gain': XGBoost gain (avg improvement per split)
            - 'lightgbm_gain': LightGBM gain importance (#1 weight source)
            - 'deeplift': DeepLIFT attribution (Captum, fast neural method)
            Slower methods:
            - 'permutation': Permutation importance (model-agnostic, stable)
            - 'kernelshap': KernelSHAP (model-agnostic, most independent)
            Long search only (slow, correlated with deeplift):
            - 'integrated_gradients': Path integral attribution (Captum)
            - 'gradientshap': Gradient + SHAP sampling (Captum)
        model: Pre-trained model to explain (optional). If None, trains a new model.
            - For tree methods: sklearn RF/XGBoost/LightGBM model
            - For NN methods: PyTorch nn.Module
            - For model-agnostic: any model with predict/predict_proba
        **kwargs: Additional arguments for the importance method
            - n_estimators: Number of trees (default: 50)
            - max_depth: Max tree depth (default: 10)
            - random_state: Random seed (default: 42)
            - n_repeats: Permutation repeats, for 'permutation' only (default: 5)
            - hidden_dims: MLP hidden layers, for NN methods (default: [64, 32])
            - epochs: Training epochs, for NN methods (default: 100)
            - n_steps: Integration steps, for 'integrated_gradients' (default: 50)
            - background_samples: Background data size, for SHAP methods (default: 100)
            - nsamples: SHAP samples per instance, for 'kernelshap' (default: 'auto')
            - max_eval_samples: Max samples to evaluate (default: 100)

    Returns:
        dict: Dictionary with representation names as keys and importance score arrays as values

    Example:
        # Feature selection (trains internally):
        scores = compute_feature_importance(reps, labels, method='treeshap')

        # Explain pre-trained model:
        my_model = XGBRegressor().fit(X, y)
        scores = compute_feature_importance(reps, labels, method='treeshap', model=my_model)

    Note:
        mutual_info was removed - it's a univariate filter method that performed
        poorly in testing (60-68% recall vs 92-96% for RF/permutation). It cannot
        capture feature interactions and has high variance with limited samples.
    """
    importance_scores = {}
    is_classification = _is_classification_task(labels)

    # PERMUTATION IMPORTANCE
    if method == 'permutation':
        from sklearn.inspection import permutation_importance

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        n_repeats = kwargs.get('n_repeats', 5)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data (match random_forest implementation)
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                perm_model = model
            else:
                if is_classification:
                    perm_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                       random_state=random_state, n_jobs=-1)
                else:
                    perm_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                      random_state=random_state, n_jobs=-1)
                perm_model.fit(X_scaled, labels)

            perm = permutation_importance(perm_model, X_scaled, labels, n_repeats=n_repeats,
                                         random_state=random_state, n_jobs=-1)
            importance_scores[rep_name] = np.nan_to_num(perm.importances_mean, nan=0.0)

    # TREESHAP - SHAP values for tree-based models
    elif method == 'treeshap':
        try:
            import shap
        except ImportError:
            raise ImportError(
                "TreeSHAP requires the 'shap' package. "
                "Install with: conda install shap (recommended) or pip install shap"
            )

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        max_features_shap = kwargs.get('max_features_shap', 2000)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Pre-screen high-dimensional reps to avoid slow RF + SHAP.
            # Variance is read off X_clipped, BEFORE standardising. Read off
            # X_scaled it is 1.0 for every non-constant feature, so the sort was
            # ordering rounding error: measured spread 4.8e-06 across all 4800
            # features of caco2/grover, and the 2800 features that lost were
            # handed importance exactly 0 and could never be selected.
            selected_idx = None
            if X_scaled.shape[1] > max_features_shap:
                variances = np.var(X_clipped, axis=0)
                selected_idx = np.argsort(variances)[-max_features_shap:]
                selected_idx.sort()
                X_fit = X_scaled[:, selected_idx]
            else:
                X_fit = X_scaled

            # Use provided model or train new one
            if model is not None:
                tree_model = model
            else:
                if is_classification:
                    tree_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                       random_state=random_state, n_jobs=-1)
                else:
                    tree_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                      random_state=random_state, n_jobs=-1)
                tree_model.fit(X_fit, labels)

            # Compute SHAP values using TreeExplainer (fast, exact for trees)
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_fit)

            # Handle different SHAP output formats:
            # - Older SHAP: list of arrays [class0_values, class1_values, ...]
            # - Newer SHAP (0.40+): 3D array (n_samples, n_features, n_classes)
            # - Regression: 2D array (n_samples, n_features)
            if is_classification:
                if isinstance(shap_values, list):
                    # Older SHAP format: list of arrays
                    if len(shap_values) == 2:
                        # Binary: use positive class
                        shap_values = shap_values[1]
                    else:
                        # Multiclass: average absolute values across classes
                        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                elif shap_values.ndim == 3:
                    # Newer SHAP format: (n_samples, n_features, n_classes)
                    if shap_values.shape[2] == 2:
                        # Binary: use positive class
                        shap_values = shap_values[:, :, 1]
                    else:
                        # Multiclass: average absolute values across classes
                        shap_values = np.mean(np.abs(shap_values), axis=2)

            # Feature importance = mean absolute SHAP value per feature
            feature_importance_fit = np.mean(np.abs(shap_values), axis=0)

            # Map back to full feature space if pre-screened
            if selected_idx is not None:
                feature_importance = np.zeros(X_scaled.shape[1])
                feature_importance[selected_idx] = feature_importance_fit
            else:
                feature_importance = feature_importance_fit
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # INTEGRATED GRADIENTS - Neural network attribution
    elif method == 'integrated_gradients':
        try:
            import torch
            import torch.nn as nn
            from captum.attr import IntegratedGradients
        except ImportError:
            raise ImportError(
                "Integrated Gradients requires 'torch' and 'captum' packages. "
                "Install with: pip install torch captum"
            )

        hidden_dims = kwargs.get('hidden_dims', [64, 32])
        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.001)
        n_steps = kwargs.get('n_steps', 50)
        random_state = kwargs.get('random_state', 42)
        n_classes_hint = kwargs.get('n_classes', None)  # For multiclass with pre-trained model

        torch.manual_seed(random_state)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Convert to tensors
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Use provided model or train new one
            if model is not None:
                nn_model = model
                if not isinstance(nn_model, nn.Module):
                    raise ValueError("Provided model for integrated_gradients must be a PyTorch nn.Module")
                nn_model.eval()
                # Infer n_classes from model output
                with torch.no_grad():
                    sample_output = nn_model(X_tensor[:1])
                    if sample_output.dim() > 1 and sample_output.shape[1] > 1:
                        n_classes = sample_output.shape[1]
                    else:
                        n_classes = 2 if is_classification else 1
            else:
                # Build MLP
                input_dim = X_scaled.shape[1]
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    prev_dim = hidden_dim

                if is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        # Binary classification: single output with sigmoid
                        layers.append(nn.Linear(prev_dim, 1))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                        criterion = nn.BCEWithLogitsLoss()
                    else:
                        # Multiclass: softmax output
                        layers.append(nn.Linear(prev_dim, n_classes))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.long)
                        criterion = nn.CrossEntropyLoss()
                else:
                    n_classes = 1
                    # Regression: single output
                    layers.append(nn.Linear(prev_dim, 1))
                    nn_model = nn.Sequential(*layers)
                    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                    criterion = nn.MSELoss()

                # Train the model
                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
                nn_model.train()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = nn_model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()

                nn_model.eval()

            # Compute Integrated Gradients
            # For multi-output models, we need a wrapper that returns scalar
            if is_classification and n_classes > 2:
                # For multiclass, compute IG for each class and average
                all_attributions = []
                for class_idx in range(n_classes):
                    def forward_func(x, idx=class_idx):
                        return nn_model(x)[:, idx]

                    ig = IntegratedGradients(forward_func)
                    baseline = torch.zeros_like(X_tensor)
                    attr = ig.attribute(X_tensor, baselines=baseline, n_steps=n_steps)
                    all_attributions.append(attr.detach().numpy())

                # Average absolute attributions across classes
                attributions = np.mean([np.abs(a) for a in all_attributions], axis=0)
            else:
                # Binary classification or regression: single output
                def forward_func(x):
                    out = nn_model(x)
                    return out if out.dim() == 1 else out.squeeze(-1)

                ig = IntegratedGradients(forward_func)
                baseline = torch.zeros_like(X_tensor)
                attributions = ig.attribute(X_tensor, baselines=baseline, n_steps=n_steps)
                attributions = np.abs(attributions.detach().numpy())

            # Feature importance = mean absolute attribution per feature
            feature_importance = np.mean(attributions, axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # KERNELSHAP - Model-agnostic SHAP (slow but works with any model)
    elif method == 'kernelshap':
        try:
            import shap
        except ImportError:
            raise ImportError(
                "KernelSHAP requires the 'shap' package. "
                "Install with: conda install shap (recommended) or pip install shap"
            )

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 10)
        random_state = kwargs.get('random_state', 42)
        background_samples = kwargs.get('background_samples', 100)
        nsamples = kwargs.get('nsamples', 'auto')
        max_eval_samples = kwargs.get('max_eval_samples', 100)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            # Clean data
            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Use provided model or train new one
            if model is not None:
                kernel_model = model
                # Determine predict function based on model type
                if hasattr(kernel_model, 'predict_proba') and is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        predict_fn = lambda x, m=kernel_model: m.predict_proba(x)[:, 1]
                    else:
                        predict_fn = kernel_model.predict_proba
                else:
                    predict_fn = kernel_model.predict
            else:
                # Train model (using RF as the underlying model)
                if is_classification:
                    kernel_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                         random_state=random_state, n_jobs=-1)
                    kernel_model.fit(X_scaled, labels)

                    # For classification, use predict_proba
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        # Binary: use probability of positive class
                        predict_fn = lambda x, m=kernel_model: m.predict_proba(x)[:, 1]
                    else:
                        # Multiclass: use full probability matrix
                        predict_fn = kernel_model.predict_proba
                else:
                    kernel_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                        random_state=random_state, n_jobs=-1)
                    kernel_model.fit(X_scaled, labels)
                    predict_fn = kernel_model.predict

            # Create background dataset (subsample for speed)
            n_background = min(background_samples, X_scaled.shape[0])
            background = shap.sample(X_scaled, n_background, random_state=random_state)

            # Create KernelExplainer
            explainer = shap.KernelExplainer(predict_fn, background)

            # Compute SHAP values (subsample evaluation data for speed)
            # Seed a local RNG for reproducible, concurrency-safe sampling.
            n_eval = min(max_eval_samples, X_scaled.shape[0])
            eval_idx = np.random.RandomState(random_state).choice(
                X_scaled.shape[0], n_eval, replace=False)
            X_eval = X_scaled[eval_idx]

            # Suppress progress bar for cleaner output
            shap_values = explainer.shap_values(X_eval, nsamples=nsamples, silent=True)

            # Handle multiclass output (list of arrays)
            if isinstance(shap_values, list):
                # Average absolute values across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

            # Feature importance = mean absolute SHAP value per feature
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # XGBOOST GAIN - Average improvement per split
    elif method == 'xgboost_gain':
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost methods require 'xgboost'. Install with: pip install xgboost")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)

        max_features_xgb = kwargs.get('max_features_xgb', 2000)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Pre-screen high-dimensional reps. Variance is read off X_clipped,
            # BEFORE standardising — after it, every non-constant feature has
            # variance 1.0 and the sort orders rounding error. See the note on
            # the same screen in the treeshap branch.
            selected_idx = None
            if X_scaled.shape[1] > max_features_xgb:
                variances = np.var(X_clipped, axis=0)
                selected_idx = np.argsort(variances)[-max_features_xgb:]
                selected_idx.sort()
                X_fit = X_scaled[:, selected_idx]
            else:
                X_fit = X_scaled

            # Use provided model or train new one
            if model is not None:
                xgb_model = model
                if not hasattr(xgb_model, 'get_booster'):
                    raise ValueError("Provided model for xgboost_gain must be an XGBoost model with get_booster()")
            else:
                if is_classification:
                    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                 random_state=random_state, n_jobs=-1, verbosity=0)
                else:
                    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                random_state=random_state, n_jobs=-1, verbosity=0)
                xgb_model.fit(X_fit, labels)

            booster = xgb_model.get_booster()
            scores = booster.get_score(importance_type='gain')

            # Convert to array
            feature_importance_fit = np.zeros(X_fit.shape[1])
            for fname, score in scores.items():
                idx = int(fname[1:])  # 'f0' -> 0
                feature_importance_fit[idx] = score

            # Map back to full feature space if pre-screened
            if selected_idx is not None:
                feature_importance = np.zeros(X_scaled.shape[1])
                feature_importance[selected_idx] = feature_importance_fit
            else:
                feature_importance = feature_importance_fit

            importance_scores[rep_name] = np.nan_to_num(feature_importance.astype(float), nan=0.0)

    # LIGHTGBM GAIN
    elif method == 'lightgbm_gain':
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM methods require 'lightgbm'. Install with: pip install lightgbm")

        n_estimators = kwargs.get('n_estimators', 50)
        max_depth = kwargs.get('max_depth', 6)
        random_state = kwargs.get('random_state', 42)
        max_features_lgb = kwargs.get('max_features_lgb', 2000)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)

            # Pre-screen high-dimensional reps to avoid slow LightGBM fits
            selected_idx = None
            if X_scaled.shape[1] > max_features_lgb:
                # Use variance as a fast pre-filter, keep top max_features_lgb.
                # Read off X_clipped, BEFORE standardising — after it every
                # non-constant feature has variance 1.0 and the sort orders
                # rounding error. See the note in the treeshap branch.
                variances = np.var(X_clipped, axis=0)
                selected_idx = np.argsort(variances)[-max_features_lgb:]
                selected_idx.sort()
                X_fit = X_scaled[:, selected_idx]
            else:
                X_fit = X_scaled

            # Use provided model or train new one
            if model is not None:
                lgb_model = model
                if not hasattr(lgb_model, 'booster_'):
                    raise ValueError("Provided model for lightgbm_gain must be a fitted LightGBM model with booster_")
            else:
                if is_classification:
                    lgb_model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  random_state=random_state, n_jobs=1, verbosity=-1)
                else:
                    lgb_model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                 random_state=random_state, n_jobs=1, verbosity=-1)
                lgb_model.fit(X_fit, labels)

            feature_importance_fit = lgb_model.booster_.feature_importance(importance_type='gain')

            # Map back to full feature space if pre-screened
            if selected_idx is not None:
                feature_importance = np.zeros(X_scaled.shape[1])
                feature_importance[selected_idx] = feature_importance_fit
            else:
                feature_importance = feature_importance_fit

            importance_scores[rep_name] = np.nan_to_num(feature_importance.astype(float), nan=0.0)

    # DEEPLIFT - DeepLIFT attribution
    elif method == 'deeplift':
        try:
            import torch
            import torch.nn as nn
            from captum.attr import DeepLift
        except ImportError:
            raise ImportError("DeepLIFT requires 'torch' and 'captum'. Install with: pip install torch captum")

        hidden_dims = kwargs.get('hidden_dims', [64, 32])
        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.001)
        random_state = kwargs.get('random_state', 42)

        torch.manual_seed(random_state)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Use provided model or train new one
            if model is not None:
                nn_model = model
                if not isinstance(nn_model, nn.Module):
                    raise ValueError("Provided model for deeplift must be a PyTorch nn.Module")
                nn_model.eval()
                # Infer n_classes from model output
                with torch.no_grad():
                    sample_output = nn_model(X_tensor[:1])
                    if sample_output.dim() > 1 and sample_output.shape[1] > 1:
                        n_classes = sample_output.shape[1]
                    else:
                        n_classes = 2 if is_classification else 1
            else:
                # Build MLP
                input_dim = X_scaled.shape[1]
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                    prev_dim = hidden_dim

                if is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        layers.append(nn.Linear(prev_dim, 1))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                        criterion = nn.BCEWithLogitsLoss()
                    else:
                        layers.append(nn.Linear(prev_dim, n_classes))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.long)
                        criterion = nn.CrossEntropyLoss()
                else:
                    n_classes = 1
                    layers.append(nn.Linear(prev_dim, 1))
                    nn_model = nn.Sequential(*layers)
                    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                    criterion = nn.MSELoss()

                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
                nn_model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = criterion(nn_model(X_tensor), y_tensor)
                    loss.backward()
                    optimizer.step()

                nn_model.eval()

            # DeepLIFT attribution - need to use model directly, not lambda
            baseline = torch.zeros_like(X_tensor)
            dl = DeepLift(nn_model)

            if is_classification and n_classes > 2:
                # Multiclass: compute attribution for each class and average
                all_attributions = []
                for class_idx in range(n_classes):
                    attr = dl.attribute(X_tensor, baselines=baseline, target=class_idx)
                    all_attributions.append(np.abs(attr.detach().numpy()))
                attributions = np.mean(all_attributions, axis=0)
            else:
                # Binary classification or regression
                attributions = dl.attribute(X_tensor, baselines=baseline)
                attributions = np.abs(attributions.detach().numpy())

            feature_importance = np.mean(attributions, axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    # GRADIENTSHAP - Gradient + SHAP sampling
    elif method == 'gradientshap':
        try:
            import torch
            import torch.nn as nn
            from captum.attr import GradientShap
        except ImportError:
            raise ImportError("GradientSHAP requires 'torch' and 'captum'. Install with: pip install torch captum")

        hidden_dims = kwargs.get('hidden_dims', [64, 32])
        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.001)
        random_state = kwargs.get('random_state', 42)
        background_samples = kwargs.get('background_samples', 50)
        n_samples = kwargs.get('n_samples', 50)

        torch.manual_seed(random_state)

        for rep_name, X in base_reps.items():
            if X.shape[1] == 0:
                raise ValueError(f"Representation '{rep_name}' has 0 features")

            X_clipped = np.clip(X, -1e10, 1e10)
            X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clipped)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Use provided model or train new one
            if model is not None:
                nn_model = model
                if not isinstance(nn_model, nn.Module):
                    raise ValueError("Provided model for gradientshap must be a PyTorch nn.Module")
                nn_model.eval()
                # Infer n_classes from model output
                with torch.no_grad():
                    sample_output = nn_model(X_tensor[:1])
                    if sample_output.dim() > 1 and sample_output.shape[1] > 1:
                        n_classes = sample_output.shape[1]
                    else:
                        n_classes = 2 if is_classification else 1
            else:
                # Build MLP
                input_dim = X_scaled.shape[1]
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                    prev_dim = hidden_dim

                if is_classification:
                    n_classes = len(np.unique(labels))
                    if n_classes == 2:
                        layers.append(nn.Linear(prev_dim, 1))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                        criterion = nn.BCEWithLogitsLoss()
                    else:
                        layers.append(nn.Linear(prev_dim, n_classes))
                        nn_model = nn.Sequential(*layers)
                        y_tensor = torch.tensor(labels, dtype=torch.long)
                        criterion = nn.CrossEntropyLoss()
                else:
                    n_classes = 1
                    layers.append(nn.Linear(prev_dim, 1))
                    nn_model = nn.Sequential(*layers)
                    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                    criterion = nn.MSELoss()

                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
                nn_model.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = criterion(nn_model(X_tensor), y_tensor)
                    loss.backward()
                    optimizer.step()

                nn_model.eval()

            # Background samples
            # Seed a local RNG for reproducible, concurrency-safe sampling.
            n_bg = min(background_samples, X_tensor.shape[0])
            bg_idx = np.random.RandomState(random_state).choice(
                X_tensor.shape[0], n_bg, replace=False)
            baseline = X_tensor[bg_idx]

            if is_classification and n_classes > 2:
                all_attributions = []
                for class_idx in range(n_classes):
                    gs = GradientShap(lambda x, idx=class_idx, m=nn_model: m(x)[:, idx])
                    attr = gs.attribute(X_tensor, baselines=baseline, n_samples=n_samples,
                                       stdevs=0.0)
                    all_attributions.append(attr.detach().numpy())
                attributions = np.mean([np.abs(a) for a in all_attributions], axis=0)
            else:
                gs = GradientShap(lambda x, m=nn_model: m(x).squeeze(-1) if m(x).dim() > 1 else m(x))
                attributions = gs.attribute(X_tensor, baselines=baseline, n_samples=n_samples,
                                           stdevs=0.0)
                attributions = np.abs(attributions.detach().numpy())

            feature_importance = np.mean(attributions, axis=0)
            importance_scores[rep_name] = np.nan_to_num(feature_importance, nan=0.0)

    else:
        raise NotImplementedError(
            f"Method '{method}' not implemented. Choose from: "
            "treeshap, xgboost_gain, lightgbm_gain, deeplift, permutation, "
            "kernelshap, integrated_gradients, gradientshap"
        )

    return importance_scores


# ============================================================================
# FEATURE CONCATENATION
# ============================================================================

def concatenate_features(base_reps, importance_scores, allocation):
    """
    Concatenate top features from each representation based on allocation.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        importance_scores: Dict with representation names as keys and importance arrays as values
        allocation: Dict with representation names as keys and n_features as values
                   OR int for same allocation across all reps
        
    Returns:
        tuple: (hybrid_features, feature_info)
            - hybrid_features: Concatenated array of selected features (clipped and cleaned)
            - feature_info: Dict with metadata about selected features
    """
    if not isinstance(allocation, dict):
        raise TypeError(f"allocation must be a dict, got {type(allocation).__name__}")
    
    selected_parts = []
    feature_info = {}
    
    for rep_name in base_reps.keys():
        if rep_name not in allocation:
            raise KeyError(f"allocation missing key '{rep_name}'")
        
        X = base_reps[rep_name]
        scores = importance_scores[rep_name]
        n_select = allocation[rep_name]
        
        if n_select == 0:
            continue
        
        # Verify dimension match
        if len(scores) != X.shape[1]:
            raise ValueError(
                f"Dimension mismatch for '{rep_name}': importance_scores has {len(scores)} values "
                f"but features has {X.shape[1]} columns. Ensure importance was computed on same data."
            )
        
        # Clean data (consistent with all other functions)
        X_clipped = np.clip(X, -1e10, 1e10)
        X_clipped = np.nan_to_num(X_clipped, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # ── Select the top N features, but THE BUDGET IS A MAXIMUM ───────────
        #
        # Owner's framing, 2026-09-10: the budget is the most features the
        # hybrid may use, not a quota it has to fill. A smaller set that scores
        # better is a better answer, not a short one.
        #
        # So a representation stops at its last feature with a POSITIVE
        # importance score. It used to take exactly n_select whatever they
        # scored, and importance methods produce large blocks of exact zeros:
        # lightgbm_gain writes 0.0 for every feature no tree split on, and 0.0
        # again for everything outside the 2000-feature variance pre-screen it
        # applies to a wider representation. Measured on caco2 with ecfp4, 201
        # of 2048 features scored above zero. Once a quota passed that count the
        # rest of the slots were filled from a block of exact ties, which
        # np.argsort breaks by column index — so the hybrid was padded with
        # features chosen by their position in the matrix.
        #
        # If NOTHING scored above zero the method had no signal to give, and
        # cutting to nothing would silently drop the representation the
        # allocator chose. Take the quota, and record that it happened.
        n_requested = int(min(n_select, X_clipped.shape[1]))
        _positive = int(np.sum(np.asarray(scores) > 0))
        if _positive > 0:
            n_select = min(n_requested, _positive)
        else:
            n_select = n_requested
        top_indices = np.argsort(scores)[-n_select:][::-1]

        # Extract selected features
        X_selected = X_clipped[:, top_indices]
        selected_parts.append(X_selected)

        # Store metadata
        feature_info[rep_name] = {
            'selected_indices': top_indices,
            'importance_scores': scores[top_indices],
            'n_features': n_select,
            # What the allocator asked for, against what the importance method
            # could actually justify. Equal on a representation whose features
            # were all scored; n_requested larger when the budget is a maximum
            # the method could not fill.
            'n_requested': n_requested,
            'n_scored_above_zero': _positive,
            'n_unfilled': int(n_requested - n_select),
        }
    
    # Concatenate all selected features
    if len(selected_parts) == 0:
        raise ValueError(
            "No features selected. Check that allocation has non-zero values "
            "for at least one representation."
        )
    
    hybrid_features = np.hstack(selected_parts).astype(np.float32)

    # Say it when the budget could not be filled. The width is recorded per
    # representation above; this is the one line a reader of the log sees, and
    # without it a hybrid that came back narrower than the budget looks like a
    # hybrid that was asked for less.
    _short = {rn: i['n_unfilled'] for rn, i in feature_info.items()
              if isinstance(i, dict) and i.get('n_unfilled')}
    if _short:
        _total = sum(_short.values())
        warn_once(
            f"the budget is a MAXIMUM and it was not filled: {_total} feature "
            f"slots went unused because no feature left in those "
            f"representations scored above zero importance "
            f"({dict(sorted(_short.items(), key=lambda kv: -kv[1])[:5])}). The "
            f"hybrid is {hybrid_features.shape[1]} features wide. Padding them "
            f"out would have added features the importance method ranked at "
            f"exactly zero, chosen by column order.")

    return hybrid_features, feature_info


def apply_feature_selection(base_reps, feature_info, augmentations=None,
                            base_only=False):
    """
    Apply pre-computed feature selection to new data (e.g., test set).

    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        feature_info: Dict from create_hybrid containing selected_indices for each rep
        augmentations: Dict of {aug_name: features} for the SAME new rows. Pass
            them when the fit admitted extra features (feature_info carries a
            non-empty 'augmentation_info') and this call should return the WHOLE
            matrix. create_hybrid appends those columns after the base ones, so
            leaving them out returns a NARROWER matrix than the fit produced --
            100 features against 140 on qm9, measured 2026-09-10 -- which is
            told apart from the real thing only by a shape error at predict
            time, or by nothing at all when the training side came through here
            too.
        base_only: say so explicitly when the caller appends the extras itself
            with apply_augmentation_selection, which about twenty experiments
            do. Without either argument a recipe carrying extras still returns
            the base block, but says so loudly.

    Returns:
        np.ndarray: Concatenated selected features (clipped and cleaned)
    """
    selected_parts = []
    
    # Iterate over feature_info keys to maintain same order as training
    # (feature_info preserves the order from create_hybrid)
    for rep_name in feature_info.keys():
        # Skip metadata keys like 'allocation', 'greedy_history', 'augmentation_info'
        if not isinstance(feature_info[rep_name], dict) or 'selected_indices' not in feature_info[rep_name]:
            continue
            
        if rep_name not in base_reps:
            raise ValueError(f"Representation '{rep_name}' found in feature_info but not in base_reps")
        
        X = base_reps[rep_name]
        indices = feature_info[rep_name]['selected_indices']
        
        # Check that indices are valid for this data
        max_idx = np.max(indices) if len(indices) > 0 else -1
        if max_idx >= X.shape[1]:
            raise ValueError(
                f"Feature index {max_idx} out of bounds for representation '{rep_name}' "
                f"with {X.shape[1]} features. Ensure test data has same number of features as training data."
            )
        
        # Apply same selection as training
        X_selected = X[:, indices]
        
        # Apply same clipping as training (CRITICAL for consistency)
        X_selected = np.clip(X_selected, -1e10, 1e10)
        X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=1e10, neginf=-1e10)
        
        selected_parts.append(X_selected)
    
    if len(selected_parts) == 0:
        # This used to say only that feature_info was empty, which names a
        # symptom and no cause. Every observed instance has been an allocation
        # that handed out no slots — so say which representations were offered,
        # what each one's entry looked like, and where to look. Expanded
        # 2026-09-12 after the same bare message came out of two different
        # experiments with two different causes.
        _offered = sorted(base_reps.keys()) if isinstance(base_reps, dict) else []
        _entries = []
        for _rn in _offered:
            _e = feature_info.get(_rn)
            if _e is None:
                _entries.append(f"{_rn}: no entry")
            elif not isinstance(_e, dict):
                _entries.append(f"{_rn}: entry is {type(_e).__name__}, not a dict")
            elif 'selected_indices' not in _e:
                _entries.append(f"{_rn}: entry has no selected_indices")
            else:
                _entries.append(
                    f"{_rn}: {len(np.asarray(_e['selected_indices']).ravel())} "
                    f"selected")
        raise ValueError(
            "No features to select: not one representation came back with a "
            "feature to take, so there is nothing to build a matrix from.\n"
            f"  representations offered: {_offered or 'none'}\n"
            f"  what feature_info holds for each: "
            f"{'; '.join(_entries) if _entries else 'nothing'}\n"
            "  The usual cause is an allocation that gave every representation "
            "a budget of zero, which happens when every allocation score is "
            "zero or non-finite. Both allocation paths now fall back to an "
            "equal split and warn when that happens, so check the run's "
            "warnings for an 'every representation weighed zero' line."
        )

    X_base = np.hstack(selected_parts).astype(np.float32)

    # The extras the fit admitted, in the same place create_hybrid put them:
    # after every base block.
    aug_info = feature_info.get('augmentation_info') or {}
    if aug_info:
        if augmentations is None:
            if not base_only:
                _widths = {k: int(len(v.get('selected_indices', [])))
                           for k, v in aug_info.items()}
                warn_once(
                    f"this recipe admitted {sum(_widths.values())} extra "
                    f"features from {len(_widths)} donor(s) {_widths} on top "
                    f"of {X_base.shape[1]} base features, and no augmentation "
                    f"arrays were passed for these rows, so what comes back is "
                    f"the base block ALONE -- a different feature set from the "
                    f"one that was fitted. Pass augmentations={{name: array}} "
                    f"to get the whole matrix, or base_only=True if you append "
                    f"them yourself with apply_augmentation_selection.")
        else:
            X_aug = apply_augmentation_selection(augmentations, aug_info)
            if X_aug is not None:
                X_base = np.hstack([X_base, X_aug]).astype(np.float32)

    return X_base


def apply_augmentation_selection(augmentations, augmentation_info):
    """
    Apply pre-computed augmentation selection to new data (e.g., test set).
    
    Args:
        augmentations: Dict of {aug_name: features}
        augmentation_info: Dict from create_hybrid containing selected_indices for each aug
        
    Returns:
        np.ndarray: Concatenated selected augmentation features
    """
    if not augmentation_info:
        return None
    
    selected_parts = []
    
    for aug_name, info in augmentation_info.items():
        if aug_name not in augmentations:
            raise ValueError(f"Augmentation '{aug_name}' found in augmentation_info but not in augmentations")
        
        X = augmentations[aug_name]
        indices = info['selected_indices']
        
        # Check bounds
        max_idx = np.max(indices) if len(indices) > 0 else -1
        if max_idx >= X.shape[1]:
            raise ValueError(
                f"Feature index {max_idx} out of bounds for augmentation '{aug_name}' "
                f"with {X.shape[1]} features."
            )
        
        X_selected = X[:, indices]
        X_selected = np.clip(X_selected, -1e10, 1e10)
        X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=1e10, neginf=-1e10)
        
        selected_parts.append(X_selected)
    
    if len(selected_parts) == 0:
        return None
    
    return np.hstack(selected_parts).astype(np.float32)


# ============================================================================
# MAIN HYBRID CREATION
# ============================================================================

def internal_holdout_split(n_samples, labels, groups=None, validation_split=0.2):
    """The train/held-out carve create_hybrid makes inside the rows it is given.

    Returns (train_idx, val_idx) as index arrays into the rows passed in.

    ONE definition, because two places need the identical carve: create_hybrid
    itself, and find_best_hybrid's precompute of the per-representation scores
    that set the budget split. Those two drifted apart on 2026-08-20, when this
    carve moved from a single GroupShuffleSplit draw to the first fold of
    GroupKFold and the precompute was left on the old draw. The precompute then
    handed create_hybrid scores measured on different held-out rows than
    create_hybrid would have used, under a comment saying the two matched.

    With groups: the first fold of GroupKFold over 1/validation_split splits,
    capped at the number of distinct groups. Deterministic, no random_state, and
    the same geometry as the outer GroupKFold cross-validation.
    Without groups: a fixed-seed random carve of validation_split of the rows.
    """
    n_samples = int(n_samples)
    if groups is not None and len(np.unique(groups)) >= 2:
        from sklearn.model_selection import GroupKFold
        n_groups = len(np.unique(groups))
        n_splits = max(2, min(int(round(1.0 / validation_split)), n_groups))
        gkf = GroupKFold(n_splits=n_splits)
        return next(gkf.split(np.zeros(n_samples), labels, groups))
    n_val = max(int(n_samples * validation_split), 1)
    indices = np.random.RandomState(42).permutation(n_samples)
    return indices[n_val:], indices[:n_val]


def create_hybrid(base_reps, labels,
                 # DEFAULT SET 2026-09-10 (FRAMEWORK_FINALIZATION.md 3.1). Was
                 # 'top_3', which welds a representation COUNT into the
                 # weighting slot and came out of DEFAULT_ALLOCATION_METHODS on
                 # 2026-09-01 while this default was left behind. Chosen off the
                 # c32 cross sweep, 54 cells (9 datasets x 3 models x budgets
                 # 500/1000): performance_weighted and power_2 each win 14 cells
                 # and second places break the tie 29 to 28. Every representation
                 # gets a share of the budget proportional to its own score.
                 allocation_method='performance_weighted',
                 n_per_rep=None,
                 budget=100,
                 step_size=5,
                 patience=20,
                 validation_split=0.2,
                 apply_filters=False,
                 importance_method='lightgbm_gain',
                 # Augmentation parameters
                 augmentations=None,
                 augmentation_strategy='uncapped',
                 augmentation_threshold=None,
                 augmentation_budget=20,
                 candidate_ranking='residual_correlation',
                 # Model-aware augmentation (capped/uncapped strategies)
                 model=None,
                 metric_fn=None,
                 lower_is_better=False,
                 # Internal evaluator for greedy/performance_weighted allocation.
                 # 'rf', not lightgbm. Set 2026-09-16: f29's per-representation
                 # fits (the nearest real numbers -- no experiment scores
                 # internal_evaluator choice directly, see
                 # FRAMEWORK_FINALIZATION.md f31) show rf 2x-20x cheaper than
                 # lightgbm on every one of 5 datasets, tying on 1, winning on
                 # 1, losing on 3 (worst: logd rmse 0.6126 vs 0.5066). The prior
                 # default, 'lightgbm', came from a 2026-07-16 sweep that
                 # replaced every 'ridge' default across this file -- never a
                 # comparison between evaluators.
                 internal_evaluator='rf',
                 # Pre-computed scores (deduplication optimisation)
                 precomputed_importance=None,
                 precomputed_alloc_scores=None,
                 # Group-aware internal split
                 groups=None,
                 # Search objective. 'predictive' (default), 'similarity',
                 # 'struct_fid', or a dict blending the last two — in geometry
                 # mode the allocators maximise the held-out geometry score
                 # instead of model performance.
                 objective=OBJECTIVE_PREDICTIVE, is_clf=False,
                 # Whole-dataset row number of each row of base_reps, in order,
                 # and the MCS pairs keyed by those numbers. Both are needed by
                 # objective='struct_fid' and by any blend containing it, so the
                 # budget is split by the quantity the run selects on.
                 row_index=None, structural_pairs=None,
                 **kwargs):
    """
    Create hybrid representation by combining features from multiple base representations.
    
    Args:
        base_reps: Dict with representation names as keys and feature arrays as values
        labels: Target values for computing importance
        
        allocation_method: Method for allocating features across representations.
            Base methods:
            - 'performance_weighted': Allocate proportional to baseline scores (default)
            - 'greedy': CUT 2026-04-09, do not use (see NAMED_ALLOCATIONS)
            - 'dominant_greedy': CUT, do not use (see NAMED_ALLOCATIONS)
            - 'fixed': Equal split of `budget` across the reps
            Named variants (see NAMED_ALLOCATIONS for full list):
            - 'top_2', 'top_3', 'top_5': Top-k reps only
            - 'pairdist_0.5', 'pairdist_1.0', 'pairdist_2.0': redundancy
              penalty measured on pairwise distances, applied to rank
              ('pairdist_score_*' applies it to the score instead)
            - 'entropy_0.5', 'entropy_1.0', 'entropy_2.0': Entropy-weighted
            - 'rank_exp_d0.1'...'rank_exp_d0.5': Rank exponential decay
            - 'greedy_fill_s5'..., 'greedy_forced_s5'...: CUT, do not use
            
        n_per_rep: Fixed number of features per rep, for 'fixed' only.
            DEFAULT None = split `budget` evenly across the reps, which is what
            'fixed' has always claimed to do. Before 2026-08-26 this defaulted
            to 100 and 'fixed' IGNORED `budget` entirely, so with 11 reps it ran
            at 1100 features while every other allocation ran at the requested
            budget — see f01, where that inflation is the whole of `fixed`'s
            apparent win. Pass an integer only to reproduce that old behaviour;
            the total width is then n_per_rep * n_reps, not `budget`.
        budget: Total feature budget (for 'greedy' method, default: 100)
        step_size: Features to add per greedy iteration (default: 5)
        patience: Early stopping patience for greedy, None to disable (default: 20)
        validation_split: Fraction for validation in greedy (default: 0.2)
        
        apply_filters: Apply quality filters (sparsity + variance) before allocation (default: False)
        importance_method: Method for feature importance (default: 'treeshap').
            NOTE: 'greedy' and 'dominant_greedy' allocations ignore this parameter —
            they use internal RF feature_importances_. All performance_weighted
            variants (softmax_T*, top_*, entropy_*, etc.) use the specified
            importance_method.
        
        augmentations: Dict of {aug_name: feature_array} for supplementary features (default: None)
            These are evaluated separately from base_reps after initial hybrid creation
        augmentation_strategy: How to handle augmentations
            - 'uncapped': validated top-k block of extra features, no budget
              limit. k is chosen from (0, 5, 10, 20, 40), each scored once on
              the internal split; k=0 (keep nothing) is reachable and wins
              ties (default)
            - 'greedy_ablation': Test each rep, keep only those that improve
            - 'capped': validated top-k block that DISPLACES base features, so
              the total stays within budget. Same k ladder and same tie rule as
              'uncapped'; the difference is that the base is cut to
              (budget - k) columns instead of the extras being appended
              outside the budget
            - 'all': Include all augmentations
            - 'none': Ignore augmentations
        augmentation_threshold: Minimum improvement to keep augmentation (default: None = any improvement)
            Can be float (e.g., 0.01 for 1% R² improvement) or None
        augmentation_budget: Max features per augmentation to select (default: 20)
        candidate_ranking: How to rank augmentation features for capped/uncapped strategies
            One of CANDIDATE_RANKINGS. The two base-aware rules
            ('residual_correlation', 'residual_deeplift') rank against what the
            base cannot explain; the four base-blind ones ('rf_importance',
            'lightgbm_gain', 'mutual_info', 'deeplift') rank against the labels.
            - 'residual_correlation': Rank by correlation with the OUT-OF-FOLD
              residuals of a cheap internal forest fitted on the base features,
              i.e. by what the base cannot already explain (default)
            - 'rf_importance': Rank by RF feature importance against the labels.
              Base-blind — it cannot see redundancy with the base.

        model: sklearn-compatible estimator for model-aware augmentation (default: None)
            Used by 'capped' and 'uncapped' strategies. Deepcopied for each evaluation.
        metric_fn: callable(y_true, y_pred) -> float for model-aware augmentation
        lower_is_better: Whether lower metric_fn values are better (default: False)
        
        **kwargs: Additional arguments for importance computation
        
    Returns:
        tuple: (hybrid_features, feature_info)
            - hybrid_features: Combined feature array
            - feature_info: Metadata about selected features (includes 'allocation' for greedy,
                           'augmentation_info' if augmentations used)
    """
    # Pop augmentation-only knobs up front. Leftover **kwargs are forwarded to
    # compute_feature_importance() further down, so they must not survive past
    # here — and the uncapped branch is not always reached.
    _aug_max_candidates = kwargs.pop('aug_max_candidates', 50)
    _aug_per_source_budget = kwargs.pop('aug_per_source_budget', None)
    # How much better than "keep no extras" a block of extras must score
    # before it is accepted. 0.0 reproduces the behaviour before 2026-08-26:
    # any improvement wins, however small, so a near-tie is settled by luck.
    # For 'capped' a lost near-tie also throws away real base features.
    _aug_margin = float(kwargs.pop('augmentation_margin', 0.0) or 0.0)
    # Size the extras as a fraction of the budget rather than by the fixed
    # ladder of 0/5/10/20/40. None keeps the ladder. Set means: take exactly
    # round(fraction * width) top-ranked extras and make no internal
    # comparison at all -- the 2026-08-20 decision, "rank once, cut at the
    # cap, no inner holdout". The grid sweeps the value; the engine never
    # chooses it.
    _aug_fraction = kwargs.pop('augmentation_fraction', None)
    # A candidate ranking the CALLER already computed, as the
    # (ranked_train, ranked_val, provenance) triple _pool_and_rank_aug_features
    # returns. None recomputes it, which is what every existing caller does.
    #
    # Only for a caller sweeping settings the ranking does not depend on: the
    # downstream model, the size rule, the margin, and capped-versus-uncapped.
    # It DOES depend on the donors, the per-donor cap, the ranking rule, the
    # base features and the labels, so a caller passing a ranking built under
    # different values of those gets silently wrong answers. Verify with a
    # bit-identical check against the recomputing path before trusting it.
    _aug_precomputed_ranking = kwargs.pop('aug_precomputed_ranking', None)
    # Filled by allocate_performance_weighted with the RAW per-representation
    # scores it computed, and surfaced on feature_info below. Recording it is
    # free and changes no stored result: feature_info is data, not a cache key.
    _alloc_score_sink = {}
    _aug_fraction = (float(_aug_fraction) if _aug_fraction is not None else None)

    # Convert labels to numpy array if needed
    labels = np.asarray(labels)

    # Input validation
    if not base_reps:
        raise ValueError("base_reps cannot be empty")

    # An objective name this file cannot score used to fall straight through
    # every geometry branch into the predictive one, so a typo — or the cut
    # 'similarity_sac' — ran a prediction search under the name that was asked
    # for and reported it as that objective.
    _check_objective_name(objective)
    
    # Check all reps have same number of samples and at least 1 feature
    n_samples_list = [X.shape[0] for X in base_reps.values()]
    if len(set(n_samples_list)) > 1:
        raise ValueError(f"All representations must have same number of samples. Got: {n_samples_list}")
    
    n_samples = n_samples_list[0]
    if len(labels) != n_samples:
        raise ValueError(f"labels length ({len(labels)}) must match number of samples ({n_samples})")
    
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}")
    
    # Check for reps with 0 features
    for rep_name, X in base_reps.items():
        if X.shape[1] == 0:
            raise ValueError(f"Representation '{rep_name}' has 0 features")
    
    # Validate augmentations if provided
    if augmentations is not None:
        for aug_name, X_aug in augmentations.items():
            if X_aug.shape[0] != n_samples:
                raise ValueError(f"Augmentation '{aug_name}' has {X_aug.shape[0]} samples, expected {n_samples}")
            if X_aug.shape[1] == 0:
                raise ValueError(f"Augmentation '{aug_name}' has 0 features")
    
    # Check for NaN/inf in labels
    if np.any(np.isnan(labels)):
        raise ValueError("labels contains NaN values")
    if np.any(np.isinf(labels)):
        raise ValueError("labels contains infinite values")
    
    # Validate numeric parameters
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if n_per_rep is not None and n_per_rep <= 0:
        raise ValueError(f"n_per_rep must be positive or None, got {n_per_rep}")
    if not (0 < validation_split < 1):
        raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be at least 1 or None, got {patience}")
    if augmentation_budget <= 0:
        raise ValueError(f"augmentation_budget must be positive, got {augmentation_budget}")

    # Resolve 'match' internal evaluator — maps downstream model type to evaluator
    # Callables pass through as-is (custom evaluator)
    if callable(internal_evaluator):
        pass  # Use as-is
    elif internal_evaluator == 'match':
        if model is not None:
            model_class = type(model).__name__.lower()
            if 'randomforest' in model_class or model_class == 'rf':
                internal_evaluator = 'rf'
            elif 'xgb' in model_class or 'xgboost' in model_class:
                internal_evaluator = 'xgboost'
            elif 'lgbm' in model_class or 'lightgbm' in model_class:
                internal_evaluator = 'lightgbm'
            elif 'dnn' in model_class:
                internal_evaluator = 'dnn'
            elif 'mlp' in model_class:
                internal_evaluator = 'mlp'
            elif 'sv' in model_class:
                internal_evaluator = 'svm'
            elif 'gaussianprocess' in model_class or model_class == 'gp':
                internal_evaluator = 'gp'
            elif 'kneighbors' in model_class or model_class == 'knn':
                internal_evaluator = 'knn'
            else:
                internal_evaluator = 'lightgbm'  # safe fallback
        else:
            internal_evaluator = 'lightgbm'  # no model provided, default

    # Apply quality filters if requested
    # We track the mapping from filtered indices back to original indices
    filter_index_maps = {}  # {rep_name: array mapping filtered_idx -> original_idx}
    original_base_reps = base_reps  # Keep reference to original data
    
    if apply_filters:
        try:
            from kirby.utils.feature_filtering import apply_quality_filters
            
            filtered_reps = {}
            for rep_name, X in base_reps.items():
                X_filtered, keep_mask = apply_quality_filters(
                    X, 
                    sparsity_threshold=kwargs.get('sparsity_threshold', 0.95),
                    variance_threshold=kwargs.get('variance_threshold', 0.01)
                )
                if X_filtered.shape[1] == 0:
                    raise ValueError(
                        f"All features filtered out for '{rep_name}'. "
                        f"Adjust sparsity_threshold or variance_threshold, or set apply_filters=False."
                    )
                filtered_reps[rep_name] = X_filtered
                filter_index_maps[rep_name] = np.where(keep_mask)[0]
            base_reps = filtered_reps
        except ImportError as e:
            raise ImportError(
                f"apply_filters=True requires kirby.utils.feature_filtering module: {e}"
            )
    
    # Common setup: create train/val split (group-aware when groups provided).
    # Lifted into internal_holdout_split so find_best_hybrid's precompute of the
    # per-representation budget scores makes the identical carve instead of its
    # own copy, which had drifted.
    train_idx, val_idx = internal_holdout_split(
        n_samples, labels, groups, validation_split)

    n_train = len(train_idx)
    n_val = len(val_idx)

    if n_train < 2:
        raise ValueError(f"Need at least 2 training samples. "
                       f"Got {n_train} with validation_split={validation_split} and {n_samples} total samples.")
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    # Scaffold groups for the internal held-out rows. Read-across needs them:
    # neighbours have to come from a different scaffold or the score is not
    # read-across. None when the caller passed no groups, which the blend path
    # rejects rather than scoring every representation identically.
    groups_val = (np.asarray(groups)[val_idx] if groups is not None else None)

    # Groups of the internal TRAINING rows. The candidate-feature ranking fits
    # out-of-fold residuals on these rows; without them a molecule's
    # scaffold-mates sit in the rows its own residual is predicted from.
    groups_train_internal = (np.asarray(groups)[train_idx]
                             if groups is not None else None)

    # Whole-dataset row numbers of those same held-out rows. MCS pairs store
    # row numbers into the whole dataset; this fold's rows are renumbered from
    # zero, so without this mapping a stored pair cannot be located and every
    # representation scores NaN.
    row_index_val = (np.asarray(row_index)[val_idx]
                     if row_index is not None else None)

    # A blend has to score every representation before it can weight any of
    # them, so only the allocators that do that can run one. Refuse the
    # rest loudly: falling back to a single component would report a number
    # nobody asked for, under the name of the blend.
    _blend_ok = {'performance_weighted', 'fixed', 'weighted_global'}
    if isinstance(objective, dict) or objective == OBJECTIVE_PREDICTIVE_CORR:
        _base_alloc = NAMED_ALLOCATIONS.get(allocation_method,
                                            (allocation_method, {}))[0]
        if _base_alloc not in _blend_ok:
            raise ValueError(
                f"allocation_method={allocation_method!r} resolves to "
                f"{_base_alloc!r}, which keeps or drops one representation at a "
                f"time against a running best. A blended objective combines "
                f"scores measured on all the representations, so it needs them "
                f"scored together. Use performance_weighted (or any of its "
                f"named variants) or weighted_global, or pass a single "
                f"objective name.")
    
    # Split base reps
    rep_dict_train = {name: X[train_idx] for name, X in base_reps.items()}
    rep_dict_val = {name: X[val_idx] for name, X in base_reps.items()}
    
    # Split augmentations if provided
    aug_dict_train = None
    aug_dict_val = None
    if augmentations is not None and augmentation_strategy != 'none':
        aug_dict_train = {name: X[train_idx] for name, X in augmentations.items()}
        aug_dict_val = {name: X[val_idx] for name, X in augmentations.items()}
    
    # ========================================================================
    # BASE HYBRID CREATION
    # ========================================================================

    # Resolve named allocation methods
    _original_allocation_name = allocation_method
    if allocation_method in NAMED_ALLOCATIONS:
        _base, _extra = NAMED_ALLOCATIONS[allocation_method]
        allocation_method = _base
        for k, v in _extra.items():
            kwargs.setdefault(k, v)

    # GREEDY ALLOCATION
    if allocation_method == 'greedy':
        # Run greedy allocation - now returns feature_info directly
        allocation, history, greedy_feature_info = allocate_greedy_forward(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, step_size=step_size, patience=patience,
            internal_evaluator=internal_evaluator,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            greedy_tolerance=kwargs.get('greedy_tolerance', 0.0),
            precomputed_importance=precomputed_importance,
            objective=objective, is_clf=is_clf,
        )
        
        # If filters were applied, remap indices from filtered space to original space
        if filter_index_maps:
            for rep_name in list(greedy_feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in greedy_feature_info[rep_name]:
                    filtered_indices = greedy_feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    greedy_feature_info[rep_name]['selected_indices'] = original_indices
        
        # Apply the SAME feature selection from greedy to full data
        # Use original_base_reps so indices (now in original space) match
        hybrid_features = apply_feature_selection(original_base_reps, greedy_feature_info)
        
        # Build feature_info with allocation metadata
        feature_info = greedy_feature_info.copy()
        feature_info['allocation'] = allocation
        feature_info['greedy_history'] = history
    
    # PERFORMANCE-WEIGHTED ALLOCATION
    elif allocation_method == 'performance_weighted':
        # Run performance-weighted allocation
        ws = kwargs.pop('weighting_scheme', 'linear')
        wp = kwargs.pop('weighting_params', None)
        allocation = allocate_performance_weighted(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, weighting_scheme=ws, weighting_params=wp,
            internal_evaluator=internal_evaluator,
            precomputed_scores=precomputed_alloc_scores,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            objective=objective, is_clf=is_clf, groups_val=groups_val,
            row_index_val=row_index_val, structural_pairs=structural_pairs,
            score_sink=_alloc_score_sink,
        )

        # Importance comes from the INTERNAL TRAINING part only, the same rows
        # allocate_weighted_global has always used. It used to be computed over
        # every row handed to create_hybrid, held-out part included -- and that
        # held-out part is what then decides whether a block of extra features
        # beats the base. The base was chosen with those rows' labels and the
        # extras were not, so "keep the base as it is" won more often than it
        # should have.
        if precomputed_importance is not None:
            importance_scores = precomputed_importance
            # If filters were applied, subset precomputed importance to match
            # filtered feature indices (precomputed on unfiltered data)
            if filter_index_maps:
                importance_scores = {
                    rn: scores[filter_index_maps[rn]]
                    if rn in filter_index_maps else scores
                    for rn, scores in importance_scores.items()
                }
        else:
            importance_scores = compute_feature_importance(
                rep_dict_train, train_labels, method=importance_method, **kwargs
            )
        
        # Select features based on train importance, then apply to full data
        _, feature_info = concatenate_features(
            rep_dict_train, importance_scores, allocation
        )
        
        # If filters were applied, remap indices from filtered space to original space
        if filter_index_maps:
            for rep_name in list(feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in feature_info[rep_name]:
                    filtered_indices = feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    feature_info[rep_name]['selected_indices'] = original_indices
        
        # Apply same feature selection to full data (use original if filters applied)
        hybrid_features = apply_feature_selection(original_base_reps, feature_info)

        # Store full importance scores for budget floor reuse
        for rep_name, imp_arr in importance_scores.items():
            if rep_name in feature_info and isinstance(feature_info[rep_name], dict):
                feature_info[rep_name]['importance_scores_full'] = imp_arr

        # Add allocation to feature_info for reference
        feature_info['allocation'] = allocation
        
    # FIXED ALLOCATION (BACKWARDS COMPATIBLE)
    elif allocation_method == 'fixed':
        # Importance from the internal training part only, for the same reason
        # as performance_weighted above.
        if precomputed_importance is not None:
            importance_scores = precomputed_importance
            if filter_index_maps:
                importance_scores = {
                    rn: scores[filter_index_maps[rn]]
                    if rn in filter_index_maps else scores
                    for rn, scores in importance_scores.items()
                }
        else:
            importance_scores = compute_feature_importance(
                rep_dict_train, train_labels, method=importance_method, **kwargs
            )
        
        # Equal split of the BUDGET across reps — the neutral reference point
        # for the allocation axis. n_per_rep is only honoured when a caller
        # explicitly asks for the pre-2026-08-26 budget-blind behaviour, which
        # spends n_per_rep * n_reps features regardless of `budget`.
        if n_per_rep is None:
            allocation = allocate_equal_split(
                {rn: X.shape[1] for rn, X in rep_dict_train.items()}, budget)
        else:
            allocation = {rep: n_per_rep for rep in base_reps.keys()}
        
        # Select features based on train importance
        _, feature_info = concatenate_features(
            rep_dict_train, importance_scores, allocation
        )
        
        # If filters were applied, remap indices from filtered space to original space
        if filter_index_maps:
            for rep_name in list(feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in feature_info[rep_name]:
                    filtered_indices = feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    feature_info[rep_name]['selected_indices'] = original_indices
        
        # Apply same feature selection to original data
        hybrid_features = apply_feature_selection(original_base_reps, feature_info)
        
        feature_info['allocation'] = allocation

    # DOMINANT-GREEDY ALLOCATION
    elif allocation_method == 'dominant_greedy':
        dominant_fraction = kwargs.pop('dominant_fraction', 0.7)
        allocation, history, dom_greedy_feature_info = allocate_dominant_greedy(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, step_size=step_size, patience=patience,
            dominant_fraction=dominant_fraction,
            internal_evaluator=internal_evaluator,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            greedy_tolerance=kwargs.get('greedy_tolerance', 0.0),
            precomputed_importance=precomputed_importance,
            objective=objective, is_clf=is_clf,
        )

        if filter_index_maps:
            for rep_name in list(dom_greedy_feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in dom_greedy_feature_info[rep_name]:
                    filtered_indices = dom_greedy_feature_info[rep_name]['selected_indices']
                    original_indices = filter_index_maps[rep_name][filtered_indices]
                    dom_greedy_feature_info[rep_name]['selected_indices'] = original_indices

        hybrid_features = apply_feature_selection(original_base_reps, dom_greedy_feature_info)

        feature_info = dom_greedy_feature_info.copy()
        feature_info['allocation'] = allocation
        feature_info['dominant_greedy_history'] = history

    # WEIGHTED GLOBAL-POOLING ALLOCATION (ADDITIVE — new branch; the per-rep
    # quota path above is untouched). Sizes each rep's slice of a shared
    # candidate pool by its performance weight, then runs ONE global importance
    # ranking over the pool to pick the budget. See allocate_weighted_global.
    elif allocation_method == 'weighted_global':
        ws = kwargs.pop('weighting_scheme', 'linear')
        wp = kwargs.pop('weighting_params', None)
        ptm = kwargs.pop('pool_total_mult', 2)

        # Subset precomputed importance to filtered space if filters applied
        # (precomputed on unfiltered data), mirroring the performance_weighted
        # branch. allocate_weighted_global also guards on width mismatch.
        _wg_precomp_imp = precomputed_importance
        if precomputed_importance is not None and filter_index_maps:
            _wg_precomp_imp = {
                rn: (scores[filter_index_maps[rn]]
                     if rn in filter_index_maps else scores)
                for rn, scores in precomputed_importance.items()
            }

        allocation, feature_info = allocate_weighted_global(
            rep_dict_train, rep_dict_val, train_labels, val_labels,
            total_budget=budget, weighting_scheme=ws, weighting_params=wp,
            pool_total_mult=ptm, importance_method=importance_method,
            internal_evaluator=internal_evaluator,
            precomputed_scores=precomputed_alloc_scores,
            precomputed_importance=_wg_precomp_imp,
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
            objective=objective, is_clf=is_clf, groups_val=groups_val,
            row_index_val=row_index_val, structural_pairs=structural_pairs,
        )

        # feature_info is in FILTERED space; remap indices to original space
        # (same remap loop as the performance_weighted branch).
        if filter_index_maps:
            for rep_name in list(feature_info.keys()):
                if rep_name in filter_index_maps and 'selected_indices' in feature_info[rep_name]:
                    filtered_indices = feature_info[rep_name]['selected_indices']
                    feature_info[rep_name]['selected_indices'] = \
                        filter_index_maps[rep_name][filtered_indices]

        hybrid_features = apply_feature_selection(original_base_reps, feature_info)
        feature_info['allocation'] = allocation

    else:
        raise ValueError(f"Unknown allocation_method: {allocation_method}. "
                        f"Choose from: 'greedy', 'performance_weighted', 'dominant_greedy', "
                        f"'fixed', 'weighted_global', or any named variant in NAMED_ALLOCATIONS "
                        f"(e.g. 'top_3', 'entropy_1.0', 'greedy_fill_s25').")

    # ========================================================================
    # BUDGET FLOOR: fill from best-performing reps (ranked by baseline score)
    # When an allocator (e.g. greedy) converges early, fill remaining budget
    # with top features from the best single reps. Walks reps in order of
    # baseline score, adds their top unselected features by importance.
    # With capped augmentation this means there is no spare budget left, so
    # every admitted extra has to displace a base column. That is intentional.
    # ========================================================================
    current_n = sum(
        info.get('n_features', len(info.get('selected_indices', [])))
        for info in feature_info.values()
        if isinstance(info, dict) and 'selected_indices' in info
    )
    _do_fill = kwargs.pop('budget_fill', False)
    # 'fixed' is excluded from the top-up ONLY in its budget-blind form
    # (n_per_rep given explicitly), where the total width is deliberately
    # n_per_rep * n_reps and has nothing to do with `budget`. In its default
    # equal-split form it is a point on the budget axis like every other
    # allocation, so it tops up the same way.
    _fill_blocked = (allocation_method == 'fixed' and n_per_rep is not None)
    if _do_fill and current_n < budget and not _fill_blocked:
        deficit = budget - current_n

        # Get baseline scores per rep for ranking
        _fill_baselines = kwargs.pop('precomputed_baselines', None)
        if _fill_baselines is None:
            _fill_baselines = {}
            for rep_name, X_tr in rep_dict_train.items():
                X_vl = rep_dict_val[rep_name]
                X_tr_c = np.nan_to_num(np.clip(X_tr, -1e10, 1e10), nan=0.0)
                X_vl_c = np.nan_to_num(np.clip(X_vl, -1e10, 1e10), nan=0.0)
                _fill_baselines[rep_name] = _evaluate_allocation(
                    X_tr_c, X_vl_c, train_labels, val_labels,
                    model=model, metric_fn=metric_fn,
                    lower_is_better=lower_is_better,
                    internal_evaluator=internal_evaluator,
                    objective=objective, is_clf=is_clf,
                    groups_val=groups_val,
                )

        # Walk reps best-first, fill with their top unselected features
        ranked_reps = sorted(_fill_baselines.keys(),
                             key=lambda r: _fill_baselines[r], reverse=True)

        filled = 0
        for rep_name in ranked_reps:
            if filled >= deficit:
                break
            X_rep = rep_dict_train[rep_name]
            info = feature_info.get(rep_name, {})
            sel_idx = info.get('selected_indices', np.array([], dtype=int))
            selected = set(sel_idx.tolist() if hasattr(sel_idx, 'tolist')
                          else list(sel_idx))
            n_total = X_rep.shape[1]
            if len(selected) >= n_total:
                continue

            # Get importance for this rep (cached or compute)
            imp = info.get('importance_scores_full', info.get('importance_scores'))
            if imp is None or len(imp) != n_total:
                X_clean = np.nan_to_num(np.clip(X_rep, -1e10, 1e10), nan=0.0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
                is_clf = _is_classification_task(train_labels)
                if is_clf:
                    rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                                random_state=42, n_jobs=-1)
                else:
                    rf = RandomForestRegressor(n_estimators=50, max_depth=10,
                                               random_state=42, n_jobs=-1)
                rf.fit(X_scaled, train_labels)
                imp = rf.feature_importances_
                imp = np.nan_to_num(imp, nan=0.0)
                del rf

            # Add top unselected features from this rep
            unselected = [(imp[i], i) for i in range(n_total) if i not in selected]
            unselected.sort(key=lambda x: -x[0])

            for _, feat_idx in unselected:
                if filled >= deficit:
                    break
                if 'selected_indices' not in feature_info.get(rep_name, {}):
                    feature_info[rep_name] = {
                        'selected_indices': np.array([feat_idx]),
                        'n_features': 1,
                    }
                else:
                    cur = feature_info[rep_name]['selected_indices']
                    cur = cur.tolist() if hasattr(cur, 'tolist') else list(cur)
                    cur.append(feat_idx)
                    feature_info[rep_name]['selected_indices'] = np.array(
                        sorted(set(cur)), dtype=int)
                    feature_info[rep_name]['n_features'] = len(
                        feature_info[rep_name]['selected_indices'])
                filled += 1

        # Rebuild hybrid_features
        hybrid_features = apply_feature_selection(original_base_reps, feature_info)

    # ========================================================================
    # AUTO-DEMOTE: zeroed-out base reps → augmentation pool (top_k only)
    # ========================================================================
    # When a top_k allocation (top_2, top_3, top_5) zeros out base reps,
    # their top features (ranked by importance, capped at augmentation_budget)
    # are added to the augmentation pool for A/B testing. This lets top_k
    # concentrate the main budget while still capturing opportunistic value
    # from borderline reps. Only triggers for top_k + active augmentation.
    _is_top_k = _original_allocation_name in ('top_2', 'top_3', 'top_5')
    _has_augmentations = (aug_dict_train is not None and len(aug_dict_train) > 0) or \
                         (augmentations is not None and len(augmentations) > 0)
    if _is_top_k and _has_augmentations and augmentation_strategy not in ('none', None):
        # Find base reps that got 0 allocated features
        _alloc = feature_info.get('allocation', {})
        demoted_reps = [name for name, count in _alloc.items() if count == 0]

        if demoted_reps:
            if aug_dict_train is None:
                aug_dict_train = {}
                aug_dict_val = {}

            for rep_name in demoted_reps:
                X_full = original_base_reps[rep_name]
                X_train_rep = X_full[train_idx]
                n_cols = X_train_rep.shape[1]

                # Rank features by importance — reuse precomputed if available
                if (precomputed_importance is not None and
                        rep_name in precomputed_importance):
                    imp_scores = precomputed_importance[rep_name]
                else:
                    X_clean = np.nan_to_num(
                        np.clip(X_train_rep, -1e10, 1e10), nan=0.0)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_clean)
                    is_clf = _is_classification_task(train_labels)
                    _rf = (RandomForestClassifier if is_clf else
                           RandomForestRegressor)(
                        n_estimators=50, max_depth=10,
                        random_state=42, n_jobs=-1)
                    _rf.fit(X_scaled, train_labels)
                    imp_scores = np.nan_to_num(_rf.feature_importances_, nan=0.0)
                    del _rf

                # Take top augmentation_budget features
                n_keep = min(augmentation_budget, n_cols)
                top_idx = np.argsort(imp_scores)[::-1][:n_keep]
                top_idx = np.sort(top_idx)  # preserve column order

                aug_dict_train[rep_name] = X_full[train_idx][:, top_idx]
                aug_dict_val[rep_name] = X_full[val_idx][:, top_idx]

                # Also add to full-data augmentations so apply_augmentation_selection
                # can reconstruct the final hybrid at the end.
                #
                # ⛔ THIS MUTATES THE CALLER'S DICT, DELIBERATELY. Do not "fix"
                # it by copying `augmentations` at the top of create_hybrid — it
                # was tried on 2026-09-10 and reverted within the hour.
                # _evaluate_hybrid_config READS the mutation: it rebuilds the
                # validation side from `train_augs`, the same object it passed
                # in, and finds a demoted representation by its full-data row
                # count (find_best_hybrid.py, the val_augs rebuild). Take a copy
                # here and that lookup falls through to the KeyError below, the
                # fold is lost, and since 2026-09-10 a lost fold refuses the
                # whole configuration.
                #
                # The LEAK — one model's demotion following the next model into
                # its rebuild — was closed on the other side instead:
                # apply_blueprint rebuilds against a copy. The block is also
                # gated on a top_k allocation, which left the default set on
                # 2026-09-01, so no default search reaches it at all.
                if augmentations is None:
                    augmentations = {}
                augmentations[rep_name] = X_full[:, top_idx]

                # Record which columns of the base rep were demoted. Callers
                # that hold a SEPARATE validation-side augmentation dict (the
                # per-fold rebuild helper does) cannot otherwise reconstruct
                # this rep's validation slice: the demotion happens inside this
                # function, so their val dict has no entry for it and
                # apply_augmentation_selection raises "found in
                # augmentation_info but not in augmentations". Storing the
                # indices lets them take the same columns from their own copy.
                feature_info.setdefault('demoted_reps', {})[rep_name] = top_idx

    # ========================================================================
    # AUGMENTATION HANDLING
    # ========================================================================

    if aug_dict_train is not None and augmentation_strategy != 'none':
        # Get base hybrid for train/val
        X_base_train = apply_feature_selection(
            {name: X[train_idx] for name, X in original_base_reps.items()},
            feature_info
        )
        X_base_val = apply_feature_selection(
            {name: X[val_idx] for name, X in original_base_reps.items()},
            feature_info
        )
        
        if augmentation_strategy == 'greedy_ablation':
            # Run greedy ablation
            aug_result = greedy_augmentation_ablation(
                X_base_train, X_base_val,
                aug_dict_train, aug_dict_val,
                train_labels, val_labels,
                augmentation_budget=augmentation_budget,
                threshold=augmentation_threshold,
                return_all_scores=True,
                model=model, metric_fn=metric_fn, lower_is_better=lower_is_better,
                objective=objective, is_clf=is_clf, groups_val=groups_val,
            )
            
            feature_info['augmentation_info'] = aug_result['augmentation_info']
            feature_info['augmentation_scores'] = aug_result['all_scores']
            # Every rung's score, so a caller sweeping the margin reads
            # them instead of re-fitting them. DATA on the recipe, not a
            # key setting: recording it changes no stored result.
            feature_info['augmentation_rung_scores'] = aug_result.get(
                'rung_scores')
            feature_info['augmentation_baseline'] = aug_result['baseline_score']
            feature_info['augmentation_final'] = aug_result['final_score']
            feature_info['kept_augmentations'] = aug_result['kept_augmentations']
            
            # Apply kept augmentations to full data
            if aug_result['kept_augmentations']:
                aug_features = apply_augmentation_selection(
                    augmentations, aug_result['augmentation_info']
                )
                if aug_features is not None:
                    hybrid_features = np.hstack([hybrid_features, aug_features])
        
        elif augmentation_strategy == 'capped':
            # Validated top-k block that displaces base features; total stays
            # within budget.
            aug_result = augment_capped(
                X_base_train, X_base_val,
                feature_info,
                aug_dict_train, aug_dict_val,
                train_labels, val_labels,
                model=model,
                metric_fn=metric_fn,
                lower_is_better=lower_is_better,
                total_budget=budget,
                candidate_ranking=candidate_ranking,
                patience=10,
                max_candidates=_aug_max_candidates,
                precomputed_ranking=_aug_precomputed_ranking,
                # Was never forwarded here, so a 2560-wide donor competed with
                # a 40-wide one on column count alone -- while the cap still
                # entered the stored result's identity, filing two runs that
                # behaved identically as different results.
                per_source_budget=_aug_per_source_budget,
                margin=_aug_margin,
                fraction=_aug_fraction,
                objective=objective, is_clf=is_clf, groups_val=groups_val,
                groups_train=groups_train_internal,
            )

            feature_info['augmentation_info'] = aug_result['augmentation_info']
            # Every rung's score, so a caller sweeping the margin reads
            # them instead of re-fitting them. DATA on the recipe, not a
            # key setting: recording it changes no stored result.
            feature_info['augmentation_rung_scores'] = aug_result.get(
                'rung_scores')
            feature_info['augmentation_baseline'] = aug_result['baseline_score']
            feature_info['augmentation_final'] = aug_result['final_score']
            feature_info['kept_augmentations'] = aug_result['kept_augmentations']
            feature_info['augmentation_replacements'] = aug_result['replacements']
            feature_info['phase_a_added'] = aug_result['phase_a_added']
            feature_info['phase_b_replaced'] = aug_result['phase_b_replaced']

            # Rebuild hybrid_features from (possibly modified) base_feature_info
            # + augmentation features.  Phase B may have removed base indices.
            # base_only: the extras go on four lines below, so the base block is
            # what this call is FOR. Without it every capped configuration of
            # every fold trips the warning saying the returned matrix is a
            # different feature set from the one that was fitted -- the loudest
            # line in a capped run, saying the opposite of what happened.
            hybrid_features = apply_feature_selection(
                original_base_reps, feature_info, base_only=True
            )
            if aug_result['kept_augmentations']:
                aug_features = apply_augmentation_selection(
                    augmentations, aug_result['augmentation_info']
                )
                if aug_features is not None:
                    hybrid_features = np.hstack([hybrid_features, aug_features])

        elif augmentation_strategy == 'uncapped':
            # Feature-level add-if-it-helps, no budget limit
            aug_result = augment_uncapped(
                X_base_train, X_base_val,
                aug_dict_train, aug_dict_val,
                train_labels, val_labels,
                model=model,
                metric_fn=metric_fn,
                lower_is_better=lower_is_better,
                candidate_ranking=candidate_ranking,
                patience=10,
                max_candidates=_aug_max_candidates,
                precomputed_ranking=_aug_precomputed_ranking,
                per_source_budget=_aug_per_source_budget,
                margin=_aug_margin,
                fraction=_aug_fraction,
                objective=objective, is_clf=is_clf, groups_val=groups_val,
                groups_train=groups_train_internal,
            )

            feature_info['augmentation_info'] = aug_result['augmentation_info']
            # Every rung's score, so a caller sweeping the margin reads
            # them instead of re-fitting them. DATA on the recipe, not a
            # key setting: recording it changes no stored result.
            feature_info['augmentation_rung_scores'] = aug_result.get(
                'rung_scores')
            feature_info['augmentation_baseline'] = aug_result['baseline_score']
            feature_info['augmentation_final'] = aug_result['final_score']
            feature_info['kept_augmentations'] = aug_result['kept_augmentations']
            feature_info['n_added'] = aug_result['n_added']

            # Append augmentation features to hybrid
            if aug_result['kept_augmentations']:
                aug_features = apply_augmentation_selection(
                    augmentations, aug_result['augmentation_info']
                )
                if aug_features is not None:
                    hybrid_features = np.hstack([hybrid_features, aug_features])

        elif augmentation_strategy == 'all':
            # No accept/reject test: take each donor's top augmentation_budget
            # features (default 20) and concatenate them all.
            #
            # The ranking is fit on the INNER-TRAINING rows only. Until
            # 2026-08-26 it was fit on `labels`, the whole label vector, which
            # includes the inner-validation rows this hybrid is then scored on.
            # Every other strategy ranks on train rows alone. The difference
            # matters because 'all' is the do-nothing arm every other strategy
            # is measured against, and on logd it BEAT real selection
            # (+0.0131 vs +0.0058) -- a comparison that is only meaningful if
            # this arm is as honest as the others.
            is_classification = _is_classification_task(train_labels)
            augmentation_info = {}
            aug_parts = []

            for aug_name, X_aug in augmentations.items():
                X_rank = (aug_dict_train.get(aug_name)
                          if aug_dict_train else None)
                if X_rank is None:
                    X_rank = X_aug[train_idx]
                _, indices, importances = _select_augmentation_features(
                    X_rank, train_labels, augmentation_budget, is_classification
                )
                augmentation_info[aug_name] = {
                    'selected_indices': indices,
                    'importance_scores': importances,
                    'n_features': len(indices)
                }
                X_sel = np.clip(X_aug[:, indices], -1e10, 1e10)
                aug_parts.append(np.nan_to_num(X_sel, nan=0.0))

            feature_info['augmentation_info'] = augmentation_info
            feature_info['kept_augmentations'] = list(augmentations.keys())

            if aug_parts:
                hybrid_features = np.hstack([hybrid_features] + aug_parts)

        else:
            # The chain above has always been if/elif with no final branch, so
            # an unrecognised name produced a base-only hybrid while the name
            # was still written into the stored recipe as though it had been
            # honoured. The allocation dispatch a few hundred lines above does
            # raise and list its valid names; this one now matches it.
            raise ValueError(
                f"Unknown augmentation_strategy: {augmentation_strategy!r}. "
                f"Choose from: 'none', 'greedy_ablation', 'capped', "
                f"'uncapped', 'all'.")
    
    # The raw per-representation scores the allocation computed, so a caller
    # sweeping settings that do not change them can hand them straight back
    # as precomputed_alloc_scores instead of paying ~240s to recompute them.
    # Empty unless a performance-weighted allocation actually ran.
    if _alloc_score_sink:
        feature_info['allocation_baseline_scores'] = dict(_alloc_score_sink)
    return hybrid_features, feature_info