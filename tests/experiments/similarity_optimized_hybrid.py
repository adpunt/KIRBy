#!/usr/bin/env python3
"""
Similarity-Optimized Hybrid — a hybrid found by optimizing SIMILARITY PRESERVATION
=================================================================================

KIRBy's hybrid search normally optimizes DOWNSTREAM PREDICTIVE PERFORMANCE. This
standalone experiment runs the SAME search machinery with a different objective —
``objective='similarity'`` (find_best_hybrid) — which selects the hybrid whose
GEOMETRY gives the best held-out ACTIVITY READ-ACROSS (scaffold-grouped kNN
Spearman). No downstream model is fit in that arm.

SAC (Spearman between pairwise rep-space distance and |delta property|) was CUT
as a SEARCH objective on 2026-08-26; see the comment block above
OBJECTIVE_PREDICTIVE in src/kirby/find_best_hybrid.py (lines 1215-1244). SAC is
still the quantity REPORTED in Payoff 1 below — reporting it and selecting by it
are different things. Asking this script for --sim-objective similarity_sac now
stops the run.

It answers two questions on small-molecule datasets, honestly (held-out, with
multi-seed spread — never the in-sample/definitional SAC of the older s02 path):

  Payoff 1 — SIMILARITY QUALITY: does the similarity-optimized hybrid preserve
             similarity on HELD-OUT folds better than the predictive-optimized
             hybrid and the best single rep? (held-out SAC + violation rate)

  Payoff 2 — NOISE MITIGATION: does its smoother geometry translate into better
             label-noise handling? Under three noise MODELS (gaussian / hetero /
             outlier) it sweeps EVERY MITIGATION method in
             kirby.utils.noise_mitigation_methods, grouped by similarity-reliance
             (SIMILARITY: knn_k5/k10/k20, distance_weighted, neighborhood_consensus,
             activity_cliffs; GEOMETRY: lof, mahalanobis; CONTROL: zscore,
             prediction_error). Per method x space it reports DETECTION vs the
             known mask (precision/recall/F1) and RECOVERY two ways — 'remove'
             (drop flagged) and 'smooth' (kNN-correct flagged labels) — plus a
             detector-independent GLOBAL kNN label-smoothing baseline.

  Payoff 2b — NOISE ESTIMATION: the ESTIMATOR_REGISTRY continuous per-sample
             noise scorers (SIMILARITY: neighbor_residual, local_variance_ratio,
             distance_weighted_residual, activity_cliff_aware; CONTROL:
             cross_val_residual, bayesian_rf), scored by AUROC + precision@k vs
             the mask and a sample-REWEIGHTING recovery.

  Payoff 3 — IMPORTANCE-WEIGHTED DISTANCES: feeds distance matrices built with
             feature_weights = f(importance) — none / sqrt / squared / top-k —
             into the distance-aware methods, testing whether emphasising the
             informative features sharpens similarity-based noise handling.

For Payoff 2/2b/3 the sim-hybrid should lift the SIMILARITY group specifically;
CONTROL is the null contrast. All gated to regression (the denoisers/estimators
are regression-designed — clf sets record a 'skipped' marker, still do Payoff 1).

The two hybrids are found by SEPARATE searches (different objective) — this is
NOT re-scoring the predictive hybrid; the similarity search discovers a
different rep composition (see the divergence check in the engine tests).

Datasets (sound, Murcko-scaffold split, precompute-built per DATASET_STATUS
2026-07-07): caco2, logd, herg (clf), herg_ki, qm9, adme_hlm, adme_rlm,
adme_mdr1. Avoid chembl_qsar (never run) and herg's stale graph_kernel rep.

Search space is kept small on purpose (budgets < 300, aggressive pruning) per
the project constraint.

Run:
    python tests/experiments/similarity_optimized_hybrid.py --datasets logd herg \
        --seeds 3 --output tests/results/sim_optimized_hybrid
"""
import argparse
import json
import os
import sys
import zlib
import numpy as np

# --- Path bootstrap so `experiments...` and `kirby...` import cleanly ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
for _p in (_TESTS, os.path.join(_TESTS, 'experiments')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.find_best_hybrid import (
    find_best_hybrid, OBJECTIVE_SIMILARITY_SAC, OBJECTIVE_PREDICTIVE,
    OBJECTIVE_SIMILARITY, _is_geometry_objective,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.utils.noise_mitigation_methods import (
    get_mitigation_method,
    # float-returning SAC (same one the engine optimizes) — the similarity_helpers
    # twin returns a 5-tuple, so use this to avoid confusion.
    similarity_activity_correlation_scalable,
)

from experiments._shared import load_data, cv_split
from experiments.dataset_policy import noise_groups_for
from experiments.similarity.similarity_helpers import violation_rate_scalable

# Small-molecule datasets that are methodologically sound (DATASET_STATUS §5/§9).
# Ordered by ascending n (roughly ascending runtime — searches scale with n), so
# the small sets finish first and the big ones (herg ~7.7k, qm9 ~10k) run last.
DEFAULT_DATASETS = ['caco2', 'adme_hlm', 'adme_rlm', 'adme_mdr1', 'herg_ki',
                    'logd', 'herg', 'qm9']

# Budgets kept < 300 (project constraint); aggressive pruning below.
SIM_BUDGETS = [int(v) for v in
               os.environ.get('KIRBY_SIMOPT_BUDGETS', '').replace(' ', '').split(',')
               if v] or [10, 25, 50, 100, 200]
# The five allocation methods the search sweeps, as set on 2026-09-01 from the
# c32 cross sweep (find_best_hybrid.DEFAULT_ALLOCATION_METHODS). All five route
# through _evaluate_allocation, so all five see the objective.
#
# Was ['greedy', 'top_2', 'top_3', 'performance_weighted', 'dominant_greedy'].
# greedy and dominant_greedy are cut (hybrid.py, "THE GREEDY FAMILY IS CUT").
# top_2/top_3 weld a representation count into the weighting slot; the number of
# representations is a separate axis and is not encoded here.
# This file carries its own search space and does NOT read the local grid, so
# the KIRBY_LOCAL_* cut that shrinks every other similarity run never reached it:
# measured 2026-09-24, simopt on caco2 took 749 minutes at 5 budgets x 5
# allocation methods. These two env vars cut it; unset reproduces every stored
# number.
SIM_ALLOCATIONS = ([v for v in
                    os.environ.get('KIRBY_SIMOPT_ALLOCATIONS', '').replace(' ', '').split(',')
                    if v] or
                   ['performance_weighted', 'power_2', 'softmax_T1',
                    'rank_exp_d0.3', 'wglobal_power2'])
ADAPTIVE_PRUNING = 0.35   # aggressive: prune strategies >35% below the running best
N_FOLDS = 3
# Payoff 2 varies the noise CONDITION at a single moderate level. Every condition
# below delivers the SAME amount of noise, so a difference between them is a
# difference of shape rather than of amount.
#
# `hetero` -- error scaling with |y| -- was here and is GONE. Its premise, that
# measurement error depends on the value being measured, was tested directly on 16,844
# repeat measurements and refuted: "The dpIC50's depend neither on the average measured
# pIC50 nor on any of the ligand properties examined" (Kalliokoski et al. 2013,
# PLoS ONE 8(4):e61007). It was also the condition this experiment called "the
# realistic noise model".
#
# What replaces it: gaussian as the control, outlier_p10 for sparse contamination, and
# grouped_wider for structure -- whole scaffold families carrying a wider error, which
# is the best-evidenced of the set (within-laboratory error must be multiplied by about
# three to reach between-laboratory error, Avdeef 2019) and the one this experiment
# actually needs, because it is the only zero-mean condition where the noise is
# predictable from structure.
NOISE_TYPES = ['gaussian', 'outlier_p10', 'grouped_wider']

# The contaminated fraction, held at one value across every condition so the detection
# numbers stay comparable: a random detector scores precision = NOISE_FRAC, and that is
# the null the DENOISE_METHODS are judged against. It OVERRIDES the p that
# `outlier_p10` carries in the registry and the group_fraction that `grouped_wider`
# carries, so the condition name here says which SHAPE and targeting rule is in use,
# not which fraction.
NOISE_FRAC = 0.25
# How much noise, as a fraction of the label SD. This experiment had no level
# axis: the magnitude was hardcoded at 0.3 of the label RANGE, which is roughly
# 4-6x the SD, so it was running far hotter than the number suggested.
NOISE_LEVEL = 0.5
KNN_SMOOTH_K = 10   # neighbours for label smoothing / correction
# The noise section does many O(n^2) distance ops per method x space x fold, so it
# subsamples each fold's train/val to a cap. The comparison is relative (which
# space/method wins), so a ~1k sample is ample and keeps the whole sweep tractable.
NOISE_MAX_N_TRAIN = 1000
NOISE_MAX_N_VAL = 500

# Noise-mitigation methods grouped by whether they rely on the SIMILARITY
# assumption (rep-space distance => nearby samples share labels). A geometry
# tuned for that assumption (the sim-hybrid) should help the 'similarity' group
# specifically; the 'geometry' group uses feature-space distance but for
# density-outlier detection (not the label-similarity assumption); the 'control'
# group ignores rep-space geometry entirely and is the null contrast.
SIMILARITY_DENOISERS = ['knn_k5', 'knn_k10', 'knn_k20', 'distance_weighted',
                        'neighborhood_consensus', 'activity_cliffs']
GEOMETRY_DENOISERS = ['lof', 'mahalanobis']
# zscore = label-only (purest null); prediction_error = model residual. Both
# ignore rep-space geometry. (cv_disagreement is a valid control too but runs an
# internal 5-fold CV — too costly for the sweep.)
CONTROL_DENOISERS = ['zscore', 'prediction_error']
DENOISE_METHODS = SIMILARITY_DENOISERS + GEOMETRY_DENOISERS + CONTROL_DENOISERS
METHOD_CLASS = {**{m: 'similarity' for m in SIMILARITY_DENOISERS},
                **{m: 'geometry' for m in GEOMETRY_DENOISERS},
                **{m: 'control' for m in CONTROL_DENOISERS}}

# Noise ESTIMATION methods (kirby.utils.noise_estimation.ESTIMATOR_REGISTRY):
# continuous per-sample noise scores (higher = noisier), scored by AUROC vs the
# known mask. Neighbour-based ones rely on the similarity assumption; model-based
# ones are the control. Neural/O(n^3) estimators (deep_ensemble, heteroscedastic_nn,
# gaussian_process, ngboost) are excluded for cost/deps.
SIMILARITY_ESTIMATORS = ['neighbor_residual', 'local_variance_ratio',
                         'distance_weighted_residual', 'activity_cliff_aware']
# The affordable non-neighbour estimators are all far too slow at scale
# (bayesian_rf ~27s/fit, cross_val_residual runs an internal 5-fold CV,
# leave_one_out is O(n) refits). So the estimation sweep is the 4 fast
# neighbour-based (similarity) estimators; the non-similarity null contrast is
# carried by the MITIGATION controls (zscore / prediction_error).
CONTROL_ESTIMATORS = []
ESTIMATION_METHODS = SIMILARITY_ESTIMATORS + CONTROL_ESTIMATORS
ESTIMATOR_CLASS = {**{m: 'similarity' for m in SIMILARITY_ESTIMATORS},
                   **{m: 'control' for m in CONTROL_ESTIMATORS}}

# Importance-weighted-distance experiment (Payoff 3): feed distance matrices
# built with feature_weights = f(importance) into the distance-aware methods, in
# several 'fashions'. Tests whether emphasising the informative features
# sharpens similarity-based noise handling — and whether it stacks with the
# sim-hybrid geometry. Run on a subset of representative similarity methods and a
# single (realistic) noise model to bound cost.
WEIGHT_TEST_METHODS = ['knn_k5', 'distance_weighted', 'neighborhood_consensus']
WEIGHT_FASHIONS = ['none', 'sqrt_importance', 'importance', 'top_k']
WEIGHT_TOPK = 50                 # 'top_k' fashion: keep the top-K features by importance
# The weighting sweep asks WHERE the noise is, so it needs a condition whose
# per-molecule noise scale actually varies. For the five shape-only conditions
# (gaussian, the three Student-t, laplace) every molecule gets the same scale, so
# that question is UNDEFINED there rather than answered with zero, and a rank
# correlation against a constant must never be reported. grouped_wider, the three
# outlier conditions and censoring are the ones that can answer it.
WEIGHT_NOISE_TYPE = 'grouped_wider'
WEIGHT_SPACES = ('sim', 'pred')  # weighting sweep runs on the low-dim hybrid spaces only

# The raw best-single-rep space can be very wide (e.g. ecfp4 = 2048-d), which
# makes the O(n^2 . d) distance/kNN/estimator ops in the noise section blow up.
# Cap it to its top-variance features for Payoff 2/2b (Payoff 1 keeps full dim —
# its SAC is pair-sampled and cheap). Hybrids are already < 300-d, so unaffected.
NOISE_REP_DIM_CAP = 256

# --- Which objective the geometry ("sim") arm may be run under -------------
# find_best_hybrid._is_geometry_objective returns True only for 'similarity'
# since SAC was cut as a search objective on 2026-08-26 (evidence on both
# sides: the comment block above OBJECTIVE_PREDICTIVE in
# src/kirby/find_best_hybrid.py, lines 1215-1244). Every objective branch in
# this file therefore has to REJECT anything outside this tuple: an unlisted
# name used to fall through the geometry test into the predictive path, so the
# script ran an RMSE-optimised search on both arms and labelled one of them the
# similarity hybrid.
SIM_OBJECTIVE_CHOICES = (OBJECTIVE_SIMILARITY,)
# The objective rides in the cache tier so two objectives never share a result
# row (reps are never in the cache key).
SIM_OBJECTIVE_TIER = {OBJECTIVE_SIMILARITY: 'general_sim_readacross'}


class SimObjectiveError(ValueError):
    """The sim arm was asked for an objective that selects nothing.

    Raised before any search runs. main() re-raises it instead of recording a
    per-dataset error entry, so a bad --sim-objective stops the run."""


def _check_sim_objective(objective):
    """Return `objective` if the sim arm accepts it; raise SimObjectiveError."""
    if objective in SIM_OBJECTIVE_CHOICES:
        return objective
    if objective == OBJECTIVE_SIMILARITY_SAC:
        raise SimObjectiveError(
            f"--sim-objective {OBJECTIVE_SIMILARITY_SAC!r} was CUT as a search "
            "objective on 2026-08-26 and nothing selects by it. Read the "
            "comment block above OBJECTIVE_PREDICTIVE in "
            "src/kirby/find_best_hybrid.py, lines 1215-1244, for the numbers "
            "on both sides of that decision. Use "
            f"--sim-objective {OBJECTIVE_SIMILARITY} (activity read-across). "
            "SAC is still reported in Payoff 1 either way."
        )
    raise SimObjectiveError(
        f"--sim-objective {objective!r} is not an objective this script can "
        f"run. Accepted: {list(SIM_OBJECTIVE_CHOICES)}."
    )


def _cli_sim_objective(value):
    """argparse `type=` wrapper so the rejection message reaches the user."""
    try:
        return _check_sim_objective(value)
    except SimObjectiveError as e:
        raise argparse.ArgumentTypeError(str(e))


def _model(is_clf):
    if is_clf:
        return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)


def _score(model, X_tr, y_tr, X_va, y_va, is_clf):
    from sklearn.metrics import f1_score, r2_score
    m = _model(is_clf)
    m.fit(X_tr, y_tr)
    p = m.predict(X_va)
    return f1_score(y_va, p, average='weighted') if is_clf else r2_score(y_va, p)


def _standardize_pair(X_tr, X_va):
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_va)


def _search(reps, labels, groups, is_clf, objective, tier, seed,
            dataset_name=None, cache_dir=None):
    """Run one objective-specific hybrid search; return (best Blueprint, baselines).

    Caching is ALWAYS on when a dataset_name is given (which it always is):
    reuses per-rep baseline caches, per-fold importance caches, and any matching
    result rows already present, and caches this run for cheap reruns. cache_dir
    is resolved by the caller to a concrete path (falls back to $KIRBY_CACHE_DIR
    or ~/.cache/kirby) so reuse of the existing precompute caches is guaranteed.
    """
    task_type = 'classification' if is_clf else 'regression'
    use_cache = dataset_name is not None
    kw = dict(
        base_reps=reps, labels=labels, groups=groups, task_type=task_type,
        models=['rf'], internal_evaluators=['match'],
        allocation_methods=SIM_ALLOCATIONS, budgets=SIM_BUDGETS,
        importance_methods=['lightgbm_gain'],
        augmentation_strategy='none', adaptive_pruning=ADAPTIVE_PRUNING,
        seed=seed, verbose=False, return_fitted=False,
        # The search's own fold count. Left unset it fell to
        # find_best_hybrid.DEFAULT_N_FOLDS (5 on the cluster), while Payoff 1
        # and 2 score on N_FOLDS=3 folds and the result says they are the same
        # folds the search selected on. cv_splitter_id below already claimed 3.
        n_folds=N_FOLDS,
        cache=use_cache, cache_dir=cache_dir, dataset_name=dataset_name,
        cv_splitter_id=(f'k{N_FOLDS}s{seed}' if use_cache else None),
    )
    if _is_geometry_objective(objective):
        # Geometry arm; find_best_hybrid forces the matching geometry metric
        # ('readacross') itself and fits no downstream model.
        res = find_best_hybrid(objective=objective, tier=tier, **kw)
    elif objective == OBJECTIVE_PREDICTIVE:
        res = find_best_hybrid(objective=objective,
                               metric=('rmse' if not is_clf else 'f1_weighted'),
                               tier=tier, **kw)
    else:
        # No silent fall-through: an objective that is neither geometry nor
        # predictive used to land in the predictive branch, which then ran an
        # RMSE search and returned it as the geometry arm's blueprint.
        _check_sim_objective(objective)
        raise SimObjectiveError(
            f"_search: objective={objective!r} is neither a geometry objective "
            f"nor {OBJECTIVE_PREDICTIVE!r}."
        )
    by_budget = res.best_by_model_budget.get('rf', {})
    # score_mean holds the RAW metric value (rmse is lower-is-better; sac and
    # f1_weighted are higher-is-better). A bare max() therefore selected the
    # WORST hybrid whenever the metric was rmse — which is why the predictive
    # comparator always landed on the smallest budget. Pick direction-aware.
    _pick = min if res.lower_is_better else max
    best = _pick(by_budget.values(),
                 key=lambda r: r['score_mean']) if by_budget else None
    return (best['blueprint'] if best else None), res.baselines


def _predictive_blueprint(reps, labels, groups, is_clf, seed, dataset_name,
                          data, cache_dir):
    """Predictive-optimized comparator blueprint.

    Deliberately runs the SAME controlled search as the similarity side (rf
    model, same small budget grid, tier 'general') and differs ONLY in the
    objective — an apples-to-apples A/B. This is both fair and safe.

    Note: we do NOT pull from the existing full precompute pool
    (get_best_search_for_downstream). On a cache miss that helper launches a
    full DEFAULT-model search (svm/mlp/…) with FULL_BUDGETS, which is heavy and
    was observed to segfault on large molecular sets — and it would also compare
    a blueprint tuned over a different model/budget space, muddying the A/B.
    Cache reuse still happens where it is safe: per-rep baseline caches, per-fold
    importance caches, and this run's own SQLite result store (tier 'general').
    """
    bp, _ = _search(reps, labels, groups, is_clf, OBJECTIVE_PREDICTIVE,
                    'general', seed, dataset_name=dataset_name, cache_dir=cache_dir)
    return bp, 'search'


def _build_hybrid_fold(reps_tr, reps_va, y_tr, bp, objective, is_clf,
                       groups_tr=None):
    """Rebuild a blueprint's hybrid on the TRAIN fold, apply selection to VAL.
    Returns (X_tr, X_va) already standardized (scaler fit on train).

    groups_tr are the scaffold groups of the TRAIN-fold rows. create_hybrid
    slices its own internal held-out rows out of them, and the read-across
    objective is scored on those rows, so the geometry arm requires them. They
    are passed on BOTH arms, so create_hybrid's internal split is GroupKFold in
    both — the same split the search itself ran under, and the A/B still
    differs only in the objective."""
    if _is_geometry_objective(objective):
        # No model and no metric_fn: the allocation is scored on held-out
        # geometry, not on predictions.
        model, metric_fn = None, None
        if groups_tr is None:
            raise SimObjectiveError(
                f"_build_hybrid_fold: objective={objective!r} scores held-out "
                "read-across and needs groups_tr, the scaffold groups of the "
                "train-fold rows."
            )
    elif objective == OBJECTIVE_PREDICTIVE:
        # The SAME metric function object the search used (_search passes
        # metric='rmse' / 'f1_weighted'). create_hybrid recognises it by
        # identity (hybrid.metric_name_for_fn) and applies the origin rule to
        # the per-representation scores. A lambda was not recognised, so the
        # rebuild split the budget by a min-shift instead, which gives the
        # worst representation zero features: not the recipe the search scored.
        from kirby.find_best_hybrid import BUILTIN_METRICS
        model = _model(is_clf)
        metric_fn, metric_lower = BUILTIN_METRICS[
            'f1_weighted' if is_clf else 'rmse']
    else:
        # Previously an unlisted objective got a model and NO metric_fn here,
        # so every candidate allocation scored -inf and the rebuild returned a
        # hybrid the objective had no say in.
        _check_sim_objective(objective)
        raise SimObjectiveError(
            f"_build_hybrid_fold: objective={objective!r} is neither a geometry "
            f"objective nor {OBJECTIVE_PREDICTIVE!r}."
        )
    if model is None:
        metric_lower = False
    y_tr = np.asarray(y_tr)
    # create_hybrid's own internal train/held-out carve of these rows, from
    # the one function that defines it (the search uses the same one).
    from kirby.hybrid import internal_holdout_split, _score_rep_for_budget
    _itr, _iva = internal_holdout_split(len(y_tr), y_tr, groups_tr, 0.2)

    # The recipe's representation count. The search restricts each fold to the
    # rep_count representations its per-representation scores rank highest
    # (find_best_hybrid, "Top-N representation restriction"). This rebuild used
    # to hand create_hybrid the whole pool, so on a pool larger than rep_count
    # it built a different hybrid from the one the search scored. The ranking
    # is recomputed here on THIS fold's training rows, with the same scorer
    # the search uses, so no held-out row decides which representations stay.
    _rep_count = getattr(bp, 'rep_count', None)
    _alloc_scores = None
    if _rep_count is not None and int(_rep_count) < len(reps_tr):
        _gva = None if groups_tr is None else np.asarray(groups_tr)[_iva]
        _scores = {}
        for _rn, _X in reps_tr.items():
            try:
                _scores[_rn] = float(_score_rep_for_budget(
                    _X[_itr], _X[_iva], y_tr[_itr], y_tr[_iva],
                    internal_evaluator=bp.internal_evaluator,
                    metric_fn=metric_fn, lower_is_better=metric_lower,
                    objective=objective, is_clf=is_clf, groups_val=_gva))
            except Exception:
                _scores[_rn] = float('nan')
        _ranked = sorted(reps_tr, key=lambda r: (
            -(_scores[r] if np.isfinite(_scores[r]) else -np.inf), r))
        _keep = set(_ranked[:int(_rep_count)])
        reps_tr = {k: v for k, v in reps_tr.items() if k in _keep}
        reps_va = {k: v for k, v in reps_va.items() if k in _keep}
        _alloc_scores = {k: _scores[k] for k in _keep}

    # Within-rep feature selection stays importance-based; the ALLOCATION across
    # reps is what the objective drives. Importance is computed on the internal
    # TRAINING part only, as the search's importance pre-pass does. It used to
    # be computed on every row of the fold, internal held-out part included.
    imp = compute_feature_importance({k: v[_itr] for k, v in reps_tr.items()},
                                     y_tr[_itr], method=bp.importance_method)
    X_tr, fi = create_hybrid(
        reps_tr, y_tr, allocation_method=bp.allocation_method, budget=bp.budget,
        step_size=bp.step_size, importance_method=bp.importance_method,
        internal_evaluator=bp.internal_evaluator, augmentation_strategy='none',
        model=model, metric_fn=metric_fn, lower_is_better=metric_lower,
        objective=objective, is_clf=is_clf, precomputed_importance=imp,
        precomputed_alloc_scores=_alloc_scores,
        groups=groups_tr,
    )
    X_va = apply_feature_selection(reps_va, fi)
    return _standardize_pair(X_tr, X_va)


def _heldout_similarity(reps, labels, groups, is_clf, bp_sim, bp_pred, best_rep, seed,
                        sim_objective=OBJECTIVE_SIMILARITY):
    """Payoff 1: per-fold held-out SAC + violation rate for sim-hybrid,
    predictive-hybrid, and best single rep. Returns per-space fold lists.

    SAC is the quantity REPORTED here; sim_objective is the quantity the sim
    hybrid was SELECTED by. They are not the same quantity."""
    _check_sim_objective(sim_objective)
    out = {k: {'sac': [], 'violation': []} for k in ('sim', 'pred', 'rep')}
    _groups = None if groups is None else np.asarray(groups)
    for tr, va in cv_split(groups, 3, next(iter(reps.values())), labels, seed=seed):
        reps_tr = {k: v[tr] for k, v in reps.items()}
        reps_va = {k: v[va] for k, v in reps.items()}
        y_va = labels[va]
        groups_tr = None if _groups is None else _groups[tr]
        spaces = {}
        if bp_sim is not None:
            _, Xs = _build_hybrid_fold(reps_tr, reps_va, labels[tr], bp_sim,
                                       sim_objective, is_clf, groups_tr=groups_tr)
            spaces['sim'] = Xs
        if bp_pred is not None:
            _, Xp = _build_hybrid_fold(reps_tr, reps_va, labels[tr], bp_pred,
                                       OBJECTIVE_PREDICTIVE, is_clf,
                                       groups_tr=groups_tr)
            spaces['pred'] = Xp
        # best single rep: standardize on train
        _, Xr = _standardize_pair(reps_tr[best_rep], reps_va[best_rep])
        spaces['rep'] = Xr
        for name, Xv in spaces.items():
            out[name]['sac'].append(
                similarity_activity_correlation_scalable(Xv, y_va, is_clf=is_clf))
            out[name]['violation'].append(
                violation_rate_scalable(Xv, y_va, is_clf=is_clf))
    return out


def _detection_scores(flagged_idx, true_mask):
    """Precision/recall/F1 of the flagged-noisy set vs the known injected mask."""
    pred = np.zeros(len(true_mask), dtype=bool)
    if len(flagged_idx):
        pred[np.asarray(flagged_idx, dtype=int)] = True
    tp = int(np.sum(pred & true_mask))
    fp = int(np.sum(pred & ~true_mask))
    fn = int(np.sum(~pred & true_mask))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _inject_typed(y, frac, noise_type, seed, is_clf, groups=None, level=None):
    """Label noise for one condition, through noiseInject.

    `noise_type` is a noiseInject condition name. `frac` is the contaminated fraction
    for the outlier conditions and the affected-molecule fraction for grouped_wider;
    it is ignored by the shape-only conditions, which perturb every molecule.
    `level` is the noise DELIVERED, as a fraction of the label SD.

    Returns (y_noisy, mask), where mask marks the badly-wrong rows -- the ones whose
    noise scale is above the base. For a shape-only condition every row is perturbed
    equally, so the mask is all-True and "which rows are corrupted" is undefined
    rather than empty.

    Classification uses random label flips; a flipped class label has no magnitude,
    so dose matching does not apply there.
    """
    import numpy as _np
    from kirby.noise_spec import (GROUPED_STRATEGIES, NoiseSpec, make_noisy_labels,
                                  make_noisy_labels_verbose)

    if level is None:
        level = NOISE_LEVEL
    if is_clf:
        spec = NoiseSpec(strategy='uniform', target_effective_noise=float(frac),
                         noise_seed=int(seed))
        y_noisy = make_noisy_labels(y, spec, is_clf=True)
        return y_noisy, _np.asarray(y_noisy) != _np.asarray(y)

    params = ()
    if noise_type.startswith('outlier'):
        params = (('p', float(frac)),)
    elif noise_type in GROUPED_STRATEGIES:
        params = (('group_fraction', float(frac)),)
    spec = NoiseSpec(strategy=noise_type, target_effective_noise=float(level),
                     noise_seed=int(seed), params=params)
    # Passed for every condition: ignored when drawing by the ones that do not
    # target groups, but it records the assignment on their rows.
    res = make_noisy_labels_verbose(y, spec, groups=groups)
    scale = _np.asarray(res.noise_scale, dtype=float)
    if scale.max() - scale.min() <= 1e-12:
        mask = _np.asarray(res.epsilon) != 0.0
    else:
        mask = scale > (scale.min() + 1e-12)
    return _np.asarray(res.y_noisy), mask

def _knn_smoothed_labels(X, y, is_clf, k=KNN_SMOOTH_K):
    """Distance-weighted kNN label smoothing IN THE GIVEN SPACE — the canonical
    'similarity for noise mitigation' operator. Each label -> distance-weighted
    mean (regression) / weighted majority (classification) of its k neighbours.
    A better geometry (higher SAC) should yield better-corrected labels."""
    from sklearn.neighbors import NearestNeighbors
    n = len(y)
    k = min(k, n - 1)
    if k < 1:
        return np.asarray(y, dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]      # drop self
    w = 1.0 / (dist + 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    if is_clf:
        out = np.empty(n, dtype=np.asarray(y).dtype)
        for i in range(n):
            labs = y[idx[i]]
            classes = np.unique(labs)
            out[i] = max(classes, key=lambda c: w[i][labs == c].sum())
        return out
    return (w * y[idx]).sum(axis=1)


def _feature_importance(X, y, is_clf):
    """Per-column importance (RF impurity) of a built feature matrix, w.r.t. y.
    Used to weight distances in the importance-weighting experiment."""
    m = (RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
         if is_clf else
         RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1))
    m.fit(X, y)
    imp = np.asarray(m.feature_importances_, dtype=float)
    return np.maximum(imp, 0.0)


def _score_weighted(X_tr, y_tr, X_va, y_va, is_clf, sample_weight):
    """Downstream score when training rows are SAMPLE-WEIGHTED (soft denoising)."""
    from sklearn.metrics import f1_score, r2_score
    m = _model(is_clf)
    m.fit(X_tr, y_tr, sample_weight=sample_weight)
    p = m.predict(X_va)
    return f1_score(y_va, p, average='weighted') if is_clf else r2_score(y_va, p)


def _weighted_distance_matrix(X, importance, fashion):
    """Distance matrix under a feature-weighting 'fashion' (importance-based)."""
    from kirby.utils.noise_mitigation_methods import compute_distance_matrix
    if fashion == 'none':
        return compute_distance_matrix(X, standardize=True)
    if fashion == 'sqrt_importance':
        return compute_distance_matrix(X, feature_weights=importance, standardize=True)
    if fashion == 'importance':
        return compute_distance_matrix(X, feature_weights=importance ** 2, standardize=True)
    if fashion == 'top_k':
        k = min(WEIGHT_TOPK, X.shape[1])
        top = np.argsort(importance)[-k:]
        w = np.zeros(X.shape[1]); w[top] = 1.0
        return compute_distance_matrix(X, feature_weights=w, standardize=True)
    raise ValueError(f"unknown fashion {fashion}")


def _estimation_eval(X_tr, y_noisy, X_va, y_va, mask, is_clf, noisy):
    """For each noise ESTIMATION method: AUROC + precision@k of the continuous
    noise score vs the known mask, and a REWEIGHTING delta (down-weight
    high-noise rows via sample_weight). Returns {estimator: dict}.

    delta_reweight is the RAW paired score change (score_after - score_noisy),
    NOT a fraction of damage repaired: the old ratio divided by (clean - noisy),
    which is near-zero on folds where the noise barely hurt and NEGATIVE on folds
    where it accidentally helped, so the per-fold ratio exploded and flipped sign.
    Normalise once at the aggregate level if a fraction is wanted."""
    from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY
    from sklearn.metrics import roc_auc_score
    k = int(mask.sum())
    res = {}
    for name in ESTIMATION_METHODS:
        d = {'auroc': np.nan, 'precision_at_k': np.nan, 'delta_reweight': np.nan}
        try:
            cls = ESTIMATOR_REGISTRY[name]
            try:
                est = cls(is_clf=is_clf)
            except TypeError:
                est = cls()
            est.fit(X_tr, y_noisy)
            scores = np.asarray(est.noise_scores_, dtype=float)
            if np.ptp(scores) > 0 and mask.any() and not mask.all():
                d['auroc'] = float(roc_auc_score(mask, scores))
                topk = np.argsort(scores)[-k:] if k else np.array([], int)
                d['precision_at_k'] = float(mask[topk].mean()) if k else np.nan
            # soft down-weighting: w in [~0,1], lowest for highest noise score.
            s = scores - scores.min()
            s = s / (s.max() + 1e-12)
            w = 1.0 - s
            if not is_clf or len(np.unique(y_noisy)) > 1:
                sc = _score_weighted(X_tr, y_noisy, X_va, y_va, is_clf, w)
                d['delta_reweight'] = sc - noisy
        except Exception:
            pass
        res[name] = d
    return res


def _noise_eval(reps, labels, groups, is_clf, bp_sim, bp_pred, best_rep,
                noise_type, seed, sim_objective=OBJECTIVE_SIMILARITY,
                noise_groups=None):
    """Payoff 2, per noise MODEL. For each feature space (sim/pred/rep):
      * DETECTION vs the known injected mask (precision/recall/F1) for every
        method in DENOISE_METHODS, plus flag_fraction (what share of rows the
        method flags). F1 is NOT comparable across methods with different
        flag_fraction: a random detector scores F1 = 2*p*r/(p+r) with precision
        p = NOISE_FRAC, so flagging everything yields F1 = 2*NOISE_FRAC/(1+
        NOISE_FRAC) with zero skill. Judge skill by PRECISION vs NOISE_FRAC.
      * DELTA two ways per method: 'remove' (drop flagged) and 'smooth'
        (correct flagged labels toward their kNN-smoothed value in this space).
        These are RAW paired score changes (after - noisy), not fractions.
      * 'damage' (clean - noisy) is recorded separately so a fraction can be
        formed once at the aggregate level if wanted.
      * A detector-independent GLOBAL kNN label-smoothing baseline per space.

    The hybrid spaces are rebuilt from the NOISY labels (the only labels a real
    deployment has at detection time); the raw-rep space is label-free anyway.
    A SAC-optimized geometry should lift the SIMILARITY methods AND make the
    smoothing/correction better; the CONTROL methods are the null contrast."""
    out = {m: {s: [] for s in ('sim', 'pred', 'rep')} for m in DENOISE_METHODS}
    glob = {s: [] for s in ('sim', 'pred', 'rep')}   # global-smoothing delta
    dmg = {s: [] for s in ('sim', 'pred', 'rep')}    # clean - noisy, per fold
    est = {e: {s: [] for s in ('sim', 'pred', 'rep')} for e in ESTIMATION_METHODS}
    # weighting sweep runs on one noise model only (WEIGHT_NOISE_TYPE) to bound cost.
    wgt = {m: {f: {s: [] for s in ('sim', 'pred', 'rep')} for f in WEIGHT_FASHIONS}
           for m in WEIGHT_TEST_METHODS}
    _check_sim_objective(sim_objective)
    _rng = np.random.RandomState(seed)
    _groups = None if groups is None else np.asarray(groups)
    for fold_i, (tr, va) in enumerate(
            cv_split(groups, N_FOLDS, next(iter(reps.values())), labels, seed=seed)):
        # Subsample the fold to bound the O(n^2) noise-section cost. Relative
        # (space/method) comparison only, so ~1k train / 500 val is ample.
        if len(tr) > NOISE_MAX_N_TRAIN:
            tr = _rng.choice(tr, NOISE_MAX_N_TRAIN, replace=False)
        if len(va) > NOISE_MAX_N_VAL:
            va = _rng.choice(va, NOISE_MAX_N_VAL, replace=False)
        reps_tr = {k: v[tr] for k, v in reps.items()}
        reps_va = {k: v[va] for k, v in reps.items()}
        y_tr, y_va = labels[tr], labels[va]
        # The seed mixes the fold and the condition in. It used to be the run seed
        # alone, so every fold shared one realisation and two conditions at one
        # level drew the same noise -- a comparison between them measured nothing.
        _nseed = (int(seed) * 1000003 + zlib.crc32(noise_type.encode('utf-8'))
                  + fold_i * 104729) & 0x7FFFFFFF
        y_noisy, mask = _inject_typed(y_tr, NOISE_FRAC, noise_type, _nseed, is_clf,
                                      groups=(noise_groups[tr]
                                              if noise_groups is not None else None))
        mask = np.asarray(mask, dtype=bool)
        # Scaffold groups of the (possibly subsampled) train rows, for the
        # read-across scoring inside create_hybrid.
        groups_tr = None if _groups is None else _groups[tr]
        spaces = {}
        # Build the hybrid spaces from y_noisy, NOT the clean y_tr: the feature
        # selection must only see the labels available at detection time. Using
        # y_tr let the space encode the clean values of the very rows it is then
        # asked to flag as corrupted.
        if bp_sim is not None:
            spaces['sim'] = _build_hybrid_fold(reps_tr, reps_va, y_noisy, bp_sim,
                                               sim_objective, is_clf,
                                               groups_tr=groups_tr)
        if bp_pred is not None:
            spaces['pred'] = _build_hybrid_fold(reps_tr, reps_va, y_noisy, bp_pred,
                                                OBJECTIVE_PREDICTIVE, is_clf,
                                                groups_tr=groups_tr)
        # Cap the raw rep space to top-variance features (label-free) to bound the
        # O(n^2 . d) noise-section cost on wide fingerprints.
        rep_tr, rep_va = reps_tr[best_rep], reps_va[best_rep]
        if rep_tr.shape[1] > NOISE_REP_DIM_CAP:
            top = np.argsort(rep_tr.var(axis=0))[-NOISE_REP_DIM_CAP:]
            rep_tr, rep_va = rep_tr[:, top], rep_va[:, top]
        spaces['rep'] = _standardize_pair(rep_tr, rep_va)
        for name, (X_tr, X_va) in spaces.items():
            clean = _score(None, X_tr, y_tr, X_va, y_va, is_clf)
            noisy = _score(None, X_tr, y_noisy, X_va, y_va, is_clf)
            # Recorded, never divided by. (clean - noisy) is near zero on folds
            # where the injected noise barely hurt and negative where it helped.
            dmg[name].append(clean - noisy)
            # kNN-smoothed labels in THIS space (reused by correct-mode + global).
            try:
                y_smooth = _knn_smoothed_labels(X_tr, y_noisy, is_clf)
            except Exception:
                y_smooth = y_noisy
            # Global smoothing baseline (no detection — correct all labels).
            try:
                gs = _score(None, X_tr, y_smooth, X_va, y_va, is_clf)
                glob[name].append(gs - noisy)
            except Exception:
                glob[name].append(np.nan)
            for method in DENOISE_METHODS:
                rec = {'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
                       'flag_fraction': np.nan,
                       'delta_remove': np.nan, 'delta_smooth': np.nan}
                # mahalanobis is O(d^3) and singular on wide spaces (e.g. the raw
                # ecfp4 rep at 2048-d takes ~16s/call and is meaningless). Skip it
                # above 500-d; hybrids (<300-d) always run.
                if method == 'mahalanobis' and X_tr.shape[1] > 500:
                    out[method][name].append(rec)
                    continue
                try:
                    filt = get_mitigation_method(method)
                    flagged = np.asarray(filt.identify_noise(X_tr, y_noisy), dtype=int)
                    p, r, f1 = _detection_scores(flagged, mask)
                    # flag_fraction makes F1 interpretable: a zero-skill detector
                    # that flags everything still scores F1 = 2f/(1+f), f=NOISE_FRAC.
                    rec.update(precision=p, recall=r, f1=f1,
                               flag_fraction=len(np.unique(flagged)) / len(y_noisy))
                    keep = np.ones(len(y_noisy), dtype=bool)
                    keep[flagged] = False
                    if keep.sum() >= 10 and (not is_clf or len(np.unique(y_noisy[keep])) > 1):
                        cl = _score(None, X_tr[keep], y_noisy[keep], X_va, y_va, is_clf)
                        rec['delta_remove'] = cl - noisy
                    # correct-mode: replace flagged labels with smoothed value.
                    y_corr = y_noisy.copy()
                    if len(flagged):
                        y_corr[flagged] = y_smooth[flagged]
                    if not is_clf or len(np.unique(y_corr)) > 1:
                        cs = _score(None, X_tr, y_corr, X_va, y_va, is_clf)
                        rec['delta_smooth'] = cs - noisy
                except Exception:
                    pass
                out[method][name].append(rec)

            # ── Noise ESTIMATION (continuous scores → AUROC + reweight) ──
            for e, ev in _estimation_eval(X_tr, y_noisy, X_va, y_va, mask,
                                          is_clf, noisy).items():
                est[e][name].append(ev)

            # ── Importance-WEIGHTED distances (Payoff 3), one noise model,
            #    low-dim hybrid spaces only (distance matrices are O(n^2.d)) ──
            if noise_type == WEIGHT_NOISE_TYPE and name in WEIGHT_SPACES:
                try:
                    imp = _feature_importance(X_tr, y_noisy, is_clf)
                    for fashion in WEIGHT_FASHIONS:
                        try:
                            D = _weighted_distance_matrix(X_tr, imp, fashion)
                        except Exception:
                            for mth in WEIGHT_TEST_METHODS:
                                wgt[mth][fashion][name].append(
                                    {'f1': np.nan, 'precision': np.nan,
                                     'flag_fraction': np.nan, 'delta_remove': np.nan})
                            continue
                        for mth in WEIGHT_TEST_METHODS:
                            w = {'f1': np.nan, 'precision': np.nan,
                                 'flag_fraction': np.nan, 'delta_remove': np.nan}
                            try:
                                filt = get_mitigation_method(
                                    mth, precomputed_distances=D)
                                fl = np.asarray(filt.identify_noise(X_tr, y_noisy),
                                                dtype=int)
                                pw, _, f1 = _detection_scores(fl, mask)
                                w['f1'] = f1
                                w['precision'] = pw
                                w['flag_fraction'] = len(np.unique(fl)) / len(y_noisy)
                                keep = np.ones(len(y_noisy), dtype=bool)
                                keep[fl] = False
                                if keep.sum() >= 10:
                                    cl = _score(None, X_tr[keep], y_noisy[keep],
                                                X_va, y_va, is_clf)
                                    w['delta_remove'] = cl - noisy
                            except Exception:
                                pass
                            wgt[mth][fashion][name].append(w)
                except Exception:
                    pass
    return out, glob, est, wgt, dmg


def _ci(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {'mean': None, 'lo': None, 'hi': None, 'n': 0}
    a = np.array(vals, float)
    return {'mean': float(a.mean()), 'lo': float(np.percentile(a, 2.5)),
            'hi': float(np.percentile(a, 97.5)), 'n': len(a),
            'std': float(a.std())}


def run_dataset(dataset_name, seeds, verbose=True, cache_dir=None, rep_names=None,
                output_dir=None, sim_objective=OBJECTIVE_SIMILARITY):
    # Reject a cut or unknown objective before anything is loaded or searched.
    _check_sim_objective(sim_objective)
    sim_tier = SIM_OBJECTIVE_TIER[sim_objective]
    if verbose:
        print(f"\n{'='*70}\n{dataset_name}\n{'='*70}")
    data = load_data(dataset_name, rep_names=rep_names)
    reps = {k: np.asarray(v) for k, v in data['reps'].items()}
    # Drop the known-stale herg graph_kernel rep (DATASET_STATUS §5).
    if dataset_name == 'herg':
        reps.pop('graph_kernel', None)
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    is_clf = data.get('task_type') == 'classification'

    sim_sac, pred_sac, rep_sac = [], [], []
    sim_vio, pred_vio, rep_vio = [], [], []
    # Scaffold groups for grouped_wider. Acyclic molecules are singletons
    # (NOISE_DESIGN.md 2a rule 2), and this grouping is deliberately NOT the one that
    # sets the fold boundaries.
    noise_groups, _n_noise_groups, _largest_group_share = noise_groups_for(data)

    noise = {nt: {m: {s: [] for s in ('sim', 'pred', 'rep')}
                  for m in DENOISE_METHODS} for nt in NOISE_TYPES}
    noise_glob = {nt: {s: [] for s in ('sim', 'pred', 'rep')} for nt in NOISE_TYPES}
    noise_dmg = {nt: {s: [] for s in ('sim', 'pred', 'rep')} for nt in NOISE_TYPES}
    noise_est = {nt: {e: {s: [] for s in ('sim', 'pred', 'rep')}
                      for e in ESTIMATION_METHODS} for nt in NOISE_TYPES}
    noise_wgt = {m: {f: {s: [] for s in ('sim', 'pred', 'rep')}
                     for f in WEIGHT_FASHIONS} for m in WEIGHT_TEST_METHODS}
    picks = []

    def _agg(dicts, key):
        return _ci([d[key] for d in dicts])

    def _build_result(seeds_done, progress):
        """Assemble the result dict from whatever has accumulated so far, so a
        killed job still leaves useful PRELIMINARY output. `progress` is a short
        note on how far the current seed got (e.g. 'noise:hetero')."""
        if is_clf:
            payoff2 = {'skipped': 'noise-mitigation denoisers are regression-designed; '
                                  'label-flip detection unsupported on classification'}
            payoff2_est = payoff2
            payoff3 = payoff2
        else:
            payoff2 = {}
            for nt in noise:
                payoff2[nt] = {
                    # RAW paired score change (after - noisy), not a fraction.
                    '_global_smoothing': {
                        sp: _ci(noise_glob[nt][sp]) for sp in ('sim', 'pred', 'rep')},
                    # (clean - noisy): how much the injected noise actually hurt.
                    '_damage': {
                        sp: _ci(noise_dmg[nt][sp]) for sp in ('sim', 'pred', 'rep')},
                }
                for m in DENOISE_METHODS:
                    payoff2[nt][m] = {'class': METHOD_CLASS[m]}
                    for sp in ('sim', 'pred', 'rep'):
                        ds = noise[nt][m][sp]
                        payoff2[nt][m][sp] = {
                            'delta_remove': _agg(ds, 'delta_remove'),
                            'delta_smooth': _agg(ds, 'delta_smooth'),
                            'detection_f1': _agg(ds, 'f1'),
                            'precision': _agg(ds, 'precision'),
                            'recall': _agg(ds, 'recall'),
                            'flag_fraction': _agg(ds, 'flag_fraction'),
                        }
            payoff2_est = {}
            for nt in noise_est:
                payoff2_est[nt] = {}
                for em in ESTIMATION_METHODS:
                    payoff2_est[nt][em] = {'class': ESTIMATOR_CLASS[em]}
                    for sp in ('sim', 'pred', 'rep'):
                        ds = noise_est[nt][em][sp]
                        payoff2_est[nt][em][sp] = {
                            'auroc': _agg(ds, 'auroc'),
                            'precision_at_k': _agg(ds, 'precision_at_k'),
                            'delta_reweight': _agg(ds, 'delta_reweight'),
                        }
            payoff3 = {'noise_type': WEIGHT_NOISE_TYPE, 'fashions': WEIGHT_FASHIONS}
            for mth in WEIGHT_TEST_METHODS:
                payoff3[mth] = {}
                for f in WEIGHT_FASHIONS:
                    payoff3[mth][f] = {}
                    for sp in ('sim', 'pred', 'rep'):
                        ds = noise_wgt[mth][f][sp]
                        payoff3[mth][f][sp] = {
                            'detection_f1': _agg(ds, 'f1'),
                            'precision': _agg(ds, 'precision'),
                            'flag_fraction': _agg(ds, 'flag_fraction'),
                            'delta_remove': _agg(ds, 'delta_remove'),
                        }
        return {
            'dataset': dataset_name,
            'task_type': 'classification' if is_clf else 'regression',
            'n_samples': int(len(labels)), 'n_reps': len(reps),
            'seeds_requested': seeds, 'seeds_completed': seeds_done,
            'progress': progress, 'partial': seeds_done < seeds,
            'picks': picks,
            'method_classes': METHOD_CLASS,
            'estimator_classes': ESTIMATOR_CLASS,
            # Chance precision for every detector = the injected corruption rate.
            'noise_frac': NOISE_FRAC,
            # On a regression dataset the folds come from GroupKFold, which is
            # deterministic and ignores the seed, so the seeds below do NOT
            # resample the folds. On a classification dataset they come from
            # StratifiedGroupKFold, which shuffles with the seed as its random
            # state — so the folds DO move between seeds there and this flag is
            # False. The search also selects its blueprint on these same folds,
            # so payoff1 is a selection-on-evaluation estimate. See §Known
            # limitations in SIM_OPTIMIZED_HYBRID_ANALYSIS_2026_07_10.md.
            'folds_are_seed_invariant': (not is_clf),
            # Which objective SELECTED the 'sim' space, and the cache tier its
            # search rows were written under. Payoff 1 REPORTS SAC for every
            # space; that is a different quantity from the one selected on.
            'sim_objective': sim_objective,
            'sim_search_cache_tier': sim_tier,
            # 'sim' and 'pred' are scored on the same GroupKFold folds their
            # searches selected on: SELECTION-OPTIMISTIC for those two spaces.
            # 'rep' is not selected on these folds; the only selection it
            # carries is the choice of best_rep, made on the sim baseline.
            'payoff1_selection_basis': (
                'sim/pred: scored on the folds their search selected on; '
                'rep: not selected on these folds'),
            'payoff1_heldout_similarity': {
                'sac': {'sim': _ci(sim_sac), 'pred': _ci(pred_sac), 'rep': _ci(rep_sac)},
                'violation': {'sim': _ci(sim_vio), 'pred': _ci(pred_vio), 'rep': _ci(rep_vio)},
            },
            # payoff2[noise_type] = {'_global_smoothing': {space: delta},
            # '_damage': {space: clean-noisy}, method: {'class', space:
            # {delta_remove, delta_smooth, detection_f1, precision, recall,
            # flag_fraction}}}. deltas are RAW paired score changes, not fractions.
            'payoff2_noise': payoff2,
            # payoff2_estimation[noise_type][estimator] = {'class', space:
            # {auroc, precision_at_k, delta_reweight}}.
            'payoff2_estimation': payoff2_est,
            # payoff3[method][fashion][space] = {detection_f1, precision,
            # flag_fraction, delta_remove}.
            'payoff3_importance_weighted': payoff3,
        }

    def _flush(seeds_done, progress):
        """Write the current partial result to disk so preliminary numbers are
        visible while the job runs (and survive an early kill)."""
        res = _build_result(seeds_done, progress)
        if output_dir:
            with open(os.path.join(output_dir, f'{dataset_name}.json'), 'w') as f:
                json.dump(res, f, indent=2)
        return res

    for seed in range(seeds):
        # The geometry arm. sim_objective is 'similarity': scaffold-grouped kNN
        # activity read-across. The cache tier carries the objective
        # (SIM_OBJECTIVE_TIER) so two objectives never share a result row; the
        # tier 'general_sim' held SAC-selected rows and is not written now.
        bp_sim, base_sim = _search(reps, labels, groups, is_clf,
                                   sim_objective, sim_tier, 42 + seed,
                                   dataset_name=dataset_name, cache_dir=cache_dir)
        # Predictive comparator: reuse the EXISTING precompute pool where possible.
        bp_pred, pred_src = _predictive_blueprint(
            reps, labels, groups, is_clf, 42 + seed, dataset_name, data, cache_dir)
        # Best single rep by held-out SAC (from the sim baseline).
        best_rep = max(reps.keys(),
                       key=lambda r: base_sim.get(r, {}).get('rf', -np.inf))
        picks.append({'seed': 42 + seed,
                      'sim_alloc': getattr(bp_sim, 'allocation_method', None),
                      'sim_budget': getattr(bp_sim, 'budget', None),
                      'pred_alloc': getattr(bp_pred, 'allocation_method', None),
                      'pred_budget': getattr(bp_pred, 'budget', None),
                      'pred_source': pred_src,
                      'best_rep': best_rep})

        p1 = _heldout_similarity(reps, labels, groups, is_clf,
                                 bp_sim, bp_pred, best_rep, 42 + seed,
                                 sim_objective=sim_objective)
        sim_sac += p1['sim']['sac']; pred_sac += p1['pred']['sac']; rep_sac += p1['rep']['sac']
        sim_vio += p1['sim']['violation']; pred_vio += p1['pred']['violation']; rep_vio += p1['rep']['violation']
        # Flush after Payoff 1 (cheap) — preliminary similarity numbers land fast.
        _flush(seed, f'seed{seed}:payoff1')

        # Payoff 2 only for regression: the noise-mitigation denoisers are
        # regression-designed, so on classification they do not detect label
        # flips (verified: they flag ~0). herg still contributes Payoff 1.
        if not is_clf:
            _noise_types = list(NOISE_TYPES)
            if noise_groups is None:
                _drop = [c for c in _noise_types if c in GROUPED_STRATEGIES]
                if _drop:
                    print(f"  {', '.join(_drop)}: DROPPED, this dataset has no SMILES "
                          f"to build scaffold groups from. Grouped noise over invented "
                          f"groups is uniform noise wearing a grouped name.")
                    _noise_types = [c for c in _noise_types if c not in _drop]
            for nt in _noise_types:
                r, g, e, wg, dg = _noise_eval(reps, labels, groups, is_clf,
                                              bp_sim, bp_pred, best_rep, nt, 42 + seed,
                                              sim_objective=sim_objective,
                                              noise_groups=noise_groups)
                for sp in ('sim', 'pred', 'rep'):
                    noise_glob[nt][sp] += g[sp]
                    noise_dmg[nt][sp] += dg[sp]
                    for m in DENOISE_METHODS:
                        noise[nt][m][sp] += r[m][sp]
                    for em in ESTIMATION_METHODS:
                        noise_est[nt][em][sp] += e[em][sp]
                    if nt == WEIGHT_NOISE_TYPE:
                        for mth in WEIGHT_TEST_METHODS:
                            for f in WEIGHT_FASHIONS:
                                noise_wgt[mth][f][sp] += wg[mth][f][sp]
                # Flush after each noise model — partial noise results stream in.
                _flush(seed, f'seed{seed}:noise:{nt}')
        _flush(seed + 1, f'seed{seed}:done')

    result = _build_result(seeds, 'complete')
    payoff2 = result['payoff2_noise']
    payoff2_est = result['payoff2_estimation']
    payoff3 = result['payoff3_importance_weighted']
    if verbose:
        s = result['payoff1_heldout_similarity']['sac']
        print(f"  held-out SAC   sim={s['sim']['mean']:.3f}  "
              f"pred={s['pred']['mean']:.3f}  best_rep={s['rep']['mean']:.3f}")
        if 'skipped' in payoff2:
            print("  payoff2 noise: skipped (classification — denoisers are regression-designed)")
            return result
        def _m(d): return d['mean'] if d and d['mean'] is not None else float('nan')
        print(f"  (chance precision = NOISE_FRAC = {NOISE_FRAC:.2f}; deltas are raw "
              f"score changes, not fractions)")
        for nt in NOISE_TYPES:
            g = payoff2[nt]['_global_smoothing']
            d = payoff2[nt]['_damage']
            print(f"  [{nt}] damage (clean-noisy)   sim={_m(d['sim']):+.3f} "
                  f"pred={_m(d['pred']):+.3f} rep={_m(d['rep']):+.3f}")
            print(f"  [{nt}] global-smooth delta    sim={_m(g['sim']):+.3f} "
                  f"pred={_m(g['pred']):+.3f} rep={_m(g['rep']):+.3f}")
            for cls in ('similarity', 'geometry', 'control'):
                ms = [m for m in DENOISE_METHODS if METHOD_CLASS[m] == cls]
                def _clsmean(sp, key):
                    vals = [_m(payoff2[nt][m][sp][key]) for m in ms]
                    vals = [v for v in vals if v == v]
                    return sum(vals) / len(vals) if vals else float('nan')
                print(f"     {cls:10s} d_remove sim={_clsmean('sim','delta_remove'):+.3f} "
                      f"pred={_clsmean('pred','delta_remove'):+.3f} | "
                      f"d_smooth sim={_clsmean('sim','delta_smooth'):+.3f} "
                      f"pred={_clsmean('pred','delta_smooth'):+.3f} | "
                      f"prec sim={_clsmean('sim','precision'):.2f} "
                      f"pred={_clsmean('pred','precision'):.2f} | "
                      f"flag% sim={100*_clsmean('sim','flag_fraction'):.0f}")
        # Estimation: AUROC by estimator class, at the realistic (hetero) model.
        et = WEIGHT_NOISE_TYPE if WEIGHT_NOISE_TYPE in payoff2_est else NOISE_TYPES[0]
        print(f"  noise ESTIMATION AUROC @{et}:")
        for cls in ('similarity', 'control'):
            es = [e for e in ESTIMATION_METHODS if ESTIMATOR_CLASS[e] == cls]
            def _emean(sp):
                vals = [_m(payoff2_est[et][e][sp]['auroc']) for e in es]
                vals = [v for v in vals if v == v]
                return sum(vals) / len(vals) if vals else float('nan')
            print(f"     {cls:10s} AUROC sim={_emean('sim'):.2f} "
                  f"pred={_emean('pred'):.2f} rep={_emean('rep'):.2f}")
        # Payoff 3: importance-weighted distances (detection F1 by fashion).
        print(f"  importance-WEIGHTED distance detF1 @{payoff3['noise_type']} "
              f"(sim space):")
        for mth in WEIGHT_TEST_METHODS:
            cells = " ".join(f"{f}={_m(payoff3[mth][f]['sim']['detection_f1']):.2f}"
                             for f in WEIGHT_FASHIONS)
            print(f"     {mth:22s} {cells}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--reps', nargs='+', default=None,
                    help='Explicit rep list override (default: DATASET_REPS general).')
    ap.add_argument('--output', default='tests/results/sim_optimized_hybrid')
    ap.add_argument('--sim-objective', default=OBJECTIVE_SIMILARITY,
                    type=_cli_sim_objective,
                    metavar='{' + ','.join(SIM_OBJECTIVE_CHOICES) + '}',
                    help="Which geometry the hybrid search optimises. "
                         "'similarity' (default, the only accepted value): "
                         "scaffold-grouped kNN activity read-across. "
                         "'similarity_sac' was cut as a search objective on "
                         "2026-08-26 and is now rejected with a message "
                         "pointing at find_best_hybrid.py lines 1215-1244. "
                         "The objective rides in the cache tier, so results "
                         "should go to their own --output dir.")
    ap.add_argument('--cache-dir', default=os.environ.get('KIRBY_CACHE_DIR'),
                    help='Cache root. Reuses per-rep baseline/importance caches; '
                         'caches the sim search under the tier SIM_OBJECTIVE_TIER '
                         'gives for --sim-objective. Defaults to $KIRBY_CACHE_DIR '
                         'or ~/.cache/kirby.')
    args = ap.parse_args()
    # Second gate, in case the argparse type= is ever dropped: a cut or unknown
    # objective stops the run here, before any dataset is loaded.
    _check_sim_objective(args.sim_objective)

    # Resolve to a concrete cache dir so caching is ALWAYS in place (an empty
    # --cache-dir / unset $KIRBY_CACHE_DIR still gets the standard cache root,
    # which is where the existing precompute baseline/importance caches live).
    cache_dir = (args.cache_dir or os.environ.get('KIRBY_CACHE_DIR')
                 or os.path.expanduser('~/.cache/kirby'))

    os.makedirs(args.output, exist_ok=True)
    print(f"[cache] caching ON, cache_dir={cache_dir}")
    all_results = {}
    for ds in args.datasets:
        try:
            # output_dir => run_dataset streams partial JSON after each seed and
            # each noise model, so preliminary results appear (and survive a kill).
            all_results[ds] = run_dataset(ds, args.seeds, cache_dir=cache_dir,
                                          rep_names=args.reps, output_dir=args.output,
                                          sim_objective=args.sim_objective)
        except SimObjectiveError:
            # A bad objective is a configuration error, not a per-dataset
            # failure: it applies to every dataset. Stop the run instead of
            # writing an error entry into each JSON.
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[ds] = {'error': str(e)}
        with open(os.path.join(args.output, f'{ds}.json'), 'w') as f:
            json.dump(all_results[ds], f, indent=2)
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote results to {args.output}")


if __name__ == '__main__':
    main()
