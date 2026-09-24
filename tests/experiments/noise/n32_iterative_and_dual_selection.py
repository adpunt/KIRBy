#!/usr/bin/env python3
"""N32: trimmed refits, two learners picking for each other, and better detectors.

Paper: Noise.

WHERE THIS CAME FROM. The second research report (2026-09-24, "Label-Noise
Mitigation ... Across Signal-to-Noise Regimes") proposed, for datasets where the
model's out-of-fold error is small:

  H2  ITLM (Shen & Sanghavi 2019): fit, keep the lowest-loss share alpha of the
      training molecules, refit on them, repeat 3-5 rounds.
  H3  a cleanlab-style score: the residual divided by the model's own predicted
      spread, instead of the raw residual.
  H4  shrink the FLAGGED labels only toward the out-of-fold prediction, by
      w = sn^2 / (sn^2 + sm^2).
  H6  two different learners, each choosing the low-loss molecules the OTHER
      one trains on.
  E   conformal scores as a detector.
  O   data valuation (Data Shapley) as a detector.

What already existed, read 2026-09-24: n09 `multistage_cleaning` is 3 rounds at
a fixed 10% drop on IN-SAMPLE loss; n09 `small_loss` is one learner;
`CoTeachingFilter` in src/kirby/utils/noise_mitigation_methods.py is not used by
this directory; n07 `js_shrink` shrinks EVERY row. None divides a residual by a
predicted spread.

THE DETECTORS. Each gives one suspicion score per training molecule, and each is
graded by `detection_scores` against the molecules the injector moved (AUROC,
average precision, precision at the true count). No detector reads the mask.

  oof_residual              |y - p|, p the group-aware out-of-fold LightGBM
                            prediction (kirby.utils.noisy_labels.oof_predict).
  ngboost_rescaled          |y - mu| / sigma from ESTIMATOR_REGISTRY['ngboost'],
                            which keeps its out-of-fold mu and sigma on the
                            fitted object (predictions_, sigma_). The registry's
                            own score is sigma alone, which reads no label; the
                            division is done here.
  quantile_forest_rescaled  |y - median| / spread from
                            ESTIMATOR_REGISTRY['quantile_forest']. Its range is
                            the 10th to 90th percentile of the labels in the
                            leaves, so spread = width / (2 * 1.2816), the SD a
                            normal with that range would have. The registry's own
                            score is the distance OUTSIDE the range, which is 0
                            for most molecules; the division is done here.
  conformal                 |y - pred| / uncertainty from
                            ESTIMATOR_REGISTRY['conformal'] (a forest's per-tree
                            spread). This is the conformity score the report's
                            conformal outlier p-values are computed from; the
                            p-value is decreasing in it within one calibration
                            fold. The registry's own score is the interval
                            WIDTH, which reads no label.
  knn_shapley               minus each molecule's Shapley value to a 5-nearest-
                            neighbour predictor, summed with every other training
                            molecule as the test point. See knn_shapley_values.

THE METHODS. Every downstream model is fitted on what each method leaves, in
every feature space, and scored on held-out molecules against their UNTOUCHED
labels.

  clean                     the training labels before injection. The most any
                            method can recover.
  no_cleaning               the corrupted labels as they are. The arm to beat.
  itlm_a<alpha>             4 rounds. Each round: a LightGBM (oof_predict's
                            settings) is fitted on the kept molecules; a kept
                            molecule's loss is its OUT-OF-FOLD squared error, a
                            dropped molecule's is its error under the model
                            fitted on all kept molecules (the report's snippet);
                            the lowest-loss share alpha is kept. The overlap of
                            the kept sets of consecutive rounds is written.
  itlm_a<alpha>_random/seed<k>
                            the same NUMBER of molecules removed at random. The
                            control, for n24's reason: a smaller training set is
                            a different fit, and sometimes a better one.
  two_learner               lightgbm and svm (the suite's own factories). Each
                            scores every molecule out of fold; each hands the
                            other its lowest-loss share; each refits on what it
                            was handed and rescores; 2 exchange rounds. Every
                            downstream model is then fitted on the INTERSECTION
                            of the two learners' final kept sets -- the molecules
                            both would keep. The union was the other option; with
                            the two learners agreeing on most molecules it would
                            remove almost nothing. How far the two flagged sets
                            agree is written as counts.
  two_learner_random/seed<k>
                            the same number removed at random.
  shrink_flagged_one_weight the flagged share (the top of |y - p|) moved toward p
                            by ONE weight w = sn^2 / (sn^2 + sm^2). sm^2 is the
                            mean square of y - p over the UNFLAGGED molecules;
                            sn^2 is how far the flagged molecules' mean square
                            exceeds it, floored at 0. Mean squares, not
                            variances: a one-direction shift has a small
                            variance and a large mean square. Both are estimated from the
                            corrupted labels; the injector's size is NOT read.
  shrink_flagged_per_row    the same flagged share, each molecule with its own
                            w_i = max(r_i^2 - sm^2, 0) / (max(r_i^2 - sm^2, 0)
                            + sm^2).

THE FLAGGED SHARE is the condition's NOMINAL share (FLAG_SHARE below), what a
user who knew the condition would ask for. It is not the realised count of moved
molecules, which a user cannot see. Under grouped_wider the nominal share is of
scaffold GROUPS, so the molecule share differs, and the realised count is
written beside it.

TAIL SCORES. Every scored fit also writes `tail_scores`: rmse on the held-out
molecules whose clean label is in the top and the bottom tenth, and the
calibration slope. The report warns that trimming and relabelling at high R^2
remove genuine extremes; the overall rmse can hide that.

WHERE THE FEATURES COME FROM. The recipe for each downstream model is read from
the precomputed pool, which was searched on clean labels. Each fold rebuilds the
hybrid from that fold's CORRUPTED training labels, once, and every detector and
method shares it. `clean` is the clean labels in corrupted-label features.

WHAT THIS CANNOT SAY. The noise is injected, so every number is confined to the
named condition. Nothing here measures real label error.

Usage:
    python n32_iterative_and_dual_selection.py --dataset logd --mode full
    python n32_iterative_and_dual_selection.py --dataset herg_ki --mode full \
        --noise-condition shift_only_p10
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time
import zlib

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream, pick_noise_base_rep,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, hybrid_for_fold,
    pool_recipe_per_downstream_model, group_models_by_recipe,
    detection_scores, detection_is_defined, tail_scores, sweep_conditions,
    DETECTION_IS_DEFINED, SHIFT_ONLY_CONDITIONS, SHIFT_ONLY_DELTAS_SD,
    inject_shift_train_labels,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.noisy_labels import oof_predict
from kirby.utils.noisy_labels.core import (
    make_model as make_oof_lgbm, group_kfold_splitter, _safe_n_splits,
)

DEFAULT_NOISE_CONDITION = 'outlier_only_p10'

# The nominal share of molecules each condition damages. See the docstring.
FLAG_SHARE = {
    'outlier_p01': 0.01, 'outlier_p05': 0.05, 'outlier_p10': 0.10,
    'outlier_only_p01': 0.01, 'outlier_only_p05': 0.05,
    'outlier_only_p10': 0.10,
    'grouped_wider': 0.20,
    **SHIFT_ONLY_CONDITIONS,
}
# censoring's share is its level, the fraction clipped.

NOISE_DRAWS = (0, 1, 2)
ITLM_ALPHAS = (0.8, 0.9, 0.95)
ITLM_ROUNDS = 4
EXCHANGE_ROUNDS = 2
EXCHANGE_LEARNERS = ('lightgbm', 'svm')
RANDOM_SEEDS = (0, 1, 2)
OOF_SPLITS = 5
KNN_SHAPLEY_K = 5
DETECTORS = ('oof_residual', 'ngboost_rescaled', 'quantile_forest_rescaled',
             'conformal', 'knn_shapley')
GAP_MIN_FRAC = 0.02

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}

# The spread of a normal whose 10th-to-90th percentile range is 1.
_Q10_Q90_WIDTH_IN_SD = 2.0 * float(norm.ppf(0.9))


# ── Detectors ────────────────────────────────────────────────────────────────

def knn_shapley_values(X, y, k=KNN_SHAPLEY_K, groups=None, block=512):
    """Each training molecule's Shapley value to a k-nearest-neighbour predictor.

    THE UTILITY. For one test molecule t with label y_t, a subset S of the other
    training molecules is worth
        U(S) = (1/k) * sum over the min(k, |S|) members of S nearest to t
                       of  -(y_i - y_t)^2
    i.e. minus the mean squared label gap to the neighbours, one term per
    neighbour. That is NOT minus the squared error of the neighbours' MEAN: the
    mean's squared error is not a sum over neighbours, and Jia et al.'s exact
    recursion needs a sum. The per-neighbour sum is the regression analogue of
    their classification utility (1/k) * sum 1[y_i = y_t], with the indicator
    replaced by -(y_i - y_t)^2.

    THE RECURSION (Jia et al. 2019, "Efficient task-specific data valuation for
    nearest neighbor algorithms", Theorem 1), with u_i = -(y_(i) - y_t)^2 and the
    others sorted nearest first, i = 1..N:
        s_N = u_N / N
        s_i = s_(i+1) + (u_i - u_(i+1)) / k * min(k, i) / i
    It is exact for this utility; _smoke_n32 checks it against a brute-force
    Shapley value over every subset on 7 molecules.

    EVERY training molecule is the test point once, with itself -- and, when
    groups are given, every molecule of its own scaffold group -- left out of
    the others, the same group rule the out-of-fold detectors follow. The values
    are summed over the test points and divided by their number.

    Returns one value per molecule; a LOW value is a molecule that hurts.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    g = None if groups is None else np.asarray(groups)
    sq = np.einsum('ij,ij->i', X, X)
    values = np.zeros(n)
    counted = 0
    for start in range(0, n, block):
        stop = min(n, start + block)
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * X[start:stop] @ X.T
        for r, t in enumerate(range(start, stop)):
            dist = d2[r].copy()
            dist[t] = np.inf
            if g is not None:
                dist[g == g[t]] = np.inf
            others = np.flatnonzero(np.isfinite(dist))
            N = others.size
            if N == 0:
                continue
            order = others[np.argsort(dist[others], kind='stable')]
            u = -(y[order] - y[t]) ** 2
            s = np.empty(N)
            s[-1] = u[-1] / N
            if N > 1:
                i = np.arange(1, N)                   # 1-indexed i = 1..N-1
                terms = (u[:-1] - u[1:]) / k * np.minimum(k, i) / i
                s[:-1] = s[-1] + np.cumsum(terms[::-1])[::-1]
            values[order] += s
            counted += 1
    return values / max(counted, 1)


def _rescaled(y, centre, spread):
    spread = np.asarray(spread, dtype=float)
    return np.abs(np.asarray(y, dtype=float) - np.asarray(centre, dtype=float)) / \
        np.clip(spread, 1e-9, None)


def detector_scores(name, X, y, groups, p_oof, oof_splits):
    """One suspicion score per training molecule; higher = more suspect."""
    from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY
    y = np.asarray(y, dtype=float)
    if name == 'oof_residual':
        return np.abs(y - p_oof)
    if name == 'ngboost_rescaled':
        est = ESTIMATOR_REGISTRY['ngboost'](n_folds=oof_splits, groups=groups)
        est.estimate(X, y)
        return _rescaled(y, est.predictions_, est.sigma_)
    if name == 'quantile_forest_rescaled':
        est = ESTIMATOR_REGISTRY['quantile_forest'](n_folds=oof_splits,
                                                    groups=groups)
        est.estimate(X, y)
        spread = np.asarray(est.interval_widths_) / _Q10_Q90_WIDTH_IN_SD
        return _rescaled(y, est.predictions_, spread)
    if name == 'conformal':
        est = ESTIMATOR_REGISTRY['conformal'](n_folds=oof_splits, groups=groups)
        est.estimate(X, y)
        return _rescaled(y, est.predictions_, est.uncertainties_)
    if name == 'knn_shapley':
        return -knn_shapley_values(X, y, groups=groups)
    raise ValueError(f'n32: unknown detector {name!r}')


# ── Methods ─────────────────────────────────────────────────────────────────

def subset_loss(make, X, y, groups, keep, oof_splits):
    """Squared error of every molecule under a model trained on the kept ones.

    A kept molecule's error is OUT OF FOLD within the kept set (group-aware); a
    dropped molecule's is under one model fitted on every kept molecule. This is
    the report's H2 snippet with the model left as an argument.
    """
    y = np.asarray(y, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    ki = np.flatnonzero(keep)
    g_k = np.asarray(groups)[ki]
    loss = np.empty(len(y))
    k = _safe_n_splits(g_k, oof_splits)
    oof = np.full(ki.size, np.nan)
    for tr, te in group_kfold_splitter(g_k, k, X[ki], y[ki]):
        m = make()
        m.fit(X[ki][tr], y[ki][tr])
        oof[te] = m.predict(X[ki][te])
    loss[ki] = (y[ki] - oof) ** 2
    di = np.flatnonzero(~keep)
    if di.size:
        m = make()
        m.fit(X[ki], y[ki])
        loss[di] = (y[di] - m.predict(X[di])) ** 2
    return loss


def lowest_share(loss, share_kept):
    """Boolean mask of the lowest-loss share of rows, ties broken by position."""
    loss = np.asarray(loss, dtype=float)
    n_keep = int(round(share_kept * loss.size))
    n_keep = min(max(n_keep, 2), loss.size)
    keep = np.zeros(loss.size, dtype=bool)
    keep[np.argsort(loss, kind='stable')[:n_keep]] = True
    return keep


def jaccard(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int(np.sum(a | b))
    return float(np.sum(a & b)) / union if union else 1.0


def itlm(X, y, groups, alpha, rounds=ITLM_ROUNDS, oof_splits=OOF_SPLITS,
         make=None):
    """ITLM: returns (final keep mask, last round's loss, per-round records)."""
    make = make or make_oof_lgbm
    keep = np.ones(len(y), dtype=bool)
    records = []
    loss = None
    for t in range(int(rounds)):
        loss = subset_loss(make, X, y, groups, keep, oof_splits)
        new_keep = lowest_share(loss, alpha)
        records.append({'round': t + 1,
                        'kept': int(new_keep.sum()),
                        'overlap_with_previous_round_kept_set': (
                            jaccard(keep, new_keep) if t > 0 else None)})
        keep = new_keep
    return keep, loss, records


def two_learner_exchange(X, y, groups, make_a, make_b, share_kept,
                         rounds=EXCHANGE_ROUNDS, oof_splits=OOF_SPLITS):
    """Each learner picks the lowest-loss share for the OTHER to train on.

    Returns (intersection keep mask, record). The record holds the counts of
    molecules each learner flags and how many both flag.
    """
    all_rows = np.ones(len(y), dtype=bool)
    loss_a = subset_loss(make_a, X, y, groups, all_rows, oof_splits)
    loss_b = subset_loss(make_b, X, y, groups, all_rows, oof_splits)
    for _ in range(int(rounds)):
        keep_for_a = lowest_share(loss_b, share_kept)
        keep_for_b = lowest_share(loss_a, share_kept)
        loss_a = subset_loss(make_a, X, y, groups, keep_for_a, oof_splits)
        loss_b = subset_loss(make_b, X, y, groups, keep_for_b, oof_splits)
    keep_a = lowest_share(loss_a, share_kept)
    keep_b = lowest_share(loss_b, share_kept)
    both = keep_a & keep_b
    flag_a, flag_b = ~keep_a, ~keep_b
    return both, {
        'flagged_by_first_learner': int(flag_a.sum()),
        'flagged_by_second_learner': int(flag_b.sum()),
        'flagged_by_both': int(np.sum(flag_a & flag_b)),
        'flagged_by_either': int(np.sum(flag_a | flag_b)),
        'kept_by_both_and_trained_on': int(both.sum()),
        'training_molecules': int(len(y)),
    }, np.maximum(loss_a, loss_b)


def shrink_flagged(y, p, share):
    """The two flagged-only shrink columns and what they estimated."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    r = y - p
    flag = ~lowest_share(np.abs(r), 1.0 - share)
    if flag.sum() < 2 or (~flag).sum() < 2:
        return None, None, {'why_none': 'the flagged or unflagged set holds '
                                        'fewer than 2 molecules'}
    # MEAN SQUARES, not variances. Under a one-direction shift every flagged
    # residual sits near +delta, so their VARIANCE is small and a variance-based
    # sn^2 came out 0 on the smoke's toy data -- the shrink moved nothing. p is
    # an out-of-fold prediction of the labels, so a residual's expected value is
    # 0 and its mean square is its spread around the truth.
    sm2 = float(np.mean(r[~flag] ** 2))
    sn2 = max(float(np.mean(r[flag] ** 2)) - sm2, 0.0)
    w = sn2 / (sn2 + sm2) if (sn2 + sm2) > 0 else 0.0
    one = y.copy()
    one[flag] = (1.0 - w) * y[flag] + w * p[flag]
    excess = np.maximum(r[flag] ** 2 - sm2, 0.0)
    w_i = excess / (excess + sm2) if sm2 > 0 else np.zeros_like(excess)
    per = y.copy()
    per[flag] = (1.0 - w_i) * y[flag] + w_i * p[flag]
    return one, per, {
        'molecules_flagged': int(flag.sum()),
        'mean_square_residual_unflagged_label_units2': sm2,
        'extra_mean_square_on_flagged_label_units2': sn2,
        'one_weight': w,
        'per_row_weight_mean': float(np.mean(w_i)),
    }


def replicate_seed(*parts):
    """zlib.crc32, not hash(): stable across processes (n24:308-322)."""
    key = '|'.join(str(p) for p in parts)
    return int(zlib.crc32(key.encode('utf-8')) & 0x7FFFFFFF)


def random_keep(n, n_remove, seed):
    keep = np.ones(n, dtype=bool)
    if n_remove > 0:
        rng = np.random.default_rng(int(seed))
        keep[rng.choice(n, int(n_remove), replace=False)] = False
    return keep


def build_arm_names(alphas, seeds):
    out = ['clean', 'no_cleaning']
    for a in alphas:
        out.append(f'itlm_a{a:g}')
        out += [f'itlm_a{a:g}_random/seed{s}' for s in seeds]
    out.append('two_learner')
    out += [f'two_learner_random/seed{s}' for s in seeds]
    out += ['shrink_flagged_one_weight', 'shrink_flagged_per_row']
    return out


def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, metric_name,
               keep=None):
    """One fit; (score, tail dict, None) or (None, None, why)."""
    try:
        m = factory()
        if keep is None:
            m.fit(X_tr, y_tr)
        else:
            m.fit(X_tr[keep], np.asarray(y_tr)[keep])
        pred = _predict_for_metric(m, X_va, metric_name)
        return (float(metric_fn(y_va, pred)),
                tail_scores(y_va, np.asarray(pred, dtype=float)), None)
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'


def _levels_and_draws(dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs. At level 0 nothing moves, so one draw is kept."""
    if levels is None:
        if condition in SHIFT_ONLY_CONDITIONS:
            levels = (0.0,) + tuple(SHIFT_ONLY_DELTAS_SD)
        else:
            levels = noise_levels(dataset_name, strategy=condition,
                                  is_clf=False)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        for d in (draws if float(lvl) > 0 else draws[:1]):
            out.append((float(lvl), int(d)))
    return out


EXP_ID = 'n32_iterative_and_dual_selection'


def run(data, config, output_dir, dataset_name, **kwargs):
    """N32 across every condition where detection is defined, one file each.

    The two shift_only_* conditions are not in the NoiseInject registry, so no
    per-condition job ever names them. They run here, in the job that names no
    condition, beside the eight registry conditions. A job that names a
    condition where every label moves gets a skip carrying the reason.
    """
    if data['task_type'] != 'regression':
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    named = kwargs.get('noise_condition')
    if named and not detection_is_defined(named):
        print(f"  SKIPPED: {named} moves every training label, so there is no "
              f"damaged subset to find.")
        return {'skipped': 'detection_undefined', 'noise_condition': named}
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=list(DETECTION_IS_DEFINED), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N32: ITLM, two-learner exchange, flagged-only shrink, five detectors."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N32: TRIMMED REFITS, TWO LEARNERS, BETTER DETECTORS")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. Every method here "
              f"ranks molecules by a squared residual, so n32 is regression "
              f"only.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    if not detection_is_defined(condition):
        raise ValueError(
            f"n32: condition {condition!r} moves every training label, so there "
            f"is no damaged subset to find. Run it under one of: "
            f"{', '.join(DETECTION_IS_DEFINED)}.")
    if condition not in FLAG_SHARE and condition != 'censoring':
        raise ValueError(f"n32: no nominal flag share for {condition!r}")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    alphas = [float(a) for a in (kwargs.get('alphas') or ITLM_ALPHAS)]
    seeds = [int(s) for s in (kwargs.get('random_seeds') or RANDOM_SEEDS)]
    detectors = list(kwargs.get('detectors') or DETECTORS)
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    itlm_rounds = int(kwargs.get('itlm_rounds', ITLM_ROUNDS))
    arm_names = build_arm_names(alphas, seeds)

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError("n32: no downstream model could be built.")
    learner_factories = get_model_factories(
        list(EXCHANGE_LEARNERS), task_type=task_type, n_samples=len(labels))
    missing_learners = [m for m in EXCHANGE_LEARNERS
                        if m not in learner_factories]
    if missing_learners:
        raise RuntimeError(f"n32: the exchange learners {missing_learners} "
                           f"could not be built.")

    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    noise_groups = noise_groups_for(data)[0]
    if condition in GROUPED_STRATEGIES and noise_groups is None:
        raise RuntimeError(
            f"n32: condition {condition!r} needs scaffold groups and "
            f"{dataset_name} has none.")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': ('label SDs of upward shift'
                       if condition in SHIFT_ONLY_CONDITIONS
                       else noise_dose_units(dataset_name)),
        'nominal_flag_share': FLAG_SHARE.get(condition,
                                             'the level (fraction clipped)'),
        'nominal_flag_share_note': (
            'The share every flag-based method removes or shrinks is the '
            "condition's NOMINAL share, not the realised count of moved "
            'molecules. Under grouped_wider the nominal share is of scaffold '
            'groups.'),
        'noise_draws': list(NOISE_DRAWS if kwargs.get('draws') is None
                            else kwargs['draws']),
        'models': downstream_models,
        'exchange_learners': list(EXCHANGE_LEARNERS),
        'exchange_rule': ('every downstream model is fitted on the '
                          'INTERSECTION of the two learners\' final kept sets'),
        'base_rep': base_rep_name,
        'arms': arm_names,
        'arm_to_beat': 'no_cleaning',
        'detectors': detectors,
        'itlm_alphas': alphas,
        'itlm_rounds': itlm_rounds,
        'exchange_rounds': EXCHANGE_ROUNDS,
        'random_seeds': seeds,
        'oof_splits': oof_splits,
        'knn_shapley_k': KNN_SHAPLEY_K,
        'knn_shapley_utility': (
            'mean over the k nearest of -(y_i - y_test)^2, one term per '
            'neighbour; not the squared error of the neighbours\' mean'),
        'shrink_note': (
            'sm^2 and sn^2 are estimated from the corrupted labels: sm^2 is '
            'the mean square of y - p over the unflagged molecules, sn^2 how '
            'far the flagged molecules\' mean square exceeds it. The injector\'s own '
            'size is not read.'),
        'features_are_fixed_note': (
            'Each feature space is built once per (level, draw, fold) from the '
            'CORRUPTED training labels and shared by every detector and '
            'method, clean included.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to this '
            'condition on this dataset. Held-out molecules are graded against '
            'untouched labels in every arm.'),
        'per_level': {},
        'tails': {},
        'detection': {},
        'method_records': {},
    }

    sample_X = reps[base_rep_name]
    splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))

    # ── The recipe per downstream model, n30's block ──
    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    _raised = [d['lookup_error'] for d in no_recipe.values()
               if d['lookup_error']]
    results['hybrid_lookup_error'] = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}
    recipes_by_model, recipe_for_space_by_model = {}, {}
    hybrid_unavailable = {}
    for _m in downstream_models:
        _m_recipes, _m_unavail = pool_recipes_by_family(
            by_model[_m]['meta'] if _m in by_model else None,
            SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipes_by_model[_m] = _m_recipes
        recipe_for_space_by_model[_m] = {r['space']: r['blueprint']
                                         for r in _m_recipes}
        for _sp, _why in _m_unavail.items():
            hybrid_unavailable.setdefault(_sp, {})[_m] = _why
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    _ordered = []
    for _m in downstream_models:
        for r in recipes_by_model.get(_m, ()):
            if r['space'] not in _ordered:
                _ordered.append(r['space'])
    spaces = ('base',) + tuple(_ordered)
    results['hybrid_available'] = bool(_ordered)
    if _ordered:
        results['hybrid_recipe_by_model'] = convert_numpy({
            m: {r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
            } for r in rs}
            for m, rs in recipes_by_model.items() if rs})
    else:
        results['hybrid_unavailable_reason'] = (
            (f"the pool lookup RAISED {results['hybrid_lookup_error']}; the "
             f"pool was never read. Every arm ran in the base representation.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from pool subset {POOL_SUBSET!r} for any '
             f'downstream model. Every arm ran in the base representation.'))
        print("  no hybrid recipe; base representation only")
    results['spaces_run'] = list(spaces)

    passes = _levels_and_draws(dataset_name, condition,
                               levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} molecules, {len(splits)} folds, models "
          f"{downstream_models}")
    print(f"  condition {condition}, levels {results['noise_levels']}, "
          f"{len(passes)} (level, draw) passes, {len(arm_names)} arms, "
          f"detectors {detectors}")

    acc, tail_acc, det_acc, rec_acc, errors = {}, {}, {}, {}, {}
    t0 = time.time()

    for level, draw in passes:
        lvl_key = f'{level:g}'
        share = (FLAG_SHARE[condition] if condition in FLAG_SHARE
                 else float(level))
        spec = (None if condition in SHIFT_ONLY_CONDITIONS
                else noise_spec_for(dataset_name, condition, level,
                                    noise_seed=draw, is_clf=False))
        print(f"\n  --- {condition} / level {lvl_key} / draw {draw} ---")
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            y_val = labels[val_idx]
            y_clean_tr = labels[train_idx]
            g_tr = (np.asarray(groups)[train_idx] if groups is not None
                    else np.arange(len(train_idx)))
            try:
                if condition in SHIFT_ONLY_CONDITIONS:
                    y0, cmask, prov = inject_shift_train_labels(
                        labels, train_idx=train_idx, condition=condition,
                        delta_sd=level, seed=draw * 1000 + fold_i)
                else:
                    y0, cmask, prov = inject_train_labels(
                        labels, spec, train_idx=train_idx, is_clf=False,
                        groups=noise_groups, dataset_name=dataset_name)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            cmask = np.asarray(cmask, dtype=bool)
            n_tr = int(len(y0))

            train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                          for k, v in reps.items()}
            val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}
            fold_spaces = {}
            sc = StandardScaler()
            fold_spaces['base'] = [(
                sc.fit_transform(train_reps[base_rep_name]),
                sc.transform(val_reps[base_rep_name]),
                list(downstream_models))]
            for _sp in spaces:
                if _sp == 'base':
                    continue
                _cells = []
                for _bp_m, _mnames, _shareable in group_models_by_recipe(
                        {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                         for _m in downstream_models}, downstream_models):
                    X_tr_h, X_va_h = hybrid_for_fold(
                        train_reps, val_reps, y0, _bp_m, g_tr,
                        None if _shareable
                        else model_factories[_mnames[0]](),
                        metric_fn, lower_is_better)
                    if X_tr_h is None:
                        errors[f'{lvl_key}|{_sp}/{_mnames[0]}|f{fold_i}|'
                               f'build'] = 'the hybrid could not be built'
                        continue
                    sh = StandardScaler()
                    _cells.append((sh.fit_transform(X_tr_h),
                                   sh.transform(X_va_h), list(_mnames)))
                if _cells:
                    fold_spaces[_sp] = _cells

            for space in spaces:
                for X_tr_s, X_va_s, cell_models in fold_spaces.get(space, ()):
                    _ekey = (space if cell_models == list(downstream_models)
                             else f'{space}/{cell_models[0]}')
                    rec = {'pass': pass_key, 'training_molecules': n_tr,
                           'molecules_moved': int(cmask.sum()),
                           'nominal_flag_share': share}

                    try:
                        p = oof_predict(X_tr_s, y0, g_tr, n_splits=oof_splits)
                    except Exception as exc:
                        errors[f'{lvl_key}|{_ekey}|f{fold_i}|oof'] = (
                            f'{type(exc).__name__}: {exc}')
                        p = None

                    # ── Detectors, graded against the mask ──
                    for det in detectors:
                        if det == 'oof_residual' and p is None:
                            continue
                        try:
                            sus = detector_scores(det, X_tr_s, y0, g_tr, p,
                                                  oof_splits)
                        except Exception as exc:
                            errors[f'{lvl_key}|{_ekey}|{det}|f{fold_i}'] = (
                                f'{type(exc).__name__}: {exc}')
                            continue
                        d = detection_scores(cmask, sus, condition)
                        if d is not None:
                            (det_acc.setdefault(lvl_key, {})
                             .setdefault(_ekey, {})
                             .setdefault(det, []).append(d))

                    # ── The training columns and keep masks ──
                    cols = {'clean': (y_clean_tr, None),
                            'no_cleaning': (y0, None)}
                    for a in alphas:
                        try:
                            keep, last_loss, rounds = itlm(
                                X_tr_s, y0, g_tr, a, rounds=itlm_rounds,
                                oof_splits=oof_splits)
                        except Exception as exc:
                            errors[f'{lvl_key}|{_ekey}|itlm_a{a:g}|'
                                   f'f{fold_i}'] = f'{type(exc).__name__}: {exc}'
                            continue
                        cols[f'itlm_a{a:g}'] = (y0, keep)
                        n_rm = int(n_tr - keep.sum())
                        rec[f'itlm_a{a:g}'] = {
                            'rounds': rounds, 'molecules_removed': n_rm,
                            'moved_molecules_among_the_removed':
                                int(np.sum(cmask & ~keep)),
                            'detection_by_last_round_loss':
                                detection_scores(cmask, last_loss, condition)}
                        for s in seeds:
                            rk = random_keep(n_tr, n_rm, replicate_seed(
                                s, lvl_key, draw, fold_i, n_tr, n_rm))
                            cols[f'itlm_a{a:g}_random/seed{s}'] = (y0, rk)
                    try:
                        both, xrec, xloss = two_learner_exchange(
                            X_tr_s, y0, g_tr,
                            learner_factories[EXCHANGE_LEARNERS[0]],
                            learner_factories[EXCHANGE_LEARNERS[1]],
                            1.0 - share, oof_splits=oof_splits)
                        cols['two_learner'] = (y0, both)
                        n_rm = int(n_tr - both.sum())
                        xrec['moved_molecules_among_the_removed'] = int(
                            np.sum(cmask & ~both))
                        xrec['detection_by_larger_of_the_two_losses'] = (
                            detection_scores(cmask, xloss, condition))
                        rec['two_learner'] = xrec
                        for s in seeds:
                            rk = random_keep(n_tr, n_rm, replicate_seed(
                                'x', s, lvl_key, draw, fold_i, n_tr, n_rm))
                            cols[f'two_learner_random/seed{s}'] = (y0, rk)
                    except Exception as exc:
                        errors[f'{lvl_key}|{_ekey}|two_learner|f{fold_i}'] = (
                            f'{type(exc).__name__}: {exc}')
                    if p is not None:
                        one, per, srec = shrink_flagged(y0, p, share)
                        rec['shrink_flagged'] = srec
                        if one is not None:
                            cols['shrink_flagged_one_weight'] = (one, None)
                            cols['shrink_flagged_per_row'] = (per, None)

                    rec_acc.setdefault(lvl_key, {}).setdefault(
                        _ekey, []).append(rec)

                    space_cell = acc.setdefault(lvl_key, {}).setdefault(
                        space, {})
                    tail_cell = tail_acc.setdefault(lvl_key, {}).setdefault(
                        space, {})
                    for arm in arm_names:
                        if arm not in cols:
                            continue
                        y_use, keep = cols[arm]
                        for mname in cell_models:
                            s_, tl, err = _fit_score(
                                model_factories[mname], X_tr_s, y_use, X_va_s,
                                y_val, metric_fn, metric_name, keep=keep)
                            if s_ is None:
                                errors[f'{lvl_key}|{_ekey}|{arm}|{mname}|'
                                       f'f{fold_i}'] = err
                                continue
                            (space_cell.setdefault(arm, {})
                             .setdefault(mname, {})[pass_key]) = s_
                            if tl is not None:
                                (tail_cell.setdefault(arm, {})
                                 .setdefault(mname, {})[pass_key]) = tl
            print(f"    level {lvl_key} draw {draw} fold {fold_i}: "
                  f"{int(cmask.sum())} of {n_tr} training molecules moved "
                  f"[{time.time() - t0:.0f}s]")

    results['per_level'] = convert_numpy(acc)
    results['tails'] = convert_numpy(tail_acc)
    results['detection'] = convert_numpy({
        lvl: {sp: {d: _mean_detection(rows) for d, rows in by_det.items()}
              for sp, by_det in by_sp.items()}
        for lvl, by_sp in det_acc.items()})
    results['method_records'] = convert_numpy(rec_acc)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, tail_acc, arm_names, downstream_models, alphas, seeds,
        metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _mean_detection(rows):
    """Average the detection readings over the passes of one level."""
    if not rows:
        return None
    keys = ('detection_auroc', 'detection_average_precision',
            'detection_precision_at_k', 'detection_n_corrupted',
            'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows
                if r.get(k) is not None and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_n_passes'] = len(rows)
    return out


def _paired(base, got, lower_is_better):
    """n24:981-1006. Positive gain means the second beat the first."""
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_passes_paired': int(gains.size),
        'passes_won': int((gains > tol).sum()),
        'passes_lost': int((gains < -tol).sum()),
        'passes_tied': int((np.abs(gains) <= tol).sum()),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_pass_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'pass_gains': [float(x) for x in gains],
    }


def recovery_share(clean, noisy, arm, lower_is_better,
                   gap_min_frac=GAP_MIN_FRAC):
    """Share of the clean-to-corrupted gap an arm recovers, or (None, why)."""
    if not all(np.isfinite(v) for v in (clean, noisy, arm)):
        return None, 'a score is missing'
    gap = (noisy - clean) if lower_is_better else (clean - noisy)
    floor = gap_min_frac * abs(clean)
    if gap < floor:
        return None, (f'the clean-to-corrupted gap is {gap:+.6f}, under '
                      f'{gap_min_frac:g} of the clean score ({floor:.6f})')
    gain = (noisy - arm) if lower_is_better else (arm - noisy)
    return float(gain / gap), None


def _mean_on(d, keys):
    vals = [d[k] for k in keys if k in d and np.isfinite(d[k])]
    return float(np.mean(vals)) if vals else float('nan')


def _tail_row(tails_arm, tails_base):
    """Mean of each tail number for the arm, and its mean change from
    no_cleaning on the passes both scored."""
    keys = sorted(set(tails_arm) & set(tails_base))
    out = {'n_passes': len(keys)}
    for f in ('tail_rmse_top', 'tail_rmse_bottom', 'calibration_slope'):
        a = [tails_arm[k][f] for k in keys]
        b = [tails_base[k][f] for k in keys]
        out[f] = float(np.mean(a)) if a else float('nan')
        out[f'{f}_change_from_no_cleaning'] = (
            float(np.mean(np.subtract(a, b))) if a else float('nan'))
    return out


def _summarise(acc, tail_acc, arm_names, models, alphas, seeds, metric_name,
               lower_is_better):
    """Every arm against no_cleaning; each removal arm against its randoms."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against no_cleaning, paired per pass; a '
                        'pass is one (draw, fold)'),
           'per_level': {}}
    removal = {f'itlm_a{a:g}': [f'itlm_a{a:g}_random/seed{s}' for s in seeds]
               for a in alphas}
    removal['two_learner'] = [f'two_learner_random/seed{s}' for s in seeds]
    for lvl, by_space in acc.items():
        lrow = {}
        for space, by_arm in by_space.items():
            base = by_arm.get('no_cleaning') or {}
            clean = by_arm.get('clean') or {}
            tails = (tail_acc.get(lvl) or {}).get(space) or {}
            srow = {}
            for m in models:
                b = base.get(m) or {}
                if not b:
                    continue
                row = {'no_cleaning_mean': _mean_on(b, list(b)),
                       'per_arm': {}}
                for arm in arm_names:
                    if arm == 'no_cleaning':
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    e = {'vs_no_cleaning': _paired(b, g, lower_is_better)}
                    if arm != 'clean':
                        keys = sorted(set(clean.get(m) or {}) & set(b) & set(g))
                        sh, why = (recovery_share(
                            _mean_on(clean[m], keys), _mean_on(b, keys),
                            _mean_on(g, keys), lower_is_better)
                            if keys else (None, 'no pass scored all three'))
                        e['recovery_share'] = sh
                        e['recovery_share_refused_because'] = why
                    if arm in removal:
                        vs = {}
                        for r_arm in removal[arm]:
                            rg = (by_arm.get(r_arm) or {}).get(m) or {}
                            if rg:
                                vs[r_arm] = _paired(rg, g, lower_is_better)
                        e['vs_random_removal'] = vs
                        e['beat_every_random_seed'] = bool(vs and all(
                            v['passes_won'] * 2 > v['n_passes_paired']
                            for v in vs.values()))
                    ta = (tails.get(arm) or {}).get(m) or {}
                    tb = (tails.get('no_cleaning') or {}).get(m) or {}
                    if ta and tb:
                        e['tails'] = _tail_row(ta, tb)
                    row['per_arm'][arm] = e
                srow[m] = row
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass against no_cleaning)")
    for lvl, by_sp in (results.get('detection') or {}).items():
        for sp, by_det in by_sp.items():
            for det, row in by_det.items():
                if row:
                    print(f"  level {lvl} {sp}/{det}: AUROC "
                          f"{row['detection_auroc']:.3f}, average precision "
                          f"{row['detection_average_precision']:.3f}, over "
                          f"{row['detection_n_corrupted']:.0f} moved molecules "
                          f"of {row['detection_n_training']:.0f}")
    for lvl, lrow in s['per_level'].items():
        print(f"\n  level {lvl}:")
        for space, srow in lrow.items():
            for m, row in srow.items():
                print(f"    {space}/{m}: no_cleaning "
                      f"{row['no_cleaning_mean']:.4f}")
                for arm, e in row['per_arm'].items():
                    if '_random/' in arm:
                        continue
                    v = e['vs_no_cleaning']
                    tail = ''
                    if 'beat_every_random_seed' in e:
                        tail = (f"; beat every random seed: "
                                f"{e['beat_every_random_seed']}")
                    print(f"      {arm:28s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, mean "
                          f"{v['mean_gain']:+.4f} (spread "
                          f"{v['spread_over_pass_gains']:.4f}){tail}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N32: trimmed refits, two learners, better detectors')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'Injected condition (default '
                             f'{DEFAULT_NOISE_CONDITION}). One of: '
                             f'{", ".join(DETECTION_IS_DEFINED)}')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated levels. Default: the shared grid; '
                             'for shift_only_* the shift in label SDs')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates')
    parser.add_argument('--alphas', type=str, default=None,
                        help='Comma-separated ITLM kept shares')
    parser.add_argument('--detectors', type=str, default=None,
                        help=f'Comma-separated detectors (default '
                             f'{",".join(DETECTORS)})')
    parser.add_argument('--random-seeds', type=str, default=None,
                        help='Comma-separated random-removal replicates')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds. Overrides the 1 or 5 --mode gives')
    parser.add_argument('--base-rep', type=str, default=None,
                        help="Base representation. Default: pick_noise_base_rep's")
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    if args.exp_folds:
        config['n_splits'] = int(args.exp_folds)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {}
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.alphas:
        kw['alphas'] = [float(x) for x in args.alphas.split(',')]
    if args.detectors:
        kw['detectors'] = [x.strip() for x in args.detectors.split(',')]
    if args.random_seeds:
        kw['random_seeds'] = [int(x) for x in args.random_seeds.split(',')]
    if args.base_rep:
        kw['base_rep'] = args.base_rep

    if args.tag:
        # A tagged run is a one-off: one condition, written under the tag.
        kw.setdefault('noise_condition', DEFAULT_NOISE_CONDITION)
        res = _run_one_condition(data, config, args.output, args.dataset, **kw)
        save_experiment_results(res, args.output, EXP_ID, args.dataset,
                                filename_suffix=f"{kw['noise_condition']}_{args.tag}")
        return
    # run() writes one file per condition itself; this is the index.
    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, EXP_ID, args.dataset)


if __name__ == '__main__':
    main()
