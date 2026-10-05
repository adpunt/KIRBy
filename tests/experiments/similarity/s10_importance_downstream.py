#!/usr/bin/env python3
"""S10: Importance Scores as Feature Weights for Distance-Dependent Downstream Tasks.

Paper: Similarity
Question: does multiplying each feature by its importance help four models
that read a distance: predicting each molecule from its nearest training
molecules (kNN), a Gaussian process with an RBF kernel, a support vector
regressor with an RBF kernel, and q-RASAR (a model given four extra features
read off each molecule's nearest training molecules)?
Dependencies: S05 (the kNN case), C04 (importance weighting).

What is computed (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT.md Part 2, chat 4).

Per outer fold (sim_outer_folds, count = config['n_splits']; 1 is one held-out
split), everything is chosen on that fold's TRAINING rows by
fold_similarity_selection(..., neighbour_spaces=True), which runs the per-fold
searches on the dataset's decided grid (DECISIONS.md), pruning off. The feature
spaces ("arms") of one fold:
  single_knn1, single_knn2   knn's 2 best single representations, ranked by
                             the per-fold distance search (Part 1 step 5)
  general_<model>_1, _2      each decided model's 2 best general hybrids (a
                             second hybrid with the same selected features as
                             the first is skipped by the selection)
  distance_hybrid            knn's top hybrid of the per-fold distance search
  similarity_hybrid          the top hybrid of the per-fold similarity-objective
                             search (features chosen so distance follows
                             activity difference)
A hybrid arm is rebuilt on the fold's training rows (build_fold_entry, with the
fold search's own representation scores) at every decided budget of the search
it came from (decided_search(ds, tier)['budgets']); the budget the recipe was
searched at is one of them and is named per fold. Every arm is scaled with
FeatureScaler fitted on the training rows (0/1 features unscaled, the rest
standardised, held-out values cut to the training range).

In every (arm, budget) the four tasks are fitted twice: on the scaled features
("unweighted") and on the scaled features multiplied by the importance weights
("importance_weighted"). A hybrid is weighted by its recipe's own importance
(flatten_fold_weights); a single representation by importance computed on the
fold's training rows with SINGLE_IMPORTANCE_PREFERENCE, restricted to the
dataset's decided methods.
  weighted_knn   each held-out molecule predicted from its nearest training
                 molecules at every decided kNN setting (knn_axes: the dataset
                 group's distances, 5/10/20 neighbours, rank and inverse_square
                 counting, the scaffold group not excluded).
  weighted_gp    sklearn GaussianProcessRegressor, ConstantKernel x RBF +
                 WhiteKernel, labels centred and scaled (normalize_y), one fit
                 per starting length scale. Regression only; skipped above
                 _GP_MAX_SAMPLES molecules.
  weighted_svr   an RBF support vector regressor at gamma='scale', C=1, on
                 standardised labels (SVRScaledTarget). Regression only;
                 skipped above _SVR_MAX_SAMPLES molecules.
  qrasar         each of the dataset's two decided models fitted on the
                 features alone, and on the features plus four neighbour
                 features (the neighbours' label mean and spread under the
                 counting scheme, the distance to the nearest, the mean
                 distance), at every decided kNN setting.

Importance weighting inside this experiment's own distance (the power the
importance is raised to, KIRBY_IMPORTANCE_EXPONENT, and the rescale,
KIRBY_S10_WEIGHT_CONVENTION) is still open (handoff 2.10); both are as they were.

Every difference is stored per fold, oriented when it is computed so that a
positive number means the importance-weighted arm (or the arm with the four
neighbour features) scored better in the dataset's own metric direction
(get_domain_metric's lower_is_better). Across folds: per-fold scores, mean and
spread; no test.

Removed on 2026-10-05: every whole-dataset pool read (get_rule_hybrids,
get_best_search_for_downstream, extract_top3_base_reps, top_hybrids_across_
models over a stored search, the top2_* blocks, fold_reselection over a stored
search); the per-representation arms over every loaded representation (now
knn_base, 2 per fold); the sweep over FULL_BUDGETS (now the decided budgets of
the search each hybrid came from); KIRBY_S10_KNN_VOTE, KIRBY_S10_K_VALUES and
KIRBY_S10_QRASAR_K (now knn_axes); sklearn's euclidean neighbours and
make_nearest_neighbors (now the decided distances); StandardScaler (now
FeatureScaler); q-RASAR's fixed random forest (now the two decided models);
the in-place sign flip _orient_deltas, which reached one dict twice on rmse
datasets (dirty-run results, s10).

GP history kept because the numbers it explains are in older result files:
until 2026-09-20 the weight vector summed to 1, which divided every distance by
the feature count and made the weighted GP a constant predictor; 'mean1' (the
default, f17's measured choice) keeps a scaled feature near its own scale.
Until 2026-10-05 normalize_y was off, so a collapsed fit predicted 0 instead of
the training mean (42.81 rmse on buchwald_hartwig in the dirty run).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np

from experiments.similarity.similarity_helpers import _CODE_FAULTS
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel, Hyperparameter, RBF, WhiteKernel, ConstantKernel)

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output,
    get_domain_metric, get_model_factories, flatten_fold_weights,
)
from experiments.decided_grid import decided_models, decided_search, is_decided
from experiments.similarity.similarity_helpers import (
    sim_outer_folds, fold_spread_note, knn_axes, scale_fold, knn_lists,
    knn_sweep, fold_similarity_selection, build_fold_entry, entry_record,
)
from kirby.hybrid import compute_feature_importance, SVRScaledTarget
# Scores for the metric. On a binary dataset scored by auc this returns the
# predicted probability of class 1, as the per-fold searches score auc; every
# other metric gets model.predict (results audit, e7d9ae2).
from kirby.find_best_hybrid import _predict_for_metric


# =============================================================================
# CONSTANTS
# =============================================================================

_GP_MAX_SAMPLES = 5000
_SVR_MAX_SAMPLES = 10000

TASKS = ('weighted_knn', 'weighted_gp', 'weighted_svr', 'qrasar')
CONDITIONS = ('unweighted', 'importance_weighted')

# Bumped whenever what a checkpoint unit holds changes, so a sidecar written
# under the stored-pool layout (one unit per (hybrid space, task)) is moved
# aside instead of read back.
CHECKPOINT_VERSION = 's10_per_fold_selection_knn_axes_2026_10_05'

# Importance method for the importance_weighted single-representation arms.
# Until 2026-10-05 this was lightgbm_gain on every dataset; DECISIONS.md Step 3
# cuts lightgbm_gain on several (buchwald_hartwig, cdr_flexibility,
# ginkgo_pr_cho, ginkgo_titer among them). The first name here that the
# dataset's distance search decided is used (knn_base is ranked by that
# search); if neither is decided, the first decided method. Same rule as s05.
SINGLE_IMPORTANCE_PREFERENCE = ('lightgbm_gain', 'treeshap')

_DELTA_CONVENTION = (
    "delta = (treatment - reference) in the dataset's metric, multiplied by -1 "
    "when lower_is_better, so a positive delta means the treatment scored "
    "better. Computed once, when the number is made; nothing negates a stored "
    "delta afterwards. Per fold, and the mean over the folds both arms scored.")


def single_importance_method(dataset_name):
    """(method, tier it was read from, the decided list)."""
    tier = ('general_distance' if is_decided(dataset_name, 'general_distance')
            else 'general')
    methods = list(decided_search(dataset_name, tier)['importance_methods'])
    for pref in SINGLE_IMPORTANCE_PREFERENCE:
        if pref in methods:
            return pref, tier, methods
    return methods[0], tier, methods


def _arm_budgets(dataset_name, source):
    """The decided budgets (features) a hybrid from `source` is built at.

    source 'distance' -> the distance search's budgets; 'general' and
    'similarity' -> the general search's (the similarity-objective search runs
    on the general tier). The owner swept every hybrid over a list of budgets
    here (2026-09-11) and kept it (results audit S10e, 2026-10-05); the list is
    now the dataset's decided one (R3), not FULL_BUDGETS.
    """
    tier = 'general_distance' if source == 'distance' else 'general'
    return [int(b) for b in decided_search(dataset_name, tier)['budgets']], tier


# =============================================================================
# IMPORTANCE WEIGHTS (open, handoff 2.10: power and rescale left as they were)
# =============================================================================

def _importance_weights(imp_scores, exponent=None, convention=None):
    """Importance scores turned into the multiplier for scaled features.

    Two steps. The importance is raised to a power, and the result is rescaled.

    The power is `exponent`. None reads KIRBY_IMPORTANCE_EXPONENT through the
    shared importance_feature_weights, and an unset environment gives 0.5. 0
    makes every feature weigh the same, which is the unweighted control.

    The rescale is `convention`, read from KIRBY_S10_WEIGHT_CONVENTION when
    None: 'mean1' (default) divides by the mean weight, 'sum1' by the sum.
    Under sum-to-1 a 2048-feature representation has every distance divided by
    2048 and the GP's length scale cannot recover (2026-09-20).
    """
    from kirby.similarity.preprocessing import normalise_weights
    from experiments._shared import importance_feature_weights
    w = importance_feature_weights(imp_scores, exponent=exponent)
    conv = convention if convention is not None else _weight_convention()
    if conv == 'sum1':
        return w / (w.sum() + 1e-10)
    return normalise_weights(w, conv)


def _weight_convention():
    """'mean1' (unset) or 'sum1'. Reads KIRBY_S10_WEIGHT_CONVENTION."""
    return os.environ.get('KIRBY_S10_WEIGHT_CONVENTION', 'mean1').strip() or 'mean1'


# =============================================================================
# FOLD ARITHMETIC
# =============================================================================

def _finite(v):
    return v is not None and isinstance(v, (int, float, np.floating)) \
        and np.isfinite(v)


def _fold_stats(fold_scores):
    """Per-fold scores (None where a fold has none), their mean and spread."""
    vals = [float(v) for v in fold_scores if _finite(v)]
    out = {'fold_scores': [None if not _finite(v) else float(v)
                           for v in fold_scores],
           'n_folds_scored': len(vals),
           'mean': float(np.mean(vals)) if vals else None,
           'std': None}
    note = fold_spread_note(vals)
    if note:
        out['spread_note'] = note
    elif vals:
        out['std'] = float(np.std(vals))
    return out


def _oriented_delta(treatment, reference, lower_is_better):
    """Per-fold oriented (treatment - reference) and its mean over the folds
    both scored. Positive = treatment better (see _DELTA_CONVENTION)."""
    sign = -1.0 if lower_is_better else 1.0
    per = []
    for t, r in zip(treatment, reference):
        per.append(sign * (float(t) - float(r))
                   if _finite(t) and _finite(r) else None)
    vals = [d for d in per if d is not None]
    out = {'delta_per_fold': per, 'n_paired_folds': len(vals),
           'delta_mean': float(np.mean(vals)) if vals else None}
    note = fold_spread_note(vals)
    if note:
        out['spread_note'] = note
    return out


# =============================================================================
# TASK 1: kNN (decided settings)
# =============================================================================

def _task_knn(A, B, w, y_tr, y_va, *, axes, metric_name, metric_fn, is_clf):
    """Every decided kNN setting, on the scaled features and on the weighted
    ones. Each row carries distance, neighbour count and counting scheme.

    knn_sweep is called once per distance so that one distance failing does
    not cost the others. The case seen on synthetic 0/1 features: a held-out
    row whose weighted features are all 0 has no cosine distance to any
    training row, every neighbour weight is 0, the prediction is NaN and the
    metric raises. That distance's rows are then written with score None and
    the error.
    """
    out = {}
    for cond, (Xt, Xv) in (('unweighted', (A, B)),
                           ('importance_weighted', (A * w, B * w))):
        rows = []
        for metric in axes['metrics']:
            one = dict(axes, metrics=[metric])
            try:
                rows.extend(knn_sweep(Xt, y_tr, Xv, y_va, axes=one,
                                      metric_name=metric_name,
                                      metric_fn=metric_fn, is_clf=is_clf,
                                      scale=False))
            except Exception as e:
                err = f'{type(e).__name__}: {e}'
                rows.extend({'distance': metric, 'k': int(k), 'scheme': s,
                             'score': None, 'error': err}
                            for k in axes['ks'] for s in axes['schemes'])
        out[cond] = rows
    return out


# =============================================================================
# TASK 2: GP WITH IMPORTANCE-WEIGHTED RBF
# =============================================================================

_GP_LS_BOUNDS = (1e-5, 1e5)


def _sqdist(X, Y):
    """Squared euclidean distance between every row of X and every row of Y."""
    d = (np.einsum('ij,ij->i', X, X)[:, None]
         + np.einsum('ij,ij->i', Y, Y)[None, :] - 2.0 * (X @ Y.T))
    np.maximum(d, 0, out=d)
    return d


class _CachedRBF(Kernel):
    """The same RBF, over squared distances computed once instead of per step.

    sklearn's RBF rebuilds every pairwise distance from all of the features at
    each step of the marginal-likelihood optimiser. Only the length scale
    changes between steps, so the distances do not: this computes them once per
    training matrix and then evaluates exp(-d / 2 l^2) on them. Same kernel,
    same gradient.

    The cached entry holds the matrix it was computed from as well as the
    distances, so the array cannot be freed and another array take its id.

    _gp_kernel_equivalence() below fits both kernels on the same small matrix
    and reports the largest difference in their held-out predictions; run()
    stores that number in the result file. KIRBY_S10_GP_KERNEL=sklearn puts
    sklearn's own RBF back.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=_GP_LS_BOUNDS):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self._cache = {}

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter('length_scale', 'numeric',
                              self.length_scale_bounds)

    def _d2(self, X):
        key = id(X)
        hit = self._cache.get(key)
        if hit is None or hit[0] is not X:
            hit = (X, _sqdist(X, X))
            self._cache[key] = hit
        return hit[1]

    def __call__(self, X, Y=None, eval_gradient=False):
        ls = float(np.squeeze(self.length_scale))
        d2 = self._d2(X) if Y is None else _sqdist(X, Y)
        K = np.exp(-0.5 * d2 / (ls * ls))
        if not eval_gradient:
            return K
        if Y is not None:
            raise ValueError('gradient is only defined when Y is None')
        return K, (K * (d2 / (ls * ls)))[:, :, np.newaxis]

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True


def _gp_kernel_in_use():
    """'cached' (the default) or 'sklearn'. Reads KIRBY_S10_GP_KERNEL."""
    return os.environ.get('KIRBY_S10_GP_KERNEL', 'cached').strip() or 'cached'


def _gp_kernel_equivalence():
    """Largest difference in prediction between the two kernels, 20 held-out rows.

    Fits both on 60 rows of 8 gaussian features and predicts 20 more. A number
    at the size of floating-point rounding is what says the cached kernel
    changed nothing; anything larger means it did.
    """
    try:
        r = np.random.default_rng(0)
        X, y = r.normal(size=(60, 8)), r.normal(size=60)
        X_test = r.normal(size=(20, 8))
        preds = []
        for K in (RBF(1.0), _CachedRBF(1.0)):
            g = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * K + WhiteKernel(1e-3),
                n_restarts_optimizer=0, alpha=1e-6, random_state=42,
                normalize_y=True)
            g.fit(X, y)
            preds.append(g.predict(X_test))
        return float(np.abs(preds[0] - preds[1]).max())
    except Exception as e:
        return f'{type(e).__name__}: {e}'


_GP_FIT_BASIS = (
    "standard_fit / weighted_fit = per fold: the fitted RBF length scale, in "
    "the units of the scaled (and weighted) features; at_bound, meaning it "
    "finished at an end of its bounds so the marginal-likelihood optimiser ran "
    "out of room rather than found an optimum; and pred_sd, the spread of that "
    "arm's own predictions over the held-out molecules, which is 0 for a "
    "constant predictor. At the top bound the kernel is constant and every "
    "held-out molecule gets the training mean (normalize_y centres the "
    "labels). A fold where either arm is at a bound or constant fitted nothing "
    "in that arm, so delta_both_fitted leaves it out; delta keeps every fold.")


def _gp_normalize_y():
    """Whether the GP centres and scales the labels before fitting.

    True (unset) puts the prior mean at the training mean, so a fit with no
    information predicts the training mean. False leaves the prior mean at 0,
    and a collapsed fit predicts 0 for every held-out molecule (42.81 rmse on
    buchwald_hartwig in the dirty run). Default True since 2026-10-05 (owner
    delegated, 6056c8d). KIRBY_S10_GP_NORMALIZE_Y=0 fits the old one.
    """
    raw = os.environ.get('KIRBY_S10_GP_NORMALIZE_Y', '').strip().lower()
    return raw not in ('0', 'false', 'no', 'off')


def _gp_lengthscale_starts():
    """Where the RBF's length scale starts, one arm each.

    'fixed' is 1.0 with bounds 1e-5 to 1e5. 'median' starts at the median
    euclidean distance between two training molecules in the features the arm
    is fitted on, with bounds that value divided and multiplied by 100 (the
    same convention as s11's KIRBY_S11_LENGTHSCALE). Each arm takes the median
    of ITS OWN training features. Reads KIRBY_S10_GP_LENGTHSCALE as a
    comma-separated list; unset gives both.
    """
    raw = os.environ.get('KIRBY_S10_GP_LENGTHSCALE', '').strip()
    if not raw:
        return ['fixed', 'median']
    return [v.strip() for v in raw.split(',') if v.strip()]


def _gp_restarts():
    """Extra starts of the marginal-likelihood optimiser. KIRBY_S10_GP_RESTARTS,
    unset 0: both arms got the same starts, so they cancelled out of the
    difference this task reports while tripling the Cholesky work."""
    return int(os.environ.get('KIRBY_S10_GP_RESTARTS', '').strip() or '0')


def _median_pair_distance(X):
    """Median euclidean distance between two rows of X, or None.

    Zero distances are dropped: two identical molecules carry no scale.
    """
    try:
        A = np.asarray(X, dtype=np.float64)
        if A.shape[0] < 3:
            return None
        d2 = _sqdist(A, A)
        iu = np.triu_indices(d2.shape[0], k=1)
        d = np.sqrt(d2[iu])
        d = d[d > 0]
        return float(np.median(d)) if d.size else None
    except Exception:
        return None


def _make_gp(X=None, start='fixed'):
    ls, bounds = 1.0, _GP_LS_BOUNDS
    if start == 'median' and X is not None:
        _m = _median_pair_distance(X)
        if _m:
            ls, bounds = _m, (_m / 100.0, _m * 100.0)
    _rbf = (RBF(length_scale=ls, length_scale_bounds=bounds)
            if _gp_kernel_in_use() == 'sklearn'
            else _CachedRBF(length_scale=ls, length_scale_bounds=bounds))
    kernel = ConstantKernel(1.0) * _rbf + WhiteKernel(1e-3)
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=_gp_restarts(),
        alpha=1e-6, random_state=42, normalize_y=_gp_normalize_y())


def _gp_fit_report(gp, preds, start=None, start_value=None):
    """What the GP settled on, so a degenerate fit is recorded, not just scored.
    'at_bound' is read against the bounds the fitted kernel actually carries."""
    try:
        ls = float(np.squeeze(gp.kernel_.k1.k2.length_scale))
    except Exception:
        return None
    try:
        lo, hi = (float(v) for v in
                  np.ravel(gp.kernel_.k1.k2.length_scale_bounds)[:2])
    except Exception:
        lo, hi = _GP_LS_BOUNDS
    sd = float(np.std(preds))
    return {'length_scale': ls,
            'length_scale_bounds': [lo, hi],
            'lengthscale_start': start,
            'lengthscale_start_value': start_value,
            'at_bound': bool(ls <= lo * 1.01 or ls >= hi * 0.99),
            'pred_sd': sd,
            'constant_predictor': bool(sd < 1e-6)}


def _gp_one(X_tr, y_tr, X_va, y_va, start, metric_fn):
    """(score, fit record, error) for one GP fit."""
    try:
        sv = _median_pair_distance(X_tr) if start == 'median' else 1.0
        gp = _make_gp(X_tr, start)
        gp.fit(X_tr, y_tr)
        p = gp.predict(X_va)
        return float(metric_fn(y_va, p)), _gp_fit_report(gp, p, start, sv), None
    except Exception as e:
        return None, None, f'{type(e).__name__}: {e}'


def _task_gp(A, B, w, y_tr, y_va, *, metric_fn):
    """One standard and one weighted GP per starting length scale."""
    out = {}
    for st in _gp_lengthscale_starts():
        s_sc, s_fit, s_err = _gp_one(A, y_tr, B, y_va, st, metric_fn)
        w_sc, w_fit, w_err = _gp_one(A * w, y_tr, B * w, y_va, st, metric_fn)
        out[st] = {'unweighted': s_sc, 'importance_weighted': w_sc,
                   'unweighted_fit': s_fit, 'importance_weighted_fit': w_fit}
        if s_err:
            out[st]['unweighted_error'] = s_err
        if w_err:
            out[st]['importance_weighted_error'] = w_err
    return out


# =============================================================================
# TASK 3: SVR WITH IMPORTANCE-WEIGHTED RBF
# =============================================================================

def _make_svr():
    """RBF support vector regressor at C=1, gamma='scale', on standardised
    labels. SVRScaledTarget is the regressor the searches fit as 'svm' since
    fb1df54; a plain SVR at C=1 and epsilon=0.1 on raw labels cannot fit a
    label spread far from 1 (dirty-run bug 4: buchwald yields, sd 27.3)."""
    return SVRScaledTarget(kernel='rbf', C=1.0, gamma='scale')


def _task_svr(A, B, w, y_tr, y_va, *, metric_fn):
    out = {}
    for cond, (Xt, Xv) in (('unweighted', (A, B)),
                           ('importance_weighted', (A * w, B * w))):
        try:
            m = _make_svr()
            m.fit(Xt, y_tr)
            out[cond] = float(metric_fn(y_va, m.predict(Xv)))
        except Exception as e:
            out[cond] = None
            out[f'{cond}_error'] = f'{type(e).__name__}: {e}'
    return out


# =============================================================================
# TASK 4: SIMILARITY-AUGMENTED FEATURES (q-RASAR STYLE)
# =============================================================================

def _label_numbers(y):
    """The labels as numbers for the neighbours' mean and spread. A class label
    that is not numeric is replaced by its index among the sorted classes."""
    y = np.asarray(y)
    try:
        return y.astype(np.float64)
    except (TypeError, ValueError):
        return np.unique(y, return_inverse=True)[1].astype(np.float64)


def _sim_descriptors(y_num, idx, dist, *, k, scheme, fill):
    """Four q-RASAR features per query row from its neighbour list cut to k:
    the neighbours' label mean and spread, both weighted by the counting
    scheme; the distance to the nearest; the mean distance to the k.

    A distance that is not finite (cosine on a row with no spread) is replaced
    by `fill`, the largest finite training-row distance under that distance.
    """
    from experiments.noise import similarity_space as ss
    i = np.asarray(idx)[:, :k]
    d = np.asarray(dist, dtype=np.float64)[:, :k]
    d = np.where(np.isfinite(d), d, fill)
    wts = ss.neighbour_weights(d, scheme)
    tot = wts.sum(1, keepdims=True)
    wts = np.divide(wts, tot, out=np.full_like(wts, 1.0 / k), where=tot > 0)
    nb = y_num[i]
    mean = (wts * nb).sum(1)
    sd = np.sqrt(np.maximum((wts * (nb - mean[:, None]) ** 2).sum(1), 0.0))
    return np.column_stack([mean, sd, d[:, 0], d.mean(1)])


def _task_qrasar(A, B, w, y_tr, y_va, *, axes, factories, metric_fn):
    """Per decided model: the features alone, and the features plus the four
    neighbour features under each decided kNN setting, unweighted and weighted.

    The neighbour lists are measured in the same matrix the model is given:
    the scaled features (unweighted) or the scaled features times the weights
    (importance_weighted). Training rows' own neighbour lists exclude the row
    itself by index (dirty-run bug 7). The four features are scaled with
    FeatureScaler fitted on the training rows before they are appended.
    """
    y_num = _label_numbers(y_tr)
    out = {'base_only': {}, 'rows': []}
    for m, fac in factories.items():
        try:
            mdl = fac()
            mdl.fit(A, y_tr)
            out['base_only'][m] = float(metric_fn(
                y_va, _predict_for_metric(mdl, B, metric_fn)))
        except Exception as e:
            out['base_only'][m] = None
            out.setdefault('errors', []).append(
                {'model': m, 'arm': 'base_only',
                 'error': f'{type(e).__name__}: {e}'})
    for cond, (Xt, Xv) in (('unweighted', (A, B)),
                           ('importance_weighted', (A * w, B * w))):
        for metric in axes['metrics']:
            try:
                i_tr, d_tr = knn_lists(Xt, None, metric=metric,
                                       k_max=axes['k_max'])
                i_va, d_va = knn_lists(Xt, Xv, metric=metric,
                                       k_max=axes['k_max'])
            except Exception as e:
                out.setdefault('errors', []).append(
                    {'condition': cond, 'distance': metric,
                     'error': f'{type(e).__name__}: {e}'})
                continue
            fin = d_tr[np.isfinite(d_tr)]
            fill = float(fin.max()) if fin.size else 0.0
            for k in axes['ks']:
                if k > i_tr.shape[1] or k > i_va.shape[1]:
                    for scheme in axes['schemes']:
                        out['rows'].append({
                            'condition': cond, 'distance': metric, 'k': int(k),
                            'scheme': scheme, 'scores': {},
                            'note': f'only {i_tr.shape[1]} training neighbours'})
                    continue
                for scheme in axes['schemes']:
                    row = {'condition': cond, 'distance': metric, 'k': int(k),
                           'scheme': scheme, 'scores': {}}
                    try:
                        D_tr = _sim_descriptors(y_num, i_tr, d_tr, k=k,
                                                scheme=scheme, fill=fill)
                        D_va = _sim_descriptors(y_num, i_va, d_va, k=k,
                                                scheme=scheme, fill=fill)
                        D_tr, D_va = scale_fold(D_tr, D_va)
                        X_aug_tr = np.hstack([Xt, D_tr])
                        X_aug_va = np.hstack([Xv, D_va])
                    except Exception as e:
                        row['error'] = f'{type(e).__name__}: {e}'
                        out['rows'].append(row)
                        continue
                    for m, fac in factories.items():
                        try:
                            mdl = fac()
                            mdl.fit(X_aug_tr, y_tr)
                            row['scores'][m] = float(metric_fn(
                                y_va, _predict_for_metric(mdl, X_aug_va,
                                                          metric_fn)))
                        except Exception as e:
                            row['scores'][m] = None
                            row.setdefault('errors', {})[m] = (
                                f'{type(e).__name__}: {e}')
                    out['rows'].append(row)
    return out


# =============================================================================
# ONE FEATURE SPACE: all four tasks
# =============================================================================

def _score_space(A, B, w, y_tr, y_va, *, axes, metric_name, metric_fn,
                 is_clf, n_samples, qrasar_factories):
    out = {'width_built': int(A.shape[1]), 'tasks': {}}
    t0 = time.time()
    out['tasks']['weighted_knn'] = _task_knn(
        A, B, w, y_tr, y_va, axes=axes, metric_name=metric_name,
        metric_fn=metric_fn, is_clf=is_clf)
    if is_clf:
        out['tasks']['weighted_gp'] = {'skipped': True,
                                       'reason': 'classification (regressor)'}
        out['tasks']['weighted_svr'] = {'skipped': True,
                                        'reason': 'classification (regressor)'}
    else:
        if n_samples > _GP_MAX_SAMPLES:
            out['tasks']['weighted_gp'] = {
                'skipped': True,
                'reason': f'{n_samples} molecules > {_GP_MAX_SAMPLES}'}
        else:
            out['tasks']['weighted_gp'] = _task_gp(A, B, w, y_tr, y_va,
                                                   metric_fn=metric_fn)
        if n_samples > _SVR_MAX_SAMPLES:
            out['tasks']['weighted_svr'] = {
                'skipped': True,
                'reason': f'{n_samples} molecules > {_SVR_MAX_SAMPLES}'}
        else:
            out['tasks']['weighted_svr'] = _task_svr(A, B, w, y_tr, y_va,
                                                     metric_fn=metric_fn)
    out['tasks']['qrasar'] = _task_qrasar(
        A, B, w, y_tr, y_va, axes=axes, factories=qrasar_factories,
        metric_fn=metric_fn)
    out['seconds'] = round(time.time() - t0, 1)
    return out


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Importance weights against no weights, in each per-fold feature space,
    for kNN, a Gaussian process, an RBF support vector regressor and q-RASAR."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S10: IMPORTANCE WEIGHTS FOR DOWNSTREAM TASKS")
    print("=" * 70)

    if kwargs.get('budget') is not None:
        print(f"  --budget {kwargs['budget']} is ignored: each hybrid is built "
              f"at the decided budgets of the search it came from")

    reps = data['reps']
    labels = np.asarray(data['labels'])
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    n_samples = len(labels)
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    axes = knn_axes(dataset_name, n_samples)
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    imp_method, imp_tier, imp_decided = single_importance_method(dataset_name)

    factories = {}
    for m in list(models) + ['knn']:
        try:
            f = get_model_factories([m], task_type=task_type,
                                    n_samples=n_samples).get(m)
        except Exception:
            f = None
        factories[m] = f
    qrasar_factories = {m: factories[m] for m in models if factories.get(m)}
    qrasar_missing = {m: 'no model factory at this molecule count'
                      for m in models if not factories.get(m)}

    folds = sim_outer_folds(data, config)
    n_folds = len(folds)

    print(f"  metric={metric_name} (lower_is_better={lower_is_better}), "
          f"{n_samples} molecules, {n_folds} outer folds, models={models}")
    print(f"  kNN settings ({axes['group']}): distances={axes['metrics']}, "
          f"neighbours={axes['ks']}, counting={axes['schemes']}, scaffold "
          f"group not excluded")
    print(f"  single-representation importance: {imp_method} "
          f"(decided on the {imp_tier} search: {imp_decided})")

    results = {
        'selection_scope': 'fold_train_rows',
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'n_outer_folds': n_folds,
        'models': models,
        'qrasar_models': sorted(qrasar_factories),
        'qrasar_models_missing': qrasar_missing,
        'knn_settings': dict(axes),
        'conditions': list(CONDITIONS),
        'tasks': list(TASKS),
        'delta_convention': _DELTA_CONVENTION,
        'gp_fit_basis': _GP_FIT_BASIS,
        'single_importance': {
            'method': imp_method, 'read_from_tier': imp_tier,
            'decided_methods': imp_decided,
            'preference': list(SINGLE_IMPORTANCE_PREFERENCE),
            'note': ('importance method for the importance_weighted single '
                     'representation arms; a hybrid arm is weighted by its own '
                     "recipe's importance"),
        },
        'per_fold': {},
    }

    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's10_importance_downstream',
                           dataset_name),
        run_identity('s10_importance_downstream', dataset_name, data, config,
                     version=CHECKPOINT_VERSION,
                     knn_axes={k: v for k, v in axes.items() if k != 'source'},
                     single_importance=imp_method,
                     weight_convention=_weight_convention(),
                     gp_normalize_y=_gp_normalize_y(),
                     gp_lengthscale_starts=_gp_lengthscale_starts(),
                     gp_kernel=_gp_kernel_in_use(),
                     gp_restarts=_gp_restarts()))

    t_total = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  Fold {fold_i} ({len(train_idx)} training rows, "
              f"{len(val_idx)} held-out rows)")
        results['per_fold'][fold_i] = _run_fold(
            data, dataset_name, fold_i, n_folds, train_idx, val_idx,
            models=models, axes=axes, factories=factories,
            qrasar_factories=qrasar_factories, imp_method=imp_method,
            metric_name=metric_name, metric_fn=metric_fn,
            lower_is_better=lower_is_better, is_clf=is_clf,
            n_samples=n_samples, ck=_ck)

    results['summary'] = _summarise(results['per_fold'], n_folds,
                                    lower_is_better)
    results['arm_reasons_missing'] = _missing_reasons(results['per_fold'])

    from kirby.similarity.preprocessing import importance_exponent_override
    _exp = importance_exponent_override()
    results['weighting_settings'] = {
        'importance_exponent': 0.5 if _exp is None else _exp,
        'weight_convention': _weight_convention(),
        'gp_restarts': _gp_restarts(),
        'gp_kernel': _gp_kernel_in_use(),
        'gp_lengthscale_starts': _gp_lengthscale_starts(),
        'gp_normalize_y': _gp_normalize_y(),
        'gp_kernel_largest_prediction_difference': _gp_kernel_equivalence(),
        'open': ('the power and the rescale of the importance weights are '
                 'open (handoff 2.10); left as they were'),
        'basis': (
            'Features are scaled with FeatureScaler on the training rows, '
            'then multiplied by the importance raised to importance_exponent '
            'and rescaled by weight_convention (mean1: divided by the mean '
            'weight; sum1: by the sum).'),
    }

    print(f"\n  Total elapsed: {time.time() - t_total:.1f}s")
    _print_summary(results['summary'], metric_name)

    # The saved config says what ran: the decided grid of each search kind,
    # and the kNN settings, in place of the engine defaults the shared config
    # carries.
    config = dict(config)
    _searched = {}
    for _kind in ('general', 'general_distance'):
        try:
            _searched[_kind] = decided_search(dataset_name, _kind)
        except Exception as _e:
            _searched[_kind] = {'error': f'{type(_e).__name__}: {_e}'}
    config['searched_grid'] = _searched
    config['allocations'] = {k: g.get('allocation_methods')
                             for k, g in _searched.items()}
    config['importance_methods'] = {k: g.get('importance_methods')
                                    for k, g in _searched.items()}
    config['budgets'] = {k: g.get('budgets') for k, g in _searched.items()}
    config['models'] = models
    config['k_values'] = list(axes['ks'])
    config['knn_settings'] = {k: v for k, v in axes.items() if k != 'source'}
    config['config_note'] = (
        "searched_grid, allocations, importance_methods and budgets are the "
        "dataset's decided grid per search kind; every per-fold search ran "
        "on it. k_values and knn_settings are the decided kNN settings.")

    return make_structured_output(
        's10_importance_downstream', dataset_name, data, config, results,
        primary_metric=metric_name, lower_is_better=lower_is_better,
    )


def _fold_arms(sel, rec, models, dataset_name):
    """The arms this fold should have: (name, kind, payload, model, source)."""
    arms = []
    for i in range(2):
        name = f'single_knn{i + 1}'
        if i < len(rec['knn_base']):
            arms.append((name, 'single', rec['knn_base'][i], None, None))
        else:
            rec['arms_missing'][name] = (
                rec['unavailable'].get('knn_base')
                or rec['unavailable'].get('distance_search')
                or 'the per-fold distance search gave knn fewer than '
                   f'{i + 1} single representations')
    for m in models:
        s = (sel.get('models') or {}).get(m)
        hyb = (s or {}).get('hybrids') or []
        for j in range(2):
            name = f'general_{m}_{j + 1}'
            if j < len(hyb):
                arms.append((name, 'hybrid', hyb[j], m, 'general'))
            else:
                rec['arms_missing'][name] = (
                    rec['unavailable'].get(f'general/{m}')
                    or f'the per-fold general search gave {m} fewer than '
                       f'{j + 1} hybrids with distinct selected features')
    for name, key, model, source in (
            ('distance_hybrid', 'distance_hybrid', 'knn', 'distance'),
            ('similarity_hybrid', 'similarity_hybrid', None, 'similarity')):
        if sel.get(key) is not None:
            arms.append((name, 'hybrid', sel[key], model, source))
        else:
            rec['arms_missing'][name] = (
                rec['unavailable'].get(key)
                or rec['unavailable'].get('distance_search')
                or 'not selected')
    return arms


def _run_fold(data, dataset_name, fold_i, n_folds, train_idx, val_idx, *,
              models, axes, factories, qrasar_factories, imp_method,
              metric_name, metric_fn, lower_is_better, is_clf, n_samples, ck):
    """One outer fold: select on the training rows, then score every arm at
    every budget on the held-out rows. One checkpoint unit per (fold, arm)."""
    reps = data['reps']
    labels = np.asarray(data['labels'])
    y_tr, y_va = labels[train_idx], labels[val_idx]

    rec = {
        'fold': int(fold_i),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'selection_scope': 'fold_train_rows',
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'models': {},
        'knn_base': [],
        'knn_base_train_scores': {},
        'distance_hybrid': None,
        'similarity_hybrid': None,
        'unavailable': {},
        'arms': {},
        'arms_missing': {},
    }

    # Selection on this fold's training rows. The searches are in the result
    # store after the first attempt, so a resumed run re-reads them.
    try:
        sel = fold_similarity_selection(
            data, dataset_name, train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_folds, neighbour_spaces=True)
    except _CODE_FAULTS:
        raise  # a fault in the code, not a failed search
    except Exception as exc:
        rec['selection_error'] = f'{type(exc).__name__}: {exc}'
        print(f"    [warn] selection raised {rec['selection_error']}")
        return rec

    rec['unavailable'] = dict(sel.get('unavailable') or {})
    for m in models:
        s = (sel.get('models') or {}).get(m)
        if s is None:
            rec['models'][m] = None
            continue
        bd = s.get('base_detail') or {}
        rec['models'][m] = {
            'base': list(s.get('base') or []),
            'base_train_scores': {e['rep']: e.get('score')
                                  for e in (bd.get('selected') or [])},
            'hybrids': [entry_record(h) for h in (s.get('hybrids') or [])],
            'n_hybrids_skipped_identical_features': len(
                s.get('hybrids_skipped_identical_features') or []),
        }
    rec['knn_base'] = list(sel.get('knn_base') or [])
    rec['knn_base_train_scores'] = {
        e['rep']: e.get('score')
        for e in ((sel.get('knn_base_detail') or {}).get('selected') or [])}
    rec['distance_hybrid'] = entry_record(sel.get('distance_hybrid'))
    rec['similarity_hybrid'] = entry_record(sel.get('similarity_hybrid'))

    for name, kind, payload, model, source in _fold_arms(sel, rec, models,
                                                         dataset_name):
        if ck.done(fold=fold_i, arm=name):
            rec['arms'][name] = ck.get(fold=fold_i, arm=name)
            print(f"    {name}: read back from the partial file")
            continue
        arm = {'kind': kind, 'widths': {}}
        threw = False
        try:
            if kind == 'single':
                arm['representation'] = payload
                X = np.asarray(reps[payload])
                A, B = scale_fold(X[train_idx], X[val_idx])
                imp = compute_feature_importance(
                    {payload: X[train_idx]}, y_tr, method=imp_method)
                w = np.asarray(_importance_weights(imp[payload]),
                               dtype=np.float64)
                arm['importance_method'] = imp_method
                if w.shape[0] != A.shape[1]:
                    raise ValueError(f'weight/feature mismatch: w={w.shape[0]} '
                                     f'vs X={A.shape[1]}')
                arm['widths']['all'] = _score_space(
                    A, B, w, y_tr, y_va, axes=axes, metric_name=metric_name,
                    metric_fn=metric_fn, is_clf=is_clf, n_samples=n_samples,
                    qrasar_factories=qrasar_factories)
                arm['recipe_width_key'] = 'all'
            else:
                arm['hybrid'] = entry_record(payload)
                arm['selected_for_model'] = model
                arm['source_search'] = source
                arm['importance_method'] = arm['hybrid'].get('importance')
                budgets, tier = _arm_budgets(dataset_name, source)
                recipe_b = arm['hybrid'].get('budget_features')
                recipe_b = int(recipe_b) if recipe_b is not None else None
                if recipe_b is not None and recipe_b not in budgets:
                    budgets = sorted(set(budgets) | {recipe_b})
                arm['budgets'] = budgets
                arm['budgets_tier'] = tier
                arm['recipe_width_key'] = (str(recipe_b) if recipe_b is not None
                                           else None)
                for b in budgets:
                    key = str(b)
                    try:
                        # The recipe's own budget is rebuilt exactly as
                        # searched (no override); every other budget applies
                        # the same recipe at that many features.
                        X_tr_h, X_va_h, feat_info = build_fold_entry(
                            data, payload, train_idx=train_idx,
                            val_idx=val_idx,
                            model_factory=factories.get(model) if model else None,
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better,
                            budget_override=(None if b == recipe_b else b))
                        A, B = scale_fold(X_tr_h, X_va_h)
                        w = np.asarray(_importance_weights(
                            flatten_fold_weights(feat_info)), dtype=np.float64)
                        if w.shape[0] != A.shape[1]:
                            raise ValueError(
                                f'weight/feature mismatch: w={w.shape[0]} '
                                f'vs X={A.shape[1]}')
                        arm['widths'][key] = _score_space(
                            A, B, w, y_tr, y_va, axes=axes,
                            metric_name=metric_name, metric_fn=metric_fn,
                            is_clf=is_clf, n_samples=n_samples,
                            qrasar_factories=qrasar_factories)
                        arm['widths'][key]['budget_features'] = b
                        arm['widths'][key]['is_recipe_budget'] = (b == recipe_b)
                    except Exception as exc:
                        threw = True
                        arm['widths'][key] = {
                            'budget_features': b,
                            'error': f'{type(exc).__name__}: {exc}'}
                        print(f"    [warn] {name} at {b} features: "
                              f"{arm['widths'][key]['error'][:120]}")
        except Exception as exc:
            threw = True
            arm['error'] = f'{type(exc).__name__}: {exc}'
            print(f"    [warn] {name} failed: {arm['error'][:120]}")
        _print_arm(name, arm, metric_name)
        rec['arms'][name] = arm
        if not threw:
            ck.put(arm, fold=fold_i, arm=name)

    for name, why in rec['arms_missing'].items():
        print(f"    {name}: not available ({why})")
    return rec


def _print_arm(name, arm, metric_name):
    """One line per (arm, budget): best kNN setting in each condition."""
    for key, sp in (arm.get('widths') or {}).items():
        if 'tasks' not in sp:
            continue
        kn = sp['tasks'].get('weighted_knn') or {}
        parts = []
        for cond in CONDITIONS:
            ok = [r for r in kn.get(cond) or [] if _finite(r.get('score'))]
            if ok:
                parts.append(f"{cond} {len(ok)} settings scored")
        label = arm.get('representation') or f'{key} features'
        print(f"    {name:<22} {label:<16} width={sp['width_built']:>5} "
              f"({sp.get('seconds', 0):.0f}s): " + '; '.join(parts))


# =============================================================================
# ACROSS FOLDS
# =============================================================================

def _missing_reasons(per_fold):
    reasons = {}
    for rec in per_fold.values():
        if rec.get('selection_error'):
            reasons.setdefault('selection', []).append(rec['selection_error'])
        for arm, why in (rec.get('arms_missing') or {}).items():
            reasons.setdefault(arm, [])
            if why not in reasons[arm]:
                reasons[arm].append(why)
        for arm, a in (rec.get('arms') or {}).items():
            errs = [a.get('error')] + [w.get('error') for w in
                                       (a.get('widths') or {}).values()]
            for e in errs:
                if e:
                    reasons.setdefault(arm, [])
                    if e not in reasons[arm]:
                        reasons[arm].append(e)
    return reasons


def _collect(per_fold, n_folds):
    """{(arm, width key): [space dict or None per fold]}, plus the recipe's own
    budget per fold under width key 'recipe' for hybrid arms."""
    spaces = {}
    for fold_i, rec in per_fold.items():
        fi = int(fold_i)
        for arm, a in (rec.get('arms') or {}).items():
            for key, sp in (a.get('widths') or {}).items():
                if 'tasks' not in sp:
                    continue
                spaces.setdefault((arm, key), [None] * n_folds)[fi] = sp
            rk = a.get('recipe_width_key')
            if a.get('kind') == 'hybrid' and rk and \
                    'tasks' in ((a.get('widths') or {}).get(rk) or {}):
                spaces.setdefault((arm, 'recipe'), [None] * n_folds)[fi] = \
                    a['widths'][rk]
    return spaces


def _summarise(per_fold, n_folds, lower_is_better):
    """Per (arm, budget, task, setting): per-fold held-out scores of both
    conditions, their means and spreads, and the oriented difference."""
    out = {}
    for (arm, key), sps in _collect(per_fold, n_folds).items():
        blk = out.setdefault(arm, {}).setdefault(key, {})
        blk['width_built_per_fold'] = [None if s is None else s['width_built']
                                       for s in sps]
        tasks = [None if s is None else s['tasks'] for s in sps]

        # kNN: one cell per (distance, neighbours, counting scheme).
        knn = {}
        for fi, t in enumerate(tasks):
            for cond in CONDITIONS:
                for r in ((t or {}).get('weighted_knn') or {}).get(cond) or []:
                    c = knn.setdefault(
                        f"{r['distance']}|k{r['k']}|{r['scheme']}",
                        {'distance': r['distance'], 'k': r['k'],
                         'scheme': r['scheme'],
                         'unweighted': [None] * n_folds,
                         'importance_weighted': [None] * n_folds})
                    c[cond][fi] = r.get('score')
        for c in knn.values():
            uw, iw = c['unweighted'], c['importance_weighted']
            c['unweighted'] = _fold_stats(uw)
            c['importance_weighted'] = _fold_stats(iw)
            c['weighted_vs_unweighted'] = _oriented_delta(iw, uw,
                                                          lower_is_better)
        blk['weighted_knn'] = {
            'cells': knn,
            'heldout_repick': _knn_heldout_repick(knn, n_folds,
                                                  lower_is_better)}

        # GP: per starting length scale.
        gp = {}
        skipped = next((t['weighted_gp'] for t in tasks if t and
                        (t.get('weighted_gp') or {}).get('skipped')), None)
        if skipped:
            gp = dict(skipped)
        else:
            for fi, t in enumerate(tasks):
                for st, g in ((t or {}).get('weighted_gp') or {}).items():
                    c = gp.setdefault(st, {
                        'unweighted': [None] * n_folds,
                        'importance_weighted': [None] * n_folds,
                        'unweighted_fit': [None] * n_folds,
                        'importance_weighted_fit': [None] * n_folds})
                    for f in ('unweighted', 'importance_weighted',
                              'unweighted_fit', 'importance_weighted_fit'):
                        c[f][fi] = g.get(f)
            for st, c in gp.items():
                uw, iw = c['unweighted'], c['importance_weighted']
                # A fold where either arm finished at a bound or predicted one
                # number for every held-out molecule is not a measurement of
                # weighting; delta_both_fitted leaves those folds out.
                fitted = [
                    bool(a and b and not a['at_bound'] and not b['at_bound']
                         and not a['constant_predictor']
                         and not b['constant_predictor'])
                    for a, b in zip(c['unweighted_fit'],
                                    c['importance_weighted_fit'])]
                c['n_folds_either_arm_at_bound_or_constant'] = sum(
                    1 for a, b, ok in zip(uw, iw, fitted)
                    if not ok and (a is not None or b is not None))
                c['both_fitted_per_fold'] = fitted
                c['delta'] = _oriented_delta(iw, uw, lower_is_better)
                c['delta_both_fitted'] = _oriented_delta(
                    [v if ok else None for v, ok in zip(iw, fitted)],
                    [v if ok else None for v, ok in zip(uw, fitted)],
                    lower_is_better)
                c['unweighted'] = _fold_stats(uw)
                c['importance_weighted'] = _fold_stats(iw)
        blk['weighted_gp'] = gp

        # SVR.
        skipped = next((t['weighted_svr'] for t in tasks if t and
                        (t.get('weighted_svr') or {}).get('skipped')), None)
        if skipped:
            blk['weighted_svr'] = dict(skipped)
        else:
            uw = [None if t is None else (t.get('weighted_svr') or {})
                  .get('unweighted') for t in tasks]
            iw = [None if t is None else (t.get('weighted_svr') or {})
                  .get('importance_weighted') for t in tasks]
            blk['weighted_svr'] = {
                'unweighted': _fold_stats(uw),
                'importance_weighted': _fold_stats(iw),
                'delta': _oriented_delta(iw, uw, lower_is_better)}

        # q-RASAR: per model, the features alone and per setting.
        qr = {}
        for fi, t in enumerate(tasks):
            q = (t or {}).get('qrasar') or {}
            for m, sc in (q.get('base_only') or {}).items():
                qr.setdefault(m, {'base_only': [None] * n_folds,
                                  'cells': {}})['base_only'][fi] = sc
            for r in q.get('rows') or []:
                for m, sc in (r.get('scores') or {}).items():
                    mm = qr.setdefault(m, {'base_only': [None] * n_folds,
                                           'cells': {}})
                    c = mm['cells'].setdefault(
                        f"{r['distance']}|k{r['k']}|{r['scheme']}",
                        {'distance': r['distance'], 'k': r['k'],
                         'scheme': r['scheme'],
                         'unweighted': [None] * n_folds,
                         'importance_weighted': [None] * n_folds})
                    c[r['condition']][fi] = sc
        for m, mm in qr.items():
            base = mm['base_only']
            for c in mm['cells'].values():
                uw, iw = c['unweighted'], c['importance_weighted']
                c['unweighted_vs_base_only'] = _oriented_delta(
                    uw, base, lower_is_better)
                c['weighted_vs_base_only'] = _oriented_delta(
                    iw, base, lower_is_better)
                c['weighted_vs_unweighted'] = _oriented_delta(
                    iw, uw, lower_is_better)
                c['unweighted'] = _fold_stats(uw)
                c['importance_weighted'] = _fold_stats(iw)
            mm['base_only'] = _fold_stats(base)
        blk['qrasar'] = qr
    return out


def _knn_heldout_repick(cells, n_folds, lower_is_better):
    """For each held-out fold, the kNN setting with the best mean over the
    OTHER folds is picked separately in each condition and scored on the
    held-out fold; the per-fold difference is weighted minus unweighted
    (oriented). Undefined with one fold."""
    if n_folds < 2:
        return {'note': 'one fold, no other folds to pick a setting on'}
    better = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)

    def pick(cond, f):
        best, best_v = None, None
        for name, c in cells.items():
            vals = [v for j, v in enumerate(c[cond]['fold_scores'])
                    if j != f and _finite(v)]
            if not vals:
                continue
            m = float(np.mean(vals))
            if best_v is None or better(m, best_v):
                best, best_v = name, m
        return best

    per = []
    for f in range(n_folds):
        pu, pw = pick('unweighted', f), pick('importance_weighted', f)
        su = cells[pu]['unweighted']['fold_scores'][f] if pu else None
        sw = cells[pw]['importance_weighted']['fold_scores'][f] if pw else None
        per.append({'fold': f, 'unweighted_setting': pu,
                    'importance_weighted_setting': pw,
                    'unweighted': su, 'importance_weighted': sw})
    d = _oriented_delta([p['importance_weighted'] for p in per],
                        [p['unweighted'] for p in per], lower_is_better)
    return {'per_fold': per, **d}


def _print_summary(summary, metric_name):
    """One line per (arm, budget): the held-out re-pick of the kNN difference,
    the GP and SVR differences, all oriented (positive = weighting better)."""
    print(f"\n  {'=' * 60}")
    print(f"  SUMMARY ({metric_name}; positive delta = importance weighting "
          f"scored better)")
    print(f"  {'=' * 60}")

    def fmt(d):
        v = (d or {}).get('delta_mean')
        return '--' if v is None else f"{v:+.4f} ({d['n_paired_folds']} folds)"

    for arm, by_w in summary.items():
        for key, blk in by_w.items():
            parts = [f"kNN re-picked {fmt(blk['weighted_knn'].get('heldout_repick'))}"]
            gp = blk.get('weighted_gp') or {}
            if not gp.get('skipped'):
                for st, c in gp.items():
                    if isinstance(c, dict) and 'delta' in c:
                        parts.append(f"GP[{st}] {fmt(c['delta_both_fitted'])}")
            svr = blk.get('weighted_svr') or {}
            if not svr.get('skipped') and 'delta' in svr:
                parts.append(f"SVR {fmt(svr['delta'])}")
            print(f"    {arm} [{key}]: " + '; '.join(parts))


if __name__ == '__main__':
    parser = create_experiment_parser(
        'S10: Importance Weights for Distance-Dependent Downstream Tasks',
        has_delta=True, has_augment=True)
    # Accepted so old command lines still parse; ignored (each hybrid is built
    # at the decided budgets of the search it came from).
    parser.add_argument('--budget', type=int, default=None,
                        help='Ignored since 2026-10-05')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  budget=args.budget)
    save_experiment_results(results, args.output,
                            's10_importance_downstream', args.dataset)
