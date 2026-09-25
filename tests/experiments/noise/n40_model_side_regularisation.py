#!/usr/bin/env python3
"""N40: under noise spread over every label, does changing the MODEL help where changing the labels did not?

Paper: Noise.

THE QUESTION. The third research report (2026-09-25,
`compass_artifact_wf-4c1b218e-…`, its A2, A3 and D3 #5-#7) argues that with
mean-zero noise on every label and no repeat measurements, every change to the
labels is a smoother in disguise, so the only levers left are on the model side:
regularisation strength retuned for the noise, seed-averaged ensembles, a stack of
several models, and a Gaussian process whose noise term is set from an estimate of
the label noise. n07, n18, n24 and n30 already test the label-side changes; this
file tests the model-side ones, paired per pass against the same model at its
fixed default settings.

THE CONDITIONS. `gaussian` from the registry, and `dense_oofmse` from
noiseInject.RELATIVE_CONDITIONS (noise variance a share of lightgbm's
out-of-fold MSE on this fold's clean training labels, in `resid_sd_rep`; see
THE FEATURE SPACES). Swept through
noise_helpers.sweep_conditions, one file per condition; a job can name any other
regression condition with --noise-condition.

THE ARMS, per downstream model, every one fitted on the noisy training labels
unless it says otherwise, and graded on held-out molecules against their untouched
labels:

  clean                the model at its fixed defaults on the labels before
                       injection. The most any arm can recover.
  fixed                the model at its fixed defaults (find_best_hybrid.get_model).
                       The arm to beat.
  retuned              TUNE_N_ITER settings drawn from TUNE_GRIDS, each scored on
                       TUNE_INNER_FOLDS scaffold-grouped inner folds of THIS
                       FOLD'S TRAINING MOLECULES ONLY; the best refitted on all of
                       them. The held-out fold never reaches the tuner.
  tuned_clean_reused   the same search on the clean training labels, the chosen
                       setting then fitted on the noisy labels. The gap between it
                       and `retuned` is what retuning FOR the noise is worth.
  seed_one             lightgbm, xgboost and dnn with row and feature subsampling
                       (SEED_SUBSAMPLE), one seed.
  seed_avg             the same, SEED_AVG_N seeds, predictions averaged. seed_one
                       is there so the averaging is not confounded with the
                       subsampling that makes the seeds differ.
  rf_1000_trees        rf only: RF_BIG_TREES trees in place of the default 100.

TUNE_GRIDS widen find_best_hybrid._HP_SEARCH_SPACES toward stronger
regularisation, as the report's D3 #5 asks: more minimum samples per leaf, larger
L2 penalties, lower learning rates with more rounds, more dropout. Early stopping is
not a separate arm; a lower learning rate with the round count in the grid covers it.

Models that are not downstream models, each fitted in the feature matrix of
HOST_MODEL (lightgbm when it runs, else the first model), stored under arm `fixed`:

  stack                       non-negative least squares on the out-of-fold
                              predictions of every downstream model (each in its
                              own feature matrix), applied to their fixed-default
                              held-out predictions. Paired against every model.
  gp_ml_nugget                find_best_hybrid's GP: RBF + WhiteKernel, the noise
                              term fitted by marginal likelihood.
  gp_nugget_from_gamma        the same kernel with the noise term FIXED at the
                              Gamma-test noise variance (noise_helpers.gamma_test)
                              of this fold's training molecules. The report's
                              "fix the nugget from a preliminary test on the
                              measurement error".
  gp_nugget_cp                the noise term chosen from GP_NUGGET_MULT x the fitted
                              one by Mallows Cp, RSS/n + 2 sigma^2 tr(S)/n, with
                              sigma^2 the Gamma-test estimate.
  gp_nugget_cv                the same grid chosen by inner-fold error instead.
  lightgbm_plus_gp_residual   regression kriging: fixed lightgbm, plus a GP fitted
                              to lightgbm's out-of-fold residuals.
  knn_k5_uniform              kNN, k = 5, equal weights: the reference for the two
                              below. Equal weights because Cp needs a linear
                              smoother whose trace is known (tr S = n / k).
  knn_k_cp                    k from KNN_K_GRID by Mallows Cp with the Gamma sigma^2.
  knn_k_cv                    k from the same grid by inner-fold error.

On datasets with repeat measurements (herg_ic50, bindingdb_kd; any dataset whose
build_replicate_groups finds repeats), two more, both fitted to each training
molecule's MEAN label so the pair differs in one thing only:

  gp_means_ml_nugget          the GP with a single fitted noise term.
  gp_means_per_molecule_alpha the same, plus a per-molecule noise term
                              sigma_e^2 / k_i, sigma_e^2 the pooled within-molecule
                              variance of the training measurements and k_i the
                              molecule's count. The report's `alpha=noise_i`.

The GP arms are skipped, with the reason written, when a fold has more training
molecules than DEFAULT_MODELS['gp']['_max_samples'] (1,000). A GP holds the full
n x n kernel. They run in base_1, base_2 and every hybrid space, never on PCA.

THE FEATURE SPACES. base_1 and base_2 are each downstream model's top 2 single
representations, ranked by that model's own score in the stored search
(_shared.noise_base_reps_by_model, owner's decision 2026-09-25);
`base_reps_by_model` names them. The extra arms above run on the host model's
matrix in each space. A model the pool holds no search for has no space.
A relative condition's unit, lightgbm's out-of-fold residual SD on the fold's
clean training labels, must be one number for every space, so it is measured in
ONE representation, `resid_sd_rep`: the rank-1 single representation of the
first downstream model that has one. That choice was made on 2026-09-25 when the
fixed preference list was deleted; the owner has not signed it off.

Ridge is not used anywhere (the owner's no-ridge rule); the report's Cp example on
ridge is replaced by the two linear smoothers the model set already has.

THE GAMMA ESTIMATE. noise_helpers.gamma_test on Euclidean distances in the matrix
the GP and kNN see, k = 1..GAMMA_K_MAX neighbours. Above GAMMA_MAX_MOLECULES
training molecules it runs on a seeded subset of that many. Its intercept is an
upper bound on the label noise: it also holds variation the features cannot
resolve.

Usage:
    python n40_model_side_regularisation.py --dataset logd --mode full
    python n40_model_side_regularisation.py --dataset herg_ic50 --noise-condition dense_oofmse
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
from scipy.optimize import nnls
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, hybrid_for_fold,
    pool_recipe_per_downstream_model, group_models_by_recipe,
    sweep_conditions, tail_scores,
    RELATIVE_CONDITIONS, relative_levels, oof_residual_sd,
    inject_relative_train_labels, heavy_atom_counts, gamma_test,
)
from experiments.noise.n07_remediation import _predict_for_metric
from experiments.noise.n30_censored_likelihood import _paired, _mean_on
from experiments.noise.n33_robust_training import _share
from experiments.noise._ground_truth_providers import build_replicate_groups
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.noisy_labels import oof_predict
# The per-pass checkpoint, cached builds and the similarity axis shared by
# n32-n41 (n32's shared block, 2026-09-25). n40 changes the MODEL, never the
# training labels, so it writes no D-versus-D' rows: there is no D'.
from experiments.noise.n32_iterative_and_dual_selection import (
    downstream_model_list, fit_predict_cached, scaled_base_cells,
    cached_hybrid, ledger_signature, PassLedger, restart_flag, sweep_settings,
    sweep_ks, is_model_input, fold_similarity, _nc, _ss,
)

EXP_ID = 'n40_model_side_regularisation'

DEFAULT_CONDITIONS = ('gaussian', 'dense_oofmse')
DEFAULT_NOISE_CONDITION = 'gaussian'

# The replicate index of the injection, as n07, n11, n13, n24 and n33.
NOISE_DRAWS = (0, 1, 2)

# The tuner: this many settings, each scored on this many scaffold-grouped inner
# folds of the training molecules. ParameterSampler with a fixed seed, so every
# fold and every pass tries the same settings.
TUNE_N_ITER = 20
TUNE_INNER_FOLDS = 3
TUNE_SEED = 0

# find_best_hybrid._HP_SEARCH_SPACES, widened toward stronger regularisation.
# lightgbm's subsample / colsample_bytree are its bagging_fraction /
# feature_fraction under the scikit-learn names; subsample_freq=1 turns bagging on.
# svm's epsilon is given in SDs of the training labels (epsilon_sd) and converted
# per fit, so one grid means the same thing on pKi and on logD. dnn has no weight
# decay (kirby.models.dnn.DNNRegressor); dropout, width and rate are its levers.
TUNE_GRIDS = {
    'lightgbm': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.02, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [20, 40, 80],
        'reg_lambda': [0.0, 5.0, 20.0, 50.0],
        'colsample_bytree': [0.3, 0.6],
        'subsample': [0.7],
        'subsample_freq': [1],
    },
    'xgboost': {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.02, 0.05, 0.1],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'min_child_weight': [1, 5, 10, 20],
        'reg_lambda': [1.0, 5.0, 20.0, 50.0],
    },
    'rf': {
        'n_estimators': [200, 500],
        'max_depth': [5, 10, 20, None],
        'max_features': ['sqrt', 0.3, 0.5],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'min_samples_split': [2, 5, 10],
    },
    'svm': {
        'C': [0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
        'epsilon_sd': [0.01, 0.05, 0.1, 0.25, 0.5],
        'gamma': ['scale', 'auto'],
    },
    'dnn': {
        'dropout': [0.1, 0.2, 0.3, 0.5],
        'lr': [1e-3, 3e-4],
        'hidden_sizes': [(256, 128, 64), (128, 64), (64,)],
    },
}

# Seed averaging. The subsampling is what makes two seeds differ.
SEED_AVG_N = 10
SEED_SUBSAMPLE = {
    'lightgbm': {'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8},
    'xgboost': {'subsample': 0.8},
    'dnn': {},
}
RF_BIG_TREES = 1000

# Inner folds for the stack's and the kriging's out-of-fold predictions.
OOF_SPLITS = 5

HOST_MODEL = 'lightgbm'
KNN_K_GRID = (1, 3, 5, 7, 11, 15, 21, 31)
KNN_REFERENCE_K = 5
GP_NUGGET_MULT = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
GP_JITTER = 1e-6
GAMMA_K_MAX = 10
GAMMA_MAX_MOLECULES = 5000

GAP_MIN_FRAC = 0.02

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}

MODEL_ARMS = ('clean', 'fixed', 'retuned', 'tuned_clean_reused',
              'seed_one', 'seed_avg', 'rf_1000_trees')

# (arm under test, the arm it is paired against). Both stored under arm 'fixed'.
EXTRA_PAIRS = (
    ('gp_nugget_from_gamma', 'gp_ml_nugget'),
    ('gp_nugget_cp', 'gp_ml_nugget'),
    ('gp_nugget_cv', 'gp_ml_nugget'),
    ('gp_nugget_cp', 'gp_nugget_cv'),
    ('lightgbm_plus_gp_residual', 'lightgbm'),
    ('knn_k_cp', 'knn_k5_uniform'),
    ('knn_k_cv', 'knn_k5_uniform'),
    ('knn_k_cp', 'knn_k_cv'),
    ('gp_means_per_molecule_alpha', 'gp_means_ml_nugget'),
)

# Set by the smoke to a callable(name, X, y, groups): every call the tuner gets,
# so a test can prove the held-out molecules never reach it.
TUNER_HOOK = None


# ── the tuner ──

def tuned_model(name, factory, params, y_train):
    """A fresh downstream model with `params` applied. epsilon_sd is converted."""
    params = dict(params)
    if 'epsilon_sd' in params:
        params['epsilon'] = float(params.pop('epsilon_sd')) * max(
            float(np.std(y_train)), 1e-9)
    m = factory()
    return m.set_params(**params) if params else m


def tune(name, factory, X, y, groups, metric_fn, lower_is_better,
         n_iter=None, inner_folds=None):
    """Pick one setting from TUNE_GRIDS[name] on scaffold-grouped inner folds.

    X, y, groups are THIS FOLD'S TRAINING MOLECULES and nothing else. Returns
    (best_params, {setting index: mean inner score}).
    """
    if TUNER_HOOK is not None:
        TUNER_HOOK(name, X, y, groups)
    n_iter = TUNE_N_ITER if n_iter is None else int(n_iter)
    inner_folds = TUNE_INNER_FOLDS if inner_folds is None else int(inner_folds)
    grid = TUNE_GRIDS[name]
    settings = list(ParameterSampler(grid, n_iter=n_iter, random_state=TUNE_SEED))
    groups = np.asarray(groups)
    k = min(inner_folds, len(np.unique(groups)))
    if k < 2:
        raise ValueError('tune: fewer than two scaffold groups in the training fold')
    splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    scores = []
    for params in settings:
        fold_scores = []
        for tr, va in splits:
            m = tuned_model(name, factory, params, y[tr])
            m.fit(X[tr], y[tr])
            fold_scores.append(float(metric_fn(y[va], m.predict(X[va]))))
        scores.append(float(np.mean(fold_scores)))
    s = np.asarray(scores)
    best = int(np.nanargmin(s) if lower_is_better else np.nanargmax(s))
    return settings[best], {'inner_scores': scores, 'best_index': best}


def _params_for_json(p):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}


# ── seed averaging ──

def seeded_model(name, factory, seed):
    m = factory()
    extra = dict(SEED_SUBSAMPLE.get(name, {}))
    extra['random_state'] = int(seed)
    return m.set_params(**extra)


# ── the Gamma estimate, Cp and the closed-form GP ──

def _sub_distance(dist_blocks, n, idx):
    """The idx-by-idx block of a training distance given as row blocks
    (n32.FoldSimilarity.distance_blocks), without holding the n x n matrix."""
    idx = np.asarray(idx)
    pos = np.full(n, -1)
    pos[idx] = np.arange(idx.size)
    D = np.empty((idx.size, idx.size))
    step = _ss._chunk_rows(n)
    for a in range(0, n, step):
        b = min(n, a + step)
        rows = idx[(idx >= a) & (idx < b)]
        if rows.size:
            D[pos[rows]] = dist_blocks(a, b)[rows - a][:, idx]
    return D


def gamma_sigma2(X, y, seed=0, k_max=GAMMA_K_MAX, dist_blocks=None):
    """noise_helpers.gamma_test in the matrix the model sees; subset above a cap.

    ⛔ THE DISTANCE CHANGED 2026-09-25. Until then this was cdist's default,
    Euclidean, whatever --distance-metric said (plan section 4: "Euclidean
    slips in unasked", n40:332). It is now the process-wide distance
    (distance_config.scipy_metric, manhattan by default), which is what the
    model_input similarity setting means everywhere else, as the owner asked
    (2026-09-25). So the Gamma estimate, knn_k_cp and the two GP arms that read
    it are NOT today's numbers under model_input.

    k_max and dist_blocks, added 2026-09-25: the similarity sweep's k in place
    of GAMMA_K_MAX, and a setting's distance (row blocks) in place of the
    model-input matrix. The same seeded subset is taken either way."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.arange(len(y))
    if len(y) > GAMMA_MAX_MOLECULES:
        idx = np.random.RandomState(seed).choice(len(y), GAMMA_MAX_MOLECULES,
                                                 replace=False)
    if dist_blocks is None:
        from kirby.similarity.distance_config import scipy_metric
        _m, _kw = scipy_metric()
        D = cdist(X[idx], X[idx], metric=_m, **_kw)
    else:
        D = _sub_distance(dist_blocks, len(y), idx)
    g = gamma_test(D, y[idx], k_max=k_max)
    if g is None:
        return None, None
    g['n_used'] = int(len(idx))
    return max(float(g['intercept']), 0.0), g


def knn_cp_choice(X, y, sigma2, k_grid=KNN_K_GRID, neighbour_idx=None):
    """k by Mallows Cp for equal-weight kNN: RSS/n + 2 sigma^2 / k.

    The in-sample fit counts the molecule itself as its first neighbour, so the
    smoother's trace is n / k and df / n = 1 / k.

    ⛔ THE DISTANCE CHANGED 2026-09-25: a bare NearestNeighbors (Euclidean)
    until then; now _shared.make_nearest_neighbors, the process-wide distance,
    as gamma_sigma2 above. neighbour_idx, added the same day: a similarity
    setting's lists (the molecule itself excluded), and the molecule is put
    back in front as its own first neighbour, as the in-sample fit counts it.
    """
    ks = [k for k in k_grid if k <= len(y)]
    if neighbour_idx is None:
        from experiments._shared import make_nearest_neighbors
        nn = make_nearest_neighbors(n_neighbors=max(ks)).fit(X)
        _, idx = nn.kneighbors(X)
    else:
        nb = np.asarray(neighbour_idx)
        idx = np.hstack([np.arange(len(y))[:, None], nb])
        ks = [k for k in ks if k <= idx.shape[1]]
    cps = {}
    for k in ks:
        fit = y[idx[:, :k]].mean(axis=1)
        cps[k] = float(np.mean((y - fit) ** 2) + 2.0 * sigma2 / k)
    best = min(cps, key=cps.get)
    return best, cps


def knn_cv_choice(X, y, groups, metric_fn, lower_is_better, k_grid=KNN_K_GRID,
                  inner_folds=TUNE_INNER_FOLDS):
    k_fold = min(inner_folds, len(np.unique(groups)))
    splits = list(GroupKFold(n_splits=k_fold).split(X, y, groups))
    ks = [k for k in k_grid if k <= min(len(tr) for tr, _ in splits)]
    scores = {}
    for k in ks:
        s = []
        for tr, va in splits:
            m = KNeighborsRegressor(n_neighbors=k, weights='uniform').fit(X[tr], y[tr])
            s.append(float(metric_fn(y[va], m.predict(X[va]))))
        scores[k] = float(np.mean(s))
    pick = (min if lower_is_better else max)(scores, key=scores.get)
    return pick, scores


class _EigenGP:
    """The GP posterior mean with a FIXED kernel and a chosen noise term, closed form.

    Takes the signal kernel of a fitted find_best_hybrid GP (kernel_.k1) and
    normalises y as that GP does (normalize_y=True). With K = U diag(lam) U^T,
    the smoother at noise term a is S = U diag(lam / (lam + a)) U^T, so every
    noise term on the grid costs one matrix product after one decomposition.
    `a` is in normalised-label units, as sklearn's WhiteKernel noise_level is.
    """

    def __init__(self, signal_kernel, X, y):
        self.k = signal_kernel
        self.X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu = float(np.mean(y))
        self.sd = float(np.std(y)) or 1.0
        self.z = (y - self.mu) / self.sd
        lam, U = np.linalg.eigh(self.k(self.X))
        self.lam = np.clip(lam, 0.0, None)
        self.U = U
        self.Utz = U.T @ self.z

    def in_sample(self, a):
        f = self.lam / (self.lam + a + GP_JITTER)
        fit = self.U @ (f * self.Utz)
        return self.mu + self.sd * fit, float(np.sum(f))

    def predict(self, Xq, a):
        coef = self.U @ (self.Utz / (self.lam + a + GP_JITTER))
        return self.mu + self.sd * (self.k(np.asarray(Xq, dtype=float), self.X) @ coef)


def gp_cp_choice(egp, y, sigma2, grid):
    """Noise term by Mallows Cp in label units: RSS/n + 2 sigma^2 tr(S)/n."""
    n = len(y)
    cps = {}
    for a in grid:
        fit, tr = egp.in_sample(a)
        cps[a] = float(np.mean((y - fit) ** 2) + 2.0 * sigma2 * tr / n)
    return min(cps, key=cps.get), cps


def gp_cv_choice(signal_kernel, X, y, groups, grid, metric_fn, lower_is_better,
                 inner_folds=TUNE_INNER_FOLDS):
    k = min(inner_folds, len(np.unique(groups)))
    splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    scores = {a: [] for a in grid}
    for tr, va in splits:
        e = _EigenGP(signal_kernel, X[tr], y[tr])
        for a in grid:
            scores[a].append(float(metric_fn(y[va], e.predict(X[va], a))))
    scores = {a: float(np.mean(v)) for a, v in scores.items()}
    pick = (min if lower_is_better else max)(scores, key=scores.get)
    return pick, scores


def make_gp(alpha=None):
    """find_best_hybrid's GP; with `alpha`, a per-molecule noise term is added."""
    from kirby.find_best_hybrid import _get_gp_regressor
    gp = _get_gp_regressor()
    if alpha is not None:
        gp.set_params(alpha=np.asarray(alpha, dtype=float) + GP_JITTER)
    return gp


def gp_max_samples():
    from kirby.find_best_hybrid import DEFAULT_MODELS
    return DEFAULT_MODELS['gp'].get('_max_samples')


def molecule_means(y, rep_ids):
    """(first row per molecule, mean label, count, pooled within variance)."""
    order = {}
    for i, g in enumerate(rep_ids):
        order.setdefault(g, []).append(i)
    gids = sorted(order, key=lambda g: order[g][0])
    first = np.array([order[g][0] for g in gids])
    means = np.array([float(np.mean(y[order[g]])) for g in gids])
    counts = np.array([len(order[g]) for g in gids], dtype=float)
    ss = sum(float(np.sum((y[order[g]] - np.mean(y[order[g]])) ** 2))
             for g in gids if len(order[g]) >= 2)
    dof = int(sum(len(order[g]) - 1 for g in gids if len(order[g]) >= 2))
    s2 = ss / dof if dof > 0 else float('nan')
    return first, means, counts, s2, dof


def _fit_predict(model, X_tr, y_tr, X_va, metric_name):
    model.fit(X_tr, y_tr)
    return np.asarray(_predict_for_metric(model, X_va, metric_name), dtype=float)


def _oof_model(factory, X, y, groups, n_splits):
    k = min(n_splits, len(np.unique(groups)))
    p = np.full(len(y), np.nan)
    for tr, va in GroupKFold(n_splits=k).split(X, y, groups):
        p[va] = factory().fit(X[tr], y[tr]).predict(X[va])
    return p


def _seed(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _levels_and_draws(dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs. At level 0.0 the draw delivers zero, so one draw."""
    if levels is None:
        levels = (relative_levels(condition) if condition in RELATIVE_CONDITIONS
                  else noise_levels(dataset_name, strategy=condition, is_clf=False))
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        for d in (draws if float(lvl) > 0 else draws[:1]):
            out.append((float(lvl), int(d)))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N40 across its conditions, one file per condition."""
    if data['task_type'] != 'regression':
        print(f"  SKIPPED: {dataset_name} is {data['task_type']}. The Gamma "
              f"estimate, Cp and the GP noise term are defined on a number, so "
              f"n40 is regression only.")
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    # sweep_conditions pops force_rerun; PassLedger reads restart_passes.
    kwargs['restart_passes'] = bool(kwargs.get('force_rerun'))
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=list(DEFAULT_CONDITIONS), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N40 under ONE injected condition."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N40: RETUNING, SEED AVERAGING, STACKING, GP NOISE TERMS")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    relative = condition in RELATIVE_CONDITIONS

    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    tune_n_iter = int(kwargs.get('tune_n_iter', TUNE_N_ITER))
    tune_inner = int(kwargs.get('tune_inner_folds', TUNE_INNER_FOLDS))
    seed_avg_n = int(kwargs.get('seed_avg_n', SEED_AVG_N))
    rf_big = int(kwargs.get('rf_big_trees', RF_BIG_TREES))
    skip_tuning = bool(kwargs.get('skip_tuning', False))
    skip_gp = bool(kwargs.get('skip_gp', False))
    skip_knn = bool(kwargs.get('skip_knn', False))
    skip_stack = bool(kwargs.get('skip_stack', False))

    downstream_models = downstream_model_list(kwargs, config)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError("n40: no downstream model could be built.")
    host = HOST_MODEL if HOST_MODEL in downstream_models else downstream_models[0]

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')
    noise_groups = noise_groups_for(data)[0]
    if condition in GROUPED_STRATEGIES and noise_groups is None:
        raise RuntimeError(f"n40: {condition!r} needs scaffold groups.")
    heavy = None
    if condition == 'hetero_seen':
        if data.get('smiles') is None:
            raise RuntimeError("n40: hetero_seen needs SMILES for heavy-atom counts.")
        heavy = heavy_atom_counts(data['smiles'])

    rep_ids = build_replicate_groups(data)
    has_repeats = (rep_ids is not None
                   and len(np.unique(rep_ids)) < len(rep_ids))
    gp_cap = gp_max_samples()

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': ('share of lightgbm out-of-fold MSE on the clean training '
                       'fold' if relative and not condition.startswith('sparse')
                       else 'residual SDs' if relative
                       else noise_dose_units(dataset_name)),
        'models': downstream_models,
        'host_model_for_extra_arms': host,
        'arms': list(MODEL_ARMS),
        'arm_to_beat': 'fixed',
        'extra_pairs': [list(p) for p in EXTRA_PAIRS],
        'settings': {
            'tune_n_iter': tune_n_iter, 'tune_inner_folds': tune_inner,
            'tune_seed': TUNE_SEED,
            'tune_grids': {k: {kk: [list(x) if isinstance(x, tuple) else x
                                    for x in vv] for kk, vv in v.items()}
                           for k, v in TUNE_GRIDS.items()},
            'seed_avg_n': seed_avg_n, 'seed_subsample': SEED_SUBSAMPLE,
            'rf_big_trees': rf_big, 'oof_splits': oof_splits,
            'knn_k_grid': list(KNN_K_GRID), 'gp_nugget_mult': list(GP_NUGGET_MULT),
            'gamma_k_max': GAMMA_K_MAX,
            'gamma_max_molecules': GAMMA_MAX_MOLECULES,
            'gp_max_training_molecules': gp_cap,
        },
        'tuning_note': ('The tuner sees only the training molecules of the '
                        'fold, split into scaffold-grouped inner folds. The '
                        'held-out molecules never reach it.'),
        'gamma_note': ('The Gamma intercept is an upper bound on the label '
                       'noise: it also holds variation the features cannot '
                       'resolve.'),
        'repeat_measurements': bool(has_repeats),
        'gp_means_arms_note': (
            'fitted to each training molecule\'s mean label; run only where '
            'build_replicate_groups finds repeat measurements'
            if has_repeats else
            'not run: this dataset has no repeat measurements, so there is no '
            'per-molecule count to set a noise term from'),
        'gap_min_frac': GAP_MIN_FRAC,
        'similarity_settings': [st.name for st in sweep_settings()],
        'similarity_ks': list(sweep_ks(GAMMA_K_MAX)),
        'similarity_note': (
            'The Gamma estimate and knn_k_cp read neighbours in the process-wide '
            'distance under model_input (Euclidean before 2026-09-25). Every '
            'other (setting, k) adds, under arm fixed: knn_k5_uniform@<setting> '
            '(a 5-neighbour mean in that setting), and knn_k_cp@, '
            'gp_nugget_from_gamma@ and gp_nugget_cp@<setting>|k<k>, whose '
            'sigma^2 is the Gamma intercept in that setting with k neighbours. '
            'knn_k_cv is not swept: see the TODO at knn_k_cv. No D-versus-D\' '
            'rows: n40 changes the model, not the training data.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to this '
            'condition on this dataset. Held-out molecules are graded against '
            'untouched labels in every arm.'),
    }

    splits = list(cv_split(groups, n_folds, next(iter(reps.values())), labels,
                           mode=mode))

    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    _raised = [d['lookup_error'] for d in no_recipe.values() if d['lookup_error']]
    results['hybrid_lookup_error'] = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {m: d['why'] for m, d in no_recipe.items()}

    recipes_by_model, recipe_for_space_by_model, hybrid_unavailable = {}, {}, {}
    for _m in downstream_models:
        _m_recipes, _m_unavail = pool_recipes_by_family(
            by_model[_m]['meta'] if _m in by_model else None,
            SPACE_FOR_FAMILY, eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipes_by_model[_m] = _m_recipes
        recipe_for_space_by_model[_m] = {r['space']: r['blueprint'] for r in _m_recipes}
        for _sp, _why in _m_unavail.items():
            hybrid_unavailable.setdefault(_sp, {})[_m] = _why
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    _ordered = []
    for _m in downstream_models:
        for r in recipes_by_model.get(_m, ()):
            if r['space'] not in _ordered:
                _ordered.append(r['space'])

    # ── The single-representation comparison, per downstream model ──
    if base_reps_override is not None:
        base_by_model = {m: list(base_reps_override[m])
                         for m in downstream_models if base_reps_override.get(m)}
        base_missing = {m: 'not given by the caller' for m in downstream_models
                        if m not in base_by_model}
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: d['search'] for m, d in by_model.items()}, downstream_models,
            reps=reps, dataset_name=dataset_name)
    results['base_reps_by_model'] = base_by_model
    results['base_reps_missing'] = base_missing
    base_spaces = tuple(
        f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1)
        if base_rank_groups(base_by_model, _k, downstream_models))
    # The one representation a relative condition's residual SD is measured
    # in (module docstring, THE FEATURE SPACES).
    _ref_model = next((m for m in downstream_models if base_by_model.get(m)),
                      None)
    resid_sd_rep = base_by_model[_ref_model][0] if _ref_model else None
    if relative:
        results['resid_sd_rep'] = resid_sd_rep
        results['resid_sd_rep_chosen_as'] = (
            f'rank-1 single representation of {_ref_model}, the first '
            f'downstream model that has one' if _ref_model else
            'none: no downstream model has a single representation, so no '
            'residual SD can be measured and nothing is injected')
    spaces = base_spaces + tuple(_ordered)
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
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in rs}
            for m, rs in recipes_by_model.items() if rs})
    else:
        results['hybrid_unavailable_reason'] = (
            (f"the pool lookup RAISED {results['hybrid_lookup_error']}; the pool "
             f"was never read. Every arm ran in the single representations "
             f"only.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from pool subset {POOL_SUBSET!r} for any '
             f'downstream model. Every arm ran in the single representations '
             f'only.'))
        print("  no hybrid recipe; single representations only")
    results['spaces_run'] = list(spaces)

    passes = _levels_and_draws(dataset_name, condition,
                               levels=kwargs.get('levels'), draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    results['noise_draws'] = sorted({d for _, d in passes})
    print(f"  {condition}: {len(labels)} molecules, {len(splits)} folds, "
          f"models {downstream_models}, {len(passes)} (level, draw) passes")

    acc, tails, pass_recs, errors, skips = {}, {}, {}, {}, {}
    t0 = time.time()
    resid_sd_by_fold = {}
    clean_tuned_cache = {}
    metric = (metric_name, metric_fn, lower_is_better)
    sim_settings = [st for st in sweep_settings()]
    sim_ks = sweep_ks(GAMMA_K_MAX)
    # The per-pass checkpoint, 2026-09-25 (n32's PassLedger; no rows). The two
    # in-run caches (resid_sd_by_fold, clean_tuned_cache) are not stored:
    # their keys are not JSON, and both recompute to the same values.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, condition,
                        signature=ledger_signature(data, config, results,
                                                   splits),
                        restart=restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        acc, tails, pass_recs, errors, skips = (
            st_['acc'], st_['tails'], st_['pass_recs'], st_['errors'],
            st_['skips'])
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    def _store(lvl_key, space, arm, mname, pass_key, pred, y_val):
        s = float(metric_fn(y_val, pred))
        (acc.setdefault(lvl_key, {}).setdefault(space, {}).setdefault(arm, {})
         .setdefault(mname, {}))[pass_key] = s
        (tails.setdefault(lvl_key, {}).setdefault(space, {}).setdefault(arm, {})
         .setdefault(mname, {}))[pass_key] = tail_scores(y_val, pred)

    for level, draw in passes:
        lvl_key = f'{level:g}'
        spec = (None if relative else
                noise_spec_for(dataset_name, condition, level,
                               noise_seed=draw, is_clf=False))
        print(f"\n  --- {condition} / level {lvl_key} / draw {draw} ---")
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if ledger.pass_done(pass_key):
                continue
            y_val = labels[val_idx]
            y_clean_tr = labels[train_idx]
            g_tr = groups[train_idx] if groups is not None else None
            try:
                if relative:
                    if fold_i not in resid_sd_by_fold:
                        if resid_sd_rep is None:
                            raise RuntimeError(results['resid_sd_rep_chosen_as'])
                        resid_sd_by_fold[fold_i] = oof_residual_sd(
                            StandardScaler().fit_transform(np.asarray(
                                reps[resid_sd_rep][train_idx], dtype=np.float64)),
                            y_clean_tr, g_tr, n_splits=oof_splits)[0]
                    y0, cmask, prov = inject_relative_train_labels(
                        labels, train_idx=train_idx, condition=condition,
                        level=level, resid_sd=resid_sd_by_fold[fold_i],
                        seed=_seed(dataset_name, condition, draw, fold_i),
                        heavy_atoms=(heavy[train_idx] if heavy is not None
                                     else None))
                else:
                    y0, cmask, prov = inject_train_labels(
                        labels, spec, train_idx=train_idx, is_clf=False,
                        groups=noise_groups, dataset_name=dataset_name)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            rec = {'pass': pass_key, 'training_molecules': int(len(y0)),
                   'labels_moved': int(np.sum(np.abs(y0 - y_clean_tr) > 1e-12)),
                   'injected_rms_label_units': float(
                       np.sqrt(np.mean((y0 - y_clean_tr) ** 2))),
                   'per_space': {}}
            if relative:
                rec['resid_sd_label_units'] = float(resid_sd_by_fold[fold_i])

            train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                          for k, v in reps.items()}
            val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}
            # Each single-representation space holds one matrix per distinct
            # representation at that rank, shared by the models that rank it
            # there.
            fold_spaces = {}
            for _sp in base_spaces:
                # Standardised once per fold through noise_cache (2026-09-25).
                fold_spaces[_sp] = scaled_base_cells(
                    dataset_name, reps, train_idx, val_idx, base_by_model,
                    int(_sp.split('_')[1]), downstream_models)
            for _sp in spaces:
                if _sp in base_spaces:
                    continue
                _cells = []
                for _bp_m, _mnames, _shareable in group_models_by_recipe(
                        {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                         for _m in downstream_models}, downstream_models):
                    # Cached per (fold, recipe, labels), 2026-09-25.
                    X_tr_h, X_va_h = cached_hybrid(
                        dataset_name, train_reps, val_reps, y0, _bp_m, g_tr,
                        _mnames[0], model_factories, metric, train_idx,
                        _shareable)
                    if X_tr_h is None:
                        errors[f'{lvl_key}|{_sp}/{_mnames[0]}|d{draw}|f{fold_i}|build'] = (
                            'the hybrid could not be built')
                        continue
                    sh = StandardScaler()
                    _cells.append((sh.fit_transform(X_tr_h), sh.transform(X_va_h),
                                   list(_mnames)))
                if _cells:
                    fold_spaces[_sp] = _cells

            # The similarity spaces of this fold, from the noisy labels (plan
            # section 1, 2026-09-25). The host model's spaces are read.
            fsim = fold_similarity(
                dataset_name, data, reps=reps, train_idx=train_idx,
                val_idx=val_idx, y_train=y0, groups_train=g_tr,
                base_by_model=base_by_model,
                recipe_for_space_by_model=recipe_for_space_by_model,
                model_factories=model_factories, metric=metric,
                task_type=task_type)

            for space in spaces:
                sp_rec = {'tuned': {}, 'seed_avg': {}}
                fixed_val, oof_by_model = {}, {}
                host_X = None
                for X_tr_s, X_va_s, cell_models in fold_spaces.get(space, ()):
                    ekey = f'{lvl_key}|{space}/{cell_models[0]}|d{draw}|f{fold_i}'
                    if host in cell_models:
                        host_X = (X_tr_s, X_va_s)
                    for mname in cell_models:
                        fac = model_factories[mname]
                        # clean and fixed
                        for arm, yy in (('clean', y_clean_tr), ('fixed', y0)):
                            try:
                                # Through noise_cache for a seeded model
                                # (2026-09-25); the same prediction.
                                pred = np.asarray(fit_predict_cached(
                                    dataset_name, mname, fac, X_tr_s, yy,
                                    X_va_s, metric_name), dtype=float)
                                _store(lvl_key, space, arm, mname, pass_key, pred, y_val)
                                if arm == 'fixed':
                                    fixed_val[mname] = pred
                            except Exception as exc:
                                errors[f'{ekey}|{arm}|{mname}'] = f'{type(exc).__name__}: {exc}'
                        # retuned and tuned_clean_reused
                        if not skip_tuning and mname in TUNE_GRIDS:
                            try:
                                p_noisy, info_n = tune(
                                    mname, fac, X_tr_s, y0, g_tr, metric_fn,
                                    lower_is_better, tune_n_iter, tune_inner)
                                # A single representation does not move with
                                # the noisy labels, so its clean tuning is
                                # reused across passes.
                                ck = ((fold_i, space, mname)
                                      if space in base_spaces else None)
                                if ck is not None and ck in clean_tuned_cache:
                                    p_clean, info_c = clean_tuned_cache[ck]
                                else:
                                    p_clean, info_c = tune(
                                        mname, fac, X_tr_s, y_clean_tr, g_tr,
                                        metric_fn, lower_is_better,
                                        tune_n_iter, tune_inner)
                                    if ck is not None:
                                        clean_tuned_cache[ck] = (p_clean, info_c)
                                for arm, params in (('retuned', p_noisy),
                                                    ('tuned_clean_reused', p_clean)):
                                    pred = _fit_predict(
                                        tuned_model(mname, fac, params, y0),
                                        X_tr_s, y0, X_va_s, metric_name)
                                    _store(lvl_key, space, arm, mname, pass_key,
                                           pred, y_val)
                                sp_rec['tuned'][mname] = {
                                    'noisy': _params_for_json(p_noisy),
                                    'clean': _params_for_json(p_clean),
                                    'same_setting': p_noisy == p_clean,
                                    'inner_scores_noisy': info_n['inner_scores']}
                            except Exception as exc:
                                errors[f'{ekey}|tune|{mname}'] = f'{type(exc).__name__}: {exc}'
                        # seed averaging
                        try:
                            if mname in SEED_SUBSAMPLE:
                                preds = []
                                for s in range(seed_avg_n):
                                    preds.append(_fit_predict(
                                        seeded_model(mname, fac, s), X_tr_s, y0,
                                        X_va_s, metric_name))
                                _store(lvl_key, space, 'seed_one', mname,
                                       pass_key, preds[0], y_val)
                                _store(lvl_key, space, 'seed_avg', mname,
                                       pass_key, np.mean(preds, axis=0), y_val)
                                sp_rec['seed_avg'][mname] = {
                                    'n_seeds': seed_avg_n,
                                    'mean_sd_across_seeds': float(
                                        np.mean(np.std(preds, axis=0)))}
                            elif mname == 'rf':
                                pred = _fit_predict(fac().set_params(n_estimators=rf_big),
                                                    X_tr_s, y0, X_va_s, metric_name)
                                _store(lvl_key, space, 'rf_1000_trees', mname,
                                       pass_key, pred, y_val)
                        except Exception as exc:
                            errors[f'{ekey}|seed|{mname}'] = f'{type(exc).__name__}: {exc}'
                        # out-of-fold predictions for the stack
                        if not skip_stack and len(downstream_models) > 1:
                            try:
                                oof_by_model[mname] = _oof_model(
                                    fac, X_tr_s, y0, g_tr, oof_splits)
                            except Exception as exc:
                                errors[f'{ekey}|stack_oof|{mname}'] = f'{type(exc).__name__}: {exc}'

                if not fold_spaces.get(space):
                    continue
                # ── the stack ──
                have = [m for m in downstream_models
                        if m in oof_by_model and m in fixed_val]
                if not skip_stack and len(have) > 1:
                    try:
                        P = np.column_stack([oof_by_model[m] for m in have])
                        w, _ = nnls(P, y0)
                        if w.sum() > 0:
                            pred = np.column_stack([fixed_val[m] for m in have]) @ w
                            _store(lvl_key, space, 'fixed', 'stack', pass_key, pred, y_val)
                            sp_rec['stack_weights'] = dict(zip(have, map(float, w)))
                    except Exception as exc:
                        errors[f'{lvl_key}|{space}|d{draw}|f{fold_i}|stack'] = (
                            f'{type(exc).__name__}: {exc}')

                if host_X is None:
                    rec['per_space'][space] = sp_rec
                    continue
                X_tr_s, X_va_s = host_X
                ekey = f'{lvl_key}|{space}|d{draw}|f{fold_i}'

                # ── the Gamma estimate ──
                sigma2 = None
                if not (skip_gp and skip_knn):
                    try:
                        sigma2, g = gamma_sigma2(X_tr_s, y0, seed=fold_i)
                        if g is not None:
                            sp_rec['gamma'] = {k: g[k] for k in (
                                'intercept', 'slope', 'k_max', 'v_ratio',
                                'var_y', 'n', 'n_used')}
                    except Exception as exc:
                        errors[f'{ekey}|gamma'] = f'{type(exc).__name__}: {exc}'

                # ── the Gamma estimate under every other (setting, k) ──
                # {name suffix: sigma^2}. model_input at GAMMA_K_MAX is the
                # sigma2 above; its other ks and every other setting are here.
                sim_sigma2, sim_lists = {}, {}
                if not (skip_gp and skip_knn):
                    sp_rec['gamma_by_similarity'] = {}
                    for setting in sim_settings:
                        try:
                            if is_model_input(setting):
                                blocks = None
                            else:
                                blocks, _n, why = fsim.distance_blocks(
                                    setting, host, X_tr_s)
                                if blocks is None:
                                    sp_rec['gamma_by_similarity'][
                                        setting.name] = {'unavailable': why}
                                    continue
                            for k in sim_ks:
                                if is_model_input(setting) and k == GAMMA_K_MAX:
                                    continue
                                s2_k, g_k = gamma_sigma2(
                                    X_tr_s, y0, seed=fold_i, k_max=k,
                                    dist_blocks=blocks)
                                nm = f'{setting.name}|k{k}'
                                sp_rec['gamma_by_similarity'][nm] = (
                                    None if g_k is None else
                                    {kk: g_k[kk] for kk in ('intercept',
                                                            'k_max', 'n_used')})
                                if s2_k is not None:
                                    sim_sigma2[nm] = (setting, s2_k)
                        except Exception as exc:
                            errors[f'{ekey}|gamma@{setting.name}'] = (
                                f'{type(exc).__name__}: {exc}')

                # ── kNN, k by Cp and by CV ──
                if not skip_knn:
                    try:
                        pred = KNeighborsRegressor(
                            n_neighbors=min(KNN_REFERENCE_K, len(y0)),
                            weights='uniform').fit(X_tr_s, y0).predict(X_va_s)
                        _store(lvl_key, space, 'fixed', 'knn_k5_uniform', pass_key, pred, y_val)
                        k_cv, cv_s = knn_cv_choice(X_tr_s, y0, g_tr, metric_fn,
                                                   lower_is_better, inner_folds=tune_inner)
                        pred = KNeighborsRegressor(n_neighbors=k_cv, weights='uniform') \
                            .fit(X_tr_s, y0).predict(X_va_s)
                        _store(lvl_key, space, 'fixed', 'knn_k_cv', pass_key, pred, y_val)
                        sp_rec['knn'] = {'k_cv': int(k_cv),
                                         'cv_scores': {str(k): v for k, v in cv_s.items()}}
                        if sigma2 is not None:
                            k_cp, cps = knn_cp_choice(X_tr_s, y0, sigma2)
                            pred = KNeighborsRegressor(n_neighbors=k_cp, weights='uniform') \
                                .fit(X_tr_s, y0).predict(X_va_s)
                            _store(lvl_key, space, 'fixed', 'knn_k_cp', pass_key, pred, y_val)
                            sp_rec['knn'].update({'k_cp': int(k_cp),
                                                  'cp': {str(k): v for k, v in cps.items()}})
                    except Exception as exc:
                        errors[f'{ekey}|knn'] = f'{type(exc).__name__}: {exc}'
                    # TODO(similarity-sweep 2026-09-25): knn_k_cv is not swept
                    # over the similarity settings. Its inner folds need each
                    # molecule's neighbours among the inner TRAINING rows only,
                    # which a fold's lists do not give; building lists per
                    # inner fold is a design choice (which space, rebuilt per
                    # inner fold or restricted) left to the owner.
                    # ── kNN under every other similarity setting ──
                    kmax_knn = max(KNN_K_GRID)
                    for nm, (setting, s2_k) in list(sim_sigma2.items()) + [
                            (st.name, (st, None)) for st in sim_settings]:
                        if is_model_input(setting):
                            if s2_k is None:
                                continue
                            try:
                                k_cp, cps = knn_cp_choice(X_tr_s, y0, s2_k)
                                pred = KNeighborsRegressor(
                                    n_neighbors=k_cp, weights='uniform') \
                                    .fit(X_tr_s, y0).predict(X_va_s)
                                _store(lvl_key, space, 'fixed',
                                       f'knn_k_cp@{nm}', pass_key, pred, y_val)
                            except Exception as exc:
                                errors[f'{ekey}|knn_k_cp@{nm}'] = (
                                    f'{type(exc).__name__}: {exc}')
                            continue
                        try:
                            if setting.name not in sim_lists:
                                wide = _ss.SimilaritySetting(
                                    setting.space, setting.weighting,
                                    setting.metric, k_max=kmax_knn)
                                sim_lists[setting.name] = fsim.neighbours(
                                    wide, host, X_tr_s, X_va_s)
                            nb = sim_lists[setting.name]
                            if not nb.get('available'):
                                continue
                            iv, _dv = nb['idx_val'], nb['dist_val']
                            if s2_k is None:
                                # knn_k5_uniform in this setting: the mean of
                                # the 5 nearest training labels.
                                kk = min(KNN_REFERENCE_K, iv.shape[1])
                                _store(lvl_key, space, 'fixed',
                                       f'knn_k5_uniform@{nm}', pass_key,
                                       y0[iv[:, :kk]].mean(axis=1), y_val)
                            else:
                                k_cp, cps = knn_cp_choice(
                                    X_tr_s, y0, s2_k,
                                    neighbour_idx=nb['idx_train'])
                                _store(lvl_key, space, 'fixed',
                                       f'knn_k_cp@{nm}', pass_key,
                                       y0[iv[:, :k_cp]].mean(axis=1), y_val)
                        except Exception as exc:
                            errors[f'{ekey}|knn@{nm}'] = (
                                f'{type(exc).__name__}: {exc}')

                # ── the GP arms ──
                if not skip_gp:
                    n_tr = len(y0)
                    if gp_cap is not None and n_tr > gp_cap:
                        skips[f'{ekey}|gp'] = (
                            f'{n_tr} training molecules, above the GP cap of '
                            f'{gp_cap} (DEFAULT_MODELS["gp"]["_max_samples"])')
                    else:
                        try:
                            gp = make_gp().fit(X_tr_s, y0)
                            pred = gp.predict(X_va_s)
                            _store(lvl_key, space, 'fixed', 'gp_ml_nugget', pass_key, pred, y_val)
                            white = float(gp.kernel_.k2.noise_level)
                            sig_kernel = gp.kernel_.k1
                            egp = _EigenGP(sig_kernel, X_tr_s, y0)
                            var_y = egp.sd ** 2
                            grid = [white * m for m in GP_NUGGET_MULT]
                            gp_rec = {'ml_noise_normalised': white,
                                      'ml_noise_label_units': white * var_y}
                            a_cv, cv_s = gp_cv_choice(sig_kernel, X_tr_s, y0, g_tr,
                                                      grid, metric_fn, lower_is_better,
                                                      inner_folds=tune_inner)
                            _store(lvl_key, space, 'fixed', 'gp_nugget_cv', pass_key,
                                   egp.predict(X_va_s, a_cv), y_val)
                            gp_rec['cv_noise_label_units'] = a_cv * var_y
                            if sigma2 is not None:
                                a_g = sigma2 / var_y
                                _store(lvl_key, space, 'fixed', 'gp_nugget_from_gamma',
                                       pass_key, egp.predict(X_va_s, a_g), y_val)
                                a_cp, cps = gp_cp_choice(egp, y0, sigma2, grid + [a_g])
                                _store(lvl_key, space, 'fixed', 'gp_nugget_cp', pass_key,
                                       egp.predict(X_va_s, a_cp), y_val)
                                # The same two arms with sigma^2 from every
                                # other (setting, k), 2026-09-25.
                                for nm, (_st, s2_k) in sim_sigma2.items():
                                    a_gk = s2_k / var_y
                                    _store(lvl_key, space, 'fixed',
                                           f'gp_nugget_from_gamma@{nm}',
                                           pass_key, egp.predict(X_va_s, a_gk),
                                           y_val)
                                    a_cpk, _c = gp_cp_choice(
                                        egp, y0, s2_k, grid + [a_gk])
                                    _store(lvl_key, space, 'fixed',
                                           f'gp_nugget_cp@{nm}', pass_key,
                                           egp.predict(X_va_s, a_cpk), y_val)
                                gp_rec.update({
                                    'gamma_noise_label_units': sigma2,
                                    'cp_noise_label_units': a_cp * var_y})
                            sp_rec['gp'] = gp_rec
                        except Exception as exc:
                            errors[f'{ekey}|gp'] = f'{type(exc).__name__}: {exc}'
                        # regression kriging on the host's residuals
                        try:
                            # Cached on the label hash (2026-09-25).
                            p_oof = _nc.cached_oof_predict(
                                dataset_name, X_tr_s, y0, g_tr,
                                n_splits=oof_splits)
                            gp_r = make_gp().fit(X_tr_s, y0 - p_oof)
                            host_pred = (fixed_val.get('lightgbm') if host == 'lightgbm'
                                         else None)
                            if host_pred is None:
                                from kirby.find_best_hybrid import get_model
                                host_pred = _fit_predict(get_model('lightgbm', 'regression'),
                                                         X_tr_s, y0, X_va_s, metric_name)
                            _store(lvl_key, space, 'fixed', 'lightgbm_plus_gp_residual',
                                   pass_key, host_pred + gp_r.predict(X_va_s), y_val)
                            if host != 'lightgbm':
                                _store(lvl_key, space, 'fixed', 'lightgbm', pass_key,
                                       host_pred, y_val)
                        except Exception as exc:
                            errors[f'{ekey}|kriging'] = f'{type(exc).__name__}: {exc}'

                    # ── molecule means, with and without a per-molecule term ──
                    if has_repeats:
                        rid = np.asarray(rep_ids)[train_idx]
                        first, ym, kcount, s2e, dof = molecule_means(y0, rid)
                        if gp_cap is not None and len(ym) > gp_cap:
                            skips[f'{ekey}|gp_means'] = (
                                f'{len(ym)} training molecules, above the GP cap '
                                f'of {gp_cap}')
                        elif not np.isfinite(s2e):
                            skips[f'{ekey}|gp_means'] = (
                                'no training molecule has two measurements')
                        else:
                            try:
                                Xm = X_tr_s[first]
                                pred = make_gp().fit(Xm, ym).predict(X_va_s)
                                _store(lvl_key, space, 'fixed', 'gp_means_ml_nugget',
                                       pass_key, pred, y_val)
                                var_m = float(np.var(ym)) or 1.0
                                alpha = (s2e / kcount) / var_m
                                pred = make_gp(alpha).fit(Xm, ym).predict(X_va_s)
                                _store(lvl_key, space, 'fixed',
                                       'gp_means_per_molecule_alpha', pass_key, pred, y_val)
                                sp_rec['gp_means'] = {
                                    'n_molecules': int(len(ym)),
                                    'within_variance_label_units': float(s2e),
                                    'within_dof': dof,
                                    'max_count': int(kcount.max())}
                            except Exception as exc:
                                errors[f'{ekey}|gp_means'] = f'{type(exc).__name__}: {exc}'
                rec['per_space'][space] = sp_rec

            pass_recs.setdefault(lvl_key, []).append(rec)
            ledger.finish_pass(pass_key, {'acc': acc, 'tails': tails,
                                          'pass_recs': pass_recs,
                                          'errors': errors, 'skips': skips})
            print(f"    level {lvl_key} draw {draw} fold {fold_i}: "
                  f"{rec['labels_moved']} of {rec['training_molecules']} labels "
                  f"moved [{time.time() - t0:.0f}s]")

    results['per_level'] = convert_numpy(acc)
    results['tails_per_level'] = convert_numpy(tails)
    results['per_pass'] = convert_numpy(pass_recs)
    results['checkpoint'] = ledger.describe()
    if skips:
        results['arm_skips'] = skips
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, downstream_models, metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _summarise(acc, models, metric_name, lower_is_better):
    """Each arm against `fixed` per model; the stack and the extra pairs."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against the same model at fixed default '
                        'settings, paired per pass; a pass is one (draw, fold)'),
           'per_level': {}}
    for lvl, by_space in acc.items():
        lrow = {}
        for space, by_arm in by_space.items():
            fixed = by_arm.get('fixed') or {}
            clean = by_arm.get('clean') or {}
            srow = {'per_model': {}, 'stack': {}, 'extra_pairs': {}}
            for m in models:
                b = fixed.get(m) or {}
                if not b:
                    continue
                row = {'fixed_mean': _mean_on(b, list(b)), 'per_arm': {}}
                for arm in MODEL_ARMS:
                    if arm == 'fixed':
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    e = {'vs_fixed': _paired(b, g, lower_is_better)}
                    if arm != 'clean':
                        e.update(_share(clean.get(m) or {}, b, g, lower_is_better))
                    row['per_arm'][arm] = e
                srow['per_model'][m] = row
                if fixed.get('stack'):
                    srow['stack'][m] = _paired(b, fixed['stack'], lower_is_better)
            for got, base in EXTRA_PAIRS:
                g, b = fixed.get(got) or {}, fixed.get(base) or {}
                if g and b:
                    srow['extra_pairs'][f'{got}_vs_{base}'] = _paired(
                        b, g, lower_is_better)
            # The similarity-sweep arms (2026-09-25), '<arm>@<setting>[|k<k>]',
            # each paired against what its own arm is paired against.
            for got in [a for a in fixed if '@' in a]:
                stem = got.split('@', 1)[0]
                for g0, base in EXTRA_PAIRS:
                    if g0 != stem:
                        continue
                    g, b = fixed.get(got) or {}, fixed.get(base) or {}
                    if g and b:
                        srow['extra_pairs'][f'{got}_vs_{base}'] = _paired(
                            b, g, lower_is_better)
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; {results['noise_condition']}; paired per pass "
          f"against fixed defaults)")
    for lvl, lrow in s['per_level'].items():
        print(f"\n  level {lvl}:")
        for space, srow in lrow.items():
            for m, row in srow['per_model'].items():
                print(f"    {space}/{m}: fixed {row['fixed_mean']:.4f}")
                for arm, e in row['per_arm'].items():
                    v = e['vs_fixed']
                    print(f"      {arm:20s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, mean "
                          f"{v['mean_gain']:+.4f} (spread "
                          f"{v['spread_over_pass_gains']:.4f})")
            for m, v in srow['stack'].items():
                print(f"    {space}/stack vs {m}: won {v['passes_won']} of "
                      f"{v['n_passes_paired']} passes, mean {v['mean_gain']:+.4f}")
            for name, v in srow['extra_pairs'].items():
                print(f"    {space}/{name}: won {v['passes_won']} of "
                      f"{v['n_passes_paired']} passes, mean {v['mean_gain']:+.4f}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N40: model-side levers under label noise (retuning, seeds, stacking, GP)')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help='One regression condition. Default: sweep '
                             f'{", ".join(DEFAULT_CONDITIONS)}, one file each')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated noise levels. Default: the '
                             "condition's grid")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates (default 0,1,2)')
    parser.add_argument('--tune-n-iter', type=int, default=None,
                        help=f'Settings the tuner tries (default {TUNE_N_ITER})')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Leave out retuned and tuned_clean_reused')
    parser.add_argument('--skip-gp', action='store_true',
                        help='Leave out every GP arm')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds. Overrides the 1 or 5 --mode gives')
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
    if args.tune_n_iter:
        kw['tune_n_iter'] = args.tune_n_iter
    if args.skip_tuning:
        kw['skip_tuning'] = True
    if args.skip_gp:
        kw['skip_gp'] = True

    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, EXP_ID, args.dataset)


if __name__ == '__main__':
    main()
