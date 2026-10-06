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

THE CONDITIONS. `gaussian`, one of the six decided kinds of damage
(KNN_SEARCH_SPACE_RECORD, "The noise grid", Adelaide 2026-10-05), at the
decided levels for the dataset's group (noise_helpers.decided_levels: 0, 0.5,
1.0, 1.5 on full_knn; 0, 1.0 on fast_knn). And `dense_oofmse` from
noiseInject.RELATIVE_CONDITIONS (noise variance a share of lightgbm's
out-of-fold MSE on the fold's clean training labels), n40's own kind: levels
0, 0.1, 0.3, 1.0 on datasets of 5,000 molecules or fewer, 0 and 1.0 above
(Adelaide delegated, 2026-10-06). Swept through noise_helpers.sweep_conditions,
one file per condition; a job can name another regression condition with
--noise-condition, and a kind the noise grid cut is skipped with the reason
written. Every injection goes through noise_helpers.inject_condition. 3 draws
per non-zero level (one under censoring, noise_helpers.draws_for); level 0 is
the labels as delivered and runs once.

THE ARMS, per decided model, every one fitted on the noisy training labels
unless it says otherwise, and graded on held-out molecules against their untouched
labels:

  clean                the model at its fixed defaults on the labels before
                       injection. The most any arm can recover.
  fixed                the model at its fixed defaults (find_best_hybrid.get_model).
                       The arm to beat.
  retuned              the model's default setting plus TUNE_N_ITER settings
                       drawn from TUNE_GRIDS, each scored on TUNE_INNER_FOLDS
                       scaffold-grouped inner folds of THIS FOLD'S TRAINING
                       MOLECULES ONLY; the best refitted on all of them. The
                       held-out fold never reaches the tuner. The default is
                       candidate 0 since 2026-10-05 (dirty-run finding "n40:
                       retuning never tries the default").
  tuned_clean_reused   the same search on the clean training labels, the chosen
                       setting then fitted on the noisy labels. The gap between it
                       and `retuned` is what retuning FOR the noise is worth.
  seed_one             lightgbm, xgboost and dnn with row and feature subsampling
                       (SEED_SUBSAMPLE), one seed.
  seed_avg             the same, SEED_AVG_N seeds, predictions averaged. seed_one
                       is there so the averaging is not confounded with the
                       subsampling that makes the seeds differ.
  rf_1000_trees        rf only: RF_BIG_TREES trees in place of the default 100.

A decided model with no grid in TUNE_GRIDS (gp) has no retuned arm, and one with
no seed or tree setting (svm, gp) has no seed arm.

TUNE_GRIDS widen find_best_hybrid._HP_SEARCH_SPACES toward stronger
regularisation, as the report's D3 #5 asks: more minimum samples per leaf, larger
L2 penalties, lower learning rates with more rounds, more dropout. Early stopping is
not a separate arm; a lower learning rate with the round count in the grid covers it.

Models that are not downstream models, each fitted in the feature matrix of
HOST_MODEL (lightgbm when it is one of the dataset's two decided models, else
the first decided model), stored under arm `fixed`:

  stack                       non-negative least squares on the out-of-fold
                              predictions of every decided model (each in its
                              own feature matrix), applied to their fixed-default
                              held-out predictions. Paired against every model.
  gp_ml_nugget                find_best_hybrid's GP: RBF + WhiteKernel, the noise
                              term fitted by marginal likelihood.
  gp_nugget_from_gamma@<setting>|k<k>
                              the same kernel with the noise term FIXED at the
                              Gamma-test noise variance (noise_helpers.gamma_test)
                              of this fold's training molecules, measured in one
                              neighbour setting with k neighbours. The report's
                              "fix the nugget from a preliminary test on the
                              measurement error".
  gp_nugget_cp@<setting>|k<k> the noise term chosen from GP_NUGGET_MULT x the fitted
                              one by Mallows Cp, RSS/n + 2 sigma^2 tr(S)/n, with
                              sigma^2 that Gamma-test estimate.
  gp_nugget_cv                the same grid chosen by inner-fold error instead.
  lightgbm_plus_gp_residual   regression kriging: fixed lightgbm, plus a GP fitted
                              to lightgbm's out-of-fold residuals.
  knn_k<k>_<scheme>@<setting> the k nearest training labels in one neighbour
                              setting, k in (5, 10, 20), each neighbour counted
                              by one of the two swept schemes (rank,
                              inverse_square; KNN_SEARCH_SPACE_RECORD axes 4
                              and 5).

DECIDED 2026-10-06 (Adelaide delegated): the kNN arms that found neighbours in
the host model's own features (knn_k<k>_<scheme>, knn_k_cv_<scheme>) and the
Gamma estimate in those features are removed; the kNN record's axis 1 cut the
model's own features as a neighbour space. The Mallows-Cp choice of k
(knn_k_cp, in the host features and under every setting) is dropped. The GP
noise-term arms take sigma^2 from the per-setting Gamma estimates only.

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
n x n kernel. They run in every fitted configuration, never on PCA.

THE CONFIGURATIONS (aligned 2026-10-05, chat 5; HANDOFF_EXPERIMENT_ALIGNMENT.md
2.4). Per outer fold (noise_helpers.noise_outer_folds) and per decided model
(DECISIONS.md Step 1, two per dataset): its 2 best single representations and
its 2 best general hybrids, all from searches on that fold's training rows
(noise_helpers.fold_configurations), so 2 models x 4 = 8 fitted configurations
(R5). Named base_1, base_2, hybrid, hybrid__general2
(noise_helpers.CONFIGURATION_SPACES). Each hybrid is rebuilt on the fold's
training rows from THIS pass's noisy labels with the per-fold recipe and its
own representation scores (noise_helpers.fold_model_inputs). Which
representations and recipes each fold used is under `folds` in the result.
Until 2026-10-05 the five models of DOWNSTREAM_MODELS (plus rf and svm) each
had 2 single representations and up to 6 hybrids chosen ONCE from the
whole-dataset stored search (get_best_search_for_downstream, pool_recipes_by_
family), reused on every fold.

THE NEIGHBOUR SPACES. The Gamma estimate and the kNN arms are computed under
every neighbour setting of the dataset's group (n07.sweep_settings: the
group's spaces and distances, no importance weighting, scaffold-mates kept in
the lists), each read through the host model's per-fold spaces. The distance
search's top hybrid (hybrid__distance1) and the similarity search's top hybrid
(hybrid_similarity) enter here ONLY, as neighbour spaces; no model is fitted
on them (handoff 2.3 item 1). No model is refitted per setting: every
@<setting> arm is a neighbour mean or a closed-form GP prediction, so every
setting is scored and the per-dataset screen (refit the top 3) has nothing to
cut here.

Ridge is not used anywhere (the owner's no-ridge rule); the report's Cp example on
ridge is replaced by the two linear smoothers the model set already has.

THE GAMMA ESTIMATE. noise_helpers.gamma_test on distances in each neighbour
setting, k = 1..<the setting's k> neighbours, k in (5, 10, 20). Until
2026-10-06 it also ran in the host model's own features (the process-wide
distance, k = 1..10); removed with the own-feature kNN arms. Above
GAMMA_MAX_MOLECULES training molecules it
runs on a seeded subset of that many. Its intercept is an upper bound on the
label noise: it also holds variation the features cannot resolve.

ROBUSTNESS. Each summary row carries damaged_minus_clean: the fixed arm's
score minus the clean arm's score, per outer fold, in the dataset's own metric
(read the sign with lower_is_better). On full_knn datasets the summary also
carries auc_norm (kirby.noise_robustness), one curve per outer fold, per space,
arm and model. The noise-grid decision (2026-10-05) asks for both on full_knn
and the plain difference only on fast_knn.

Usage:
    python n40_model_side_regularisation.py --dataset logd --mode full
    python n40_model_side_regularisation.py --dataset caco2 --noise-condition dense_oofmse
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import zlib

import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold, ParameterSampler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_levels, noise_groups_for
from experiments.noise.noise_helpers import (
    sweep_conditions, tail_scores, RELATIVE_CONDITIONS, relative_levels,
    gamma_test,
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, inject_condition, decided_levels,
    dataset_knn_group, level_unit, CONFIGURATION_SPACES,
    DECIDED_REGRESSION_CONDITIONS, draws_for,
)
from experiments.noise.n07_remediation import (
    _predict_for_metric, sweep_settings, sweep_ks, sweep_schemes,
)
from experiments.noise.n30_censored_likelihood import _paired, _mean_on
from experiments.noise.n33_robust_training import _share
from experiments.noise._ground_truth_providers import build_replicate_groups
from kirby.noise_spec import GROUPED_STRATEGIES
# The per-pass checkpoint, the cached fits and the neighbour spaces shared by
# n32-n41 (n32's shared block, 2026-09-25). n40 changes the MODEL, never the
# training labels, so it writes no D-versus-D' rows: there is no D'.
from experiments.noise.n32_iterative_and_dual_selection import (
    fit_predict_cached, ledger_signature, PassLedger, restart_flag,
    fold_similarity, _nc, _ss,
)

EXP_ID = 'n40_model_side_regularisation'

# Identity of this version's checkpoints. It is in the result header, which
# ledger_signature hashes, so a checkpoint written before the alignment is set
# aside and never read back (handoff 2.4 item 11).
# 2026-10-06: dense_oofmse levels split by molecule count, own-feature kNN and
# Gamma removed, knn_k_cp dropped, one draw under censoring.
RESULT_LAYOUT = 'fold_aligned_2026-10-06'

# DECIDED 2026-10-06 (Adelaide delegated): dense_oofmse is kept as n40's own
# kind (noise on every label, sized by lightgbm's out-of-fold error; f37 never
# measured it). Levels 0, 0.1, 0.3, 1.0 (shares of the out-of-fold MSE) on
# datasets of DENSE_OOFMSE_LARGE_N molecules or fewer; 0 and 1.0 on datasets
# above that. The count is distinct molecules when SMILES are loaded, as
# noise_helpers.dataset_knn_group counts them. The unit is measured in the dataset's first pruned representation as loaded
# (noise_helpers.inject_condition), one choice for every fold and model.
# Until 2026-10-05 it was the rank-1 single representation of the first model
# in the whole-dataset search, which the owner never signed off.
DEFAULT_CONDITIONS = ('gaussian', 'dense_oofmse')
DEFAULT_NOISE_CONDITION = 'gaussian'
DENSE_OOFMSE_LARGE_N = 5000
DENSE_OOFMSE_LARGE_LEVELS = (0.0, 1.0)

# The sixteen kinds the noise grid cut (KNN_SEARCH_SPACE_RECORD, "The noise
# grid", Adelaide 2026-10-05). A job naming one gets a skip with the reason.
CUT_BY_NOISE_GRID = frozenset({
    'laplace', 'student_t_nu3', 'student_t_nu5', 'student_t_nu10',
    'outlier_p01', 'outlier_p10', 'outlier_only_p05', 'grouped_wider',
    'shift_only_p05', 'sparse_dprime_p02', 'sparse_dprime_p05',
    'sparse_dprime_p10', 'sparse_dprime_up_p02', 'sparse_dprime_up_p05',
})
# Kinds noise_helpers.inject_condition cannot inject: hetero_seen needs each
# molecule's heavy-atom count, which inject_condition does not take.
NOT_INJECTABLE = {'hetero_seen': (
    'noise_helpers.inject_condition takes no heavy-atom counts, which '
    'hetero_seen sizes its noise by')}

# The replicate index of the injection, as n07, n11, n13, n24 and n33.
NOISE_DRAWS = (0, 1, 2)

# The tuner: the model's default setting, then this many settings, each scored
# on this many scaffold-grouped inner folds of the training molecules.
# ParameterSampler with a fixed seed, so every fold and every pass tries the
# same settings.
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

# DECIDED 2026-10-06 (Adelaide delegated): the GP, kNN and kriging arms stay
# in the host model's features, as before. They are models outside the
# dataset's two decided ones, fitted as this experiment's own levers (gp runs
# on every fold of 1,000 training molecules or fewer; the kriging arm fits a
# lightgbm even where lightgbm is not decided). The host is lightgbm where it
# is decided, else the first decided model.
HOST_MODEL = 'lightgbm'

# DECIDED 2026-10-06 (Adelaide delegated): kNN arms take k from (5, 10, 20),
# the neighbour counts of the kNN record's axis 4. Until 2026-10-06 knn_k_cp
# and knn_k_cv chose from (1, 3, 5, 7, 11, 15, 21, 31); both arms are removed
# (knn_k_cv found neighbours in the host model's own features; knn_k_cp is
# dropped), so every kNN arm now reads each of these three counts.
KNN_K_GRID = (5, 10, 20)
GP_NUGGET_MULT = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
GP_JITTER = 1e-6
GAMMA_MAX_MOLECULES = 5000

GAP_MIN_FRAC = 0.02

MODEL_ARMS = ('clean', 'fixed', 'retuned', 'tuned_clean_reused',
              'seed_one', 'seed_avg', 'rf_1000_trees')

# (arm under test, the arm it is paired against). Both stored under arm 'fixed'.
# gp_nugget_from_gamma and gp_nugget_cp exist only as '<arm>@<setting>|k<k>'
# since 2026-10-06; _summarise pairs each by its stem.
GP_PAIRS = (
    ('gp_nugget_from_gamma', 'gp_ml_nugget'),
    ('gp_nugget_cp', 'gp_ml_nugget'),
    ('gp_nugget_cv', 'gp_ml_nugget'),
    ('gp_nugget_cp', 'gp_nugget_cv'),
    ('lightgbm_plus_gp_residual', 'lightgbm'),
    ('gp_means_per_molecule_alpha', 'gp_means_ml_nugget'),
)


def extra_pairs(schemes=None):
    """GP_PAIRS. Until 2026-10-06 also three kNN pairs per counting scheme
    (knn_k_cp and knn_k_cv against knn_k5, and against each other); those arms
    are removed. `schemes` is accepted and unused."""
    return tuple(GP_PAIRS)


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
    """Pick one setting on scaffold-grouped inner folds: the model's default
    (candidate 0, {}) or one of TUNE_N_ITER drawn from TUNE_GRIDS[name].

    X, y, groups are THIS FOLD'S TRAINING MOLECULES and nothing else. Returns
    (best_params, {'inner_scores', 'best_index', 'chose_default'}).

    2026-10-05 (chat 5, dirty-run finding "n40: retuning never tries the
    default"): the default is a candidate, so `retuned` can only lose to
    `fixed` through the inner folds' choice, never because the default was out
    of reach. Until then the candidates were the drawn settings only.
    """
    if TUNER_HOOK is not None:
        TUNER_HOOK(name, X, y, groups)
    n_iter = TUNE_N_ITER if n_iter is None else int(n_iter)
    inner_folds = TUNE_INNER_FOLDS if inner_folds is None else int(inner_folds)
    grid = TUNE_GRIDS[name]
    settings = [{}] + list(ParameterSampler(grid, n_iter=n_iter,
                                            random_state=TUNE_SEED))
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
    return settings[best], {'inner_scores': scores, 'best_index': best,
                            'chose_default': best == 0}


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


def gamma_sigma2(X, y, k_max, dist_blocks, seed=0):
    """noise_helpers.gamma_test in one neighbour setting; subset above a cap.

    dist_blocks: the setting's training distance as row blocks
    (n32.FoldSimilarity.distance_blocks); k_max: the setting's neighbour count.
    X only sets the number of training molecules.

    2026-10-06 (Adelaide delegated): the estimate in the host model's own
    features (the process-wide distance, k = 1..10, the default when
    dist_blocks was None) is removed with the own-feature kNN arms. Until
    2026-09-25 that default was Euclidean whatever --distance-metric said."""
    y = np.asarray(y, dtype=float)
    idx = np.arange(len(y))
    if len(y) > GAMMA_MAX_MOLECULES:
        idx = np.random.RandomState(seed).choice(len(y), GAMMA_MAX_MOLECULES,
                                                 replace=False)
    D = _sub_distance(dist_blocks, len(y), idx)
    g = gamma_test(D, y[idx], k_max=k_max)
    if g is None:
        return None, None
    g['n_used'] = int(len(idx))
    return max(float(g['intercept']), 0.0), g


# ── kNN under the swept counting schemes (KNN_SEARCH_SPACE_RECORD axis 5) ──
# Every kNN prediction here is the neighbours' label under one swept scheme
# (similarity_space.weighted_neighbour_label), read off a neighbour setting's
# lists. Until 2026-10-05 the host-feature kNN arms were sklearn
# KNeighborsRegressor(weights='uniform') with sklearn's default Euclidean.

def knn_scheme_predict(y_fit, idx, dist, k, scheme):
    """The k nearest neighbours' label under one counting scheme."""
    return _ss.weighted_neighbour_label(
        np.asarray(y_fit, dtype=float), np.asarray(idx)[:, :k],
        np.asarray(dist)[:, :k], scheme)


# Removed 2026-10-06 (Adelaide delegated): _host_lists, knn_cv_choice and
# knn_cp_choice. The first two found neighbours in the host model's own
# features (the kNN record's axis 1 cut that space); the Cp choice of k is
# dropped.


class _EigenGP:
    """The GP posterior mean with a FIXED kernel and a chosen noise term, closed form.

    Takes the signal kernel of a fitted find_best_hybrid GP (kernel_.k1) and
    normalises y as that GP does (normalize_y=True). With K = U diag(lam) U^T,
    the smoother at noise term a is S = U diag(lam / (lam + a)) U^T, so every
    noise term on the grid costs one matrix product after one decomposition.
    `a` is in normalised-label units, as sklearn's WhiteKernel noise_level is.
    The features keep their own dtype (2026-10-05); the kernel is float64.
    """

    def __init__(self, signal_kernel, X, y):
        self.k = signal_kernel
        self.X = np.asarray(X)
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
        return self.mu + self.sd * (self.k(np.asarray(Xq), self.X) @ coef)


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


def _molecule_count(data):
    """Distinct molecules when SMILES are loaded, else rows (the count
    noise_helpers.dataset_knn_group uses)."""
    smi = data.get('smiles')
    if smi is not None and len(smi):
        return len(set(smi))
    return len(data['labels'])


def _condition_levels(data, dataset_name, condition):
    """The levels one condition is swept at on this dataset, 0 first.

    A decided kind: noise_helpers.decided_levels (the noise grid, Adelaide
    2026-10-05). dense_oofmse: relative_levels (0, 0.1, 0.3, 1.0) on datasets
    of DENSE_OOFMSE_LARGE_N molecules or fewer, DENSE_OOFMSE_LARGE_LEVELS
    above (DECIDED 2026-10-06, Adelaide delegated). Another relative kind the
    grid did not measure (hetero_hidden): relative_levels. Anything else a
    job names: the dataset_policy grid it had before."""
    if condition in DECIDED_REGRESSION_CONDITIONS:
        return tuple(decided_levels(data, dataset_name, condition))
    if condition == 'dense_oofmse':
        if _molecule_count(data) > DENSE_OOFMSE_LARGE_N:
            return tuple(DENSE_OOFMSE_LARGE_LEVELS)
        return tuple(relative_levels(condition))
    if condition in RELATIVE_CONDITIONS:
        return tuple(relative_levels(condition))
    return tuple(noise_levels(dataset_name, strategy=condition, is_clf=False))


def _levels_and_draws(data, dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs for one fold. Level 0 is the labels as delivered,
    so one draw; every other level NOISE_DRAWS draws, one under censoring
    (noise_helpers.draws_for, DECIDED 2026-10-06, Adelaide delegated)."""
    if levels is None:
        levels = _condition_levels(data, dataset_name, condition)
    draws = draws_for(condition, NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in sorted({float(l) for l in levels}):
        for d in (draws if lvl > 0 else draws[:1]):
            out.append((lvl, int(d)))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N40 across its conditions, one file per condition."""
    if data['task_type'] != 'regression':
        print(f"  SKIPPED: {dataset_name} is {data['task_type']}. The Gamma "
              f"estimate, Cp and the GP noise term are defined on a number, so "
              f"n40 is regression only.")
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    named = kwargs.get('noise_condition')
    if named and named in CUT_BY_NOISE_GRID:
        print(f"  SKIPPED: {named} is one of the sixteen kinds the noise grid "
              f"cut (KNN_SEARCH_SPACE_RECORD, 2026-10-05).")
        return {'skipped': 'condition_cut_by_noise_grid',
                'noise_condition': named}
    if kwargs.get('noise_conditions'):
        asked = [str(c) for c in kwargs['noise_conditions']]
        cut = [c for c in asked if c in CUT_BY_NOISE_GRID]
        if cut:
            print(f"  dropped {cut}: cut by the noise grid (2026-10-05)")
        kwargs['noise_conditions'] = [c for c in asked
                                      if c not in CUT_BY_NOISE_GRID]
    # sweep_conditions pops force_rerun; PassLedger reads restart_passes.
    kwargs['restart_passes'] = bool(kwargs.get('force_rerun'))
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=list(DEFAULT_CONDITIONS), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N40 under ONE injected condition.

    Per outer fold (noise_helpers.noise_outer_folds):
      1. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels) and gives each decided model 2 single
         representations, 2 general hybrids and the neighbour-space recipes.
      2. Per (level, draw), noise_helpers.inject_condition damages the
         training labels, and fold_model_inputs builds each model's 4
         configurations, the hybrids rebuilt from the damaged labels.
      3. A configuration two models share (the same matrix) is one cell.
      4. Every model arm runs in every cell; the GP and kriging arms in the
         host model's cell of each configuration; the Gamma estimate and the
         kNN arms under every neighbour setting, read through the host
         model's per-fold spaces.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N40: RETUNING, SEED AVERAGING, STACKING, GP NOISE TERMS")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    labels = np.asarray(data['labels'], dtype=float)
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        return {'skipped': 'not_regression', 'task_type': task_type}
    if condition in CUT_BY_NOISE_GRID:
        return {'skipped': 'condition_cut_by_noise_grid',
                'noise_condition': condition}
    if condition in NOT_INJECTABLE:
        print(f"  SKIPPED: {condition}: {NOT_INJECTABLE[condition]}")
        return {'skipped': 'condition_not_injectable',
                'noise_condition': condition,
                'reason': NOT_INJECTABLE[condition]}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    # Outer folds from config['n_splits'] (KIRBY_NOISE_N_FOLDS overrides), the
    # splits the per-fold searches are filed under. Until 2026-10-05:
    # cv_split with mode 'intermediary' at 1 fold and 'full' otherwise.
    folds, n_folds = noise_outer_folds(data, config)

    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    tune_n_iter = int(kwargs.get('tune_n_iter', TUNE_N_ITER))
    tune_inner = int(kwargs.get('tune_inner_folds', TUNE_INNER_FOLDS))
    seed_avg_n = int(kwargs.get('seed_avg_n', SEED_AVG_N))
    rf_big = int(kwargs.get('rf_big_trees', RF_BIG_TREES))
    skip_tuning = bool(kwargs.get('skip_tuning', False))
    skip_gp = bool(kwargs.get('skip_gp', False))
    skip_knn = bool(kwargs.get('skip_knn', False))
    skip_stack = bool(kwargs.get('skip_stack', False))

    # The dataset's two decided models (DECISIONS.md Step 1, handoff R1).
    # Until 2026-10-05: the config's five plus rf and svm
    # (downstream_model_list). A `models` kwarg (a smoke) can only narrow them.
    decided = list(decided_models(dataset_name, 'general'))
    asked = kwargs.get('models')
    downstream_models = ([m for m in decided if m in asked] if asked
                         else list(decided))
    if asked and [m for m in asked if m not in decided]:
        print(f"  models {[m for m in asked if m not in decided]} dropped: "
              f"not decided for {dataset_name} ({decided})")
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    models_unavailable = {
        m: 'get_model_factories gave no factory (size limit or unknown name)'
        for m in downstream_models if m not in model_factories}
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError("n40: no decided model could be built.")
    host = HOST_MODEL if HOST_MODEL in downstream_models else downstream_models[0]

    # The neighbour settings (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05):
    # the dataset group's spaces and distances, no weighting, ks 5, 10, 20,
    # schemes rank and inverse_square, scaffold-mates kept in every list.
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    knn_group = dataset_knn_group(data, dataset_name)
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sim_settings = list(sweep_settings(dataset_name, n_molecules,
                                       has_smiles=has_smiles))
    sim_ks = tuple(int(k) for k in sweep_ks())
    schemes = tuple(sweep_schemes())
    pairs = extra_pairs(schemes)

    noise_groups = noise_groups_for(data)[0]
    if condition in GROUPED_STRATEGIES and noise_groups is None:
        raise RuntimeError(f"n40: {condition!r} needs scaffold groups.")

    rep_ids = build_replicate_groups(data)
    has_repeats = (rep_ids is not None
                   and len(np.unique(rep_ids)) < len(rep_ids))
    gp_cap = gp_max_samples()

    results = {
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': level_unit(condition),
        'knn_group': knn_group,
        'models': downstream_models,
        'models_decided': decided,
        'models_unavailable': models_unavailable,
        'host_model_for_extra_arms': host,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'per outer fold and decided model: the 2 best single '
            'representations and the 2 best general hybrids from searches on '
            'that fold\'s training rows with the clean training labels '
            '(noise_helpers.fold_configurations); which ones, per fold, is '
            'under folds. Each hybrid\'s training-row score is its score_mean '
            'there; held_out_score is the model at its fixed defaults fitted '
            'on the clean training labels at level 0.'),
        'n_folds': n_folds,
        'folds_run': [int(i) for i, _, _ in folds],
        'arms': list(MODEL_ARMS),
        'arm_to_beat': 'fixed',
        'extra_pairs': [list(p) for p in pairs],
        'settings': {
            'tune_n_iter': tune_n_iter, 'tune_inner_folds': tune_inner,
            'tune_seed': TUNE_SEED,
            'tune_default_is_candidate_0': True,
            'tune_grids': {k: {kk: [list(x) if isinstance(x, tuple) else x
                                    for x in vv] for kk, vv in v.items()}
                           for k, v in TUNE_GRIDS.items()},
            'seed_avg_n': seed_avg_n, 'seed_subsample': SEED_SUBSAMPLE,
            'rf_big_trees': rf_big, 'oof_splits': oof_splits,
            'knn_k_grid': list(KNN_K_GRID),
            'knn_schemes': list(schemes),
            'gp_nugget_mult': list(GP_NUGGET_MULT),
            'gamma_max_molecules': GAMMA_MAX_MOLECULES,
            'gp_max_training_molecules': gp_cap,
        },
        'tuning_note': ('The tuner sees only the training molecules of the '
                        'fold, split into scaffold-grouped inner folds. The '
                        'held-out molecules never reach it. Candidate 0 is '
                        'the model\'s default setting.'),
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
        'similarity_settings': [st.name for st in sim_settings],
        'similarity_ks': list(sim_ks),
        'similarity_note': (
            'Under arm fixed, every neighbour setting adds '
            'knn_k<k>_<scheme>@<setting> (the k nearest training labels under '
            'that counting scheme, k in 5, 10, 20), and '
            'gp_nugget_from_gamma@ and gp_nugget_cp@<setting>|k<k>, whose '
            'sigma^2 is the Gamma intercept in that setting with k '
            'neighbours. Scaffold-mates are kept in every list. '
            'hybrid__distance1 and hybrid_similarity are neighbour spaces '
            'only. Removed 2026-10-06 (Adelaide delegated): the kNN arms and '
            'the Gamma estimate in the host model\'s own features '
            '(knn_k<k>_<scheme>, knn_k_cv_<scheme>, gp_nugget_from_gamma and '
            'gp_nugget_cp without @), and knn_k_cp everywhere. No '
            'D-versus-D\' rows: n40 changes the model, not the training '
            'data.'),
        'neighbour_screen': (
            'not run: no arm refits a model per neighbour setting (each '
            '@<setting> arm is a neighbour mean or a closed-form GP '
            'prediction), so every setting is scored'),
        'robustness_note': (
            'damaged_minus_clean in each summary row is the fixed arm\'s '
            'score minus the clean arm\'s, per outer fold, in the dataset\'s '
            'own metric (direction: lower_is_better). On full_knn datasets '
            'summary.auc_norm is the retention score per outer fold, per '
            'space, arm and model; absent on fast_knn by decision.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to this '
            'condition on this dataset. Held-out molecules are graded against '
            'untouched labels in every arm.'),
        'folds': {},
    }

    passes = _levels_and_draws(data, dataset_name, condition,
                               levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    results['noise_draws'] = sorted({d for _, d in passes})
    print(f"  {condition}: {len(labels)} molecules, {len(folds)} of {n_folds} "
          f"folds, models {downstream_models} (host {host}), {knn_group}, "
          f"{len(passes)} (level, draw) passes per fold, "
          f"{len(sim_settings)} neighbour settings")

    acc, tails, pass_recs, errors, skips = {}, {}, {}, {}, {}
    fold_records, prov_acc = {}, {}
    t0 = time.time()
    clean_tuned_cache = {}
    # The per-pass checkpoint, 2026-09-25 (n32's PassLedger; no rows). The
    # in-run cache (clean_tuned_cache) is not stored: its keys are not JSON,
    # and it recomputes to the same values. The signature hashes the header
    # above, which carries RESULT_LAYOUT, so a pre-alignment checkpoint is
    # set aside.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, condition,
                        signature=ledger_signature(
                            data, config, results,
                            [(tr, va) for _, tr, va in folds]),
                        restart=restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        acc, tails, pass_recs, errors, skips = (
            st_['acc'], st_['tails'], st_['pass_recs'], st_['errors'],
            st_['skips'])
        fold_records = st_.get('fold_records') or {}
        prov_acc = st_.get('prov_acc') or {}
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    def _state():
        return {'acc': acc, 'tails': tails, 'pass_recs': pass_recs,
                'errors': errors, 'skips': skips,
                'fold_records': fold_records, 'prov_acc': prov_acc}

    def _store(lvl_key, space, arm, mname, pass_key, pred, y_val):
        s = float(metric_fn(y_val, pred))
        (acc.setdefault(lvl_key, {}).setdefault(space, {}).setdefault(arm, {})
         .setdefault(mname, {}))[pass_key] = s
        (tails.setdefault(lvl_key, {}).setdefault(space, {}).setdefault(arm, {})
         .setdefault(mname, {}))[pass_key] = tail_scores(y_val, pred)
        return s

    for fold_i, train_idx, val_idx in folds:
        fold_keys = [f'l{l:g}d{d}f{fold_i}' for l, d in passes]
        if all(ledger.pass_done(k) for k in fold_keys):
            print(f"\n  fold {fold_i}: every pass on disk in the checkpoint")
            continue
        print(f"\n  === fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows ===")
        # The per-fold searches, on this fold's training rows. Stored per fold
        # (ocv{n}f{i}s{seed}) and shared with every experiment on the dataset.
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            errors[f'f{fold_i}|fold_configurations'] = (
                f'{type(exc).__name__}: {exc}')
            fold_records[str(fold_i)] = {
                'error': f'per-fold searches failed: {type(exc).__name__}: '
                         f'{exc}'}
            continue
        frec = fold_records.get(str(fold_i))
        if not frec or 'models' not in frec:
            frec = fold_records[str(fold_i)] = {
                'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in downstream_models:
            rec_m = configuration_record(entries.get(m))
            rec_m['metric'] = metric_name
            rec_m['lower_is_better'] = bool(lower_is_better)
            frec['models'].setdefault(m, rec_m)
            print(f"    {m}: base {rec_m.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec_m.get('hybrids') or []]}")

        y_val = labels[val_idx]
        y_clean_tr = labels[train_idx]
        g_tr = (np.asarray(groups)[train_idx] if groups is not None
                else np.arange(len(train_idx)))

        for level, draw in passes:
            lvl_key = f'{level:g}'
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if ledger.pass_done(pass_key):
                continue
            print(f"\n    --- {condition} / level {lvl_key} / draw {draw} / "
                  f"fold {fold_i} ---")
            try:
                y0, _cmask, prov = inject_condition(
                    data, dataset_name, condition, level, draw,
                    train_idx=train_idx, groups=noise_groups)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            prov_acc[pass_key] = convert_numpy(prov)
            y0 = np.asarray(y0, dtype=float)
            rec = {'pass': pass_key, 'training_molecules': int(len(y0)),
                   'labels_moved': int(np.sum(np.abs(y0 - y_clean_tr) > 1e-12)),
                   'injected_rms_label_units': float(
                       np.sqrt(np.mean((y0 - y_clean_tr) ** 2))),
                   'per_space': {}}

            # Each model's 4 configurations, the hybrids rebuilt on this
            # fold's training rows from THIS pass's damaged labels with the
            # per-fold recipe (its own representation scores: dirty-run bug
            # 1), float32 kept. 0/1 features stay unscaled in base_1 and
            # base_2 only: the two hybrids come from n07.fold_similarity_spaces,
            # which standardises them with StandardScaler (n07 _scale_pair),
            # while the neighbour-space hybrids (n32 FoldSimilarity._hyb) use
            # FeatureScaler. Shared code, not changed here (2026-10-05).
            inputs = {}
            for m in downstream_models:
                ins, unav, names = fold_model_inputs(
                    data, dataset_name, m, entries.get(m),
                    train_idx=train_idx, val_idx=val_idx, y_train=y0,
                    factory=model_factories[m], metric=metric)
                inputs[m] = ins
                mrec = frec['models'][m]
                if names:
                    mrec.setdefault('configuration_names', dict(names))
                mrec.setdefault('n_features', {})
                for sp, (Xa, _Xb) in ins.items():
                    mrec['n_features'].setdefault(sp, int(np.shape(Xa)[1]))
                for sp in CONFIGURATION_SPACES:
                    if sp not in ins:
                        why = unav.get(sp) or 'not built'
                        (mrec.setdefault('configurations_unavailable', {})
                         .setdefault(sp, why))
                        errors[f'{lvl_key}|d{draw}|f{fold_i}|{sp}/{m}|build'] = (
                            why)

            # A configuration two models share (the same matrix) is one cell.
            fold_spaces = {}
            for space in CONFIGURATION_SPACES:
                by_hash = {}
                for m in downstream_models:
                    if space not in inputs[m]:
                        continue
                    X_tr_s, X_va_s = inputs[m][space]
                    h = (_nc.array_hash(np.asarray(X_tr_s)),
                         _nc.array_hash(np.asarray(X_va_s)))
                    if h in by_hash:
                        by_hash[h][2].append(m)
                    else:
                        by_hash[h] = [X_tr_s, X_va_s, [m]]
                if by_hash:
                    fold_spaces[space] = [tuple(v) for v in by_hash.values()]

            # The host model's neighbour spaces for this pass, from the noisy
            # labels: its base and hybrid configurations handed over, the
            # distance and similarity hybrids built from their per-fold
            # recipes.
            fsim = None
            entry_h = entries.get(host)
            if entry_h is not None:
                fsim = fold_similarity(
                    dataset_name, data, reps=data['reps'],
                    train_idx=train_idx, val_idx=val_idx, y_train=y0,
                    groups_train=g_tr,
                    base_by_model={host: list(entry_h.get('base') or [])},
                    recipe_for_space_by_model={
                        host: dict(entry_h.get('recipes') or {})},
                    model_factories=model_factories, metric=metric,
                    task_type=task_type)
                for sp, (Xa, Xb) in inputs[host].items():
                    fsim._spaces[(host, sp)] = (Xa, Xb, None)
                for sp, why in (entry_h.get('unavailable') or {}).items():
                    if sp not in inputs[host]:
                        fsim._spaces[(host, sp)] = (None, None, why)
                        fsim.unavailable[f'{host}|{sp}'] = why
            # Per pass, a setting's Gamma estimate and neighbour lists do not
            # depend on the configuration, except forest proximity, whose
            # forest is grown on the configuration's matrix. Computed once.
            sim_memo, list_memo, block_memo = {}, {}, {}

            for space in CONFIGURATION_SPACES:
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
                                s_ = _store(lvl_key, space, arm, mname,
                                            pass_key, pred, y_val)
                                if arm == 'fixed':
                                    fixed_val[mname] = pred
                                if arm == 'clean' and level == 0.0:
                                    (frec['models'][mname]
                                     .setdefault('held_out_score', {})
                                     [space]) = s_
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
                                # reused across passes of the fold.
                                ck = ((fold_i, space, mname)
                                      if space in ('base_1', 'base_2')
                                      else None)
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
                                    'noisy_chose_default': info_n['chose_default'],
                                    'clean_chose_default': info_c['chose_default'],
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

                # DECIDED 2026-10-06 (Adelaide delegated): no Gamma estimate
                # and no kNN arm in the host model's own features (the kNN
                # record's axis 1 cut that space as a neighbour setting). The
                # Gamma estimate and the kNN arms below read the decided
                # neighbour settings only. Until 2026-10-06 the host features
                # also carried a Gamma estimate (k = 1..10), knn_k<k>_<scheme>,
                # knn_k_cv_<scheme> and knn_k_cp_<scheme>.
                # ── the Gamma estimate under every (setting, k) ──
                # {'<setting>|k<k>': (setting, sigma^2)}.
                sim_sigma2 = {}
                if not (skip_gp and skip_knn) and fsim is not None:
                    sp_rec['gamma_by_similarity'] = {}
                    for setting in sim_settings:
                        if setting.space == _ss.MODEL_INPUT_SPACE:
                            sp_rec['gamma_by_similarity'][setting.name] = {
                                'unavailable': 'model_input is the host '
                                               'model\'s own features, cut '
                                               'as a neighbour space '
                                               '(2026-10-06)'}
                            continue
                        per_config = setting.space == _ss.FOREST_SPACE
                        bkey = ((setting.name, space) if per_config
                                else setting.name)
                        try:
                            if bkey not in block_memo:
                                block_memo[bkey] = fsim.distance_blocks(
                                    setting, host, X_tr_s)
                            blocks, _n, why = block_memo[bkey]
                            if blocks is None:
                                sp_rec['gamma_by_similarity'][
                                    setting.name] = {'unavailable': why}
                                continue
                            for k in sim_ks:
                                if (bkey, k) not in sim_memo:
                                    sim_memo[(bkey, k)] = gamma_sigma2(
                                        X_tr_s, y0, seed=fold_i, k_max=k,
                                        dist_blocks=blocks)
                                s2_k, g_k = sim_memo[(bkey, k)]
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

                if not skip_knn:
                    # ── kNN under every neighbour setting ──
                    kmax_knn = max(max(KNN_K_GRID), max(sim_ks))
                    for setting in (sim_settings if fsim is not None else ()):
                        if setting.space == _ss.MODEL_INPUT_SPACE:
                            continue
                        per_config = setting.space == _ss.FOREST_SPACE
                        lkey = ((setting.name, space) if per_config
                                else setting.name)
                        try:
                            if lkey not in list_memo:
                                wide = _ss.SimilaritySetting(
                                    setting.space, setting.weighting,
                                    setting.metric, k_max=kmax_knn)
                                list_memo[lkey] = fsim.neighbours(
                                    wide, host, X_tr_s, X_va_s)
                            nb = list_memo[lkey]
                            if not nb.get('available'):
                                continue
                            iv, dv = nb['idx_val'], nb['dist_val']
                            for sch in schemes:
                                for k in sim_ks:
                                    if k > iv.shape[1]:
                                        continue
                                    _store(lvl_key, space, 'fixed',
                                           f'knn_k{k}_{sch}@{setting.name}',
                                           pass_key,
                                           knn_scheme_predict(y0, iv, dv, k,
                                                              sch), y_val)
                        except Exception as exc:
                            errors[f'{ekey}|knn@{setting.name}'] = (
                                f'{type(exc).__name__}: {exc}')
                    # knn_k_cp_<scheme>@<setting> dropped 2026-10-06
                    # (Adelaide delegated).

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
                            # The Gamma-nugget and Cp arms, sigma^2 from
                            # every (setting, k). The versions with sigma^2
                            # from the host features are removed 2026-10-06
                            # (Adelaide delegated).
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

            if fsim is not None and fsim.unavailable:
                (frec.setdefault('neighbour_spaces_unavailable', {})
                 .update(dict(fsim.unavailable)))
            pass_recs.setdefault(lvl_key, []).append(rec)
            ledger.finish_pass(pass_key, _state())
            print(f"    level {lvl_key} draw {draw} fold {fold_i}: "
                  f"{rec['labels_moved']} of {rec['training_molecules']} labels "
                  f"moved [{time.time() - t0:.0f}s]")

    results['folds'] = convert_numpy(fold_records)
    results['injection_provenance'] = convert_numpy(prov_acc)
    results['per_level'] = convert_numpy(acc)
    results['tails_per_level'] = convert_numpy(tails)
    results['per_pass'] = convert_numpy(pass_recs)
    results['checkpoint'] = ledger.describe()
    if skips:
        results['arm_skips'] = skips
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, downstream_models, metric_name, lower_is_better, pairs))
    # The retention score on full_knn datasets, beside the plain difference
    # (noise-grid decision, KNN_SEARCH_SPACE_RECORD 'Levels split by dataset
    # group', 2026-10-05). Absent on fast_knn datasets by that decision.
    if knn_group == 'full_knn':
        results['summary']['auc_norm'] = convert_numpy(
            _auc_norm_by_fold(acc, lower_is_better))
    else:
        results['summary']['auc_norm'] = (
            f'not computed: {knn_group} dataset (noise-grid decision '
            f'2026-10-05: the plain difference only)')
    _print_summary(results, metric_name)
    return results


def _outer_fold_of(pass_key):
    """The outer fold number of a pass key 'l<level>d<draw>f<fold>'."""
    return int(str(pass_key).rsplit('f', 1)[1])


def _damaged_minus_clean(clean_d, fixed_d):
    """The fixed arm's score minus the clean arm's, per outer fold (the mean
    over that fold's draws), in the dataset's own metric; read the sign with
    lower_is_better. noise_helpers.fold_spread over the folds, so one fold
    gives 'one fold, no spread'. The robustness number the noise-grid
    decision asks for on fast_knn datasets (2026-10-05)."""
    by = {}
    for k in sorted(set(clean_d) & set(fixed_d)):
        d = fixed_d[k] - clean_d[k]
        if np.isfinite(d):
            by.setdefault(_outer_fold_of(k), []).append(float(d))
    return fold_spread([float(np.mean(v)) for _f, v in sorted(by.items())])


def _auc_norm_by_fold(acc, lower_is_better):
    """The retention score auc_norm (kirby.noise_robustness.robustness_metrics)
    per space, arm and model, one curve per outer fold.

    One fold's curve: its score at each level (the mean over that level's
    draws) against the level, its level-0 score the clean reference. Level 0
    runs once, as draw 0. robustness_metrics returns NaN below four levels,
    when there is no level 0, or when the level-0 score is at or below zero; no
    skill floor is applied (baseline_threshold=None, as c10). The 'clean' arm
    is left out: it trains on clean labels at every level. noise_helpers.
    fold_spread over the folds, so one fold gives 'one fold, no spread'.
    Called on full_knn datasets only (noise-grid decision, 2026-10-05)."""
    from kirby.noise_robustness import robustness_metrics
    levels = sorted(acc, key=float)
    out = {}
    spaces = {sp for lvl in levels for sp in acc[lvl]}
    for space in sorted(spaces):
        arms = {a for lvl in levels for a in (acc[lvl].get(space) or {})}
        for arm in sorted(arms - {'clean'}):
            names = {m for lvl in levels
                     for m in ((acc[lvl].get(space) or {}).get(arm) or {})}
            for m in sorted(names):
                by_fold = {}
                for lvl in levels:
                    d = ((acc[lvl].get(space) or {}).get(arm) or {}).get(m) or {}
                    for k, v in d.items():
                        if np.isfinite(v):
                            (by_fold.setdefault(_outer_fold_of(k), {})
                             .setdefault(float(lvl), []).append(float(v)))
                per_fold = {}
                for f, curve in sorted(by_fold.items()):
                    xs = sorted(curve)
                    r = robustness_metrics(
                        xs, [float(np.mean(curve[x])) for x in xs],
                        baseline_threshold=None,
                        lower_is_better=lower_is_better)
                    per_fold[f] = r['auc_norm']
                vals = [v for v in per_fold.values() if np.isfinite(v)]
                e = {'per_fold': {str(f): (None if not np.isfinite(v) else
                                           float(v))
                                  for f, v in per_fold.items()},
                     'over_folds': fold_spread(vals) if vals else
                     {'note': 'auc_norm undefined on every fold (fewer than '
                              'four levels, no level 0, or a level-0 score '
                              'at or below zero)'}}
                out.setdefault(space, {}).setdefault(arm, {})[m] = e
    return out


def _summarise(acc, models, metric_name, lower_is_better, pairs=None):
    """Each arm against `fixed` per model; the stack and the extra pairs.

    Spreads are over outer folds (n30._paired, noise_helpers.fold_spread). No
    significance test."""
    pairs = tuple(pairs if pairs is not None else extra_pairs(sweep_schemes()))
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
                row = {'fixed_mean': _mean_on(b, list(b)), 'per_arm': {},
                       'damaged_minus_clean': _damaged_minus_clean(
                           clean.get(m) or {}, b)}
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
            for got, base in pairs:
                g, b = fixed.get(got) or {}, fixed.get(base) or {}
                if g and b:
                    srow['extra_pairs'][f'{got}_vs_{base}'] = _paired(
                        b, g, lower_is_better)
            # The neighbour-setting arms (2026-09-25), '<arm>@<setting>[|k<k>]',
            # each paired against what its own arm is paired against.
            for got in [a for a in fixed if '@' in a]:
                stem = got.split('@', 1)[0]
                for g0, base in pairs:
                    if g0 != stem:
                        continue
                    g, b = fixed.get(got) or {}, fixed.get(base) or {}
                    if g and b:
                        srow['extra_pairs'][f'{got}_vs_{base}'] = _paired(
                            b, g, lower_is_better)
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _spread_text(fs):
    """'sd 0.0123 over 5 folds' or the fold_spread note."""
    if not fs or fs.get('sd') is None:
        return (fs or {}).get('note') or 'no spread'
    return f"sd {fs['sd']:.4f} over {fs['n_folds']} folds"


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; {results['noise_condition']}; paired per pass "
          f"against fixed defaults)")
    for lvl, lrow in s['per_level'].items():
        print(f"\n  level {lvl}:")
        for space, srow in lrow.items():
            for m, row in srow['per_model'].items():
                dmc = row.get('damaged_minus_clean') or {}
                print(f"    {space}/{m}: fixed {row['fixed_mean']:.4f}; "
                      f"damaged minus clean "
                      f"{(dmc.get('mean') if dmc.get('mean') is not None else float('nan')):+.4f} "
                      f"({_spread_text(dmc)})")
                for arm, e in row['per_arm'].items():
                    v = e['vs_fixed']
                    print(f"      {arm:20s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, mean "
                          f"{v['mean_gain']:+.4f} "
                          f"({_spread_text(v.get('gain_over_folds'))})")
            for m, v in srow['stack'].items():
                print(f"    {space}/stack vs {m}: won {v['passes_won']} of "
                      f"{v['n_passes_paired']} passes, mean {v['mean_gain']:+.4f}")
            for name, v in srow['extra_pairs'].items():
                print(f"    {space}/{name}: won {v['passes_won']} of "
                      f"{v['n_passes_paired']} passes, mean {v['mean_gain']:+.4f}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-fold gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N40: model-side levers under label noise (retuning, seeds, stacking, GP)')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help='One regression condition. Default: sweep '
                             f'{", ".join(DEFAULT_CONDITIONS)}, one file each')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated noise levels. Default: the '
                             "condition's decided grid")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates (default '
                             '0,1,2; censoring runs the first only)')
    parser.add_argument('--tune-n-iter', type=int, default=None,
                        help=f'Settings the tuner draws besides the default '
                             f'(default {TUNE_N_ITER})')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Leave out retuned and tuned_clean_reused')
    parser.add_argument('--skip-gp', action='store_true',
                        help='Leave out every GP arm')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds; sets config n_splits '
                             '(KIRBY_NOISE_N_FOLDS overrides it)')
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
