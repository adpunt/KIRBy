#!/usr/bin/env python3
"""N33: does a robust loss, a target transform, a weight or an augmentation undo injected label noise without finding the noisy molecules?

Paper: Noise.

THE QUESTION. Every cleaning method in n07, n11, n24 and n32 first decides WHICH
training molecules are wrong. This file tries the other road: leave every label
where it is, and change how the model reads them. The second research report
(2026-09-24, `compass_artifact_wf-d7ecc1f8-…`) lists these as the levers that do
not depend on the model being accurate enough to find the bad rows: its A4 (log
or Box-Cox target), A6 (robust losses with a tuned scale), A10 (weights from a
predicted spread), D (Tukey biweight, Student-t likelihood), F and H (C-Mixup,
target-noise bagging) and G (fit toward a robust teacher). It ties the gain from
a robust loss to how heavy-tailed the residuals are, so the residual kurtosis is
written beside every pass.

THE CONDITIONS. The six regression kinds of damage the noise grid keeps
(noise_helpers.conditions_for_task: gaussian, outlier_p05, grouped_shifted,
censoring, shift_only_p10, sparse_dprime_up_p10; KNN_SEARCH_SPACE_RECORD "The
noise grid", Adelaide 2026-10-05), one at a time, through
noise_helpers.sweep_conditions, so a job can name one with --noise-condition
and an interrupted sweep resumes. Each is injected through
noise_helpers.inject_condition at noise_helpers.decided_levels, 3 draws per
non-zero level and level 0 (the labels as delivered) once.
Until 2026-10-05 n33 swept every registry condition, including `laplace` and
the `student_t_*` shapes, where the report expects a robust loss to help. Both
are cut from the noise grid; see the OPEN note at DEFAULT_NOISE_CONDITION.

THE SCALE EVERY ROBUST LOSS IS SET FROM. In each fold and feature space, p is a
group-aware out-of-fold LightGBM prediction on the NOISY training labels
(`kirby.utils.noisy_labels.oof_predict`, the same p n30 uses), and

    s = 1.4826 * median |y - p|

which is the residual SD a normal would have if the residuals were normal. The
Huber thresholds are c * s for c in HUBER_C. A Huber threshold in raw label units
would mean a different thing on pKi and on logD; this makes it the same share of
the model's own error everywhere.

THE ARMS. Every arm is scored on the same held-out molecules against their
untouched labels, and paired against `noisy`: the same model on the same
features fitted on the noisy labels as they are.

  clean                   the same model on the labels before injection. The
                          most any arm can recover.
  noisy                   the arm to beat.

  Label and weight arms, every downstream model:
  yeo_johnson             the training labels put through a Yeo-Johnson power
                          transform (sklearn PowerTransformer, fitted on this
                          fold's noisy training labels), the model fitted on
                          that, its predictions transformed back. The report
                          asked for log or Box-Cox; both are undefined on
                          negative labels, and logd's are negative.
  invvar_ngboost          weight 1 / sigma(x)^2, sigma from an out-of-fold
                          NGBoost (Normal) on the noisy labels, normalised to
                          average one. A molecule NGBoost thinks sits in a noisy
                          region counts for less.
  c_mixup                 C-Mixup (Yao et al., NeurIPS 2022). Every training
                          molecule gets one partner, drawn with probability
                          proportional to exp(-(y_i - y_j)^2 / (2 bw^2)), and
                          one new molecule is made from the pair,
                          lam * (x_i, y_i) + (1 - lam) * (x_j, y_j) with
                          lam ~ Beta(2, 2). The training set doubles. bw is
                          CMIXUP_BW_SD label SDs, a stated default, not tuned.
  target_noise_bagging    TNB_BAGS fits, each on the labels plus fresh
                          N(0, (TNB_SCALE * sd(y - p))^2), predictions averaged.
                          No row is resampled; only the labels are perturbed.
  huber_teacher_blend     fitted to 0.5 * y + 0.5 * q, q an out-of-fold LightGBM
                          Huber prediction at c = 1. The report's G: a student
                          fitted toward a robust teacher.

  Loss changes (the "swaps"), each paired against the model it replaces, on
  the noisy labels, in that model's feature matrix:
  lightgbm_huber_c<c>     LightGBM objective huber, alpha = c * s
  lightgbm_fair           LightGBM objective fair, fair_c = s
  lightgbm_quantile_median  LightGBM objective quantile, alpha 0.5 (least
                          absolute error)
  lightgbm_tukey          Tukey's biweight as a LightGBM custom objective, with
                          threshold TUKEY_K * s. See tukey_objective for why its
                          second derivative is floored.
  lightgbm_expectile_t<t> an asymmetric squared loss (expectile regression) as a
                          LightGBM custom objective, at t in EXPECTILE_TAUS. Added
                          2026-09-25 from the third research report (its B3),
                          which names it for errors in one direction. Of the
                          conditions n33 sweeps, censoring is the one-direction
                          case: it moves labels down, so t = 0.6 leans toward the
                          molecules above the prediction. Both sides are run
                          because a real dataset does not say which way it errs.
  xgboost_pseudohuber_c<c>  XGBoost reg:pseudohubererror, huber_slope = c * s
  xgboost_absolute        XGBoost reg:absoluteerror
  rf_l1                   a random forest with absolute-error splits
  dnn_huber_c<c>          the DNN under a Huber loss with delta = c * s
                          (`DNNRegressor(loss='huber')`, added 2026-09-24)
  ngboost_t               NGBoost with a Student-t likelihood (3 degrees of
                          freedom, fixed; see make_ngboost), paired against
                          ngboost_normal, NGBoost with its default normal. Both
                          are fitted in lightgbm's feature matrices where
                          lightgbm is a decided model, otherwise in the first
                          decided model's (2026-10-05). NGBoost is not a
                          downstream model and has no recipe of its own.

The swaps other than rf_l1 have no entry in `find_best_hybrid.DEFAULT_MODELS`.
Each is built here from the settings of the model it replaces
(`_get_lgbm_regressor`, `_get_xgb_regressor`, `_get_dnn_regressor`), changing
only the loss, so the pair differs in the loss and in nothing else. KIRBy changes
the input, not the model, so every swap is a comparator for the label arms, not
a proposal. A swap runs only where the model it replaces is one of the
dataset's two decided models (2026-10-05); svm and gp have no swap.

THE RECOVERY SHARE. Per (level, space, model), over the passes all three scored,
(noisy - arm) / (noisy - clean) where lower wins, the sign flipped where higher
wins. Refused, with its reason, when the clean-to-noisy gap is under 2% of the
clean score (n30's floor, from n07 and n11). For a swap the denominator is the
replaced model's own gap.

TAIL ERROR. Every score is written with `noise_helpers.tail_scores`: the rmse on
the held-out molecules whose clean label is in the top and the bottom tenth, and
the calibration slope. A robust loss that wins overall by pulling the extremes
inward shows up there.

WHERE THE FEATURES COME FROM (2026-10-05, chat 5, HANDOFF_EXPERIMENT_ALIGNMENT
2.4). The models are the dataset's two decided models (DECISIONS.md Step 1,
decided_grid.decided_models). Per outer fold (noise_helpers.noise_outer_folds)
and per model, four feature spaces, all chosen by searches on that fold's
training rows with the clean labels (noise_helpers.fold_configurations):
`base_1` and `base_2`, the model's 2 best single representations, and `hybrid`
and `hybrid__general2`, its 2 best general hybrids (a second hybrid whose
selected features equal the first's is skipped for the next). 2 models x 4 =
8 fitted configurations. Each hybrid is rebuilt on the fold's training rows
from the NOISY training labels of the pass (noise_helpers.fold_model_inputs),
once per pass, and every arm of that model shares it. n33 finds no
neighbours, so the distance hybrid and the similarity hybrid are not used.
Until 2026-10-05 the recipes and base representations were read from the
whole-dataset stored searches of five models (up to 6 hybrids per model).
Which configurations each fold used is under `folds` in the result file.

Usage:
    python n33_robust_training.py --dataset herg_ki --mode full
    python n33_robust_training.py --dataset logd --noise-condition student_t_nu3
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time

import numpy as np
from scipy.stats import kurtosis
from sklearn.preprocessing import PowerTransformer

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, as_rep_dtype,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import (
    noise_groups_for, noise_dose_units,
)
# CHAT 5 ALIGNMENT (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT 2.4): folds, the
# 8 fitted configurations per fold, and the decided noise grid all come from
# noise_helpers' shared block. Removed: cv_split, DOWNSTREAM_MODELS,
# get_best_search_for_downstream, noise_base_reps_by_model,
# pool_recipe_per_downstream_model, pool_recipes_by_family and the per-cell
# hybrid build (whole-dataset searches, five models).
from experiments.noise.noise_helpers import (
    sweep_conditions, conditions_for_task, tail_scores,
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, inject_condition, decided_levels,
    decided_conditions, dataset_knn_group, level_unit, CONFIGURATION_SPACES,
)
from experiments.noise.n07_remediation import _predict_for_metric
from experiments.noise.n30_censored_likelihood import (
    _paired, _mean_on, recovery_share,
)
# The D-versus-D' rows, per-pass checkpoint, cached fits and controls shared
# by n32-n41 (n32's shared block, 2026-09-25).
from experiments.noise.n32_iterative_and_dual_selection import (
    fit_predict_cached, ledger_signature, PassLedger, restart_flag, arm_rows,
    arm_result, control_twins, twin_scores, controls_enabled, replicate_seed,
)
from experiments.noise import noise_cache as _nc
from kirby.noise_spec import GROUPED_STRATEGIES

EXP_ID = 'n33_robust_training'

# One condition when a caller asks for one run without a sweep.
DEFAULT_NOISE_CONDITION = 'gaussian'
# OPEN (chat 5): n33's question names `laplace` and the `student_t_*` shapes as
# the damage where the report expects a robust loss to help (heavy-tailed
# residuals). The noise grid decision (2026-10-05) cut both. Options: (a) n33
# sweeps the six decided kinds only (what the code does now; outlier_p05 is the
# heavy-tailed kind left); (b) n33 also sweeps laplace and student_t_nu3 at the
# decided levels, as an exception for this question. A caller can still name
# one with --noise-condition; it then runs at the decided levels with a note.

# Identity of this version's checkpoints and condition files. Files written
# before the alignment carry another signature and are set aside, not read
# back (handoff 2.4 item 11).
ALIGNMENT_VERSION = 'chat5_2026-10-05'

# The replicate index of the injection. n07, n11, n13 and n24 sweep the same three.
NOISE_DRAWS = (0, 1, 2)

# Robust-loss thresholds, in units of s = 1.4826 * median |y - p|. The report's
# A6 asks for "a grid of multiples of MAD(residual)".
HUBER_C = (0.5, 1.0, 1.5, 2.0)

# Tukey's biweight threshold, in units of s. 4.685 is the textbook constant that
# gives 95% efficiency at the normal (Beaton & Tukey 1974).
TUKEY_K = 4.685

# Tukey's second derivative, (1 - u^2)(1 - 5u^2) with u = r / threshold, is
# NEGATIVE for |u| between 1/sqrt(5) and 1 and zero beyond 1. A gradient booster
# divides the gradient by the summed second derivative, so a leaf whose rows all
# sit out there gets a negative or zero denominator. It is floored at this, which
# makes the step for those leaves a small gradient step instead. Stated because it
# is a choice: the loss being minimised is still Tukey's; only the step size is not
# Newton's in those leaves.
TUKEY_HESS_FLOOR = 1e-2

# C-Mixup kernel bandwidth, in SDs of this fold's noisy training labels. Yao et al.
# tune it per dataset; this is a fixed default, stated as one.
CMIXUP_BW_SD = 0.2
CMIXUP_BETA = 2.0

# Target-noise bagging: how many fits, and the added noise as a share of the SD of
# the out-of-fold residual y - p.
TNB_BAGS = 5
TNB_SCALE = 0.5

# The weight of the teacher in huber_teacher_blend, and which Huber threshold (in
# units of s) the teacher uses.
TEACHER_WEIGHT = 0.5
TEACHER_HUBER_C = 1.0

# Inner folds for p, the teacher and NGBoost's spread. Same as n19 and n30.
OOF_SPLITS = 5

# Floor under the NGBoost spread before 1 / sigma^2, in units of s, so one molecule
# with a near-zero predicted spread cannot take the whole weight.
SIGMA_FLOOR_S = 0.05

GAP_MIN_FRAC = 0.02

LABEL_ARMS = ('yeo_johnson', 'invvar_ngboost', 'c_mixup',
              'target_noise_bagging', 'huber_teacher_blend')

# Arms a model's fit cannot take. invvar_ngboost passes sample_weight, and
# sklearn's GaussianProcessRegressor.fit has no sample_weight (checked
# 2026-10-05), so the arm is undefined for gp. gp is decided on abagym_Ang2,
# abagym_VEGF, ginkgo_acsins, ginkgo_hic and ginkgo_pr_cho.
ARMS_UNDEFINED_FOR = {'gp': {'invvar_ngboost': (
    'GaussianProcessRegressor.fit takes no sample_weight')}}

# NGBoost is fitted in this model's feature matrices when it is one of the
# dataset's two decided models. Before 2026-10-05 lightgbm was always among
# the five models, so the NGBoost pair (ngboost_t against ngboost_normal) ran
# on every regression dataset. lightgbm is decided on only 5 of 27 regression
# datasets, so on the others the pair is fitted in the first decided model's
# feature matrices instead (2026-10-05). NGBoost is not searched and is not
# one of the 8 fitted configurations; it only borrows a host's matrices.
# OPEN (chat 5): which decided model hosts the pair where lightgbm is not
# decided. Now: the first in decided_models order. The host is written to
# the result file as ngboost_host.
NGBOOST_HOST = 'lightgbm'


def ngboost_host(downstream_models):
    """The model whose feature matrices the NGBoost pair is fitted in:
    lightgbm when decided, otherwise the first decided model."""
    if NGBOOST_HOST in downstream_models:
        return NGBOOST_HOST
    return downstream_models[0] if downstream_models else None

# Expectile levels. 0.5 is the plain squared loss, so it is not repeated here.
EXPECTILE_TAUS = (0.4, 0.6)

# NGBoost stages, the registry's 500 (find_best_hybrid.DEFAULT_MODELS['ngboost']).
# NGBoost's own TFixedDf fixes the degrees of freedom at 3.
NGB_ESTIMATORS = 500
NGB_T_DF = 3


def build_swaps(huber_c):
    """{swap name: the downstream model it replaces}."""
    out = {}
    for c in huber_c:
        out[f'lightgbm_huber_c{c:g}'] = 'lightgbm'
    out['lightgbm_fair'] = 'lightgbm'
    out['lightgbm_quantile_median'] = 'lightgbm'
    out['lightgbm_tukey'] = 'lightgbm'
    for t in EXPECTILE_TAUS:
        out[f'lightgbm_expectile_t{t:g}'] = 'lightgbm'
    for c in huber_c:
        out[f'xgboost_pseudohuber_c{c:g}'] = 'xgboost'
    out['xgboost_absolute'] = 'xgboost'
    out['rf_l1'] = 'rf'
    for c in huber_c:
        out[f'dnn_huber_c{c:g}'] = 'dnn'
    return out


def robust_scale(y, p):
    """s = 1.4826 * median |y - p|: the residual SD if the residuals were normal."""
    r = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    return float(1.4826 * np.median(np.abs(r)))


def residual_kurtosis(y, p):
    """Pearson kurtosis of y - p. 3.0 for normal residuals; above 3, heavier tails."""
    r = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    if r.size < 4 or np.std(r) == 0:
        return float('nan')
    return float(kurtosis(r, fisher=False))


def tukey_objective(threshold, hess_floor=TUKEY_HESS_FLOOR):
    """LightGBM custom objective for Tukey's biweight at `threshold` label units.

    With r = prediction - label and u = r / threshold, for |u| < 1:
        gradient  = r (1 - u^2)^2
        hessian   = (1 - u^2)(1 - 5 u^2), floored at hess_floor
    and both are zero for |u| >= 1 (the hessian then takes the floor). A row whose
    residual is past the threshold stops pulling the fit at all -- that is what a
    redescending loss does, and why it has to start from a sensible prediction:
    see fit_tukey, which fits around the median.
    """
    c = float(threshold)

    def _obj(y_true, y_pred):
        r = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
        u = r / c
        inside = np.abs(u) < 1.0
        one_m = 1.0 - u * u
        grad = np.where(inside, r * one_m * one_m, 0.0)
        hess = np.where(inside, one_m * (1.0 - 5.0 * u * u), 0.0)
        return grad, np.maximum(hess, hess_floor)
    return _obj


def expectile_objective(tau):
    """LightGBM custom objective for the expectile loss at level tau.

    loss = w * (y - f)^2 with w = tau where the label is above the prediction and
    1 - tau where it is below. With r = f - y: gradient 2 w r, hessian 2 w.
    """
    t = float(tau)

    def _obj(y_true, y_pred):
        r = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
        w = np.where(r <= 0.0, t, 1.0 - t)
        return 2.0 * w * r, 2.0 * w
    return _obj


class _Centred:
    """A regressor fitted to y - centre, predicting prediction + centre.

    LightGBM does not boost from the average under a custom objective, so Tukey's
    loss would start every row at 0, where on pKi every residual is past the
    threshold and every gradient is zero. Centring at the median fixes the start.
    """

    def __init__(self, model):
        self.model = model
        self.centre_ = 0.0

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        self.centre_ = float(np.median(y))
        self.model.fit(X, y - self.centre_, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X), dtype=float) + self.centre_


def make_swap(name, s, huber_c):
    """A fresh instance of one loss-changed model, built from its parent's settings.

    `s` is the robust residual scale in label units for this fold and space.
    Raises for an unknown name.
    """
    from kirby.find_best_hybrid import (
        _get_lgbm_regressor, _get_xgb_regressor, get_model)
    s = max(float(s), 1e-6)
    if name.startswith('lightgbm_huber_c'):
        c = float(name[len('lightgbm_huber_c'):])
        return _get_lgbm_regressor().set_params(objective='huber', alpha=c * s)
    if name == 'lightgbm_fair':
        return _get_lgbm_regressor().set_params(objective='fair', fair_c=s)
    if name == 'lightgbm_quantile_median':
        return _get_lgbm_regressor().set_params(objective='quantile', alpha=0.5)
    if name == 'lightgbm_tukey':
        return _Centred(_get_lgbm_regressor().set_params(
            objective=tukey_objective(TUKEY_K * s)))
    if name.startswith('lightgbm_expectile_t'):
        t = float(name[len('lightgbm_expectile_t'):])
        return _Centred(_get_lgbm_regressor().set_params(
            objective=expectile_objective(t)))
    if name.startswith('xgboost_pseudohuber_c'):
        c = float(name[len('xgboost_pseudohuber_c'):])
        return _Centred(_get_xgb_regressor().set_params(
            objective='reg:pseudohubererror', huber_slope=c * s, base_score=0.0))
    if name == 'xgboost_absolute':
        return _Centred(_get_xgb_regressor().set_params(
            objective='reg:absoluteerror', base_score=0.0))
    if name == 'rf_l1':
        return get_model('rf_l1', 'regression')
    if name.startswith('dnn_huber_c'):
        c = float(name[len('dnn_huber_c'):])
        from kirby.models.dnn import DNNRegressor
        return DNNRegressor(loss='huber', huber_delta=c * s)
    raise ValueError(f'n33: unknown swap {name!r}')


def make_ngboost(dist_name):
    """NGBoost at the registry's settings, with a Normal or a Student-t likelihood.

    The Student-t is ngboost's TFixedDf, degrees of freedom fixed at NGB_T_DF.
    Its free-df `T` was tried first and MEASURED 2026-09-24 on 100 toy molecules:
    the df parameter diverged and the fit raised "Input y contains NaN" inside
    the base learner.
    """
    from ngboost import NGBRegressor
    from ngboost.distns import Normal, TFixedDf
    dist = {'normal': Normal, 't': TFixedDf}[dist_name]
    return NGBRegressor(Dist=dist, n_estimators=NGB_ESTIMATORS,
                        learning_rate=0.05, random_state=42, verbose=False)


def oof_ngboost_sigma(X, y, groups, n_splits, seed=0):
    """Out-of-fold NGBoost (Normal) predicted spread for every training row."""
    from kirby.utils.noisy_labels.core import group_kfold_splitter, _safe_n_splits
    # float32 features (handoff 2.3 item 18, 2026-10-05); labels stay float64.
    X = as_rep_dtype(X)
    y = np.asarray(y, dtype=float)
    k = _safe_n_splits(groups, n_splits)
    sig = np.full(len(y), np.nan)
    for tr, te in group_kfold_splitter(groups, k, X, y, seed=seed):
        m = make_ngboost('normal').fit(X[tr], y[tr])
        sig[te] = np.asarray(m.pred_dist(X[te]).params['scale'], dtype=float)
    return sig


def invvar_weights(sigma, s):
    """1 / sigma^2, sigma floored at SIGMA_FLOOR_S * s, normalised to average one."""
    sig = np.maximum(np.asarray(sigma, dtype=float), SIGMA_FLOOR_S * max(s, 1e-9))
    w = 1.0 / sig ** 2
    return w / float(np.mean(w))


def c_mixup(X, y, bw, rng, beta=CMIXUP_BETA, chunk=1000):
    """One C-Mixup partner per row; returns (X_aug, y_aug) with 2n rows.

    Partner j of row i is drawn with probability proportional to
    exp(-(y_i - y_j)^2 / (2 bw^2)), j != i. Computed in chunks of rows so the
    n x n kernel is never held at once.
    """
    # float32 features (handoff 2.3 item 18, 2026-10-05); labels stay float64.
    X = as_rep_dtype(X)
    y = np.asarray(y, dtype=float)
    n = len(y)
    bw = max(float(bw), 1e-9)
    partners = np.empty(n, dtype=int)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = y[a:b, None] - y[None, :]
        k = np.exp(-(d * d) / (2.0 * bw * bw))
        k[np.arange(b - a), np.arange(a, b)] = 0.0
        tot = k.sum(axis=1)
        # A row with no partner inside the kernel's reach (all weights underflow)
        # draws uniformly among the others.
        bad = tot <= 0
        k[bad] = 1.0
        k[bad, np.arange(a, b)[bad]] = 0.0
        k /= k.sum(axis=1, keepdims=True)
        cum = np.cumsum(k, axis=1)
        u = rng.random(b - a)[:, None]
        partners[a:b] = np.minimum((cum < u).sum(axis=1), n - 1)
    lam = rng.beta(beta, beta, size=n)
    X_new = as_rep_dtype(lam[:, None] * X + (1.0 - lam[:, None]) * X[partners])
    y_new = lam * y + (1.0 - lam) * y[partners]
    return np.vstack([X, X_new]), np.concatenate([y, y_new])


def _fit_predict(model, X_tr, y_tr, X_va, metric_name, sample_weight=None):
    if sample_weight is None:
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    return np.asarray(_predict_for_metric(model, X_va, metric_name), dtype=float)


def label_arm_predict(arm, factory, X_tr, y0, X_va, metric_name, ctx, rng):
    """Held-out predictions for one label/weight arm with one downstream model."""
    if arm == 'yeo_johnson':
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        yt = pt.fit_transform(np.asarray(y0, dtype=float).reshape(-1, 1)).ravel()
        pred = _fit_predict(factory(), X_tr, yt, X_va, metric_name)
        return pt.inverse_transform(pred.reshape(-1, 1)).ravel()
    if arm == 'invvar_ngboost':
        return _fit_predict(factory(), X_tr, y0, X_va, metric_name,
                            sample_weight=ctx['invvar_w'])
    if arm == 'c_mixup':
        Xa, ya = c_mixup(X_tr, y0, CMIXUP_BW_SD * float(np.std(y0)), rng)
        return _fit_predict(factory(), Xa, ya, X_va, metric_name)
    if arm == 'target_noise_bagging':
        sd = TNB_SCALE * ctx['resid_sd']
        preds = [_fit_predict(factory(), X_tr,
                              y0 + rng.normal(0.0, sd, size=len(y0)),
                              X_va, metric_name)
                 for _ in range(TNB_BAGS)]
        return np.mean(preds, axis=0)
    if arm == 'huber_teacher_blend':
        y_bl = (1.0 - TEACHER_WEIGHT) * y0 + TEACHER_WEIGHT * ctx['teacher']
        return _fit_predict(factory(), X_tr, y_bl, X_va, metric_name)
    raise ValueError(f'n33: unknown arm {arm!r}')


def _levels_and_draws(data, dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs. Level 0 is the labels as delivered, so one draw.

    Levels: noise_helpers.decided_levels (the noise grid decision, 2026-10-05:
    by the dataset's kNN group and, for censoring, its own grid), unless the
    caller names them."""
    levels = (decided_levels(data, dataset_name, condition)
              if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        for d in (draws if float(lvl) > 0 else draws[:1]):
            out.append((float(lvl), int(d)))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N33 across every regression condition, one file per condition."""
    if data['task_type'] != 'regression':
        print(f"  SKIPPED: {dataset_name} is {data['task_type']}. Every arm here "
              f"changes a loss on a number or the number itself, so n33 is "
              f"regression only.")
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    # sweep_conditions pops force_rerun; PassLedger reads restart_passes.
    kwargs['restart_passes'] = bool(kwargs.get('force_rerun'))
    # The noise grid and this version go into the input fingerprint every
    # condition file is stamped with (2026-10-05, chat 5), so a condition file
    # written before the alignment (whole-dataset recipes, five models, the old
    # grid) no longer matches and is recomputed instead of being kept.
    config = dict(config)
    config.setdefault('strategies', list(decided_conditions(data)))
    config['noise_seeds'] = {'draws': list(kwargs.get('draws') or NOISE_DRAWS),
                             'alignment': ALIGNMENT_VERSION}
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=conditions_for_task(data), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _space_context(dataset_name, X_tr_s, y0, g_tr, oof_splits, label_arms,
                   errors, ekey):
    """What the robust losses and label arms read for ONE training matrix of
    one pass: p (out-of-fold LightGBM on the noisy labels), s, the teacher and
    the NGBoost weights. Returns (ctx, sp_rec, s).

    Split out of the fold loop 2026-10-05 (chat 5): each decided model now has
    its own four matrices per fold, and two models whose rank-1 single
    representation is the same share one matrix, so the caller memoises this
    on the matrix's content and computes it once."""
    ctx, sp_rec = {}, {}
    try:
        # Cached on the label hash (2026-09-25): the same p n30, n32, n34,
        # n38, n40 and n41 compute for the draw.
        p = _nc.cached_oof_predict(dataset_name, X_tr_s, y0, g_tr,
                                   n_splits=oof_splits)
    except Exception as exc:
        errors[f'{ekey}|oof'] = f'{type(exc).__name__}: {exc}'
        p = None
    if p is not None:
        s = robust_scale(y0, p)
        ctx['resid_sd'] = float(np.std(y0 - p))
        sp_rec.update({
            'robust_scale_s_label_units': s,
            'residual_sd_label_units': ctx['resid_sd'],
            'residual_kurtosis': residual_kurtosis(y0, p)})
    else:
        s = float('nan')
    have_s = np.isfinite(s) and s > 0

    if have_s and 'huber_teacher_blend' in label_arms:
        try:
            ctx['teacher'] = _nc.cached_oof_predict(
                dataset_name, X_tr_s, y0, g_tr, n_splits=oof_splits,
                params={'objective': 'huber', 'alpha': TEACHER_HUBER_C * s})
        except Exception as exc:
            errors[f'{ekey}|teacher'] = f'{type(exc).__name__}: {exc}'
    if have_s and 'invvar_ngboost' in label_arms:
        try:
            sig = oof_ngboost_sigma(X_tr_s, y0, g_tr, oof_splits)
            ctx['invvar_w'] = invvar_weights(sig, s)
            sp_rec['ngboost_sigma_median_label_units'] = float(np.median(sig))
            sp_rec['invvar_weight_max'] = float(np.max(ctx['invvar_w']))
        except Exception as exc:
            errors[f'{ekey}|ngboost_sigma'] = f'{type(exc).__name__}: {exc}'
    return ctx, sp_rec, s


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N33 under ONE injected condition."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N33: ROBUST LOSSES, TARGET TRANSFORMS, WEIGHTS, MIXUP")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    labels = np.asarray(data['labels'], dtype=float)
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    # A kind outside the noise grid runs only when a caller names it (see the
    # OPEN note at DEFAULT_NOISE_CONDITION); the sweep never picks one.
    condition_note = None
    if condition not in decided_conditions(data):
        condition_note = (
            f'{condition!r} is not one of the six kinds of damage the noise '
            f'grid keeps (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05). It ran '
            f'because a caller named it, at the decided levels.')
        print(f"  NOTE: {condition_note}")

    huber_c = [float(c) for c in (kwargs.get('huber_c') or HUBER_C)]
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    label_arms = [a for a in (kwargs.get('label_arms') or LABEL_ARMS)]
    run_ngboost = not kwargs.get('skip_ngboost', False)
    if not run_ngboost:
        label_arms = [a for a in label_arms if a != 'invvar_ngboost']

    # ── R1: the dataset's two decided models (DECISIONS.md Step 1),
    # 2026-10-05 ──
    # Replaces config['models'] / DOWNSTREAM_MODELS (five models) and the rf
    # and svm n32's downstream_model_list appended. A caller's `models` can
    # only narrow the decided pair (a smoke); a name outside it is dropped and
    # printed.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    if asked:
        dropped = [m for m in asked if m not in decided]
        if dropped:
            print(f"  n33: models {dropped} are not this dataset's decided "
                  f"models {decided}; dropped")
        downstream_models = [m for m in decided if m in asked]
    else:
        downstream_models = list(decided)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    no_factory = [m for m in downstream_models if m not in model_factories]
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n33: no downstream model could be built. Decided models "
            f"{decided}, asked {asked}, built {sorted(model_factories)}.")

    all_swaps = build_swaps(huber_c)
    wanted = kwargs.get('swaps')
    swaps = {k: v for k, v in all_swaps.items()
             if v in downstream_models and (wanted is None or k in wanted)}
    swaps_not_run = {}
    if 'rf_l1' in swaps and not get_model_factories(
            ['rf_l1'], task_type=task_type, n_samples=len(labels)):
        swaps_not_run['rf_l1'] = (
            f'rf_l1 is capped at 5,000 molecules in DEFAULT_MODELS and this '
            f'dataset has {len(labels)}')
        swaps.pop('rf_l1')
    ngb_host = ngboost_host(downstream_models)
    ngb_pair = run_ngboost and ngb_host is not None

    arms_not_run = {}
    for _m in downstream_models:
        _undef = {a: why for a, why in ARMS_UNDEFINED_FOR.get(_m, {}).items()
                  if a in label_arms}
        if _undef:
            arms_not_run[_m] = _undef

    noise_groups = noise_groups_for(data)[0]
    # ── Folds: noise_helpers.noise_outer_folds (handoff 2.4 item 14) ──
    # The count is config['n_splits'] (KIRBY_NOISE_N_FOLDS overrides); the
    # splits are the ones the per-fold searches are filed under. Was
    # cv_split(groups, n_folds, ...), which is not those splits.
    folds, n_folds = noise_outer_folds(data, config)
    knn_grp = dataset_knn_group(data, dataset_name)

    arm_names = ['clean', 'noisy'] + label_arms
    results = {
        'alignment': ALIGNMENT_VERSION,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'condition_note': condition_note,
        'level_unit': level_unit(condition),
        'knn_group': knn_grp,
        'dose_units': noise_dose_units(dataset_name),
        'models': downstream_models,
        'models_decided': decided,
        'models_without_factory': no_factory,
        'n_folds': int(n_folds),
        'folds_run': [int(i) for i, _, _ in folds],
        'selection_scope': 'fold_train_rows',
        'configuration_spaces': list(CONFIGURATION_SPACES),
        'configurations_note': (
            'Per outer fold and decided model: base_1 and base_2 are its 2 '
            'best single representations and hybrid and hybrid__general2 its 2 '
            "best general hybrids, all ranked by searches on that fold's "
            'training rows with clean labels (noise_helpers.'
            'fold_configurations). 2 models x 4 = 8 fitted configurations. '
            'Which ones, per fold, is under folds. n33 finds no neighbours, '
            'so no distance or similarity hybrid is used.'),
        'arms': arm_names,
        'arm_to_beat': 'noisy',
        'label_arms_not_run_by_model': arms_not_run,
        'model_swaps': swaps,
        'model_swaps_not_run': swaps_not_run,
        'ngboost_pair': (['ngboost_normal', 'ngboost_t'] if ngb_pair else []),
        'ngboost_host': ngb_host if ngb_pair else None,
        'ngboost_pair_note': (
            f'ngboost_t (Student-t, df fixed at {NGB_T_DF}) is paired against '
            'ngboost_normal, both at the registry '
            'settings (500 stages, rate 0.05), in the feature matrices built '
            f'for ngboost_host ({ngb_host}): lightgbm when it is a decided '
            'model, otherwise the first decided model. NGBoost is not a '
            'downstream model.'),
        'huber_c': huber_c,
        'scale_note': ('s = 1.4826 * median |y - p|, p an out-of-fold LightGBM '
                       'on the noisy labels, per (fold, model, space). Every '
                       'robust threshold is a multiple of s.'),
        'settings': {
            'tukey_k': TUKEY_K, 'tukey_hess_floor': TUKEY_HESS_FLOOR,
            'cmixup_bw_label_sd': CMIXUP_BW_SD, 'cmixup_beta': CMIXUP_BETA,
            'tnb_bags': TNB_BAGS, 'tnb_scale_of_residual_sd': TNB_SCALE,
            'teacher_weight': TEACHER_WEIGHT,
            'teacher_huber_c': TEACHER_HUBER_C,
            'sigma_floor_in_s': SIGMA_FLOOR_S, 'oof_splits': oof_splits,
        },
        'settings_note': (
            'bw, the bag count, the bag noise and the teacher weight are fixed '
            'defaults, not tuned. The Tukey hessian is floored where it is '
            'negative or zero; see tukey_objective.'),
        'gap_min_frac': GAP_MIN_FRAC,
        'features_are_fixed_note': (
            "Each model's four feature spaces are built once per (level, "
            'draw, fold) from the NOISY training labels and shared by every '
            'arm of that model, clean included. The configurations were '
            'chosen by per-fold searches on the clean training labels.'),
        'robustness_note': (
            'noisy_minus_clean is the plain difference, the noisy arm minus '
            'the clean arm, per outer fold (mean over draws) in the '
            "dataset's own metric; read its sign with lower_is_better. On "
            'fast_knn datasets it is the robustness number (auc_norm is '
            'absent there by decision, 2026-10-05); it is written on every '
            'dataset.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to this '
            'condition on this dataset. Held-out molecules are graded against '
            'untouched labels in every arm.'),
        'similarity_note': ('n33 finds no neighbours, so it has no similarity '
                            'axis; its rows carry model_input|none|process.'),
        'controls_run': controls_enabled(),
        'controls_note': (
            'With KIRBY_NOISE_CONTROLS=1, invvar_ngboost (a weighting) and '
            'huber_teacher_blend (a label move) are twinned by controls.py. '
            'yeo_johnson, c_mixup and target_noise_bagging have no twin: see '
            'the TODO at their rows.'),
    }

    passes = _levels_and_draws(data, dataset_name, condition,
                               levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    results['noise_draws'] = sorted({d for _, d in passes})
    print(f"  {condition}: {len(labels)} molecules, {len(folds)} of {n_folds} "
          f"folds, models {downstream_models} ({knn_grp})")
    print(f"  {len(passes)} (level, draw) passes, {len(arm_names)} label arms, "
          f"{len(swaps)} loss swaps{', plus the NGBoost pair' if ngb_pair else ''}")

    # acc[level][space][arm][model][pass] = score; tails the same shape.
    acc, tails, pass_recs, errors = {}, {}, {}, {}
    # {fold: {model: configuration_record}}: what each fold fitted (handoff
    # 2.4 item 11).
    fold_records = {}
    t0 = time.time()
    # Rows and the per-pass checkpoint, 2026-09-25 (n32's PassLedger). The
    # signature carries the header above (ALIGNMENT_VERSION included) and the
    # fold indices, so a checkpoint from before 2026-10-05 is set aside.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, condition,
                        signature=ledger_signature(
                            data, config, results,
                            [(tr, va) for _i, tr, va in folds]),
                        restart=restart_flag(kwargs))
    if ledger.restored():
        st = ledger.restored()
        acc, tails, pass_recs, errors = (st['acc'], st['tails'],
                                         st['pass_recs'], st['errors'])
        fold_records = st.get('fold_records') or {}
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    def _store(lvl_key, space, arm, mname, pass_key, pred, y_val):
        s = float(metric_fn(y_val, pred))
        (acc.setdefault(lvl_key, {}).setdefault(space, {}).setdefault(arm, {})
         .setdefault(mname, {}))[pass_key] = s
        (tails.setdefault(lvl_key, {}).setdefault(space, {}).setdefault(arm, {})
         .setdefault(mname, {}))[pass_key] = tail_scores(y_val, pred)

    # Per-fold configurations, once per fold for every (level, draw): the
    # searches run on the fold's training rows with the clean labels and are
    # stored on disk by fold_searches, so a resumed run reads them back.
    fold_entries = {}

    def _entries_for(fold_i, train_idx):
        if fold_i not in fold_entries:
            try:
                fold_entries[fold_i] = fold_configurations(
                    data, dataset_name, fold_i=fold_i, train_idx=train_idx,
                    n_folds=n_folds)
            except Exception as exc:
                errors[f'f{fold_i}|fold_configurations'] = (
                    f'{type(exc).__name__}: {exc}')
                print(f"  fold {fold_i}: fold_configurations raised "
                      f"{type(exc).__name__}: {exc}")
                fold_entries[fold_i] = None
        return fold_entries[fold_i]

    needs = {'huber_teacher_blend': 'teacher',
             'invvar_ngboost': 'invvar_w',
             'target_noise_bagging': 'resid_sd'}

    for level, draw in passes:
        lvl_key = f'{level:g}'
        print(f"\n  --- {condition} / level {lvl_key} / draw {draw} ---")
        for fold_i, train_idx, val_idx in folds:
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if ledger.pass_done(pass_key):
                continue
            entries = _entries_for(fold_i, train_idx)
            if entries is None:
                continue
            pass_rows = []
            y_val = labels[val_idx]
            y_clean_tr = labels[train_idx]
            g_tr = groups[train_idx] if groups is not None else None
            try:
                # The decided kinds through one call (noise_helpers.
                # inject_condition, 2026-10-05); level 0 is the labels as
                # delivered.
                y0, cmask, prov = inject_condition(
                    data, dataset_name, condition, level, draw,
                    train_idx=train_idx, groups=noise_groups)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            rec = {'pass': pass_key, 'training_molecules': int(len(y0)),
                   'labels_moved': int(np.sum(np.abs(y0 - y_clean_tr) > 1e-12)),
                   'injected_rms_label_units': float(
                       np.sqrt(np.mean((y0 - y_clean_tr) ** 2))),
                   'per_space': {}, 'unavailable': {}}
            fold_rec = fold_records.setdefault(str(fold_i), {})
            # One context per distinct training matrix in this pass.
            ctx_memo = {}

            for mname in downstream_models:
                entry = entries.get(mname)
                fac = model_factories[mname]
                cell_models = [mname]
                # The model's 4 configurations on this fold, hybrids rebuilt
                # from this pass's noisy labels with the fold search's own
                # representation scores (dirty-run bug 1).
                try:
                    inputs, unavail, names = fold_model_inputs(
                        data, dataset_name, mname, entry, train_idx=train_idx,
                        val_idx=val_idx, y_train=y0, factory=fac,
                        metric=metric)
                except Exception as exc:
                    errors[f'{lvl_key}|{mname}|d{draw}|f{fold_i}|build'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                if mname not in fold_rec:
                    fold_rec[mname] = convert_numpy(
                        configuration_record(entry, names))
                if unavail:
                    rec['unavailable'][mname] = dict(unavail)

                for space in CONFIGURATION_SPACES:
                    if space not in inputs:
                        continue
                    X_tr_s, X_va_s = inputs[space]
                    ekey = f'{lvl_key}|{space}/{mname}|d{draw}|f{fold_i}'
                    xkey = _nc.array_hash(X_tr_s)
                    if xkey not in ctx_memo:
                        ctx_memo[xkey] = _space_context(
                            dataset_name, X_tr_s, y0, g_tr, oof_splits,
                            label_arms, errors, ekey)
                    ctx, sp_rec, s = ctx_memo[xkey]
                    have_s = np.isfinite(s) and s > 0
                    if have_s:
                        sp_rec = dict(sp_rec, huber_thresholds_label_units={
                            f'c{c:g}': c * s for c in huber_c})
                    rec['per_space'].setdefault(space, {})[mname] = sp_rec

                    pass_scores = {}
                    for arm in arm_names:
                        if arm in needs and needs[arm] not in ctx:
                            continue
                        if arm in arms_not_run.get(mname, {}):
                            continue
                        rng = np.random.default_rng(
                            [fold_i, draw, int(round(level * 1000)),
                             arm_names.index(arm)])
                        try:
                            # clean and noisy through noise_cache for a
                            # seeded model (2026-09-25): the clean fit in a
                            # base space is the same every (level, draw).
                            if arm == 'clean':
                                pred = fit_predict_cached(
                                    dataset_name, mname, fac, X_tr_s,
                                    y_clean_tr, X_va_s, metric_name)
                            elif arm == 'noisy':
                                pred = fit_predict_cached(
                                    dataset_name, mname, fac, X_tr_s, y0,
                                    X_va_s, metric_name)
                            else:
                                pred = label_arm_predict(
                                    arm, fac, X_tr_s, y0, X_va_s,
                                    metric_name, ctx, rng)
                            pred = np.asarray(pred, dtype=float)
                            _store(lvl_key, space, arm, mname, pass_key,
                                   pred, y_val)
                            pass_scores.setdefault(arm, {})[mname] = \
                                acc[lvl_key][space][arm][mname][pass_key]
                        except Exception as exc:
                            errors[f'{ekey}|{arm}|{mname}'] = (
                                f'{type(exc).__name__}: {exc}')

                    # ── Loss swaps, on the noisy labels, paired with the
                    # parent ──
                    for swap, parent in swaps.items():
                        if parent != mname:
                            continue
                        if not have_s and swap != 'rf_l1':
                            continue
                        try:
                            pred = _fit_predict(make_swap(swap, s, huber_c),
                                                X_tr_s, y0, X_va_s, metric_name)
                            _store(lvl_key, space, 'noisy', swap, pass_key,
                                   pred, y_val)
                        except Exception as exc:
                            errors[f'{ekey}|{swap}'] = (
                                f'{type(exc).__name__}: {exc}')
                    if ngb_pair and mname == ngb_host:
                        for dist in ('normal', 't'):
                            try:
                                m = make_ngboost(dist).fit(X_tr_s, y0)
                                pred = np.asarray(m.predict(X_va_s), dtype=float)
                                _store(lvl_key, space, 'noisy', f'ngboost_{dist}',
                                       pass_key, pred, y_val)
                            except Exception as exc:
                                errors[f'{ekey}|ngboost_{dist}'] = (
                                    f'{type(exc).__name__}: {exc}')

                    # ── Controls (KIRBY_NOISE_CONTROLS=1), 2026-09-25 ──
                    # invvar_ngboost is a weighting and huber_teacher_blend a
                    # label move, so controls.py twins them.
                    # TODO(similarity-sweep 2026-09-25): yeo_johnson,
                    # c_mixup and target_noise_bagging get no twin. Each
                    # changes EVERY label or adds rows by construction, so
                    # "the same number of molecules at random" is the method
                    # itself; what a control for them should hold fixed is a
                    # design decision for the owner.
                    twin_src = {}
                    if ('invvar_w' in ctx
                            and 'invvar_ngboost' not in arms_not_run.get(
                                mname, {})):
                        twin_src['invvar_ngboost'] = arm_result(
                            y0, weight=ctx['invvar_w'])
                    if 'teacher' in ctx:
                        twin_src['huber_teacher_blend'] = arm_result(
                            y0, y=(1.0 - TEACHER_WEIGHT) * y0
                            + TEACHER_WEIGHT * ctx['teacher'])
                    for arm, res_ in twin_src.items():
                        if arm not in label_arms:
                            continue
                        twins = control_twins(res_, X_tr_s, y0, (
                            replicate_seed(arm), draw, fold_i,
                            int(round(level * 1000))))
                        for tw, by_m in twin_scores(
                                dataset_name, twins, model_factories,
                                cell_models, X_va_s, y_val, metric).items():
                            pass_scores[f'{arm}|{tw}'] = by_m

                    # ── D against D': noisy is D, every label arm a D'.
                    # The loss swaps change the model, not the data, and are
                    # read against their parent in the summary, not here.
                    # model_input is the configuration name alone (base_1,
                    # hybrid, ...) and the model is its own field, 2026-10-05:
                    # n32-n35 used to write 'base_1/rf' even on svm rows and
                    # n30/n31 'base_1', so a join across them missed rows
                    # (dirty-run bug 13, two row formats). ──
                    pass_rows += arm_rows(
                        dataset_name=dataset_name, exp_id=EXP_ID,
                        metric=metric, condition=condition, level=level,
                        draw=draw, fold=fold_i, model_input=space,
                        scores_by_arm=pass_scores, reference='noisy')

            pass_recs.setdefault(lvl_key, []).append(rec)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(pass_key, {'acc': acc, 'tails': tails,
                                          'pass_recs': pass_recs,
                                          'errors': errors,
                                          'fold_records': fold_records})
            print(f"    level {lvl_key} draw {draw} fold {fold_i}: "
                  f"{rec['labels_moved']} of {rec['training_molecules']} labels "
                  f"moved, {len(pass_rows)} D-vs-D' rows "
                  f"[{time.time() - t0:.0f}s]")

    results['folds'] = convert_numpy(fold_records)

    swap_parents = dict(swaps)
    if ngb_pair:
        swap_parents['ngboost_t'] = 'ngboost_normal'
    results['per_level'] = convert_numpy(acc)
    results['tails_per_level'] = convert_numpy(tails)
    results['per_pass'] = convert_numpy(pass_recs)
    results['d_vs_dprime'] = ledger.describe()
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, arm_names, downstream_models, swap_parents, pass_recs,
        metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _share(clean_d, noisy_d, arm_d, lower_is_better):
    keys = sorted(set(clean_d) & set(noisy_d) & set(arm_d))
    if not keys:
        return {'recovery_share': None,
                'recovery_share_refused_because': 'no pass scored all three',
                'n_passes': 0}
    c, k, a = (_mean_on(clean_d, keys), _mean_on(noisy_d, keys),
               _mean_on(arm_d, keys))
    share, why = recovery_share(c, k, a, lower_is_better)
    return {'recovery_share': share, 'recovery_share_refused_because': why,
            'clean_mean': c, 'noisy_mean': k, 'arm_mean': a,
            'n_passes': len(keys)}


def _outer_fold_of(pass_key):
    """The outer fold number of a pass key 'l<level>d<draw>f<fold>'."""
    return int(str(pass_key).rsplit('f', 1)[1])


def _noisy_minus_clean(clean_d, noisy_d):
    """The plain difference, noisy arm minus clean arm, per outer fold (mean
    over that fold's draws), in the dataset's own metric; read the sign with
    lower_is_better. noise_helpers.fold_spread over the folds, so one fold
    gives 'one fold, no spread'. The robustness number the noise-grid
    decision asks for on fast_knn datasets (2026-10-05)."""
    by = {}
    for k in sorted(set(clean_d) & set(noisy_d)):
        d = noisy_d[k] - clean_d[k]
        if np.isfinite(d):
            by.setdefault(_outer_fold_of(k), []).append(float(d))
    return fold_spread([float(np.mean(v)) for _f, v in sorted(by.items())])


def _spread_text(fs):
    """'sd 0.0123 over 5 folds' or the fold_spread note."""
    if not fs or fs.get('sd') is None:
        return (fs or {}).get('note') or 'no spread'
    return f"sd {fs['sd']:.4f} over {fs['n_folds']} folds"


def _summarise(acc, arm_names, models, swap_parents, pass_recs, metric_name,
               lower_is_better):
    """Every arm against noisy, every swap against its parent, per level and space."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against the noisy labels, paired per pass; '
                        'a pass is one (draw, fold)'),
           'per_level': {}}
    for lvl, by_space in acc.items():
        kurt = {}
        for rec in pass_recs.get(lvl, ()):
            # per_space is {space: {model: record}} since 2026-10-05 (each
            # decided model has its own four matrices per fold).
            for sp, by_m in rec['per_space'].items():
                for m, r in by_m.items():
                    if np.isfinite(r.get('residual_kurtosis', np.nan)):
                        kurt.setdefault(f'{sp}/{m}', []).append(
                            r['residual_kurtosis'])
        lrow = {'residual_kurtosis_per_pass': kurt}
        for space, by_arm in by_space.items():
            noisy = by_arm.get('noisy') or {}
            clean = by_arm.get('clean') or {}
            srow = {'per_model': {}, 'model_swaps': {}}
            for m in models:
                b = noisy.get(m) or {}
                if not b:
                    continue
                row = {'noisy_mean': _mean_on(b, list(b)),
                       'noisy_minus_clean': _noisy_minus_clean(
                           clean.get(m) or {}, b),
                       'per_arm': {}}
                for arm in arm_names:
                    if arm == 'noisy':
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    e = {'vs_noisy': _paired(b, g, lower_is_better)}
                    if arm != 'clean':
                        e.update(_share(clean.get(m) or {}, b, g,
                                        lower_is_better))
                    row['per_arm'][arm] = e
                srow['per_model'][m] = row
            for swap, parent in swap_parents.items():
                g = noisy.get(swap) or {}
                b = noisy.get(parent) or {}
                if not g or not b:
                    continue
                e = {'replaces': parent,
                     'vs_parent_on_noisy': _paired(b, g, lower_is_better)}
                if parent in models:
                    e.update(_share(clean.get(parent) or {}, b, g,
                                    lower_is_better))
                srow['model_swaps'][swap] = e
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; {results['noise_condition']}; paired per pass "
          f"against the noisy labels)")
    for lvl, lrow in s['per_level'].items():
        print(f"\n  level {lvl}:")
        for sp, ks in lrow['residual_kurtosis_per_pass'].items():
            print(f"    residual kurtosis in {sp}: "
                  f"{', '.join(f'{k:.2f}' for k in ks)} (3.0 is normal)")
        for space, srow in lrow.items():
            if space == 'residual_kurtosis_per_pass':
                continue
            for m, row in srow['per_model'].items():
                nmc = row.get('noisy_minus_clean') or {}
                nmc_txt = ('' if nmc.get('mean') is None else
                           f", noisy minus clean {nmc['mean']:+.4f} "
                           f"({_spread_text(nmc)})")
                print(f"    {space}/{m}: noisy {row['noisy_mean']:.4f}{nmc_txt}")
                for arm, e in row['per_arm'].items():
                    v = e['vs_noisy']
                    sh = e.get('recovery_share')
                    tail = (f", recovers {sh:.2f} of the clean-to-noisy gap"
                            if sh is not None else '')
                    # n30's _paired gives the spread over outer folds
                    # (gain_over_folds) since 2026-10-05; it used to be
                    # spread_over_pass_gains, which no longer exists.
                    print(f"      {arm:24s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, mean "
                          f"{v['mean_gain']:+.4f} "
                          f"({_spread_text(v.get('gain_over_folds'))}){tail}")
            for swap, e in srow['model_swaps'].items():
                v = e['vs_parent_on_noisy']
                print(f"    {space}/{swap} vs {e['replaces']}: won "
                      f"{v['passes_won']} of {v['n_passes_paired']} passes, "
                      f"mean {v['mean_gain']:+.4f}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than its spread over the outer folds is not a gain.")


def main():
    parser = create_experiment_parser(
        'N33: robust losses, target transforms, weights and mixup under label noise')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help='One regression condition. Default: sweep the six '
                             'decided kinds, one file each')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated noise levels. Default: the '
                             "decided levels for this dataset's kNN group")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates (default 0,1,2)')
    parser.add_argument('--huber-c', type=str, default=None,
                        help='Comma-separated Huber thresholds in units of '
                             f's (default {",".join(f"{c:g}" for c in HUBER_C)})')
    parser.add_argument('--skip-ngboost', action='store_true',
                        help='Leave out invvar_ngboost and the NGBoost pair')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds. Overrides the count --mode gives '
                             '(KIRBY_NOISE_N_FOLDS overrides both)')
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
    if args.huber_c:
        kw['huber_c'] = [float(x) for x in args.huber_c.split(',')]
    if args.skip_ngboost:
        kw['skip_ngboost'] = True

    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, EXP_ID, args.dataset)


if __name__ == '__main__':
    main()
