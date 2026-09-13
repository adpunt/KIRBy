"""f30_scorers — every feature importance scoring method f30 tests, one function each.

Each scorer has the SAME contract as the engine's own `compute_feature_importance`
output, which is the only thing the hybrid consumes:

    scorer(X, y, ctx) -> np.ndarray, shape (X.shape[1],), higher = keep

`X` is one representation's train-fold matrix, `y` the train-fold labels.  Only the
ORDER WITHIN the representation is ever read (`hybrid.py:1981`
`np.argsort(importances)[-n_select:][::-1]`), and `concatenate_features`
(`hybrid.py:4054`) raises if the length is wrong.  Magnitude, sign and scale are
never used downstream.

Methods that do not natively emit a per-feature number are converted by one of the
three rules recorded in the plan:
  S1 continuous score      -> used directly
  S2 greedy / nested order -> score = (K - step) for the feature picked at `step`,
                              0 for never-picked (the convention already used at
                              tests/experiments/feature_selectors/selector_prediction.py:177)
  S3 subset / mask         -> the CONTINUOUS statistic the method thresholds, never
                              the boolean (Boruta hit count, stability-selection
                              frequency, lasso |coef|)

Reference implementations followed are named in the docstring of each scorer.  No
new pip dependency is introduced: methods whose upstream package is absent are
implemented in numpy from the cited reference file.

Import-only from src/kirby — this module never edits hybrid.py / find_best_hybrid.py.
"""
# lightgbm FIRST — macOS OpenMP segfault guard (src/kirby/hybrid.py:41-66).
import lightgbm as _lgb  # noqa: F401

import os
import sys
import time
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = str(_HERE.parents[3])
for _p in (os.path.join(_REPO, "src"),
           os.path.join(_REPO, "tests"),
           os.path.join(_REPO, "tests", "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from kirby.hybrid import compute_feature_importance  # noqa: E402

# Greedy / kernel methods are run behind a relevance pre-screen; this is the
# "hybrid filter-wrapper" structure from the survey (§7) and the size already used
# at selector_prediction.py:177.  Recorded in every result file.
N_CAND = 400
# Relief samples m instances (Kira & Rendell 1992); O(n^2) otherwise.
RELIEF_M = 600
# Equal-frequency bins for the discretisation-based information filters.
N_BINS = 8
# Bootstraps for stability selection (Meinshausen & Buhlmann 2010 use ~100; 20 is
# what the engine's own allocate_stability_selection uses at hybrid.py:1835).
N_BOOT = 20


def _clean(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.clip(X, -1e10, 1e10)
    return np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)


def _zs(X):
    """Standardise; constant columns become all-zero rather than NaN."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def _is_clf(ctx):
    return ctx.get("task_type") == "classification"


def _order_to_score(order, d):
    """S2: nested greedy order -> per-feature score. First picked scores highest."""
    s = np.zeros(d, dtype=float)
    k = len(order)
    for step, j in enumerate(order):
        s[int(j)] = float(k - step)
    return s


# ===========================================================================
# INCUMBENTS — thin wrappers over the engine, so hybrid.py stays the source of
# truth for the methods already in production.
# ===========================================================================
def _engine(method, X, y, **kw):
    return np.asarray(
        compute_feature_importance({"_": _clean(X)}, np.asarray(y), method=method,
                                   **kw)["_"], dtype=float)


def s_lightgbm_gain(X, y, ctx):
    """hybrid.py:3749. WITH the 2000-feature variance pre-screen, as production runs it."""
    return _engine("lightgbm_gain", X, y)


def s_lightgbm_gain_nocap(X, y, ctx):
    """hybrid.py:3749 with the pre-screen lifted.

    The screen ranks by variance AFTER StandardScaler, which sets every
    non-constant column's variance to 1 (measured spread 5e-06 on caco2/grover),
    so it keeps 2000 features on floating-point rounding error and writes exactly
    0.0 for the rest.  `max_features_lgb` is an existing kwarg (hybrid.py:3758) —
    raising it is a call-site change, not a code change.
    """
    return _engine("lightgbm_gain", X, y, max_features_lgb=int(X.shape[1]))


def s_lightgbm_gain_deep(X, y, ctx):
    """hybrid.py:3749 with more trees AND the pre-screen lifted.

    Answers "can lightgbm_gain be made to work on a wide representation?" by
    measurement instead of argument.  Gain is exactly 0 for any feature no tree
    split on, so the number of features that CAN be scored is bounded by
    n_estimators x (num_leaves - 1).  Production uses n_estimators=50, max_depth=6
    (hybrid.py:3755-3756) with LightGBM's default num_leaves=31 -> a ceiling of
    1500, and MEASURED only 132-201 of 2048 features actually scored.

    `n_estimators` is an existing documented kwarg (hybrid.py:3319), so this is a
    call-site change, not a code change.
    """
    return _engine("lightgbm_gain", X, y, n_estimators=500,
                   max_features_lgb=int(X.shape[1]))


def s_xgboost_gain(X, y, ctx):
    """hybrid.py:3684. booster.get_score(importance_type='gain'), zero-filled there."""
    return _engine("xgboost_gain", X, y)


def s_treeshap(X, y, ctx):
    """hybrid.py:3386. shap.TreeExplainer, mean |SHAP|. Observational by default."""
    return _engine("treeshap", X, y)


def s_kernelshap(X, y, ctx):
    """hybrid.py:3597. shap.KernelExplainer on an RF surrogate."""
    return _engine("kernelshap", X, y)


def s_gradientshap(X, y, ctx):
    """hybrid.py:3904. captum GradientShap on an internally trained MLP."""
    return _engine("gradientshap", X, y)


def s_deeplift(X, y, ctx):
    """hybrid.py:3807. captum DeepLift. The one incumbent with NO 2000-feature cap."""
    return _engine("deeplift", X, y)


def s_integrated_gradients(X, y, ctx):
    """hybrid.py:3471. captum IntegratedGradients, zero baseline."""
    return _engine("integrated_gradients", X, y)


def s_permutation(X, y, ctx):
    """hybrid.py:3350. sklearn permutation_importance; CAN BE NEGATIVE, which is
    fine — only the order is read."""
    return _engine("permutation", X, y)


# ===========================================================================
# TREE / BOOSTING — the two gradient-boosting libraries the engine does not
# already expose, plus RandomForest MDI which has no branch at all.
# ===========================================================================
def s_catboost_importance(X, y, ctx):
    """CatBoost get_feature_importance(type='PredictionValuesChange') — the default
    non-ranking importance. https://catboost.ai/docs/en/concepts/fstr
    """
    from catboost import CatBoostClassifier, CatBoostRegressor
    Xz = _zs(_clean(X))
    cls = CatBoostClassifier if _is_clf(ctx) else CatBoostRegressor
    m = cls(iterations=200, depth=6, random_seed=ctx["seed"], verbose=0,
            thread_count=ctx.get("n_jobs", 1), allow_writing_files=False)
    m.fit(Xz, np.asarray(y))
    return np.asarray(m.get_feature_importance(type="PredictionValuesChange"),
                      dtype=float)


def s_rf_mdi(X, y, ctx):
    """sklearn RandomForest.feature_importances_ (Gini / MDI).

    NOT a branch in compute_feature_importance today — a call with
    method='random_forest' raises at hybrid.py:4008 despite the docstring at :3345.
    Biased toward high-cardinality and continuous features (Strobl et al. 2007,
    BMC Bioinformatics 8:25), which is testable here because caco2 and
    buchwald_hartwig mix binary fingerprints with continuous embeddings.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    Xz = _zs(_clean(X))
    cls = RandomForestClassifier if _is_clf(ctx) else RandomForestRegressor
    m = cls(n_estimators=100, max_depth=10, random_state=ctx["seed"],
            n_jobs=ctx.get("n_jobs", 1))
    m.fit(Xz, np.asarray(y))
    return np.asarray(m.feature_importances_, dtype=float)


def s_treeshap_interventional(X, y, ctx):
    """shap.TreeExplainer(feature_perturbation='interventional').

    The engine's treeshap branch (hybrid.py:3434) leaves feature_perturbation at
    its default.  Interventional is 'true to the model' — it breaks feature
    dependencies with the do-operator and gives an unused feature exactly zero;
    observational ('tree_path_dependent') respects correlations but can credit a
    feature the model never referenced (Chen, Janizek, Lundberg & Lee 2020,
    arXiv:2006.16234; Sundararajan & Najmi, ICML 2020).  Interventional REQUIRES a
    background sample.
    https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
    """
    import shap
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    Xz = _zs(_clean(X))
    y = np.asarray(y)
    cls = RandomForestClassifier if _is_clf(ctx) else RandomForestRegressor
    m = cls(n_estimators=50, max_depth=10, random_state=ctx["seed"],
            n_jobs=ctx.get("n_jobs", 1)).fit(Xz, y)
    rng = np.random.default_rng(ctx["seed"])
    bg = Xz[rng.choice(len(Xz), size=min(100, len(Xz)), replace=False)]
    ev = Xz[rng.choice(len(Xz), size=min(200, len(Xz)), replace=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expl = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional")
        sv = expl.shap_values(ev, check_additivity=False)
    sv = np.asarray(sv)
    # multiclass -> (n, d, k) or list of (n, d): average |value| over classes too.
    if sv.ndim == 3:
        return np.abs(sv).mean(axis=(0, 2))
    return np.abs(sv).mean(axis=0)


# ===========================================================================
# UNIVARIATE FILTERS
# ===========================================================================
def s_f_test(X, y, ctx):
    """sklearn f_classif / f_regression.
    sklearn/feature_selection/_univariate_selection.py
    For regression F = r^2 (n-2)/(1-r^2), strictly increasing in |r|, so this arm
    also stands in for Pearson and point-biserial (asserted rank-identical in the
    experiment's own check).
    """
    from sklearn.feature_selection import f_classif, f_regression
    Xz = _zs(_clean(X))
    y = np.asarray(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, _ = (f_classif(Xz, y) if _is_clf(ctx) else f_regression(Xz, y))
    return np.nan_to_num(np.asarray(F, dtype=float), nan=0.0)


def s_spearman_marginal(X, y, ctx):
    """|Spearman rho| per feature. Monotone rather than linear, so NOT rank-equal
    to the F-test arm.  Computed as Pearson on the rank-transformed columns, which
    is the definition."""
    from scipy.stats import rankdata
    Xr = np.apply_along_axis(rankdata, 0, _clean(X))
    yr = rankdata(np.asarray(y, dtype=float))
    Xr = _zs(Xr)
    yr = (yr - yr.mean()) / (yr.std() if yr.std() else 1.0)
    return np.abs(Xr.T @ yr) / len(yr)


def s_mutual_info(X, y, ctx):
    """sklearn mutual_info_classif / mutual_info_regression — Kraskov k-NN estimator.
    sklearn/feature_selection/_mutual_info.py
    This is also 'information gain': IG of a feature about the target IS I(X;Y).
    `discrete_features` is set from the measured feature type so fingerprints use
    the exact discrete estimator rather than the k-NN one.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    Xc = _clean(X)
    y = np.asarray(y)
    disc = bool(ctx.get("is_binary_rep", False))
    fn = mutual_info_classif if _is_clf(ctx) else mutual_info_regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = fn(Xc, y, discrete_features=disc, random_state=ctx["seed"])
    return np.asarray(mi, dtype=float)


def s_variance(X, y, ctx):
    """CONTROL. sklearn VarianceThreshold.variances_ — label-free.

    Any scorer that cannot beat 'rank features by their own variance, ignoring the
    labels' is not selecting.  Also quantifies the pre-screen defect directly:
    computed on the RAW matrix here, not the standardised one, which is what the
    engine's screen gets wrong (hybrid.py:3774).
    """
    return np.var(_clean(X), axis=0)


def s_random(X, y, ctx):
    """CONTROL. Uniform random scores, fixed seed. The floor."""
    rng = np.random.default_rng(ctx["seed"] + 977)
    return rng.random(X.shape[1])


# ===========================================================================
# INFORMATION-THEORETIC GREEDY FILTERS — redundancy-aware.
# Implemented in numpy from the reference criteria rather than depending on
# skfeature (the upstream repo is unmaintained and is not on PyPI under its own
# name).  Criteria follow Brown, Pocock, Zhao & Lujan 2012, JMLR 13:27-66, and
# the scikit-feature files JMI.py / CMIM.py / LCSI.py.
# ===========================================================================
def _discretise(X, n_bins=N_BINS):
    """Equal-frequency binning. Columns with <= n_bins distinct values are left
    as integer codes so binary fingerprints are exact, not re-binned."""
    X = _clean(X)
    n, d = X.shape
    out = np.zeros((n, d), dtype=np.int32)
    for j in range(d):
        col = X[:, j]
        u = np.unique(col)
        if len(u) <= n_bins:
            out[:, j] = np.searchsorted(u, col)
        else:
            qs = np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1])
            out[:, j] = np.searchsorted(np.unique(qs), col)
    return out


def _discretise_y(y, ctx, n_bins=N_BINS):
    y = np.asarray(y)
    if _is_clf(ctx):
        return np.unique(y, return_inverse=True)[1].astype(np.int32)
    yf = np.asarray(y, dtype=float)
    qs = np.unique(np.quantile(yf, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(qs, yf).astype(np.int32)


def _entropy_from_counts(counts, n):
    """Rows of `counts` are independent histograms; returns one entropy each."""
    p = counts / float(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(p > 0, np.log(p), 0.0)
    return -(p * lg).sum(axis=1)


def _mi_1d(xd, yd, n):
    """I(x;y) for a single discrete column pair."""
    nx, ny = xd.max() + 1, yd.max() + 1
    joint = np.bincount(xd * ny + yd, minlength=nx * ny).astype(float)
    hx = _entropy_from_counts(np.bincount(xd, minlength=nx)[None, :].astype(float), n)[0]
    hy = _entropy_from_counts(np.bincount(yd, minlength=ny)[None, :].astype(float), n)[0]
    hxy = _entropy_from_counts(joint[None, :], n)[0]
    return hx + hy - hxy


def _relevance_mi(Xd, yd):
    n = len(yd)
    return np.array([_mi_1d(Xd[:, j], yd, n) for j in range(Xd.shape[1])])


def _triple_counts(Xd_cands, xs, yd, nb):
    """3-way histograms of (candidate, selected, y) for every candidate at once.

    Returns counts of shape (n_cand, nb*nb*ny) via one bincount, which keeps the
    greedy inner loop out of Python.
    """
    n, nc = Xd_cands.shape
    ny = yd.max() + 1
    base = (xs.astype(np.int64) * ny + yd.astype(np.int64))[:, None]   # (n,1)
    code = Xd_cands.astype(np.int64) * (nb * ny) + base                # (n,nc)
    flat = code + (np.arange(nc, dtype=np.int64) * (nb * nb * ny))[None, :]
    counts = np.bincount(flat.ravel(), minlength=nc * nb * nb * ny)
    return counts.reshape(nc, nb * nb * ny).astype(float), ny


def _greedy_info(X, y, ctx, criterion):
    """Shared greedy skeleton for JMI and CMIM.

    Both are run behind an MI relevance pre-screen to N_CAND features — the
    hybrid filter-wrapper structure of the survey §7, and the same shape as
    selector_prediction.py:177.  Score by S2.
    """
    Xc = _clean(X)
    d = Xc.shape[1]
    yd = _discretise_y(y, ctx)
    n = len(yd)
    budget = int(min(ctx.get("budget", N_CAND), d))

    Xd_all = _discretise(Xc)
    rel = _relevance_mi(Xd_all, yd)
    cand = np.argsort(rel)[::-1][:min(N_CAND, d)].copy()
    Xd = Xd_all[:, cand]
    nb = int(Xd.max()) + 1
    rel_c = rel[cand]

    ny = int(yd.max()) + 1
    hy = _entropy_from_counts(np.bincount(yd, minlength=ny)[None, :].astype(float), n)[0]

    selected, remaining = [], list(range(len(cand)))
    # first pick = max relevance, identical for both criteria
    first = int(np.argmax(rel_c))
    selected.append(first)
    remaining.remove(first)
    running = np.full(len(cand), np.inf if criterion == "cmim" else 0.0)

    while len(selected) < budget and remaining:
        s = selected[-1]
        xs = Xd[:, s]
        rem = np.asarray(remaining)
        counts, ny_ = _triple_counts(Xd[:, rem], xs, yd, nb)

        c3 = counts.reshape(len(rem), nb, nb * ny_)
        h_ks_y = _entropy_from_counts(counts, n)                    # H(xk,xs,y)
        h_ks = _entropy_from_counts(c3.sum(axis=2), n)              # H(xk,xs)
        c_sy = c3.sum(axis=1)                                       # (nrem, nb*ny)
        h_sy = _entropy_from_counts(c_sy, n)                        # H(xs,y)
        h_s = _entropy_from_counts(c_sy.reshape(len(rem), nb, ny_).sum(axis=2), n)

        if criterion == "jmi":
            # J = sum_{s in S} I(xk,xs ; y) = sum [H(xk,xs) + H(y) - H(xk,xs,y)]
            running[rem] += h_ks + hy - h_ks_y
            scores = running[rem]
        else:  # cmim: J = min_{s in S} I(xk ; y | xs)
            cmi = h_ks + h_sy - h_s - h_ks_y
            running[rem] = np.minimum(running[rem], cmi)
            scores = running[rem]

        nxt = int(rem[int(np.argmax(scores))])
        selected.append(nxt)
        remaining.remove(nxt)

    return _order_to_score(cand[np.asarray(selected)], d)


def s_jmi(X, y, ctx):
    """JMI (Yang & Moody 1999): J(x_k) = sum_{s in S} I(x_k, x_s ; y).
    scikit-feature information_theoretical_based/JMI.py (LCSI beta=gamma=1/|S|).
    Brown et al. 2012 report JMI as the best accuracy/stability tradeoff on small
    samples, which is a direct claim about tcga_brca_pam50 (n=821).
    Continuous features are equal-frequency binned; that approximation is recorded.
    """
    return _greedy_info(X, y, ctx, "jmi")


def s_cmim(X, y, ctx):
    """CMIM (Fleuret 2004): J(x_k) = min_{s in S} I(x_k ; y | x_s).
    scikit-feature information_theoretical_based/CMIM.py.
    """
    return _greedy_info(X, y, ctx, "cmim")


def s_mrmr(X, y, ctx):
    """mRMR, FCQ variant: argmax_j relevance(j) / mean_{s in S} |corr(j, s)|.

    Follows smazzanti/mrmr `mrmr/pandas.py` (relevance = F-statistic, redundancy =
    mean absolute Pearson correlation to the already-selected set), which is the
    fast variant; Peng, Long & Ding 2005, IEEE TPAMI 27(8):1226-1238 is the
    original MID/MIQ form.  The repo already has a MID version at
    selector_prediction.py:177.  Score by S2.
    """
    Xc = _zs(_clean(X))
    d = Xc.shape[1]
    budget = int(min(ctx.get("budget", N_CAND), d))
    rel = s_f_test(X, y, ctx)
    cand = np.argsort(rel)[::-1][:min(N_CAND, d)].copy()
    Z = Xc[:, cand]
    n = len(Z)
    rel_c = np.maximum(rel[cand], 0.0)

    selected = [int(np.argmax(rel_c))]
    red_sum = np.zeros(len(cand))
    mask = np.ones(len(cand), dtype=bool)
    mask[selected[0]] = False

    while len(selected) < budget and mask.any():
        last = Z[:, selected[-1]]
        c = np.abs((Z.T @ last) / n)              # columns are standardised
        red_sum += np.nan_to_num(c, nan=0.0)
        denom = np.maximum(red_sum / len(selected), 1e-8)
        score = np.where(mask, rel_c / denom, -np.inf)
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break
        selected.append(nxt)
        mask[nxt] = False

    return _order_to_score(cand[np.asarray(selected)], d)


# ===========================================================================
# KERNEL / DISTANCE FILTERS
# ===========================================================================
def _subsample(n, cap, seed):
    if n <= cap:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=cap, replace=False)


def _double_centre(D):
    """A_ij = D_ij - rowmean - colmean + grandmean, on the last two axes."""
    r = D.mean(axis=-1, keepdims=True)
    c = D.mean(axis=-2, keepdims=True)
    g = D.mean(axis=(-2, -1), keepdims=True)
    return D - r - c + g


def s_dcor(X, y, ctx, n_sub=500, chunk=64):
    """Distance correlation per feature (Szekely, Rizzo & Bakirov 2007,
    Ann. Statist. 35(6):2769-2794).  dCor = 0 iff independent, so it catches
    nonlinear dependence a correlation misses.

    Reference: vnmabus/dcor `dcor/_dcor.py`.  The O(n log n) univariate algorithms
    (`_fast_dcov_avl.py`, `_fast_dcov_mergesort.py`) are the scalable route; here
    the naive doubly-centred form is used on `n_sub` subsampled rows and
    vectorised over blocks of features, which is exact on the subsample.
    """
    Xc = _clean(X)
    idx = _subsample(len(Xc), n_sub, ctx["seed"])
    Z = Xc[idx]
    yv = np.asarray(y, dtype=float)[idx]
    m = len(idx)

    B = _double_centre(np.abs(yv[:, None] - yv[None, :]))
    dvar_y = float((B * B).mean())
    if dvar_y <= 0:
        return np.zeros(Xc.shape[1])

    d = Z.shape[1]
    out = np.zeros(d)
    for a in range(0, d, chunk):
        b = min(a + chunk, d)
        col = Z[:, a:b].T                                   # (k, m)
        D = np.abs(col[:, :, None] - col[:, None, :])       # (k, m, m)
        A = _double_centre(D)
        dcov2 = (A * B[None, :, :]).mean(axis=(1, 2))
        dvar_x = (A * A).mean(axis=(1, 2))
        denom = np.sqrt(np.maximum(dvar_x * dvar_y, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[a:b] = np.sqrt(np.maximum(dcov2, 0.0) / np.where(denom > 0, denom, np.nan))
        del D, A
    return np.nan_to_num(out, nan=0.0)


def _centred_gram(v, sigma):
    """Centred Gaussian Gram matrix for one 1-D block of values."""
    D = (v[:, None] - v[None, :]) ** 2
    K = np.exp(-D / (2.0 * sigma ** 2))
    b = len(v)
    H = np.eye(b) - np.ones((b, b)) / b
    return H @ K @ H


def s_hsic_lasso(X, y, ctx, n_sub=1000, block=20, n_perm=3):
    """Block HSIC-Lasso (Yamada, Jitkrittum, Sigal, Xing & Sugiyama 2014,
    Neural Computation 26(1):185-207; block variant Climente-Gonzalez et al. 2019).

    min_{beta >= 0} || Kbar_y - sum_j beta_j Kbar_{x_j} ||_F^2 + lambda ||beta||_1

    Redundancy-aware like mRMR but kernel-based, so it needs NO discretisation —
    which matters because flab_expression_koenig and tcga_brca_pam50 are entirely
    continuous.  Reference: riken-aip/pyHSICLasso `pyHSICLasso/hsic_lasso.py`
    (the B / M arguments there ARE the block algorithm) and `nlars.py` for the
    non-negative solve; sklearn's positive Lasso solves the same objective.

    Run behind an HSIC relevance pre-screen to N_CAND features: the full design is
    O(n*B*M*d) and does not fit at d=4800.  Score is beta_j, zero-filled (S1) —
    the lasso solution is genuinely zero off the support.
    """
    from sklearn.linear_model import Lasso
    Xc = _clean(X)
    d = Xc.shape[1]
    idx = _subsample(len(Xc), n_sub, ctx["seed"])
    Z = _zs(Xc[idx])
    yv = np.asarray(y, dtype=float)[idx]
    if _is_clf(ctx):
        yv = np.unique(yv, return_inverse=True)[1].astype(float)
    yv = (yv - yv.mean()) / (yv.std() if yv.std() else 1.0)
    m = len(idx)
    rng = np.random.default_rng(ctx["seed"])

    sig_x, sig_y = 1.0, 1.0                      # standardised inputs
    nblk = m // block
    if nblk < 2:
        return np.zeros(d)

    # --- relevance pre-screen: block HSIC(x_j, y), cheap, d evaluations --------
    perm0 = rng.permutation(m)[:nblk * block].reshape(nblk, block)
    Ly0 = np.stack([_centred_gram(yv[p], sig_y).ravel() for p in perm0])   # (nblk,b^2)
    rel = np.zeros(d)
    for j in range(d):
        kj = np.stack([_centred_gram(Z[p, j], sig_x).ravel() for p in perm0])
        rel[j] = float((kj * Ly0).sum())
    cand = np.argsort(rel)[::-1][:min(N_CAND, d)]

    # --- full block design on the survivors -----------------------------------
    Krows, Lrows = [], []
    for _ in range(n_perm):
        perm = rng.permutation(m)[:nblk * block].reshape(nblk, block)
        Ly = np.concatenate([_centred_gram(yv[p], sig_y).ravel() for p in perm])
        cols = []
        for j in cand:
            cols.append(np.concatenate(
                [_centred_gram(Z[p, j], sig_x).ravel() for p in perm]))
        Krows.append(np.stack(cols, axis=1))
        Lrows.append(Ly)
    K = np.vstack(Krows).astype(np.float64)
    L = np.concatenate(Lrows).astype(np.float64)

    nrm = np.linalg.norm(K, axis=0)
    nrm[nrm == 0] = 1.0
    K = K / nrm
    alpha = 1e-3 * float(np.abs(K.T @ L).max()) / max(len(L), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beta = Lasso(alpha=alpha, positive=True, max_iter=3000,
                     fit_intercept=False).fit(K, L).coef_
    out = np.zeros(d)
    out[cand] = np.abs(beta)
    if not np.any(out > 0):                # degenerate lasso -> fall back to HSIC
        out[cand] = np.maximum(rel[cand], 0.0)
    return out


def s_laplacian(X, y, ctx, n_sub=1000, k=5):
    """CONTROL, unsupervised. Laplacian score (He, Cai & Niyogi, NIPS 2005):
    L_j = (f~^T L f~) / (f~^T D f~) on a kNN heat-kernel graph, LOWER is better —
    returned negated so higher = keep, matching the repo's own convention at
    tests/experiments/similarity_metrics/rankers.py:110.
    Reference: scikit-feature `similarity_based/lap_score.py`.

    Never sees the labels.  With the variance control it brackets how much of the
    hybrid's performance comes from label information at all.
    """
    Xc = _clean(X)
    idx = _subsample(len(Xc), n_sub, ctx["seed"])
    Z = _zs(Xc[idx])
    m = len(idx)
    sq = (Z * Z).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    np.fill_diagonal(D2, np.inf)
    kk = min(k, m - 1)
    nn = np.argpartition(D2, kk, axis=1)[:, :kk]
    t = float(np.median(D2[np.arange(m)[:, None], nn])) or 1.0
    W = np.zeros((m, m))
    rows = np.repeat(np.arange(m), kk)
    W[rows, nn.ravel()] = np.exp(-D2[rows, nn.ravel()] / t)
    W = np.maximum(W, W.T)

    dg = W.sum(axis=1)
    tot = dg.sum() or 1.0
    ft = Z - (Z.T @ dg / tot)[None, :]
    num = (ft * (dg[:, None] * ft)).sum(axis=0) - np.einsum("ij,jk,ik->i", ft.T, W, ft.T)
    den = (ft * (dg[:, None] * ft)).sum(axis=0)
    den[den == 0] = 1.0
    return -(num / den)


# ===========================================================================
# RELIEF FAMILY — interaction-sensitive without evaluating subsets.
# ===========================================================================
def _relief(X, y, ctx, cls_name):
    """skrebate. EpistasisLab/scikit-rebate `relieff.py` / `multisurf.py`.

    NOTE there is no separate RReliefF class in skrebate — ReliefF detects a
    continuous endpoint and switches to the RReliefF update rule internally, so
    one arm covers the three regression datasets and tcga alike.

    O(n^2 * d), so instances are subsampled to RELIEF_M — which is Relief's own
    definition (Kira & Rendell 1992 specify m sampled instances), not a shortcut.
    """
    import skrebate
    Xc = _zs(_clean(X))
    yv = np.asarray(y)
    idx = _subsample(len(Xc), RELIEF_M, ctx["seed"])
    Z, yy = Xc[idx], yv[idx]
    if _is_clf(ctx):
        yy = np.unique(yy, return_inverse=True)[1]
    else:
        yy = np.asarray(yy, dtype=float)
    est = getattr(skrebate, cls_name)
    kw = {"n_jobs": ctx.get("n_jobs", 1)}
    if cls_name == "ReliefF":
        kw["n_neighbors"] = 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = est(**kw).fit(Z, yy)
    return np.asarray(m.feature_importances_, dtype=float)


def s_relieff(X, y, ctx):
    """ReliefF / RReliefF (Kononenko 1994). skrebate ReliefF."""
    return _relief(X, y, ctx, "ReliefF")


def s_multisurf(X, y, ctx):
    """MultiSURF. Urbanowicz et al. 2018, J. Biomed. Informatics 85:168-188:
    'MultiSURF yields the most reliable feature selection performance across a
    wide range of problem types.'  skrebate MultiSURF — per-instance adaptive
    neighbour threshold, no k."""
    return _relief(X, y, ctx, "MultiSURF")


# ===========================================================================
# PERTURBATION — permutation's correlation fix.
# ===========================================================================
def s_conditional_permutation(X, y, ctx, n_cond=64, n_repeats=3):
    """Conditional permutation / CFI (Strobl et al. 2008; Chamma, Engemann &
    Thirion, NeurIPS 2023).  Reference: mind-inria/hidimstat
    `src/hidimstat/conditional_feature_importance.py`.

    Instead of permuting x_j marginally — which forces the model off-manifold and
    inflates correlated features (Hooker & Mentch 2021) — regress x_j on the other
    features, permute only the RESIDUAL, reconstruct, and measure the loss
    increase.  The conditioning set is a random subset of `n_cond` other features
    with a ridge conditioner, which is the published cheap variant; the subset size
    is recorded in every result file.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    Xz = _zs(_clean(X))
    yv = np.asarray(y)
    n, d = Xz.shape
    rng = np.random.default_rng(ctx["seed"])

    clf = _is_clf(ctx)
    cls = RandomForestClassifier if clf else RandomForestRegressor
    m = cls(n_estimators=50, max_depth=10, random_state=ctx["seed"],
            n_jobs=ctx.get("n_jobs", 1)).fit(Xz, yv)

    if clf:
        classes = list(m.classes_)
        yi = np.array([classes.index(v) for v in yv])

        def loss(Z):
            p = np.clip(m.predict_proba(Z), 1e-12, 1.0)
            return float(-np.log(p[np.arange(len(yi)), yi]).mean())
    else:
        yf = np.asarray(yv, dtype=float)

        def loss(Z):
            return float(((m.predict(Z) - yf) ** 2).mean())

    base = loss(Xz)
    out = np.zeros(d)
    kcond = min(n_cond, d - 1)
    for j in range(d):
        others = rng.choice(np.delete(np.arange(d), j), size=kcond, replace=False)
        resid = Xz[:, j] - Ridge(alpha=1.0).fit(Xz[:, others], Xz[:, j]).predict(Xz[:, others])
        fitted = Xz[:, j] - resid
        acc = 0.0
        for _ in range(n_repeats):
            Zp = Xz.copy()
            Zp[:, j] = fitted + rng.permutation(resid)
            acc += loss(Zp) - base
        out[j] = acc / n_repeats
    return out


# ===========================================================================
# WRAPPER / META — subset methods converted by S3 (never the boolean).
# ===========================================================================
def s_boruta_hits(X, y, ctx, max_iter=40):
    """Boruta (Kursa & Rudnicki 2010, J. Stat. Software 36(11)).
    Reference: scikit-learn-contrib/boruta_py `boruta/boruta_py.py`.

    S3 conversion: the score is the SHADOW HIT COUNT — the fraction of iterations
    in which the real feature's importance beat the maximum shadow importance —
    NOT the confirmed / tentative / rejected label, which is a three-way decision
    and would tie thousands of features.  The hit count is exactly the statistic
    Boruta's binomial test is run on, so nothing is invented.

    Implemented directly against the paper because BorutaPy exposes `ranking_`
    rather than the raw hit counts, and `ranking_` collapses every rejected
    feature to one value.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    Xz = _zs(_clean(X))
    yv = np.asarray(y)
    n, d = Xz.shape
    rng = np.random.default_rng(ctx["seed"])
    cls = RandomForestClassifier if _is_clf(ctx) else RandomForestRegressor
    hits = np.zeros(d)
    for it in range(max_iter):
        shadow = np.column_stack([rng.permutation(Xz[:, j]) for j in range(d)])
        m = cls(n_estimators=50, max_depth=10, random_state=ctx["seed"] + it,
                n_jobs=ctx.get("n_jobs", 1)).fit(np.hstack([Xz, shadow]), yv)
        imp = np.asarray(m.feature_importances_, dtype=float)
        hits += (imp[:d] > imp[d:].max()).astype(float)
    return hits / max_iter


def s_stability_selection(X, y, ctx):
    """Stability selection (Meinshausen & Buhlmann 2010, JRSS-B 72(4):417-473).
    Reference: scikit-learn-contrib/stability-selection `stability_selection.py`
    (repo archived, so the algorithm is implemented here rather than depended on);
    the engine already has the same idea as an allocator at hybrid.py:1835.

    Subsample n/2 rows N_BOOT times, re-run the base scorer, count how often each
    feature lands in the top K.  The selection frequency IS a per-feature score in
    [0,1] — S3 with no conversion.  Also the 'aggregate across bootstraps' half of
    ensemble feature selection (Saeys et al. 2008).
    """
    Xc = _clean(X)
    n, d = Xc.shape
    rng = np.random.default_rng(ctx["seed"])
    topk = int(min(max(ctx.get("budget", 50), 10), d))
    base = ctx.get("stability_base", "lightgbm_gain")
    counts = np.zeros(d)
    for b in range(N_BOOT):
        sub = rng.choice(n, size=max(n // 2, 20), replace=False)
        sc = SCORERS[base](Xc[sub], np.asarray(y)[sub],
                           dict(ctx, seed=ctx["seed"] + b))
        counts[np.argsort(sc)[-topk:]] += 1.0
    return counts / N_BOOT


def s_lasso(X, y, ctx):
    """Lasso (Tibshirani 1996). |coef_| is the score, S3 — exactly zero for dropped
    features.  Under correlation it arbitrarily keeps one of a group, which is the
    behaviour the elastic-net arm is paired against.
    sklearn.linear_model.LassoCV / LogisticRegression(penalty='l1')."""
    from sklearn.linear_model import LassoCV, LogisticRegression
    Xz = _zs(_clean(X))
    yv = np.asarray(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _is_clf(ctx):
            m = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=400,
                                   random_state=ctx["seed"]).fit(Xz, yv)
            return np.abs(np.asarray(m.coef_)).mean(axis=0)
        m = LassoCV(n_alphas=20, cv=3, random_state=ctx["seed"],
                    n_jobs=ctx.get("n_jobs", 1), max_iter=2000).fit(Xz, yv)
        return np.abs(np.asarray(m.coef_))


def s_elastic_net(X, y, ctx):
    """Elastic net (Zou & Hastie 2005). The grouping effect keeps correlated
    features together rather than picking one — the direct counterpoint to lasso
    at a fixed budget.  sklearn ElasticNetCV / LogisticRegression(l1_ratio=0.5)."""
    from sklearn.linear_model import ElasticNetCV, LogisticRegression
    Xz = _zs(_clean(X))
    yv = np.asarray(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _is_clf(ctx):
            m = LogisticRegression(penalty="elasticnet", solver="saga", C=0.1,
                                   l1_ratio=0.5, max_iter=400,
                                   random_state=ctx["seed"]).fit(Xz, yv)
            return np.abs(np.asarray(m.coef_)).mean(axis=0)
        m = ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=3, random_state=ctx["seed"],
                         n_jobs=ctx.get("n_jobs", 1), max_iter=2000).fit(Xz, yv)
        return np.abs(np.asarray(m.coef_))


def s_matching_pursuit(X, y, ctx):
    """Greedy forward / matching pursuit on (X, y) — the complementarity-aware arm.

    Repeatedly take the feature most correlated with the RESIDUAL of y, then
    project that direction out, so a feature is chosen because it supplies signal
    the already-chosen features lack.  This is the closed-form substitute for
    sequential forward selection: SFS refits the model once per candidate per
    step (~100,000 fits for one representation), this needs none.

    Already implemented in the repo at
    tests/experiments/feature_selectors/selector_prediction.py:135 and re-derived
    here so f30 owns its own copy.  Score by S2.
    """
    Xz = _zs(_clean(X))
    d = Xz.shape[1]
    yv = np.asarray(y, dtype=float)
    if _is_clf(ctx):
        yv = np.unique(yv, return_inverse=True)[1].astype(float)
    r = yv - yv.mean()
    budget = int(min(ctx.get("budget", N_CAND), d))

    norms = np.linalg.norm(Xz, axis=0)
    norms[norms == 0] = 1.0
    B = Xz / norms
    picked, mask = [], np.ones(d, dtype=bool)
    for _ in range(budget):
        c = np.abs(B.T @ r)
        c[~mask] = -np.inf
        j = int(np.argmax(c))
        if not np.isfinite(c[j]) or c[j] <= 1e-12:
            break
        picked.append(j)
        mask[j] = False
        v = B[:, j]
        r = r - v * float(v @ r)
    return _order_to_score(np.asarray(picked, dtype=int), d)


# ===========================================================================
# BLENDS — the three published aggregation families from the survey (§7).
# All three already exist in this repo; nothing is invented.
#   rank averaging / Borda  -> tests/experiments/similarity_metrics/rankers.py, `combine_ranks`
#   reciprocal rank fusion  -> tests/experiments/similarity_metrics/aggregate.py:90
#   bootstrap ensembling    -> s_stability_selection above
# WHICH methods get blended is decided from pass 1's agreement matrix, not here:
# fusing two methods that already rank features the same way does nothing.
# ===========================================================================
def blend_borda(score_vectors):
    """Mean rank across methods (Saeys et al. 2008 ensemble FS; the fusion already
    used at `combine_ranks` in similarity_metrics/rankers.py).  Reference didayolo/ranky
    `ranky/ranking.py` `borda`."""
    from scipy.stats import rankdata
    R = np.vstack([rankdata(np.asarray(s, dtype=float)) for s in score_vectors])
    return R.mean(axis=0)


def blend_rrf(score_vectors, k=60):
    """Reciprocal rank fusion, sum 1/(k + rank). Top-weighted alternative to Borda;
    already in the repo at aggregate.py:90 for neighbour rankings."""
    from scipy.stats import rankdata
    out = np.zeros(len(score_vectors[0]))
    for s in score_vectors:
        r = rankdata(-np.asarray(s, dtype=float))       # rank 1 = best
        out += 1.0 / (k + r)
    return out


BLENDS = {"borda": blend_borda, "rrf": blend_rrf}


# ===========================================================================
# REGISTRY
# ===========================================================================
SCORERS = {
    # incumbents (engine branches)
    "lightgbm_gain": s_lightgbm_gain,
    "lightgbm_gain_nocap": s_lightgbm_gain_nocap,
    "lightgbm_gain_deep": s_lightgbm_gain_deep,
    "xgboost_gain": s_xgboost_gain,
    "treeshap": s_treeshap,
    "kernelshap": s_kernelshap,
    "gradientshap": s_gradientshap,
    "deeplift": s_deeplift,
    "integrated_gradients": s_integrated_gradients,
    "permutation": s_permutation,
    # new
    "catboost_importance": s_catboost_importance,
    "rf_mdi": s_rf_mdi,
    "treeshap_interventional": s_treeshap_interventional,
    "f_test": s_f_test,
    "spearman_marginal": s_spearman_marginal,
    "mutual_info": s_mutual_info,
    "mrmr": s_mrmr,
    "jmi": s_jmi,
    "cmim": s_cmim,
    "hsic_lasso": s_hsic_lasso,
    "dcor": s_dcor,
    "relieff": s_relieff,
    "multisurf": s_multisurf,
    "conditional_permutation": s_conditional_permutation,
    "boruta_hits": s_boruta_hits,
    "stability_selection": s_stability_selection,
    "lasso": s_lasso,
    "elastic_net": s_elastic_net,
    "matching_pursuit": s_matching_pursuit,
    # controls
    "variance": s_variance,
    "laplacian": s_laplacian,
    "random": s_random,
}

# family / redundancy-awareness / conversion rule, carried into every result file
# so the analysis never has to re-derive them.
META = {
    "lightgbm_gain":            ("embedded_tree",  False, "S1", "incumbent"),
    "lightgbm_gain_nocap":      ("embedded_tree",  False, "S1", "incumbent"),
    "lightgbm_gain_deep":       ("embedded_tree",  False, "S1", "incumbent"),
    "xgboost_gain":             ("embedded_tree",  False, "S1", "incumbent"),
    "treeshap":                 ("posthoc_shapley", True, "S1", "incumbent"),
    "kernelshap":               ("posthoc_shapley", True, "S1", "incumbent"),
    "gradientshap":             ("posthoc_gradient", False, "S1", "incumbent"),
    "deeplift":                 ("posthoc_gradient", False, "S1", "incumbent"),
    "integrated_gradients":     ("posthoc_gradient", False, "S1", "incumbent"),
    "permutation":              ("wrapper",        False, "S1", "incumbent"),
    "catboost_importance":      ("embedded_tree",  False, "S1", "new"),
    "rf_mdi":                   ("embedded_tree",  False, "S1", "new"),
    "treeshap_interventional":  ("posthoc_shapley", False, "S1", "new"),
    "f_test":                   ("filter_univariate", False, "S1", "new"),
    "spearman_marginal":        ("filter_univariate", False, "S1", "new"),
    "mutual_info":              ("filter_info",    False, "S1", "new"),
    "mrmr":                     ("filter_info",     True, "S2", "new"),
    "jmi":                      ("filter_info",     True, "S2", "new"),
    "cmim":                     ("filter_info",     True, "S2", "new"),
    "hsic_lasso":               ("filter_kernel",   True, "S1", "new"),
    "dcor":                     ("filter_kernel",  False, "S1", "new"),
    "relieff":                  ("filter_relief",   True, "S1", "new"),
    "multisurf":                ("filter_relief",   True, "S1", "new"),
    "conditional_permutation":  ("wrapper",         True, "S1", "new"),
    "boruta_hits":              ("wrapper",        False, "S3", "new"),
    "stability_selection":      ("meta",           False, "S3", "new"),
    "lasso":                    ("embedded_linear", True, "S3", "new"),
    "elastic_net":              ("embedded_linear", True, "S3", "new"),
    "matching_pursuit":         ("wrapper",         True, "S2", "new"),
    "variance":                 ("control",        False, "S1", "control"),
    "laplacian":                ("control",        False, "S1", "control"),
    "random":                   ("control",        False, "S1", "control"),
}

# USER DECISION 2026-09-03: xgboost_gain is not run. It was the single dearest arm
# on tcga (804.5s of scoring, the largest of eleven) and lightgbm_gain and
# treeshap already cover gradient-boosting gain. The scorer stays defined so it can
# be re-added by naming it in --arms.
DROPPED = {"xgboost_gain"}
PASS1 = [m for m in SCORERS if m not in DROPPED]

# Arms that build a torch model.
#
# ⚠ macOS ONLY. On darwin LightGBM loads libomp.dylib and torch loads
# libiomp5.dylib — two OpenMP runtimes in one process (src/kirby/hybrid.py:41-66,
# which states plainly: "Leave it unset on the server, where there is one OpenMP
# runtime and nothing to work around"). MEASURED on this Mac, same script both ways:
#   OMP_NUM_THREADS=1 -> exit 0, deeplift 8.65s
#   OMP_NUM_THREADS=8 -> exit 139 (SIGSEGV), dies at the first torch call
# NOT measured on Linux and not expected there. The guard below is therefore
# darwin-only: on the cluster these arms keep every thread the task was given.
TORCH_ARMS = {"deeplift", "integrated_gradients", "gradientshap"}


def assert_omp_safe(method):
    if sys.platform != "darwin":
        return                     # one OpenMP runtime on Linux; nothing to guard
    if method in TORCH_ARMS and os.environ.get("OMP_NUM_THREADS") not in ("1", None):
        raise RuntimeError(
            f"{method} builds a torch model and would SIGSEGV here: "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, needs 1. "
            f"Two OpenMP runtimes in one process (hybrid.py:41-66). "
            f"Re-run this arm with OMP_NUM_THREADS=1.")


def score(method, X, y, ctx):
    """Run one scorer and return (score_vector, seconds).

    A failure RAISES — f30 does not copy the silent drop at
    find_best_hybrid.py:4473.  The caller records which method failed on which
    representation and fold.
    """
    assert_omp_safe(method)
    t0 = time.time()
    v = SCORERS[method](_clean(X), np.asarray(y), ctx)
    v = np.asarray(v, dtype=float).ravel()
    if v.shape[0] != X.shape[1]:
        raise ValueError(
            f"{method}: returned {v.shape[0]} scores for {X.shape[1]} features — "
            f"concatenate_features (hybrid.py:4054) requires an exact match")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{method}: non-finite scores")
    return v, time.time() - t0
