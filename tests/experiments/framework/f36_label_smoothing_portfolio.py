#!/usr/bin/env python3
"""f36 -- which label-smoothing combinations the noise suite should carry.

Owner's question, 2026-09-29: do different tasks (dataset, model input, model,
noise condition) prefer different label smoothing? If not, the noise suite
sweeps one. If they do, which smallest set must every noise experiment try so
that each task's best is inside it? A framework experiment: it decides what is
built, nothing here is published. Design and the owner's answers: F36.md.

WHAT IS COMPUTED. Every label smoothing in the repo replaces a training label y
with

    new y = y + a * (v - y)

v is the value the label is pulled toward, a is how far it moves (0 leaves it,
1 replaces it). f36 runs every combination of:

  v, 12 ways        VALUES (below). Two more were read and are the same
                    computation as one already here, so they are not run
                    twice: n23's gaussian_process prediction is
                    GaussianProcessEstimator(n_folds=5, groups) exactly as
                    gp_oof; n23's activity_cliff_aware sets no predictions_ and
                    falls back to the neighbour mean (n23:1066-1075).
  a, 31 ways        RULES (below). Each one returns the new label column.
  preserve, 4 ways  none, variance, group_mean, rank -- applied after the move.
  scale, 3 ways     raw, residual, rank -- the space the move is made in.
  passes, 3 ways    once, fixed (refit v on the moved labels and repeat until
                    no label moves by more than 1e-3 x the label SD, at most 5
                    passes), anneal (3 passes, the move halved each pass, v
                    refitted each pass).

NOT RUN, because the result is the same numbers as another combination:
scale=residual with preserve none or group_mean. With p fixed, the residual is
y - p and v - p, so y - p + a((v - p) - (y - p)) + p = y + a(v - y): every rule
here reads y only through y - v, the suspicion score or quantities fixed from
D, so the column is identical to scale=raw. Group-mean preservation subtracts
each group's mean CHANGE, and a change is the same in either space. The smoke
(_smoke_f36.py) checks both identities numerically.

On a classification dataset (herg) a label is a class. Only one move is
defined there: replace a flagged label with its neighbours' majority class
(n07's knn_smooth and prox_smooth; n07's label_smooth on a class label is the
same computation as knn_smooth, n07:967-984). So herg runs v in {9-neighbour
majority, forest-proximity majority} x flag share {0.05, 0.10, 0.20} x passes
{once, fixed}: 12 combinations. Preserve and scale are not defined on a class.

THE MEASURE is D vs D' (tests/experiments/noise/d_vs_dprime.py): the model
fitted on the noisy training fold (D) and on the smoothed one (D'), scored on
the same held-out fold in the dataset's own metric, beside the model fitted on
the clean training labels. lower_is_better from get_domain_metric.

TASKS. Datasets logd, flab_expression_koenig, herg (precompute DONE on all
three, checked 2026-09-30 with tests/tools/precompute_status.py). Models rf and
svm. Model inputs: each model's top-2 single representations and its
prediction-built hybrid, read from the precompute (noise_base_reps_by_model,
pool_recipe_per_downstream_model; the hybrid is built per fold from the noisy
training labels by n07.fold_similarity_spaces, as n42 does). Conditions:
outlier_p10, outlier_only_p10, censoring on regression; uniform on herg. Every
level of each condition's grid (dataset_policy.noise_levels), draws 0, 1, 2
(one draw at level 0 on regression, n23._levels_and_draws). 5 folds (config
n_splits). No search ever starts: KIRBY_NO_COLD_SEARCH=1 is set here.

The suspicion score every score-reading rule uses is n07's detector,
cross_val (n07:483), computed once on D and held fixed across passes.
Neighbours are today's: Euclidean in the model input, 9 neighbours
(make_nearest_neighbors(10) minus the molecule itself, as n07), and the
forest's 10 nearest by leaf co-occurrence (as n07 prox_smooth). n42 sweeps
every other neighbour choice for these two, so f36 does not.

ORDER OF WORK. All units under passes=once first, then fixed, then anneal
(KIRBY_F36_PASSES picks). Within a schedule, units are dealt to parallel
processes by stride (KIRBY_F36_SLICE=K/N). Rows are written per unit file and
a finished (schedule, unit) is recorded, so a killed run resumes.

OUTPUT: <output>/f36_label_smoothing_portfolio/<dataset>/<condition>/
<unit>.rows.jsonl, one d_vs_dprime row per (combination, model).
method = 'f36/smooth', method_setting = 'v=..|a=..|preserve=..|scale=..|passes=..'.
f36_analyse.py reads them.
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

EXP_ID = 'f36_label_smoothing_portfolio'

# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

# v: what the label is pulled toward. Each is computed on the training fold
# only, out of fold where a model is fitted.
VALUES = (
    'knn9_mean',           # mean label of 9 nearest, Euclidean (n07 label_smooth)
    'forest10_mean',       # mean label of 10 nearest by forest leaves (n07 prox_smooth)
    'lgbm_oof',            # out-of-fold LightGBM, 5 group-aware inner folds (n07 oof_smooth)
    'lgbm_huber_oof',      # the same with a Huber loss, alpha = 1.0 x s (n33 teacher)
    'gp_oof',              # Gaussian process posterior mean (n07 gp_smooth)
    'qrf_q50',             # quantile forest, conditional median
    'qrf_q25',             # quantile forest, 0.25 quantile
    'qrf_q75',             # quantile forest, 0.75 quantile
    'pair_mean',           # mean of the molecule and its near-duplicate partners (n35 pairs)
    'hetero_gp_oof',       # n23 source's own out-of-fold prediction
    'ngboost_oof',         # n23 source's own out-of-fold prediction
    'heteroscedastic_nn_oof',  # n23 source's own out-of-fold prediction
)

# Two v's that were read and are the same computation as one above.
VALUES_NOT_RUN = {
    'n23 gaussian_process': (
        'GaussianProcessEstimator(n_folds=5, groups) -- the call gp_oof makes '
        '(n07:1163 and n23._estimator_kwargs), so the same posterior mean.'),
    'n23 activity_cliff_aware': (
        'sets no predictions_, so n23 moves toward the 10-neighbour mean '
        '(n23:1066-1075): the computation knn9_mean already is.'),
}

# The values swept for each rule family. Every value is one the suite already
# runs somewhere, except the redescending zero points, the uncertainty gate and
# the agreement gate, which are not built anywhere else (F36.md section 1d).
PROP = (0.25, 0.5, 1.0)          # n18, n23, n25 strengths
CONST = (0.25, 0.5, 1.0)         # n27 ALPHAS
FLAG = (0.05, 0.10, 0.20)        # n07 FLAG_SHARES
JS = (0.5, 1.0, 2.0)             # n25 js_shrink strengths
WINSOR = (1.0, 1.25, 1.5)        # n30 WINSOR_C
SBS_RULES = ('threshold', 'soft_threshold', 'confidence')  # smooth_by_scores
SBS_ALPHA = 0.5
REDESC_ZERO = (3.0, 4.685)       # where the share returns to 0, in units of s
GATE_SHARE = 0.5                 # the most a gated rule moves a label
UNC_FLOOR_PCT = 10.0             # floor on the uncertainty, its 10th percentile


def rule_names():
    out = [f'prop{a:g}' for a in PROP]
    out += [f'const{a:g}' for a in CONST]
    out += [f'flagtop{f:g}' for f in FLAG]
    out += [f'js{s:g}' for s in JS]
    out += [f'winsor_c{c:g}' for c in WINSOR]
    out += [f'{r}{SBS_ALPHA:g}' for r in SBS_RULES]
    out += [f'n31_bayes_shrink_flag{f:g}' for f in FLAG]
    out += [f'n32_one_weight_flag{f:g}' for f in FLAG]
    out += [f'n32_per_row_flag{f:g}' for f in FLAG]
    out += [f'redesc_zero{c:g}' for c in REDESC_ZERO]
    out += ['uncertainty_gate', 'agreement_gate']
    return tuple(out)


RULES = rule_names()
PRESERVES = ('none', 'variance', 'group_mean', 'rank')
SCALES = ('raw', 'residual', 'rank')
PASSES = ('once', 'fixed', 'anneal')
FIXED_MAX_PASSES = 5
FIXED_TOL_SD = 1e-3
ANNEAL_PASSES = 3
ANNEAL_DECAY = 0.5

# Combinations that are the same numbers as another, by algebra (module doc).
DEGENERATE = {('residual', 'none'): 'identical to scale=raw, preserve=none',
              ('residual', 'group_mean'): 'identical to scale=raw, '
                                          'preserve=group_mean'}

CLF_VALUES = ('knn9_majority', 'forest10_majority')
CLF_PASSES = ('once', 'fixed')

REGRESSION_CONDITIONS = ('outlier_p10', 'outlier_only_p10', 'censoring')
CLASSIFICATION_CONDITIONS = ('uniform',)
MODELS = ('rf', 'svm')
MODEL_INPUTS = ('base_1', 'base_2', 'hybrid')

# Which stored searches the model inputs are read from: the prediction-built
# ones only. n42 reads 'predictive_and_similarity', and that made the noise
# loader complete the similarity-built search for logd, 74% stored, inside the
# first f36 job (2026-10-02 log). f36's hybrid input is the prediction-built
# recipe and its single representations are ranked by prediction, so the
# similarity-built search is never read.
POOL_SUBSET = 'predictive_only'
SPACE_FOR_FAMILY = {'predictive': 'hybrid'}
NOISE_DRAWS = (0, 1, 2)
INNER_FOLDS = 5
PAIR_PERCENTILE = 1.0            # n35 CLOSE_PERCENTILE
HUBER_C = 1.0                    # n33 TEACHER_HUBER_C


def combinations(is_clf):
    """Every (v, rule, preserve, scale, passes) this task type runs, and the
    ones skipped with their reason."""
    if is_clf:
        return ([(v, f'flagtop{f:g}', 'none', 'raw', p)
                 for p in CLF_PASSES for v in CLF_VALUES for f in FLAG], {})
    run, skipped = [], {}
    for p in PASSES:
        for v in VALUES:
            for r in RULES:
                for pr in PRESERVES:
                    for sc in SCALES:
                        why = DEGENERATE.get((sc, pr))
                        if why:
                            skipped[(v, r, pr, sc, p)] = why
                            continue
                        run.append((v, r, pr, sc, p))
    return run, skipped


def setting_label(v, r, pr, sc, p):
    return f'v={v}|a={r}|preserve={pr}|scale={sc}|passes={p}'


# ---------------------------------------------------------------------------
# Pure computations: no I/O, tested by the smoke
# ---------------------------------------------------------------------------

def robust_s(r):
    """1.4826 x median |r|, n30's residual scale."""
    return float(1.4826 * np.median(np.abs(np.asarray(r, dtype=float))))


def norm01(sc):
    sc = np.asarray(sc, dtype=float)
    rng = np.ptp(sc)
    if not np.isfinite(rng) or rng <= 0:
        return np.zeros_like(sc)
    return (sc - sc.min()) / rng


def top_share(score, share):
    """n07's flag: score at or above its (1 - share) percentile."""
    thr = np.percentile(score, 100.0 * (1.0 - float(share)))
    return np.asarray(score) >= thr


def apply_rule(rule, z, vz, score, ctx):
    """The moved column for one rule, in whatever space z and vz are in.

    score: the suspicion score of D, raw. ctx: 'unc' (uncertainty per molecule,
    for the gate), 'disagree' (|two predictions| per molecule) and 'disagree_s'
    (its scale), all fixed from D.
    """
    from kirby.utils.noise_estimation import smooth_by_scores
    z = np.asarray(z, dtype=float)
    vz = np.asarray(vz, dtype=float)
    r = z - vz

    def move(share):
        return z + np.asarray(share, dtype=float) * (vz - z)

    if rule.startswith('prop'):
        return move(float(rule[4:]) * norm01(score))
    if rule.startswith('const'):
        return move(np.full(z.size, float(rule[5:])))
    if rule.startswith('flagtop'):
        return move(top_share(score, float(rule[7:])).astype(float))
    if rule.startswith('js'):
        # n07:950-964 with v in place of the LightGBM prediction.
        tau2 = float(np.var(r))
        if not np.isfinite(tau2) or tau2 <= 0:
            return z.copy()
        sigma2 = norm01(score) * tau2 * float(rule[2:])
        return move(sigma2 / (sigma2 + tau2))
    if rule.startswith('winsor_c'):
        # n30.winsorise with v in place of the LightGBM prediction.
        c = float(rule[len('winsor_c'):])
        s = robust_s(r)
        if not np.isfinite(s) or s <= 0:
            return z.copy()
        return vz + np.clip(r, -c * s, c * s)
    for name in SBS_RULES:
        if rule.startswith(name) and rule[len(name):].replace('.', '').isdigit():
            # smooth_by_scores' own formula; its share is read off by moving
            # 0 toward 1.
            share = smooth_by_scores(np.zeros(z.size),
                                     np.asarray(score, dtype=float),
                                     np.ones(z.size),
                                     alpha=float(rule[len(name):]),
                                     method=name)
            return move(share)
    if rule.startswith('n31_bayes_shrink_flag'):
        from experiments.noise.n31_regime_and_breakeven import (
            estimated_shrink_weight)
        flag = top_share(score, float(rule[len('n31_bayes_shrink_flag'):]))
        w = estimated_shrink_weight(z, vz, flag)
        return move(np.where(flag, w, 0.0))
    if rule.startswith('n32_one_weight_flag') or rule.startswith('n32_per_row_flag'):
        from experiments.noise.n32_iterative_and_dual_selection import (
            shrink_flagged)
        share = float(rule.rsplit('flag', 1)[1])
        one, per, _info = shrink_flagged(z, vz, share)
        if one is None:
            return z.copy()
        return one if rule.startswith('n32_one') else per
    if rule.startswith('redesc_zero'):
        # Share 4u(1-u) for u = |r| / (c s) below 1, 0 above: rises from 0,
        # peaks at 1 when |r| = c s / 2, falls back to 0 at |r| = c s.
        c = float(rule[len('redesc_zero'):])
        s = robust_s(r)
        if not np.isfinite(s) or s <= 0:
            return z.copy()
        u = np.abs(r) / (c * s)
        return move(np.where(u < 1.0, 4.0 * u * (1.0 - u), 0.0))
    if rule == 'uncertainty_gate':
        unc = ctx.get('unc')
        if unc is None:
            return None
        unc = np.asarray(unc, dtype=float)
        floor = max(float(np.percentile(unc, UNC_FLOOR_PCT)), 1e-12)
        return move(GATE_SHARE * floor / np.maximum(unc, floor))
    if rule == 'agreement_gate':
        dis = ctx.get('disagree')
        s = ctx.get('disagree_s')
        if dis is None or s is None or not np.isfinite(s) or s <= 0:
            return None
        return move(GATE_SHARE * np.exp(-np.asarray(dis, dtype=float) / s))
    raise ValueError(f'f36: unknown rule {rule!r}')


def preserve(pr, zn, z_before, groups=None):
    """What the move may not change, restored after it."""
    zn = np.asarray(zn, dtype=float)
    z_before = np.asarray(z_before, dtype=float)
    if pr == 'none':
        return zn
    if pr == 'variance':
        sd_new, sd_old = float(np.std(zn)), float(np.std(z_before))
        if sd_new <= 0 or not np.isfinite(sd_new):
            return zn
        mu = float(np.mean(zn))
        return mu + (zn - mu) * (sd_old / sd_new)
    if pr == 'group_mean':
        if groups is None:
            return None
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        d = zn - z_before
        mean_d = np.bincount(inv, weights=d) / np.bincount(inv)
        return zn - mean_d[inv]
    if pr == 'rank':
        out = np.empty_like(zn)
        out[np.argsort(zn, kind='stable')] = np.sort(z_before)
        return out
    raise ValueError(f'f36: unknown preserve {pr!r}')


class Scale:
    """The space the move is made in, fixed from D's labels."""

    def __init__(self, sc, y0, p0=None):
        self.sc = sc
        if sc == 'residual':
            if p0 is None:
                raise ValueError('residual scale needs the base prediction')
            self.p0 = np.asarray(p0, dtype=float)
        elif sc == 'rank':
            y0 = np.asarray(y0, dtype=float)
            n = y0.size
            ys = np.sort(y0)
            grid = (np.arange(n) + 0.5) / n
            # Tied labels (a censoring limit) share one grid position: the
            # mean of theirs, so the map is a function.
            self.uy, inv = np.unique(ys, return_inverse=True)
            self.gy = np.bincount(inv, weights=grid) / np.bincount(inv)

    def fwd(self, y):
        y = np.asarray(y, dtype=float)
        if self.sc == 'raw':
            return y.copy()
        if self.sc == 'residual':
            return y - self.p0
        if self.uy.size == 1:
            return np.full(y.size, 0.5)
        return np.interp(y, self.uy, self.gy)

    def back(self, z):
        z = np.asarray(z, dtype=float)
        if self.sc == 'raw':
            return z.copy()
        if self.sc == 'residual':
            return z + self.p0
        if self.uy.size == 1:
            return np.full(z.size, float(self.uy[0]))
        return np.interp(z, self.gy, self.uy)


def smooth(y0, v0, rule, pr, sc, passes, *, score, ctx, groups=None,
           p0=None, refit_v=None):
    """D' labels for one combination. refit_v(y) returns v for labels y; it
    is called before every pass after the first. Returns (labels, passes run)
    or (None, why)."""
    y0 = np.asarray(y0, dtype=float)
    scale = Scale(sc, y0, p0)
    n_pass = {'once': 1, 'fixed': FIXED_MAX_PASSES,
              'anneal': ANNEAL_PASSES}[passes]
    tol = FIXED_TOL_SD * float(np.std(y0))
    y = y0.copy()
    v = np.asarray(v0, dtype=float)
    ran = 0
    for k in range(n_pass):
        if k > 0:
            v = refit_v(y)
            if v is None:
                return None, f'v could not be refitted at pass {k + 1}'
            v = np.asarray(v, dtype=float)
        z, vz = scale.fwd(y), scale.fwd(v)
        zn = apply_rule(rule, z, vz, score, ctx)
        if zn is None:
            return None, f'rule {rule} has no input on this unit'
        zn = preserve(pr, zn, z, groups)
        if zn is None:
            return None, f'preserve={pr} needs scaffold groups and there are none'
        y_new = scale.back(zn)
        if passes == 'anneal':
            y_new = y + (ANNEAL_DECAY ** k) * (y_new - y)
        delta = float(np.max(np.abs(y_new - y))) if y.size else 0.0
        y = y_new
        ran = k + 1
        if passes == 'fixed' and delta < tol:
            break
    if not np.all(np.isfinite(y)):
        return None, 'non-finite labels'
    return y, ran


def majority(y, nbrs):
    """Each row's neighbours' most common class (scipy mode, as n07)."""
    from scipy.stats import mode as scipy_mode
    return scipy_mode(np.asarray(y)[nbrs], axis=1).mode.ravel()


def smooth_clf(y0, nbrs, share, passes, score):
    """n07 knn_smooth on a class label: flagged labels take the neighbours'
    majority; under passes=fixed the majority is retaken from the moved labels
    until nothing changes, at most FIXED_MAX_PASSES."""
    flag = top_share(score, share)
    y = np.asarray(y0).copy()
    n_pass = 1 if passes == 'once' else FIXED_MAX_PASSES
    ran = 0
    for k in range(n_pass):
        new = y.copy()
        new[flag] = majority(y, nbrs)[flag]
        changed = int(np.sum(new != y))
        y = new
        ran = k + 1
        if changed == 0:
            break
    return y, ran


def pair_mean_values(y, pairs_i, pairs_j):
    """Mean label of each molecule and its close partners; a molecule in no
    close pair keeps its own label (so it does not move)."""
    y = np.asarray(y, dtype=float)
    s = y.copy()
    c = np.ones(y.size)
    np.add.at(s, pairs_i, y[pairs_j])
    np.add.at(c, pairs_i, 1.0)
    np.add.at(s, pairs_j, y[pairs_i])
    np.add.at(c, pairs_j, 1.0)
    return s / c


# ---------------------------------------------------------------------------
# The values v, for one unit
# ---------------------------------------------------------------------------

class Values:
    """v for every name in VALUES on labels y, for one unit's training rows.

    Labels y0 (D): model fits go through noise_cache, shared with every other
    experiment that fits the same thing on the same draw. Any other labels
    (a later pass): fitted directly, never stored, because each combination's
    later passes see labels nobody else will.
    Neighbour lists, forest neighbours and close pairs are found once on D
    and kept across passes; only the labels averaged over them change.
    """

    def __init__(self, ds, X, y0, groups, errors):
        self.ds, self.X, self.y0, self.g = ds, X, np.asarray(y0, float), groups
        self.errors = errors
        self._nbrs = self._forest = self._pairs = None
        self._d0 = {}
        self.unavailable = {}

    # -- fixed structure --
    def nbrs(self):
        if self._nbrs is None:
            from experiments._shared import make_nearest_neighbors
            nn = make_nearest_neighbors(n_neighbors=min(10, len(self.y0) - 1))
            nn.fit(self.X)
            self._nbrs = nn.kneighbors(self.X)[1][:, 1:]
        return self._nbrs

    def forest(self, is_clf=False, y=None):
        if self._forest is None:
            from experiments.noise import noise_cache as nc
            from kirby.utils.noise_estimation import forest_proximity_distance
            leaves = nc.cached_forest_leaf_ids(
                self.ds, self.X, self.y0 if y is None else y, is_clf)[0]
            D = forest_proximity_distance(leaves)
            np.fill_diagonal(D, np.inf)
            self._forest = np.argsort(D, axis=1)[:, :min(10, len(self.y0) - 1)]
        return self._forest

    def pairs(self):
        if self._pairs is None:
            from scipy.spatial.distance import cdist
            from experiments.noise.n35_near_duplicate_disagreement import (
                close_pairs)
            D = cdist(self.X, self.X).astype(np.float32)
            i, j, _d, thr, _tot = close_pairs(D, PAIR_PERCENTILE)
            del D
            self._pairs = (i, j, thr)
        return self._pairs

    # -- the values --
    def at(self, name, y):
        y = np.asarray(y, dtype=float)
        first = np.array_equal(y, self.y0)
        if first and name in self._d0:
            return self._d0[name]
        try:
            v = self._compute(name, y, first)
        except Exception as exc:
            self.errors.add(f'v={name}', exc)
            v = None
        if first:
            self._d0[name] = v
        return v

    def _oof(self, y, first, params=None):
        from experiments.noise import noise_cache as nc
        from kirby.utils.noisy_labels import oof_predict
        k = min(INNER_FOLDS, len(y))
        if first:
            return nc.cached_oof_predict(self.ds, self.X, y, self.g,
                                         n_splits=k, params=params)
        return np.asarray(oof_predict(self.X, y, self.g, n_splits=k,
                                      params=params))

    def _cached_or_not(self, first, kind, key, fn):
        from experiments.noise import noise_cache as nc
        if not first:
            return fn()
        key = dict(key, X=nc.array_hash(self.X), y=nc.array_hash(self.y0),
                   groups=nc.array_hash(self.g))
        return nc.cached(self.ds, kind, key, fn)

    def _compute(self, name, y, first):
        if name == 'knn9_mean':
            return np.mean(y[self.nbrs()], axis=1)
        if name == 'forest10_mean':
            return np.mean(y[self.forest()], axis=1)
        if name == 'lgbm_oof':
            return self._oof(y, first)
        if name == 'lgbm_huber_oof':
            s = robust_s(y - self._oof(y, first))
            if not np.isfinite(s) or s <= 0:
                return None
            return self._oof(y, first, params={'objective': 'huber',
                                               'alpha': HUBER_C * s})
        if name == 'gp_oof':
            from experiments.noise.n07_remediation import GP_SMOOTH_MAX_ROWS
            if len(y) > GP_SMOOTH_MAX_ROWS:
                self.unavailable['gp_oof'] = (
                    f'{len(y)} training molecules is over GP_SMOOTH_MAX_ROWS='
                    f'{GP_SMOOTH_MAX_ROWS} (n07:328)')
                return None

            def fit():
                from kirby.utils.noise_estimation import GaussianProcessEstimator
                est = GaussianProcessEstimator(n_folds=INNER_FOLDS, groups=self.g)
                est.estimate(self.X, y)
                return np.asarray(est.predictions_, dtype=float)
            return self._cached_or_not(first, 'f36_gp_oof', {}, fit)
        if name.startswith('qrf_q'):
            def fit():
                from kirby.utils.noise_estimation import QuantileForestEstimator
                est = QuantileForestEstimator(quantile_low=0.25,
                                              quantile_high=0.75,
                                              n_folds=INNER_FOLDS, groups=self.g)
                est.estimate(self.X, y)
                return np.stack([est.quantile_lo_, est.predictions_,
                                 est.quantile_hi_]).astype(float)
            q = self._cached_or_not(first, 'f36_qrf_25_50_75', {}, fit)
            return q[{'qrf_q25': 0, 'qrf_q50': 1, 'qrf_q75': 2}[name]]
        if name == 'pair_mean':
            i, j, _thr = self.pairs()
            return pair_mean_values(y, i, j)
        if name.endswith('_oof') and name[:-4] in ('hetero_gp', 'ngboost',
                                                    'heteroscedastic_nn'):
            from experiments.noise import n23_uncertainty_cleaning as n23
            src = name[:-4]
            if first:
                sc, tgt, tsrc = n23._uncertainty_cached(
                    self.ds, src, self.X, y, self.g, False, INNER_FOLDS)
            else:
                sc, tgt, tsrc = n23._uncertainty(src, self.X, y, self.g, False,
                                                 INNER_FOLDS)
            if sc is None or tsrc != n23.OWN_TARGET:
                self.unavailable[name] = (f'{src}: no out-of-fold prediction '
                                          f'of its own ({tsrc})')
                return None
            return np.asarray(tgt, dtype=float)
        raise ValueError(f'f36: unknown value {name!r}')

    def uncertainty(self):
        """hetero_gp's out-of-fold latent variance on D: how unsure the mean
        model is about each molecule. The gate's input."""
        def fit():
            from kirby.utils.noise_estimation import HeteroscedasticGPEstimator
            est = HeteroscedasticGPEstimator(n_folds=INNER_FOLDS, groups=self.g)
            est.estimate(self.X, self.y0)
            return np.asarray(est.latent_variance_, dtype=float)
        try:
            return self._cached_or_not(True, 'f36_hetero_gp_latent_var', {}, fit)
        except Exception as exc:
            self.errors.add('uncertainty_gate/hetero_gp', exc)
            return None


# ---------------------------------------------------------------------------
# One unit: (condition, level, draw, fold, model, model input)
# ---------------------------------------------------------------------------

class _Errors:
    def __init__(self):
        self.d = {}

    def add(self, where, exc):
        msg = exc if isinstance(exc, str) else f'{type(exc).__name__}: {exc}'
        e = self.d.setdefault(where, {'count': 0, 'first': msg})
        e['count'] += 1


def unit_rows_path(output_dir, ds, cond, unit):
    fold, level, draw, m, mi = unit
    return os.path.join(str(output_dir), EXP_ID, ds, cond,
                        f'f{fold}_l{level:g}_d{draw}_{m}_{mi}.rows.jsonl')


def run_unit(*, ds, cond, unit, passes, X_tr, X_va, y0, y_clean, y_val, g_tr,
             metric, task_type, factory, output_dir, errors, unavailable,
             flush_every=200, flush_seconds=60.0):
    """Every combination of one passes schedule on one unit. Rows already on
    disk are skipped. Returns the number of rows written.

    Rows go to disk every flush_every combinations or every flush_seconds,
    whichever comes first, so a job killed at its wall clock loses at most a
    minute of fits. One combination under passes fixed or anneal refits its
    value up to 4 times, so a count alone could hold hours of work in memory.
    """
    from experiments.noise import noise_cache as nc
    from experiments.noise.d_vs_dprime import (
        make_row, row_key, append_rows, done_keys)
    from experiments.noise.n07_remediation import DETECTOR
    fold, level, draw, m, mi = unit
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lib = metric
    mode = nc.predict_mode_for(metric_name)
    path = unit_rows_path(output_dir, ds, cond, unit)
    done = done_keys(path)

    combos, _skipped = combinations(is_clf)
    combos = [c for c in combos if c[-1] == passes]

    def ident(setting, sim):
        return {'dataset': ds, 'experiment': EXP_ID, 'method': 'f36/smooth',
                'method_setting': setting, 'similarity_setting': sim,
                'model_input': f'{m}:{mi}', 'noise_condition': cond,
                'level': float(level), 'draw': int(draw), 'fold': int(fold)}

    def sim_of(v):
        if v.startswith('knn9'):
            return 'model_input|euclidean|k=9'
        if v.startswith('forest10'):
            return 'forest_proximity|k=10'
        if v == 'pair_mean':
            return f'model_input|euclidean|pair_pct={PAIR_PERCENTILE:g}'
        return 'none'

    todo = [c for c in combos
            if row_key(dict(ident(setting_label(*c), sim_of(c[0])), model=m))
            not in done]
    if not todo:
        return 0

    def fit_score(y_fit):
        mdl = factory()
        mdl.fit(X_tr, y_fit)
        return float(metric_fn(y_val, nc._predict(mdl, X_va, mode)))

    def cached_score(y_fit):
        pred = nc.cached_fit_predict(ds, m, factory, X_tr, y_fit, X_va,
                                     predict_mode=mode)
        return float(metric_fn(y_val, pred))

    s_D = cached_score(y0)
    s_C = cached_score(y_clean)

    try:
        score = nc.cached_detector_scores(ds, X_tr, y0, is_clf=is_clf,
                                          detector=DETECTOR, groups=g_tr)
        score = np.asarray(score, dtype=float)
    except Exception as exc:
        errors.add(f'detector={DETECTOR}', exc)
        score = None

    vals = Values(ds, X_tr, y0, g_tr, errors)
    ctx = {}
    p0 = None
    if not is_clf:
        p0 = vals.at('lgbm_oof', y0)
        if any(c[1] == 'uncertainty_gate' for c in todo):
            ctx['unc'] = vals.uncertainty()
        if any(c[1] == 'agreement_gate' for c in todo):
            pk = vals.at('knn9_mean', y0)
            if p0 is not None and pk is not None:
                ctx['disagree'] = np.abs(np.asarray(p0) - np.asarray(pk))
                ctx['disagree_s'] = robust_s(y0 - p0)

    buf = []
    written = 0
    last_flush = time.time()

    def flush():
        nonlocal buf, written, last_flush
        if buf:
            append_rows(path, buf)
            written += len(buf)
            buf = []
        last_flush = time.time()

    for c in todo:
        v, r, pr, sc, p = c
        setting = setting_label(*c)
        y_new, why = None, None
        if score is None:
            why = 'no suspicion score'
        elif is_clf:
            nbrs = (vals.nbrs() if v.startswith('knn9')
                    else vals.forest(is_clf=True, y=y0))
            y_new, _ran = smooth_clf(y0, nbrs, float(r[len('flagtop'):]), p,
                                     score)
        else:
            v0 = vals.at(v, y0)
            if v0 is None:
                why = vals.unavailable.get(v, f'v={v} could not be computed')
            elif sc == 'residual' and p0 is None:
                why = 'residual scale needs the LightGBM prediction'
            else:
                y_new, why = smooth(
                    y0, v0, r, pr, sc, p, score=score, ctx=ctx, groups=g_tr,
                    p0=p0, refit_v=functools.partial(vals.at, v))
        if y_new is None:
            unavailable.setdefault(f'{v}|{r}|{pr}|{sc}|{p}', why)
            s_P = float('nan')
        else:
            try:
                s_P = fit_score(y_new)
            except Exception as exc:
                errors.add(f'fit|{m}', exc)
                s_P = float('nan')
        buf.append(make_row(**ident(setting, sim_of(v)), model=m,
                            metric=metric_name, lower_is_better=lib,
                            score_D=s_D, score_Dprime=s_P, score_clean=s_C))
        if (len(buf) >= flush_every
                or time.time() - last_flush >= flush_seconds):
            flush()
    flush()
    unavailable.update({f'v={k}': w for k, w in vals.unavailable.items()})
    return written


# ---------------------------------------------------------------------------
# Entry point through noise_helpers.run_all
# ---------------------------------------------------------------------------

def _recipes_for(by_model, models):
    """n42._recipes_for with POOL_SUBSET and SPACE_FOR_FAMILY above."""
    import copy
    from experiments._shared import pool_recipes_by_family, POOL_SUBSETS
    recipes, unavail = {}, {}
    for m in models:
        meta = (by_model.get(m) or {}).get('meta')
        try:
            rs, un = pool_recipes_by_family(
                meta, SPACE_FOR_FAMILY,
                eligible_families=POOL_SUBSETS[POOL_SUBSET])
        except Exception as exc:
            recipes[m] = {}
            unavail[m] = {'_all': f'{type(exc).__name__}: {exc}'}
            continue
        per = {}
        for r in rs:
            bp = r['blueprint']
            aug = r['aug_condition'] or 'uncapped'
            if getattr(bp, 'augmentation_strategy', None) != aug:
                try:
                    bp = copy.copy(bp)
                    bp.augmentation_strategy = aug
                except Exception:
                    pass
            per[r['space']] = (bp, aug)
        recipes[m] = per
        if un:
            unavail[m] = dict(un)
    return recipes, unavail


def _slice():
    raw = os.environ.get('KIRBY_F36_SLICE', '0/1')
    k, n = (int(x) for x in raw.split('/'))
    if not 0 <= k < n:
        raise ValueError(f'KIRBY_F36_SLICE={raw!r}: need 0 <= K < N')
    return k, n


def _passes_asked():
    raw = os.environ.get('KIRBY_F36_PASSES')
    if not raw:
        return PASSES
    got = tuple(t.strip() for t in raw.split(',') if t.strip())
    bad = [g for g in got if g not in PASSES]
    if bad:
        raise ValueError(f'KIRBY_F36_PASSES: unknown {bad}; known {PASSES}')
    return tuple(p for p in PASSES if p in got)


def _conditions_asked(is_clf):
    default = CLASSIFICATION_CONDITIONS if is_clf else REGRESSION_CONDITIONS
    raw = os.environ.get('KIRBY_F36_CONDITIONS')
    if not raw:
        return default
    got = tuple(t.strip() for t in raw.split(',') if t.strip())
    bad = [g for g in got if g not in default]
    if bad:
        raise ValueError(f'KIRBY_F36_CONDITIONS: {bad} is not one of {default}')
    return got


def run(data, config, output_dir, dataset_name, **kwargs):
    """Every condition, every passes schedule, this process's slice of the
    units. Called by main() on data from noise_helpers.load_noise_data."""
    from experiments._shared import (
        cv_split, get_domain_metric, get_model_factories,
        get_best_search_for_downstream, noise_base_reps_by_model,
        convert_numpy)
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        noise_spec_for, inject_train_labels, pool_recipe_per_downstream_model)
    from experiments.noise import n07_remediation as n07
    from experiments.noise import n23_uncertainty_cleaning as n23
    from experiments.noise import similarity_space as ss

    t_start = time.time()
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    n_folds = int(config['n_splits'])
    metric = get_domain_metric(data)
    models = list(MODELS)
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    k_slice, n_slice = _slice()
    passes_list = _passes_asked()
    conditions = _conditions_asked(is_clf)

    lookup = functools.partial(
        get_best_search_for_downstream, reps, labels, groups,
        budget=kwargs.get('budget'), augmentations=augmentations,
        dataset_name=dataset_name, task_type=task_type, data=data,
        pool_subset=POOL_SUBSET)
    by_model, no_recipe = pool_recipe_per_downstream_model(lookup, models)
    recipes, recipes_unavailable = _recipes_for(by_model, models)
    base_by_model, base_missing = noise_base_reps_by_model(
        {m: by_model[m]['search'] for m in by_model}, models, reps=reps,
        dataset_name=dataset_name, verbose=False)
    for m in models:
        print(f"  {m}: base {base_by_model.get(m)}, recipe spaces "
              f"{sorted(recipes.get(m, {}))}")

    noise_groups = noise_groups_for(data)[0]
    sample_X = next(iter(reps.values()))
    folds = [(np.asarray(a), np.asarray(b)) for a, b in
             cv_split(groups, n_folds, sample_X, labels)]
    mi_settings = [ss.SimilaritySetting(sp, 'none', 'manhattan')
                   for sp in MODEL_INPUTS]

    combos, skipped = combinations(is_clf)
    errors = _Errors()
    unavailable = {}
    written_total = 0
    units_done = units_resumed = 0
    index = {}

    for passes in passes_list:
        for cond in conditions:
            ckpt = n07.FoldCheckpoint(
                output_dir, EXP_ID, dataset_name, cond,
                tag=f'units_{passes}_slice{k_slice}of{n_slice}',
                fingerprint={'combos': len([c for c in combos
                                            if c[-1] == passes]),
                             'models': models})
            level_draws = n23._levels_and_draws(dataset_name, cond, is_clf,
                                                draws=NOISE_DRAWS)
            units = [(f, float(lv), int(dr), m, mi)
                     for f in range(len(folds)) for lv, dr in level_draws
                     for m in models for mi in MODEL_INPUTS]
            mine = units[k_slice::n_slice]
            print(f"\n  f36 {dataset_name} {cond} passes={passes}: "
                  f"{len(units)} units, this slice {len(mine)}")
            built = {}
            for unit in mine:
                fold, level, draw, m, mi = unit
                if ckpt.get(list(unit)) is not None:
                    units_resumed += 1
                    continue
                t0 = time.time()
                tr, va = folds[fold]
                y_clean = labels[tr]
                y_val = labels[va]
                g_tr = None if groups is None else np.asarray(groups)[tr]
                key = (fold, level, draw, m)
                if key not in built:
                    built.clear()
                    try:
                        spec = noise_spec_for(dataset_name, cond, level,
                                              noise_seed=draw, is_clf=is_clf)
                        y0, _mask, _prov = inject_train_labels(
                            labels, spec, train_idx=tr, is_clf=is_clf,
                            groups=noise_groups, dataset_name=dataset_name)
                        y0 = np.asarray(y0)
                        sp_tr, sp_va, sp_un = n07.fold_similarity_spaces(
                            dataset_name, mi_settings, model_input_tr=None,
                            model_input_va=None, reps=reps, train_idx=tr,
                            val_idx=va, y_train=y0, groups_train=g_tr,
                            base_reps=base_by_model.get(m, []),
                            recipes=recipes.get(m, {}),
                            augmentations=augmentations, model_name=m,
                            model_factory=factories.get(m), metric=metric,
                            task_type=task_type)
                        built[key] = (y0, sp_tr, sp_va, sp_un)
                    except Exception as exc:
                        errors.add(f'build|{cond}|{m}', exc)
                        built[key] = None
                if built[key] is None:
                    continue
                y0, sp_tr, sp_va, sp_un = built[key]
                X_tr = sp_tr.get(mi)
                if X_tr is None:
                    unavailable.setdefault(
                        f'{m}:{mi}', sp_un.get(mi)
                        or (base_missing.get(m) if mi.startswith('base')
                            else (recipes_unavailable.get(m) or {}).get(mi)
                            or (no_recipe.get(m) or {}).get('why'))
                        or 'not built')
                    continue
                try:
                    n = run_unit(
                        ds=dataset_name, cond=cond, unit=unit, passes=passes,
                        X_tr=X_tr, X_va=sp_va[mi], y0=y0, y_clean=y_clean,
                        y_val=y_val, g_tr=g_tr, metric=metric,
                        task_type=task_type, factory=factories[m],
                        output_dir=output_dir, errors=errors,
                        unavailable=unavailable)
                except Exception as exc:
                    errors.add(f'unit|{cond}|{m}:{mi}', exc)
                    continue
                written_total += n
                ckpt.put(list(unit), {'rows_written': int(n),
                                      'seconds': round(time.time() - t0, 1)})
                units_done += 1
                print(f"    {cond} {passes} fold {fold} level {level:g} draw "
                      f"{draw} {m}:{mi}: {n} rows, {time.time() - t0:.0f}s",
                      flush=True)
            index[f'{cond}|{passes}'] = {'units': len(units),
                                         'this_slice': len(mine)}

    summary = {
        'metric': metric[0], 'lower_is_better': bool(metric[2]),
        'task_type': task_type, 'n_folds': n_folds, 'models': models,
        'model_inputs': list(MODEL_INPUTS), 'conditions': list(conditions),
        'passes': list(passes_list), 'slice': f'{k_slice}/{n_slice}',
        'n_combinations': len(combos),
        'n_combinations_not_run_identical': len(skipped),
        'why_not_run': sorted(set(skipped.values())),
        'values_not_run': VALUES_NOT_RUN,
        'base_reps_by_model': base_by_model, 'base_reps_missing': base_missing,
        'recipe_spaces_by_model': {m: sorted(v) for m, v in recipes.items()},
        'units': index, 'units_computed_this_run': units_done,
        'units_resumed': units_resumed, 'rows_written': written_total,
        'errors': errors.d, 'unavailable': unavailable,
        'seconds': round(time.time() - t_start, 1),
    }
    out = os.path.join(str(output_dir), EXP_ID,
                       f'{dataset_name}__slice{k_slice}of{n_slice}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(convert_numpy(summary), fh, indent=1, default=str)
    print(f"\n  f36 {dataset_name}: {units_done} units computed, "
          f"{units_resumed} resumed, {written_total} rows, "
          f"{len(errors.d)} distinct errors -> {out}")
    return summary


def status(output_dir, datasets=('logd', 'flab_expression_koenig', 'herg')):
    """Units finished against units asked, per dataset, condition and passes
    schedule, read from the unit checkpoint files only. Opens no data."""
    import glob
    from experiments.dataset_policy import DATASET_REGISTRY_DATA
    from experiments.noise import n23_uncertainty_cleaning as n23
    root = os.path.join(str(output_dir), EXP_ID)
    print(f'{"dataset":<24} {"condition":<18} {"passes":<7} done / asked')
    for ds in datasets:
        is_clf = DATASET_REGISTRY_DATA[ds].task_type == 'classification'
        conds = CLASSIFICATION_CONDITIONS if is_clf else REGRESSION_CONDITIONS
        for p in (CLF_PASSES if is_clf else PASSES):
            for c in conds:
                n_ld = len(n23._levels_and_draws(ds, c, is_clf,
                                                 draws=NOISE_DRAWS))
                asked = n_ld * 5 * len(MODELS) * len(MODEL_INPUTS)
                done = set()
                for f in glob.glob(os.path.join(
                        root, f'{ds}__{c}.units_{p}_slice*.jsonl')):
                    with open(f) as fh:
                        for line in fh:
                            try:
                                done.add(json.dumps(json.loads(line)['ckpt_key']))
                            except (ValueError, KeyError):
                                continue
                print(f'{ds:<24} {c:<18} {p:<7} {len(done):>5} / {asked}')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='f36 on one dataset')
    ap.add_argument('dataset', nargs='?')
    ap.add_argument('--status', action='store_true',
                    help='print units finished per dataset, condition and '
                         'passes, then exit')
    ap.add_argument('--output', required=True)
    ap.add_argument('--slice', default=None,
                    help='K/N: this process takes every Nth unit from K')
    ap.add_argument('--passes', default=None,
                    help='comma list of once,fixed,anneal (default all)')
    ap.add_argument('--conditions', default=None)
    args = ap.parse_args()
    if args.status:
        status(args.output)
        return
    if not args.dataset:
        ap.error('a dataset is needed unless --status')
    # NEVER A SEARCH. 1.0: a stored search counts only when every one of its
    # configurations is stored; COLD_SEARCH_MAX_CONFIGS 0: the 'small search
    # runs here' exception (find_best_hybrid._refuse_cold_search) is off. A
    # search that is not complete is skipped with a '[pool] WARNING' line in
    # the log, never filled in.
    os.environ['KIRBY_NO_COLD_SEARCH'] = '1.0'
    os.environ['KIRBY_COLD_SEARCH_MAX_CONFIGS'] = '0'
    if args.slice:
        os.environ['KIRBY_F36_SLICE'] = args.slice
    if args.passes:
        os.environ['KIRBY_F36_PASSES'] = args.passes
    if args.conditions:
        os.environ['KIRBY_F36_CONDITIONS'] = args.conditions
    # The data as every noise experiment loads it, WITHOUT noise_helpers.run_all:
    # run_all precomputes every search family before any experiment runs,
    # and that is where the first f36 jobs spent their 2 hours.
    from experiments.noise.noise_helpers import load_noise_data
    from kirby.find_best_hybrid import DEFAULT_N_FOLDS
    data, _reps = load_noise_data(args.dataset)
    run(data, {'n_splits': DEFAULT_N_FOLDS}, args.output, args.dataset)


if __name__ == '__main__':
    main()
