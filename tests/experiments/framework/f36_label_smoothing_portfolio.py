#!/usr/bin/env python3
"""f36 -- the label-smoothing settings every noise experiment uses.

A framework experiment: it decides settings, it is not published. Every noise
experiment smooths labels through n07.cleaned_training_set, whose settings
were fixed by hand. f36 measures them so they are set by measurement: if one
setting is within tolerance of the best everywhere it becomes the default; if
not, the smallest set that is.

Redesigned 2026-10-05 to fit 2 rounds of 5 interactive sruns x 4 hours, from
timings measured on the server that day (F36.md section 8). The earlier f36
(every combination, two stages) is replaced.

EVERY LABEL SMOOTHING IS  new y = y + a x (v - y).  The settings decided:

  v      the value pulled toward. Mean label of the k nearest molecules
         (k = 5, 9, 10, 20; Euclidean or Manhattan in the model input);
         mean label of the forest's k nearest (k as above); near-duplicate
         pair mean (n35 pairs); out-of-fold LightGBM (n07 oof_smooth);
         Gaussian process (n07 gp_smooth, refused above 1,000 molecules);
         quantile forest 0.25, 0.5, 0.75 quantile.
  a      how far, 30 rules (RULES): strength x suspicion score, the same
         share for every label, the flagged share replaced, js_shrink's
         variance ratio, winsorise, smooth_by_scores' other three curves,
         n31 and n32's shrinks, the redescending curve, the agreement gate.
  detector   which detector gives the suspicion score a rule reads:
         cross_val (n07's), neighbor_residual, activity_cliff_aware.
  preserve, scale   tested on one rule (strength 0.5 x score, cross_val) for
         every v: preserve none / variance / group mean / rank; scale raw /
         residual / rank (residual with none or group mean is the same labels
         as raw, by algebra, and is not run).
  passes  repeat-and-recalculate (fixed point, annealed) for the neighbour
         averages only, whose recalculation is re-averaging (owner,
         2026-10-04). Every rule, cross_val.
  also   quantile_clip at share 0.1, 0.3, 0.5, 0.7, 1.0 (n07, n11); n30's three
         lifts under censoring.

On herg (classification): v = the k nearest's or the forest's majority class,
the flagged share replaced, every detector, passes once or fixed.

NOT RUN, owner's decisions 2026-10-05 against the time limit: the three
uncertainty models' values and the uncertainty gate (about 26 session-hours),
LightGBM with Huber loss (about 7). n19's shrinks need repeat measurements and
n34's group centring needs the grouped noise conditions; neither is here.

TASKS. Model: SVM. Model input: the SVM's best single representation and its
best prediction-built hybrid, read from the stored precompute (herg: the base
representation only). Noise: none, gaussian 0.5, outlier_p10 0.5, censoring
0.25 (herg: none, uniform 0.1). Folds 0 and 1 of the 5-fold split, draw 0.
No search ever starts.

The regression SVM uses noise_cache's kernel-once fit. herg's SVM is scored on
its decision function (probability=False): AUC needs only the order of the
molecules; it is about 5 times faster and drops the unseeded probability step
(owner, 2026-10-05). D, D' and the clean fit go the same way.

OUTPUT <output>/f36_label_smoothing_settings/<dataset>/<condition>/<unit>.rows.jsonl,
d_vs_dprime rows; method_setting names every setting. f36_analyse.py reads them.

    python tests/experiments/framework/f36_label_smoothing_portfolio.py logd --slice 0/2 --output tests/results/f36
    python tests/experiments/framework/f36_label_smoothing_portfolio.py --status --output tests/results/f36
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

EXP_ID = 'f36_label_smoothing_settings'

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
MODEL = 'svm'
POOL_SUBSET = 'predictive_only'
SPACE_FOR_FAMILY = {'predictive': 'hybrid'}
REG_INPUTS = ('base_1', 'hybrid')
CLF_INPUTS = ('base_1',)
REG_CONDITIONS = (('none', 0.0), ('gaussian', 0.5), ('outlier_p10', 0.5),
                  ('censoring', 0.25))
CLF_CONDITIONS = (('none', 0.0), ('uniform', 0.1))
FOLDS = (0, 1)
N_SPLITS = 5
DRAW = 0

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
KS = (5, 9, 10, 20)
METRICS = ('euclidean', 'manhattan')
KNN_VALUES = tuple(f'knn{k}_{m}' for m in METRICS for k in KS)
FOREST_VALUES = tuple(f'forest{k}' for k in KS)
NEIGHBOUR_VALUES = KNN_VALUES + FOREST_VALUES + ('pair_mean',)
MODEL_VALUES = ('lgbm_oof', 'gp_oof', 'qrf_q25', 'qrf_q50', 'qrf_q75')
VALUES = NEIGHBOUR_VALUES + MODEL_VALUES
CLF_VALUES = tuple(f'knn{k}_{m}_majority' for m in METRICS for k in KS) + \
    tuple(f'forest{k}_majority' for k in KS)
DETECTORS = ('cross_val', 'neighbor_residual', 'activity_cliff_aware')

PROP = (0.25, 0.5, 1.0)
CONST = (0.25, 0.5, 1.0)
FLAG = (0.05, 0.10, 0.20)
JS = (0.5, 1.0, 2.0)
WINSOR = (1.0, 1.25, 1.5)
SBS_RULES = ('threshold', 'soft_threshold', 'confidence')
SBS_ALPHA = 0.5
REDESC_ZERO = (3.0, 4.685)
GATE_SHARE = 0.5
UNC_FLOOR_PCT = 10.0

SCORE_RULES = (tuple(f'prop{a:g}' for a in PROP)
               + tuple(f'flagtop{f:g}' for f in FLAG)
               + tuple(f'js{s:g}' for s in JS)
               + tuple(f'{r}{SBS_ALPHA:g}' for r in SBS_RULES)
               + tuple(f'n31_bayes_shrink_flag{f:g}' for f in FLAG))
FREE_RULES = (tuple(f'const{a:g}' for a in CONST)
              + tuple(f'winsor_c{c:g}' for c in WINSOR)
              + tuple(f'n32_one_weight_flag{f:g}' for f in FLAG)
              + tuple(f'n32_per_row_flag{f:g}' for f in FLAG)
              + tuple(f'redesc_zero{c:g}' for c in REDESC_ZERO)
              + ('agreement_gate',))
RULES = SCORE_RULES + FREE_RULES

PRESERVES = ('none', 'variance', 'group_mean', 'rank')
SCALES = ('raw', 'residual', 'rank')
DEGENERATE = {('residual', 'none'), ('residual', 'group_mean')}
REFERENCE_RULE = 'prop0.5'
REFERENCE_DETECTOR = 'cross_val'
FIXED_MAX_PASSES = 5
FIXED_TOL_SD = 1e-3
ANNEAL_PASSES = 3
ANNEAL_DECAY = 0.5
QCLIP = (0.1, 0.3, 0.5, 0.7, 1.0)
LIFTS = ('constant_lift', 'per_molecule_lift', 'truncated_normal_mean')
INNER_FOLDS = 5
PAIR_PERCENTILE = 1.0
GP_MAX_ROWS = 1000


def label(kind, v, rule, det='none', pr='none', sc='raw', p='once'):
    return f'{kind}|v={v}|a={rule}|detector={det}|preserve={pr}|scale={sc}|passes={p}'


def settings(is_clf, cond):
    """[(label, spec)] for one task type and noise condition. spec is a dict
    the unit code reads."""
    out = []

    def add(kind, v, rule, det='none', pr='none', sc='raw', p='once'):
        out.append((label(kind, v, rule, det, pr, sc, p),
                    dict(kind=kind, v=v, rule=rule, det=det, pr=pr, sc=sc,
                         p=p)))
    if is_clf:
        for p in ('once', 'fixed'):
            for v in CLF_VALUES:
                for f in FLAG:
                    for d in DETECTORS:
                        add('smooth', v, f'flagtop{f:g}', d, p=p)
        return out
    for v in VALUES:
        for r in RULES:
            for d in (DETECTORS if r in SCORE_RULES else ('none',)):
                add('smooth', v, r, d)
        for pr in PRESERVES:
            for sc in SCALES:
                if (pr, sc) == ('none', 'raw') or (sc, pr) in DEGENERATE:
                    continue
                add('smooth', v, REFERENCE_RULE, REFERENCE_DETECTOR, pr, sc)
    for v in NEIGHBOUR_VALUES:
        for r in RULES:
            d = REFERENCE_DETECTOR if r in SCORE_RULES else 'none'
            for p in ('fixed', 'anneal'):
                add('smooth', v, r, d, p=p)
    for a in QCLIP:
        add('quantile_clip', 'qrf_q10_q90', f'share{a:g}')
    if cond == 'censoring':
        for name in LIFTS:
            add('lift', 'lgbm_oof', name)
    return out


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

class _Errors:
    def __init__(self):
        self.d = {}

    def add(self, where, exc):
        msg = exc if isinstance(exc, str) else f'{type(exc).__name__}: {exc}'
        e = self.d.setdefault(where, {'count': 0, 'first': msg})
        e['count'] += 1


def qrf_five(X, y, groups):
    """The quantile forest's out-of-fold 0.1, 0.25, 0.5, 0.75, 0.9 quantiles,
    from ONE fit per inner fold: QuantileForestEstimator's forest, inner split
    and seed, asked for five quantiles instead of three, so quantile_clip's
    edges (0.1, 0.9) and the three values come from one forest. Rows: q10,
    q25, q50, q75, q90."""
    from quantile_forest import RandomForestQuantileRegressor
    from kirby.utils.noise_estimation import QuantileForestEstimator, _rf_jobs
    est = QuantileForestEstimator(n_folds=INNER_FOLDS, groups=groups)
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]

    def fold_fn(X_tr, y_tr, X_ev):
        qrf = RandomForestQuantileRegressor(
            n_estimators=est.n_estimators, random_state=est.random_state,
            n_jobs=_rf_jobs())
        qrf.fit(X_tr, y_tr)
        pred = np.atleast_2d(qrf.predict(X_ev, quantiles=qs))
        return {f'q{i}': pred[:, i] for i in range(len(qs))}
    oof = est._oof_apply(X, np.asarray(y, dtype=float), fold_fn,
                         n_folds=INNER_FOLDS, groups=groups)
    return np.stack([oof[f'q{i}'] for i in range(len(qs))]).astype(float)


class Values:
    """Every v on labels y for one unit. Neighbour lists, forest neighbours
    and close pairs are found once on D and kept across passes; on D's own
    labels the model-fitted values go through the shared noise cache."""

    def __init__(self, ds, X, y0, groups, is_clf, errors):
        self.ds, self.X, self.g, self.is_clf = ds, X, groups, is_clf
        self.y0 = np.asarray(y0)
        self.errors = errors
        self._nb, self._forest, self._pairs, self._q = {}, None, None, None
        self._d0 = {}
        self.unavailable = {}

    def nbrs(self, metric):
        if metric not in self._nb:
            from sklearn.neighbors import NearestNeighbors
            k = min(max(KS) + 1, len(self.y0))
            nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(self.X)
            self._nb[metric] = nn.kneighbors(self.X)[1][:, 1:]
        return self._nb[metric]

    def forest(self):
        if self._forest is None:
            from experiments.noise import noise_cache as nc
            from kirby.utils.noise_estimation import forest_proximity_distance
            leaves = nc.cached_forest_leaf_ids(self.ds, self.X, self.y0,
                                               self.is_clf)[0]
            D = forest_proximity_distance(leaves)
            np.fill_diagonal(D, np.inf)
            self._forest = np.argsort(D, axis=1)[:, :max(KS)]
            del D
        return self._forest

    def pairs(self):
        if self._pairs is None:
            from scipy.spatial.distance import cdist
            from experiments.noise.n35_near_duplicate_disagreement import (
                close_pairs)
            D = cdist(self.X, self.X).astype(np.float32)
            i, j, _d, _thr, _tot = close_pairs(D, PAIR_PERCENTILE)
            del D
            self._pairs = (i, j)
        return self._pairs

    def quantiles(self):
        if self._q is None:
            from experiments.noise import noise_cache as nc
            key = {'X': nc.array_hash(self.X), 'y': nc.array_hash(self.y0),
                   'groups': nc.array_hash(self.g)}
            self._q = nc.cached(self.ds, 'f36_qrf_five', key,
                                lambda: qrf_five(self.X, self.y0, self.g))
        return self._q

    def neighbour_list(self, v):
        """(neighbour indices) for a knn/forest value name."""
        base = v.replace('_majority', '')
        if base.startswith('knn'):
            k, metric = base[3:].split('_')
            return self.nbrs(metric)[:, :int(k)]
        return self.forest()[:, :int(base[len('forest'):])]

    def at(self, v, y):
        y = np.asarray(y)
        first = np.array_equal(y, self.y0)
        if first and v in self._d0:
            return self._d0[v]
        try:
            out = self._compute(v, y, first)
        except Exception as exc:
            self.errors.add(f'v={v}', exc)
            out = None
        if first:
            self._d0[v] = out
        return out

    def _compute(self, v, y, first):
        from experiments.noise import noise_cache as nc
        if v.endswith('_majority'):
            return majority(y, self.neighbour_list(v))
        y = np.asarray(y, dtype=float)
        if v.startswith('knn') or v.startswith('forest'):
            return np.mean(y[self.neighbour_list(v)], axis=1)
        if v == 'pair_mean':
            i, j = self.pairs()
            return pair_mean_values(y, i, j)
        if v == 'lgbm_oof':
            k = min(INNER_FOLDS, len(y))
            if first:
                return nc.cached_oof_predict(self.ds, self.X, y, self.g,
                                             n_splits=k)
            from kirby.utils.noisy_labels import oof_predict
            return np.asarray(oof_predict(self.X, y, self.g, n_splits=k))
        if v == 'gp_oof':
            if len(y) > GP_MAX_ROWS:
                self.unavailable['gp_oof'] = (
                    f'{len(y)} training molecules is over {GP_MAX_ROWS} '
                    f'(n07 GP_SMOOTH_MAX_ROWS)')
                return None
            from kirby.utils.noise_estimation import GaussianProcessEstimator

            def fit():
                est = GaussianProcessEstimator(n_folds=INNER_FOLDS,
                                               groups=self.g)
                est.estimate(self.X, y)
                return np.asarray(est.predictions_, dtype=float)
            if not first:
                return fit()
            key = {'X': nc.array_hash(self.X), 'y': nc.array_hash(y),
                   'groups': nc.array_hash(self.g)}
            return nc.cached(self.ds, 'f36_gp_oof', key, fit)
        if v.startswith('qrf_q'):
            if not first:
                raise ValueError('quantile values are not refitted')
            row = {'qrf_q25': 1, 'qrf_q50': 2, 'qrf_q75': 3}[v]
            return self.quantiles()[row]
        raise ValueError(f'f36: unknown value {v!r}')


# ---------------------------------------------------------------------------
# One unit: (fold, condition, level, model input)
# ---------------------------------------------------------------------------

def unit_rows_path(output_dir, ds, unit):
    fold, cond, level, mi = unit
    return os.path.join(str(output_dir), EXP_ID, ds, cond,
                        f'f{fold}_l{level:g}_{MODEL}_{mi}.rows.jsonl')


def run_unit(*, ds, unit, X_tr, X_va, y0, y_clean, y_val, g_tr, metric,
             task_type, factory, output_dir, errors, unavailable, beat=None,
             flush_seconds=60.0, flush_every=200):
    """Every setting of one unit. Rows already on disk are skipped; rows go to
    disk every flush_seconds or flush_every rows. Returns rows written."""
    from experiments.noise import noise_cache as nc
    from experiments.noise.d_vs_dprime import (
        make_row, row_key, append_rows, done_keys)
    from kirby.utils.noise_estimation import clip_to_range
    fold, cond, level, mi = unit
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lib = metric
    path = unit_rows_path(output_dir, ds, unit)
    done = done_keys(path)

    def ident(setting, sim):
        return {'dataset': ds, 'experiment': EXP_ID, 'method': 'f36/smooth',
                'method_setting': setting, 'similarity_setting': sim,
                'model_input': f'{MODEL}:{mi}', 'noise_condition': cond,
                'level': float(level), 'draw': DRAW, 'fold': int(fold)}

    def sim_of(v):
        if v.startswith('knn'):
            k, m = v.replace('_majority', '')[3:].split('_')
            return f'model_input|{m}|k={k}'
        if v.startswith('forest'):
            return f'forest_proximity|k={v.replace("_majority", "")[6:]}'
        if v == 'pair_mean':
            return f'model_input|euclidean|pair_pct={PAIR_PERCENTILE:g}'
        return 'none'

    todo = [(lab, sp) for lab, sp in settings(is_clf, cond)
            if row_key(dict(ident(lab, sim_of(sp['v'])), model=MODEL))
            not in done]
    if not todo:
        return 0

    # The fits. Regression: noise_cache's kernel-once SVM. Classification:
    # the SVM's decision function, probability=False (module docstring).
    mode = nc.predict_mode_for(metric_name)
    reg_kernel = (not is_clf) and nc.svm_kernel_applies(factory(), X_tr)
    seen = {}

    def predict(y_fit):
        if is_clf:
            mdl = factory()
            mdl.set_params(probability=False)
            mdl.fit(X_tr, y_fit)
            return mdl.decision_function(X_va)
        if reg_kernel:
            return nc.svm_kernel_fit_predict(factory(), X_tr, y_fit, X_va)
        mdl = factory()
        mdl.fit(X_tr, y_fit)
        return nc._predict(mdl, X_va, mode)

    def score(y_fit):
        h = hashlib.sha1(np.ascontiguousarray(np.asarray(y_fit)).tobytes()
                         ).hexdigest()
        if h not in seen:
            seen[h] = float(metric_fn(y_val, predict(y_fit)))
        return seen[h]

    if beat is not None:
        beat['now'] = f'{ds} {unit}: fitting D and the clean labels'
    s_D = score(y0)
    s_C = score(y_clean)

    scores = {}
    for d in DETECTORS:
        if beat is not None:
            beat['now'] = f'{ds} {unit}: detector {d}'
        try:
            scores[d] = np.asarray(nc.cached_detector_scores(
                ds, X_tr, y0, is_clf=is_clf, detector=d, groups=g_tr),
                dtype=float)
        except Exception as exc:
            errors.add(f'detector={d}', exc)
            scores[d] = None

    vals = Values(ds, X_tr, y0, g_tr, is_clf, errors)
    ctx, p0 = {}, None
    if not is_clf:
        if beat is not None:
            beat['now'] = f'{ds} {unit}: LightGBM out-of-fold'
        p0 = vals.at('lgbm_oof', y0)
        pk = vals.at('knn9_euclidean', y0)
        if p0 is not None and pk is not None:
            ctx['disagree'] = np.abs(np.asarray(p0) - np.asarray(pk))
            ctx['disagree_s'] = robust_s(np.asarray(y0, float) - p0)

    buf, written, last = [], 0, time.time()

    def flush():
        nonlocal buf, written, last
        if buf:
            append_rows(path, buf)
            written += len(buf)
            buf = []
        last = time.time()

    for i, (lab, sp) in enumerate(todo):
        if beat is not None:
            beat['now'] = f'{ds} {unit}: setting {i + 1} of {len(todo)}'
        y_new, why = None, None
        try:
            kind, v = sp['kind'], sp['v']
            if kind == 'quantile_clip':
                q = vals.quantiles()
                y_new = clip_to_range(np.asarray(y0, float), q[0], q[4],
                                      alpha=float(sp['rule'][5:]))
            elif kind == 'lift':
                from experiments.noise import n30_censored_likelihood as n30
                limit = n30.limit_from_labels(y0)
                if limit is None or p0 is None:
                    why = 'no limit found in the labels' if p0 is not None \
                        else 'no LightGBM prediction'
                else:
                    at = n30.rows_at_limit(y0, limit)
                    sd = n30.residual_sd(y0, p0, at)
                    cols, _info = n30.lift_columns(y0, p0, at, limit, sd, (),
                                                   (fold,))
                    y_new = cols[sp['rule']]
            else:
                sc_ = scores.get(sp['det']) if sp['det'] != 'none' else \
                    scores.get(REFERENCE_DETECTOR)
                v0 = vals.at(v, y0)
                if v0 is None:
                    why = vals.unavailable.get(v, f'v={v} not computed')
                elif sc_ is None and sp['rule'] in SCORE_RULES + tuple(
                        f'flagtop{f:g}' for f in FLAG):
                    why = f"detector {sp['det']} gave no score"
                elif is_clf:
                    y_new, _ran = smooth_clf(
                        y0, vals.neighbour_list(v),
                        float(sp['rule'][len('flagtop'):]), sp['p'], sc_)
                else:
                    y_new, why = smooth(
                        y0, v0, sp['rule'], sp['pr'], sp['sc'], sp['p'],
                        score=sc_, ctx=ctx, groups=g_tr, p0=p0,
                        refit_v=functools.partial(vals.at, v))
                    if y_new is not None:
                        why = None
        except Exception as exc:
            errors.add(f"{sp['kind']}|{sp['rule']}", exc)
            y_new, why = None, f'{type(exc).__name__}: {exc}'
        if y_new is None:
            unavailable.setdefault(lab, why)
            s_P = float('nan')
        else:
            try:
                s_P = score(y_new)
            except Exception as exc:
                errors.add('fit', exc)
                s_P = float('nan')
        buf.append(make_row(**ident(lab, sim_of(sp['v'])), model=MODEL,
                            metric=metric_name, lower_is_better=lib,
                            score_D=s_D, score_Dprime=s_P, score_clean=s_C))
        if len(buf) >= flush_every or time.time() - last >= flush_seconds:
            flush()
    flush()
    unavailable.update({f'v={k}': w for k, w in vals.unavailable.items()})
    return written


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def units_for(is_clf):
    conds = CLF_CONDITIONS if is_clf else REG_CONDITIONS
    inputs = CLF_INPUTS if is_clf else REG_INPUTS
    return [(f, c, float(lv), mi) for f in FOLDS for c, lv in conds
            for mi in inputs]


def _recipes_for(by_model, models):
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


def run(data, output_dir, dataset_name, k_slice=0, n_slice=1):
    from experiments._shared import (
        cv_split, get_domain_metric, get_model_factories,
        get_best_search_for_downstream, noise_base_reps_by_model,
        convert_numpy)
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        noise_spec_for, inject_train_labels, pool_recipe_per_downstream_model)
    from experiments.noise import n07_remediation as n07
    from experiments.noise import similarity_space as ss

    t_start = time.time()
    beat = {'now': 'reading the stored searches'}

    def heartbeat():
        while True:
            time.sleep(60)
            print(f"  [f36 {time.strftime('%H:%M:%S')}] still running: "
                  f"{beat['now']}", flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()

    task_type = data['task_type']
    is_clf = task_type == 'classification'
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    metric = get_domain_metric(data)
    factory = get_model_factories([MODEL], task_type=task_type,
                                  n_samples=len(labels))[MODEL]

    lookup = functools.partial(
        get_best_search_for_downstream, reps, labels, groups, budget=None,
        augmentations=data.get('augmentations'), dataset_name=dataset_name,
        task_type=task_type, data=data, pool_subset=POOL_SUBSET)
    by_model, no_recipe = pool_recipe_per_downstream_model(lookup, [MODEL])
    recipes, rec_un = _recipes_for(by_model, [MODEL])
    base, base_missing = noise_base_reps_by_model(
        {m: by_model[m]['search'] for m in by_model}, [MODEL], reps=reps,
        dataset_name=dataset_name, verbose=False)
    print(f"  {MODEL}: best single representation "
          f"{(base.get(MODEL) or [None])[0]}; hybrid recipe "
          f"{'found' if 'hybrid' in recipes.get(MODEL, {}) else 'MISSING'}",
          flush=True)

    folds = [(np.asarray(a), np.asarray(b)) for a, b in
             cv_split(groups, N_SPLITS, next(iter(reps.values())), labels)]
    noise_groups = noise_groups_for(data)[0]
    units = units_for(is_clf)
    mine = units[k_slice::n_slice]
    n_settings = {c: len(settings(is_clf, c)) for _f, c, _l, _m in units}
    ckpt = n07.FoldCheckpoint(output_dir, EXP_ID, dataset_name, None,
                              tag=f'units_slice{k_slice}of{n_slice}',
                              fingerprint={'settings': n_settings})
    print(f"  f36 {dataset_name}: {len(units)} units, this slice "
          f"{len(mine)}; settings per unit {n_settings}", flush=True)

    errors, unavailable = _Errors(), {}
    written = done_now = resumed = 0
    built_key, built = None, None
    for unit in mine:
        fold, cond, level, mi = unit
        if ckpt.get(list(unit)) is not None:
            resumed += 1
            continue
        t0 = time.time()
        tr, va = folds[fold]
        g_tr = None if groups is None else np.asarray(groups)[tr]
        if built_key != (fold, cond):
            built_key = (fold, cond)
            beat['now'] = f'{dataset_name} fold {fold} {cond}: building inputs'
            try:
                if cond == 'none':
                    y0 = labels[tr].copy()
                else:
                    spec = noise_spec_for(dataset_name, cond, level,
                                          noise_seed=DRAW, is_clf=is_clf)
                    y0 = np.asarray(inject_train_labels(
                        labels, spec, train_idx=tr, is_clf=is_clf,
                        groups=noise_groups, dataset_name=dataset_name)[0])
                wanted = CLF_INPUTS if is_clf else REG_INPUTS
                sp_tr, sp_va, sp_un = n07.fold_similarity_spaces(
                    dataset_name,
                    [ss.SimilaritySetting(x, 'none', 'manhattan')
                     for x in wanted],
                    model_input_tr=None, model_input_va=None, reps=reps,
                    train_idx=tr, val_idx=va, y_train=y0, groups_train=g_tr,
                    base_reps=base.get(MODEL, []),
                    recipes=recipes.get(MODEL, {}),
                    augmentations=data.get('augmentations'),
                    model_name=MODEL, model_factory=factory, metric=metric,
                    task_type=task_type)
                built = (y0, sp_tr, sp_va, sp_un)
            except Exception as exc:
                errors.add(f'build|{cond}', exc)
                built = None
        if built is None:
            continue
        y0, sp_tr, sp_va, sp_un = built
        if sp_tr.get(mi) is None:
            unavailable.setdefault(f'input {mi}', sp_un.get(mi) or (
                base_missing.get(MODEL) if mi.startswith('base') else
                (rec_un.get(MODEL) or {}).get(mi)
                or (no_recipe.get(MODEL) or {}).get('why')) or 'not built')
            continue
        try:
            n = run_unit(ds=dataset_name, unit=unit, X_tr=sp_tr[mi],
                         X_va=sp_va[mi], y0=y0, y_clean=labels[tr],
                         y_val=labels[va], g_tr=g_tr, metric=metric,
                         task_type=task_type, factory=factory,
                         output_dir=output_dir, errors=errors,
                         unavailable=unavailable, beat=beat)
        except Exception as exc:
            errors.add(f'unit|{cond}|{mi}', exc)
            continue
        written += n
        done_now += 1
        ckpt.put(list(unit), {'rows_written': int(n),
                              'seconds': round(time.time() - t0, 1)})
        print(f"  DONE fold {fold} {cond} level {level:g} {mi}: {n} rows, "
              f"{time.time() - t0:.0f}s", flush=True)

    summary = {'units': len(units), 'this_slice': len(mine),
               'units_computed_this_run': done_now,
               'units_resumed': resumed, 'rows_written': written,
               'best_single_representation': base.get(MODEL),
               'settings_per_unit': n_settings, 'errors': errors.d,
               'unavailable': unavailable,
               'seconds': round(time.time() - t_start, 1)}
    out = os.path.join(str(output_dir), EXP_ID,
                       f'{dataset_name}__slice{k_slice}of{n_slice}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(convert_numpy(summary), fh, indent=1, default=str)
    print(f"\n  f36 {dataset_name} slice {k_slice}/{n_slice} FINISHED: "
          f"{done_now} units computed, {resumed} already done, "
          f"{len(errors.d)} distinct errors -> {out}", flush=True)
    return summary


def status(output_dir, datasets=('logd', 'flab_expression_koenig', 'herg')):
    """Units finished out of units asked, per dataset. Reads the checkpoint
    files only."""
    import glob
    from experiments.dataset_policy import DATASET_REGISTRY_DATA
    root = os.path.join(str(output_dir), EXP_ID)
    for ds in datasets:
        is_clf = DATASET_REGISTRY_DATA[ds].task_type == 'classification'
        wanted = {json.dumps(list(u)) for u in units_for(is_clf)}
        done = set()
        for f in glob.glob(os.path.join(root, f'{ds}.units_slice*.jsonl')):
            with open(f) as fh:
                for line in fh:
                    try:
                        k = json.dumps(json.loads(line)['ckpt_key'])
                    except (ValueError, KeyError):
                        continue
                    if k in wanted:
                        done.add(k)
        print(f'{ds:<24} {len(done):>3} of {len(wanted)} units done')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='f36 on one dataset')
    ap.add_argument('dataset', nargs='?')
    ap.add_argument('--output', required=True)
    ap.add_argument('--slice', default='0/1',
                    help='K/N: this process takes every Nth unit from K')
    ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    if a.status:
        status(a.output)
        return
    if not a.dataset:
        ap.error('a dataset is needed unless --status')
    k, n = (int(x) for x in a.slice.split('/'))
    if not 0 <= k < n:
        ap.error('--slice K/N needs 0 <= K < N')
    # Never a search: a stored search counts only when complete, and the
    # small-search exception is off.
    os.environ['KIRBY_NO_COLD_SEARCH'] = '1.0'
    os.environ['KIRBY_COLD_SEARCH_MAX_CONFIGS'] = '0'
    from experiments.noise.noise_helpers import load_noise_data
    data, _ = load_noise_data(a.dataset)
    run(data, a.output, a.dataset, k, n)


if __name__ == '__main__':
    main()
