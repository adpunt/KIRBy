#!/usr/bin/env python3
"""N31: how good can a residual detector be on this dataset, and how good does it need to be?

Paper: Noise.

WHERE THIS COMES FROM. The second research report of 2026-09-24
(`~/Downloads/compass_artifact_wf-d7ecc1f8-…_text_markdown.md`), its section 2,
its diagnostics D2 and D4, its H1 and the last row of its section 6. That report
was written from `/Volumes/seagate/local_noise_run/loop_headroom/FINDINGS.md`,
which is herg_ki only, R^2 about 0.51. Its claim is that a residual detector gets
better as the model's own out-of-fold error falls, so every negative result from
herg_ki has to be re-measured on a dataset where R^2 is high (logd 0.857, qm9
0.912 in `tests/results/phase5_baselines/`).

WHAT IS MEASURED, per (condition, level, draw, fold, feature space):

  s_m                  the model's error in label-SD units: the rmse of a
                       group-aware out-of-fold LightGBM prediction on the CLEAN
                       training labels, divided by the SD of those labels.
  realised noise       the injected error on the molecules the injector moved,
                       in label-SD units: the root-mean-square for a variance
                       condition, the mean displacement for a shift.
  predicted AUROC      the report's two formulas, with their inputs beside them:
                         shift       Phi(delta / (sqrt(2) * s_m))
                         wider error (2/pi) * arctan(sqrt(1 + rho^2)),
                                     rho = realised RMS / s_m
                       Both assume Gaussian residuals and a model fitted on
                       clean labels; the report says so and so does this file.
  measured AUROC       `detection_scores` on the absolute out-of-fold residual
                       of the NOISY labels, against the injector's mask. The
                       signed residual is graded too, because the shift formula
                       is written for a signed score.
  residual shape       excess kurtosis of the out-of-fold residual, its one-way
                       intraclass correlation by scaffold group, and its
                       correlation with the prediction (the report's D4). On
                       the noisy labels and on the clean ones.

THE BREAK-EVEN PRECISION. `headroom_probe.py` (FINDINGS.md section 9) built flag
sets whose precision is SET: a chosen share of the flagged molecules is from the
injector's mask, the rest from outside it. Here the flag set holds each share
in FLAG_SHARES of the training molecules, at precision 0, 0.25, 0.5, 0.75 and
1.0, over FLAG_SEEDS draws. When the injector moved fewer molecules than the
precision asks for, the rest come from outside the mask and the realised
precision, written beside it, is lower. Four things are done with it:

  remove           the flagged molecules dropped
  downweight       the flagged molecules weighted 0.25, every other row 1
  rewrite          the flagged labels replaced by the out-of-fold prediction
  bayes_shrink     the flagged labels moved toward the out-of-fold prediction by
                   w = sigma_n^2 / (sigma_n^2 + s_m^2), the report's H4 weight,
                   both ESTIMATED from the model's own out-of-fold errors on
                   the labels as given (owner's decision 2026-09-28): s_m^2 is
                   the mean squared error on the molecules NOT flagged, and
                   sigma_n^2 the mean squared error on the flagged ones minus
                   s_m^2, floored at 0. So w = 1 - s_m^2 / (flagged MSE). No
                   injected quantity and no clean label enters it. The weight
                   the injector's own numbers would have given is written
                   beside it, for evaluation only.

Each is paired against the noisy fit on the same pass. The break-even precision
is where the mean paired gain crosses zero, interpolated linearly between the
two realised precisions either side. It is printed beside the precision the real
detector reached at the same flag count. The same four operations are also run
on the REAL detector's flag set (the top k by absolute residual), so its outcome
is not read off the curve alone.

HOW MANY MOLECULES ARE FLAGGED. Each share in FLAG_SHARES = 0.05, 0.10, 0.20 of
the training molecules, n24's grid, the same at every condition and level: arms
detector/share<s>/<op> and flags_share<s>_p<q>/seed<k>/<op>. Owner's decision
2026-09-28: no method may read what only exists because the noise was injected
-- how many molecules moved, the condition's nominal share, or the injected
variance. Until then k was the number the injector moved (a tenth of the
training molecules under gaussian). Grading a flag set against the mask is
evaluation and stays.

THE CONDITIONS.
  outlier_only_p05, outlier_only_p10   a share of molecules gets a wider error,
                                       the rest are untouched (registry)
  shift_only_p05, shift_only_p10       a share of molecules is moved UP by
                                       delta label SDs, the rest untouched.
                                       Defined in noise_helpers for n31 and n32
                                       only, not in the NoiseInject registry.
  gaussian                             every label moves. Nothing separates a
                                       moved molecule from an untouched one, so
                                       detection, the predicted AUROC and the
                                       synthetic flags are undefined here and
                                       are not written. The detector's own flag
                                       set is still acted on at every swept
                                       share: the case where the report says
                                       nothing should work.

Levels: the dataset's own dose grid for the registry conditions; 0 followed by
SHIFT_ONLY_DELTAS_SD for the shifts. Level 0, one draw, is where no molecule
moved: the detector arms clean the recorded labels, and detection, the predicted
AUROC and the synthetic flags are not written (nothing moved, so a flag set has
no precision).

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, untouched, in
every arm. Every scored fit also writes the error on the top and bottom tenth of
clean held-out labels and the calibration slope (`tail_scores`), because the
report warns that rewriting labels toward a prediction pulls the extremes in.

WHERE THE FEATURES COME FROM. As n30: the recipe per downstream model is read
from the precomputed pool, searched on clean labels. Each pass rebuilds the
hybrid from that pass's NOISY training labels, once, and every arm shares it.

THE SINGLE-REPRESENTATION COMPARISON is each downstream model's top 2 base
representations by that model's own score in the stored search -- spaces
`base_1` and `base_2` -- the same rule as the rest of the study
(_shared.noise_base_reps_by_model, owner's decision 2026-09-25). Which
representations those are is written to `base_reps_by_model`; a model the pool
holds no stored search for has none, and `base_reps_missing` says why.

ROWS, CACHES, CHECKPOINT (2026-09-25, plan sections 2-3). Every detector and
synthetic-flag arm writes one D-versus-D' row per pass against `noisy`, with
`clean` as score_clean. The accumulators are checkpointed per pass; the caches
of plan section 3 are used. controls.py's twins run with
KIRBY_NOISE_CONTROLS=1.

Usage:
    python n31_regime_and_breakeven.py --dataset logd --mode full
    python n31_regime_and_breakeven.py --dataset herg_ki --mode full \
        --conditions shift_only_p10 --levels 1.0
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
from scipy.stats import norm, kurtosis
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
    detection_scores, tail_scores,
    inject_shift_train_labels, SHIFT_ONLY_CONDITIONS, SHIFT_ONLY_DELTAS_SD,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.utils.noisy_labels import oof_predict
# The D-versus-D' rows, the per-pass checkpoint, the caches and the controls
# (2026-09-25, plan sections 2-3). They live in n23 for the reason given at
# the top of its shared block.
from experiments.noise import noise_cache as nc
from experiments.noise.controls import controls_enabled
from experiments.noise.n23_uncertainty_cleaning import (
    RowSink, PassCheckpoint, ensure_dprime_models, control_twins,
    as_cleanup_dict, cached_base_cell, cached_hybrid_cell, CACHEABLE_MODELS,
    NO_NEIGHBOURS,
)

# The conditions, in the order they run. See the docstring for what each does.
VARIANCE_CONDITIONS = ('outlier_only_p05', 'outlier_only_p10')
SHIFT_CONDITIONS = tuple(SHIFT_ONLY_CONDITIONS)
EVERY_ROW_CONDITIONS = ('gaussian',)
CONDITIONS = VARIANCE_CONDITIONS + SHIFT_CONDITIONS + EVERY_ROW_CONDITIONS

# The injection replicates. n07, n11, n13 and n24 sweep the same three.
NOISE_DRAWS = (0, 1, 2)

# Precision of the synthetic flag set, as headroom_probe.py swept it.
PRECISIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Draws of WHICH molecules fill each synthetic flag set. At precision 1.0 the set
# is the injector's mask and every seed gives the same set; at the others the
# seed changes which moved and which untouched molecules are picked.
FLAG_SEEDS = (0, 1)

# The four operations on a flag set, and the weight downweight gives.
OPERATIONS = ('remove', 'downweight', 'rewrite', 'bayes_shrink')
DOWNWEIGHT_TO = 0.25

# The share of training molecules every detector and synthetic flag set holds,
# swept. n24's FLAG_RATES; the same at every condition and level, so no method
# reads how many molecules the injector moved (owner's decision 2026-09-28).
FLAG_SHARES = (0.05, 0.10, 0.20)

# Inner folds for the out-of-fold prediction. Same as n19 and n30.
OOF_SPLITS = 5

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}
# base_1 and base_2 are each downstream model's rank-1 and rank-2 single
# representation (see the module docstring).
BASE_SPACES = tuple(f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1))


# ── The report's formulas ────────────────────────────────────────────────────

def predicted_auroc_shift(delta_sd, s_m):
    """Phi(delta / (sqrt(2) * s_m)), both in label-SD units. Signed score."""
    if not (np.isfinite(delta_sd) and np.isfinite(s_m)) or s_m <= 0:
        return float('nan')
    return float(norm.cdf(float(delta_sd) / (np.sqrt(2.0) * float(s_m))))


def predicted_auroc_wider(sigma_n_sd, s_m):
    """(2/pi) * arctan(sqrt(1 + rho^2)), rho = sigma_n / s_m. Absolute score."""
    if not (np.isfinite(sigma_n_sd) and np.isfinite(s_m)) or s_m <= 0:
        return float('nan')
    rho = float(sigma_n_sd) / float(s_m)
    return float((2.0 / np.pi) * np.arctan(np.sqrt(1.0 + rho ** 2)))


# ── Residual shape (the report's D4) ─────────────────────────────────────────

def one_way_icc(values, groups):
    """ICC(1) of `values` by `groups`, from the one-way ANOVA mean squares.

    Uses every group; a group of one molecule adds to the between-group sum
    and nothing to the within. Returns (icc, why) -- icc None with the reason
    when fewer than two groups hold two or more molecules.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    if v.size != g.size:
        return None, 'values and groups disagree on length'
    labels, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    if int((counts >= 2).sum()) < 2:
        return None, (f'{int((counts >= 2).sum())} scaffold groups hold two or '
                      f'more molecules; the within-group spread is undefined')
    k = labels.size
    n = v.size
    means = np.bincount(inv, weights=v) / counts
    grand = float(v.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((v - means[inv]) ** 2))
    if n - k <= 0 or k <= 1:
        return None, 'no degrees of freedom within groups'
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - float(np.sum(counts ** 2)) / n) / (k - 1)
    denom = msb + (n0 - 1.0) * msw
    if denom <= 0:
        return None, 'the mean squares give a zero denominator'
    return float((msb - msw) / denom), None


def residual_shape(y, p, groups):
    """Kurtosis, intraclass correlation by group, and correlation with p."""
    r = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    out = {'excess_kurtosis': float(kurtosis(r, fisher=True, bias=False))
           if r.size > 3 else float('nan')}
    if np.std(p) > 0 and np.std(r) > 0:
        out['corr_residual_with_prediction'] = float(np.corrcoef(r, p)[0, 1])
    else:
        out['corr_residual_with_prediction'] = float('nan')
    if groups is None:
        out['icc_by_scaffold_group'] = None
        out['icc_refused_because'] = ('this dataset has no SMILES, so '
                                      'noise_groups_for gives no scaffold groups')
    else:
        icc, why = one_way_icc(r, groups)
        out['icc_by_scaffold_group'] = icc
        if why:
            out['icc_refused_because'] = why
    return out


# ── Flag sets and what is done with them ─────────────────────────────────────

def synthetic_flags(mask, precision, seed, k):
    """A flag set of k molecules, `precision` of them from the mask.

    Returns (flag, realised precision). The count from the mask is
    round(precision * k), capped at the mask's size; the rest come from
    outside it. When inside holds too few, the realised precision is lower
    than asked; when outside holds too few, the set is smaller. The realised
    precision says so either way. k is a swept share of the training
    molecules, never the mask's size (owner's decision 2026-09-28).
    """
    mask = np.asarray(mask, dtype=bool)
    k = int(k)
    rng = np.random.default_rng(int(seed))
    inside = np.flatnonzero(mask)
    outside = np.flatnonzero(~mask)
    n_in = min(int(round(float(precision) * k)), inside.size)
    n_out = min(k - n_in, outside.size)
    flag = np.zeros(mask.size, dtype=bool)
    if n_in:
        flag[rng.choice(inside, n_in, replace=False)] = True
    if n_out:
        flag[rng.choice(outside, n_out, replace=False)] = True
    total = int(flag.sum())
    return flag, (float(mask[flag].sum()) / total if total else float('nan'))


def detector_flags(abs_residual, k):
    """The top k training molecules by absolute out-of-fold residual."""
    order = np.argsort(-np.asarray(abs_residual, dtype=float), kind='stable')
    flag = np.zeros(len(order), dtype=bool)
    flag[order[:int(k)]] = True
    return flag


def estimated_shrink_weight(y, p, flag):
    """bayes_shrink's weight from the model's own out-of-fold errors.

    s_m^2 = mean squared error on the unflagged molecules (the model's error on
    ordinary labels); sigma_n^2 = mean squared error on the flagged ones minus
    s_m^2, floored at 0. w = sigma_n^2 / (sigma_n^2 + s_m^2). Reads only the
    labels as given and the out-of-fold prediction. 0 when either side is
    empty.
    """
    r2 = (np.asarray(y, dtype=float) - np.asarray(p, dtype=float)) ** 2
    flag = np.asarray(flag, dtype=bool)
    if not flag.any() or flag.all():
        return 0.0
    s_m2 = float(np.mean(r2[~flag]))
    flagged = float(np.mean(r2[flag]))
    sigma_n2 = max(flagged - s_m2, 0.0)
    return (sigma_n2 / (sigma_n2 + s_m2)) if sigma_n2 + s_m2 > 0 else 0.0


def apply_operation(op, y, flag, p, w_shrink=None):
    """(labels, keep, sample_weight) for one operation on one flag set.
    bayes_shrink's weight is estimated from this flag set when w_shrink is
    None, which is how every arm calls it."""
    y = np.asarray(y, dtype=float)
    if op == 'remove':
        return y, ~flag, None
    if op == 'downweight':
        w = np.ones(y.size)
        w[flag] = DOWNWEIGHT_TO
        return y, None, w
    if op == 'rewrite':
        out = y.copy()
        out[flag] = np.asarray(p, dtype=float)[flag]
        return out, None, None
    if op == 'bayes_shrink':
        if w_shrink is None:
            w_shrink = estimated_shrink_weight(y, p, flag)
        out = y.copy()
        out[flag] = ((1.0 - w_shrink) * y[flag]
                     + w_shrink * np.asarray(p, dtype=float)[flag])
        return out, None, None
    raise ValueError(f'unknown operation {op!r}')


def break_even_precision(precisions, gains):
    """Lowest precision where the mean gain crosses from below zero to above.

    Linear interpolation between the two precisions either side. Returns
    (value, why): value None when the gain never crosses, with which side it
    stays on.
    """
    pts = [(float(q), float(g)) for q, g in zip(precisions, gains)
           if g is not None and np.isfinite(g)
           and q is not None and np.isfinite(q)]
    pts.sort()
    if len(pts) < 2:
        return None, 'fewer than two precisions scored'
    if pts[0][1] >= 0:
        return None, 'the gain is at or above zero at the lowest precision'
    for (q0, g0), (q1, g1) in zip(pts, pts[1:]):
        if g0 < 0 <= g1:
            return float(q0 + (0.0 - g0) * (q1 - q0) / (g1 - g0)), None
    return None, 'the gain stays below zero at every precision scored'


def _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn, metric_name,
               keep=None, sample_weight=None):
    """One fit, (score, tail dict) or (None, why)."""
    try:
        m = factory()
        y_tr = np.asarray(y_tr, dtype=float)
        if keep is not None:
            X_tr, y_tr = X_tr[keep], y_tr[keep]
        if sample_weight is None:
            m.fit(X_tr, y_tr)
        else:
            m.fit(X_tr, y_tr, sample_weight=sample_weight)
        pred = _predict_for_metric(m, X_va, metric_name)
        return (float(metric_fn(y_va, pred)),
                tail_scores(y_va, m.predict(X_va))), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _fit_score_cached(dataset_name, model_name, factory, X_tr, y_tr, X_va,
                     y_va, metric_fn, metric_name):
    """_fit_score for the two fits plan section 3 lists as repeated -- the
    clean-label fit and the no-cleaning fit on D -- through
    noise_cache.cached_fit_predict for a seeded model (CACHEABLE_MODELS).

    Same numbers: cached_fit_predict fits factory() on the same arrays, and
    for a regression metric (n31 is regression only) its prediction is
    model.predict, which is both what _predict_for_metric returns and what
    tail_scores reads. Anything else goes to _fit_score. Added 2026-09-25."""
    if (dataset_name is None or model_name not in CACHEABLE_MODELS
            or nc.predict_mode_for(metric_name) != 'predict'):
        return _fit_score(factory, X_tr, y_tr, X_va, y_va, metric_fn,
                          metric_name)
    try:
        pred = nc.cached_fit_predict(
            dataset_name, model_name, factory, X_tr,
            np.asarray(y_tr, dtype=float), X_va, predict_mode='predict')
        return (float(metric_fn(y_va, pred)), tail_scores(y_va, pred)), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _passes(dataset_name, conditions, levels=None, draws=None):
    """(condition, level, draw) triples. At level 0 nothing moves, so one draw
    is kept. Level 0 was left out until 2026-09-28, when the flag count stopped
    being the number moved; at a swept share it cleans the recorded labels."""
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for c in conditions:
        if levels is not None:
            grid = levels
        elif c in SHIFT_ONLY_CONDITIONS:
            grid = (0.0,) + tuple(SHIFT_ONLY_DELTAS_SD)
        else:
            grid = noise_levels(dataset_name, strategy=c, is_clf=False)
        for lvl in grid:
            for d in (draws if float(lvl) > 0 else draws[:1]):
                out.append((c, float(lvl), int(d)))
    return out


# _shift_seed (crc32 of condition|draw|fold) was removed 2026-09-30: every
# experiment now seeds shift noise through noise_helpers.noise_draw_seed, and
# the fold is not in it.


def run(data, config, output_dir, dataset_name, **kwargs):
    """N31: predicted detector ceiling, measured detector, break-even precision."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N31: DETECTOR CEILING AND BREAK-EVEN PRECISION")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. The report's formulas "
              f"are for a regression residual.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    conditions = list(kwargs.get('conditions') or CONDITIONS)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f'n31: unknown condition(s) {unknown}; this file runs '
                         f'{list(CONDITIONS)}')
    precisions = [float(q) for q in (kwargs.get('precisions') or PRECISIONS)]
    flag_shares = [float(x) for x in (kwargs.get('flag_shares') or FLAG_SHARES)]
    flag_seeds = [int(s) for s in (kwargs.get('flag_seeds') or FLAG_SEEDS)]
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))

    # rf and svm are added when the caller left them out: every D-versus-D'
    # row needs them (ensure_dprime_models, owner decision 2026-09-25).
    downstream_models = ensure_dprime_models(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError('n31: no downstream model could be built.')

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')
    noise_groups = noise_groups_for(data)[0]

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'conditions': conditions,
        'dose_units': noise_dose_units(dataset_name),
        'models': downstream_models,
        'precisions': precisions,
        'flag_seeds': flag_seeds,
        'operations': list(OPERATIONS),
        'downweight_to': DOWNWEIGHT_TO,
        'flag_shares': flag_shares,
        'flag_share_note': (
            'Every detector and synthetic flag set holds each share in '
            'flag_shares of the training molecules, the same at every '
            'condition and level (owner\'s decision 2026-09-28). The number '
            'the injector moved is used only to grade a flag set.'),
        'oof_splits': oof_splits,
        'shift_levels_are': 'delta in SDs of the full clean label column',
        'formula_note': (
            'Predicted AUROC is the research report\'s derivation, not a '
            'measurement: shift Phi(delta/(sqrt(2) s_m)) for a signed score, '
            'wider error (2/pi) arctan(sqrt(1+rho^2)) for an absolute score. '
            's_m is the clean-label out-of-fold rmse over the clean label SD, '
            'in the same fold and feature space. Both assume Gaussian '
            'residuals.'),
        'bayes_shrink_note': (
            'w = sigma_n^2/(sigma_n^2 + s_m^2), estimated per flag set from the '
            'out-of-fold errors on the labels as given: s_m^2 the mean squared '
            'error on the unflagged molecules, sigma_n^2 the flagged mean '
            'squared error minus s_m^2, floored at 0 (2026-09-28). '
            'bayes_shrink_weight_from_injection in the diagnostics is the '
            'weight the injector would have given, for comparison only.'),
        'features_are_fixed_note': (
            'Each feature space is built once per pass from the NOISY training '
            'labels and shared by every arm, clean included. The recipe was '
            'searched on clean labels.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to the '
            'condition named on it. Held-out molecules are graded against '
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
    base_spaces_run = tuple(
        f'base_{k}' for k in range(1, NOISE_BASE_REP_COUNT + 1)
        if base_rank_groups(base_by_model, k, downstream_models))

    _ordered = []
    for _m in downstream_models:
        for r in recipes_by_model.get(_m, ()):
            if r['space'] not in _ordered:
                _ordered.append(r['space'])
    spaces = base_spaces_run + tuple(_ordered)
    results['hybrid_available'] = bool(_ordered)
    if _ordered:
        results['hybrid_recipe_by_model'] = convert_numpy({
            m: {r['space']: {
                'built_under_objective': r['objective'],
                # Which of the owner's set this space is (top 2 general + top 1
                # distance per model and family, 2026-10-02).
                'slot': r.get('slot'), 'lineage': r.get('lineage'),
                'budget_relaxed': r.get('budget_relaxed'),
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
            (f"the pool lookup RAISED {results['hybrid_lookup_error']}; the "
             f"pool was never read. Every arm ran in the single "
             f"representations only.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from pool subset {POOL_SUBSET!r} for any '
             f'downstream model. Every arm ran in the single '
             f'representations only.'))
        print("  no hybrid recipe; single representations only")
    results['spaces_run'] = list(spaces)

    passes = _passes(dataset_name, conditions, levels=kwargs.get('levels'),
                     draws=kwargs.get('draws'))
    print(f"  {len(labels)} molecules, {len(splits)} folds, models "
          f"{downstream_models}")
    print(f"  {len(passes)} (condition, level, draw) passes; spaces "
          f"{list(spaces)}")

    # acc[cond][level][space][arm][model][pass] = score; tails the same shape.
    acc, tails = {}, {}
    # diag[cond][level][space][pass] = the per-pass record (formulas, detection,
    # residual shape). One per (pass, space): none of it depends on the model.
    diag = {}
    errors = {}
    t0 = time.time()

    # ── Rows and the per-pass checkpoint (2026-09-25) ──
    # D is `noisy`; every detector/share<s>/<op> and
    # flags_share<s>_p<q>/seed<k>/<op> arm is a
    # D'; `clean` is score_clean. No arm here reads neighbours: the flags come
    # from the absolute out-of-fold residual or from the injector's mask.
    _metric = (metric_name, metric_fn, lower_is_better)
    rows = RowSink(output_dir, 'n31_regime_and_breakeven', dataset_name,
                   experiment='n31', metric=_metric, condition=kwargs.get('tag'),
                   data=data, config=config)
    ckpt = PassCheckpoint(
        output_dir, 'n31_regime_and_breakeven', dataset_name,
        condition=kwargs.get('tag'), data=data, config=config,
        source_file=__file__,
        key={'conditions': conditions, 'precisions': precisions,
             'flag_shares': flag_shares,
             'flag_seeds': flag_seeds, 'oof_splits': oof_splits,
             'passes': passes, 'models': list(downstream_models),
             'base_reps': base_by_model, 'n_splits': n_folds})
    done_passes = set()
    _state = ckpt.load()
    if _state:
        acc, tails, diag, errors = (_state['acc'], _state['tails'],
                                    _state['diag'], _state['errors'])
        done_passes = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_passes)} (condition, "
              f"level, draw, fold) passes already done")

    for condition, level, draw in passes:
        lvl_key = f'{level:g}'
        is_shift = condition in SHIFT_ONLY_CONDITIONS
        every_row = condition in EVERY_ROW_CONDITIONS
        spec = (None if is_shift else
                noise_spec_for(dataset_name, condition, level,
                               noise_seed=draw, is_clf=False))
        print(f"\n  --- {condition} / level {lvl_key} / draw {draw} ---")

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pass_key = f'd{draw}f{fold_i}'
            _done_id = f'{condition}|{lvl_key}|{pass_key}'
            if _done_id in done_passes:
                print(f"    {_done_id}: on disk in the checkpoint")
                continue
            y_val = labels[val_idx]
            y_clean_tr = labels[train_idx]
            g_tr = groups[train_idx] if groups is not None else None
            ng_tr = (np.asarray(noise_groups)[train_idx]
                     if noise_groups is not None else None)
            try:
                if is_shift:
                    y0, mask, prov = inject_shift_train_labels(
                        labels, train_idx=train_idx, condition=condition,
                        delta_sd=level, draw=draw)
                else:
                    y0, mask, prov = inject_train_labels(
                        labels, spec, train_idx=train_idx, is_clf=False,
                        groups=noise_groups, dataset_name=dataset_name)
            except Exception as exc:
                errors[f'{condition}|{lvl_key}|{pass_key}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            label_sd = float(np.std(y_clean_tr))
            disp = y0 - y_clean_tr
            moved = np.abs(disp) > 0 if every_row else mask
            n_moved = int(moved.sum())
            sigma_n2 = (float(np.mean(disp[moved] ** 2)) if n_moved
                        else float('nan'))

            train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                          for k, v in reps.items()}
            val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}

            fold_spaces = {}
            # Each single-representation space holds one matrix per distinct
            # representation at that rank, shared by the models that rank it
            # there.
            for _k, _bsp in enumerate(BASE_SPACES, start=1):
                if _bsp not in spaces:
                    continue
                _bcells = []
                for _rn, _ms in base_rank_groups(base_by_model, _k,
                                                 downstream_models).items():
                    # noise_cache.cached_scaled since 2026-09-25.
                    _Xt, _Xv = cached_base_cell(dataset_name, reps[_rn],
                                                train_idx, val_idx)
                    _bcells.append((_Xt, _Xv, list(_ms)))
                fold_spaces[_bsp] = _bcells
            for _sp in spaces:
                if _sp in BASE_SPACES:
                    continue
                _cells = []
                for _bp_m, _mnames, _shareable in group_models_by_recipe(
                        {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                         for _m in downstream_models}, downstream_models):
                    # noise_cache.cached_hybrid_for_fold since 2026-09-25
                    # (plan section 3).
                    X_tr_h, X_va_h = cached_hybrid_cell(
                        dataset_name, train_reps, val_reps, y0, _bp_m, g_tr,
                        None if _shareable else _mnames[0],
                        None if _shareable
                        else model_factories[_mnames[0]](),
                        _metric, train_idx)
                    if X_tr_h is None:
                        errors[f'{condition}|{lvl_key}|{_sp}/{_mnames[0]}|'
                               f'{pass_key}|build'] = 'the hybrid could not be built'
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
                    ekey = f'{condition}|{lvl_key}|{_ekey}|{pass_key}'
                    try:
                        # noise_cache.cached_oof_predict since 2026-09-25:
                        # the same calls (seed 0), stored. p_clean depends
                        # on the fold alone, so it is computed once for every
                        # condition, level and draw.
                        p_clean = nc.cached_oof_predict(
                            dataset_name, X_tr_s, y_clean_tr, g_tr,
                            n_splits=oof_splits)
                        p = nc.cached_oof_predict(dataset_name, X_tr_s, y0,
                                                  g_tr, n_splits=oof_splits)
                    except Exception as exc:
                        errors[f'{ekey}|oof'] = f'{type(exc).__name__}: {exc}'
                        continue
                    clean_mse = float(np.mean((y_clean_tr - p_clean) ** 2))
                    s_m = (np.sqrt(clean_mse) / label_sd if label_sd > 0
                           else float('nan'))
                    resid = y0 - p
                    rec = {
                        'training_molecules': int(len(y0)),
                        'molecules_moved': n_moved,
                        'label_sd': label_sd,
                        's_m_label_sd_units': float(s_m),
                        'clean_oof_rmse_label_units': float(np.sqrt(clean_mse)),
                        'noisy_oof_rmse_on_untouched_label_units': (
                            float(np.sqrt(np.mean(resid[~moved] ** 2)))
                            if (~moved).any() else float('nan')),
                        'realised_mean_squared_injected_error_label_units': sigma_n2,
                        'residual_shape_noisy': residual_shape(y0, p, ng_tr),
                        'residual_shape_clean': residual_shape(
                            y_clean_tr, p_clean, ng_tr),
                    }
                    # Evaluation only, never handed to an arm: the weight the
                    # injector's own variance and the clean labels would give.
                    rec['bayes_shrink_weight_from_injection'] = float(
                        sigma_n2 / (sigma_n2 + clean_mse)
                        if np.isfinite(sigma_n2)
                        and sigma_n2 + clean_mse > 0 else 0.0)
                    # At level 0 nothing moved: no detection, no predicted
                    # AUROC, no synthetic flags. The detector arms still run.
                    nothing_moved = float(level) <= 0
                    if nothing_moved:
                        rec['detection_not_defined_because'] = (
                            'level 0: no training label moved')
                        rec['synthetic_flags_not_written_because'] = (
                            'level 0: no training label moved, so a flag set '
                            'has no precision to set')
                    elif not every_row:
                        if is_shift:
                            delta = float(np.mean(disp[mask])) / label_sd
                            rec['realised_shift_label_sd_units'] = delta
                            rec['predicted_auroc'] = predicted_auroc_shift(
                                delta, s_m)
                            rec['predicted_auroc_formula'] = (
                                'Phi(delta/(sqrt(2) s_m)), signed score')
                        else:
                            sig = np.sqrt(sigma_n2) / label_sd
                            rec['realised_rms_injected_error_label_sd_units'] = (
                                float(sig))
                            rec['rho'] = (float(sig / s_m) if s_m > 0
                                          else float('nan'))
                            rec['predicted_auroc'] = predicted_auroc_wider(
                                sig, s_m)
                            rec['predicted_auroc_formula'] = (
                                '(2/pi) arctan(sqrt(1+rho^2)), absolute score')
                        rec['detection_abs_residual'] = detection_scores(
                            mask, np.abs(resid), condition)
                        rec['detection_signed_residual'] = detection_scores(
                            mask, resid, condition)
                    else:
                        rec['detection_not_defined_because'] = (
                            'every training label moved, so no molecule is '
                            'untouched to tell apart')
                    # One flag count per swept share of the training
                    # molecules, whatever the condition moved.
                    k_by_share = {sh: int(round(sh * len(y0)))
                                  for sh in flag_shares}
                    rec['flag_count_by_share'] = {f'{sh:g}': k for sh, k
                                                  in k_by_share.items()}
                    (diag.setdefault(condition, {}).setdefault(lvl_key, {})
                     .setdefault(_ekey, {}))[pass_key] = rec

                    # Every column this (pass, space) fits, as
                    # arm -> (labels, keep, weight).
                    cols = {'clean': (y_clean_tr, None, None),
                            'noisy': (y0, None, None)}
                    det_prec = {}
                    for sh, k_flag in k_by_share.items():
                        if k_flag <= 0:
                            continue
                        dflag = detector_flags(np.abs(resid), k_flag)
                        if not every_row and not nothing_moved:
                            # Evaluation only: how many of the flagged
                            # molecules the injector moved.
                            det_prec[f'{sh:g}'] = float(mask[dflag].mean())
                        for op in OPERATIONS:
                            cols[f'detector/share{sh:g}/{op}'] = (
                                apply_operation(op, y0, dflag, p))
                    if det_prec:
                        rec['detector_precision_by_share'] = det_prec
                    if not every_row and not nothing_moved:
                        realised = {}
                        for sh, k_flag in k_by_share.items():
                            if k_flag <= 0:
                                continue
                            for q in precisions:
                                for s in flag_seeds:
                                    seed = zlib.crc32(
                                        f'{sh}|{q}|{s}|{condition}|{lvl_key}|'
                                        f'{pass_key}'.encode()) & 0x7FFFFFFF
                                    flag, rp = synthetic_flags(mask, q, seed,
                                                               k_flag)
                                    realised[f'share{sh:g}/p{q:g}/seed{s}'] = rp
                                    for op in OPERATIONS:
                                        cols[f'flags_share{sh:g}_p{q:g}/'
                                             f'seed{s}/{op}'] = (
                                            apply_operation(op, y0, flag, p))
                        rec['synthetic_realised_precision'] = realised

                    sc_cell = (acc.setdefault(condition, {})
                               .setdefault(lvl_key, {}).setdefault(space, {}))
                    tl_cell = (tails.setdefault(condition, {})
                               .setdefault(lvl_key, {}).setdefault(space, {}))
                    for arm, (y_use, keep, w) in cols.items():
                        for mname in cell_models:
                            if arm in ('clean', 'noisy'):
                                got, err = _fit_score_cached(
                                    dataset_name, mname,
                                    model_factories[mname], X_tr_s, y_use,
                                    X_va_s, y_val, metric_fn, metric_name)
                            else:
                                got, err = _fit_score(
                                    model_factories[mname], X_tr_s, y_use,
                                    X_va_s, y_val, metric_fn, metric_name,
                                    keep=keep, sample_weight=w)
                            if got is None:
                                errors[f'{ekey}|{arm}|{mname}'] = err
                                continue
                            s_val, tail = got
                            (sc_cell.setdefault(arm, {})
                             .setdefault(mname, {}))[pass_key] = s_val
                            (tl_cell.setdefault(arm, {})
                             .setdefault(mname, {}))[pass_key] = tail

                    # ── The D-versus-D' rows and the controls (2026-09-25) ──
                    for mname in cell_models:
                        sD = (sc_cell.get('noisy') or {}).get(
                            mname, {}).get(pass_key)
                        sC = (sc_cell.get('clean') or {}).get(
                            mname, {}).get(pass_key)
                        for arm in cols:
                            if arm in ('clean', 'noisy'):
                                continue
                            sP = (sc_cell.get(arm) or {}).get(
                                mname, {}).get(pass_key)
                            if sP is None:
                                continue
                            rows.add(method=arm.rsplit('/', 1)[-1],
                                     method_setting=_arm_setting(arm),
                                     similarity_setting=NO_NEIGHBOURS,
                                     model_input=space, model=mname,
                                     noise_condition=condition,
                                     level=float(level), draw=int(draw),
                                     fold=fold_i, score_D=sD, score_Dprime=sP,
                                     score_clean=sC)
                    # controls.py's twins of every arm that changes the
                    # training data, when KIRBY_NOISE_CONTROLS=1.
                    if controls_enabled():
                        for arm, (y_use, keep, w) in cols.items():
                            if arm in ('clean', 'noisy'):
                                continue
                            _cd = as_cleanup_dict(X_tr_s, y0, keep=keep,
                                                  weight=w, labels=y_use)
                            _seed = (int(draw), int(fold_i), int(zlib.crc32(
                                f'{condition}|{lvl_key}|{_ekey}|{arm}'
                                .encode()) & 0x7FFFFFFF))
                            for cname, tw, why in control_twins(
                                    _cd, X_tr_s, y0, _seed):
                                if tw is None:
                                    errors[f'{ekey}|{arm}|{cname}'] = why
                                    continue
                                for mname in cell_models:
                                    got, err = _fit_score(
                                        model_factories[mname], tw['X'],
                                        tw['y'], X_va_s, y_val, metric_fn,
                                        metric_name,
                                        sample_weight=tw.get('sample_weight'))
                                    rows.add(
                                        method=arm.rsplit('/', 1)[-1],
                                        method_setting=(f'{_arm_setting(arm)}'
                                                        f'|control={cname}'),
                                        similarity_setting=NO_NEIGHBOURS,
                                        model_input=space, model=mname,
                                        noise_condition=condition,
                                        level=float(level), draw=int(draw),
                                        fold=fold_i,
                                        score_D=(sc_cell.get('noisy') or {})
                                        .get(mname, {}).get(pass_key),
                                        score_Dprime=(None if got is None
                                                      else got[0]),
                                        score_clean=(sc_cell.get('clean') or {})
                                        .get(mname, {}).get(pass_key))

            # The pass is finished: its rows to disk, then the accumulators.
            rows.flush()
            done_passes.add(_done_id)
            ckpt.save({'acc': acc, 'tails': tails, 'diag': diag,
                       'errors': errors, 'done': sorted(done_passes)})
            print(f"    {condition} {lvl_key} {pass_key}: {n_moved} of "
                  f"{len(y0)} training labels moved [{time.time() - t0:.0f}s]")

    results['per_condition'] = convert_numpy(acc)
    results['tails'] = convert_numpy(tails)
    results['diagnostics'] = convert_numpy(diag)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, diag, downstream_models, precisions, flag_seeds, metric_name,
        lower_is_better, flag_shares))
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results, metric_name)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    ckpt.clear()
    return results


def _arm_setting(arm):
    """The rows' method_setting for an arm: the arm name. (Until 2026-09-28
    bayes_shrink rows carried |uses_injection_knowledge; its weight is now
    estimated from the model's errors.)"""
    return arm


def _paired(base, got, lower_is_better):
    """One arm against another on the passes both scored. Positive = better."""
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


def _mean(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def _summarise(acc, diag, models, precisions, flag_seeds, metric_name,
               lower_is_better, flag_shares=FLAG_SHARES):
    """Per (condition, level, space, model, flag share): the detector ops, the
    synthetic curve and its break-even precision, beside the predicted and
    measured detector numbers for the same space.

    The break-even is interpolated on the mean REALISED precision of each
    point: at a share larger than the moved set, precision 1.0 cannot be
    reached and the realised value is what the curve was measured at."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('every gain is paired against the noisy fit on the '
                        'same pass; a pass is one (draw, fold)'),
           'per_condition': {}}
    for cond, by_lvl in acc.items():
        crow = {}
        for lvl, by_space in by_lvl.items():
            lrow = {}
            for space, by_arm in by_space.items():
                # The detection record lives under the space key or a
                # space/model key when models had separate matrices.
                drecs = {}
                for ek, recs in (diag.get(cond, {}).get(lvl, {})).items():
                    if ek == space or ek.startswith(space + '/'):
                        drecs.update({f'{ek}|{pk}': r for pk, r in recs.items()})
                det = {
                    'predicted_auroc_mean': _mean(
                        r.get('predicted_auroc') for r in drecs.values()),
                    'measured_auroc_abs_residual_mean': _mean(
                        (r.get('detection_abs_residual') or {}).get(
                            'detection_auroc') for r in drecs.values()),
                    'measured_auroc_signed_residual_mean': _mean(
                        (r.get('detection_signed_residual') or {}).get(
                            'detection_auroc') for r in drecs.values()),
                    'detector_precision_at_k_mean': _mean(
                        (r.get('detection_abs_residual') or {}).get(
                            'detection_precision_at_k') for r in drecs.values()),
                    's_m_mean': _mean(r.get('s_m_label_sd_units')
                                      for r in drecs.values()),
                    'excess_kurtosis_noisy_mean': _mean(
                        r['residual_shape_noisy'].get('excess_kurtosis')
                        for r in drecs.values()),
                    'n_pass_records': len(drecs),
                    'detector_precision_by_share_mean': {
                        f'{sh:g}': _mean(
                            (r.get('detector_precision_by_share') or {}).get(
                                f'{sh:g}') for r in drecs.values())
                        for sh in flag_shares},
                }
                srow = {'detection': det, 'per_model': {}}
                noisy = by_arm.get('noisy') or {}
                clean = by_arm.get('clean') or {}
                for m in models:
                    b = noisy.get(m) or {}
                    if not b:
                        continue
                    row = {'noisy_mean': _mean(b.values()),
                           'clean_mean': _mean((clean.get(m) or {}).values()),
                           'clean_vs_noisy': _paired(b, clean.get(m) or {},
                                                     lower_is_better),
                           'detector_ops': {}, 'synthetic': {}}
                    for sh in flag_shares:
                        shk = f'share{sh:g}'
                        for op in OPERATIONS:
                            g = (by_arm.get(f'detector/{shk}/{op}')
                                 or {}).get(m) or {}
                            if g:
                                row['detector_ops'].setdefault(shk, {})[op] = (
                                    _paired(b, g, lower_is_better))
                        for op in OPERATIONS:
                            curve = {}
                            for q in precisions:
                                gains = []
                                n_paired = 0
                                for s in flag_seeds:
                                    g = ((by_arm.get(
                                        f'flags_{shk}_p{q:g}/seed{s}/{op}')
                                          or {}).get(m) or {})
                                    if not g:
                                        continue
                                    pr = _paired(b, g, lower_is_better)
                                    gains += pr['pass_gains']
                                    n_paired += pr['n_passes_paired']
                                if n_paired:
                                    curve[f'{q:g}'] = {
                                        'mean_gain': _mean(gains),
                                        'spread_over_pass_gains': (
                                            float(np.std(gains)) if gains
                                            else float('nan')),
                                        'n_paired': n_paired,
                                        'passes_won': int(sum(x > 0
                                                              for x in gains)),
                                        'realised_precision_mean': _mean(
                                            (r.get('synthetic_realised_precision')
                                             or {}).get(f'{shk}/p{q:g}/seed{s}')
                                            for r in drecs.values()
                                            for s in flag_seeds),
                                    }
                            if not curve:
                                continue
                            be, why = break_even_precision(
                                [curve[k]['realised_precision_mean']
                                 for k in curve],
                                [curve[k]['mean_gain'] for k in curve])
                            row['synthetic'].setdefault(shk, {})[op] = {
                                'curve': curve,
                                'break_even_precision': be,
                                'break_even_refused_because': why,
                                'detector_precision_at_this_share_mean': det[
                                    'detector_precision_by_share_mean'].get(
                                        f'{sh:g}'),
                            }
                    srow['per_model'][m] = row
                lrow[space] = srow
            crow[lvl] = lrow
        out['per_condition'][cond] = crow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; gains paired per pass against the noisy fit)")
    for cond, crow in s['per_condition'].items():
        for lvl, lrow in crow.items():
            print(f"\n  {cond} level {lvl}:")
            for space, srow in lrow.items():
                d = srow['detection']
                print(f"    {space}: model error {d['s_m_mean']:.3f} label SDs; "
                      f"predicted AUROC {d['predicted_auroc_mean']:.3f}, "
                      f"measured {d['measured_auroc_abs_residual_mean']:.3f} "
                      f"(absolute residual) / "
                      f"{d['measured_auroc_signed_residual_mean']:.3f} (signed); "
                      f"detector precision at k "
                      f"{d['detector_precision_at_k_mean']:.3f}")
                for m, row in srow['per_model'].items():
                    print(f"      {m}: noisy {row['noisy_mean']:.4f}, clean "
                          f"{row['clean_mean']:.4f}")
                    for shk, by_op in row['detector_ops'].items():
                        for op, e in by_op.items():
                            print(f"        detector flags {shk}, {op:12s} won "
                                  f"{e['passes_won']} of "
                                  f"{e['n_passes_paired']} passes, mean "
                                  f"{e['mean_gain']:+.4f}")
                    for shk, by_op in row['synthetic'].items():
                        for op, e in by_op.items():
                            be = e['break_even_precision']
                            print(f"        break-even precision {shk}, "
                                  f"{op:12s} "
                                  + (f"{be:.2f}" if be is not None
                                     else f"none "
                                          f"({e['break_even_refused_because']})"))
    print("\n  A mean gain no larger than the spread of the per-pass gains "
          "behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N31: predicted detector ceiling, measured detector, break-even precision')
    parser.add_argument('--conditions', type=str, default=None,
                        help=f'Comma-separated; default {",".join(CONDITIONS)}')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated levels applied to every condition '
                             '(delta in label SDs for the shifts). Default: each '
                             'condition\'s own grid')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates (default 0,1,2)')
    parser.add_argument('--precisions', type=str, default=None,
                        help='Comma-separated synthetic flag precisions')
    parser.add_argument('--flag-seeds', type=str, default=None,
                        help='Comma-separated synthetic flag draws (default 0,1)')
    parser.add_argument('--flag-shares', type=str, default=None,
                        help='Comma-separated shares of training molecules '
                             'flagged (default '
                             f'{",".join(f"{x:g}" for x in FLAG_SHARES)})')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds. Overrides the 1 or 5 --mode gives')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    if args.exp_folds:
        config['n_splits'] = int(args.exp_folds)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {}
    if args.conditions:
        kw['conditions'] = args.conditions.split(',')
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.precisions:
        kw['precisions'] = [float(x) for x in args.precisions.split(',')]
    if args.flag_seeds:
        kw['flag_seeds'] = [int(x) for x in args.flag_seeds.split(',')]
    if args.flag_shares:
        kw['flag_shares'] = [float(x) for x in args.flag_shares.split(',')]

    res = run(data, config, args.output, args.dataset, **kw)
    suffix = 'all' if not args.conditions else args.conditions.replace(',', '+')
    if args.tag:
        suffix = f'{suffix}_{args.tag}'
    save_experiment_results(res, args.output, 'n31_regime_and_breakeven',
                            args.dataset, filename_suffix=suffix)


if __name__ == '__main__':
    main()
