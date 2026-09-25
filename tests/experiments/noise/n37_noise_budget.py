#!/usr/bin/env python3
"""N37: how much of the model's error is label noise, read without cleaning anything.

Paper: Noise.

THE QUESTION. The third research report (2026-09-25, "Dense Gaussian Noise vs
Sparse Gross Errors") says label noise can only be worth handling where it is a
large share of the model's error, and asks for that share to be measured before
any cleaning is tried. This file measures it on clean labels, per dataset, and
makes no change to any label. Nothing here is a pruning rule: the report's "stop
noise work where the share is under 0.1" is NOT applied; the share is written.

WHAT IS COMPUTED, per fold, on the fold's training molecules as recorded:

  1. THE GAMMA TEST (Stefansson, Koncar & Jones 1997; noise_helpers.gamma_test).
     For each molecule, the k-th nearest neighbour, k = 1..K_MAX. delta_k is the
     mean squared distance to it, gamma_k the mean of half the squared label
     difference. A straight line through the K_MAX (delta, gamma) points; its
     intercept estimates the label-noise variance, in label units squared. It
     also holds real activity differences the features cannot resolve, so it is
     an upper bound. In three distance spaces, as n35:
       base              Tanimoto when every value is 0 or 1, else Euclidean on
                         the fold-standardised base representation
       <hybrid space>    Euclidean on each standardised hybrid matrix, built
                         from the fold's clean training labels
       forest_proximity  one minus the share of 200 trees putting the two
                         molecules in one leaf; refused above n07's row cap.
                         The forest reads the labels, so its close molecules
                         already agree on the label: biased low by construction.
     Differs from n35: per-molecule neighbours rather than the closest 1% of all
     pairs, squared distance, K_MAX points.

  2. THE CHECK. `gaussian` at the dataset's level grid, injected into the
     training labels only. The base distances do not depend on the labels, so
     the base intercept minus its level-0 value should equal the realised
     injected variance on that fold. Only the base space is checked: the other
     two spaces would move their neighbours with the labels.

  3. THE MODEL'S ERROR. Every downstream model in every space, fitted on the
     fold's training labels, scored on the held-out molecules: the domain metric
     and the mean squared error, both against the held-out labels as recorded.
     Written beside it: the base Gamma intercept divided by that MSE, the share
     of the error the noise estimate could account for. On a dataset with
     repeat measurements, the same with n26's pooled within-molecule variance
     of the training rows (n26.direct_noise_variance) in place of the intercept.

  4. SIMEX (Cook & Stefanski 1994), as a sensitivity curve. Extra noise
     N(0, lambda * s2) is added to the training labels, s2 the fold's base Gamma
     intercept, at lambda in SIMEX_LAMBDAS, SIMEX_DRAWS draws each. Every model
     in every space is refitted and scored on the ORIGINAL held-out labels.
     Straight-line and quadratic fits of MSE on lambda (lambda = 0 is the plain
     fit) are carried to lambda = -1: the MSE with the estimated noise gone.
     ⛔ The hybrid features are built ONCE per fold from the clean training
     labels and held fixed across lambda. This curve therefore measures the
     model's sensitivity to label noise, not the feature choice's. Rebuilding
     the hybrid at every (lambda, draw) would multiply the hybrid builds by
     1 + len(SIMEX_LAMBDAS) * len(SIMEX_DRAWS).

  5. RESIDUAL TAILS, on the out-of-fold residual of each (space, model): the
     held-out residuals of every fold put together, so every molecule appears
     once. SD divided by 1.4826 * MAD (about 1.0 for Gaussian residuals),
     excess kurtosis, the Student-t degrees of freedom by maximum likelihood
     (scipy.stats.t.fit), and the Hill tail index over the largest HILL_SHARE of
     |residual|. The first two are also written per fold.

  6. RESIDUAL SPREAD AGAINST THE PREDICTION. The squared out-of-fold residual
     regressed on the prediction: R^2 and n * R^2, the Breusch-Pagan statistic.
     NO p-value is computed. On a dataset with repeat measurements, a second fit
     with the molecule's measurement count and the SD of its measurements added.

WHERE THE FEATURES COME FROM. As in n35: each downstream model's recipe is read
from the precomputed pool, searched on clean labels; each fold rebuilds the
hybrid from that fold's training labels.

Usage:
    python n37_noise_budget.py --dataset logd --mode full
    python n37_noise_budget.py --dataset herg_ic50 --mode full --simex-draws 0
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
from scipy import stats as st
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
    noise_spec_for, inject_train_labels, hybrid_for_fold, gamma_test,
    pool_recipe_per_downstream_model, group_models_by_recipe,
)
from experiments.noise.n07_remediation import (
    _predict_for_metric, PROX_SMOOTH_MAX_ROWS,
)
from experiments.noise.n35_near_duplicate_disagreement import (
    is_binary, tanimoto_distance_matrix, euclidean_distance_matrix,
    MAX_PAIR_ROWS,
)
from experiments.noise.n26_noise_variance_from_repeats import (
    measurements_by_molecule, direct_noise_variance,
)
from experiments.noise._ground_truth_providers import build_replicate_groups

EXP_ID = 'n37_noise_budget'
# The only injected condition, for the check.
CHECK_CONDITION = 'gaussian'
NOISE_DRAWS = (0,)
# Neighbours in the Gamma test.
K_MAX = 10
# SIMEX: extra noise at lambda * the fold's base Gamma intercept.
SIMEX_LAMBDAS = (0.5, 1.0, 2.0)
SIMEX_DRAWS = (0, 1, 2)
# Hill estimator over this share of the largest |residual|.
HILL_SHARE = 0.10
MAD_TO_SD = 1.4826

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}
PROXIMITY_SPACE = 'forest_proximity'


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def hill_index(r, share=HILL_SHARE):
    """Hill's tail index alpha over the largest `share` of |r|.

    H = mean over the top k of log(|r|_(i) / |r|_(k+1)); alpha = 1 / H. A
    Student-t with nu degrees of freedom has alpha = nu; a Gaussian's alpha
    grows with k and n rather than settling.
    """
    a = np.sort(np.abs(np.asarray(r, dtype=float)))[::-1]
    a = a[a > 0]
    k = int(np.ceil(share * a.size))
    if k < 2 or k >= a.size:
        return None
    h = float(np.mean(np.log(a[:k] / a[k])))
    return {'alpha': (1.0 / h) if h > 0 else None, 'k': k, 'n': int(a.size)}


def tail_diagnostics(r, fit_t=True):
    """SD / (1.4826 MAD), excess kurtosis, Student-t nu and Hill alpha of r."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return {'n': int(r.size), 'too_few': True}
    mad = MAD_TO_SD * float(np.median(np.abs(r - np.median(r))))
    out = {
        'n': int(r.size),
        'sd': float(np.std(r)),
        'mad_sd': mad,
        'sd_over_mad_sd': (float(np.std(r)) / mad) if mad > 0 else None,
        'excess_kurtosis': float(st.kurtosis(r)),
        'hill': hill_index(r),
    }
    if fit_t:
        try:
            nu, loc, scale = st.t.fit(r)
            out.update({'student_t_nu': float(nu), 'student_t_loc': float(loc),
                        'student_t_scale': float(scale)})
        except Exception as exc:
            out['student_t_error'] = f'{type(exc).__name__}: {exc}'
    return out


def spread_regression(r, X_aux):
    """R^2 and n*R^2 of r^2 on X_aux (with intercept). No p-value."""
    r2 = np.asarray(r, dtype=float) ** 2
    A = np.column_stack([np.ones(r2.size), np.asarray(X_aux, dtype=float)])
    ok = np.all(np.isfinite(A), axis=1) & np.isfinite(r2)
    A, r2 = A[ok], r2[ok]
    if r2.size <= A.shape[1] + 1 or np.var(r2) == 0:
        return {'n': int(r2.size), 'too_few': True}
    coef, *_ = np.linalg.lstsq(A, r2, rcond=None)
    fit = A @ coef
    rsq = 1.0 - float(np.sum((r2 - fit) ** 2)) / float(np.sum((r2 - r2.mean()) ** 2))
    return {'n': int(r2.size), 'r_squared': rsq,
            'n_times_r_squared': float(r2.size * rsq),
            'coefficients': [float(c) for c in coef]}


def simex_extrapolate(lams, mses):
    """Straight-line and quadratic fits of MSE on lambda, carried to lambda = -1.

    lams includes 0 (the plain fit). Returns the extrapolated MSE from each fit
    and the drop from the plain fit, in label units squared and as a share.
    """
    x = np.asarray(lams, dtype=float)
    y = np.asarray(mses, dtype=float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {'lambdas': x.tolist(), 'mse': y.tolist()}
    if x.size < 2:
        return out
    y0 = float(y[x == 0][0]) if np.any(x == 0) else float('nan')
    for name, deg in (('linear', 1), ('quadratic', 2)):
        if x.size <= deg:
            continue
        c = np.polyfit(x, y, deg)
        at = float(np.polyval(c, -1.0))
        out[name] = {'mse_at_minus_one': at, 'coefficients': c.tolist(),
                     'drop_from_plain': y0 - at,
                     'drop_share_of_plain': ((y0 - at) / y0) if y0 > 0 else None}
    return out


def _seed(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _fit_predict(factory, X_tr, y_tr, X_va, metric_name):
    m = factory()
    m.fit(X_tr, y_tr)
    return np.asarray(_predict_for_metric(m, X_va, metric_name), dtype=float)


def _levels_and_draws(dataset_name, levels=None, draws=None):
    levels = (noise_levels(dataset_name, strategy=CHECK_CONDITION,
                           is_clf=False) if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    return [(float(l), int(d)) for l in levels for d in draws]


def _molecule_count_and_spread(labels, rep_groups):
    """Per row: how many measurements its molecule has, and their SD (ddof 0)."""
    g = np.asarray(rep_groups)
    labels = np.asarray(labels, dtype=float)
    k = np.zeros(len(g))
    sd = np.zeros(len(g))
    by = {}
    for i, gi in enumerate(g):
        by.setdefault(gi, []).append(i)
    for rows in by.values():
        k[rows] = len(rows)
        sd[rows] = float(np.std(labels[rows]))
    return k, sd


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(data, config, output_dir, dataset_name, **kwargs):
    """N37: the label-noise share of the model's error, on clean labels."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N37: NOISE BUDGET")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A noise variance in "
              f"label units is not defined for class labels.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    run_proximity = not kwargs.get('skip_proximity', False)
    run_simex = not kwargs.get('skip_simex', False)
    simex_lambdas = tuple(float(x) for x in
                          (kwargs.get('simex_lambdas') or SIMEX_LAMBDAS))
    simex_draws = tuple(int(x) for x in
                        (SIMEX_DRAWS if kwargs.get('simex_draws') is None
                         else kwargs['simex_draws']))
    k_max = int(kwargs.get('k_max') or K_MAX)

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n37: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    noise_groups = noise_groups_for(data)[0]
    rep_groups = build_replicate_groups(data)
    base_binary = is_binary(reps[base_rep_name])
    if rep_groups is not None:
        row_count, row_spread = _molecule_count_and_spread(labels, rep_groups)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'check_condition': CHECK_CONDITION,
        'dose_units': noise_dose_units(dataset_name),
        'models': downstream_models,
        'base_rep': base_rep_name,
        'base_distance': ('tanimoto' if base_binary
                          else 'euclidean_on_standardised_features'),
        'k_max': k_max,
        'simex_lambdas': list(simex_lambdas),
        'simex_draws': list(simex_draws),
        'hill_share': HILL_SHARE,
        'proximity_row_cap': int(PROX_SMOOTH_MAX_ROWS),
        'has_repeat_measurements': rep_groups is not None,
        'gamma_note': (
            'intercept of gamma_k on delta_k, k = 1..k_max nearest neighbours; '
            'label units squared. An upper bound on the label-noise variance: '
            'it also holds activity differences the features cannot resolve.'),
        'simex_note': (
            'extra noise at lambda * the fold base Gamma intercept, scored on '
            'the original held-out labels; the hybrid features are built once '
            'per fold from the clean training labels and held fixed across '
            'lambda, so the curve is the model sensitivity, not the feature '
            'choice sensitivity.'),
        'spread_note': 'R^2 and n*R^2 of r^2 on the inputs; no p-value is computed.',
        'no_pruning_note': ('the report proposes stopping noise work where the '
                            'noise share is under 0.1; that rule is not applied '
                            'here, the share is written.'),
    }
    if rep_groups is not None:
        results['repeat_noise_variance_whole_dataset'] = direct_noise_variance(
            measurements_by_molecule(labels, rep_groups))

    sample_X = reps[base_rep_name]
    splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))

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
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
            } for r in rs}
            for m, rs in recipes_by_model.items() if rs})
    else:
        results['hybrid_unavailable_reason'] = (
            (f"the pool lookup RAISED {results['hybrid_lookup_error']}; the "
             f"pool was never read. Everything ran in the base representation.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from pool subset {POOL_SUBSET!r} for any '
             f'downstream model. Everything ran in the base representation.'))
        print("  no hybrid recipe; base representation only")
    results['spaces_run'] = list(spaces)

    passes = _levels_and_draws(dataset_name, levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['check_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} rows, {len(splits)} folds, models "
          f"{downstream_models}")
    print(f"  base representation {base_rep_name}, distance "
          f"{results['base_distance']}; check levels {results['check_levels']}")

    per_fold = []
    # {space_key: {model: {'res': [...], 'pred': [...], 'rows': [...]}}}
    oof = {}
    errors = {}
    t0 = time.time()

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        y_tr = labels[train_idx]
        y_va = labels[val_idx]
        g_tr = groups[train_idx] if groups is not None else None
        n_tr = len(train_idx)
        frec = {'fold': fold_i, 'training_rows': int(n_tr),
                'held_out_rows': int(len(val_idx)),
                'gamma': {}, 'check': [], 'model_error': {}, 'simex': {},
                'tails_per_fold': {}}
        if rep_groups is not None:
            frec['repeat_noise_variance_training'] = direct_noise_variance(
                measurements_by_molecule(y_tr, np.asarray(rep_groups)[train_idx]))
        if n_tr > MAX_PAIR_ROWS:
            frec['refused'] = (f'{n_tr} training rows is over MAX_PAIR_ROWS='
                               f'{MAX_PAIR_ROWS}; nothing was computed')
            per_fold.append(frec)
            print(f"    fold {fold_i}: {frec['refused']}")
            continue

        train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                      for k, v in reps.items()}
        val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                    for k, v in reps.items()}

        sc = StandardScaler()
        Xb_tr = sc.fit_transform(train_reps[base_rep_name])
        Xb_va = sc.transform(val_reps[base_rep_name])
        Db = (tanimoto_distance_matrix(train_reps[base_rep_name])
              if base_binary else euclidean_distance_matrix(Xb_tr))
        g_base = gamma_test(Db, y_tr, k_max)
        frec['gamma']['base'] = g_base
        s2_base = (g_base or {}).get('intercept', float('nan'))

        # The check: base neighbours are fixed, so the intercept should rise
        # by the realised injected variance.
        for level, draw in passes:
            try:
                spec = noise_spec_for(dataset_name, CHECK_CONDITION, level,
                                      noise_seed=draw, is_clf=False)
                y_n, _m, _p = inject_train_labels(
                    labels, spec, train_idx=train_idx, is_clf=False,
                    groups=noise_groups, dataset_name=dataset_name)
                y_n = np.asarray(y_n, dtype=float)
                gn = gamma_test(Db, y_n, k_max)
                frec['check'].append({
                    'level': level, 'draw': draw,
                    'intercept': (gn or {}).get('intercept'),
                    'realised_injected_variance': float(
                        np.mean((y_n - y_tr) ** 2))})
            except Exception as exc:
                errors[f'f{fold_i}|check|{level:g}|d{draw}'] = (
                    f'{type(exc).__name__}: {exc}')
        del Db

        if run_proximity:
            if n_tr > PROX_SMOOTH_MAX_ROWS:
                frec['gamma'][PROXIMITY_SPACE] = {
                    'refused': (f'{n_tr} training rows is over '
                                f'PROX_SMOOTH_MAX_ROWS={PROX_SMOOTH_MAX_ROWS}')}
            else:
                try:
                    from kirby.utils.noise_estimation import (
                        forest_leaf_ids, forest_proximity_distance)
                    Dp = forest_proximity_distance(
                        forest_leaf_ids(Xb_tr, y_tr, False))
                    frec['gamma'][PROXIMITY_SPACE] = gamma_test(Dp, y_tr, k_max)
                    del Dp
                except Exception as exc:
                    errors[f'f{fold_i}|{PROXIMITY_SPACE}'] = (
                        f'{type(exc).__name__}: {exc}')

        # {space_key: (X_tr, X_va, models)}
        cells = {'base': (Xb_tr, Xb_va, list(downstream_models))}
        for _sp in spaces:
            if _sp == 'base':
                continue
            for _bp_m, _mnames, _shareable in group_models_by_recipe(
                    {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                     for _m in downstream_models}, downstream_models):
                X_tr_h, X_va_h = hybrid_for_fold(
                    train_reps, val_reps, y_tr, _bp_m, g_tr,
                    None if _shareable else model_factories[_mnames[0]](),
                    metric_fn, lower_is_better)
                _key = (_sp if list(_mnames) == list(downstream_models)
                        else f'{_sp}/{_mnames[0]}')
                if X_tr_h is None:
                    errors[f'f{fold_i}|{_key}|build'] = (
                        'the hybrid could not be built')
                    continue
                sh = StandardScaler()
                Xh_tr = sh.fit_transform(X_tr_h)
                Xh_va = sh.transform(X_va_h)
                frec['gamma'][_key] = gamma_test(
                    euclidean_distance_matrix(Xh_tr), y_tr, k_max)
                cells[_key] = (Xh_tr, Xh_va, list(_mnames))

        for key, (X_tr_s, X_va_s, cell_models) in cells.items():
            for mname in cell_models:
                fac = model_factories[mname]
                try:
                    pred = _fit_predict(fac, X_tr_s, y_tr, X_va_s, metric_name)
                except Exception as exc:
                    errors[f'f{fold_i}|{key}|{mname}|fit'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                res = y_va - pred
                mse = float(np.mean(res ** 2))
                row = {'metric': float(metric_fn(y_va, pred)), 'mse': mse,
                       'gamma_base_over_mse': (s2_base / mse
                                               if mse > 0 and np.isfinite(s2_base)
                                               else None)}
                gk = frec['gamma'].get(key)
                if isinstance(gk, dict) and 'intercept' in gk:
                    row['gamma_this_space_over_mse'] = (
                        gk['intercept'] / mse if mse > 0 else None)
                if rep_groups is not None:
                    s2e = frec['repeat_noise_variance_training']['s2e']
                    row['repeat_variance_over_mse'] = (
                        s2e / mse if mse > 0 and np.isfinite(s2e) else None)
                frec['model_error'].setdefault(key, {})[mname] = row
                frec['tails_per_fold'].setdefault(key, {})[mname] = \
                    tail_diagnostics(res, fit_t=False)
                o = oof.setdefault(key, {}).setdefault(
                    mname, {'res': [], 'pred': [], 'rows': []})
                o['res'].append(res)
                o['pred'].append(pred)
                o['rows'].append(np.asarray(val_idx))

                if run_simex and np.isfinite(s2_base) and s2_base > 0:
                    curve_l, curve_m, curve_metric = [0.0], [mse], [row['metric']]
                    per_draw = {}
                    for lam in simex_lambdas:
                        ms, mets = [], []
                        for d in simex_draws:
                            rng = np.random.RandomState(_seed(
                                dataset_name, fold_i, lam, d))
                            y_x = y_tr + rng.normal(size=n_tr) * np.sqrt(
                                lam * s2_base)
                            try:
                                p_x = _fit_predict(fac, X_tr_s, y_x, X_va_s,
                                                   metric_name)
                            except Exception as exc:
                                errors[f'f{fold_i}|{key}|{mname}|simex|'
                                       f'{lam:g}|d{d}'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            ms.append(float(np.mean((y_va - p_x) ** 2)))
                            mets.append(float(metric_fn(y_va, p_x)))
                        per_draw[f'{lam:g}'] = {'mse': ms, 'metric': mets}
                        if ms:
                            curve_l.append(lam)
                            curve_m.append(float(np.mean(ms)))
                            curve_metric.append(float(np.mean(mets)))
                    ex = simex_extrapolate(curve_l, curve_m)
                    ex['metric_curve'] = curve_metric
                    ex['per_draw'] = per_draw
                    ex['s2_used'] = float(s2_base)
                    frec['simex'].setdefault(key, {})[mname] = ex

        per_fold.append(frec)
        gb = frec['gamma'].get('base') or {}
        print(f"    fold {fold_i}: {n_tr} training rows, base Gamma intercept "
              f"{gb.get('intercept', float('nan')):.4f} (label variance "
              f"{gb.get('var_y', float('nan')):.4f}) [{time.time() - t0:.0f}s]")

    results['per_fold'] = convert_numpy(per_fold)

    # Out-of-fold residuals: every held-out molecule once.
    oof_block = {}
    for key, by_m in oof.items():
        for mname, o in by_m.items():
            res = np.concatenate(o['res'])
            pred = np.concatenate(o['pred'])
            rows = np.concatenate(o['rows'])
            rec = {'tails': tail_diagnostics(res),
                   'spread_on_prediction': spread_regression(res, pred[:, None]),
                   'mse': float(np.mean(res ** 2))}
            if rep_groups is not None:
                rec['spread_on_prediction_count_and_spread'] = spread_regression(
                    res, np.column_stack([pred, row_count[rows],
                                          row_spread[rows]]))
            oof_block.setdefault(key, {})[mname] = rec
    results['out_of_fold'] = convert_numpy(oof_block)
    if errors:
        results['_errors'] = errors
    results['check'] = convert_numpy(_check(per_fold))
    results['summary'] = convert_numpy(_summarise(per_fold, oof_block))
    _print_summary(results, metric_name)
    return results


def _check(per_fold):
    """Base intercept rise over level 0 beside the realised injected variance."""
    out = {}
    for f in per_fold:
        rows = f.get('check') or []
        ref = {r['draw']: r['intercept'] for r in rows if r['level'] == 0.0}
        for r in rows:
            r0 = ref.get(r['draw'])
            if r0 is None or r['intercept'] is None:
                continue
            lv = out.setdefault(f"{r['level']:g}", {
                'folds': [], 'intercept_rise': [],
                'realised_injected_variance': []})
            lv['folds'].append(f['fold'])
            lv['intercept_rise'].append(r['intercept'] - r0)
            lv['realised_injected_variance'].append(
                r['realised_injected_variance'])
    for lv in out.values():
        for k in ('intercept_rise', 'realised_injected_variance'):
            v = np.asarray(lv[k], dtype=float)
            lv[f'{k}_mean'] = float(v.mean()) if v.size else float('nan')
    return out


def _vals(xs):
    v = np.asarray([x for x in xs if x is not None], dtype=float)
    return v[np.isfinite(v)]


def _summarise(per_fold, oof_block):
    """Per space and model: the fold values side by side, and their mean."""
    out = {'gamma_intercept': {}, 'noise_share_of_mse': {}, 'simex': {},
           'tails': {}}
    for f in per_fold:
        for sp, g in (f.get('gamma') or {}).items():
            if isinstance(g, dict) and 'intercept' in g:
                out['gamma_intercept'].setdefault(sp, []).append(g['intercept'])
        for sp, by_m in (f.get('model_error') or {}).items():
            for m, row in by_m.items():
                (out['noise_share_of_mse'].setdefault(sp, {})
                 .setdefault(m, []).append(row.get('gamma_base_over_mse')))
        for sp, by_m in (f.get('simex') or {}).items():
            for m, ex in by_m.items():
                d = (ex.get('quadratic') or {}).get('drop_share_of_plain')
                dl = (ex.get('linear') or {}).get('drop_share_of_plain')
                s = out['simex'].setdefault(sp, {}).setdefault(
                    m, {'linear_drop_share': [], 'quadratic_drop_share': []})
                s['linear_drop_share'].append(dl)
                s['quadratic_drop_share'].append(d)
    for sp, v in list(out['gamma_intercept'].items()):
        a = _vals(v)
        out['gamma_intercept'][sp] = {'per_fold': v, 'mean': (
            float(a.mean()) if a.size else None)}
    for sp, by_m in out['noise_share_of_mse'].items():
        for m, v in list(by_m.items()):
            a = _vals(v)
            by_m[m] = {'per_fold': v,
                       'mean': float(a.mean()) if a.size else None}
    for sp, by_m in out['simex'].items():
        for m, s in by_m.items():
            for k in list(s):
                a = _vals(s[k])
                s[f'{k}_mean'] = float(a.mean()) if a.size else None
    for sp, by_m in oof_block.items():
        for m, rec in by_m.items():
            t = rec['tails']
            out['tails'].setdefault(sp, {})[m] = {
                k: t.get(k) for k in ('sd_over_mad_sd', 'excess_kurtosis',
                                      'student_t_nu')}
            out['tails'][sp][m]['hill_alpha'] = (t.get('hill') or {}).get('alpha')
            out['tails'][sp][m]['spread_r_squared'] = (
                rec['spread_on_prediction'].get('r_squared'))
    return out


def _fmt(x, p=4):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.{p}f}'


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print("CHECK: rise of the base Gamma intercept over level 0, beside the "
          "realised injected variance (label units squared)")
    for lvl in sorted(results['check'], key=float):
        r = results['check'][lvl]
        print(f"  level {lvl:>5s}: rise {r['intercept_rise_mean']:+.4f}, "
              f"injected {r['realised_injected_variance_mean']:.4f} "
              f"({len(r['folds'])} fold draws)")
    print("\nGAMMA INTERCEPT per space (label units squared; upper bound on "
          "the noise variance), mean over folds:")
    for sp, g in s['gamma_intercept'].items():
        print(f"  {sp:28s} {_fmt(g['mean'])}")
    print("\nBASE GAMMA INTERCEPT / held-out MSE, mean over folds:")
    for sp, by_m in s['noise_share_of_mse'].items():
        for m, v in by_m.items():
            print(f"  {sp}/{m}: {_fmt(v['mean'], 3)}")
    if s['simex']:
        print("\nSIMEX: share of held-out MSE gone at lambda = -1 "
              "(linear, quadratic), mean over folds:")
        for sp, by_m in s['simex'].items():
            for m, v in by_m.items():
                print(f"  {sp}/{m}: {_fmt(v.get('linear_drop_share_mean'), 3)}, "
                      f"{_fmt(v.get('quadratic_drop_share_mean'), 3)}")
    print("\nOUT-OF-FOLD RESIDUALS: SD/MAD-SD, excess kurtosis, t nu, Hill "
          "alpha, R^2 of r^2 on prediction")
    for sp, by_m in s['tails'].items():
        for m, t in by_m.items():
            print(f"  {sp}/{m}: {_fmt(t['sd_over_mad_sd'], 2)}, "
                  f"{_fmt(t['excess_kurtosis'], 2)}, "
                  f"{_fmt(t['student_t_nu'], 1)}, {_fmt(t['hill_alpha'], 2)}, "
                  f"{_fmt(t['spread_r_squared'], 3)}")


def main():
    parser = create_experiment_parser(
        'N37: the label-noise share of the model error, on clean labels')
    parser.add_argument('--levels', type=str, default=None,
                        help="Comma-separated gaussian levels for the check. "
                             "Default: the dataset's level grid")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated check replicates (default 0)')
    parser.add_argument('--simex-lambdas', type=str, default=None,
                        help='Comma-separated SIMEX multipliers (default 0.5,1,2)')
    parser.add_argument('--simex-draws', type=str, default=None,
                        help='Comma-separated SIMEX draws (default 0,1,2)')
    parser.add_argument('--k-max', type=int, default=None,
                        help=f'Neighbours in the Gamma test (default {K_MAX})')
    parser.add_argument('--skip-proximity', action='store_true',
                        help='Do not compute the forest proximity')
    parser.add_argument('--skip-simex', action='store_true',
                        help='Do not run the SIMEX curve')
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
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.simex_lambdas:
        kw['simex_lambdas'] = [float(x) for x in args.simex_lambdas.split(',')]
    if args.simex_draws:
        kw['simex_draws'] = [int(x) for x in args.simex_draws.split(',')]
    if args.k_max:
        kw['k_max'] = args.k_max
    if args.skip_proximity:
        kw['skip_proximity'] = True
    if args.skip_simex:
        kw['skip_simex'] = True
    if args.base_rep:
        kw['base_rep'] = args.base_rep

    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, EXP_ID, args.dataset,
                            filename_suffix=args.tag or None)


if __name__ == '__main__':
    main()
