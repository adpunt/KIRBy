#!/usr/bin/env python3
"""N26: how large is the label noise, measured from repeat measurements?

Paper: Noise.

THE QUESTION. Every cleaning method in this suite spends model error to remove
label error. That trade only pays when the label error is larger than the model
error. Nothing in the suite has measured the label error on real labels. This
file measures it three ways and puts the model error beside it.

WHERE IT COMES FROM. The research report of 2026-09-24 (items D1, D2 and D3)
proposed these three measurements. It described the data as assets valued by
one to four sources. Here a source is a ChEMBL record, and a molecule's count
is how many records it has.

NOTHING IS INJECTED IN THE MAIN MEASUREMENT. The training label of a molecule is
the mean of ALL its measurements. Its error variance is therefore the
per-measurement noise variance divided by its measurement count.

THE DATASETS. Only a loader that keeps repeat measurements can be used:
`herg_ic50` (one row per ChEMBL record) and `bindingdb_kd` (one row per
measured drug-protein pair). Any other dataset writes
{'skipped': 'no_replicate_structure'}, the same as n19.

WHAT IS COMPUTED.

  1. ERROR BY MEASUREMENT COUNT. For each molecule, the out-of-fold prediction
     from each downstream model in each feature space. The residual is the mean
     label minus that prediction, in label units. Molecules are binned by count
     (1, 2, 3, 4, 5 or more). Each bin reports its molecule count, the mean
     residual, the mean absolute residual and the mean squared residual.

  2. NOISE VARIANCE FROM THE COUNT. If a molecule's mean label carries noise of
     variance s2e / k, then its expected squared residual is s2u + s2e / k.
     s2u is the model's own error variance. Regressing the squared residual on
     1/k gives s2e as the slope and s2u as the intercept, both in label units
     squared. Three fits:
       ols      ordinary least squares over every molecule
       binned   one point per distinct count, the mean squared residual of the
                molecules with that count, weighted by how many there are
       gamma    a gamma GLM with an identity link, because a squared residual
                is skewed. Recorded as failed when statsmodels does not converge.
     Beside them, s2e measured DIRECTLY: the pooled within-molecule variance of
     the repeat measurements, over molecules measured two or more times, each
     molecule weighted by its count minus one.

  3. RELIABILITY AND THE THRESHOLD. With sL2 = s2e * mean(1/k), the average
     noise variance of a training label:
       reliability     1 - sL2 / Var(mean label). The share of the label
                       variance that is not noise.
       r2_over_reliability  the model's out-of-fold R2 divided by the
                       reliability, per reliability estimate (added
                       2026-09-25): the R2 corrected for attenuation.
       r2_threshold    1 - sL2 / Var(true label), with
                       Var(true label) = Var(mean label) - sL2. A model whose
                       R2 is below this has an out-of-fold error larger than the
                       average label noise.
     Each is computed from the direct s2e and from each fitted s2e. The model's
     own out-of-fold R2 against the mean label is written beside them.
     Two estimates that need no model at all:
       split_half      on molecules measured two or more times, two different
                       measurements are drawn at random. The Pearson
                       correlation across molecules is the reliability of ONE
                       measurement. It is projected to k measurements with
                       Spearman-Brown, k r / (1 + (k - 1) r). Repeated with
                       different draws, and every repeat is written.
       icc             the one-way random-effects intraclass correlation, over
                       the same molecules, with the unequal-count correction.

A CHECK OF THE ESTIMATOR, NOT A RESULT. The report states it found no paper
using the 1/k regression and asks that it be checked first. So noise of known
variance s2_inj / k is added to each molecule's mean label. The out-of-fold
prediction is refitted for the first downstream model in its rank-1 single
representation (space base_1), and the ols slope is refitted. What is written is the change in
slope against the injected s2_inj. If the estimator works, the two are close.
This arm says nothing about the real noise.

WHAT THIS CANNOT SEPARATE. A bias shared by every measurement of a molecule --
the same paper reporting the same number twice, or two papers copying a third --
does not shrink with the count. It lands in the intercept, not the slope. So s2e
here is the part of the noise that differs between records. The count of
molecules whose repeats are all identical is written, because those repeats
push the direct s2e toward zero.

WHERE THE HYBRID COMES IN. Each downstream model's out-of-fold prediction is
made in its top 2 single representations -- spaces base_1 and base_2, ranked
by that model's own score in the stored search (_shared.noise_base_reps_by_model,
owner's decision 2026-09-25) -- and in one hybrid per family of stored search,
each recipe the one the pool picked FOR that model. A model the pool holds no
search for has neither. The hybrid is rebuilt per
fold on the training molecules' mean labels, as n19 and n23 do. The model error
s2u depends on the feature space. The noise variance s2e should not. How far the
fitted slopes move between spaces is written as a spread.

THE HEADLINE IS NOT A MEAN. Every (space, model) pair writes its own three
fitted slopes. Nothing is averaged across models or across spaces.

Usage:
    python n26_noise_variance_from_repeats.py --dataset herg_ic50 --mode full
    python n26_noise_variance_from_repeats.py --dataset bindingdb_kd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n17_replicate_removal import build_entities
from experiments.noise.n19_measurement_count import measurement_stats
from experiments.noise.noise_helpers import (
    hybrid_for_fold, pool_recipe_per_downstream_model, group_models_by_recipe,
)

# Which searches the recipe may come from. Same as n19, n23 and n24.
POOL_SUBSET = 'predictive_and_similarity'

# One feature space per family of stored search.
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}

# Measurement-count bins for the error-by-count table. The last one is open.
COUNT_BINS = (1, 2, 3, 4, 5)

# The binned fit uses a count only when this many molecules carry it. One
# molecule measured nine times is one squared residual, not a bin mean.
MIN_MOLECULES_PER_BIN = 5

# How many times two measurements are drawn for the split-half correlation.
N_SPLIT_HALF_REPEATS = 20

# The noise variances, in label units squared, the estimator check injects. A
# molecule measured k times gets noise of variance s2_inj / k.
INJECTED_VARIANCES = (0.1, 0.25, 0.5)

# The counts the split-half reliability is projected to.
PROJECT_TO_COUNTS = (1, 2, 3, 4, 5)


def measurements_by_molecule(labels, replicate_groups):
    """Every molecule's own measurements, in `measurement_stats`' order.

    That order is molecules sorted by the row of their first measurement. It is
    rebuilt here the same way, and the caller checks the counts agree.
    """
    labels = np.asarray(labels, dtype=np.float64)
    order = {}
    for i, g in enumerate(np.asarray(replicate_groups)):
        order.setdefault(g, []).append(i)
    gids = sorted(order, key=lambda g: order[g][0])
    return [labels[np.asarray(order[g], dtype=np.int64)] for g in gids]


def direct_noise_variance(values_per_molecule):
    """The pooled within-molecule variance of the repeats, label units squared.

    Over molecules measured two or more times. Each molecule's ddof=1 variance
    is weighted by its count minus one, which is the pooled variance of a
    one-way layout.
    """
    num, den, n_mol, n_identical = 0.0, 0, 0, 0
    for v in values_per_molecule:
        if len(v) < 2:
            continue
        n_mol += 1
        var = float(np.var(v, ddof=1))
        if var == 0.0:
            n_identical += 1
        num += (len(v) - 1) * var
        den += len(v) - 1
    return {
        's2e': (num / den) if den > 0 else float('nan'),
        'molecules_used': n_mol,
        'degrees_of_freedom': int(den),
        'molecules_whose_repeats_are_all_identical': n_identical,
    }


def error_by_count(residual, k):
    """One row per count bin: molecules, mean residual, mean |residual|, mean r2."""
    out = {}
    for i, lo in enumerate(COUNT_BINS):
        last = i == len(COUNT_BINS) - 1
        sel = (k >= lo) if last else (k == lo)
        key = f'{lo}+' if last else str(lo)
        r = residual[sel]
        out[key] = {
            'molecules': int(sel.sum()),
            'mean_residual': float(np.mean(r)) if r.size else float('nan'),
            'mean_abs_residual': (float(np.mean(np.abs(r))) if r.size
                                  else float('nan')),
            'mean_squared_residual': (float(np.mean(r ** 2)) if r.size
                                      else float('nan')),
        }
    return out


def fit_ols(r2, inv_k):
    """Squared residual on 1/k by least squares. (slope, intercept)."""
    if np.unique(inv_k).size < 2:
        return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                'molecules': int(r2.size),
                'failed': 'every molecule has the same measurement count'}
    slope, intercept = np.polyfit(inv_k, r2, 1)
    return {'slope_s2e': float(slope), 'intercept_s2u': float(intercept),
            'molecules': int(r2.size), 'failed': None}


def fit_binned(r2, k, min_per_bin=MIN_MOLECULES_PER_BIN):
    """One point per distinct count, weighted by the molecules carrying it."""
    xs, ys, ws, used = [], [], [], {}
    for c in np.unique(k):
        sel = k == c
        if int(sel.sum()) < min_per_bin:
            continue
        xs.append(1.0 / float(c))
        ys.append(float(np.mean(r2[sel])))
        ws.append(float(sel.sum()))
        used[int(c)] = int(sel.sum())
    if len(xs) < 2:
        return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                'molecules_by_count_used': used,
                'failed': (f'fewer than 2 measurement counts carry '
                           f'{min_per_bin} or more molecules')}
    # np.polyfit weights multiply the residual, so the square root of the
    # molecule count gives each bin a squared-error weight of its count.
    slope, intercept = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
    return {'slope_s2e': float(slope), 'intercept_s2u': float(intercept),
            'molecules_by_count_used': used, 'failed': None}


def fit_gamma(r2, inv_k):
    """Gamma GLM with identity link. Recorded as failed rather than raised."""
    if np.unique(inv_k).size < 2:
        return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                'failed': 'every molecule has the same measurement count'}
    try:
        import statsmodels.api as sm
        y = np.maximum(r2, 1e-12)
        X = sm.add_constant(inv_k)
        fam = sm.families.Gamma(link=sm.families.links.Identity())
        # The identity link can step outside the positive mean the gamma
        # family needs. Starting from the least-squares line keeps the first
        # step inside it when the data allow.
        start = np.polyfit(inv_k, y, 1)[::-1]
        res = sm.GLM(y, X, family=fam).fit(start_params=start, maxiter=200)
        if not res.converged:
            return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                    'failed': 'statsmodels did not converge'}
        intercept, slope = res.params
        return {'slope_s2e': float(slope), 'intercept_s2u': float(intercept),
                'failed': None}
    except Exception as exc:
        return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                'failed': f'{type(exc).__name__}: {exc}'}


def _ratio(a, b):
    """a / b, NaN when either is not finite or b is not positive."""
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return float('nan')
    return float(a / b)


def reliability_from(s2e, mean_inv_k, var_y):
    """Reliability and the R2 threshold for one s2e, in the units they need."""
    if not np.isfinite(s2e):
        return {'s2e': float('nan'), 'label_noise_variance_sL2': float('nan'),
                'reliability': float('nan'), 'r2_threshold': float('nan')}
    sL2 = s2e * mean_inv_k
    var_true = var_y - sL2
    return {
        's2e': float(s2e),
        'label_noise_variance_sL2': float(sL2),
        'reliability': float(1.0 - sL2 / var_y) if var_y > 0 else float('nan'),
        'r2_threshold': (float(1.0 - sL2 / var_true) if var_true > 0
                         else float('nan')),
        'variance_of_the_true_label': float(var_true),
    }


def split_half(values_per_molecule, n_repeats, seed=0,
               project_to=PROJECT_TO_COUNTS):
    """Two measurements drawn per molecule, correlated across molecules.

    Over molecules measured two or more times. Returns one entry per repeat.
    The correlation is the reliability of ONE measurement, projected to k
    measurements by Spearman-Brown.
    """
    multi = [v for v in values_per_molecule if len(v) >= 2]
    if len(multi) < 3:
        return {'molecules': len(multi), 'repeats': [],
                'failed': 'fewer than 3 molecules measured twice or more'}
    rng = np.random.default_rng(seed)
    repeats = []
    for _ in range(int(n_repeats)):
        a = np.empty(len(multi))
        b = np.empty(len(multi))
        for j, v in enumerate(multi):
            i1, i2 = rng.choice(len(v), 2, replace=False)
            a[j], b[j] = v[i1], v[i2]
        if np.std(a) == 0 or np.std(b) == 0:
            repeats.append({'r_one_measurement': float('nan')})
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        repeats.append({
            'r_one_measurement': r,
            'spearman_brown_projected': {
                str(c): float(c * r / (1.0 + (c - 1.0) * r))
                for c in project_to},
        })
    return {'molecules': len(multi), 'repeats': repeats, 'failed': None}


def one_way_icc(values_per_molecule):
    """ICC(1) over molecules measured two or more times, unequal-count form."""
    multi = [np.asarray(v, dtype=float) for v in values_per_molecule
             if len(v) >= 2]
    g = len(multi)
    if g < 2:
        return {'icc': float('nan'), 'molecules': g,
                'failed': 'fewer than 2 molecules measured twice or more'}
    ks = np.array([len(v) for v in multi], dtype=float)
    N = float(ks.sum())
    grand = float(np.concatenate(multi).mean())
    ssb = float(sum(len(v) * (v.mean() - grand) ** 2 for v in multi))
    ssw = float(sum(((v - v.mean()) ** 2).sum() for v in multi))
    msb = ssb / (g - 1)
    msw = ssw / (N - g)
    n0 = (N - float((ks ** 2).sum()) / N) / (g - 1)
    denom = msb + (n0 - 1.0) * msw
    return {'icc': float((msb - msw) / denom) if denom > 0 else float('nan'),
            'molecules': g, 'measurements': int(N),
            'mean_square_between': msb, 'mean_square_within': msw,
            'effective_count_n0': float(n0), 'failed': None}


def _fit_predict(factory, X_tr, y_tr, X_va):
    m = factory()
    m.fit(X_tr, y_tr)
    return np.asarray(m.predict(X_va), dtype=np.float64).ravel()


def _oof_base_one_model(factory, X_base, y, splits):
    """Out-of-fold prediction in one single representation, one model."""
    oof = np.full(len(y), np.nan)
    for tr, va in splits:
        sc = StandardScaler()
        oof[va] = _fit_predict(factory, sc.fit_transform(X_base[tr]), y[tr],
                               sc.transform(X_base[va]))
    return oof


def run(data, config, output_dir, dataset_name, **kwargs):
    """N26: label-noise variance, reliability and the R2 threshold, from repeats."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N26: HOW LARGE IS THE LABEL NOISE, FROM REPEATS")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    reps = data['reps']
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A repeat of a class "
              f"label has no variance in label units.")
        return {'skipped': 'not_regression', 'task_type': task_type}

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        print(f"  SKIPPED: {dataset_name}'s loader keeps no repeat "
              f"measurements, so no molecule has a measurement count.")
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    n_split_half = int(kwargs.get('n_split_half_repeats', N_SPLIT_HALF_REPEATS))
    injected = [float(x) for x in
                (kwargs.get('injected_variances') or INJECTED_VARIANCES)]

    stats = measurement_stats(labels, rep_groups)
    probe = build_entities(labels, rep_groups, seed=0)
    if not np.array_equal(probe['n_measurements'], stats['n_measurements']):
        raise RuntimeError(
            "n26: measurement_stats and build_entities disagree about how many "
            "times each molecule was measured, so they are not in the same "
            "order.")
    if not np.array_equal(probe['row_index'], stats['row_index']):
        raise RuntimeError(
            "n26: measurement_stats and build_entities disagree about which "
            "row represents each molecule.")
    values = measurements_by_molecule(labels, rep_groups)
    if not np.array_equal([len(v) for v in values], stats['n_measurements']):
        raise RuntimeError(
            "n26: the per-molecule measurement lists are not in "
            "measurement_stats' order.")

    k = stats['n_measurements'].astype(np.float64)
    inv_k = 1.0 / k
    y = stats['mean_all'].astype(np.float64)
    n_ent = len(y)
    var_y = float(np.var(y, ddof=1))
    mean_inv_k = float(np.mean(inv_k))

    row_groups = data.get('groups')
    if row_groups is None:
        print("  SKIPPED: the loader gives no fold groups, and random folds "
              "are not used in this suite.")
        return {'skipped': 'no_fold_groups', 'dataset': dataset_name}
    ent_groups = np.asarray(row_groups)[stats['row_index']]
    ent_reps = {name: np.asarray(mat[stats['row_index']], dtype=np.float64)
                for name, mat in reps.items()}

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=n_ent)
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n26: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')

    direct = direct_noise_variance(values)
    counts = {int(c): int((stats['n_measurements'] >= c).sum())
              for c in (2, 3, 4, 5)}
    print(f"  {len(labels)} measurements over {n_ent} molecules")
    for c, v in counts.items():
        print(f"    molecules measured {c}+ times: {v} of {n_ent}")
    print(f"  molecules whose repeats are all identical: "
          f"{direct['molecules_whose_repeats_are_all_identical']} of "
          f"{direct['molecules_used']} measured twice or more")
    print(f"  variance of the mean label: {var_y:.4f} label units squared; "
          f"mean of 1/count: {mean_inv_k:.4f}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none in the main measurement',
        'training_label': 'the mean of all of a molecule\'s measurements',
        'residual_units': 'label units; squared residuals in label units squared',
        'models': downstream_models,
        'feasibility': {
            'n_measurements': int(len(labels)),
            'n_molecules': n_ent,
            'molecules_by_min_measurements': counts,
        },
        'variance_of_the_mean_label': var_y,
        'mean_of_one_over_count': mean_inv_k,
        'direct': {
            **direct,
            'reliability': reliability_from(direct['s2e'], mean_inv_k, var_y),
            'note': ('s2e measured from the repeats alone, with no model. '
                     'Repeats that are all identical push it toward zero.'),
        },
        'split_half': split_half(values, n_split_half, seed=0),
        'icc': one_way_icc(values),
        'shared_bias_note': (
            'A bias shared by every measurement of a molecule does not shrink '
            'with the count. The 1/count regression puts it in the intercept, '
            'not the slope, so each fitted s2e is only the part of the noise '
            'that differs between records.'),
        'per_space': {},
    }

    # ── The hybrid recipes, one per downstream model. n23's block. ──
    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, row_groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    _raised = [d['lookup_error'] for d in no_recipe.values()
               if d['lookup_error']]
    results['hybrid_lookup_error'] = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}

    recipe_for_space_by_model, hybrid_unavailable = {}, {}
    ordered_spaces, recipe_record = [], {}
    for _m in downstream_models:
        _recipes, _unavail = pool_recipes_by_family(
            by_model[_m]['meta'] if _m in by_model else None,
            SPACE_FOR_FAMILY, eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipe_for_space_by_model[_m] = {r['space']: r['blueprint']
                                         for r in _recipes}
        for r in _recipes:
            if r['space'] not in ordered_spaces:
                ordered_spaces.append(r['space'])
            recipe_record.setdefault(_m, {})[r['space']] = {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
            }
        for _sp, _why in _unavail.items():
            hybrid_unavailable.setdefault(_sp, {})[_m] = _why

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

    spaces = base_spaces + tuple(ordered_spaces)
    results['spaces_run'] = list(spaces)
    results['hybrid_available'] = len(ordered_spaces) > 0
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    results['hybrid_recipe_by_model'] = convert_numpy(recipe_record)
    results['features_note'] = (
        'Each hybrid is rebuilt per fold on the training molecules\' mean '
        'labels, from the recipe the pool picked for that downstream model. '
        'Each single-representation space (base_1, base_2) holds one matrix '
        'per distinct representation at that rank, shared by the models that '
        'rank it there.')
    print(f"  spaces {list(spaces)}")

    splits = list(cv_split(ent_groups, n_folds, next(iter(ent_reps.values())),
                           y, task_type='regression'))

    # oof[space][model] = one prediction per molecule
    oof = {s: {m: np.full(n_ent, np.nan) for m in downstream_models}
           for s in spaces}
    fold_scores = {s: {m: {} for m in downstream_models} for s in spaces}
    errors = {}
    t0 = time.time()

    for fold_i, (tr, va) in enumerate(splits):
        fold_key = f'f{fold_i}'
        g_tr = ent_groups[tr]
        train_reps = {n: v[tr] for n, v in ent_reps.items()}
        val_reps = {n: v[va] for n, v in ent_reps.items()}

        # {space: [(X_tr, X_va, models scored on it)]}
        fold_spaces = {}
        for _k in range(1, NOISE_BASE_REP_COUNT + 1):
            _bsp = f'base_{_k}'
            if _bsp not in spaces:
                continue
            _bcells = []
            for _rn, _ms in base_rank_groups(base_by_model, _k,
                                             downstream_models).items():
                sc = StandardScaler()
                _bcells.append((sc.fit_transform(ent_reps[_rn][tr]),
                                sc.transform(ent_reps[_rn][va]), list(_ms)))
            fold_spaces[_bsp] = _bcells
        for _sp in spaces:
            if _sp.startswith('base_'):
                continue
            _cells = []
            for _bp, _mnames, _shareable in group_models_by_recipe(
                    {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                     for _m in downstream_models}, downstream_models):
                X_tr_h, X_va_h = hybrid_for_fold(
                    train_reps, val_reps, y[tr], _bp, g_tr,
                    None if _shareable else model_factories[_mnames[0]](),
                    metric_fn, lower_is_better)
                if X_tr_h is None:
                    errors[f'{_sp}/{_mnames[0]}|{fold_key}|build'] = (
                        'the hybrid could not be built on this fold')
                    continue
                sh = StandardScaler()
                _cells.append((sh.fit_transform(X_tr_h), sh.transform(X_va_h),
                               list(_mnames)))
            if _cells:
                fold_spaces[_sp] = _cells

        for space in spaces:
            for X_tr_s, X_va_s, cell_models in fold_spaces.get(space, ()):
                for mname in cell_models:
                    try:
                        pred = _fit_predict(model_factories[mname], X_tr_s,
                                            y[tr], X_va_s)
                    except Exception as exc:
                        errors[f'{space}|{mname}|{fold_key}'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    oof[space][mname][va] = pred
                    fold_scores[space][mname][fold_key] = float(
                        metric_fn(y[va], pred))
        print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    # ── The three measurements, per (space, model) ──
    for space in spaces:
        srow = {}
        for mname in downstream_models:
            p = oof[space][mname]
            have = np.isfinite(p)
            if not have.any():
                continue
            res = y[have] - p[have]
            r2 = res ** 2
            kk, ik = k[have], inv_k[have]
            fits = {'ols': fit_ols(r2, ik),
                    'binned': fit_binned(r2, kk),
                    'gamma': fit_gamma(r2, ik)}
            var_here = float(np.var(y[have]))
            r2_model = (float(1.0 - np.mean(r2) / var_here) if var_here > 0
                        else float('nan'))
            rel_fit = {f: reliability_from(fits[f]['slope_s2e'], mean_inv_k, var_y)
                       for f in fits}
            srow[mname] = {
                'molecules_with_a_prediction': int(have.sum()),
                'fold_scores': fold_scores[space][mname],
                'model_r2_against_the_mean_label': r2_model,
                # ADDED 2026-09-25 (third research report, A6): the model's R2
                # divided by the reliability of the label it is scored on, so
                # datasets with noisier labels are not read as worse models.
                # One per reliability estimate; the reliability is on the
                # mean label, which is what the model is scored against.
                'r2_over_reliability': {
                    'direct': _ratio(r2_model, results['direct']['reliability']
                                     ['reliability']),
                    **{f: _ratio(r2_model, rel_fit[f]['reliability'])
                       for f in rel_fit}},
                'mean_squared_residual': float(np.mean(r2)),
                'error_by_count': error_by_count(res, kk),
                'fits': fits,
                'reliability_by_fit': rel_fit,
            }
        results['per_space'][space] = srow

    # How far each fit's slope moves between feature spaces, per model. The
    # noise variance is a property of the labels, so a large spread says the
    # slope is picking up model error.
    slope_spread = {}
    for mname in downstream_models:
        for f in ('ols', 'binned', 'gamma'):
            vals = [results['per_space'][s][mname]['fits'][f]['slope_s2e']
                    for s in spaces if mname in results['per_space'].get(s, {})]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                slope_spread.setdefault(mname, {})[f] = {
                    'spaces': len(vals), 'min': float(min(vals)),
                    'max': float(max(vals))}
    results['slope_across_spaces'] = slope_spread

    # ── The estimator check ──
    check_model = downstream_models[0]
    check_rep = (base_by_model.get(check_model) or [None])[0]
    base_oof = (oof['base_1'][check_model] if 'base_1' in oof
                else np.full(n_ent, np.nan))
    check = {'model': check_model, 'space': 'base_1', 'rep': check_rep,
             'what_it_is': ('a check of the 1/count regression, not a '
                            'measurement of the real noise'),
             'per_injected_variance': {}}
    if check_rep is not None and np.isfinite(base_oof).all():
        X_base = ent_reps[check_rep]
        s0 = fit_ols((y - base_oof) ** 2, inv_k)['slope_s2e']
        check['slope_without_injection'] = s0
        rng = np.random.default_rng(12345)
        for s2 in injected:
            y_pert = y + rng.normal(0.0, np.sqrt(s2 * inv_k))
            try:
                p = _oof_base_one_model(model_factories[check_model], X_base,
                                        y_pert, splits)
                s1 = fit_ols((y_pert - p) ** 2, inv_k)['slope_s2e']
                check['per_injected_variance'][f'{s2:g}'] = {
                    'injected_s2': s2,
                    'slope_with_injection': s1,
                    'slope_change': float(s1 - s0),
                }
            except Exception as exc:
                errors[f'estimator_check|{s2:g}'] = (
                    f'{type(exc).__name__}: {exc}')
    else:
        check['failed'] = ('the first model has no out-of-fold prediction for '
                           'every molecule in its rank-1 single '
                           'representation'
                           + ('' if check_rep is not None else
                              f" -- it has none: "
                              f"{base_missing.get(check_model)}"))
    results['estimator_check'] = check

    if errors:
        results['_errors'] = errors
    results = convert_numpy(results)
    _print_summary(results)
    save_experiment_results(results, output_dir,
                            'n26_noise_variance_from_repeats', dataset_name,
                            filename_suffix=kwargs.get('tag'))
    return results


def _print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY (label units squared unless stated)")
    d = results['direct']
    rel = d['reliability']
    print(f"  s2e measured directly from repeats: {d['s2e']:.4f}, over "
          f"{d['molecules_used']} molecules measured twice or more")
    print(f"    reliability {rel['reliability']:.4f}, R2 threshold "
          f"{rel['r2_threshold']:.4f}")
    sh = results['split_half']
    rs = [r['r_one_measurement'] for r in sh['repeats']
          if np.isfinite(r['r_one_measurement'])]
    if rs:
        print(f"  split-half correlation of one measurement, {len(rs)} draws "
              f"over {sh['molecules']} molecules: {min(rs):.4f} to "
              f"{max(rs):.4f}")
    icc = results['icc']
    print(f"  one-way ICC over {icc['molecules']} molecules: {icc['icc']:.4f}")
    for space, srow in results['per_space'].items():
        for m, row in srow.items():
            f = row['fits']
            print(f"  {space}/{m}: out-of-fold R2 "
                  f"{row['model_r2_against_the_mean_label']:.4f}; slope s2e "
                  f"ols {f['ols']['slope_s2e']:.4f}, binned "
                  f"{f['binned']['slope_s2e']:.4f}, gamma "
                  f"{f['gamma']['slope_s2e']:.4f}; intercept s2u ols "
                  f"{f['ols']['intercept_s2u']:.4f}")
    chk = results['estimator_check']
    for key, row in chk.get('per_injected_variance', {}).items():
        print(f"  estimator check ({chk['model']}, base_1 {chk.get('rep')}): "
              f"injected {key}, "
              f"slope moved {row['slope_change']:+.4f}")


def main():
    parser = create_experiment_parser(
        'N26: label-noise variance, reliability and the R2 threshold, from '
        'repeat measurements')
    parser.add_argument('--n-split-half-repeats', type=int,
                        default=N_SPLIT_HALF_REPEATS,
                        help='How many times two measurements are drawn per '
                             'molecule for the split-half correlation')
    parser.add_argument('--injected-variances', type=str, default=None,
                        help='Comma-separated noise variances, label units '
                             'squared, for the estimator check (default '
                             f'{",".join(str(v) for v in INJECTED_VARIANCES)})')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {'n_split_half_repeats': args.n_split_half_repeats}
    if args.injected_variances:
        kw['injected_variances'] = [
            float(x) for x in args.injected_variances.split(',')]
    if args.tag:
        kw['tag'] = args.tag
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
