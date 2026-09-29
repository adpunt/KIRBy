#!/usr/bin/env python3
"""N27: what removing training molecules costs, and how much label error reaches the test error.

Paper: Noise.

THE QUESTION. Every cleanup in this suite either removes training molecules or
moves their labels. Two numbers decide whether either can pay, and neither has
been measured in the suite:

  1. what it costs to train on fewer molecules when every label is correct;
  2. how much of an error put into the training labels comes out in the
     held-out error.

This file measures both on the same outer folds. It cleans nothing.

THREE PARTS, ALL PAIRED ON THE SAME OUTER FOLDS.

  removal_cost   train on a random share of the training molecules, at shares
                 0.25, 0.5, 0.7, 0.8 and 0.9, 20 random draws of the molecules
                 at each share, every label correct. Each is paired against
                 training on all of them. A removal cleanup that drops a tenth
                 of the training molecules pays at least the 0.9 row, before
                 any benefit from which molecules it dropped. 0.25 and 0.5 were
                 added 2026-09-24 from a second research report (its D3): a
                 steep curve down to a quarter says the dataset is short of
                 molecules or features, and cleaning comes second.

  injected       gaussian noise at the dataset's level grid, put into the
                 training labels only. The injected variance is MEASURED on
                 each pass as the mean of (noisy label - correct label) squared
                 over the training molecules, in label units squared. The
                 held-out mean squared error minus the correct-label fit's
                 held-out mean squared error, divided by that variance, is how
                 much of each unit of injected variance reaches the test error.

  relabelled     the correct training labels moved a fraction alpha of the way
                 toward an out-of-fold prediction, alpha 0.25, 0.5 and 1.0. The
                 prediction is a LightGBM fitted on the other inner folds of the
                 training molecules, group-aware (`oof_predict`). The same ratio
                 is computed, dividing by the mean of (moved label - correct
                 label) squared.

The injected and relabelled ratios are the two numbers a relabelling cleanup is
trading between. A cleanup that moves a wrong label toward a prediction removes
some of the first kind of error and adds some of the second. The file reports
both ratios side by side, per downstream model and per feature space, with the
variance each was divided by. It does not say which is larger.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, as the
loader produced them, in every part. No moved label and no injected label is
ever scored against.

WHERE THE FEATURES COME FROM, PART BY PART. Each part says it on the result
file too.
  removal_cost and relabelled: each feature space is built ONCE per fold, from
    ALL the fold's training molecules at their correct labels. The molecules
    are removed, or the labels moved, after the features were chosen. So a
    removal row prices a smaller training set for the downstream model only,
    not for the feature choice.
  injected: each hybrid space is rebuilt on every pass from the NOISY training
    labels, the same way n24 builds it. The single-representation spaces do
    not depend on the labels.

THE SINGLE-REPRESENTATION COMPARISON is each downstream model's top 2 base
representations by that model's own score in the stored search -- spaces
base_1 and base_2 (_shared.noise_base_reps_by_model, owner's decision
2026-09-25). Which representations those are is written to
`base_reps_by_model`. A model the pool holds no search for has neither.

WHAT IS NOT DONE HERE. No random share is taken of the held-out molecules. The
share 1.0 row is fitted once per fold, not 20 times: it is the same training
set every time. The file does not compare the removal cost to any cleanup's
gain; n24's random-deletion arms and n17's floor arms do that at one share.

WHICH DATASETS. Any regression dataset with one label per molecule. The ones
this was written for are herg_ki, caco2, logd and flab_thermo_tresanco. A
classification label is refused: a variance of the label error is not defined
for a class.

THE HEADLINE IS PER DRAW AND PER PASS, NOT A MEAN. The removal part writes one
row per random draw of the molecules, with the best and the worst draw. The
propagation parts write one row per pass, and a pass is one (noise draw, fold)
for injected and one fold for relabelled.

ROWS, CACHES, CHECKPOINT (2026-09-25, plan sections 2-3). removal_cost and
relabelled change the training data, so each writes one D-versus-D' row per
fold (D = every molecule at its correct label, so score_D = score_clean).
injected adds noise and writes none. The accumulators are checkpointed per
fold; the scaled representations, hybrid builds, out-of-fold targets and the
fits of seeded models on correct and injected labels go through noise_cache.
controls.py's twins of relabelled run with KIRBY_NOISE_CONTROLS=1.

Usage:
    python n27_removal_cost_and_propagation.py --dataset herg_ki --mode full
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
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import noise_levels, noise_dose_units
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, hybrid_for_fold,
    pool_recipe_per_downstream_model, group_models_by_recipe,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.utils.noisy_labels import oof_predict
# The D-versus-D' rows, the per-fold checkpoint, the caches and the controls
# (2026-09-25, plan sections 2-3). They live in n23 for the reason given at
# the top of its shared block.
from experiments.noise import noise_cache as nc
from experiments.noise.controls import controls_enabled
from experiments.noise.n23_uncertainty_cleaning import (
    RowSink, PassCheckpoint, ensure_dprime_models, control_twins,
    as_cleanup_dict, cached_base_cell, cached_hybrid_cell, CACHEABLE_MODELS,
    NO_NEIGHBOURS,
)

# Shares of the training molecules kept in the removal part. 1.0 is the
# reference every other share is paired against.
SHARES = (0.25, 0.5, 0.7, 0.8, 0.9, 1.0)

# Random draws of which molecules are kept, per share below 1.0.
REPEATS = 20

# The condition the injected part puts into the training labels. Gaussian moves
# every training label by a draw of mean zero, which is the error the ratio is
# defined for.
CONDITION = 'gaussian'

# The replicate index of the injection. n07, n11, n13 and n24 use the same three.
NOISE_DRAWS = (0, 1, 2)

# How far each training label is moved toward its out-of-fold prediction.
ALPHAS = (0.25, 0.5, 1.0)

# Inner folds of the out-of-fold prediction. n19's OOF_SPLITS.
OOF_SPLITS = 5

# Which searches the recipe may come from. Same as n02, n07, n11, n12, n13, n17,
# n18, n19, n23 and n24.
POOL_SUBSET = 'predictive_and_similarity'

# Each downstream model's rank-1 and rank-2 single representation.
BASE_SPACES = tuple(f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1))

SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}


def removal_seed(share, repeat, fold_i, n_training):
    """The seed for one random draw of kept molecules on one fold.

    zlib.crc32 and not hash(), for n24.replicate_seed's reason: hash() of a
    string changes per process, so the same command would keep different
    molecules on every run. The feature space is not in the key, so the same
    molecules are kept in every space and the space is the only difference
    between two rows.
    """
    key = f'{share:g}|{repeat}|{fold_i}|{n_training}'
    return int(zlib.crc32(key.encode('utf-8')) & 0x7FFFFFFF)


def kept_rows(share, repeat, fold_i, n_training):
    """Indices of the training molecules kept at this share, sorted."""
    n_keep = int(round(float(share) * int(n_training)))
    n_keep = min(max(n_keep, 2), int(n_training))
    rng = np.random.default_rng(removal_seed(share, repeat, fold_i, n_training))
    return np.sort(rng.choice(int(n_training), n_keep, replace=False))


def moved_toward(y, oof, alpha):
    """Each label moved a fraction alpha of the way to its out-of-fold prediction."""
    y = np.asarray(y, dtype=float)
    return (1.0 - float(alpha)) * y + float(alpha) * np.asarray(oof, dtype=float)


def _fit_predict(factory, X_tr, y_tr, X_va, metric_name):
    """One fit and its held-out predictions, or (None, why)."""
    try:
        m = factory()
        m.fit(X_tr, y_tr)
        return np.asarray(_predict_for_metric(m, X_va, metric_name),
                          dtype=float), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _fit_predict_cached(dataset_name, model_name, factory, X_tr, y_tr, X_va,
                        metric_name):
    """_fit_predict through noise_cache.cached_fit_predict for a seeded model
    (CACHEABLE_MODELS), and _fit_predict itself otherwise. Used for the fits
    plan section 3 lists as repeated: the correct-label fit and the fit on
    injected labels, which other experiments make for the same draw. The
    prediction rule is n07._predict_for_metric's (noise_cache.predict_mode_for),
    so a stored value equals a refit. Added 2026-09-25."""
    if dataset_name is None or model_name not in CACHEABLE_MODELS:
        return _fit_predict(factory, X_tr, y_tr, X_va, metric_name)
    try:
        return np.asarray(nc.cached_fit_predict(
            dataset_name, model_name, factory, X_tr, y_tr, X_va,
            predict_mode=nc.predict_mode_for(metric_name)),
            dtype=float), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _score(pred, y_va, metric_fn):
    """(the dataset's own metric, mean squared error), both against y_va."""
    y_va = np.asarray(y_va, dtype=float)
    return (float(metric_fn(y_va, pred)),
            float(np.mean((np.asarray(pred, dtype=float) - y_va) ** 2)))


def _levels_and_draws(dataset_name, levels=None, draws=None):
    """(level, draw) pairs to inject. Level 0.0 is dropped: the correct-label
    fit already on every fold is that row."""
    levels = (noise_levels(dataset_name, strategy=CONDITION)
              if levels is None else levels)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    return [(float(l), int(d)) for l in levels if float(l) > 0 for d in draws]


def _build_spaces(train_reps, val_reps, y_tr, g_tr, base_by_model, spaces,
                  recipe_for_space_by_model, downstream_models,
                  model_factories, metric_fn, lower_is_better, *,
                  dataset_name=None, reps=None, train_idx=None, val_idx=None,
                  metric_name=None, augmentations=None):
    """{space: [(training matrix, held-out matrix, models scored on it)]}.

    base_k holds one matrix per distinct representation at rank k, shared by
    the models that rank it there. A hybrid space is one matrix per distinct
    recipe, n23's layout: a model with no recipe gets no matrix rather than
    another model's.

    Since 2026-09-25 (plan section 3), given dataset_name, reps, train_idx,
    val_idx and metric_name, a single-representation matrix goes through
    noise_cache.cached_scaled and a hybrid through
    noise_cache.cached_hybrid_for_fold: the same values, stored.
    """
    _cache = (dataset_name is not None and reps is not None
              and train_idx is not None)
    out = {}
    for k in range(1, NOISE_BASE_REP_COUNT + 1):
        bsp = f'base_{k}'
        if bsp not in spaces:
            continue
        cells = []
        for rn, ms in base_rank_groups(base_by_model, k,
                                       downstream_models).items():
            if _cache:
                _Xt, _Xv = cached_base_cell(dataset_name, reps[rn],
                                            train_idx, val_idx)
                cells.append((_Xt, _Xv, list(ms)))
                continue
            sc = StandardScaler()
            cells.append((sc.fit_transform(train_reps[rn]),
                          sc.transform(val_reps[rn]), list(ms)))
        out[bsp] = cells
    # The augmentation representations for this fold. An augmented recipe was
    # rebuilt without them until 2026-09-29, i.e. narrower than the recipe
    # the search chose.
    aug_tr = aug_va = None
    if augmentations and train_idx is not None and val_idx is not None:
        aug_tr = {k: np.asarray(v[train_idx], dtype=np.float64)
                  for k, v in augmentations.items()}
        aug_va = {k: np.asarray(v[val_idx], dtype=np.float64)
                  for k, v in augmentations.items()}
    for sp in spaces:
        if sp in BASE_SPACES:
            continue
        cells = []
        for bp_m, mnames, shareable in group_models_by_recipe(
                {m: recipe_for_space_by_model.get(m, {}).get(sp)
                 for m in downstream_models}, downstream_models):
            if _cache:
                X_tr_h, X_va_h = cached_hybrid_cell(
                    dataset_name, train_reps, val_reps, y_tr, bp_m, g_tr,
                    None if shareable else mnames[0],
                    None if shareable else model_factories[mnames[0]](),
                    (metric_name, metric_fn, lower_is_better), train_idx,
                    augmentations_train=aug_tr, augmentations_val=aug_va)
            else:
                X_tr_h, X_va_h = hybrid_for_fold(
                    train_reps, val_reps, y_tr, bp_m, g_tr,
                    None if shareable else model_factories[mnames[0]](),
                    metric_fn, lower_is_better,
                    augmentations_train=aug_tr, augmentations_val=aug_va)
            if X_tr_h is not None:
                sh = StandardScaler()
                cells.append((sh.fit_transform(X_tr_h), sh.transform(X_va_h),
                              list(mnames)))
        if cells:
            out[sp] = cells
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N27: the cost of fewer training molecules, and how label error propagates."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N27: REMOVAL COST AND HOW LABEL ERROR REACHES THE TEST ERROR")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A variance of the "
              f"label error is not defined for a class label.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    labels = labels.astype(float)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    shares = sorted({float(s) for s in (kwargs.get('shares') or SHARES)}
                    | {1.0})
    repeats = int(kwargs.get('repeats') or REPEATS)
    alphas = [float(a) for a in (kwargs.get('alphas') or ALPHAS)]
    oof_splits = int(kwargs.get('oof_splits') or OOF_SPLITS)
    parts = list(kwargs.get('parts')
                 or ('removal_cost', 'injected', 'relabelled'))

    # ⛔ config FIRST, then the kwarg -- n24's comment at :418 says why.
    # rf and svm are added when the caller left them out: every D-versus-D'
    # row needs them (ensure_dprime_models, owner decision 2026-09-25).
    downstream_models = ensure_dprime_models(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n27: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')

    passes = (_levels_and_draws(dataset_name, levels=kwargs.get('levels'),
                                draws=kwargs.get('draws'))
              if 'injected' in parts else [])

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'models': downstream_models,
        'parts': parts,
        'shares': shares,
        'repeats': repeats,
        'noise_condition': CONDITION,
        'dose_units': noise_dose_units(dataset_name),
        'noise_levels': sorted({l for l, _ in passes}),
        'noise_draws': sorted({d for _, d in passes}),
        'alphas': alphas,
        'oof_splits': oof_splits,
        'held_out_graded_against': (
            'labels[val_idx] as the loader produced them, in every part. No '
            'moved or injected label is ever scored against.'),
        'features_note': (
            'removal_cost and relabelled: each feature space is built once per '
            'fold from ALL the training molecules at their correct labels, and '
            'the molecules are removed or the labels moved after the features '
            'were chosen. injected: each hybrid space is rebuilt on every pass '
            'from the noisy training labels, as n24 does. The single-'
            'representation spaces base_1 and base_2 do not depend on the '
            'labels.'),
        'removal_note': (
            'One row per random draw of the kept molecules. A draw is a '
            'replicate of the whole procedure: each fold draws its own '
            'molecules under the same draw index. The loss is paired against '
            'the share 1.0 fit on the same fold and the same features. '
            'Positive loss means the smaller training set scored worse.'),
        'ratio_note': (
            'propagation ratio = (held-out mean squared error of the fit on '
            'changed labels - held-out mean squared error of the fit on correct '
            'labels) / (mean over training molecules of (changed label - '
            'correct label) squared). Numerator and denominator are both in '
            'label units squared, and both are written beside every ratio.'),
        'per_fold': {},
    }

    splits = list(cv_split(groups, n_folds, next(iter(reps.values())), labels,
                           mode=mode))

    # ── The hybrid recipe, fetched once PER DOWNSTREAM MODEL, n23's block ──
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
        _rs, _un = pool_recipes_by_family(
            by_model[_m]['meta'] if _m in by_model else None,
            SPACE_FOR_FAMILY, eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipes_by_model[_m] = _rs
        recipe_for_space_by_model[_m] = {r['space']: r['blueprint']
                                         for r in _rs}
        for _sp, _why in _un.items():
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

    spaces = base_spaces + tuple(_ordered)
    results['hybrid_available'] = bool(_ordered)
    if not _ordered:
        results['hybrid_unavailable_reason'] = (
            (f"the pool lookup RAISED {results['hybrid_lookup_error']} -- the "
             f"pool was never read, so this is not a statement about what it "
             f"holds. Every part ran in the single representations only.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from the pool subset {POOL_SUBSET!r} for '
             f'any downstream model, so every part ran in the single '
             f'representations only.'))
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"single representations only")
    else:
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
        for m, rs in recipes_by_model.items():
            for r in rs:
                print(f"  {m} / {r['space']}: family {r['objective']!r}, "
                      f"budget {r['budget']} features")
    results['spaces_run'] = list(spaces)

    print(f"  {len(labels)} molecules, {len(splits)} folds, models "
          f"{downstream_models}, spaces {list(spaces)}")
    print(f"  parts {parts}; shares {shares} x {repeats} draws; "
          f"{len(passes)} injected (level, draw) passes; alphas {alphas}")

    # ref[space][model][fold key] = {'score', 'mse'}
    ref = {}
    # removal[space][share][model][repeat][fold key] = {'score', 'mse'}
    removal = {}
    # injected[space][level][model][pass key] = {..., 'injected_variance'}
    injected = {}
    # relabelled[space][alpha][model][fold key] = {..., 'moved_variance'}
    relabelled = {}
    errors = {}
    t0 = time.time()

    # ── Rows and the per-fold checkpoint (2026-09-25) ──
    # The two parts that CHANGE the training data write D-versus-D' rows:
    # removal_cost (D' = a random share of the molecules) and relabelled (D'
    # = the labels moved toward the out-of-fold prediction). D is the fold's
    # training molecules at their correct labels, so score_D and score_clean
    # are both the share-1.0 fit. The injected part adds noise and removes
    # none, so it is a measurement, not a D', and writes no row.
    _metric = (metric_name, metric_fn, lower_is_better)
    rows = RowSink(output_dir, 'n27_removal_cost_and_propagation',
                   dataset_name, experiment='n27', metric=_metric,
                   condition=kwargs.get('tag'), data=data, config=config)
    ckpt = PassCheckpoint(
        output_dir, 'n27_removal_cost_and_propagation', dataset_name,
        condition=kwargs.get('tag'), data=data, config=config,
        source_file=__file__,
        key={'parts': parts, 'shares': shares, 'repeats': repeats,
             'alphas': alphas, 'oof_splits': oof_splits,
             'passes': passes, 'models': list(downstream_models),
             'base_reps': base_by_model, 'n_splits': n_folds})
    done_folds = set()
    _state = ckpt.load()
    if _state:
        ref, removal, injected, relabelled, errors = (
            _state['ref'], _state['removal'], _state['injected'],
            _state['relabelled'], _state['errors'])
        done_folds = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_folds)} folds already "
              f"done")
    _cache_kw = dict(dataset_name=dataset_name, reps=reps,
                     metric_name=metric_name)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        fkey = f'f{fold_i}'
        if fkey in done_folds:
            print(f"    fold {fold_i}: on disk in the checkpoint")
            continue
        y_tr = labels[train_idx]
        y_va = labels[val_idx]
        g_tr = groups[train_idx] if groups is not None else None
        n_tr = len(train_idx)
        train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                      for k, v in reps.items()}
        val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                    for k, v in reps.items()}

        clean_spaces = _build_spaces(
            train_reps, val_reps, y_tr, g_tr, base_by_model, spaces,
            recipe_for_space_by_model, downstream_models, model_factories,
            metric_fn, lower_is_better, train_idx=train_idx, val_idx=val_idx,
            augmentations=data.get('augmentations'), **_cache_kw)
        for sp in spaces:
            if sp not in clean_spaces:
                errors[f'{sp}|{fkey}|build'] = (
                    'no feature matrix could be built for this space on this '
                    'fold')

        for space, cells in clean_spaces.items():
            for X_tr_s, X_va_s, cell_models in cells:
                # ── The correct-label fit: share 1.0, alpha 0, level 0 ──
                for mname in cell_models:
                    # Through noise_cache for a seeded model (2026-09-25).
                    pred, err = _fit_predict_cached(
                        dataset_name, mname, model_factories[mname], X_tr_s,
                        y_tr, X_va_s, metric_name)
                    if pred is None:
                        errors[f'{space}|{mname}|{fkey}|reference'] = err
                        continue
                    s, mse = _score(pred, y_va, metric_fn)
                    ref.setdefault(space, {}).setdefault(mname, {})[fkey] = {
                        'score': s, 'mse': mse}

                # ── removal_cost ──
                if 'removal_cost' in parts:
                    for share in shares:
                        if share >= 1.0:
                            continue
                        skey = f'{share:g}'
                        for rep_i in range(repeats):
                            keep = kept_rows(share, rep_i, fold_i, n_tr)
                            for mname in cell_models:
                                pred, err = _fit_predict(
                                    model_factories[mname], X_tr_s[keep],
                                    y_tr[keep], X_va_s, metric_name)
                                if pred is None:
                                    errors[f'{space}|{skey}|r{rep_i}|{mname}|'
                                           f'{fkey}'] = err
                                    continue
                                s, mse = _score(pred, y_va, metric_fn)
                                (removal.setdefault(space, {})
                                 .setdefault(skey, {}).setdefault(mname, {})
                                 .setdefault(f'r{rep_i}', {}))[fkey] = {
                                    'score': s, 'mse': mse,
                                    'training_molecules_kept': int(len(keep)),
                                    'training_molecules': int(n_tr)}
                                # D' = this random share. It is random by
                                # construction, so it has no random twin.
                                _ref = ((ref.get(space) or {}).get(mname)
                                        or {}).get(fkey) or {}
                                rows.add(method='removal_cost',
                                         method_setting=(f'share={share:g}|'
                                                         f'draw=r{rep_i}'),
                                         similarity_setting=NO_NEIGHBOURS,
                                         model_input=space, model=mname,
                                         noise_condition='none', level=None,
                                         draw=0, fold=fold_i,
                                         score_D=_ref.get('score'),
                                         score_Dprime=s,
                                         score_clean=_ref.get('score'))

                # ── relabelled ──
                if 'relabelled' in parts:
                    try:
                        # noise_cache.cached_oof_predict since 2026-09-25:
                        # the same call (seed 0), stored.
                        oof = nc.cached_oof_predict(dataset_name, X_tr_s, y_tr,
                                                    g_tr, n_splits=oof_splits)
                    except Exception as exc:
                        errors[f'{space}|{cell_models[0]}|{fkey}|oof'] = (
                            f'{type(exc).__name__}: {exc}')
                        oof = None
                    if oof is not None:
                        for alpha in alphas:
                            akey = f'{alpha:g}'
                            y_mv = moved_toward(y_tr, oof, alpha)
                            moved_var = float(np.mean((y_mv - y_tr) ** 2))
                            for mname in cell_models:
                                pred, err = _fit_predict(
                                    model_factories[mname], X_tr_s, y_mv,
                                    X_va_s, metric_name)
                                if pred is None:
                                    errors[f'{space}|a{akey}|{mname}|'
                                           f'{fkey}'] = err
                                    continue
                                s, mse = _score(pred, y_va, metric_fn)
                                (relabelled.setdefault(space, {})
                                 .setdefault(akey, {})
                                 .setdefault(mname, {}))[fkey] = {
                                    'score': s, 'mse': mse,
                                    'moved_variance': moved_var}
                                _ref = ((ref.get(space) or {}).get(mname)
                                        or {}).get(fkey) or {}
                                rows.add(method='move_toward_oof',
                                         method_setting=f'alpha={alpha:g}',
                                         similarity_setting=NO_NEIGHBOURS,
                                         model_input=space, model=mname,
                                         noise_condition='none', level=None,
                                         draw=0, fold=fold_i,
                                         score_D=_ref.get('score'),
                                         score_Dprime=s,
                                         score_clean=_ref.get('score'))
                            # Controls (KIRBY_NOISE_CONTROLS=1 only): the
                            # label changes y' - y dealt to random molecules
                            # and permuted (controls.py).
                            _cd = as_cleanup_dict(X_tr_s, y_tr, labels=y_mv)
                            _seed = (int(fold_i), int(zlib.crc32(
                                f'{space}|{cell_models[0]}|{akey}'.encode())
                                & 0x7FFFFFFF))
                            for cname, tw, why in control_twins(
                                    _cd, X_tr_s, y_tr, _seed):
                                if tw is None:
                                    errors[f'{space}|a{akey}|{cname}|'
                                           f'{fkey}'] = why
                                    continue
                                for mname in cell_models:
                                    pred, err = _fit_predict(
                                        model_factories[mname], X_tr_s,
                                        tw['y'], X_va_s, metric_name)
                                    _ref = ((ref.get(space) or {}).get(mname)
                                            or {}).get(fkey) or {}
                                    rows.add(
                                        method='move_toward_oof',
                                        method_setting=(f'alpha={alpha:g}|'
                                                        f'control={cname}'),
                                        similarity_setting=NO_NEIGHBOURS,
                                        model_input=space, model=mname,
                                        noise_condition='none', level=None,
                                        draw=0, fold=fold_i,
                                        score_D=_ref.get('score'),
                                        score_Dprime=(None if pred is None else
                                                      _score(pred, y_va,
                                                             metric_fn)[0]),
                                        score_clean=_ref.get('score'))

        # ── injected ──
        for level, draw in passes:
            lkey = f'{level:g}'
            pkey = f'l{lkey}d{draw}{fkey}'
            spec = noise_spec_for(dataset_name, CONDITION, level,
                                  noise_seed=draw)
            try:
                y_noisy, _mask, _prov = inject_train_labels(
                    labels, spec, train_idx=train_idx,
                    dataset_name=dataset_name)
            except Exception as exc:
                errors[f'l{lkey}|d{draw}|{fkey}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y_noisy = np.asarray(y_noisy, dtype=float)
            inj_var = float(np.mean((y_noisy - y_tr) ** 2))

            # The single-representation matrices do not depend on the labels,
            # so they are reused. A hybrid space is rebuilt on the noisy
            # labels, as n24 does.
            noisy_spaces = {sp: c for sp, c in clean_spaces.items()
                            if sp in BASE_SPACES}
            if any(sp not in BASE_SPACES for sp in spaces):
                noisy_spaces.update({
                    sp: c for sp, c in _build_spaces(
                        train_reps, val_reps, y_noisy, g_tr, base_by_model,
                        tuple(sp for sp in spaces if sp not in BASE_SPACES),
                        recipe_for_space_by_model, downstream_models,
                        model_factories, metric_fn, lower_is_better,
                        train_idx=train_idx, val_idx=val_idx,
                        augmentations=data.get('augmentations'),
                        **_cache_kw).items()
                    if sp not in BASE_SPACES})
            for space, cells in noisy_spaces.items():
                for X_tr_s, X_va_s, cell_models in cells:
                    for mname in cell_models:
                        # Through noise_cache for a seeded model: the fit on
                        # the injected labels is the no-cleaning fit other
                        # experiments make for the same draw (2026-09-25).
                        pred, err = _fit_predict_cached(
                            dataset_name, mname, model_factories[mname],
                            X_tr_s, y_noisy, X_va_s, metric_name)
                        if pred is None:
                            errors[f'{space}|{pkey}|{mname}|injected'] = err
                            continue
                        s, mse = _score(pred, y_va, metric_fn)
                        (injected.setdefault(space, {}).setdefault(lkey, {})
                         .setdefault(mname, {}))[pkey] = {
                            'score': s, 'mse': mse,
                            'injected_variance': inj_var, 'fold': fkey}

        # The fold is finished: its rows to disk, then the accumulators.
        rows.flush()
        done_folds.add(fkey)
        ckpt.save({'ref': ref, 'removal': removal, 'injected': injected,
                   'relabelled': relabelled, 'errors': errors,
                   'done': sorted(done_folds)})
        print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['per_fold'] = convert_numpy({
        'reference': ref, 'removal_cost': removal,
        'injected': injected, 'relabelled': relabelled})
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        ref, removal, injected, relabelled, lower_is_better, metric_name))
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    ckpt.clear()
    return results


def _loss(base, got, lower_is_better):
    """How much worse `got` scored than `base`, in the dataset's metric.

    Positive means worse, whichever way the metric runs: (got - base) where
    lower wins and (base - got) where higher wins.
    """
    return (got - base) if lower_is_better else (base - got)


def _summarise(ref, removal, injected, relabelled, lower_is_better,
               metric_name):
    """The three parts, each per (space, model), paired on the fold."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'removal_cost': {}, 'propagation': {}}

    # ── removal: one row per draw, best and worst, no mean across draws ──
    for space, by_share in removal.items():
        for share, by_model in by_share.items():
            for m, by_rep in by_model.items():
                base = (ref.get(space) or {}).get(m) or {}
                rows = {}
                for rep, by_fold in by_rep.items():
                    shared = sorted(set(base) & set(by_fold))
                    if not shared:
                        continue
                    losses = [_loss(base[k]['score'], by_fold[k]['score'],
                                    lower_is_better) for k in shared]
                    mse_losses = [by_fold[k]['mse'] - base[k]['mse']
                                  for k in shared]
                    rows[rep] = {
                        'n_folds_paired': len(shared),
                        'folds_worse': int(sum(l > 0 for l in losses)),
                        'loss_per_fold': [float(x) for x in losses],
                        'mean_loss_over_folds': float(np.mean(losses)),
                        'mean_mse_loss_over_folds': float(np.mean(mse_losses)),
                    }
                if not rows:
                    continue
                means = [r['mean_loss_over_folds'] for r in rows.values()]
                out['removal_cost'].setdefault(space, {}).setdefault(
                    share, {})[m] = {
                    'per_draw': rows,
                    'draws_read': len(rows),
                    'best_draw_mean_loss': float(min(means)),
                    'worst_draw_mean_loss': float(max(means)),
                    'draws_worse_than_every_molecule': int(
                        sum(x > 0 for x in means)),
                    'share_1_mean_score': float(np.mean(
                        [v['score'] for v in base.values()])),
                }

    # ── propagation: injected and relabelled, per pass, side by side ──
    def _ratio_rows(block, var_key):
        per = {}
        for space, by_setting in block.items():
            for setting, by_model in by_setting.items():
                for m, by_pass in by_model.items():
                    base = (ref.get(space) or {}).get(m) or {}
                    rows = []
                    for pkey, v in sorted(by_pass.items()):
                        fk = v.get('fold', pkey)
                        if fk not in base:
                            continue
                        d_mse = v['mse'] - base[fk]['mse']
                        var = v[var_key]
                        rows.append({
                            'pass': pkey,
                            'held_out_mse_change': float(d_mse),
                            'training_label_variance_added': float(var),
                            'ratio': (float(d_mse / var) if var > 0
                                      else float('nan')),
                            'held_out_score_loss': float(_loss(
                                base[fk]['score'], v['score'],
                                lower_is_better)),
                        })
                    if not rows:
                        continue
                    num = float(np.mean([r['held_out_mse_change']
                                         for r in rows]))
                    den = float(np.mean([r['training_label_variance_added']
                                         for r in rows]))
                    ratios = np.array([r['ratio'] for r in rows], dtype=float)
                    ratios = ratios[np.isfinite(ratios)]
                    per.setdefault(space, {}).setdefault(m, {})[setting] = {
                        'per_pass': rows,
                        'n_passes': len(rows),
                        'mean_held_out_mse_change': num,
                        'mean_training_label_variance_added': den,
                        'ratio_of_the_two_means': (num / den if den > 0
                                                   else float('nan')),
                        'spread_over_pass_ratios': (float(ratios.std())
                                                    if ratios.size
                                                    else float('nan')),
                    }
        return per

    inj = _ratio_rows(injected, 'injected_variance')
    rel = _ratio_rows(relabelled, 'moved_variance')
    for space in sorted(set(inj) | set(rel)):
        for m in sorted(set(inj.get(space, {})) | set(rel.get(space, {}))):
            out['propagation'].setdefault(space, {})[m] = {
                'injected_by_level': inj.get(space, {}).get(m, {}),
                'relabelled_by_alpha': rel.get(space, {}).get(m, {}),
            }
    return out


def _print_summary(results):
    s = results['summary']
    metric = s['metric']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric}; every comparison paired on the fold)")
    if not results.get('hybrid_available', True):
        print("  no hybrid recipe was available; single representations only")

    print("\n  REMOVAL COST: loss against training on every molecule, "
          "positive = worse")
    for space, by_share in s['removal_cost'].items():
        for share, by_model in sorted(by_share.items()):
            for m, row in by_model.items():
                print(f"    {space}/{m} keeping {share} of the training "
                      f"molecules: over {row['draws_read']} draws, best "
                      f"{row['best_draw_mean_loss']:+.4f}, worst "
                      f"{row['worst_draw_mean_loss']:+.4f} {metric} "
                      f"(every molecule: {row['share_1_mean_score']:.4f}); "
                      f"{row['draws_worse_than_every_molecule']} of "
                      f"{row['draws_read']} draws scored worse")

    print("\n  PROPAGATION: held-out MSE change, training-label variance "
          "added, ratio")
    for space, by_model in s['propagation'].items():
        for m, blk in by_model.items():
            for lvl, r in sorted(blk['injected_by_level'].items()):
                print(f"    {space}/{m} injected level {lvl}: "
                      f"{r['mean_held_out_mse_change']:+.5f} / "
                      f"{r['mean_training_label_variance_added']:.5f} = "
                      f"{r['ratio_of_the_two_means']:+.4f} over "
                      f"{r['n_passes']} passes (spread of per-pass ratios "
                      f"{r['spread_over_pass_ratios']:.4f})")
            for a, r in sorted(blk['relabelled_by_alpha'].items()):
                print(f"    {space}/{m} moved toward prediction, alpha {a}: "
                      f"{r['mean_held_out_mse_change']:+.5f} / "
                      f"{r['mean_training_label_variance_added']:.5f} = "
                      f"{r['ratio_of_the_two_means']:+.4f} over "
                      f"{r['n_passes']} folds (spread of per-fold ratios "
                      f"{r['spread_over_pass_ratios']:.4f})")
    print("\n  Both variances are in label units squared, averaged over "
          "training molecules.")


def main():
    parser = create_experiment_parser(
        'N27: removal cost and how label error reaches the test error')
    parser.add_argument('--shares', type=str, default=None,
                        help='Comma-separated shares of training molecules kept '
                             f'(default {",".join(str(s) for s in SHARES)}); '
                             '1.0 is always added as the reference')
    parser.add_argument('--repeats', type=int, default=None,
                        help=f'Random draws of kept molecules per share '
                             f'(default {REPEATS})')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated gaussian noise levels. Default: '
                             'the shared grid for this dataset, 0.0 dropped')
    parser.add_argument('--alphas', type=str, default=None,
                        help='Comma-separated fractions each label is moved '
                             'toward its out-of-fold prediction '
                             f'(default {",".join(str(a) for a in ALPHAS)})')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates '
                             f'(default {",".join(str(d) for d in NOISE_DRAWS)})')
    parser.add_argument('--parts', type=str, default=None,
                        help='Comma-separated: removal_cost, injected, '
                             'relabelled (default all three)')
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
    if args.shares:
        kw['shares'] = [float(x) for x in args.shares.split(',')]
    if args.repeats:
        kw['repeats'] = int(args.repeats)
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.alphas:
        kw['alphas'] = [float(x) for x in args.alphas.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.parts:
        kw['parts'] = [x.strip() for x in args.parts.split(',')]

    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output,
                            'n27_removal_cost_and_propagation', args.dataset,
                            filename_suffix=args.tag)


if __name__ == '__main__':
    main()
