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
                 0.7, 0.8 and 0.9, 20 random draws of the molecules at each
                 share, every label correct. Each is paired against training on
                 all of them. A removal cleanup that drops a tenth of the
                 training molecules pays at least the 0.9 row, before any
                 benefit from which molecules it dropped.

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
    labels, the same way n24 builds it. The base representation does not
    depend on the labels.

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
    get_best_search_for_downstream, pick_noise_base_rep,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import noise_levels, noise_dose_units
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, hybrid_for_fold,
    pool_recipe_per_downstream_model, group_models_by_recipe,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.utils.noisy_labels import oof_predict

# Shares of the training molecules kept in the removal part. 1.0 is the
# reference every other share is paired against.
SHARES = (0.7, 0.8, 0.9, 1.0)

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


def _build_spaces(train_reps, val_reps, y_tr, g_tr, base_rep_name, spaces,
                  recipe_for_space_by_model, downstream_models,
                  model_factories, metric_fn, lower_is_better):
    """{space: [(training matrix, held-out matrix, models scored on it)]}.

    The base representation is one matrix for every model. A hybrid space is
    one matrix per distinct recipe, n23's layout: a model with no recipe gets
    no matrix rather than another model's.
    """
    out = {}
    sc = StandardScaler()
    out['base'] = [(sc.fit_transform(train_reps[base_rep_name]),
                    sc.transform(val_reps[base_rep_name]),
                    list(downstream_models))]
    for sp in spaces:
        if sp == 'base':
            continue
        cells = []
        for bp_m, mnames, shareable in group_models_by_recipe(
                {m: recipe_for_space_by_model.get(m, {}).get(sp)
                 for m in downstream_models}, downstream_models):
            X_tr_h, X_va_h = hybrid_for_fold(
                train_reps, val_reps, y_tr, bp_m, g_tr,
                None if shareable else model_factories[mnames[0]](),
                metric_fn, lower_is_better)
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
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n27: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))

    passes = (_levels_and_draws(dataset_name, levels=kwargs.get('levels'),
                                draws=kwargs.get('draws'))
              if 'injected' in parts else [])

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'models': downstream_models,
        'base_rep': base_rep_name,
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
            'from the noisy training labels, as n24 does. The base '
            'representation does not depend on the labels.'),
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

    splits = list(cv_split(groups, n_folds, reps[base_rep_name], labels,
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
    spaces = ('base',) + tuple(_ordered)
    results['hybrid_available'] = bool(_ordered)
    if not _ordered:
        results['hybrid_unavailable_reason'] = (
            (f"the pool lookup RAISED {results['hybrid_lookup_error']} -- the "
             f"pool was never read, so this is not a statement about what it "
             f"holds. Every part ran in the base representation only.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from the pool subset {POOL_SUBSET!r} for '
             f'any downstream model, so every part ran in the base '
             f'representation only.'))
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"base representation only")
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
          f"{downstream_models}, base representation {base_rep_name}")
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

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        fkey = f'f{fold_i}'
        y_tr = labels[train_idx]
        y_va = labels[val_idx]
        g_tr = groups[train_idx] if groups is not None else None
        n_tr = len(train_idx)
        train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                      for k, v in reps.items()}
        val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                    for k, v in reps.items()}

        clean_spaces = _build_spaces(
            train_reps, val_reps, y_tr, g_tr, base_rep_name, spaces,
            recipe_for_space_by_model, downstream_models, model_factories,
            metric_fn, lower_is_better)
        for sp in spaces:
            if sp not in clean_spaces:
                errors[f'{sp}|{fkey}|build'] = (
                    'no feature matrix could be built for this space on this '
                    'fold')

        for space, cells in clean_spaces.items():
            for X_tr_s, X_va_s, cell_models in cells:
                # ── The correct-label fit: share 1.0, alpha 0, level 0 ──
                for mname in cell_models:
                    pred, err = _fit_predict(model_factories[mname], X_tr_s,
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

                # ── relabelled ──
                if 'relabelled' in parts:
                    try:
                        oof = oof_predict(X_tr_s, y_tr, g_tr,
                                          n_splits=oof_splits)
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

            # The base matrix does not depend on the labels, so it is reused.
            # A hybrid space is rebuilt on the noisy labels, as n24 does.
            noisy_spaces = {'base': clean_spaces['base']}
            if len(spaces) > 1:
                noisy_spaces.update({
                    sp: c for sp, c in _build_spaces(
                        train_reps, val_reps, y_noisy, g_tr, base_rep_name,
                        spaces, recipe_for_space_by_model, downstream_models,
                        model_factories, metric_fn, lower_is_better).items()
                    if sp != 'base'})
            for space, cells in noisy_spaces.items():
                for X_tr_s, X_va_s, cell_models in cells:
                    for mname in cell_models:
                        pred, err = _fit_predict(model_factories[mname],
                                                 X_tr_s, y_noisy, X_va_s,
                                                 metric_name)
                        if pred is None:
                            errors[f'{space}|{pkey}|{mname}|injected'] = err
                            continue
                        s, mse = _score(pred, y_va, metric_fn)
                        (injected.setdefault(space, {}).setdefault(lkey, {})
                         .setdefault(mname, {}))[pkey] = {
                            'score': s, 'mse': mse,
                            'injected_variance': inj_var, 'fold': fkey}

        print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['per_fold'] = convert_numpy({
        'reference': ref, 'removal_cost': removal,
        'injected': injected, 'relabelled': relabelled})
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        ref, removal, injected, relabelled, lower_is_better, metric_name))
    _print_summary(results)
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
        print("  no hybrid recipe was available; base representation only")

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
    parser.add_argument('--base-rep', type=str, default=None,
                        help="Base representation. Default: "
                             "pick_noise_base_rep's")
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
    if args.base_rep:
        kw['base_rep'] = args.base_rep

    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output,
                            'n27_removal_cost_and_propagation', args.dataset,
                            filename_suffix=args.tag)


if __name__ == '__main__':
    main()
