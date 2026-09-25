#!/usr/bin/env python3
"""n12's hybrid space against a base representation cut to the same 50 features.

WHAT THE TWO SPACES ARE. The first is the feature matrix n12 builds from the
stored search: 50 features drawn from 10 representations, one recipe, rebuilt on
each fold's noisy labels. The second is the onehot representation alone, its
5600 features ranked by importance fitted on that same fold's noisy labels and
cut to its highest 50. Neither is weighted. Both have their neighbours found by
the same two estimators at the same three neighbour counts.

WHY. n12's result file of 2026-09-20 compared the hybrid against onehot at its
full 5600 features, so two things differed at once: the mix of representations
and the number of features the distance is computed over. This run puts the
width-matched space beside it inside one process, on the same folds and the same
corruption, so the mix is the only difference left of those two.

⛔ A THIRD DIFFERENCE REMAINS AND THIS RUN CANNOT REMOVE IT. The hybrid's recipe
was chosen by a search fitted on CLEAN labels and stored in the result store.
The 50-feature cut ranks its features by an importance fitted on that fold's
NOISY labels. So a hybrid win here still leaves two things tangled: mixing
representations, and choosing features with the clean labels in hand. The report
below says so in its own line and the result file carries the same sentence.

⛔ THREE CORRUPTION DRAWS, NOT ONE. The before number averaged one draw per
fold. _build_fold_shared is handed y_noisy, so the hybrid is rebuilt for every
(fold, draw) pair: 9 builds here against 3 in the before run.

⛔ ROWS ARE NEVER CUT. All 672 antibodies. What is cut is the search space: one
injected level, one downstream model, budget 50 features.

⛔ It deletes, moves and renames nothing.
"""
import os

for _k, _v in {
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'KMP_DUPLICATE_LIB_OK': 'TRUE',
        'KIRBY_LGBM_THREADS': '1', 'KIRBY_RF_JOBS': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'KIRBY_CACHE_DIR': '/Volumes/seagate/kirby_cache',
}.items():
    os.environ.setdefault(_k, _v)

if os.environ.get('KIRBY_SMOKE_N', '').strip():
    raise SystemExit(
        "KIRBY_SMOKE_N is set. It is a ROW CAP inside load_data, not a speed "
        "flag, and a detector graded against a corrupted subset has nothing to "
        "find once the rows are cut. Unset it.")

import argparse
import json
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments', 'noise'))
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np

DATASET = 'flab_thermo_tresanco'
CONDITION = 'outlier_p10'
LEVEL = 0.5
FOLDS = 3
NOISE_SEEDS = [0, 1, 2]
BUDGET = 50
MODEL = 'rf'
OUT = '/Volumes/seagate/local_noise_run/loop_n12'

# The before run's recipe, read out of
# flab_thermo_tresanco__outlier_p10__exponent_0.5.json on 2026-09-23. If the
# pool hands this run a different recipe the two numbers are not comparable, so
# the difference is printed and written into the result file rather than
# silently absorbed.
BEFORE_RECIPE = {
    'budget': 50,
    'allocation_method': 'performance_weighted',
    'importance_method': 'lightgbm_gain',
    'n_reps': 10,
    'search_score_mean': 0.5079169245589681,
}
# The same file's readings, for the report to print beside the new ones.
BEFORE = {
    'hybrid_unweighted': 0.2427,
    'top50_unweighted': 0.2207,
    'base_full_width_unweighted': 0.1937,
    'rows': 12,
    'draws': 1,
}


def result_path(out):
    return os.path.join(out, f'{DATASET}__{CONDITION}__hybrid_vs_top50.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=OUT)
    p.add_argument('--folds', type=int, default=FOLDS)
    p.add_argument('--seeds', default=','.join(str(s) for s in NOISE_SEEDS))
    p.add_argument('--base-rep', required=True,
                   help=('the single representation to compare against. '
                         'Required: the fixed preference list that used to '
                         'choose it was deleted 2026-09-25; the noise '
                         'experiments now compare against each downstream '
                         "model's top 2 single representations by its "
                         'stored score'))
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    path = result_path(a.out)
    if os.path.exists(path):
        print('already on disk: ' + path + '; delete it to re-measure')
        return 0

    import _local_run as LR
    print('ITERATION SETTINGS:', LR.shrink(models=None, budgets=[BUDGET],
                                           search_folds=2), flush=True)
    import experiments.dataset_policy as DP
    DP._DEFAULT_NOISE_LEVELS = (LEVEL,)
    DP.CENSORING_LEVELS_DETECT = (LEVEL,)

    from experiments._shared import (
        load_data, cv_split, get_domain_metric,
        get_model_factories, get_best_search_for_downstream,
        pool_recipes_by_family, POOL_SUBSETS, convert_numpy,
    )
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        inject_train_labels, noise_spec_for, detection_scores,
    )
    from experiments.noise import n12_importance_weighted_distance as n12
    from kirby.hybrid import compute_feature_importance
    from kirby.result_cache import StaleResultDB
    from kirby.similarity.distance_config import get_distance_metric

    rep_names = DATASET_REPS[DATASET]['general']
    t0 = time.time()
    data = load_data(DATASET, rep_preset='all', rep_names=rep_names)
    for r, X in list(data['reps'].items()):
        if int(np.isnan(X).sum()):
            data['reps'][r] = np.nan_to_num(X, nan=0.0)
    data['augmentations'] = None
    reps, labels, groups = data['reps'], data['labels'], data['groups']
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    base_rep_name = a.base_rep
    if base_rep_name not in reps:
        raise SystemExit(f'--base-rep {base_rep_name!r} is not among the loaded '
                         f'representations {sorted(reps)}')
    full_width = int(reps[base_rep_name].shape[1])
    print(f'loaded {len(labels)} molecules in {time.time() - t0:.1f}s; base '
          f'representation {base_rep_name}, {full_width} features; '
          f'neighbours found under {get_distance_metric()}; dataset metric '
          f'{metric_name}', flush=True)

    model_factories = get_model_factories([MODEL], task_type=task_type,
                                          n_samples=len(labels))
    if MODEL not in model_factories:
        print(f'no {MODEL} estimator available; nothing built')
        return 0

    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups, downstream_model=MODEL, budget=None,
            augmentations=None, dataset_name=DATASET, task_type=task_type,
            data=data, pool_subset=n12.POOL_SUBSET)
    except StaleResultDB as e:
        print('BLOCKED: the stored search this run reads cannot be opened. '
              f'Nothing was deleted or moved.\n  {e}', flush=True)
        return 0
    if not meta:
        print('the pool returned no recipe; nothing built')
        return 0

    recipes, unavailable = pool_recipes_by_family(
        meta, n12.SPACE_FOR_FAMILY,
        eligible_families=POOL_SUBSETS.get(n12.POOL_SUBSET))
    rec = next((r for r in recipes if r['space'] == 'hybrid'), None)
    for sp, why in (unavailable or {}).items():
        print(f'  no {sp} space: {why}')
    if rec is None:
        print('the pool holds no prediction-built recipe, which is the space '
              'the before number was measured in; nothing built')
        return 0

    bp_reps = list(getattr(rec['blueprint'], 'reps', None) or [])
    now_recipe = {
        'budget': rec['budget'],
        'allocation_method': rec['allocation_method'],
        'importance_method': rec['importance_method'],
        'n_reps': len(bp_reps),
        'search_score_mean': rec['search_score_mean'],
    }
    print(f'\n  hybrid recipe: budget {rec["budget"]} features, allocation '
          f'{rec["allocation_method"]}, importance {rec["importance_method"]}, '
          f'{len(bp_reps)} representations, search score '
          f'{rec["search_score_mean"]} in {rec["search_metric_name"]!r}')
    print(f'  representations in the recipe: {", ".join(bp_reps)}')
    recipe_differs = {k: [BEFORE_RECIPE[k], now_recipe[k]]
                      for k in BEFORE_RECIPE if BEFORE_RECIPE[k] != now_recipe[k]}
    if recipe_differs:
        print('  ⛔ THIS IS NOT THE RECIPE THE BEFORE NUMBER USED. Fields that '
              'differ, before then now: ' + json.dumps(recipe_differs))
    else:
        print('  this is the recipe the before number used, on all five fields '
              'recorded from it')

    noise_groups, _ng, _lg = noise_groups_for(data)
    sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, a.folds, sample_X, labels, mode='full'))

    # {(importance scorer, estimator_k, space): [one reading per pass]}. The
    # hybrid space does not use the per-fold importance at all, so it is stored
    # once per pass under the scorer name 'none' and paired against BOTH
    # scorers' cuts in the report.
    rows = {}
    moved, n_train, hyb_width, build_failures = [], [], [], 0

    def score(space_name, X, y, mask, imp_name):
        got = 0
        for est_name, est_cls in n12.ESTIMATOR_FACTORIES.items():
            for k in n12.K_VALUES:
                try:
                    sc = est_cls(k=k, metric=None, is_clf=is_clf).estimate(X, y)
                    det = detection_scores(mask, sc, CONDITION)
                except Exception as e:
                    print(f'    {space_name} {est_name}_k{k} failed: {e}')
                    continue
                if det is None:
                    continue
                rows.setdefault((imp_name, f'{est_name}_k{k}', space_name),
                                []).append(
                    float(det['detection_average_precision']))
                got += 1
        return got

    for fold_idx, (tr, va) in enumerate(splits):
        for seed in seeds:
            t_pass = time.time()
            spec = noise_spec_for(DATASET, CONDITION, LEVEL,
                                  noise_seed=seed, is_clf=is_clf)
            y_noisy, cmask, _prov = inject_train_labels(
                labels, spec, train_idx=tr, is_clf=is_clf,
                groups=noise_groups, dataset_name=DATASET)
            mask = np.asarray(cmask).ravel().astype(bool)
            if mask.size == len(labels):
                mask = mask[tr]
            y_tr = y_noisy[tr] if len(y_noisy) == len(labels) else y_noisy
            moved.append(int(mask.sum()))
            n_train.append(int(mask.size))

            train_reps_fold = {k: v[tr] for k, v in reps.items()}
            val_reps_fold = {k: v[va] for k, v in reps.items()}

            shared = n12._build_fold_shared(
                train_reps_fold, y_tr, None, rec['blueprint'], base_rep_name,
                groups=(groups[tr] if groups is not None else None),
                val_reps=val_reps_fold, augmentations_val=None,
                model=model_factories[MODEL](),
                metric_fn=metric_fn, lower_is_better=lower_is_better)
            X_base_std = shared['X_base_std']
            X_hyb = shared['X_hyb_std']
            if X_hyb is None:
                build_failures += 1
                print(f'  fold {fold_idx + 1} draw {seed}: the hybrid build '
                      f'returned nothing; this pass has no hybrid reading',
                      flush=True)
            else:
                hyb_width.append(int(np.asarray(X_hyb).shape[1]))
                score('hybrid_unweighted', X_hyb, y_tr, mask, 'none')

            for imp_method in n12.IMPORTANCE_METHODS:
                t = time.time()
                try:
                    imp = compute_feature_importance(
                        {base_rep_name: train_reps_fold[base_rep_name]}, y_tr,
                        method=imp_method)[base_rep_name]
                except Exception as e:
                    print(f'  {imp_method} importance failed: {e}')
                    continue
                imp = np.asarray(imp)
                nonzero = int((imp > 0).sum())
                order = np.argsort(imp)[::-1]
                keep = np.sort(order[:BUDGET])
                score(f'top{BUDGET}_unweighted', X_base_std[:, keep], y_tr,
                      mask, imp_method)
                score(f'full_{full_width}_unweighted', X_base_std, y_tr, mask,
                      imp_method)
                print(f'  fold {fold_idx + 1} draw {seed} {imp_method}: '
                      f'{nonzero} of {full_width} features scored above zero; '
                      f'importance and both cuts in {time.time() - t:.0f}s',
                      flush=True)
            print(f'  fold {fold_idx + 1} draw {seed} done in '
                  f'{time.time() - t_pass:.0f}s', flush=True)

    report(a, base_rep_name, full_width, rows, moved, n_train, hyb_width,
           build_failures, now_recipe, recipe_differs, bp_reps, path)
    return 0


def report(a, base_rep_name, full_width, rows, moved, n_train, hyb_width,
           build_failures, now_recipe, recipe_differs, bp_reps, path):
    hyb_key = 'hybrid_unweighted'
    cut_key = f'top{BUDGET}_unweighted'
    full_key = f'full_{full_width}_unweighted'
    ests = sorted({k[1] for k in rows})
    imps = sorted({k[0] for k in rows if k[0] != 'none'})
    n_passes = len(moved)

    print('\n' + '=' * 104)
    print('DETECTION AVERAGE PRECISION against the molecules whose labels '
          'moved. Higher is better, whatever the dataset\'s own metric does.')
    print(f'Dataset {DATASET}, condition {CONDITION}, injected level {LEVEL}, '
          f'{a.folds} outer folds by {len(a.seeds.split(","))} corruption '
          f'draws = {n_passes} passes.')
    if moved:
        print(f'{float(np.mean(moved)):.1f} of {float(np.mean(n_train)):.1f} '
              f'training molecules had their label moved, averaged over those '
              f'{n_passes} passes.')
    if hyb_width:
        print(f'The hybrid space carried {sorted(set(hyb_width))} features and '
              f'was built {len(hyb_width)} times, once per pass; '
              f'{build_failures} builds returned nothing.')
    print(f'The cut space is the {base_rep_name} representation, z-scored, '
          f'kept at its {BUDGET} highest-importance features. The full space is '
          f'the same representation at all {full_width}.')
    print('One row is one importance scorer by one estimator by one neighbour '
          'count. The hybrid space uses no per-fold importance, so its reading '
          'is the same in both halves of the table.')
    print('=' * 104)
    print(f'{"scorer":<15}{"neighbours":<26}{"hybrid":>12}{"cut to " + str(BUDGET):>14}'
          f'{"difference":>13}{"full width":>13}')

    diffs, hyb_m, cut_m, full_m = [], [], [], []
    per_row = {}
    for imp in imps:
        for est in ests:
            h = rows.get(('none', est, hyb_key))
            c = rows.get((imp, est, cut_key))
            f_ = rows.get((imp, est, full_key))
            if not h or not c:
                continue
            hv, cv = float(np.mean(h)), float(np.mean(c))
            fv = float(np.mean(f_)) if f_ else float('nan')
            diffs.append(hv - cv)
            hyb_m.append(hv)
            cut_m.append(cv)
            if f_:
                full_m.append(fv)
            per_row[f'{imp}|{est}'] = {'hybrid': hv, 'cut': cv,
                                       'difference': hv - cv, 'full': fv}
            print(f'{imp:<15}{est:<26}{hv:>12.4f}{cv:>14.4f}{hv - cv:>+13.4f}'
                  f'{fv:>13.4f}')
    if not diffs:
        print('no paired row; nothing written')
        return

    d = np.asarray(diffs)
    n = len(d)
    ahead = int((d > 0).sum())
    span = float(d.max() - d.min())
    mean_diff = float(d.mean())
    clears = abs(mean_diff) > span
    print(f'\nRows above: {n}. Each is one importance scorer, one estimator and '
          f'one neighbour count, averaged over the {n_passes} passes.')
    print(f'  hybrid space, {BUDGET} features from {len(bp_reps)} '
          f'representations: mean average precision {float(np.mean(hyb_m)):.4f}')
    print(f'  {base_rep_name} cut to its top {BUDGET} features: mean average '
          f'precision {float(np.mean(cut_m)):.4f}')
    if full_m:
        print(f'  {base_rep_name} at all {full_width} features: mean average '
              f'precision {float(np.mean(full_m)):.4f}')
    print(f'  rows where the hybrid scored higher than the cut: {ahead} of {n}')
    print(f'  mean difference, hybrid minus cut: {mean_diff:+.4f} average '
          f'precision')
    print(f'  smallest and largest of those {n} row differences: '
          f'{float(d.min()):+.4f} and {float(d.max()):+.4f}, {span:.4f} apart')
    print(f'\nTHE CLAIM THIS RUN WAS ARMED TO TEST: mixing representations is '
          f'what wins, kept only if the mean difference clears the spread of '
          f'the {n} row differences.')
    print(f'  {abs(mean_diff):.4f} against a spread of {span:.4f}: '
          + ('the mean CLEARS the spread' if clears
             else 'the mean sits INSIDE its own spread, so this run does not '
                  'carry the claim'))
    print('\n⛔ WHAT IS STILL TANGLED. The hybrid\'s recipe was chosen by a '
          'search fitted on clean labels; the cut ranks its features by an '
          'importance fitted on that fold\'s noisy labels. A hybrid win here '
          'is mixing representations AND clean-label feature choice together.')
    print('\nBEFORE, from the two files of 2026-09-20 and 2026-09-21 on 3 '
          f'folds and 1 corruption draw: hybrid {BEFORE["hybrid_unweighted"]}, '
          f'cut to 50 {BEFORE["top50_unweighted"]}, full width '
          f'{BEFORE["base_full_width_unweighted"]}, hybrid ahead in 9 of 12 '
          'rows, mean difference +0.0220, those 12 differences 0.0708 apart.')

    os.makedirs(a.out, exist_ok=True)
    payload = {
        'dataset': DATASET, 'condition': CONDITION, 'level': LEVEL,
        'folds': a.folds, 'noise_seeds': [int(s) for s in a.seeds.split(',')],
        'passes': n_passes,
        'reading': 'detection_average_precision',
        'higher_is_better': True,
        'base_rep': base_rep_name, 'full_width': full_width,
        'budget': BUDGET,
        'hybrid_recipe': now_recipe,
        'hybrid_recipe_reps': bp_reps,
        'hybrid_recipe_differs_from_before': recipe_differs or None,
        'hybrid_space_widths': sorted(set(hyb_width)),
        'hybrid_builds': len(hyb_width),
        'hybrid_build_failures': build_failures,
        'labels_moved_per_training_fold': float(np.mean(moved)) if moved else None,
        'training_molecules_per_fold': float(np.mean(n_train)) if n_train else None,
        'n_rows': n,
        'mean_hybrid': float(np.mean(hyb_m)),
        'mean_cut_to_budget': float(np.mean(cut_m)),
        'mean_full_width': float(np.mean(full_m)) if full_m else None,
        'rows_hybrid_ahead_of_cut': ahead,
        'mean_difference_hybrid_minus_cut': mean_diff,
        'row_difference_min': float(d.min()),
        'row_difference_max': float(d.max()),
        'row_difference_span': span,
        'mean_clears_the_spread': bool(clears),
        'still_tangled': (
            "The hybrid's recipe was chosen by a search fitted on clean "
            "labels; the top-50 cut ranks its features by an importance "
            "fitted on that fold's noisy labels. Mixing representations and "
            "clean-label feature choice are not separated by this run."),
        'before': BEFORE,
        'per_row': per_row,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    sys.exit(main())
