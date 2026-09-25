#!/usr/bin/env python3
"""N36: which molecules should be measured a second time?

Paper: Noise.

THE QUESTION. A lab with budget for B more measurements has to choose which
molecules get them. The second research report of 2026-09-24 (items L2 and N,
after Sheng, Provost & Ipeirotis 2008, "get another label") proposes ranking
molecules by how much a second measurement would reduce the model's risk. This
file measures whether any ranking beats handing the extra measurements out at
random.

NOTHING IS INJECTED. The second measurement is a real one: a repeat the loader
already holds, held back and then handed out.

THE DATASETS. Only a loader that keeps repeat measurements can be used:
`herg_ic50` (one row per ChEMBL record) and `bindingdb_kd`. Any other dataset
writes {'skipped': 'no_replicate_structure'}, the same as n19 and n26.

THE SET-UP, PER OUTER FOLD AND PER SEED.
  - Training molecules measured once keep their one measurement.
  - Training molecules measured two or more times (the CANDIDATES) start from
    ONE of their measurements, drawn at random by the seed. A second, different
    measurement is also drawn and held back.
  - A budget of BUDGET_SHARES of the candidates get their held-back measurement.
    Their training label becomes the mean of the two.
  - Held-out molecules never train. Each is graded against the mean of ALL its
    measurements.

THE RULES THAT CHOOSE WHO GETS THE BUDGET. Each ranks the candidates; the top
B get the second measurement.
  random                    a seeded draw. RANDOM_DRAWS of them, and every rule
                            is paired against each one. This is the thing to beat.
  disagreement              |start label - out-of-fold prediction|, largest
                            first. The prediction is a group-aware LightGBM
                            out-of-fold fit inside the training molecules, on the
                            start labels, in the space being scored.
  leverage                  leverage (defined below), largest first.
  leverage_x_disagreement   leverage x disagreement, largest first.
  sparsest                  the mean distance to the 5 nearest training
                            molecules, largest first.

WHY NO VARIANCE TERM. The report multiplies leverage by the drop in label
variance a second measurement buys, s2e/1 - s2e/2, where s2e is the pooled
within-molecule variance of the training molecules' measurements. Every
candidate here starts at one measurement and goes to two. So that drop is s2e/2
for every candidate, and it cannot change the order. It is left out, and the
rules are named for what they rank by. Until 2026-09-25 they were called
`variance_x_leverage` and `variance_x_leverage_x_disagreement`, with the
constant still multiplied in; the order is the same.

LEVERAGE, AND WHY THIS ONE. The report says "model leverage or influence" and
names no formula. Used here: 1 + the number of training molecules that have this
candidate among their 5 nearest neighbours, Euclidean, in the standardised
space being scored. It counts how many other molecules' neighbourhoods a label
feeds, and needs no held-out molecule. `sparsest` reads the same neighbour
search the other way round: a molecule far from its neighbours feeds few of
them.

TWO REFERENCES, NOT RULES.
  zero_budget        every candidate on its one start measurement.
  all_measurements   every training molecule on the mean of ALL its
                     measurements. What a loader normally hands you.
A rule's gain over random is read beside the whole gap between the two.

WHERE THE HYBRID COMES IN. Each downstream model runs in the base
representation and in one hybrid per family of stored search, each recipe the
one the pool picked FOR that model, as n23 and n26 do. The hybrid is rebuilt per
(fold, seed) on the zero-budget training labels, and every rule and budget
shares it. So the only thing differing between two rules is which candidates got
a second measurement. Whether a hybrid rebuilt on the re-measured labels picks
other features is not asked here.

THE HEADLINE IS NOT A MEAN. Every (space, model, budget, rule) writes its paired
per-(fold, seed) difference against each random draw. Nothing is averaged across
models, spaces or datasets.

Usage:
    python n36_remeasure_priority.py --dataset herg_ic50 --mode full
    python n36_remeasure_priority.py --dataset bindingdb_kd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream, pick_noise_base_rep,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n19_measurement_count import measurement_stats
from experiments.noise.n26_noise_variance_from_repeats import (
    measurements_by_molecule, direct_noise_variance,
)
from experiments.noise.noise_helpers import (
    hybrid_for_fold, pool_recipe_per_downstream_model, group_models_by_recipe,
    tail_scores,
)
from kirby.utils.noisy_labels import oof_predict

# Which searches the recipe may come from. Same as n19, n23, n24 and n26.
POOL_SUBSET = 'predictive_and_similarity'

SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}

# The share of the candidates (training molecules measured 2+ times) that get
# their second measurement.
BUDGET_SHARES = (0.10, 0.25, 0.50)

# The ranking rules. 'random' is the reference every other rule is paired with.
RULES = ('random', 'disagreement', 'leverage', 'leverage_x_disagreement',
         'sparsest')

# How many random allocations each rule is paired against.
RANDOM_DRAWS = 3

# Seeds for which measurement is a candidate's start and which is its spare.
SEEDS = (0, 1)

# Neighbours for leverage and for sparsest.
N_NEIGHBOURS = 5

# Inner folds for the out-of-fold prediction the disagreement rule reads.
OOF_SPLITS = 5


def draw_start_and_spare(values_per_molecule, seed):
    """One start measurement and one different spare per molecule.

    Returns (start, spare). spare is NaN for a molecule measured once.
    """
    rng = np.random.default_rng(int(seed))
    n = len(values_per_molecule)
    start = np.empty(n)
    spare = np.full(n, np.nan)
    for j, v in enumerate(values_per_molecule):
        if len(v) == 1:
            start[j] = v[0]
            continue
        i1, i2 = rng.choice(len(v), 2, replace=False)
        start[j], spare[j] = v[i1], v[i2]
    return start, spare


def neighbour_stats(X_tr, n_neighbours=N_NEIGHBOURS):
    """Leverage and sparsity for every training molecule, from one kNN search.

    leverage  1 + how many OTHER training molecules have this one among their
              n_neighbours nearest
    sparsity  mean distance to this molecule's own n_neighbours nearest
    """
    n = X_tr.shape[0]
    k = min(n_neighbours, n - 1)
    if k < 1:
        return np.ones(n), np.zeros(n)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_tr)
    dist, idx = nn.kneighbors(X_tr)
    # Column 0 is the molecule itself, except when duplicates tie at distance
    # 0; dropping the molecule's own index wherever it sits handles both.
    own = idx == np.arange(n)[:, None]
    keep_d = np.empty((n, k))
    keep_i = np.empty((n, k), dtype=np.int64)
    for r in range(n):
        m = ~own[r]
        d, i = dist[r][m][:k], idx[r][m][:k]
        keep_d[r], keep_i[r] = d, i
    leverage = 1.0 + np.bincount(keep_i.ravel(), minlength=n).astype(float)
    sparsity = keep_d.mean(axis=1)
    return leverage, sparsity


def rank_candidates(rule, cand, *, disagreement, leverage, sparsity,
                    random_seed):
    """The candidates' indices, best first, under one rule.

    cand          training-local indices of the candidates
    disagreement  per training molecule, |start - oof|
    leverage      per training molecule, from neighbour_stats
    Ties are broken by a seeded draw so no rule inherits molecule order.
    """
    rng = np.random.default_rng(int(random_seed))
    tie = rng.random(len(cand))
    if rule == 'random':
        score = tie
    elif rule == 'disagreement':
        score = disagreement[cand]
    elif rule == 'leverage':
        score = leverage[cand]
    elif rule == 'leverage_x_disagreement':
        score = leverage[cand] * disagreement[cand]
    elif rule == 'sparsest':
        score = sparsity[cand]
    else:
        raise ValueError(f'n36: unknown rule {rule!r}')
    order = np.lexsort((tie, -np.asarray(score, dtype=float)))
    return cand[order]


def labels_after_budget(start_tr, spare_tr, chosen):
    """Training labels once `chosen` (training-local) got their spare."""
    y = start_tr.copy()
    y[chosen] = 0.5 * (start_tr[chosen] + spare_tr[chosen])
    return y


def _fit_predict(factory, X_tr, y_tr, X_va):
    m = factory()
    m.fit(X_tr, y_tr)
    return np.asarray(m.predict(X_va), dtype=np.float64).ravel()


def _score(metric_fn, y_true, pred):
    out = {'score': float(metric_fn(y_true, pred))}
    t = tail_scores(y_true, pred)
    if t is not None:
        out.update(t)
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N36: which molecules to measure a second time."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N36: WHICH MOLECULES TO MEASURE AGAIN")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    reps = data['reps']
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A second measurement "
              f"of a class label has no mean to average into.")
        return {'skipped': 'not_regression', 'task_type': task_type}

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        print(f"  SKIPPED: {dataset_name}'s loader keeps no repeat "
              f"measurements, so there is no second measurement to hand out.")
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    row_groups = data.get('groups')
    if row_groups is None:
        print("  SKIPPED: the loader gives no fold groups, and random folds "
              "are not used in this suite.")
        return {'skipped': 'no_fold_groups', 'dataset': dataset_name}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    budgets = [float(b) for b in (kwargs.get('budget_shares') or BUDGET_SHARES)]
    seeds = [int(s) for s in (kwargs.get('seeds') or SEEDS)]
    n_random = int(kwargs.get('random_draws', RANDOM_DRAWS))
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    rules = list(RULES)

    stats = measurement_stats(labels, rep_groups)
    values = measurements_by_molecule(labels, rep_groups)
    if not np.array_equal([len(v) for v in values], stats['n_measurements']):
        raise RuntimeError(
            "n36: the per-molecule measurement lists are not in "
            "measurement_stats' order.")
    k_all = stats['n_measurements']
    y_mean_all = stats['mean_all'].astype(np.float64)
    n_ent = len(y_mean_all)
    n_multi = int((k_all >= 2).sum())
    ent_groups = np.asarray(row_groups)[stats['row_index']]
    ent_reps = {name: np.asarray(mat[stats['row_index']], dtype=np.float64)
                for name, mat in reps.items()}

    print(f"  {len(labels)} measurements over {n_ent} molecules; "
          f"{n_multi} of them measured twice or more")
    if n_multi < 10:
        print("  SKIPPED: fewer than 10 molecules measured twice or more, so "
              "no budget share hands out a second measurement to anyone.")
        return {'skipped': 'too_few_repeated_molecules',
                'molecules_measured_twice_or_more': n_multi}

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=n_ent)
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n36: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none',
        'grading_label': 'the mean of ALL a held-out molecule\'s measurements',
        'models': downstream_models,
        'base_rep': base_rep_name,
        'rules': rules,
        'budget_shares': budgets,
        'budget_unit': ('share of the training molecules measured twice or '
                        'more that get their second measurement'),
        'seeds': seeds,
        'random_draws': n_random,
        'n_neighbours': N_NEIGHBOURS,
        'feasibility': {
            'n_measurements': int(len(labels)),
            'n_molecules': n_ent,
            'molecules_measured_twice_or_more': n_multi,
        },
        'leverage_definition': (
            '1 + the number of training molecules that have this candidate '
            f'among their {N_NEIGHBOURS} nearest neighbours, Euclidean, in the '
            'standardised space being scored'),
        'variance_term_note': (
            'No variance term is used: every candidate goes from 1 measurement '
            'to 2, so the variance drop s2e/1 - s2e/2 = s2e/2 is the same for '
            'all of them and cannot change the order.'),
        'hybrid_note': (
            'The hybrid is rebuilt per (fold, seed) on the zero-budget training '
            'labels and shared by every rule and budget.'),
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
    spaces = ('base',) + tuple(ordered_spaces)
    results['spaces_run'] = list(spaces)
    results['hybrid_available'] = len(spaces) > 1
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    results['hybrid_recipe_by_model'] = convert_numpy(recipe_record)
    print(f"  base representation: {base_rep_name}; spaces {list(spaces)}")

    X_base = ent_reps[base_rep_name]
    splits = list(cv_split(ent_groups, n_folds, X_base, y_mean_all,
                           task_type='regression'))

    # scores[space][model][arm][pass_key] = {'score': ..., tail ...}
    # arm is 'zero_budget', 'all_measurements', or f'{budget}|{rule}|{r}'. r is
    # the random draw for 'random' and always 0 for a ranking rule.
    scores = {s: {m: {} for m in downstream_models} for s in spaces}
    s2e_by_pass, errors = {}, {}
    t0 = time.time()

    for fold_i, (tr, va) in enumerate(splits):
        g_tr = ent_groups[tr]
        train_reps = {n: v[tr] for n, v in ent_reps.items()}
        val_reps = {n: v[va] for n, v in ent_reps.items()}
        y_va = y_mean_all[va]
        cand = np.where(k_all[tr] >= 2)[0]
        if len(cand) == 0:
            errors[f'f{fold_i}'] = 'no training molecule measured twice or more'
            continue
        # The training fold's own pooled within-molecule variance. Only
        # training molecules' measurements enter it. Recorded for the reader;
        # no rule uses it (see WHY NO VARIANCE TERM in the header).
        s2e = direct_noise_variance([values[j] for j in tr])['s2e']

        for seed in seeds:
            pass_key = f'f{fold_i}s{seed}'
            s2e_by_pass[pass_key] = s2e
            start_all, spare_all = draw_start_and_spare(values, seed)
            start_tr, spare_tr = start_all[tr], spare_all[tr]

            # {space: [(X_tr, X_va, models)]}, built on the zero-budget labels.
            fold_spaces = {}
            sc = StandardScaler()
            fold_spaces['base'] = [(sc.fit_transform(X_base[tr]),
                                    sc.transform(X_base[va]),
                                    list(downstream_models))]
            for _sp in spaces:
                if _sp == 'base':
                    continue
                _cells = []
                for _bp, _mnames, _shareable in group_models_by_recipe(
                        {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                         for _m in downstream_models}, downstream_models):
                    X_tr_h, X_va_h = hybrid_for_fold(
                        train_reps, val_reps, start_tr, _bp, g_tr,
                        None if _shareable else model_factories[_mnames[0]](),
                        metric_fn, lower_is_better)
                    if X_tr_h is None:
                        errors[f'{_sp}/{_mnames[0]}|{pass_key}|build'] = (
                            'the hybrid could not be built on this fold')
                        continue
                    sh = StandardScaler()
                    _cells.append((sh.fit_transform(X_tr_h),
                                   sh.transform(X_va_h), list(_mnames)))
                if _cells:
                    fold_spaces[_sp] = _cells

            for space in spaces:
                for X_tr_s, X_va_s, cell_models in fold_spaces.get(space, ()):
                    # What every rule reads, once per cell.
                    try:
                        oof = oof_predict(X_tr_s, start_tr, g_tr,
                                          n_splits=oof_splits, seed=seed)
                        disagreement = np.abs(start_tr - oof)
                    except Exception as exc:
                        errors[f'{space}|{pass_key}|oof'] = (
                            f'{type(exc).__name__}: {exc}')
                        disagreement = None
                    leverage, sparsity = neighbour_stats(X_tr_s)

                    # Every label column this cell fits on.
                    columns = {'zero_budget': start_tr,
                               'all_measurements': y_mean_all[tr]}
                    for b in budgets:
                        n_b = int(round(b * len(cand)))
                        if n_b < 1:
                            continue
                        for rule in rules:
                            if disagreement is None and 'disagreement' in rule:
                                continue
                            # A ranking rule is fitted once, with draw 0's
                            # tie-break, and paired against every random
                            # draw. Only 'random' is redrawn.
                            draws = range(n_random) if rule == 'random' else (0,)
                            for r in draws:
                                ranked = rank_candidates(
                                    rule, cand, disagreement=disagreement,
                                    leverage=leverage, sparsity=sparsity,
                                    random_seed=1000 * seed + 17 * fold_i + r)
                                columns[f'{b:g}|{rule}|{r}'] = \
                                    labels_after_budget(start_tr, spare_tr,
                                                        ranked[:n_b])
                    for mname in cell_models:
                        for arm, y_fit in columns.items():
                            try:
                                pred = _fit_predict(model_factories[mname],
                                                    X_tr_s, y_fit, X_va_s)
                            except Exception as exc:
                                errors[f'{space}|{mname}|{arm}|{pass_key}'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            scores[space][mname].setdefault(arm, {})[
                                pass_key] = _score(metric_fn, y_va, pred)
            print(f"    fold {fold_i} seed {seed} done "
                  f"[{time.time() - t0:.0f}s]")

    results['s2e_training_fold_by_pass'] = s2e_by_pass
    results['per_space'] = _summarise(scores, spaces, downstream_models,
                                      budgets, rules, n_random,
                                      lower_is_better)
    if errors:
        results['_errors'] = errors
    results = convert_numpy(results)
    _print_summary(results, metric_name)
    save_experiment_results(results, output_dir, 'n36_remeasure_priority',
                            dataset_name, filename_suffix=kwargs.get('tag'))
    return results


def _gain(a, b, lower_is_better):
    """How much a beats b, positive = a better, in the metric's own units."""
    return (b - a) if lower_is_better else (a - b)


def _summarise(scores, spaces, models, budgets, rules, n_random,
               lower_is_better):
    """Per (space, model): raw scores, and every paired difference.

    rule_vs_random[budget][rule] holds, per random draw r, the per-pass gain of
    the rule over the random allocation of draw r, on the same (fold, seed).
    The rule itself is fitted once and paired with every draw.
    """
    out = {}
    for space in spaces:
        srow = {}
        for m in models:
            arms = scores[space][m]
            if not arms:
                continue
            row = {'scores': arms, 'rule_vs_random': {},
                   'all_measurements_vs_zero_budget': {}}
            zb, am = arms.get('zero_budget', {}), arms.get('all_measurements', {})
            row['all_measurements_vs_zero_budget'] = {
                p: _gain(am[p]['score'], zb[p]['score'], lower_is_better)
                for p in zb if p in am}
            for b in budgets:
                bkey = f'{b:g}'
                for rule in rules:
                    if rule == 'random':
                        # Random against zero budget, so the size of the
                        # budget's own effect sits beside the rules' gains.
                        per = {}
                        for r in range(n_random):
                            ra = arms.get(f'{bkey}|random|{r}', {})
                            per[str(r)] = {p: _gain(ra[p]['score'],
                                                    zb[p]['score'],
                                                    lower_is_better)
                                           for p in ra if p in zb}
                        row.setdefault('random_vs_zero_budget', {})[bkey] = per
                        continue
                    per = {}
                    for r in range(n_random):
                        ru = arms.get(f'{bkey}|{rule}|0', {})
                        ra = arms.get(f'{bkey}|random|{r}', {})
                        per[str(r)] = {p: _gain(ru[p]['score'], ra[p]['score'],
                                                lower_is_better)
                                       for p in ru if p in ra}
                    row['rule_vs_random'].setdefault(bkey, {})[rule] = per
            srow[m] = row
        out[space] = srow
    return out


def _span(d):
    vals = [v for per in d.values() for v in per.values()]
    if not vals:
        return None
    return min(vals), max(vals), len(vals)


def _print_summary(results, metric_name):
    print("\n" + "=" * 70)
    print(f"SUMMARY: gain in {metric_name} over random allocation, positive = "
          f"the rule is better. Min to max over every (fold, seed, random "
          f"draw) pair; the count is how many pairs.")
    for space, srow in results['per_space'].items():
        for m, row in srow.items():
            am = list(row['all_measurements_vs_zero_budget'].values())
            if am:
                print(f"  {space}/{m}: all measurements vs one each "
                      f"{min(am):+.4f} to {max(am):+.4f} over {len(am)} passes")
            for bkey, by_rule in row['rule_vs_random'].items():
                rnd = _span(row.get('random_vs_zero_budget', {}).get(bkey, {}))
                if rnd:
                    print(f"    budget {bkey} of the repeated molecules: "
                          f"random vs none {rnd[0]:+.4f} to {rnd[1]:+.4f}")
                for rule, per in by_rule.items():
                    s = _span(per)
                    if s:
                        print(f"      {rule}: {s[0]:+.4f} to {s[1]:+.4f} "
                              f"over {s[2]} pairs")


def main():
    parser = create_experiment_parser(
        'N36: which molecules to measure a second time')
    parser.add_argument('--budget-shares', type=str, default=None,
                        help='Comma-separated shares of the repeated training '
                             'molecules that get a second measurement (default '
                             f'{",".join(f"{b:g}" for b in BUDGET_SHARES)})')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for which measurement is '
                             'the start and which the spare (default '
                             f'{",".join(str(s) for s in SEEDS)})')
    parser.add_argument('--random-draws', type=int, default=RANDOM_DRAWS,
                        help='Random allocations each rule is paired against')
    parser.add_argument('--oof-splits', type=int, default=OOF_SPLITS,
                        help='Inner folds for the disagreement rule')
    parser.add_argument('--base-rep', type=str, default=None,
                        help="Which representation is the base space. Default: "
                             "pick_noise_base_rep's")
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {'random_draws': args.random_draws, 'oof_splits': args.oof_splits}
    if args.budget_shares:
        kw['budget_shares'] = [float(x) for x in args.budget_shares.split(',')]
    if args.seeds:
        kw['seeds'] = [int(x) for x in args.seeds.split(',')]
    if args.base_rep:
        kw['base_rep'] = args.base_rep
    if args.tag:
        kw['tag'] = args.tag
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
