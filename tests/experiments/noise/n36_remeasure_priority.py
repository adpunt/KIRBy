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

WHERE THE HYBRID COMES IN. Each downstream model runs in its own top 2 single
representations and in one hybrid per family of stored search, each recipe the
one the pool picked FOR that model, as n23 and n26 do. The hybrid is rebuilt per
(fold, seed) on the zero-budget training labels, and every rule and budget
shares it. So the only thing differing between two rules is which candidates got
a second measurement. Whether a hybrid rebuilt on the re-measured labels picks
other features is not asked here.

THE SINGLE-REPRESENTATION COMPARISON is each downstream model's top 2 base
representations by that model's own score in the stored search -- spaces
`base_1` and `base_2` -- the same rule as the rest of the study
(_shared.noise_base_reps_by_model, owner's decision 2026-09-25). Which
representations those are is written to `base_reps_by_model`; a model the pool
holds no stored search for has none, and `base_reps_missing` says why.

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
from experiments.noise.n19_measurement_count import measurement_stats
from experiments.noise.n26_noise_variance_from_repeats import (
    measurements_by_molecule, direct_noise_variance,
)
from experiments.noise.noise_helpers import (
    hybrid_for_fold, pool_recipe_per_downstream_model, group_models_by_recipe,
    tail_scores,
)
from kirby.utils.noisy_labels import oof_predict
# The D-versus-D' rows, per-pass checkpoint, cached builds, controls and the
# similarity axis shared by n32-n41 (n32's shared block, 2026-09-25).
from experiments.noise.n32_iterative_and_dual_selection import (
    method_setting, downstream_model_list, fit_predict_cached,
    scaled_base_cells, cached_hybrid, ledger_signature, PassLedger,
    restart_flag, arm_rows, arm_result, control_twins, twin_scores,
    controls_enabled, replicate_seed, sweep_settings, sweep_ks,
    is_model_input, setting_models, similarity_key, FoldSimilarity, _nc, _ss,
)

EXP_ID = 'n36_remeasure_priority'

# Which searches the recipe may come from. Same as n19, n23, n24 and n26.
POOL_SUBSET = 'predictive_and_similarity'

SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}
# base_1 and base_2 are each downstream model's rank-1 and rank-2 single
# representation (see the module docstring).
BASE_SPACES = tuple(f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1))

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

    ⛔ THE DISTANCE CHANGED 2026-09-25. Until then this was a bare
    sklearn NearestNeighbors, whose default is Minkowski p=2 -- Euclidean --
    whatever --distance-metric said (plan section 4: "Euclidean slips in
    unasked", this line). It is now _shared.make_nearest_neighbors, the
    process-wide distance (manhattan by default), which is what the
    model_input similarity setting means everywhere else. So n36's leverage
    and sparsest rules under model_input are NOT today's numbers: they move to
    the process-wide distance, as the owner asked (2026-09-25). Every other
    setting reads its neighbours through neighbour_stats_from_lists.
    """
    from experiments._shared import make_nearest_neighbors
    n = X_tr.shape[0]
    k = min(n_neighbours, n - 1)
    if k < 1:
        return np.ones(n), np.zeros(n)
    nn = make_nearest_neighbors(n_neighbors=k + 1).fit(X_tr)
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


def neighbour_stats_from_lists(idx, dist):
    """Leverage and sparsity from neighbour lists a similarity setting gave
    (similarity_space.neighbours_at): the molecule itself is already excluded
    by index there. Same two quantities as neighbour_stats."""
    idx = np.asarray(idx)
    n = idx.shape[0]
    if idx.shape[1] < 1:
        return np.ones(n), np.zeros(n)
    leverage = 1.0 + np.bincount(idx.ravel(), minlength=n).astype(float)
    sparsity = np.asarray(dist, dtype=float).mean(axis=1)
    return leverage, sparsity


# The rules that read the neighbour search, and so take a similarity setting.
NEIGHBOUR_RULES = ('leverage', 'leverage_x_disagreement', 'sparsest')


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


def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N36: which molecules to measure a second time.

    force_rerun (run_dataset.py --force-rerun, via run_all) sets aside the
    rows file and the per-pass checkpoint and recomputes every pass.

    THE SIMILARITY AXIS, added 2026-09-25 (plan section 1). leverage,
    leverage_x_disagreement and sparsest read a neighbour search. Under the
    model_input setting at k = N_NEIGHBOURS they keep their names; every other
    (setting, k) of the sweep (n32.sweep_settings, n32.sweep_ks) adds the
    rules <rule>@<setting>|k<k>, paired against the same random draws. A
    per-model space is built for each model of the cell and that model fits
    its own columns."""
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

    downstream_models = downstream_model_list(kwargs, config)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=n_ent)
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n36: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none',
        'grading_label': 'the mean of ALL a held-out molecule\'s measurements',
        'models': downstream_models,
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
            f'among their {N_NEIGHBOURS} nearest neighbours in the similarity '
            'setting; under model_input, the process-wide distance '
            '(manhattan by default; Euclidean before 2026-09-25) in the '
            'standardised space being scored'),
        'similarity_settings': [st.name for st in sweep_settings()],
        'similarity_ks': list(sweep_ks(N_NEIGHBOURS)),
        'controls_run': controls_enabled(),
        'rows_note': ('D is zero_budget, D\' each rule\'s column, '
                      'score_clean all_measurements: the mean of every '
                      'measurement, the nearest this dataset has to a clean '
                      'label.'),
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

    spaces = base_spaces_run + tuple(ordered_spaces)
    results['spaces_run'] = list(spaces)
    results['hybrid_available'] = bool(ordered_spaces)
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    results['hybrid_recipe_by_model'] = convert_numpy(recipe_record)
    print(f"  spaces {list(spaces)}")

    splits = list(cv_split(ent_groups, n_folds, next(iter(ent_reps.values())),
                           y_mean_all, task_type='regression'))

    # scores[space][model][arm][pass_key] = {'score': ..., tail ...}
    # arm is 'zero_budget', 'all_measurements', or f'{budget}|{rule}|{r}'. r is
    # the random draw for 'random' and always 0 for a ranking rule.
    scores = {s: {m: {} for m in downstream_models} for s in spaces}
    s2e_by_pass, errors = {}, {}
    t0 = time.time()
    metric = (metric_name, metric_fn, lower_is_better)
    sim_settings = sweep_settings()
    sim_ks = sweep_ks(N_NEIGHBOURS)
    # The neighbour rules added under the similarity axis, in the order they
    # were first scored; handed to _summarise beside the base rules.
    extra_rules = []
    # The augmentations and SMILES aligned to the molecule rows (ent_*).
    ent_aug = ({a: np.asarray(v)[stats['row_index']]
                for a, v in (data.get('augmentations') or {}).items()}
               or None)
    ent_smiles = (None if data.get('smiles') is None
                  else [data['smiles'][i] for i in stats['row_index']])
    # Rows and the per-pass checkpoint, 2026-09-25 (n32's PassLedger).
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, kwargs.get('tag'),
                        signature=ledger_signature(data, config, results,
                                                   splits),
                        restart=bool(force_rerun) or restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        scores, s2e_by_pass, errors = (st_['scores'], st_['s2e_by_pass'],
                                       st_['errors'])
        extra_rules = list(st_['extra_rules'])
        print(f"  resumed {len(ledger.passes_done)} finished passes")

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
            if ledger.pass_done(pass_key):
                continue
            pass_rows = []
            s2e_by_pass[pass_key] = s2e
            start_all, spare_all = draw_start_and_spare(values, seed)
            start_tr, spare_tr = start_all[tr], spare_all[tr]

            # {space: [(X_tr, X_va, models)]}, built on the zero-budget labels.
            # Each single-representation space holds one matrix per distinct
            # representation at that rank, shared by the models that rank it
            # there.
            fold_spaces = {}
            for _k, _bsp in enumerate(BASE_SPACES, start=1):
                if _bsp not in spaces:
                    continue
                # Standardised once per fold through noise_cache (2026-09-25).
                fold_spaces[_bsp] = scaled_base_cells(
                    dataset_name, ent_reps, tr, va, base_by_model, _k,
                    downstream_models)
            for _sp in spaces:
                if _sp in BASE_SPACES:
                    continue
                _cells = []
                for _bp, _mnames, _shareable in group_models_by_recipe(
                        {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                         for _m in downstream_models}, downstream_models):
                    # Cached per (fold, recipe, labels), 2026-09-25.
                    X_tr_h, X_va_h = cached_hybrid(
                        dataset_name, train_reps, val_reps, start_tr, _bp,
                        g_tr, _mnames[0], model_factories, metric, tr,
                        _shareable)
                    if X_tr_h is None:
                        errors[f'{_sp}/{_mnames[0]}|{pass_key}|build'] = (
                            'the hybrid could not be built on this fold')
                        continue
                    sh = StandardScaler()
                    _cells.append((sh.fit_transform(X_tr_h),
                                   sh.transform(X_va_h), list(_mnames)))
                if _cells:
                    fold_spaces[_sp] = _cells

            # The similarity spaces of this fold, from the zero-budget
            # labels (plan section 1, 2026-09-25).
            fsim = FoldSimilarity(
                dataset_name, data, reps=ent_reps, train_idx=tr, val_idx=va,
                y_train=start_tr, groups_train=g_tr,
                base_by_model=base_by_model,
                recipe_for_space_by_model=recipe_for_space_by_model,
                model_factories=model_factories, metric=metric,
                task_type=task_type, augmentations=ent_aug,
                smiles=ent_smiles)

            for space in spaces:
                for X_tr_s, X_va_s, cell_models in fold_spaces.get(space, ()):
                    _ekey = (space if cell_models == list(downstream_models)
                             else f'{space}/{cell_models[0]}')
                    # What every rule reads, once per cell.
                    try:
                        # Cached on the label hash (2026-09-25).
                        oof = _nc.cached_oof_predict(
                            dataset_name, X_tr_s, start_tr, g_tr,
                            n_splits=oof_splits, seed=seed)
                        disagreement = np.abs(start_tr - oof)
                    except Exception as exc:
                        errors[f'{_ekey}|{pass_key}|oof'] = (
                            f'{type(exc).__name__}: {exc}')
                        disagreement = None

                    # {(suffix, model or None): (leverage, sparsity)}: the
                    # neighbour search of every (setting, k). Suffix '' with
                    # model None is model_input at N_NEIGHBOURS, today's rule
                    # names; a per-model space gives one entry per model.
                    nstats, sim_unavail = {}, {}
                    for setting in sim_settings:
                        for model in setting_models(setting, cell_models):
                            if is_model_input(setting):
                                for k in sim_ks:
                                    nstats[(similarity_key(
                                        setting, k, today_k=N_NEIGHBOURS),
                                        None)] = neighbour_stats(X_tr_s, k)
                                continue
                            nb = fsim.neighbours(setting, model, X_tr_s)
                            if not nb.get('available'):
                                sim_unavail[f'{setting.name}|{model}'] = \
                                    nb.get('reason')
                                continue
                            per_m = (model if setting.space
                                     in _ss.FEATURE_SPACES else None)
                            for k in sim_ks:
                                if nb['k_eff'] < k:
                                    continue
                                i_, d_ = _ss.neighbours_at(nb, k)
                                nstats[(similarity_key(setting, k), per_m)] = \
                                    neighbour_stats_from_lists(i_, d_)
                    if sim_unavail:
                        errors.setdefault('similarity_unavailable', {}).update(
                            sim_unavail)

                    # Every label column this cell fits on. model_columns
                    # holds the columns a per-model space gave one model.
                    columns = {'zero_budget': start_tr,
                               'all_measurements': y_mean_all[tr]}
                    model_columns = {m: {} for m in cell_models}
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
                            variants = ([(k_, m_) for k_, m_ in nstats]
                                        if rule in NEIGHBOUR_RULES
                                        else [('', None)])
                            for sfx, per_m in variants:
                                lev, spa = nstats.get((sfx, per_m),
                                                      nstats.get(('', None),
                                                                 (None, None)))
                                rname = f'{rule}{sfx}'
                                if sfx and rname not in extra_rules:
                                    extra_rules.append(rname)
                                for r in draws:
                                    ranked = rank_candidates(
                                        rule, cand, disagreement=disagreement,
                                        leverage=lev, sparsity=spa,
                                        random_seed=1000 * seed + 17 * fold_i + r)
                                    col = labels_after_budget(
                                        start_tr, spare_tr, ranked[:n_b])
                                    if per_m is None:
                                        columns[f'{b:g}|{rname}|{r}'] = col
                                    else:
                                        model_columns[per_m][
                                            f'{b:g}|{rname}|{r}'] = col
                    pass_scores = {}
                    for mname in cell_models:
                        for arm, y_fit in list(columns.items()) + list(
                                model_columns[mname].items()):
                            try:
                                # Through noise_cache for a seeded model
                                # (2026-09-25); the same prediction.
                                pred = np.asarray(fit_predict_cached(
                                    dataset_name, mname,
                                    model_factories[mname], X_tr_s, y_fit,
                                    X_va_s, metric_name),
                                    dtype=np.float64).ravel()
                            except Exception as exc:
                                errors[f'{space}|{mname}|{arm}|{pass_key}'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            sc_ = _score(metric_fn, y_va, pred)
                            scores[space][mname].setdefault(arm, {})[
                                pass_key] = sc_
                            pass_scores.setdefault(arm, {})[mname] = \
                                sc_['score']
                    # ── Controls (KIRBY_NOISE_CONTROLS=1): every budget
                    # column is a label move, twinned by controls.py. n36's own
                    # 'random' rule is kept; it draws among the candidates,
                    # the twins among every training molecule. ──
                    if controls_enabled():
                        for mname in cell_models:
                            for arm, y_fit in list(columns.items()) + list(
                                    model_columns[mname].items()):
                                if arm in ('zero_budget', 'all_measurements'):
                                    continue
                                twins = control_twins(
                                    arm_result(start_tr, y=y_fit), X_tr_s,
                                    start_tr, (replicate_seed(arm), seed,
                                               fold_i))
                                for tw, by_m in twin_scores(
                                        dataset_name, twins, model_factories,
                                        [mname], X_va_s, y_va,
                                        metric).items():
                                    pass_scores.setdefault(
                                        f'{arm}|{tw}', {}).update(by_m)
                    pass_rows += arm_rows(
                        dataset_name=dataset_name, exp_id=EXP_ID,
                        metric=metric,
                        condition=results['noise_regime'], level=None,
                        draw=seed, fold=fold_i, model_input=_ekey,
                        scores_by_arm=pass_scores, reference='zero_budget',
                        clean='all_measurements',
                        setting_of=_n36_row_setting)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(pass_key, {
                'scores': scores, 's2e_by_pass': s2e_by_pass,
                'errors': errors, 'extra_rules': extra_rules})
            print(f"    fold {fold_i} seed {seed} done, {len(pass_rows)} "
                  f"D-vs-D' rows [{time.time() - t0:.0f}s]")

    results['s2e_training_fold_by_pass'] = s2e_by_pass
    results['per_space'] = _summarise(scores, spaces, downstream_models,
                                      budgets, rules + extra_rules, n_random,
                                      lower_is_better)
    results['rules_under_similarity_settings'] = extra_rules
    results['d_vs_dprime'] = ledger.describe()
    if errors:
        results['_errors'] = errors
    results = convert_numpy(results)
    _print_summary(results, metric_name)
    save_experiment_results(results, output_dir, 'n36_remeasure_priority',
                            dataset_name, filename_suffix=kwargs.get('tag'))
    return results


def _n36_row_setting(arm):
    """(method, method_setting, similarity_setting) of an n36 budget column
    '<budget>|<rule>[@<setting>|k<k>]|<draw>[|<twin>]' for its row."""
    parts = arm.split('|')
    twin = ''
    if parts[-1] in ('random_twin', 'shuffled_twin'):
        twin = '|' + parts.pop()
    budget, draw = parts[0], parts[-1]
    rule_part = '|'.join(parts[1:-1])
    rule, _, sim = rule_part.partition('@')
    k = None
    sim_name = _ss.MODEL_INPUT.name
    if sim:
        bits = sim.split('|')
        k = bits[3][1:] if len(bits) > 3 else None
        sim_name = '|'.join(bits[:3])
    elif rule in NEIGHBOUR_RULES:
        k = N_NEIGHBOURS
    return (f'{rule}{twin}',
            method_setting(budget_share=budget, random_draw=draw, k=k),
            sim_name)


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
    if args.tag:
        kw['tag'] = args.tag
    run(data, config, args.output, args.dataset, **kw)
    # run() saves its own result file (n36 has no sweep_conditions).


if __name__ == '__main__':
    main()
