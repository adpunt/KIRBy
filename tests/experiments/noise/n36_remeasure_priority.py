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
  sparsest                  the mean distance to the k nearest training
                            molecules, largest first.
The three neighbour rules are named <rule>@<neighbour setting>|k<k> on every
row and in every summary (see THE NEIGHBOURS below).

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
candidate among their k nearest neighbours under one neighbour setting. It
counts how many other molecules' neighbourhoods a label feeds, and needs no
held-out molecule. `sparsest` reads the same neighbour search the other way
round: a molecule far from its neighbours feeds few of them.

THE NEIGHBOURS (since 2026-10-05, KNN_SEARCH_SPACE_RECORD.md, Adelaide). The
settings are the dataset group's (n07.sweep_settings): more than 5000 molecule
rows is fast_knn. No importance weighting, the group's distances, k in 5, 10
and 20, and a molecule's scaffold-mates stay in its list. The model's own
matrix (model_input) is cut in both groups, so the bare names `leverage` and
`sparsest` -- 5 neighbours in the model's own standardised features, used
until 2026-10-05 -- are no longer computed. Each neighbour setting's leverage
ranking changes the training labels and so needs its own model fits; only the
3 best (setting, k, counting scheme) combinations of the per-model screen
(n07.screen_settings: how well each setting's neighbours predict each training
molecule's zero-budget label, no model fit) are fitted. The screen table is
written per (fold, seed, configuration, model) under `screen`.

TWO REFERENCES, NOT RULES.
  zero_budget        every candidate on its one start measurement.
  all_measurements   every training molecule on the mean of ALL its
                     measurements. What a loader normally hands you.
A rule's gain over random is read beside the whole gap between the two.

THE MODELS AND CONFIGURATIONS (since 2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT
section 2.4). The dataset's two decided models (DECISIONS.md Step 1). Per outer
fold, each model gets its 2 best single representations (`base_1`, `base_2`)
and its 2 best general hybrids (`hybrid`, `hybrid__general2`), ranked by
searches on that fold's training molecules only
(noise_helpers.fold_configurations): 2 models x 4 = 8 fitted configurations.
The same per-fold searches give the distance hybrid (`hybrid__distance1`) and
the similarity hybrid (`hybrid_similarity`); they are neighbour spaces for the
leverage and sparsest rules only, never fitted. Until 2026-10-05 every recipe
and both single representations came from a search over every row of the
dataset, held-out molecules and held-back second measurements included.

The hybrids are rebuilt per (fold, seed) on the zero-budget training labels,
and every rule and budget shares them. So the only thing differing between two
rules is which candidates got a second measurement. Whether a hybrid rebuilt on
the re-measured labels picks other features is not asked here.

THE ROWS THE SEARCH RUNS ON. One row per molecule (or per drug-target pair on
the DTA sets), the rows n36 fits and grades on, with all of a molecule's
measurements on one side of every split (handoff 2.3 item 14). Where that
collapses the loader's rows (herg_ic50 keeps one row per ChEMBL record), the
per-fold searches are filed under their own name (SEARCH_CONDITION_SUFFIX).

THE HEADLINE IS NOT A MEAN. Every (space, model, budget, rule) writes its paired
per-(fold, seed) difference against each random draw. Nothing is averaged across
models, spaces or datasets. Beside it, `by_fold` gives each fold's mean over its
(seed, random draw) pairs and the spread over folds (noise_helpers.fold_spread;
one fold gives no spread). No significance test is run.

Usage:
    python n36_remeasure_priority.py --dataset herg_ic50 --mode full
    python n36_remeasure_priority.py --dataset bindingdb_kd --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, as_rep_dtype,
    get_domain_metric, get_model_factories,
)
from experiments.decided_grid import decided_models
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n19_measurement_count import measurement_stats
from experiments.noise.n26_noise_variance_from_repeats import (
    measurements_by_molecule, direct_noise_variance,
)
from experiments.noise.noise_helpers import (
    tail_scores, noise_outer_folds, fold_spread, fold_configurations,
    fold_model_inputs, configuration_record, CONFIGURATION_SPACES,
)
from experiments.noise.n07_remediation import (
    sweep_settings, sweep_ks, screen_settings, screen_top, usable_ks,
)
# The D-versus-D' rows, per-pass checkpoint, cached fits, controls and the
# per-fold neighbour spaces shared by n32-n41 (n32's shared block, 2026-09-25).
from experiments.noise.n32_iterative_and_dual_selection import (
    method_setting, fit_predict_cached, ledger_signature, PassLedger,
    restart_flag, arm_rows, arm_result, control_twins, twin_scores,
    controls_enabled, replicate_seed, FoldSimilarity, _nc, _ss,
)

EXP_ID = 'n36_remeasure_priority'

# What this file writes since chat 5 (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT
# section 2.4). In the result header, so the per-pass checkpoint's signature
# (ledger_signature hashes the header) changes and a checkpoint or rows file
# from the whole-dataset design is set aside, never read back.
RESULT_LAYOUT = 'fold_aligned_2026-10-05'

# The per-fold searches run on one row per molecule with its mean label, not on
# the loader's one row per measurement, so they are filed apart from the
# dataset's own per-fold searches (handoff 2.3 item 14, dirty-run bug 11). Used
# only when those rows differ from the loader's.
SEARCH_CONDITION_SUFFIX = 'n36_one_row_per_molecule_mean_label'

# The 4 fitted configurations of each decided model (2 models x 4 = 8, R5):
# base_1, base_2, hybrid, hybrid__general2, all from the per-fold search.
BASE_SPACES = ('base_1', 'base_2')
SPACES = tuple(CONFIGURATION_SPACES)

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

# Neighbours for leverage and for sparsest in neighbour_stats, which run() no
# longer calls (the model's own matrix is cut as a neighbour space, 2026-10-05);
# kept for _smoke_n36's helper check. run() sweeps k in n07.sweep_ks().
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

    Not called by run() since 2026-10-05: it searches the model's own matrix
    (model_input), which the kNN record cut in both groups. run() reads every
    neighbour setting through neighbour_stats_from_lists.

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


def molecule_rows_data(data, stats, dataset_name):
    """The dataset as n36 fits and grades it: one row per molecule.

    2026-10-05 (chat 5, handoff 2.3 item 14, dirty-run bug 11). The loader
    keeps one row per measurement on herg_ic50 (12,008 rows over 9,802
    molecules) and one per measured pair on the DTA sets. Until 2026-10-05 the
    recipes came from a whole-dataset search over those rows while n36 fitted on
    molecules. Now the outer folds, the per-fold searches and every feature
    matrix are on these rows, so all of a molecule's measurements sit on one
    side of every split.

    Representations and augmentation matrices are each molecule's first row
    (`row_index`), float32 (handoff 2.3 item 18; until 2026-10-05 n36 cast them
    to float64 here). Groups are the loader's groups on that row. The label is
    the mean of all the molecule's measurements (`mean_all`), the label n36
    grades held-out molecules on.
    """
    idx = np.asarray(stats['row_index'])
    groups = data.get('groups')
    augs = data.get('augmentations') or None
    smiles = data.get('smiles')
    return {
        'reps': {n: as_rep_dtype(np.asarray(m)[idx])
                 for n, m in data['reps'].items()},
        'augmentations': ({n: as_rep_dtype(np.asarray(m)[idx])
                           for n, m in augs.items()} if augs else None),
        'labels': np.asarray(stats['mean_all'], dtype=np.float64),
        'groups': None if groups is None else np.asarray(groups)[idx],
        'smiles': (None if smiles is None
                   else [smiles[int(i)] for i in idx]),
        'task_type': data['task_type'],
        'dataset_name': dataset_name,
    }


def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N36: which molecules to measure a second time.

    force_rerun (run_dataset.py --force-rerun, via run_all) sets aside the
    rows file and the per-pass checkpoint and recomputes every pass.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    Per outer fold (noise_helpers.noise_outer_folds, on one row per molecule):
      1. noise_helpers.fold_configurations runs this fold's searches on its
         training molecules (decided grid, pruning off) and gives each of the
         dataset's two decided models 2 base representations, 2 general
         hybrids, and the recipes of the two neighbour-only hybrids.
      2. Per seed, the start and spare measurements are drawn, and each
         model's 4 configurations are built on the zero-budget labels
         (fold_model_inputs; hybrids rebuilt from the per-fold recipe).
      3. Configurations two models share (the same matrix) are one cell.
         Per cell and model: the out-of-fold disagreement, the neighbour
         screen, leverage and sparsity under the screen's kept settings, then
         every budget column fitted and scored on the held-out molecules.
    Each (fold, seed) pass is checkpointed as it finishes (PassLedger)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N36: WHICH MOLECULES TO MEASURE AGAIN")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
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

    if data.get('groups') is None:
        print("  SKIPPED: the loader gives no fold groups, and random folds "
              "are not used in this suite.")
        return {'skipped': 'no_fold_groups', 'dataset': dataset_name}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    budgets = [float(b) for b in (kwargs.get('budget_shares') or BUDGET_SHARES)]
    seeds = [int(s) for s in (kwargs.get('seeds') or SEEDS)]
    n_random = int(kwargs.get('random_draws', RANDOM_DRAWS))
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    rules = list(RULES)
    # The rules that read no neighbour list: fitted once per budget whatever
    # the screen keeps.
    plain_rules = [r for r in rules if r not in NEIGHBOUR_RULES]

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

    print(f"  {len(labels)} measurements over {n_ent} molecules; "
          f"{n_multi} of them measured twice or more")
    if n_multi < 10:
        print("  SKIPPED: fewer than 10 molecules measured twice or more, so "
              "no budget share hands out a second measurement to anyone.")
        return {'skipped': 'too_few_repeated_molecules',
                'molecules_measured_twice_or_more': n_multi}

    # ── Models: the dataset's two decided models (R1, DECISIONS.md Step 1) ──
    # Until 2026-10-05: config['models'] (the five DOWNSTREAM_MODELS) with rf
    # and svm appended (n32.downstream_model_list). A `models` kwarg (a smoke)
    # can only narrow the decided pair. knn is never fitted here.
    models_decided = [m for m in decided_models(dataset_name, 'general')
                      if m != 'knn']
    asked = kwargs.get('models')
    downstream_models = ([m for m in models_decided if m in asked] if asked
                         else list(models_decided))
    if asked and [m for m in asked if m not in models_decided]:
        print(f"  models {[m for m in asked if m not in models_decided]} "
              f"dropped: not decided for {dataset_name} ({models_decided})")
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=n_ent)
    models_not_built = {
        m: ('get_model_factories returned no factory (over its row limit at '
            f'{n_ent} molecules, or not defined)')
        for m in downstream_models if m not in model_factories}
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n36: no decided model could be built. Decided {models_decided}, "
            f"got {sorted(model_factories)}.")

    # ── One row per molecule, and the outer folds on those rows ──
    ent_data = molecule_rows_data(data, stats, dataset_name)
    ent_groups = ent_data['groups']
    same_rows = (n_ent == len(labels)
                 and np.array_equal(stats['row_index'], np.arange(n_ent)))
    search_suffix = None if same_rows else SEARCH_CONDITION_SUFFIX
    folds, n_folds = noise_outer_folds(ent_data, config)

    # ── The neighbour settings (KNN_SEARCH_SPACE_RECORD, 2026-10-05) ──
    # The group comes from the molecule rows the neighbours are searched
    # over (n_ent): herg_ic50's 9,802 molecules and the DTA sets' pairs are
    # fast_knn.
    ent_smiles = ent_data['smiles']
    has_smiles = ent_smiles is not None and len(ent_smiles) > 0
    sim_settings = sweep_settings(dataset_name, n_ent, has_smiles)
    sim_ks = tuple(sweep_ks())
    knn_grp = _ss.knn_group(dataset_name, n_ent)

    results = {
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none',
        'grading_label': 'the mean of ALL a held-out molecule\'s measurements',
        'models': downstream_models,
        'models_decided': models_decided,
        'models_not_built': models_not_built,
        'configurations_per_model': list(SPACES),
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'Per outer fold and decided model: the 2 best base '
            'representations and the 2 best general hybrids, ranked by '
            "searches on that fold's training molecules (decided grid, "
            'pruning off; noise_helpers.fold_configurations). Which ones, per '
            'fold, is under folds; a hybrid\'s training-row score is its '
            'score_mean there. Nothing is read from a whole-dataset search.'),
        'search_rows': 'one row per molecule (its first measurement row)',
        'search_labels': ("each training molecule's mean over ALL its "
                          'measurements, the held-back spare included'),
        'search_condition_suffix': search_suffix,
        'n_folds': n_folds,
        'folds_run': [int(f) for f, _, _ in folds],
        'rules': rules,
        'budget_shares': budgets,
        'budget_unit': ('share of the training molecules measured twice or '
                        'more that get their second measurement'),
        'seeds': seeds,
        'random_draws': n_random,
        'oof_splits': oof_splits,
        'feasibility': {
            'n_measurements': int(len(labels)),
            'n_molecules': n_ent,
            'molecules_measured_twice_or_more': n_multi,
        },
        'leverage_definition': (
            '1 + the number of training molecules that have this candidate '
            'among their k nearest neighbours under the neighbour setting '
            'named in the rule (<rule>@<setting>|k<k>)'),
        'knn_group': knn_grp,
        'similarity_settings': [st.name for st in sim_settings],
        'similarity_ks': list(sim_ks),
        'neighbour_screen': {
            'top': screen_top(),
            'note': ('Per (fold, seed, configuration, model): every '
                     '(setting, k, counting scheme) scored with no model fit '
                     'on how well its neighbours predict each training '
                     "molecule's zero-budget label (n07.screen_settings). "
                     'Only the kept combinations get leverage, '
                     'leverage_x_disagreement and sparsest columns. Leverage '
                     'and sparsity do not read the counting scheme, so kept '
                     'combinations that differ only in it are fitted once.'),
        },
        'neighbour_spaces_note': (
            'hybrid__distance1 and hybrid_similarity are neighbour spaces '
            'only; no model is fitted on them. Scaffold-mates stay in every '
            'neighbour list.'),
        'controls_run': controls_enabled(),
        'rows_note': ('D is zero_budget, D\' each rule\'s column, '
                      'score_clean all_measurements: the mean of every '
                      'measurement, the nearest this dataset has to a clean '
                      'label. model_input on a row is the configuration name '
                      '(base_1, hybrid, ...); the model is its own field.'),
        'variance_term_note': (
            'No variance term is used: every candidate goes from 1 measurement '
            'to 2, so the variance drop s2e/1 - s2e/2 = s2e/2 is the same for '
            'all of them and cannot change the order.'),
        'hybrid_note': (
            'The hybrids are rebuilt per (fold, seed) on the zero-budget '
            'training labels from the per-fold recipe and shared by every '
            'rule and budget.'),
        'per_space': {},
    }
    print(f"  models {downstream_models}; {n_folds} outer folds on {n_ent} "
          f"molecule rows, running {results['folds_run']}; {knn_grp}, "
          f"{len(sim_settings)} neighbour settings x ks {list(sim_ks)}, top "
          f"{screen_top()} fitted")

    # scores[space][model][arm][pass_key] = {'score': ..., tail ...}
    # arm is 'zero_budget', 'all_measurements', or f'{budget}|{rule}|{r}'. r is
    # the random draw for 'random' and always 0 for a ranking rule.
    scores = {s: {m: {} for m in downstream_models} for s in SPACES}
    s2e_by_pass, errors, fold_records, screens = {}, {}, {}, []
    t0 = time.time()
    # The neighbour rules, in the order they were first scored; handed to
    # _summarise beside the plain rules.
    extra_rules = []
    # Rows and the per-pass checkpoint, 2026-09-25 (n32's PassLedger). The
    # signature hashes the header above, which carries RESULT_LAYOUT.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, kwargs.get('tag'),
                        signature=ledger_signature(
                            data, config, results,
                            [(tr, va) for _, tr, va in folds]),
                        restart=bool(force_rerun) or restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        scores, s2e_by_pass, errors = (st_['scores'], st_['s2e_by_pass'],
                                       st_['errors'])
        extra_rules = list(st_['extra_rules'])
        fold_records = st_.get('fold_records') or {}
        screens = list(st_.get('screens') or [])
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    def _state():
        return {'scores': scores, 's2e_by_pass': s2e_by_pass,
                'errors': errors, 'extra_rules': extra_rules,
                'fold_records': fold_records, 'screens': screens}

    for fold_i, tr, va in folds:
        if all(ledger.pass_done(f'f{fold_i}s{seed}') for seed in seeds):
            print(f"  fold {fold_i}: every seed on disk in the checkpoint")
            continue
        g_tr = ent_groups[tr]
        y_va = y_mean_all[va]
        cand = np.where(k_all[tr] >= 2)[0]
        if len(cand) == 0:
            errors[f'f{fold_i}'] = 'no training molecule measured twice or more'
            continue
        # The training fold's own pooled within-molecule variance. Only
        # training molecules' measurements enter it. Recorded for the reader;
        # no rule uses it (see WHY NO VARIANCE TERM in the header).
        s2e = direct_noise_variance([values[j] for j in tr])['s2e']

        # The per-fold searches, on this fold's training molecules.
        # OPEN (chat 5): they rank on each training molecule's mean over ALL
        # its measurements (ent_data['labels']), so a candidate's held-back
        # spare enters the choice of representations and hybrid recipes,
        # while the zero_budget arm trains on the start measurement only. The
        # other choice is a search per seed on that seed's start labels (two
        # searches per fold instead of one). Left on the mean, the label
        # n19 and n26 search on. 2026-10-05.
        try:
            entries = fold_configurations(
                ent_data, dataset_name, fold_i=fold_i, train_idx=tr,
                n_folds=n_folds, condition_suffix=search_suffix)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            errors[f'f{fold_i}|fold_configurations'] = (
                f'{type(exc).__name__}: {exc}')
            fold_records[str(fold_i)] = {
                'error': f'per-fold searches failed: {type(exc).__name__}: '
                         f'{exc}'}
            continue
        frec = fold_records.setdefault(str(fold_i), {
            'n_train': int(len(tr)), 'n_val': int(len(va)),
            'candidates': int(len(cand)), 'models': {}})
        for m in downstream_models:
            rec = configuration_record(entries.get(m))
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            frec['models'].setdefault(m, rec)
            print(f"    fold {fold_i} {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")

        for seed in seeds:
            pass_key = f'f{fold_i}s{seed}'
            if ledger.pass_done(pass_key):
                continue
            pass_rows = []
            s2e_by_pass[pass_key] = s2e
            start_all, spare_all = draw_start_and_spare(values, seed)
            start_tr, spare_tr = start_all[tr], spare_all[tr]

            # Each model's 4 configurations, built on the zero-budget labels.
            inputs = {}
            for m in downstream_models:
                ins, unav, names = fold_model_inputs(
                    ent_data, dataset_name, m, entries.get(m),
                    train_idx=tr, val_idx=va, y_train=start_tr,
                    factory=model_factories[m], metric=metric)
                inputs[m] = ins
                mrec = frec['models'][m]
                if names:
                    mrec.setdefault('configuration_names', {})[
                        f's{seed}'] = dict(names)
                for sp, (Xa, _Xb) in ins.items():
                    mrec.setdefault('n_features', {}).setdefault(
                        sp, int(np.shape(Xa)[1]))
                for sp in SPACES:
                    if sp not in ins:
                        why = unav.get(sp) or 'not built'
                        (mrec.setdefault('configurations_unavailable', {})
                         .setdefault(sp, why))
                        errors[f'{sp}/{m}|{pass_key}|build'] = why

            # A configuration two models share (the same matrix) is one cell:
            # the disagreement and the neighbour lists read only the features
            # and the labels, so they are computed once and both models are
            # fitted on it.
            cells = []
            for space in SPACES:
                by_hash = {}
                for m in downstream_models:
                    if space not in inputs[m]:
                        continue
                    X_tr_s, X_va_s = inputs[m][space]
                    h = (_nc.array_hash(np.asarray(X_tr_s)),
                         _nc.array_hash(np.asarray(X_va_s)))
                    if h in by_hash:
                        by_hash[h][2].append(m)
                    else:
                        by_hash[h] = [X_tr_s, X_va_s, [m]]
                for X_tr_s, X_va_s, ms in by_hash.values():
                    cells.append((space, X_tr_s, X_va_s, ms))

            # The neighbour spaces of this fold, one per model, built on the
            # zero-budget labels. Its 4 configurations are handed over, not
            # rebuilt; the distance and similarity hybrids are built here.
            fsims = {}
            for m in downstream_models:
                entry = entries.get(m)
                if entry is None:
                    continue
                fsim = FoldSimilarity(
                    dataset_name, ent_data, reps=ent_data['reps'],
                    train_idx=tr, val_idx=va, y_train=start_tr,
                    groups_train=g_tr,
                    base_by_model={m: list(entry.get('base') or [])},
                    recipe_for_space_by_model={
                        m: dict(entry.get('recipes') or {})},
                    model_factories=model_factories, metric=metric,
                    task_type=task_type,
                    augmentations=ent_data['augmentations'],
                    smiles=ent_smiles)
                for sp, (Xa, Xb) in inputs[m].items():
                    fsim._spaces[(m, sp)] = (Xa, Xb, None)
                for sp, why in (entry.get('unavailable') or {}).items():
                    if sp not in inputs[m]:
                        fsim._spaces[(m, sp)] = (None, None, why)
                        fsim.unavailable[f'{m}|{sp}'] = why
                fsims[m] = fsim

            for space, X_tr_s, X_va_s, cell_models in cells:
                _ekey = (space if cell_models == list(downstream_models)
                         else f'{space}/{cell_models[0]}')
                # What every rule reads, once per cell.
                try:
                    # Cached on the label hash (2026-09-25).
                    # OPEN (chat 5): the disagreement rule's out-of-fold
                    # prediction is LightGBM (noisy_labels.oof_predict),
                    # which is not a decided model on bindingdb_kd, kiba
                    # or herg_ic50. It only ranks candidates and is never
                    # scored. Left as it was. 2026-10-05.
                    oof = _nc.cached_oof_predict(
                        dataset_name, X_tr_s, start_tr, g_tr,
                        n_splits=oof_splits, seed=seed)
                    disagreement = np.abs(start_tr - oof)
                except Exception as exc:
                    errors[f'{_ekey}|{pass_key}|oof'] = (
                        f'{type(exc).__name__}: {exc}')
                    disagreement = None

                # {model: [(rule suffix, leverage, sparsity)]}: the screen's
                # kept (setting, k) of each model. The neighbour lists of a
                # per-model space (base, augmented, hybrid) belong to that
                # model, so each model has its own screen.
                kept_stats = {m: [] for m in cell_models}
                for m in cell_models:
                    fsim = fsims.get(m)
                    if fsim is None:
                        continue
                    nbrs = {}
                    for setting in sim_settings:
                        try:
                            nb = fsim.neighbours(setting, m, X_tr_s)
                        except Exception as exc:
                            nb = {'available': False,
                                  'reason': f'{type(exc).__name__}: {exc}'}
                        if not nb.get('available'):
                            errors.setdefault('similarity_unavailable', {})[
                                f'{setting.name}|{m}'] = nb.get('reason')
                            continue
                        nbrs[setting.name] = nb
                    kept, table = screen_settings(
                        sim_settings, nbrs, start_tr, metric=metric,
                        is_clf=False, ks=sim_ks)
                    screens.append({
                        'fold': int(fold_i), 'seed': int(seed),
                        'configuration': space, 'model': m,
                        'kept': [f'{s_.name}|k={k_}|{sch}'
                                 for s_, k_, sch in kept],
                        'table': table})
                    # Leverage and sparsity do not read the counting scheme:
                    # kept combinations differing only in it are one column.
                    # OPEN (chat 5): so fewer than 3 (setting, k) can be
                    # fitted on a pass. Whether to fill up to 3 distinct
                    # (setting, k) from further down the screen is
                    # Adelaide's call. 2026-10-05.
                    seen = set()
                    for s_, k_, _sch in kept:
                        if (s_.name, k_) in seen:
                            continue
                        seen.add((s_.name, k_))
                        try:
                            i_, d_ = _ss.neighbours_at(nbrs[s_.name], k_)
                        except Exception as exc:
                            errors[f'{_ekey}|{m}|{s_.name}|k{k_}|{pass_key}'] = (
                                f'{type(exc).__name__}: {exc}')
                            continue
                        lev, spa = neighbour_stats_from_lists(i_, d_)
                        kept_stats[m].append((f'@{s_.name}|k{int(k_)}',
                                              lev, spa))

                # Every label column this cell fits on. model_columns holds
                # the neighbour-rule columns of one model.
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
                        # tie-break, and paired against every random draw.
                        # Only 'random' is redrawn.
                        draws = range(n_random) if rule == 'random' else (0,)
                        if rule in plain_rules:
                            variants = [(None, '', None, None)]
                        else:
                            variants = [(m, sfx, lev, spa)
                                        for m in cell_models
                                        for sfx, lev, spa in kept_stats[m]]
                        for per_m, sfx, lev, spa in variants:
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
                            errors[f'{_ekey}|{mname}|{arm}|{pass_key}'] = (
                                f'{type(exc).__name__}: {exc}')
                            continue
                        sc_ = _score(metric_fn, y_va, pred)
                        scores[space][mname].setdefault(arm, {})[
                            pass_key] = sc_
                        pass_scores.setdefault(arm, {})[mname] = \
                            sc_['score']
                        if arm in ('zero_budget', 'all_measurements'):
                            # The configuration's held-out score, per seed.
                            (frec['models'][mname]
                             .setdefault('held_out_score', {})
                             .setdefault(space, {})
                             .setdefault(f's{seed}', {}))[arm] = sc_['score']
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
                # model_input is the configuration name, the model its own
                # field. Until 2026-10-05 it was '<space>/<first model of the
                # cell>' on every model's rows, as n32 wrote it (dirty-run
                # bug 13).
                pass_rows += arm_rows(
                    dataset_name=dataset_name, exp_id=EXP_ID,
                    metric=metric,
                    condition=results['noise_regime'], level=None,
                    draw=seed, fold=fold_i, model_input=space,
                    scores_by_arm=pass_scores, reference='zero_budget',
                    clean='all_measurements',
                    setting_of=_n36_row_setting)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(pass_key, _state())
            print(f"    fold {fold_i} seed {seed} done, {len(cells)} "
                  f"configuration cells, {len(pass_rows)} D-vs-D' rows "
                  f"[{time.time() - t0:.0f}s]")

    results['folds'] = convert_numpy(fold_records)
    results['screen'] = convert_numpy(screens)
    results['s2e_training_fold_by_pass'] = s2e_by_pass
    results['per_space'] = _summarise(scores, SPACES, downstream_models,
                                      budgets, plain_rules + extra_rules,
                                      n_random,
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
    '<budget>|<rule>[@<setting>|k<k>]|<draw>[|<twin>]' for its row.

    A rule that reads no neighbour list (random, disagreement) is written with
    similarity setting 'none' since 2026-10-05; until then it carried
    model_input's name although it read no neighbours."""
    parts = arm.split('|')
    twin = ''
    if parts[-1] in ('random_twin', 'shuffled_twin'):
        twin = '|' + parts.pop()
    budget, draw = parts[0], parts[-1]
    rule_part = '|'.join(parts[1:-1])
    rule, _, sim = rule_part.partition('@')
    k = None
    sim_name = 'none'
    if sim:
        bits = sim.split('|')
        k = bits[3][1:] if len(bits) > 3 else None
        sim_name = '|'.join(bits[:3])
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
            # Over folds (2026-10-05): each fold's mean over its (seed,
            # random draw) pairs, then noise_helpers.fold_spread across the
            # folds. One fold gives 'one fold, no spread'.
            row['by_fold'] = {
                'all_measurements_vs_zero_budget': _fold_spread_of(
                    {'0': row['all_measurements_vs_zero_budget']}),
                'random_vs_zero_budget': {
                    bk: _fold_spread_of(per) for bk, per in
                    (row.get('random_vs_zero_budget') or {}).items()},
                'rule_vs_random': {
                    bk: {ru: _fold_spread_of(per) for ru, per in by.items()}
                    for bk, by in row['rule_vs_random'].items()},
            }
            srow[m] = row
        out[space] = srow
    return out


def _fold_of(pass_key):
    """The fold number of a pass key 'f<fold>s<seed>'."""
    return int(str(pass_key)[1:].split('s', 1)[0])


def _fold_spread_of(per_draw):
    """fold_spread over folds of {draw: {pass_key: gain}}: each fold's value is
    the mean of its gains over seeds and draws."""
    by_fold = {}
    for per in (per_draw or {}).values():
        for p, v in (per or {}).items():
            if v is not None and np.isfinite(v):
                by_fold.setdefault(_fold_of(p), []).append(float(v))
    folds = sorted(by_fold)
    out = fold_spread([float(np.mean(by_fold[f])) for f in folds])
    out['folds'] = folds
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
                        fs = (row.get('by_fold', {}).get('rule_vs_random', {})
                              .get(bkey, {}).get(rule) or {})
                        tail = ''
                        if fs.get('n_folds', 0) > 1:
                            tail = (f"; mean over {fs['n_folds']} folds "
                                    f"{fs['mean']:+.4f}, sd {fs['sd']:.4f}")
                        elif fs.get('n_folds') == 1:
                            tail = '; one fold, no spread'
                        print(f"      {rule}: {s[0]:+.4f} to {s[1]:+.4f} "
                              f"over {s[2]} pairs{tail}")


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
