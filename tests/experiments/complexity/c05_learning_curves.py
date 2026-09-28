#!/usr/bin/env python3
"""C05: Learning curves -- sample efficiency.

Paper: Complexity
Conclusion: Hybrids are more sample-efficient than individual reps.
Dependencies: None (standalone).

At each sample size fraction, runs find_best_hybrid on subsample to discover
the optimal config for that data regime. Tests whether the methodology adapts
to different data scales and whether the hybrid advantage persists.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, fit_learning_curve,
    find_best_hybrid, get_domain_metric,
    extract_top3_base_reps,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    resolve_comparison_models,
    INTERNAL_EVALUATORS,
    get_best_search_for_downstream, build_fold_hybrid, cv_split,
    build_fold_reselection_input, fold_reselection,
    hr_exclude_folds,
)
from sklearn.preprocessing import StandardScaler
from kirby.hybrid import _graded_scores_for_clf


# find_best_hybrid's default seed. c05 never passes one, so the search's folds
# are built with this; _min_train_rows must build the same folds.
_SEARCH_SEED = 42


def _min_train_rows(sub_groups, sub_labels, n_folds, task_type):
    """Training rows in the smallest training partition of the search's folds.

    Uses the splitter find_best_hybrid uses (_make_cv_splitter: GroupKFold, or
    StratifiedGroupKFold with seed 42 for classification), on the rows the
    group draw actually returned. Falls back to (k-1)/k of those rows if the
    splitter cannot be built here; the search then raises on its own.
    """
    n = len(sub_labels)
    try:
        from kirby.find_best_hybrid import _make_cv_splitter
        splitter, use_groups = _make_cv_splitter(
            sub_groups, n_folds, _SEARCH_SEED, task_type=task_type)
        X = np.zeros((n, 1))
        args = (X, sub_labels, sub_groups) if use_groups else (X, sub_labels)
        return int(min(len(tr) for tr, _ in splitter.split(*args)))
    except Exception:
        return int(n * (n_folds - 1) / n_folds) if n_folds > 1 else int(n * 0.8)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Learning curves: hybrid advantage vs sample size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C05: LEARNING CURVES")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    models = get_model_factories(config['models'],
                                    task_type=data['task_type'],
                                    n_samples=len(data['labels']))
    n_folds = config['n_splits']
    # Canonical default (FULL_BUDGETS) — old default [25, 50, 100, 200, 500] had
    # 200 which is NOT in FULL_BUDGETS, causing silent blueprint misses.
    from experiments._shared import FULL_BUDGETS
    sweep_budgets = config.get('budgets', FULL_BUDGETS)
    n_repeats = kwargs.get('n_repeats', 3)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_total = len(labels)
    model_names = list(models.keys())

    from experiments._shared import LARGE_DATASET_N, LARGE_SUBSAMPLE_K
    if n_total > LARGE_DATASET_N:
        # Large dataset: keep a multi-point curve but cap it at K rows so we
        # never run the giant full-data searches. Absolute sizes, not %-of-n.
        sample_sizes = [s for s in [250, 500, 1000, LARGE_SUBSAMPLE_K]
                        if s < n_total]
        print(f"  [size-gate] n={n_total} > {LARGE_DATASET_N}: learning curve "
              f"capped at {LARGE_SUBSAMPLE_K} rows ({sample_sizes})")
    else:
        # ⛔ CURVE POINTS, NOT ROWS. Every point still draws from the whole
        # dataset and the last point is every row of it; what falls is how many
        # points the curve has.
        #
        # Shipped: five points at 0.10, 0.25, 0.50, 0.75 and 1.0 of the rows.
        # Under KIRBY_LOCAL_GRID: three, at 0.10, 0.50 and 1.0. Three is the
        # floor — fit_learning_curve is only called where at least three points
        # scored, and below that this experiment has no curve to fit.
        #
        # Measured 2026-09-25 on flab_thermo_tresanco, all 672 antibodies, on a
        # laptop running about 20 other experiment processes (load average 800,
        # this process getting a quarter of one core): 26 s for the
        # 67-antibody point and 414 s for the 168-antibody point, both over two
        # seeds, against a 2400 s cap for the whole experiment. The 168- and
        # 504-antibody points are the two dropped, and 504 is the most
        # expensive point below the full-data one: it is the first size whose
        # training partition (336 rows) admits both swept budgets, so its
        # search is 12 configurations where 168's is 6.
        #
        # KIRBY_C05_FRACTIONS overrides both, e.g. 0.10,0.25,0.50,0.75,1.0 to
        # put the five back. Unset on the cluster, nothing changes.
        _fr = os.environ.get('KIRBY_C05_FRACTIONS')
        if _fr:
            fractions = [float(x) for x in _fr.split(',') if x.strip()]
        elif os.environ.get('KIRBY_LOCAL_GRID'):
            fractions = [0.10, 0.50, 1.0]
        else:
            fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
        sample_sizes = [max(50, int(n_total * f)) for f in fractions]
    # Deduplicate and cap at total
    sample_sizes = sorted(set(min(s, n_total) for s in sample_sizes))

    print(f"  Models: {model_names}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Budgets: {sweep_budgets}")
    print(f"  Repeats: {n_repeats}")
    print(f"  Search ")

    last_search = None
    # The number of molecules subsampled, and the seed, of the search that
    # `last_search` ends up holding. Tracked because `last_search` is reset to
    # None at the top of every sample size, so which sample size and seed it
    # came from is not readable from the key names alone.
    last_search_n = None
    last_search_seed = None
    search_failures = []
    results = {'sample_sizes': sample_sizes, 'budgets': sweep_budgets,
                'models': {},
                # FIX 4d: explicit log of (n, model) all-rows-blueprint
                # second-pass points that had too few scaffold groups for the
                # requested fold count. cv_split raises (no random-split
                # fallback) on groups < fold count, so rather than let that
                # propagate or be swallowed into an empty score we reduce folds
                # to the available groups (min 2) or record a skip here.
                'skipped_insufficient_groups': [],
                # Honest per-fold rescore blocks (CONTRACT shape) nested by
                # sample size. For each sample size we store one block PER
                # subsampling seed (subsamples differ across seeds, so their
                # per-fold scores must not be merged into one candidate list).
                # Each block is exactly what build_fold_reselection_input returns
                # (candidates_by_model + baselines_by_model + lower_is_better),
                # so a reader can feed each straight to fold_reselection. Purely
                # additive: the mean-based 'hybrid'/'best_single' arrays are
                # UNCHANGED.
                'fold_reselection_by_n': {},
                # The sample-efficiency curve to read. At each sample size,
                # each held-out fold's hybrid recipe and single representation
                # are picked on the other folds only, so both sides pay the
                # same tax for being a maximum. The 'hybrid'/'best_single'
                # arrays below are each a maximum taken on the folds they are
                # then reported on, over ~600 recipes against ~10
                # representations, so their difference runs high.
                'reselected_curve': {},
                # Why hybrid_all_rows_bp is nan wherever it is nan. Filled in
                # below, per model, per sample size.
                'all_rows_bp_skips': {}}

    # Initialize per-model result structure
    for model_name in model_names:
        results['models'][model_name] = {
            'hybrid': [], 'best_single': [],
            # Leave-one-fold-out arrays, aligned with sample_sizes. Read these.
            'hybrid_reselected': [], 'best_single_reselected': [],
            'reselected_delta_mean_of_folds': [],
            'best_configs': [],
            # Second pass: the all-rows-selected blueprint applied at each n.
            'hybrid_all_rows_bp': [],
            'all_rows_bp_meta': [],
        }

    # ⛔ AUGMENTATION CONDITION, NOT ROWS. Every row of the dataset is still
    # used; what is dropped here is one CONDITION of the search.
    #
    # c05 hardcoded augmentation_strategy='capped' and passed the augmentation
    # representations into every one of its find_best_hybrid calls, so
    # KIRBY_LOCAL_AUG_CONDITIONS -- which the local run sets to 'none' and which
    # cuts the augmented conditions out of the shared precompute -- never
    # reached c05. _shared._local_aug_conditions states the cost: a 'capped' or
    # 'uncapped' configuration-fold is about eight model fits where 'none' is
    # one, because the extra feature block is offered at several widths and each
    # one is fitted and scored on the internal split.
    #
    # Measured here 2026-09-25 on flab_thermo_tresanco, all 672 antibodies, on a
    # laptop already running 18 other experiment processes (load average 700,
    # this process getting about a quarter of one core): with 'capped' the
    # 67-antibody point alone took 333 s for its two seeds and the run had not
    # finished the 168-antibody point after 615 s, against a 2400 s cap for the
    # whole experiment. Five curve points x 2 seeds at that rate is hours.
    #
    # This is the same fault c02 had and the same remedy: c02 ran four searches
    # and took 33,174 s on caco2 for exactly this reason
    # (c02_complexity_grid.py:73).
    #
    # With this cut and the two below, the same command finished on the same
    # machine in 1884 s end to end (c05 itself 1645 s), all 672 antibodies, zero
    # refused searches, result_status ok.
    #
    # WHAT IT COSTS: the augmentation question has no answer in a run with
    # KIRBY_LOCAL_AUG_CONDITIONS set. 'hybrid', 'best_single', the
    # leave-one-fold-out arrays and the all-rows-blueprint column are then all
    # measured on the base representations alone. Unset -- the cluster -- this
    # block does nothing and every search still runs 'capped' with the
    # augmentations, so no server run changes.
    from experiments._shared import _local_aug_conditions
    _aug_conditions_kept = _local_aug_conditions(['none', 'uncapped', 'capped'])
    _drop_augs = bool(data.get('augmentations')) and 'capped' not in _aug_conditions_kept
    if _drop_augs:
        print("  [local] KIRBY_LOCAL_AUG_CONDITIONS keeps "
              f"{_aug_conditions_kept} and c05 asked for 'capped', so the "
              "augmentation representations are dropped from every search "
              "here. Every antibody is still used; the augmentation question "
              "has no answer in this run.")
    results['augmentation_conditions_kept'] = list(_aug_conditions_kept)
    results['augmentations_dropped_by_local_grid'] = bool(_drop_augs)

    # The allocation method, importance method, internal evaluator and budget
    # below are chosen once by scoring every row of the dataset, so every label
    # a subsample later holds out was used to pick them. At each subsample only
    # the feature selection is redone, on that subsample's rows. So the scores
    # under 'hybrid_all_rows_bp' are not held-out scores: they answer "how does
    # a recipe picked with all the labels do at size n", not "how does a recipe
    # picked at size n generalise".
    all_rows_blueprints = {}
    all_rows_meta = {}
    # {model: [ {n_sub, train_rows, blueprint_budget, reason}, ... ]} for every
    # point where the second pass was skipped. Without this a nan in
    # hybrid_all_rows_bp is indistinguishable from a point that was attempted
    # and failed.
    all_rows_bp_skips = {}
    # ⛔ A SECOND SCORE PER POINT, NOT ROWS. Switching this off empties the
    # 'hybrid_all_rows_bp' column and changes nothing else: 'hybrid',
    # 'best_single' and the leave-one-fold-out arrays are untouched, and every
    # antibody is still used at every point.
    #
    # What it costs: the column that answers "how does a recipe picked with all
    # the labels do at size n". It is not a held-out score — the paragraph above
    # says so — and on this dataset it was already nan at every point for rf,
    # whose all-rows recipe asks for 250 features while the largest training
    # partition here is 448 only at the full-data point.
    #
    # What it saves, measured 2026-09-25 on flab_thermo_tresanco: the
    # 67-antibody point, where neither model's recipe fits the 44 training rows
    # so this pass never ran, took 26 s over two seeds. The 168-antibody point,
    # where svm's 100-feature recipe does fit the 112 training rows and the pass
    # therefore rebuilt and refitted it over 3 folds per seed, took 414 s. The
    # search grew by no configurations between those two points — both are 6 —
    # but the rows grew 2.5x as well, so that gap is this pass and the larger
    # fits together and I did not separate them.
    #
    # KIRBY_C05_ALL_ROWS_BP=1 puts it back. Unset on the cluster, it runs.
    _bp_env = os.environ.get('KIRBY_C05_ALL_ROWS_BP')
    _all_rows_bp = (_bp_env not in ('0', 'false', 'no') if _bp_env
                    else not os.environ.get('KIRBY_LOCAL_GRID'))
    results['all_rows_bp_second_pass_ran'] = bool(_all_rows_bp)
    if not _all_rows_bp:
        print("  [local] the all-rows-blueprint second pass is off under "
              "KIRBY_LOCAL_GRID, so hybrid_all_rows_bp is nan at every sample "
              "size. Set KIRBY_C05_ALL_ROWS_BP=1 to run it.")
    full_augs_top = None if _drop_augs else data.get('augmentations')
    for model_name in (model_names if _all_rows_bp else []):
        try:
            _s_fd, _m_fd = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=model_name, budget=None,
                augmentations=full_augs_top, 
                dataset_name=dataset_name,
                task_type=data['task_type'], metric=metric_name,
                data=data,
            )
            if _m_fd:
                all_rows_blueprints[model_name] = _m_fd['best_config']['blueprint']
                all_rows_meta[model_name] = {
                    'tier': _m_fd['tier'], 'aug_condition': _m_fd['aug_condition'],
                    'budget': _m_fd['best_config']['blueprint'].budget,
                }
                print(f"  All-rows-selected BP for {model_name}: tier={_m_fd['tier']} "
                      f"aug={_m_fd['aug_condition']} budget={_m_fd['best_config']['blueprint'].budget}")
        except Exception as e:
            print(f"  All-rows-selected BP for {model_name}: pool helper failed ({e})")

    for n_sub in sample_sizes:
        print(f"\n  n={n_sub}:")
        # Collect scores across repeats per model
        hybrid_by_model = {m: [] for m in model_names}
        single_by_model = {m: [] for m in model_names}
        configs_by_model = {m: [] for m in model_names}
        # THE HYBRID SIDE MAXIMISES OVER A BUDGET LIST THAT GROWS WITH n.
        # sub_budgets below keeps every sweep budget at or below the training
        # rows, so at a 50-molecule point it is [10, 25] and at a 1000-molecule
        # point it is [10, 25, 50, 100, 250, 500]. search.best_by_model is the
        # argmax across all of them. The single-representation side has no
        # budget axis at all -- each representation is one score at its own
        # full width. So part of the rise in delta with n is the number of
        # configurations maximised over, not sample efficiency. Recorded here
        # per seed so the confound is visible in the table; NOT corrected,
        # because which way to correct it is the user's call.
        budgets_searched_this_n = []
        rows_drawn_this_n = []
        # Rows drawn by the seeds whose search returned, for the curve's x axis.
        rows_scored_this_n = []
        # Second pass: the all-rows-selected blueprint, features re-selected
        # on the subsample. The recipe itself still came from every label.
        hybrid_fd_by_model = {m: [] for m in model_names}
        # One CONTRACT-shape fold_reselection block per seed at this sample size.
        reselection_blocks_this_n = []

        last_search = None
        full_augs = None if _drop_augs else data.get('augmentations')
        # Pre-compute group-level pool so subsampling preserves scaffold
        # integrity (subsampling individual rows shatters scaffold groups
        # across train/val of the same fold).
        _unique_groups = np.unique(groups)
        _group_to_idx = {g: np.where(groups == g)[0] for g in _unique_groups}
        # FIX 3h: at the full-data point the group sampling below picks EVERY
        # group regardless of seed, so all n_repeats seeds draw the identical
        # dataset and run the identical (expensive) search under different cache
        # names. The mean of n_repeats identical runs is just the value, so
        # collapse to a single seed here. <100% fractions still use n_repeats.
        n_repeats_eff = 1 if n_sub >= n_total else n_repeats
        for seed in range(n_repeats_eff):
            rng = np.random.RandomState(seed)
            # Draw whole scaffold groups until we hit ~n_sub samples (the
            # exact size is approximate; group sizes vary).
            shuffled_groups = rng.permutation(_unique_groups)
            picked, picked_count = [], 0
            for g in shuffled_groups:
                gi = _group_to_idx[g]
                if picked_count + len(gi) > n_sub and picked:
                    break
                picked.extend(gi.tolist())
                picked_count += len(gi)
                if picked_count >= n_sub:
                    break
            idx = np.array(sorted(picked))
            rows_drawn_this_n.append(int(len(idx)))
            sub_reps = {k: v[idx] for k, v in reps.items()}
            sub_labels = labels[idx]
            sub_groups = groups[idx]
            sub_augs = {k: v[idx] for k, v in full_augs.items()} if full_augs else None
            n_unique_groups = len(np.unique(sub_groups))

            if n_unique_groups < 2:
                # RECORDED, NOT SILENT (2026-09-28). The draw above always
                # takes the first group whole, so on a dataset with a few large
                # groups a small point is one group and cannot be split. On
                # flab_expression_koenig (5 groups of about 855 rows) the 10%
                # point asks for 427 rows and draws one group of 855 on every
                # seed. Before this, that point was nan with nothing saying why.
                search_failures.append({
                    'n_sub': int(n_sub), 'seed': int(seed),
                    'n_groups': int(n_unique_groups),
                    'n_rows': int(len(idx)),
                    'error': ('not searched: the group draw returned one '
                              'group, so no group-aware split exists'),
                })
                print(f"    seed={seed}: draw returned {len(idx)} rows in one "
                      f"group; no search at n={n_sub}")
                continue

            sub_folds = min(n_folds, n_unique_groups)

            # find_best_hybrid discovers optimal config per model at this sample size.
            # Cap budgets to the TRAINING partition, not the whole subsample: a
            # budget > train rows yields a degenerate selection -> NaN (this was
            # the all-NaN-at-small-n bug, e.g. budget 50 at n_sub=50 leaves only
            # ~40 train rows). See COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
            #
            # ⛔ FROM THE ROWS DRAWN, NOT THE ROWS ASKED FOR (2026-09-28). This
            # used n_sub, but the group draw stops short of n_sub, so the cap
            # could exceed every fold's training rows. buchwald_hartwig at 10%
            # asks for 395 reactions and draws two additive groups of 180: the
            # cap said 197 while each fold trains on 180. Now it is the
            # smallest training partition of the folds the search itself
            # builds (same splitter, same seed 42 as find_best_hybrid's
            # default), so no fold is handed a budget above its own rows.
            train_size = _min_train_rows(sub_groups, sub_labels, sub_folds,
                                         data['task_type'])
            sub_budgets = [b for b in sweep_budgets if b <= train_size]
            budgets_searched_this_n.append(len(sub_budgets))
            if not sub_budgets:
                # No sweep budget fits the training partition. The old line
                # here re-evaluated the same empty generator inside min(...),
                # so its `default` always won -- it never picked a sweep
                # budget. Say so: this budget is NOT one of the swept ones.
                sub_budgets = [max(2, train_size)]
                print(f"    n={n_sub}: no sweep budget fits {train_size} "
                      f"training rows; using {sub_budgets[0]}, which is not "
                      f"in the sweep")
            # Subsample-specific cache key — won't collide with main precompute
            # cache. Capped strategy is fine for small subsamples but projects
            # 'capped' at every subsample size. The 250-row gate was removed
            # 2026-08-27; the rebuilt capped costs about a tenth of the version
            # it was written for.
            #
            # 'none' instead when the block near the top of run() dropped the
            # augmentation representations. find_best_hybrid normalises the
            # strategy to 'none' itself when it is handed no augmentations, but
            # it does that with a warning, and the strategy rides in the search
            # metadata and the cache key, so it is set here rather than left to
            # be corrected.
            sub_aug_strategy = 'none' if sub_augs is None else 'capped'
            search = None
            try:
                search = find_best_hybrid(
                    sub_reps, sub_labels, groups=sub_groups,
                    models=model_names,
                    budgets=sub_budgets,
                    n_folds=sub_folds,
                    augmentations=sub_augs,
                    augmentation_strategy=sub_aug_strategy,
                    internal_evaluators=INTERNAL_EVALUATORS,
                    metric=metric_name,
                    task_type=data['task_type'],
                    verbose=False,
                    cache=True,
                    dataset_name=f"{dataset_name}_n{n_sub}_s{seed}",
                    tier='general',
                )
                last_search = search
                last_search_n, last_search_seed = n_sub, seed
                rows_scored_this_n.append(int(len(idx)))
                # Per-fold scores for BOTH arms (hybrid candidates +
                # per-rep baselines) in the CONTRACT shape. Derives models
                # only from what the search actually evaluated, so model
                # gating is preserved automatically.
                try:
                    reselection_blocks_this_n.append(
                        build_fold_reselection_input(search))
                except Exception as _he:
                    print(f"    seed={seed}: fold_reselection build failed - {_he}")
                for model_name in model_names:
                    if model_name in search.best_by_model:
                        mbest = search.best_by_model[model_name]
                        hybrid_by_model[model_name].append(mbest['score_mean'])
                        configs_by_model[model_name].append({
                            'budget': mbest['blueprint'].budget,
                            'importance': mbest['blueprint'].importance_method,
                            'allocation': mbest['blueprint'].allocation_method,
                        })
            except Exception as e:
                # ⛔ RECORDED, NOT JUST PRINTED. At small subsample sizes
                # allocate_performance_weighted refuses when not one
                # representation can be scored on a held-out block -- under ten
                # held-out rows, one scaffold group, or constant activity. It is
                # right to refuse: splitting equally instead would be filed
                # under performance_weighted's name and be indistinguishable
                # from a method that chose an equal split. But c05 caught it per
                # configuration and carried on, so the point was simply missing
                # and a NaN at n=67 read the same as a NaN from a model that
                # scored nothing.
                search_failures.append({
                    'n_sub': int(n_sub), 'seed': int(seed),
                    'n_groups': int(n_unique_groups),
                    'n_rows': int(len(idx)),
                    'error': f'{type(e).__name__}: {e}',
                })
                print(f"    seed={seed}: find_best_hybrid failed - {e}")

            # Baselines per model, read out of the search that just ran.
            # search.baselines is {rep: {model: score}} on the SAME folds and
            # the same subsample rows the hybrid arm used. Re-scoring every
            # representation here instead cost 4,550 fits per dataset and got
            # three things wrong: it ran each representation through
            # cap_dimensions, so anything wider than the fold's training rows
            # was PCA-compressed before it was scored; it predicted hard class
            # labels, so an AUROC dataset was scored from 0/1 instead of
            # probabilities; and it keyed the result off model_names resolved at
            # full n while iterating factories resolved at the subsample size,
            # so a model admitted at a small subsample and gated out at full n
            # raised a KeyError. The models are now whatever the search itself
            # evaluated.
            if search is not None:
                best_single_by_model = {}
                for rep_name, rep_scores in (getattr(search, 'baselines', None) or {}).items():
                    if rep_name.startswith('_'):
                        continue  # reserved keys, mirrors extract_top3_base_reps
                    for _m, _score in (rep_scores or {}).items():
                        if _score is None or not np.isfinite(_score):
                            continue
                        _cur = best_single_by_model.get(_m)
                        if _cur is None or (_score < _cur if lower_is_better
                                            else _score > _cur):
                            best_single_by_model[_m] = float(_score)
                for _m, _score in best_single_by_model.items():
                    single_by_model.setdefault(_m, []).append(_score)

            # Second pass: reapply the all-rows-selected blueprint at this
            # subsample. alloc/imp/eval/budget/strategy stay fixed from the
            # whole-dataset selection above; only the feature selection runs on
            # subsample rows. The resulting score is NOT held out from the rows
            # that chose the recipe.
            for model_name, model_factory in models.items():
                bp_fd = all_rows_blueprints.get(model_name)
                if bp_fd is None:
                    continue
                bp_fd_budget = bp_fd.budget
                # ⛔ THIS SKIP WROTE NOTHING, AND IT FIRES FOR A WHOLE MODEL AT
                # EVERY SAMPLE SIZE. Measured on flab_thermo_tresanco
                # 2026-09-24: the whole-dataset winner for rf carries
                # budget=1000 while the curve's sample sizes are 67, 168, 336,
                # 504 and 672 -- so rf's hybrid_all_rows_bp column was nan at
                # every point, and nothing said why. lightgbm's winner carries
                # budget=50 and its column is populated throughout. A reader
                # comparing the two models sees one full column and one empty
                # one with no explanation.
                #
                # Also compared against n_sub, the whole subsample, while the
                # hybrid is built on the TRAINING partition -- so a budget of
                # exactly n_sub passed and was then applied to about 0.8 n_sub
                # rows. It now compares against the training partition, the
                # same number the other arm caps against.
                _fd_train = train_size
                if bp_fd_budget is not None and bp_fd_budget > _fd_train:
                    all_rows_bp_skips.setdefault(model_name, []).append({
                        'n_sub': int(n_sub),
                        'train_rows': int(_fd_train),
                        'blueprint_budget': int(bp_fd_budget),
                        'reason': ('the all-rows recipe asks for more features '
                                   'than this subsample has training rows'),
                    })
                    continue
                # FIX 4d: cv_split RAISES (no random-split fallback) when the
                # subsample has fewer distinct groups than the requested fold
                # count. Reduce the fold count to the available groups (min 2);
                # if fewer than 2 distinct groups exist, record an explicit skip
                # flag instead of crashing or silently emitting nothing.
                n_sub_groups = len(np.unique(sub_groups))
                if n_sub_groups < 2:
                    results['skipped_insufficient_groups'].append(
                        {'n': int(n_sub), 'model': model_name, 'seed': int(seed),
                         'n_groups': int(n_sub_groups),
                         'reason': 'fewer than 2 distinct groups for all-rows-bp CV'})
                    continue
                sub_folds_fd = min(sub_folds, n_sub_groups)
                try:
                    fold_scores = []
                    sample_sub_X = next(iter(sub_reps.values()))
                    try:
                        _fd_splits = list(cv_split(sub_groups, sub_folds_fd,
                                                   sample_sub_X, sub_labels))
                    except ValueError:
                        # Group count fell below the (already-reduced) fold
                        # count after StratifiedGroupKFold's own checks — skip
                        # cleanly with a flag rather than propagate.
                        results['skipped_insufficient_groups'].append(
                            {'n': int(n_sub), 'model': model_name, 'seed': int(seed),
                             'n_groups': int(n_sub_groups),
                             'reason': 'cv_split rejected all-rows-bp folds'})
                        continue
                    for tr_i, va_i in _fd_splits:
                        train_reps_fold = {k: v[tr_i] for k, v in sub_reps.items()}
                        val_reps_fold = {k: v[va_i] for k, v in sub_reps.items()}
                        train_augs_fold = ({k: v[tr_i] for k, v in sub_augs.items()}
                                           if sub_augs else None)
                        val_augs_fold = ({k: v[va_i] for k, v in sub_augs.items()}
                                         if sub_augs else None)
                        X_tr, X_va, _fi = build_fold_hybrid(
                            train_reps_fold, val_reps_fold,
                            sub_labels[tr_i], bp_fd,
                            augmentations_train=train_augs_fold,
                            augmentations_val=val_augs_fold,
                            groups=(sub_groups[tr_i] if sub_groups is not None else None),
                            model=model_factory(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better)
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr)
                        X_va_s = scaler.transform(X_va)
                        mdl = model_factory()
                        mdl.fit(X_tr_s, sub_labels[tr_i])
                        # Ranking metrics need graded scores. An AUROC read off
                        # hard 0/1 labels collapses to the mean of sensitivity
                        # and specificity (hybrid.py:2661). The baselines arm
                        # above was fixed for this; this one was not, so the
                        # hybrid_all_rows_bp column was a different quantity
                        # from the hybrid column on every auc dataset.
                        _hard = mdl.predict(X_va_s)
                        _y = (_graded_scores_for_clf(mdl, X_va_s, _hard)
                              if data['task_type'] == 'classification' else _hard)
                        fold_scores.append(metric_fn(sub_labels[va_i], _y))
                    if fold_scores:
                        hybrid_fd_by_model[model_name].append(float(np.mean(fold_scores)))
                except Exception:
                    pass

        if reselection_blocks_this_n:
            results['fold_reselection_by_n'][str(n_sub)] = reselection_blocks_this_n

        # Leave-one-fold-out re-pick at this sample size, per seed, then
        # averaged over seeds. Pure arithmetic on per-fold scores that already
        # exist, so it costs no fits. Subsamples differ between seeds, so each
        # seed's block is rescored on its own and only the resulting numbers
        # are averaged.
        reselected_by_model_this_n = {m: {'h': [], 's': [], 'd': []}
                                  for m in model_names}
        for _blk in reselection_blocks_this_n:
            try:
                _hr = fold_reselection(
                    _blk.get('candidates_by_model', {}),
                    _blk.get('baselines_by_model', {}),
                    _blk.get('lower_is_better'),
                    metric_name=_blk.get('metric_name'),
                    exclude_folds=hr_exclude_folds(_blk))
            except Exception as _e:
                print(f"    per-fold reselection unavailable at n={n_sub}: {_e}")
                continue
            for _m, _v in _hr.items():
                if _m not in reselected_by_model_this_n:
                    reselected_by_model_this_n[_m] = {'h': [], 's': [], 'd': []}
                for _k, _f in (('h', 'hybrid_reselected'),
                               ('s', 'best_single_reselected'),
                               ('d', 'delta_mean_of_folds')):
                    _val = _v.get(_f)
                    if _val is not None and not np.isnan(_val):
                        reselected_by_model_this_n[_m][_k].append(float(_val))

        # Aligned with sample_sizes. n_budgets_searched is how many budgets the
        # hybrid side took its maximum over at this point; the single-rep side
        # always takes its maximum over the representations only. n_rows_drawn
        # is what the group draw actually returned -- the draw stops at the
        # first scaffold group that would overshoot, so it can land well under
        # the size asked for, and the seeds averaged at one point of the curve
        # are then not all at one size.
        results.setdefault('n_budgets_searched', []).append(
            int(max(budgets_searched_this_n)) if budgets_searched_this_n else 0)
        results.setdefault('n_rows_drawn', []).append(
            sorted(rows_drawn_this_n))
        results.setdefault('n_rows_scored_mean', []).append(
            float(np.mean(rows_scored_this_n)) if rows_scored_this_n
            else float('nan'))

        if last_search is not None:
            top3 = extract_top3_base_reps(last_search)
            if top3:
                print(f"    Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

        # Aggregate across repeats
        for model_name in model_names:
            h = (float(np.mean(hybrid_by_model[model_name]))
                 if hybrid_by_model[model_name] else float('nan'))
            s = (float(np.mean(single_by_model[model_name]))
                 if single_by_model[model_name] else float('nan'))

            h_fd = (float(np.mean(hybrid_fd_by_model[model_name]))
                    if hybrid_fd_by_model[model_name] else float('nan'))
            results['models'][model_name]['hybrid'].append(h)
            results['models'][model_name]['best_single'].append(s)
            results['models'][model_name]['best_configs'].append(
                configs_by_model[model_name] if configs_by_model[model_name] else None)
            _hb = reselected_by_model_this_n.get(model_name, {'h': [], 's': [], 'd': []})
            for _k, _field in (('h', 'hybrid_reselected'),
                               ('s', 'best_single_reselected'),
                               ('d', 'reselected_delta_mean_of_folds')):
                results['models'][model_name][_field].append(
                    float(np.mean(_hb[_k])) if _hb[_k] else float('nan'))
            results['models'][model_name]['hybrid_all_rows_bp'].append(h_fd)
            results['models'][model_name]['all_rows_bp_meta'].append(
                all_rows_meta.get(model_name))

            delta = (s - h if lower_is_better else h - s) if not (np.isnan(h) or np.isnan(s)) else float('nan')
            hd = results['models'][model_name]['reselected_delta_mean_of_folds'][-1]
            print(f"    {model_name}: hybrid={h:.4f}, hybrid_all_rows_bp={h_fd:.4f}, "
                  f"best_single={s:.4f}, same-fold delta={delta:+.4f}, "
                  f"leave-one-fold-out delta={hd:+.4f}")

    # Top-3 from the last (full-data) subsample search
    # Carry the skip record out with the numbers.
    results['all_rows_bp_skips'] = all_rows_bp_skips
    for _m, _sk in sorted(all_rows_bp_skips.items()):
        print(f"  {_m}: hybrid_all_rows_bp skipped at {len(_sk)} of "
              f"{len(sample_sizes)} sample sizes — its all-rows recipe asks "
              f"for {_sk[0]['blueprint_budget']} features and the largest "
              f"training partition here is "
              f"{max(x['train_rows'] for x in _sk)}")

    top3_full = extract_top3_base_reps(last_search) if last_search is not None else []

    # Fit learning curves per model.
    #
    # ⛔ X AXIS IS THE ROWS DRAWN, NOT THE ROWS ASKED FOR (2026-09-28). The
    # group draw stops short of n_sub, by up to one whole group: on
    # flab_expression_koenig the 50% point asks for 2,137 antibodies and draws
    # two groups, 1,710. The fit read n_sub. It now reads the mean rows drawn
    # by the seeds whose search returned, falling back to n_sub where none did.
    #
    # BOTH ARMS ARE FITTED (2026-09-28). Only the hybrid had a fit, so "more
    # sample-efficient than individual representations" had no curve to be
    # compared against. The single-representation arrays are fitted the same
    # way. Each fit has 3 parameters, so 3 points leave no residual degrees of
    # freedom and fit_R2 near 1 says nothing; that needs 4 or more points.
    _n_scored = results.get('n_rows_scored_mean') or []

    def _x(i):
        v = _n_scored[i] if i < len(_n_scored) else float('nan')
        return v if np.isfinite(v) else float(sample_sizes[i])

    for model_name in model_names:
        model_result = results['models'][model_name]
        for _arr, _fit in (('hybrid', 'hybrid_fit'),
                           ('best_single', 'best_single_fit'),
                           ('hybrid_reselected', 'hybrid_reselected_fit'),
                           ('best_single_reselected',
                            'best_single_reselected_fit')):
            _valid = [i for i, v in enumerate(model_result[_arr])
                      if not np.isnan(v)]
            if len(_valid) >= 3:
                n_arr = np.array([_x(i) for i in _valid])
                v_arr = np.array([model_result[_arr][i] for i in _valid])
                model_result[_fit] = fit_learning_curve(
                    n_arr, v_arr, lower_is_better=lower_is_better)
                model_result[_fit]['n_points'] = len(_valid)
                model_result[_fit]['n_rows'] = [float(x) for x in n_arr]
                # Which parameters stopped on a bound of fit_learning_curve
                # (asymptote within one observed range of the scores, a in
                # [0, 100], b in [0.1, 2]). A parameter on its bound was set by
                # the bound, not the data. Checked on made-up rmse values in
                # yield units (30, 22, 18): a stopped at 100 and the asymptote
                # at the edge of its range.
                _f = model_result[_fit]
                if 'error' not in _f:
                    _span = float(np.max(v_arr) - np.min(v_arr)) or 1.0
                    _lo = float(np.min(v_arr)) - _span
                    _hi = float(np.max(v_arr)) + _span
                    _tol = 1e-6 * max(1.0, abs(_span))
                    _f['at_bound'] = [
                        _p for _p, _hit in (
                            ('asymptote', min(abs(_f['r2_max'] - _lo),
                                              abs(_f['r2_max'] - _hi)) < _tol),
                            ('a', _f['a'] <= 1e-9 or _f['a'] >= 100 - 1e-6),
                            ('b', _f['b'] <= 0.1 + 1e-9 or _f['b'] >= 2 - 1e-9))
                        if _hit]

    # Summary table. Each model gets its same-fold pair and the
    # leave-one-fold-out delta beside it; the two differ by how much the
    # hybrid side gains from maximising over ~600 recipes on the folds it is
    # then scored on.
    print("\n" + "-" * 70)
    print("LEARNING CURVE SUMMARY")
    print("-" * 70)
    print("  n = molecules asked for; drawn = molecules the group draw "
          "returned; nb = budgets the hybrid side maximised over")
    header = f"  {'n':>6} {'drawn':>11} {'nb':>3}"
    for m in model_names:
        header += f"  {m+'_h':>10} {m+'_s':>10} {'delta':>8} {'lofo_d':>8}"
    print(header)
    print("  " + "-" * (22 + (10 + 10 + 8 + 8 + 5) * len(model_names)))
    _nb_all = results.get('n_budgets_searched') or []
    _drawn_all = results.get('n_rows_drawn') or []
    for i, n_sub in enumerate(sample_sizes):
        _dr = _drawn_all[i] if i < len(_drawn_all) else []
        _dr_s = (f"{min(_dr)}-{max(_dr)}" if _dr and min(_dr) != max(_dr)
                 else (str(_dr[0]) if _dr else '?'))
        _nb = _nb_all[i] if i < len(_nb_all) else '?'
        row = f"  {n_sub:>6} {_dr_s:>11} {_nb:>3}"
        for m in model_names:
            h = results['models'][m]['hybrid'][i]
            s = results['models'][m]['best_single'][i]
            d = (s - h if lower_is_better else h - s) if not (np.isnan(h) or np.isnan(s)) else float('nan')
            hd = results['models'][m]['reselected_delta_mean_of_folds'][i]
            row += f"  {h:>10.4f} {s:>10.4f} {d:>+8.4f} {hd:>+8.4f}"
        print(row)

    # Learning curve fit summary
    print(f"\n  LEARNING CURVE FITS (score_max = asymptotic {metric_name}):")
    for model_name in model_names:
        fit = results['models'][model_name].get('hybrid_fit', {})
        if 'r2_max' in fit and 'error' not in fit:
            print(f"    {model_name}: {metric_name}_max={fit['r2_max']:.4f}, "
                  f"a={fit['a']:.3f}, b={fit['b']:.3f}, fit_R2={fit['fit_r2']:.3f}")
        elif 'r2_max' in fit:
            print(f"    {model_name}: {metric_name}_max~{fit['r2_max']:.4f} (fit failed: {fit.get('error', '?')})")

    results['top3_base_reps'] = [(n, s) for n, s in top3_full] if top3_full else []

    # --- The top TWO of each arm, and the two reference numbers -------------
    # The curve above keeps ONE hybrid and ONE base representation per
    # downstream model per sample size: results['models'][m]['hybrid'] is the
    # mean over seeds of search.best_by_model, and ['best_single'] the mean over
    # seeds of the single highest-scoring representation. Nothing said what came
    # second, so a rung where the top two representations are a thousandth apart
    # reads the same as one where the second is far behind.
    #
    # `last_search` is the search at the LAST sample size in `sample_sizes` and
    # the last seed that did not raise -- the same SearchResult top3_base_reps
    # above is read from. On a dataset under LARGE_DATASET_N that last size is
    # every row; on a larger one it is the LARGE_SUBSAMPLE_K cap, so this block
    # describes that subsample and not the whole dataset.
    #
    # All four read rows that search already stored, so this costs 0 new model
    # fits on top of the searches the curve already paid for -- at 5 sample
    # sizes x 3 seeds that is 15 find_best_hybrid calls, and none of them is
    # repeated here.
    #
    # require_full=False everywhere: a downstream model this subsample's search
    # never scored yields a short list carrying its own shortfall_reason,
    # instead of raising at the end of the run.
    def _no_blueprint(obj):
        """Drop the pickled Blueprint from the blocks added below.

        It does serialise -- convert_numpy routes it through Blueprint.to_dict
        (find_best_hybrid.py:1438) -- but it carries kept_augmentations, one
        list of feature indices per fold per donor representation, and every
        recipe field on it is already a key of the entry it hangs off. Which
        representations actually fed a hybrid is recorded by hybrid_deltas under
        'source_reps'.
        """
        if isinstance(obj, dict):
            return {k: _no_blueprint(v) for k, v in obj.items()
                    if k != 'blueprint'}
        if isinstance(obj, list):
            return [_no_blueprint(v) for v in obj]
        return obj

    # New top-level keys. Nothing is written into results['models'], which is
    # looped as downstream models in three places in this file and in
    # firstpass_notready.py's read_c05, nor into
    # results['fold_reselection_by_n'], whose keys are str(sample size) and are
    # read back with int(float(nkey)).
    results['top2_base_reps'] = {}
    results['top2_hybrids'] = {}
    results['hybrid_deltas_by_model'] = {}
    results['top2_pairs_by_model'] = {}
    if last_search is not None:
        # WHICH SEARCH THE FOUR BLOCKS BELOW ARE READ OUT OF, recorded
        # 2026-09-28. `last_search` is reset to None at the top of every sample
        # size, so it is the search of the LAST size in `sample_sizes`, at the
        # last seed there that did not raise. The saved result named neither
        # that number of molecules nor that seed, so a reader could not tell how
        # many molecules these rankings were ranked on.
        #
        # One find_best_hybrid call, given models=model_names, scored every
        # downstream model in it, so this search was not fetched for one model
        # the way n13's and n17's per-model searches are. The list is written
        # under the same key name those files use.
        results['top2_pairs_search_of_model'] = list(model_names)
        results['top2_search_n_molecules_subsampled'] = last_search_n
        results['top2_search_seed'] = last_search_seed
        # WHICH DOWNSTREAM MODELS THESE ROWS COVER, resolved 2026-09-28.
        # `model_names` is what run() was configured with. The search scored
        # whatever it scored -- svm drops itself above 10,000 molecules, and a
        # dataset's skip policy can drop a name -- and the two disagree whenever
        # that happened. The configured list stays what this file asks for, so
        # it goes in as `requested`: a name the search never scored is dropped
        # and named, never swapped for another model's rows.
        _cmp_models, _cmp_dropped = resolve_comparison_models(
            last_search, model_names)
        results['top2_models_compared'] = list(_cmp_models)
        # {downstream model: why it is not in the rows below}. Until now every
        # one of these reasons was printed to the terminal only.
        results['top2_models_dropped'] = {_n: _w for _n, _w in _cmp_dropped}
        for _n, _w in _cmp_dropped:
            print(f"  top-2 rows for {_n} unavailable: {_w}")
        for _m in _cmp_models:
            try:
                results['top2_base_reps'][_m] = top_base_reps(
                    last_search, _m, 2, require_full=False)
            except Exception as _e:
                print(f"  top-2 base representations for {_m} unavailable: {_e}")
                results['top2_models_dropped'][_m] = (
                    'no top-2 base representations: %s: %s'
                    % (type(_e).__name__, _e))
            _th = None
            try:
                _th = top_hybrids(last_search, _m, 2, require_full=False)
                results['top2_hybrids'][_m] = _no_blueprint(_th)
            except Exception as _e:
                print(f"  top-2 hybrid configurations for {_m} unavailable: {_e}")
                results['top2_models_dropped'][_m] = (
                    'no top-2 hybrid configurations: %s: %s'
                    % (type(_e).__name__, _e))
            # The rank-1 hybrid is handed to hybrid_deltas explicitly. Called
            # without one it picks its own and reads it as
            # _picked['selected'][0]['row'] (_shared.py:7432), and a selected
            # entry from top_hybrids has no 'row' key -- 24 keys, none of them
            # 'row' -- so that path raises KeyError('row'). Checked by running
            # both calls in this session. The entry carries 'score_mean' and
            # 'blueprint', which is all hybrid_deltas reads off the hybrid it is
            # given.
            if _th and _th['selected']:
                try:
                    results['hybrid_deltas_by_model'][_m] = hybrid_deltas(
                        last_search, _m, hybrid=_th['selected'][0])
                except Exception as _e:
                    print(f"  hybrid deltas for {_m} unavailable: {_e}")
                    results['top2_models_dropped'][_m] = (
                        'no hybrid deltas: %s: %s' % (type(_e).__name__, _e))
        try:
            results['top2_pairs_by_model'] = _no_blueprint(
                select_model_rep_pairs(last_search, models=list(_cmp_models),
                                       n_base=2, n_hybrids=2,
                                       require_full=False))
        except Exception as _e:
            print(f"  top-2 model-and-representation grid unavailable: {_e}")
            results['top2_pairs_by_model_absent_reason'] = \
                f"{type(_e).__name__}: {_e}"
    else:
        print("  top-2 rows and hybrid deltas skipped: every find_best_hybrid "
              "call raised, so no SearchResult was kept")
        results['top2_extra_absent_reason'] = (
            'every find_best_hybrid call at the last sample size raised, so no '
            'SearchResult was kept and there are no stored rows to rank')
        results['top2_models_compared'] = []
        results['top2_models_dropped'] = {}

    # Printed because the two deltas answer different questions and a reader
    # given one number cannot tell which was asked. The first is the hybrid
    # against the mean of the representations the budget actually gave features
    # to, read off Blueprint.allocation_per_fold, so a representation given 0
    # features is not in the mean. The second is the hybrid against the single
    # best base representation for the same downstream model. A hybrid can win
    # the first and lose the second.
    if results['hybrid_deltas_by_model']:
        print(f"\n  TOP HYBRID vs TWO REFERENCE NUMBERS at n={sample_sizes[-1]} "
              f"({metric_name}, + means the hybrid won):")
        for _m in model_names:
            _hd = results['hybrid_deltas_by_model'].get(_m)
            if not _hd:
                continue
            _dsm = _hd.get('delta_vs_source_mean')
            _dbr = _hd.get('delta_vs_best_rep')
            _n_src = _hd.get('n_source_reps')
            _src_mean = _hd.get('source_mean')
            _best_rep = _hd.get('best_rep')
            _best_sc = _hd.get('best_rep_score')
            _left = (f"vs the mean of the {_n_src} representations that fed it "
                     f"({_src_mean:.4f}): {_dsm:+.4f}"
                     if _dsm is not None and _src_mean is not None
                     else "vs the mean of its source representations: --")
            _right = (f"vs the best single representation {_best_rep} "
                      f"({_best_sc:.4f}): {_dbr:+.4f}"
                      if _dbr is not None and _best_sc is not None
                      else "vs the best single representation: --")
            print(f"    {_m}: {_left}; {_right}")

    # Every configuration whose search refused, with the size and the group
    # count that caused it, so a missing point on the curve says why.
    results['search_failures'] = search_failures
    results['search_failures_by_n'] = {
        int(n): sum(1 for f in search_failures if f['n_sub'] == int(n))
        for n in sample_sizes}
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C05: Learning Curves',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Number of random subsampling repeats per size')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Repeats: {args.n_repeats}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, n_repeats=args.n_repeats)
    save_experiment_results(results, args.output, 'c05_learning_curves', args.dataset)
