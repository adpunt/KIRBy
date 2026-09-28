#!/usr/bin/env python3
"""C10: Noise Robustness & Feature Stability — auc_norm across noise strategies.

Paper: Complexity
Conclusion: Hybrids are more noise-robust than individual reps.
Dependencies: None (standalone).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import zlib

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, _cv_split,
    get_domain_metric, convert_numpy, get_model_factories,
    build_fold_hybrid, build_fold_reselection_input, fold_reselection,
    hr_exclude_folds,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
)
from kirby.find_best_hybrid import find_best_hybrid
from experiments.dataset_policy import (
    noise_dose_units, noise_groups_for, noise_levels as dataset_noise_levels,
)
from kirby.noise_robustness import robustness_metrics
from kirby.noise_spec import (
    REGRESSION_STRATEGIES, NoiseSpec, TrainingNoise, label_rows,
    noise_provenance_row,
)

# Optional NoiseInject import
try:
    import noiseInject  # noqa: F401
    HAS_NOISEINJECT = True
except ImportError:
    HAS_NOISEINJECT = False


def _nogueira_stability(feature_sets, total_features):
    """Compute Nogueira et al. (2018) stability index from per-fold feature sets.

    Args:
        feature_sets: list of sets, each containing (rep_name, feature_idx) tuples.
        total_features: int, total number of features available.

    Returns:
        (phi, se): stability index in [-1, 1] and standard error.
        phi=1 means identical selection every fold.

    Reference:
        Nogueira, Sechidis & Brown (2018). "On the Stability of Feature
        Selection Algorithms." JMLR 18(174), 1-54.
    """
    M = len(feature_sets)
    if M < 2 or total_features == 0:
        return float('nan'), float('nan')

    # Build unified feature index from all selected features across folds
    all_features = set()
    for fs in feature_sets:
        all_features.update(fs)
    all_features = sorted(all_features)
    feat_to_idx = {f: i for i, f in enumerate(all_features)}
    d = len(all_features)

    if d == 0:
        return float('nan'), float('nan')

    # Binary matrix Z: M x d
    Z = np.zeros((M, d), dtype=float)
    for i, fs in enumerate(feature_sets):
        for f in fs:
            if f in feat_to_idx:
                Z[i, feat_to_idx[f]] = 1.0

    k_bar = Z.sum(axis=1).mean()
    if k_bar == 0 or k_bar == d:
        return float('nan'), float('nan')

    p_f = Z.mean(axis=0)
    numerator = (M / (M - 1)) * np.sum(p_f * (1 - p_f))
    denominator = d * k_bar * (1 - k_bar / d)

    if denominator == 0:
        return float('nan'), float('nan')

    phi = 1 - numerator / denominator
    var_terms = p_f * (1 - p_f)
    se = np.sqrt((4 / (M * d**2 * (k_bar * (1 - k_bar / d))**2)) *
                 np.sum(var_terms**2)) if M > 2 else float('nan')

    return float(phi), float(se)


def run(data, config, output_dir, dataset_name, levels=None, strategies=None,
        noise_seed=42, **retired):
    """Exp 10: Noise robustness (auc_norm / Weibull) across noise strategies.

    Tests whether hybrids are more noise-robust than individual reps, and whether
    the optimal hybrid config changes under noise. Also tracks feature stability
    across noise levels (do the same features stay important when noise increases?).

    ONE SEARCH PER (CONDITION, LEVEL), on that cell's own corrupted labels. The
    level-0 cells are clean labels whatever the condition's shape, so they share
    a single search. Each reported fold is then read with the recipe that scored
    best on the other folds, for the hybrid and for the representation arm
    alike. Fold 0 contributes no term to either arm — the search filters its own
    recipe list using fold 0's scores — so four folds are reported, not five.

    Robustness is summarised by robustness_metrics(levels, scores): auc_norm
    (PRIMARY, higher = more robust) plus the supplementary Weibull tau/beta shape
    descriptors. This replaces the old NDS (linear slope of R² vs the noise level),
    which assumed a straight-line degradation and re-encoded baseline skill.

    NOISE LEVELS ARE NO LONGER SIGMAS. Under noiseInject 1.0.0 a level is the noise
    actually DELIVERED, as a root-mean-square, and every condition solves in closed
    form for the internal scale that delivers it. Before that each condition delivered
    whatever it happened to deliver -- between 0.49x and 2.00x the Gaussian amount --
    so the apparent severity ordering between conditions was entirely explained by
    amount rather than shape. `--sigmas` is REFUSED by name rather than ignored, so a
    job script written against the old scheme fails instead of running silently under
    the new one, where the number means something different.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: NOISE ROBUSTNESS & FEATURE STABILITY")
    print("=" * 70)

    # Order matters: gate on task_type BEFORE the noiseInject availability
    # check. A classification dataset is a clean skip regardless of whether
    # noiseInject is installed; and a missing noiseInject is a skip (not an
    # 'error') so it never surfaces as a wave failure.
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  Skipping: c10 is regression-only "
              f"(task_type={task_type}). Noise injection mangles class labels.")
        return {'skipped': True, 'reason': 'regression_only',
                'task_type': task_type}

    if not HAS_NOISEINJECT:
        print("  Skipping: noiseInject not installed "
              "(install: pip install noiseInject).")
        return {'skipped': True, 'reason': 'noiseInject_unavailable'}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations') or {}
    budgets = config['budgets']
    n_folds = config['n_splits']

    # The recipe is searched under each condition's own corrupted labels — see
    # the block that runs the searches, below. Reading one clean-label recipe
    # out of the stored pool and applying it at every level gave the hybrid arm
    # a recipe chosen by a search that saw the true activity value of every
    # molecule, while the representations it is compared against had no recipe
    # chosen for them on any labels.

    for _dead in ('sigmas', 'strategy'):
        if _dead in retired:
            raise TypeError(
                f"c10 no longer takes {_dead!r}. A noise level is now the amount of "
                f"noise DELIVERED, not a Gaussian sigma, so the old numbers mean "
                f"something different. Pass levels=... and strategies=... instead; "
                f"the per-dataset grids are in experiments.dataset_policy.")
    if retired:
        raise TypeError(f"unexpected arguments: {sorted(retired)}")

    if strategies is None:
        strategies = list(REGRESSION_STRATEGIES)
    dose_units = noise_dose_units(dataset_name)

    def _levels_for(strategy):
        lv = list(levels) if levels is not None else list(
            dataset_noise_levels(dataset_name, strategy=strategy))
        if len(lv) < 4:
            # auc_norm summarises the curve of score against noise level, and
            # robustness_metrics returns nan for every cell when it is handed
            # fewer than FOUR points. This said three until 2026-09-18, so a
            # three-level run passed the check and published a table of nan --
            # the exact thing the check exists to stop.
            raise ValueError(
                f"c10 needs at least 4 noise levels to compute auc_norm; got "
                f"{len(lv)}: {lv}. Fewer give nan in every cell.")
        return lv

    # Scaffold groups for grouped_wider / grouped_shifted. Acyclic molecules are
    # singletons (NOISE_DESIGN.md 2a rule 2) and this grouping is deliberately NOT the
    # one that sets the fold boundaries. None when the dataset has no SMILES, in which
    # case the two grouped conditions are dropped rather than run over invented groups.
    noise_groups, n_noise_groups, largest_group_share = noise_groups_for(data)
    if noise_groups is None:
        dropped = [c for c in strategies if NoiseSpec(c, 0.1).needs_groups]
        if dropped:
            print(f"  {', '.join(dropped)}: DROPPED, this dataset has no SMILES to "
                  f"build scaffold groups from. Grouped noise over invented groups is "
                  f"uniform noise wearing a grouped name.")
            strategies = [c for c in strategies if c not in dropped]
    else:
        print(f"  noise groups: {n_noise_groups} scaffolds, largest holds "
              f"{largest_group_share:.1%} of molecules")

    provenance = {}

    _noisy_columns = {}

    def _spec_for(strategy, level):
        """The noise condition for one (condition, level). None at level 0.

        A fresh seed per (condition, level): the injector advances its generator
        on every draw, so re-using one across the sweep would make a level's
        realisation depend on how many levels ran before it. The seed depends on
        the condition as well, or two conditions at one level would share a
        realisation and the comparison between them would measure nothing.

        WHICH molecules are hit is seeded from the condition alone, with the
        level left out, so one condition's levels hit the same molecules and
        auc_norm summarises a curve of rising dose rather than a curve of
        re-drawn victims. The fold index is not part of either seed: the random
        draws are made once over every row, and each fold's training labels
        are then corrupted with the amount, censoring limit and grouped-wider
        families taken from that fold's training labels alone
        (kirby.noise_spec.inject_training_rows). find_best_hybrid does the same
        with a spec, so the recipe a search picks and the features phase 3
        rebuilds come from the same corrupted training labels.
        """
        if float(level) == 0.0:
            return None
        digest = zlib.crc32(strategy.encode('utf-8'))
        selection_seed = (int(noise_seed) * 1000003 + digest) & 0x7FFFFFFF
        seed = (selection_seed
                + int(round(float(level) * 1000)) * 7919) & 0x7FFFFFFF
        return NoiseSpec(strategy=strategy, target_effective_noise=float(level),
                         noise_seed=seed, selection_seed=selection_seed,
                         dose_units=dose_units)

    def _noisy_column(spec):
        """The noise source for one condition: `src[train_idx]` is that fold's
        training labels, corrupted from those labels alone
        (kirby.noise_spec.inject_training_rows). Held-out rows are never
        corrupted. Provenance is recorded per fold, one row each.
        """
        if spec is None:
            return labels
        if spec.spec_id not in _noisy_columns:
            # Passed for EVERY condition, not just the grouped two. The other
            # nine ignore it when drawing and it populates n_groups /
            # largest_group_share on their rows too.
            src = TrainingNoise(labels, spec, is_clf=False, groups=noise_groups)
            provenance.setdefault(spec.spec_id, [
                noise_provenance_row(src.result(tr), spec)
                for tr, _ in _cv_split(groups, n_folds, sample_X, labels)])
            _noisy_columns[spec.spec_id] = src
        return _noisy_columns[spec.spec_id]

    results = {
        'noise_levels': {s: _levels_for(s) for s in strategies},
        'dose_units': dose_units,
        'strategies': strategies,
        'noise_seed': int(noise_seed),
        'n_noise_groups': n_noise_groups,
        'largest_group_share': largest_group_share,
        'noise_provenance': provenance,
        'individual_reps': {},
        'hybrids': {},
        'best_single': {},
        'feature_stability': {},
    }

    sample_X = next(iter(reps.values()))

    # The model the feature-stability measurement builds its hybrid against.
    # It used to be `from sklearn.ensemble import RandomForestRegressor` and a
    # bare RandomForestRegressor(n_estimators=100), which bypassed the roster in
    # two ways: the size guards that stop a model being handed more rows than it
    # can take never ran, and the classifier dispatch was bypassed. The second of
    # those never bit here, because this file returns at the top on any dataset
    # that is not regression. The first did. Owner's decision 2026-09-12.
    _rf_factory = get_model_factories(
        ['rf'], task_type=task_type, n_samples=len(labels))['rf']

    # --- Phase 0: one search per (condition, level) ---
    # Every score below is read off these searches. Level 0 is clean labels
    # whatever the condition's shape, so all of the level-0 cells share ONE
    # search instead of running the same one eleven times.
    #
    # speed='fast' by the owner's instruction: 12 recipes per (model, budget)
    # instead of 75. models=['rf'] and every budget in one call, so the budget
    # axis stays in the search rather than being fixed outside it.
    print("\n  Phase 0: one search per (condition, level)")
    _searches = {}
    search_failures = {}

    def _search_for(strategy, level):
        key = 'level0' if float(level) == 0.0 else (strategy, float(level))
        if key in _searches:
            return _searches[key]
        spec = _spec_for(strategy, level)
        label = 'clean labels' if spec is None else f'{strategy} at level {level}'
        try:
            if spec is not None:
                # Drawn here first so a condition this dataset cannot support —
                # the two grouped shapes need one scaffold-group id per molecule
                # and this dataset may have no SMILES — says so with its own
                # message instead of raising from inside the search.
                _noisy_column(spec)
            s = find_best_hybrid(
                reps, labels, groups=groups,
                budgets=budgets,
                models=['rf'],
                metric=metric_name,
                task_type='regression',
                dataset_name=dataset_name,
                tier='general',
                speed='fast',
                n_folds=n_folds,
                augmentations=augmentations or None,
                label_noise=spec,
                # Scaffold families for grouped_wider / grouped_shifted; the
                # search raised on both before it could be given them.
                noise_groups=noise_groups,
                cache=True,
                verbose=False,
            )
            print(f"    {label}: searched")
        except Exception as e:
            print(f"    {label}: SEARCH FAILED, {type(e).__name__}: {e}")
            search_failures[str(key)] = f'{type(e).__name__}: {e}'
            s = None
        _searches[key] = s
        return s

    def _reselected_cell(search):
        """Held-out scores for one (condition, level), hybrid and reps alike.

        Each reported fold is read with the recipe that scored best on the OTHER
        folds. Fold 0 contributes no term to either arm: the search filters its
        own recipe list using fold 0's scores, so fold 0's hybrid score would be
        read off recipes fold 0 helped keep while the representations passed
        through no such filter. Four folds are reported, not five.

        Returns per_budget (one fold_reselection entry per budget, the hybrid arm
        restricted to that budget's recipes), per_rep (each representation's own
        mean over the reported folds) and best_single (the representation
        re-picked per fold, which is the arm the hybrid is compared against).
        """
        block = build_fold_reselection_input(search)
        excluded = hr_exclude_folds(block)
        _drop = {int(f) for f in excluded}
        bases = (block['baselines_by_model'].get('rf') or {})

        # Candidates split by budget, straight off the search rows, so the
        # budget axis survives the honest re-pick instead of the re-pick
        # choosing a budget for every cell.
        cands_by_budget = {}
        for r in (getattr(search, 'all_results', None) or []):
            if r.get('model') != 'rf' or not r.get('fold_scores'):
                continue
            name = "{}|{}|{}|s{}".format(
                r.get('allocation_method'), r.get('importance_method'),
                r.get('internal_evaluator'), r.get('step_size'))
            cands_by_budget.setdefault(r.get('budget'), []).append(
                {'name': name, 'fold_scores': list(r['fold_scores'])})

        per_budget = {}
        for budget in budgets:
            per_budget[budget] = fold_reselection(
                {'rf': cands_by_budget.get(budget, [])},
                {'rf': bases},
                block['lower_is_better'],
                metric_name=block['metric_name'],
                exclude_folds=excluded,
            )['rf']

        per_rep = {}
        for rep_name, folds in bases.items():
            kept = [f for i, f in enumerate(folds)
                    if i not in _drop and f is not None and not np.isnan(f)]
            per_rep[rep_name] = float(np.mean(kept)) if kept else float('nan')

        return {
            'per_budget': per_budget,
            'per_rep': per_rep,
            'best_single': per_budget[budgets[0]]['best_single_reselected'],
            'excluded_folds': excluded,
            'recipes': {
                budget: [bp.allocation_method, bp.importance_method,
                         bp.internal_evaluator]
                for budget, bp in (
                    (b, (search.best_by_model_budget.get('rf', {}).get(b) or {})
                        .get('blueprint'))
                    for b in budgets)
                if bp is not None
            },
        }

    cells = {}
    for strategy in strategies:
        for level in _levels_for(strategy):
            s = _search_for(strategy, level)
            if s is None:
                continue
            cells[(strategy, float(level))] = _reselected_cell(s)

    # ── THE TOP 2 BASE REPRESENTATIONS AND THE TOP 2 HYBRIDS ──────────────
    # Added 2026-09-18. Additive: every key below this block is untouched and
    # every number c10 already reports is unchanged. All four helpers read rows
    # the search already stored, so this costs 0 new model fits.
    #
    # One entry per (noise strategy, noise level), because c10 runs one search
    # per cell and the ranking can move as the labels get noisier — which is
    # the thing c10 exists to measure. c10 searches with models=['rf'] (:311),
    # so rf is the only downstream model any of these can rank.
    #
    # hybrid_deltas carries the two reference numbers: the hybrid against the
    # mean of the base representations that fed it, and against the best base
    # representation for rf.
    results['top2_by_cell'] = {}
    for _key, _s in _searches.items():
        if _s is None:
            continue
        _cell = {}
        try:
            _b = top_base_reps(_s, 'rf', 2, require_full=False,
                               context=f'c10 {dataset_name} {_key}')
            _cell['top2_base_reps'] = [(e['rep'], e['score'])
                                       for e in _b['selected']]
            _h = top_hybrids(_s, 'rf', 2, require_full=False,
                             context=f'c10 {dataset_name} {_key}')
            _cell['top2_hybrids'] = [
                {'label': e['label'], 'score_mean': e['score_mean'],
                 'budget': e['budget'], 'n_features': e['n_features']}
                for e in _h['selected']]
            if _h['selected']:
                _cell['hybrid_deltas'] = {
                    k: v for k, v in hybrid_deltas(
                        # ⛔ WAS ['row'], FIXED 2026-09-28. top_hybrids returns
                        # each entry as a flat dict and never nests the row under
                        # a 'row' key, so this raised KeyError, the except below
                        # caught it, and every cell in top2_by_cell carried an
                        # error string instead of its two deltas. hybrid_deltas
                        # reads score_mean and blueprint off what it is given and
                        # the entry carries both.
                        _s, 'rf', hybrid=_h['selected'][0],
                        context=f'c10 {dataset_name} {_key}').items()
                    if k != 'notes' or v}
        except Exception as _e:
            _cell['error'] = f'{type(_e).__name__}: {_e}'
        results['top2_by_cell'][str(_key)] = _cell

    results['search_failures'] = search_failures
    # TODO(similarity-sweep 2026-09-25): every score in c10 comes from a
    # find_best_hybrid search per (condition, level) with label_noise=, and the
    # prelim runs set KIRBY_NO_COLD_SEARCH=1. Those noisy-label searches are
    # not in the precompute, so under that setting a cell whose search is not
    # already stored is refused and lands in search_failures. Whether c10 runs
    # in the prelim sweep, and with which searches, is the owner's call.
    _any_cell = next(iter(cells.values()), None)
    results['excluded_folds'] = _any_cell['excluded_folds'] if _any_cell else None
    results['scores_are'] = (
        "held out, leave-one-fold-out: each reported fold is read with the "
        "recipe that scored best on the other folds, for the hybrid and for "
        "the representation arm alike. The folds in 'excluded_folds' "
        "contribute no term to either arm, so four folds are reported.")

    def _curve(level_scores, _levels):
        """auc_norm, Weibull shape and retention for one score-against-level curve.

        level_scores maps a level to that level's held-out score. Returns None
        when fewer than two levels scored, which is not a curve.
        """
        if len(level_scores) < 2:
            return None
        # Robustness metrics: auc_norm (primary) + Weibull shape, computed
        # on the score-vs-level degradation curve (no linear assumption).
        s_arr = np.array(list(level_scores.keys()))
        r2_arr = np.array(list(level_scores.values()))
        # baseline_threshold=None: no skill floor, on any metric. The
        # default of 0.3 was applied on ONE side of the direction —
        # a higher-is-better baseline under 0.3 was dropped and
        # returned NaN, while a lower-is-better baseline only had to
        # be above zero, so every representation was admitted on the
        # rmse datasets and some were silently removed on the
        # spearman ones. Ranking under two different
        # admission rules by dataset is what this changes.
        # Fixed 2026-09-10 (metric-direction audit).
        #
        # ONE rule survives and robustness_metrics applies it whatever is
        # passed here: a clean score at or below zero is refused. Retention
        # divides by it, so a negative clean score flips the curve's sign and a
        # representation that IMPROVES under noise reads as a clean decay.
        # Owner's decision 2026-09-18, measured on ginkgo_hic, where `aac` is
        # in the signed-off pool and scores -0.154 spearman clean.
        rmetrics = robustness_metrics(s_arr, r2_arr,
                                      baseline_threshold=None,
                                      lower_is_better=lower_is_better)

        baseline_r2 = level_scores.get(0.0, level_scores.get(min(_levels)))
        if baseline_r2 is None:
            # The clean cell of this condition produced no score, so there is
            # nothing to divide by. Say nan rather than report a retention.
            baseline_r2 = float('nan')
        max_noise_r2 = level_scores.get(max(_levels), float('nan'))
        if baseline_r2 and not np.isnan(baseline_r2) and not np.isnan(max_noise_r2):
            if lower_is_better:
                # For lower-is-better (MAE/RMSE): retention = baseline / noisy * 100
                # (lower noisy score = better retention)
                retention = float(baseline_r2 / max_noise_r2 * 100)
            else:
                # For higher-is-better (R²): retention = noisy / baseline * 100
                retention = float(max_noise_r2 / baseline_r2 * 100)
        else:
            retention = 0.0

        return {
            'scores_by_level': level_scores,
            'auc_norm': rmetrics['auc_norm'],
            'weibull_tau': rmetrics['weibull_tau'],
            'weibull_beta': rmetrics['weibull_beta'],
            'baseline_r2': baseline_r2,
            'retention_pct': retention,
        }

    # --- Phase 1: Individual rep noise robustness ---
    # One representation on its own has no recipe to re-pick, so its held-out
    # score on a fold is that fold's own score; the reported number is the mean
    # over the folds that contribute.
    print("\n  Phase 1: Individual rep robustness (auc_norm)")
    for rep_name, X_rep in reps.items():
        print(f"    {rep_name} (dim={X_rep.shape[1]}):")
        rep_results = {}

        for strategy in strategies:
            level_scores = {}
            _levels = _levels_for(strategy)

            for level in _levels:
                cell = cells.get((strategy, float(level)))
                if cell is None:
                    continue
                score = cell['per_rep'].get(rep_name)
                if score is None or np.isnan(score):
                    continue
                level_scores[level] = float(score)

            curve = _curve(level_scores, _levels)
            if curve is None:
                print(f"      {strategy}: fewer than 2 levels scored, no curve")
                rep_results[strategy] = {'skipped': True,
                                         'reason': 'fewer_than_2_levels',
                                         'scores_by_level': level_scores}
                continue
            rep_results[strategy] = curve
            print(f"      {strategy}: auc_norm={curve['auc_norm']:.4f}, "
                  f"base={curve['baseline_r2']:.4f}, "
                  f"retention={curve['retention_pct']:.1f}%")

        results['individual_reps'][rep_name] = rep_results

    # The arm the hybrid is actually compared against: on each reported fold,
    # whichever representation scored best on the other folds.
    print("\n  Phase 1b: best single representation, re-picked per fold")
    for strategy in strategies:
        _levels = _levels_for(strategy)
        level_scores = {}
        for level in _levels:
            cell = cells.get((strategy, float(level)))
            if cell is None:
                continue
            score = cell['best_single']
            if score is None or np.isnan(score):
                continue
            level_scores[level] = float(score)
        curve = _curve(level_scores, _levels)
        if curve is None:
            results['best_single'][strategy] = {
                'skipped': True, 'reason': 'fewer_than_2_levels',
                'scores_by_level': level_scores}
            continue
        results['best_single'][strategy] = curve
        print(f"    {strategy}: auc_norm={curve['auc_norm']:.4f}, "
              f"base={curve['baseline_r2']:.4f}, "
              f"retention={curve['retention_pct']:.1f}%")

    # --- Phase 2: Hybrid noise robustness at each budget ---
    # One number per (budget, condition, level), read off that cell's own
    # search: on each reported fold, whichever of that budget's recipes scored
    # best on the other folds.
    print("\n  Phase 2: Hybrid robustness (auc_norm) by budget")

    for budget in budgets:
        print(f"\n    Budget={budget}:")
        budget_results = {}

        for strategy in strategies:
            level_scores = {}
            _levels = _levels_for(strategy)

            for level in _levels:
                cell = cells.get((strategy, float(level)))
                if cell is None:
                    continue
                score = cell['per_budget'][budget]['hybrid_reselected']
                if score is None or np.isnan(score):
                    continue
                level_scores[level] = float(score)

            if not level_scores:
                # No cell of this condition produced a hybrid score. Reporting
                # 0.0 here is how a broken arm looks identical to one that
                # genuinely scored nothing.
                print(f"      {strategy}: NO SCORES -- no cell of this "
                      f"condition produced a hybrid score at this budget.")
                budget_results[strategy] = {'skipped': True,
                                            'reason': 'no_scored_cells'}
                continue

            curve = _curve(level_scores, _levels)
            if curve is None:
                print(f"      {strategy}: fewer than 2 levels scored, no curve")
                budget_results[strategy] = {'skipped': True,
                                            'reason': 'fewer_than_2_levels',
                                            'scores_by_level': level_scores}
                continue
            budget_results[strategy] = curve
            print(f"      {strategy}: auc_norm={curve['auc_norm']:.4f}, "
                  f"base={curve['baseline_r2']:.4f}, "
                  f"retention={curve['retention_pct']:.1f}%")

        results['hybrids'][budget] = budget_results

    # --- Phase 2a: does the recipe move under noise? ---
    # Every hybrid number above now comes from a search run on that cell's own
    # corrupted labels, so the clean-label advantage the old version of this
    # phase measured is gone from the scores. What is left worth recording is
    # which recipe each condition picks, against the recipe the clean-label
    # search picks — at every level, not only the strongest one.
    print("\n  Phase 2a: does the recipe move under noise?")
    # Every condition's level-0 cell came from the one clean-label search, so
    # whichever one is present is the clean recipe.
    _clean_cell = None
    for _s in strategies:
        _c = cells.get((_s, 0.0))
        if _c is not None:
            _clean_cell = _c
            break
    recipe_bias = {}
    for strategy in strategies:
        _rows = {}
        for level in _levels_for(strategy):
            if float(level) == 0.0:
                continue
            cell = cells.get((strategy, float(level)))
            if cell is None or _clean_cell is None:
                continue
            _per_level = {}
            for budget in budgets:
                _clean_recipe = _clean_cell['recipes'].get(budget)
                _noisy_recipe = cell['recipes'].get(budget)
                if _clean_recipe is None or _noisy_recipe is None:
                    continue
                _per_level[budget] = {
                    # Both recipes are named, so a reader can see whether they
                    # even differ before reading the scores.
                    'clean_recipe': _clean_recipe,
                    'noisy_recipe': _noisy_recipe,
                    'recipes_differ': _clean_recipe != _noisy_recipe,
                    # The held-out score this condition's own recipe reaches at
                    # this level, the same number phase 2 reports.
                    'noisy_recipe_score': cell['per_budget'][budget]['hybrid_reselected'],
                }
            if _per_level:
                _rows[float(level)] = _per_level
        recipe_bias[strategy] = _rows
        _n_diff = sum(1 for lv in _rows.values()
                      for r in lv.values() if r.get('recipes_differ'))
        _n_cmp = sum(len(lv) for lv in _rows.values())
        print(f"    {strategy}: of the {_n_cmp} (level, budget) pairs compared, "
              f"{_n_diff} picked a different recipe from the clean-label search")

    results['recipe_bias'] = convert_numpy(recipe_bias)
    results['recipe_bias_schema'] = (
        "One entry per noise condition, then per level, then per budget. "
        "'clean_recipe' is what the level-0 search picked at that budget and "
        "'noisy_recipe' is what this condition's own search picked on its own "
        "corrupted labels. 'noisy_recipe_score' is that recipe's held-out "
        "score, the same number in hybrids[budget][condition]['scores_by_level']."
    )

    # --- Phase 2b: Noise × budget interaction ---
    # For each strategy, extract auc_norm per budget to test whether budget
    # trades off against noise robustness.
    print("\n  Phase 2b: Noise × budget interaction")
    noise_budget = {}
    for strategy in strategies:
        budget_auc_pairs = []
        for budget in budgets:
            bdata = results['hybrids'].get(budget, {}).get(strategy)
            if bdata and not np.isnan(bdata.get('auc_norm', float('nan'))):
                budget_auc_pairs.append((budget, bdata['auc_norm']))
        if len(budget_auc_pairs) >= 2:
            b_arr = np.array([p[0] for p in budget_auc_pairs])
            # auc_norm is already direction-normalized: HIGHER = more robust for
            # any metric (it works on the retention curve). So a POSITIVE
            # budget-vs-auc_norm correlation = higher budget → more robust.
            auc_arr = np.array([p[1] for p in budget_auc_pairs])
            corr = float(np.corrcoef(b_arr, auc_arr)[0, 1]) if np.std(auc_arr) > 0 else 0.0
            noise_budget[strategy] = {
                'budget_auc_norm': {int(b): float(a) for b, a in budget_auc_pairs},
                'correlation': corr,  # positive = higher budget → more robust
            }
            print(f"    {strategy}: budget-auc_norm correlation={corr:.3f} "
                  f"({'higher budget = more robust' if corr > 0.3 else 'weak/no trend'})")
    results['noise_budget_interaction'] = noise_budget

    # --- Phase 3: Feature stability across noise levels ---
    print("\n  Phase 3: Feature stability across noise")
    budget_for_stability = budgets[len(budgets) // 2]  # middle budget
    print(f"    Using budget={budget_for_stability}")

    # One condition only -- feature stability re-runs the whole allocation at every
    # level, so the full set would be eleven times the cost for a question that is
    # about the level, not the shape.
    # Each (level, fold) selection is written to a checkpoint as it finishes
    # and read back on resume (2026-09-25; n07.FoldCheckpoint). c10 wrote its
    # result once, at the end. The searches themselves are already stored by
    # find_best_hybrid (cache=True), so this is the one piece of work a
    # resumed run would otherwise redo in full. c10 changes no training data
    # and finds no neighbours, so it has no similarity setting to sweep and
    # writes no D-versus-D' rows.
    from experiments.noise.n07_remediation import FoldCheckpoint
    from experiments._shared import input_fingerprint
    ckpt = FoldCheckpoint(output_dir, 'c10_noise_robustness', dataset_name,
                          tag='stability', fingerprint={
                              'input': input_fingerprint(data, config),
                              'noise_seed': int(noise_seed),
                              'budget': budget_for_stability})

    for strategy in strategies[:1]:
        features_by_level = {}
        _levels_stability = _levels_for(strategy)

        for level in _levels_stability:
            fold_features = []
            _search_stab = _search_for(strategy, level)
            if _search_stab is None:
                features_by_level[level] = fold_features
                continue
            bp_stab = (_search_stab.best_by_model_budget.get('rf', {})
                       .get(budget_for_stability) or {}).get('blueprint')
            if bp_stab is None:
                print(f"    level={level}: this level's search has no recipe at "
                      f"budget={budget_for_stability}, no selection recorded")
                features_by_level[level] = fold_features
                continue
            # The same corrupted label column the search trained on, sliced to
            # this fold's training rows.
            y_noisy_full = _noisy_column(_spec_for(strategy, level))
            for fold_idx, (train_idx, val_idx) in enumerate(
                    _cv_split(groups, n_folds, sample_X, labels)):
                _stored = ckpt.get([strategy, float(level), int(fold_idx)])
                if _stored is not None:
                    fold_features.append({(r, int(i)) for r, i in _stored})
                    continue
                y_train = label_rows(y_noisy_full, train_idx)

                train_reps = {k: v[train_idx] for k, v in reps.items()}
                val_reps_stab = {k: v[val_idx] for k, v in reps.items()}
                train_augs_stab = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                val_augs_stab = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                try:
                    # build_fold_hybrid carries the whole recipe — evaluator,
                    # augmentation strategy, candidate ranking, margin — where
                    # a bare create_hybrid call passed four of them and took
                    # the engine's defaults for the rest. model and metric_fn
                    # are the estimator and the metric this experiment scores
                    # with, so the internal accept/reject is decided by them
                    # rather than by r2 on the internal evaluator.
                    _, _, fi = build_fold_hybrid(
                        train_reps, val_reps_stab, y_train, bp_stab,
                        augmentations_train=train_augs_stab,
                        augmentations_val=val_augs_stab,
                        budget_override=budget_for_stability,
                        groups=groups[train_idx],
                        model=_rf_factory(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better,
                    )
                    # Collect selected feature identifiers: (rep_name, feature_idx) tuples
                    selected = set()
                    for rep_name, rep_info in fi.items():
                        if isinstance(rep_info, dict) and 'selected_indices' in rep_info:
                            for idx in rep_info['selected_indices']:
                                selected.add((rep_name, int(idx)))
                    fold_features.append(selected)
                    ckpt.put([strategy, float(level), int(fold_idx)],
                             sorted([r, int(i)] for r, i in selected))
                except Exception as _e:
                    print(f"    level={level} fold {fold_idx}: rebuild failed, "
                          f"{type(_e).__name__}: {_e}")
                    continue

            features_by_level[level] = fold_features

        # Feature stability across folds (at level=0)
        if features_by_level.get(0.0) and len(features_by_level[0.0]) >= 2:
            fold_sets = features_by_level[0.0]
            jaccard_across_folds = []
            for i in range(len(fold_sets)):
                for j in range(i + 1, len(fold_sets)):
                    inter = len(fold_sets[i] & fold_sets[j])
                    union = len(fold_sets[i] | fold_sets[j])
                    jaccard_across_folds.append(inter / union if union > 0 else 0)
            results['feature_stability']['fold_jaccard_level0'] = float(np.mean(jaccard_across_folds))
            print(f"    Fold stability (level=0): Jaccard={np.mean(jaccard_across_folds):.3f}")

            # Nogueira stability index (Nogueira, Sechidis & Brown, JMLR 2018)
            total_features = sum(v.shape[1] for v in reps.values())
            nogueira_phi, nogueira_se = _nogueira_stability(fold_sets, total_features)
            results['feature_stability']['nogueira_phi'] = nogueira_phi
            results['feature_stability']['nogueira_se'] = nogueira_se
            if not np.isnan(nogueira_phi):
                # SE can be NaN when M<=2 folds (Nogueira formula needs M>2);
                # writing NaN to JSON is not a valid value, so skip the CI.
                if not np.isnan(nogueira_se):
                    ci_lo = nogueira_phi - 1.96 * nogueira_se
                    ci_hi = nogueira_phi + 1.96 * nogueira_se
                    results['feature_stability']['nogueira_ci_95'] = [ci_lo, ci_hi]
                    print(f"    Nogueira stability (level=0): phi={nogueira_phi:.3f} "
                          f"(95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
                else:
                    print(f"    Nogueira stability (level=0): phi={nogueira_phi:.3f} "
                          f"(CI undefined: requires M>2 folds)")

        # Feature stability across noise levels (compare level=0 features to each level)
        if features_by_level.get(0.0):
            ref_features = set.union(*features_by_level[0.0])  # union across folds at level=0
            noise_stability = {}
            for level in _levels_stability:
                if level == 0.0 or level not in features_by_level:
                    continue
                if not features_by_level[level]:
                    continue
                noisy_features = set.union(*features_by_level[level])
                inter = len(ref_features & noisy_features)
                union = len(ref_features | noisy_features)
                jaccard = inter / union if union > 0 else 0.0
                noise_stability[level] = float(jaccard)
                print(f"    level={level:.1f} vs level=0: Jaccard={jaccard:.3f} "
                      f"(shared={inter}/{len(ref_features)} ref features)")

            results['feature_stability']['noise_jaccard'] = noise_stability

            # NDS of feature stability (does feature overlap degrade linearly with noise?)
            if len(noise_stability) >= 2:
                s_arr = np.array(list(noise_stability.keys()))
                j_arr = np.array(list(noise_stability.values()))
                slope, _ = np.polyfit(s_arr, j_arr, 1)
                results['feature_stability']['feature_nds'] = float(slope)
                print(f"    Feature stability NDS: {slope:.4f} (negative = features change with noise)")

        # Per-rep allocation stability across noise
        if features_by_level.get(0.0):
            print("\n    Per-rep feature count stability:")
            for level in [0.0] + [v for v in _levels_stability if v > 0]:
                if level not in features_by_level or not features_by_level[level]:
                    continue
                # Average feature count per rep across folds
                rep_counts = {}
                for fold_feats in features_by_level[level]:
                    fold_reps = set(r for r, _ in fold_feats)
                    for rep_name in fold_reps:
                        rep_counts.setdefault(rep_name, [])
                        count = sum(1 for r, _ in fold_feats if r == rep_name)
                        rep_counts[rep_name].append(count)
                counts_str = ", ".join(
                    f"{r}={np.mean(c):.0f}" for r, c in sorted(rep_counts.items()))
                print(f"      level={level:.1f}: {counts_str}")

    # 2026-09-25: c10 finds no neighbours and changes no training data, so no
    # similarity setting is swept here; the checkpoint is recorded.
    results['similarity'] = {
        'swept': False,
        'why_not_swept': ('c10 finds no neighbours and changes no training '
                          'data; its scores are held-out search scores'),
        'checkpoint_file': ckpt.path,
        'selections_read_back': ckpt.n_resumed}

    # --- Summary table ---
    # auc_norm = normalised area under the retention curve. It is already
    # direction-normalized: HIGHER = more robust for any metric direction.
    print("\n  NOISE ROBUSTNESS SUMMARY (auc_norm, higher = more robust):")
    print(f"  {'Representation':<20} " +
          " ".join(f"{s:<12}" for s in strategies))
    print("  " + "-" * (20 + 13 * len(strategies)))

    # Individual reps
    for rep_name in sorted(results['individual_reps'].keys()):
        row = f"  {rep_name:<20} "
        for s in strategies:
            auc = results['individual_reps'][rep_name].get(s, {}).get('auc_norm', float('nan'))
            row += f"{auc:>+8.4f}    "
        print(row)

    # The representation re-picked per fold — the arm the hybrid is compared
    # against, on the same held-out folds.
    row = f"  {'best_single':<20} "
    for s in strategies:
        auc = results['best_single'].get(s, {}).get('auc_norm', float('nan'))
        row += f"{auc:>+8.4f}    "
    print(row)

    # Hybrids
    for budget in sorted(b for b in results['hybrids'] if isinstance(b, (int, float))):
        row = f"  hybrid_b{budget:<11} "
        for s in strategies:
            auc = results['hybrids'][budget].get(s, {}).get('auc_norm', float('nan'))
            row += f"{auc:>+8.4f}    "
        print(row)

    return results


if __name__ == '__main__':
    import argparse
    parser = create_experiment_parser('C10: Noise Robustness & Feature Stability',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--levels', type=float, nargs='+', default=None,
                        help='Noise levels. Default is the grid for this dataset in '
                             'experiments.dataset_policy: anchored to the published '
                             'assay error in the label\'s own units for herg_ki, '
                             'caco2 and logd, a fraction of the label SD otherwise, '
                             'and the censored fraction for censoring.')
    parser.add_argument('--strategies', type=str, nargs='+', default=None,
                        choices=list(REGRESSION_STRATEGIES),
                        help='Noise conditions (default: all eleven).')
    parser.add_argument('--noise-seed', type=int, default=42,
                        help='Base seed for the noise draw only.')
    # Refused by name, not ignored: a job script written against the old scheme would
    # otherwise run silently under the new one, where the number means something else.
    parser.add_argument('--sigmas', type=float, nargs='+', default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sigmas is not None:
        parser.error("--sigmas is gone. A noise level is now the amount of noise "
                     "DELIVERED (a root-mean-square), not a Gaussian sigma, so the "
                     "old numbers mean something different. Use --levels.")
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  levels=args.levels, strategies=args.strategies,
                  noise_seed=args.noise_seed)
    save_experiment_results(results, args.output, 'c10_noise_robustness', args.dataset)
