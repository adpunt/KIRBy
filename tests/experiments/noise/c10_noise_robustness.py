#!/usr/bin/env python3
"""C10: Noise Robustness & Feature Stability — how far each configuration's
held-out score falls when its training labels are damaged.

Paper: Complexity
Question: are hybrids more noise-robust than single representations, and do
the selected features stay the same as noise rises?
Dependencies: None (standalone).

ALIGNED 2026-10-05 (chat 5, HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). What changed:
- Every configuration is chosen per outer fold, on that fold's training rows
  (noise_helpers.fold_configurations). Until now c10 ran one whole-dataset
  search per (condition, level) and re-picked recipes per fold from it, so the
  held-out fold's rows had helped choose every recipe (R2).
- The dataset's two decided models only, each with its 2 best base
  representations and 2 best hybrids: 8 fitted configurations per fold (R5).
  Was every downstream model (5 or 6), every representation, one best-single
  arm and one hybrid per budget (8 budgets).
- The decided noise grid: six kinds, levels by dataset group, 3 draws per
  non-zero level, level 0 once (KNN_SEARCH_SPACE_RECORD.md, "The noise grid").
  Was eleven kinds, one draw.
- Robustness is reported as the plain difference damaged-minus-clean in the
  dataset's own metric on every dataset, and as auc_norm where 4 levels exist
  (full_knn). The dirty run found that auc_norm divides each side by its own
  clean score, so it could show the hybrid losing less while its rmse was
  worse at every level; the raw scores by level and the plain difference are
  now written next to it.
- DECIDED 2026-10-06 (Adelaide delegated): the "does the recipe move under
  damage" part (a per-fold search on damaged training labels per kind, level
  and draw) is dropped; n15 asks a close question. KIRBY_C10_RECIPE_SEARCHES
  is gone. The one-hybrid-per-budget question stays dropped.
- DECIDED 2026-10-06 (Adelaide delegated): one draw under censoring
  (noise_helpers.draws_for); its draws gave identical labels.
- The result file is written after every fold (it was written once, at the
  end, and the dirty run left a stale herg_ki.json beside a newer partial
  file).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_domain_metric, convert_numpy,
    get_model_factories, hybrid_feature_signature,
    input_fingerprint,
)
from experiments.dataset_policy import noise_groups_for
from experiments.noise.noise_helpers import (
    CONFIGURATION_SPACES, DECIDED_REGRESSION_CONDITIONS, configuration_record,
    dataset_knn_group, decided_conditions, decided_levels, draws_for,
    fold_configurations, fold_model_inputs, fold_spread, inject_condition,
    level_unit, noise_outer_folds, recipe_is_geometry,
)
from kirby.noise_robustness import robustness_metrics

# Optional NoiseInject import
try:
    import noiseInject  # noqa: F401
    HAS_NOISEINJECT = True
except ImportError:
    HAS_NOISEINJECT = False


EXP_FULL_NAME = 'c10_noise_robustness'

# Three draws per non-zero level (owner's decision 2026-09-12; the noise grid of
# 2026-10-05 keeps it). Level 0 is the labels as delivered and runs once.
# Censoring runs draw 0 only (draws_for, decided 2026-10-06).
NOISE_DRAWS = (0, 1, 2)

# The identity every checkpoint line and the result file carry. Bumped
# 2026-10-05 so no partial file from the whole-dataset design is read back.
# Bumped 2026-10-06: recipe-movement part dropped, one draw under censoring.
RESULT_SCHEMA = 'c10_per_fold_8_configurations_2026_10_06'

RECIPE_MOVEMENT_DROPPED = (
    'DECIDED 2026-10-06 (Adelaide delegated): the per-fold searches on '
    'damaged training labels (does the recipe move under damage) are not '
    'run; n15 asks a close question')

# The hybrid slots whose selected features phase 3 records.
HYBRID_SLOTS = ('hybrid', 'hybrid__general2')

# The cell that stands for clean labels: level 0, one draw.
CLEAN_CELL = ('clean', 0.0, 0)


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

    # Binary matrix Z: M x d (a selection indicator, not a representation)
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


def _hybrid_detail(entry):
    """The recipe fields of each chosen hybrid, under the names top_hybrids
    gives them. configuration_record reads 'evaluator' and 'n_reps', which a
    top_hybrids entry does not carry (it has 'internal_evaluator' and
    'rep_count'), so those two come out as None there; written here as well
    (2026-10-05)."""
    if entry is None:
        return []
    return [{k: h.get(k) for k in (
        'label', 'score_mean', 'budget', 'n_features', 'allocation_method',
        'importance_method', 'internal_evaluator', 'rep_count', 'step_size',
        'aug_condition')} for h in (entry.get('hybrids') or [])]


def _scalar_items(d):
    return {k: v for k, v in (d or {}).items()
            if isinstance(v, (bool, int, float, str, type(None)))}


def run(data, config, output_dir, dataset_name, levels=None, strategies=None,
        noise_seed=42, **retired):
    """Exp 10: noise robustness of each fitted configuration, per outer fold.

    Per outer fold (noise_outer_folds), per decided model: its 2 best base
    representations and 2 best hybrids, ranked by a search on that fold's
    training rows (fold_configurations). Each of these 8 configurations is
    fitted on the fold's training rows with damaged labels and scored on the
    held-out rows' clean labels, for every (kind, level, draw) of the decided
    grid. Hybrids are rebuilt on the damaged training labels from the fold's
    recipe, so their feature selection sees the damage; the recipe itself is
    the one the clean-label search picked.

    Two parts:
      1. robustness   score against level per configuration slot; the plain
                      difference damaged-minus-clean on every dataset; auc_norm
                      (and the Weibull shape) where the dataset has 4 levels.
      2. feature stability   the selected features of each hybrid slot, under
                      the first kind swept, at every level and draw.
    The recipe-movement part (searches on damaged labels) was dropped
    2026-10-06.

    Robustness numbers are in the dataset's own metric and direction
    (get_domain_metric). Summaries over folds are mean and SD (fold_spread);
    no significance test.
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

    for _dead in ('sigmas', 'strategy'):
        if _dead in retired:
            raise TypeError(
                f"c10 no longer takes {_dead!r}. A noise level is now the amount of "
                f"noise DELIVERED, not a Gaussian sigma, so the old numbers mean "
                f"something different. Pass levels=... and strategies=... instead; "
                f"the per-dataset grids are in noise_helpers.decided_levels.")
    if retired:
        raise TypeError(f"unexpected arguments: {sorted(retired)}")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    labels = np.asarray(data['labels'])
    reps = data['reps']
    augmentations = data.get('augmentations') or {}
    knn_group = dataset_knn_group(data, dataset_name)

    # ── THE NOISE GRID (KNN_SEARCH_SPACE_RECORD.md "The noise grid", 2026-10-05)
    # Six kinds; strategies= may name a subset of them, never a cut kind.
    allowed = decided_conditions(data)
    if strategies is None:
        strategies = list(allowed)
    else:
        _bad = [s for s in strategies if s not in allowed]
        if _bad:
            raise ValueError(
                f"c10: {_bad} are not among the decided kinds of damage "
                f"{allowed} (noise grid, 2026-10-05).")
        strategies = list(strategies)

    # Scaffold groups for grouped_shifted. Acyclic molecules are singletons
    # (NOISE_DESIGN.md 2a rule 2) and this grouping is deliberately NOT the one
    # that sets the fold boundaries. None when the dataset has no SMILES, in
    # which case the grouped kind is dropped rather than run over invented groups.
    noise_groups, n_noise_groups, largest_group_share = noise_groups_for(data)
    if noise_groups is None:
        dropped = [c for c in strategies if c.startswith('grouped_')]
        if dropped:
            print(f"  {', '.join(dropped)}: DROPPED, this dataset has no SMILES to "
                  f"build scaffold groups from. Grouped noise over invented groups is "
                  f"uniform noise wearing a grouped name.")
            strategies = [c for c in strategies if c not in dropped]
    else:
        print(f"  noise groups: {n_noise_groups} scaffolds, largest holds "
              f"{largest_group_share:.1%} of molecules")

    def _levels_for(strategy):
        """The decided levels of one kind, 0 first. levels= may pick a subset
        of them (a quick run); a level outside the decided grid raises."""
        dec = tuple(float(x) for x in decided_levels(data, dataset_name, strategy))
        if levels is None:
            return dec
        lv = tuple(sorted({float(x) for x in levels} | {0.0}))
        _bad = [x for x in lv if x not in dec]
        if _bad:
            raise ValueError(
                f"c10: levels {_bad} are not in the decided grid for "
                f"{strategy} on {dataset_name} ({knn_group}): {dec}.")
        return lv

    noise_grid = {s: list(_levels_for(s)) for s in strategies}
    # Every (kind, level, draw) a configuration is fitted under; level 0 once.
    cells = [CLEAN_CELL] + [(s, float(lv), int(d)) for s in strategies
                            for lv in noise_grid[s] if float(lv) != 0.0
                            for d in draws_for(s, NOISE_DRAWS)]
    print(f"  {knn_group}: {len(strategies)} kinds of damage, "
          f"{len(cells)} label sets per configuration and fold "
          f"(level 0 once, {len(NOISE_DRAWS)} draws per non-zero level"
          f"{', 1 under censoring' if 'censoring' in strategies else ''})")

    # ── FOLDS (handoff 2.4 item 14) ───────────────────────────────────────
    folds, n_folds = noise_outer_folds(data, config)
    print(f"  outer folds: {n_folds} (running {[f for f, _, _ in folds]})")

    from experiments.noise.n07_remediation import FoldCheckpoint
    ckpt = FoldCheckpoint(output_dir, EXP_FULL_NAME, dataset_name, tag='cells',
                          fingerprint={'schema': RESULT_SCHEMA,
                                       'input': input_fingerprint(data, config),
                                       'noise_seed': int(noise_seed),
                                       'n_folds': int(n_folds)})

    stability_condition = strategies[0] if strategies else None

    state = {'rows': [], 'per_fold': {}, 'stability': [],
             'folds_done_main': []}

    def _factory_for(model, n_train):
        return get_model_factories([model], task_type='regression',
                                   n_samples=int(n_train)).get(model)

    def _score_cell(model, entry, factory, fold_i, tr, va, cell):
        """Fit the model's 4 configurations under one label set; held-out
        score against the clean held-out labels."""
        cond, level, draw = cell
        y_tr, mask, prov = inject_condition(
            data, dataset_name, cond, level, draw, train_idx=tr,
            groups=noise_groups, base_seed=int(noise_seed))
        inputs, unavail, names = fold_model_inputs(
            data, dataset_name, model, entry, train_idx=tr, val_idx=va,
            y_train=y_tr, factory=factory, metric=metric)
        y_va = labels[va]
        rows, errors = [], {}
        for space in CONFIGURATION_SPACES:
            if space not in inputs:
                continue
            X_tr, X_va = inputs[space]
            try:
                est = factory()
                est.fit(X_tr, y_tr)
                held = float(metric_fn(y_va, est.predict(X_va)))
                train = float(metric_fn(y_tr, est.predict(X_tr)))
            except Exception as exc:
                errors[space] = f'{type(exc).__name__}: {exc}'
                continue
            rows.append({
                'fold': int(fold_i), 'model': model, 'space': space,
                'configuration': names.get(space), 'condition': cond,
                'level': float(level), 'draw': int(draw),
                'n_features': int(X_tr.shape[1]),
                'n_corrupted_train': int(np.asarray(mask).sum()),
                # Scored against the labels the model was trained on, i.e.
                # the damaged ones at a non-zero level.
                'train_score': train,
                'heldout_score': held,
            })
        return {'rows': rows, 'unavailable': dict(unavail),
                'fit_errors': errors,
                'provenance': _scalar_items(prov)}

    def _selected_features(model, factory, recipe, tr, va, y_tr):
        """The selected features of one hybrid recipe rebuilt on these
        training labels: [[representation, [feature numbers]], ...]."""
        from experiments.noise import noise_cache as nc
        bp, aug_cond = recipe
        geometry = recipe_is_geometry(bp)
        augs = augmentations if (aug_cond or 'none') != 'none' else None
        groups = data.get('groups')
        built = nc.cached_hybrid_for_fold(
            dataset_name, {k: v[tr] for k, v in reps.items()},
            {k: v[va] for k, v in reps.items()}, y_tr, bp,
            None if groups is None else np.asarray(groups)[tr],
            None if (geometry or factory is None) else factory(),
            metric_fn, lower_is_better,
            model_name=None if geometry else model,
            metric_name=metric_name, train_idx=tr,
            augmentations_train=(None if augs is None else
                                 {a: X[tr] for a, X in augs.items()}),
            augmentations_val=(None if augs is None else
                               {a: X[va] for a, X in augs.items()}),
            want_feature_info=True)
        if built is None or built[0] is None or len(built) < 3:
            return None
        return [[name, [int(i) for i in idx]]
                for name, idx in hybrid_feature_signature(built[2])]

    def _write_snapshot(complete):
        """The result file as it stands. A snapshot is marked
        result_status='empty' so run_all's resume check recomputes it; the
        finished cells are then read back from the checkpoint."""
        res = _assemble(complete)
        if complete or output_dir is None:
            return res
        snap = dict(res, result_status='empty', has_computed_results=False,
                    snapshot_note=(
                        'written per fold while the run is in progress; marked '
                        'empty so a resumed run recomputes it, reading the '
                        'finished cells back from the checkpoint'))
        try:
            save_experiment_results(convert_numpy(snap), output_dir,
                                    EXP_FULL_NAME, dataset_name,
                                    metric_name=metric_name,
                                    lower_is_better=lower_is_better)
        except Exception as exc:
            print(f"  snapshot not written: {type(exc).__name__}: {exc}")
        return res

    def _assemble(complete):
        rows = state['rows']
        models = sorted({r['model'] for r in rows})

        # ── robustness, per model, per configuration slot, per kind ──────
        robustness = {}
        for model in models:
            robustness[model] = {}
            for space in CONFIGURATION_SPACES:
                rs = [r for r in rows if r['model'] == model and r['space'] == space]
                if not rs:
                    continue
                by_fold = {}
                for r in rs:
                    by_fold.setdefault(r['fold'], {})[
                        (r['condition'], r['level'], r['draw'])] = r['heldout_score']
                robustness[model][space] = {}
                for cond in strategies:
                    lvls = [float(x) for x in noise_grid[cond]]
                    nz = [x for x in lvls if x != 0.0]
                    score_by_level = {0.0: []}
                    diff_by_level = {x: [] for x in nz}
                    auc_fold, tau_fold, beta_fold = [], [], []
                    for f, sc in sorted(by_fold.items()):
                        clean = sc.get(CLEAN_CELL)
                        if clean is not None:
                            score_by_level[0.0].append(clean)
                        per_draw_auc, per_draw_tau, per_draw_beta = [], [], []
                        for x in nz:
                            vals = [sc[(cond, x, d)]
                                    for d in draws_for(cond, NOISE_DRAWS)
                                    if (cond, x, d) in sc]
                            if vals:
                                score_by_level.setdefault(x, []).append(
                                    float(np.mean(vals)))
                            if vals and clean is not None:
                                # damaged minus clean, in the metric's own
                                # units; mean over the draws of this fold
                                diff_by_level[x].append(
                                    float(np.mean([v - clean for v in vals])))
                        if len(lvls) >= 4 and clean is not None:
                            for d in draws_for(cond, NOISE_DRAWS):
                                pts = [(x, sc.get((cond, x, d))) for x in nz]
                                if any(v is None for _, v in pts):
                                    continue
                                m = robustness_metrics(
                                    np.array([0.0] + [x for x, _ in pts]),
                                    np.array([clean] + [v for _, v in pts]),
                                    baseline_threshold=None,
                                    lower_is_better=lower_is_better)
                                per_draw_auc.append(m['auc_norm'])
                                per_draw_tau.append(m['weibull_tau'])
                                per_draw_beta.append(m['weibull_beta'])

                        def _fmean(v):
                            v = [x for x in v if x is not None and np.isfinite(x)]
                            return float(np.mean(v)) if v else None
                        if per_draw_auc:
                            auc_fold.append(_fmean(per_draw_auc))
                            tau_fold.append(_fmean(per_draw_tau))
                            beta_fold.append(_fmean(per_draw_beta))
                    out = {
                        'heldout_score_by_level': {
                            str(x): fold_spread(v) for x, v in sorted(score_by_level.items())},
                        'difference_damaged_minus_clean': {
                            str(x): fold_spread(v) for x, v in sorted(diff_by_level.items())},
                    }
                    if len(lvls) >= 4:
                        out['auc_norm'] = fold_spread(auc_fold)
                        out['weibull_tau'] = fold_spread(tau_fold)
                        out['weibull_beta'] = fold_spread(beta_fold)
                    else:
                        out['auc_norm'] = None
                        out['auc_norm_absent'] = (
                            f'{knn_group}: {len(lvls)} levels; auc_norm needs 4 '
                            f'including level 0 (decided 2026-10-05)')
                    robustness[model][space][cond] = out

        # ── feature stability ────────────────────────────────────────────
        stability = {}
        widths = sum(int(np.asarray(v).shape[1]) for v in reps.values()) + sum(
            int(np.asarray(v).shape[1]) for v in augmentations.values())
        for model in sorted({s['model'] for s in state['stability']}):
            stability[model] = {}
            for slot in HYBRID_SLOTS:
                ss_ = [s for s in state['stability']
                       if s['model'] == model and s['slot'] == slot]
                if not ss_:
                    continue
                sets = {}
                for s in ss_:
                    sets[(s['fold'], s['level'], s['draw'])] = {
                        (name, int(i)) for name, idx in s['features'] for i in idx}
                lvl0 = {f: v for (f, lv, d), v in sets.items() if lv == 0.0}
                blk = {'condition': stability_condition,
                       'note': ('each fold has its own recipe in this slot, '
                                'chosen by that fold\'s search, so the '
                                'across-fold numbers include the recipe '
                                'changing between folds')}
                if len(lvl0) >= 2:
                    fs_ = [lvl0[f] for f in sorted(lvl0)]
                    jac = []
                    for i in range(len(fs_)):
                        for j in range(i + 1, len(fs_)):
                            u = len(fs_[i] | fs_[j])
                            jac.append(len(fs_[i] & fs_[j]) / u if u else 0.0)
                    # One value per PAIR of folds; fold_spread's n_folds
                    # counts pairs here.
                    blk['fold_jaccard_level0'] = fold_spread(jac)
                    blk['fold_jaccard_level0_unit'] = 'one value per pair of folds'
                    blk['nogueira_phi_level0'] = _nogueira_stability(fs_, widths)[0]
                else:
                    blk['fold_jaccard_level0'] = None
                    blk['fold_jaccard_level0_absent'] = 'fewer than 2 folds'
                # Same fold, same recipe: level 0 against each damaged level.
                by_level = {}
                for (f, lv, d), v in sets.items():
                    if lv == 0.0 or f not in lvl0:
                        continue
                    u = len(lvl0[f] | v)
                    by_level.setdefault(lv, {}).setdefault(f, []).append(
                        len(lvl0[f] & v) / u if u else 0.0)
                blk['jaccard_vs_level0'] = {
                    str(lv): fold_spread([float(np.mean(x)) for x in per_f.values()])
                    for lv, per_f in sorted(by_level.items())}
                if len(by_level) >= 2:
                    lv_arr = np.array(sorted(by_level))
                    j_arr = np.array([blk['jaccard_vs_level0'][str(x)]['mean']
                                      for x in lv_arr])
                    blk['feature_nds'] = float(np.polyfit(lv_arr, j_arr, 1)[0])
                counts = {}
                for (f, lv, d), v in sets.items():
                    c = counts.setdefault(str(lv), {})
                    for name, _ in v:
                        c[name] = c.get(name, 0) + 1
                n_sets = {}
                for (f, lv, d) in sets:
                    n_sets[str(lv)] = n_sets.get(str(lv), 0) + 1
                blk['mean_features_per_representation'] = {
                    lv: {name: c / n_sets[lv] for name, c in sorted(cs.items())}
                    for lv, cs in sorted(counts.items())}
                stability[model][slot] = blk

        return {
            'result_schema': RESULT_SCHEMA,
            'complete': bool(complete),
            'design': ('per outer fold, per decided model: 2 base '
                       'representations and 2 hybrids chosen by a search on '
                       'the fold\'s training rows (clean labels); each fitted '
                       'on damaged training labels, scored on the held-out '
                       'rows\' clean labels'),
            'selection_scope': 'fold_train_rows',
            'metric_name': metric_name,
            'lower_is_better': bool(lower_is_better),
            'n_folds': int(n_folds),
            'folds_run': [int(f) for f, _, _ in folds],
            'folds_done': list(state['folds_done_main']),
            'configuration_spaces': list(CONFIGURATION_SPACES),
            'knn_group': knn_group,
            'noise_grid': noise_grid,
            'level_units': {s: level_unit(s) for s in strategies},
            'draws_per_nonzero_level': {
                s: len(draws_for(s, NOISE_DRAWS)) for s in strategies},
            'noise_seed': int(noise_seed),
            'n_noise_groups': n_noise_groups,
            'largest_group_share': largest_group_share,
            'per_fold': {str(k): v for k, v in sorted(state['per_fold'].items())},
            'rows': rows,
            'robustness': robustness,
            'robustness_schema': (
                'robustness[model][space][condition]: '
                'heldout_score_by_level (level -> fold_spread of the fold\'s '
                'mean over draws), difference_damaged_minus_clean (level -> '
                'fold_spread of the fold\'s mean over draws of damaged minus '
                'clean, in the metric\'s own units and direction), auc_norm '
                '(fold_spread of the fold\'s mean over draws; absent on '
                'fast_knn by decision). space base_1 is each fold\'s best '
                'representation for that model, so it can name a different '
                'representation on each fold; rows[] names it per fold.'),
            'feature_stability': stability,
            'recipe_movement': None,
            'recipe_movement_dropped': RECIPE_MOVEMENT_DROPPED,
            'budget_question_dropped': (
                'one hybrid per budget is not run under the 8-configuration '
                'rule (2026-10-05); stays dropped 2026-10-06'),
            # c10 finds no neighbours and changes no training data beyond
            # the injected damage, so no neighbour setting is swept and no
            # distance or similarity hybrid is used (handoff 2.3 item 1).
            'similarity': {
                'swept': False,
                'why_not_swept': ('c10 finds no neighbours; its scores are '
                                  'held-out scores of fitted configurations'),
                'checkpoint_file': ckpt.path,
                'cells_read_back': ckpt.n_resumed},
        }

    # ── PART 1 AND 2: the 8 configurations, every label set; feature stability
    print("\n  Part 1: the 8 configurations under every label set, per fold")
    for fold_i, tr, va in folds:
        print(f"\n  fold {fold_i}: {len(tr)} training rows, {len(va)} held out")
        entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                      train_idx=tr, n_folds=n_folds)
        frec = {'n_train': int(len(tr)), 'n_held_out': int(len(va)),
                'configurations': {}, 'not_fitted': {}, 'unavailable': {},
                'fit_errors': {}, 'provenance': {}}
        for model, entry in entries.items():
            rec = configuration_record(entry)
            rec['hybrid_detail'] = _hybrid_detail(entry)
            frec['configurations'][model] = rec
            if entry is None:
                frec['not_fitted'][model] = 'the per-fold search failed for this model'
                print(f"    {model}: NOT FITTED, the per-fold search failed")
                continue
            factory = _factory_for(model, len(tr))
            if factory is None:
                frec['not_fitted'][model] = (
                    'no model factory at this training size '
                    '(get_model_factories skipped it)')
                print(f"    {model}: NOT FITTED, no factory at {len(tr)} rows")
                continue
            print(f"    {model}: base {entry.get('base')}, "
                  f"{len(entry.get('hybrids') or [])} hybrids")
            for cell in cells:
                key = ['score', int(fold_i), model, cell[0], cell[1], cell[2]]
                got = ckpt.get(key)
                if got is None:
                    try:
                        got = _score_cell(model, entry, factory, fold_i, tr, va,
                                          cell)
                    except Exception as exc:
                        print(f"      {cell}: FAILED {type(exc).__name__}: {exc}")
                        frec['fit_errors'][f'{model}|{cell}'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    ckpt.put(key, got)
                state['rows'].extend(got['rows'])
                if got.get('unavailable'):
                    frec['unavailable'].setdefault(model, {}).update(
                        got['unavailable'])
                if got.get('fit_errors'):
                    frec['fit_errors'][f'{model}|{cell}'] = got['fit_errors']
                if cell != CLEAN_CELL:
                    frec['provenance'].setdefault(
                        f'{cell[0]}|{cell[1]}|{cell[2]}', got.get('provenance'))

            # Part 2: selected features of each hybrid slot, under the first
            # kind swept, at every level and draw (was one recipe at the
            # middle of the whole budget list, one draw).
            if stability_condition is None:
                continue
            stab_cells = [CLEAN_CELL] + [c for c in cells
                                         if c[0] == stability_condition]
            for slot in HYBRID_SLOTS:
                recipe = (entry.get('recipes') or {}).get(slot)
                if recipe is None or recipe[0] is None:
                    continue
                for cell in stab_cells:
                    key = ['features', int(fold_i), model, slot, cell[0],
                           cell[1], cell[2]]
                    sel = ckpt.get(key)
                    if sel is None:
                        try:
                            y_tr, _, _ = inject_condition(
                                data, dataset_name, cell[0], cell[1], cell[2],
                                train_idx=tr, groups=noise_groups,
                                base_seed=int(noise_seed))
                            feats = _selected_features(model, factory, recipe,
                                                       tr, va, y_tr)
                        except Exception as exc:
                            print(f"      {slot} {cell}: rebuild failed, "
                                  f"{type(exc).__name__}: {exc}")
                            continue
                        if feats is None:
                            continue
                        sel = {'features': feats}
                        ckpt.put(key, sel)
                    state['stability'].append({
                        'fold': int(fold_i), 'model': model, 'slot': slot,
                        'level': float(cell[1]), 'draw': int(cell[2]),
                        'features': sel['features']})
        state['per_fold'][int(fold_i)] = frec
        state['folds_done_main'].append(int(fold_i))
        _write_snapshot(False)

    results = _write_snapshot(True)

    # --- Summary table, one per model: plain difference at the top level ---
    for model, blk in results['robustness'].items():
        print(f"\n  NOISE ROBUSTNESS, model {model}: held-out {metric_name} "
              f"damaged minus clean at each kind's top level, mean over folds "
              f"({'lower' if lower_is_better else 'higher'} is better, so "
              f"{'positive' if lower_is_better else 'negative'} = worse)")
        print(f"  {'configuration':<20} " + " ".join(f"{s[:12]:<13}" for s in strategies))
        for space, per_c in blk.items():
            row = f"  {space:<20} "
            for s in strategies:
                top = str(float(max(noise_grid[s])))
                v = ((per_c.get(s) or {}).get('difference_damaged_minus_clean')
                     or {}).get(top) or {}
                mean = v.get('mean')
                row += f"{mean:>+9.4f}     " if mean is not None else f"{'n/a':>9}     "
            print(row)

    return results


if __name__ == '__main__':
    import argparse
    parser = create_experiment_parser('C10: Noise Robustness & Feature Stability',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--levels', type=float, nargs='+', default=None,
                        help='A subset of the decided levels (a quick run). Default: '
                             'every decided level for this dataset '
                             '(noise_helpers.decided_levels).')
    parser.add_argument('--strategies', type=str, nargs='+', default=None,
                        choices=list(DECIDED_REGRESSION_CONDITIONS),
                        help='Kinds of damage (default: the six decided).')
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
