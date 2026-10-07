#!/usr/bin/env python3
"""C05: Learning curves -- sample efficiency.

Paper: Complexity
Conclusion: Hybrids are more sample-efficient than individual reps.
Dependencies: None (standalone).

At each sample size, whole scaffold groups are drawn until about that many
molecules are held (three draws below full size, one at full size). Each draw
is split into outer folds (nested_cv_outer_folds, config['n_splits']). On each
outer fold's TRAINING rows one search runs on the dataset's decided grid
(fold_standard_selection), and per decided model it names the 2 best single
representations and the 2 best hybrids. Each is fitted on those training rows
and scored on the fold's held-out rows.

The curve compares, per decided model, the top hybrid with the top single
representation at each sample size (owner, 2026-10-05, handoff 2.3 (B)). The
second-ranked of each is scored and stored beside them.

Aligned 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). Until then: five
downstream models; one search over every row of each draw, whose own folds
both picked the winner and scored it; a distance search; 'capped'
augmentation on every dataset with donors; a second pass that rebuilt recipes
chosen by the whole-dataset search; budgets and grid from the fast tier.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    first_outer_folds,
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
    fit_learning_curve, get_domain_metric,
    hybrid_deltas,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_standard_selection,
)
from experiments.decided_grid import (
    decided_models, decided_search, UndecidedSearch)
from kirby.find_best_hybrid import (
    nested_cv_outer_folds, nested_cv_cache_namespace,
    get_model, _fit_model, _predict_for_metric, _impute_nan)
from kirby.result_cache import StaleResultDB
from kirby.scaling import FeatureScaler

# Bumped 2026-10-05 when the experiment moved to per-fold searches. A result
# file without this value was computed by the old design (one search over every
# row of a draw, five models) and is not comparable.
RESULT_VERSION = 'c05_per_fold_2026-10-05'

# The outer split's seed. The same 42 every other experiment's per-fold
# searches use, so at the full-size point (the draw is every row) c05's
# searches are the dataset's standard per-fold searches and are read from the
# store, not run again.
SPLIT_SEED = 42

# The recipe fields written beside each hybrid's scores.
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count')

# fit_learning_curve fits score = asymptote - a / n^b: three numbers. With three
# points or fewer it passes through every point exactly whatever they are, so
# its asymptote says nothing (dirty run, 2026-10-02). Fitted only from four.
CURVE_FIT_PARAMETERS = 3


def _heldout_score(X_tr, X_va, y_tr, y_va, model_name, task_type, metric_fn,
                   train_groups=None):
    """Fit one decided model on the training rows, score the held-out rows.

    Features: missing values imputed from the training rows, 0/1 features left
    as they are and the rest standardised on the training rows (FeatureScaler,
    handoff 2.3 item 17), and each held-out feature capped at the range it
    takes on the scaled training rows (owner delegated, 2026-10-05; on adme_hlm
    one held-out molecule sat 83 million training standard deviations out).
    Predictions follow the engine's rule (_predict_for_metric): probabilities
    for AUROC, labels otherwise.

    Returns (score, None) or (None, why).
    """
    X_tr, X_va, _ = _impute_nan(np.asarray(X_tr), np.asarray(X_va))
    scaler = FeatureScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    if X_tr_s.shape[1]:
        X_va_s = np.clip(X_va_s, X_tr_s.min(0), X_tr_s.max(0))
    m = get_model(model_name, task_type, n_samples=len(y_tr))
    if m is None:
        return None, (f'{model_name} is not fitted on {len(y_tr)} training '
                      f'rows (model size limit)')
    _fit_model(m, X_tr_s, y_tr, groups=train_groups)
    y_pred = _predict_for_metric(m, X_va_s, metric_fn)
    return float(metric_fn(y_va, y_pred)), None


def _spread(values, lower_is_better):
    """Mean and spread of per-fold scores. One fold has no spread."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {'mean': float('nan'), 'n_folds': 0, 'std_over_folds': None,
                'spread_note': 'no fold scored'}
    out = {'mean': float(np.mean(vals)), 'n_folds': len(vals),
           'min': float(np.min(vals)), 'max': float(np.max(vals))}
    if len(vals) == 1:
        out['std_over_folds'] = None
        out['spread_note'] = 'one fold, no spread'
    else:
        out['std_over_folds'] = float(np.std(vals))
    return out


def _score_fold(sel, *, reps, labels, groups, augmentations, tr, va,
                task_type, metric_fn, lower_is_better):
    """Held-out scores of the 2 base representations and 2 hybrids per model
    that fold_standard_selection chose on this fold's training rows.

    Each hybrid is rebuilt on the training rows from its recipe with the
    per-representation scores THIS fold's search measured (blueprint.rep_scores
    of a recipe from this search; dirty-run bug 1), then scored on the held-out
    rows.
    """
    tr_reps = {r: np.asarray(X)[tr] for r, X in reps.items()}
    va_reps = {r: np.asarray(X)[va] for r, X in reps.items()}
    y_tr, y_va = labels[tr], labels[va]
    g_tr = groups[tr]
    out = {}
    for m, s in sel.items():
        if s is None:
            out[m] = {'error': ("the search on this fold's training rows "
                                "failed for this model")}
            continue
        base_rows = []
        for i, e in enumerate(s['base_detail']['selected'], 1):
            rep = e['rep']
            row = {'slot': f'base{i}', 'rep': rep,
                   'train_rows_score': e.get('score')}
            try:
                sc, why = _heldout_score(tr_reps[rep], va_reps[rep], y_tr,
                                         y_va, m, task_type, metric_fn,
                                         train_groups=g_tr)
                row['heldout_score'] = sc
                if why:
                    row['error'] = why
            except Exception as ex:
                row['heldout_score'] = None
                row['error'] = f'{type(ex).__name__}: {ex}'
            base_rows.append(row)

        hyb_rows = []
        for i, e in enumerate(s['hybrids'], 1):
            bp = e['blueprint']
            cond = e.get('aug_condition') or 'none'
            row = {'slot': f'hybrid{i}',
                   **{f: e.get(f) for f in HYBRID_FIELDS},
                   'aug_condition': cond,
                   'train_rows_score': e.get('score_mean')}
            augs = augmentations if cond != 'none' else None
            try:
                X_tr_h, X_va_h, fi = build_fold_hybrid(
                    tr_reps, va_reps, y_tr, bp,
                    augmentations_train=(None if augs is None else
                                         {a: np.asarray(X)[tr]
                                          for a, X in augs.items()}),
                    augmentations_val=(None if augs is None else
                                       {a: np.asarray(X)[va]
                                        for a, X in augs.items()}),
                    groups=g_tr,
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=get_model(m, task_type,
                                            n_samples=len(y_tr)),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                row['n_features'] = int(np.asarray(X_tr_h).shape[1])
                row['features_by_rep'] = {
                    r: int(len(info['selected_indices']))
                    for r, info in fi.items()
                    if isinstance(info, dict) and 'selected_indices' in info}
                sc, why = _heldout_score(X_tr_h, X_va_h, y_tr, y_va, m,
                                         task_type, metric_fn,
                                         train_groups=g_tr)
                row['heldout_score'] = sc
                if why:
                    row['error'] = why
            except Exception as ex:
                row['heldout_score'] = None
                row['error'] = f'{type(ex).__name__}: {ex}'
            hyb_rows.append(row)

        rec = {'base': base_rows, 'hybrids': hyb_rows,
               'base_shortfall': s['base_detail'].get('shortfall_reason'),
               'hybrids_skipped_identical_features':
                   s.get('hybrids_skipped_identical_features', [])}
        # The top hybrid against the mean of the representations that fed it
        # and against the best single representation, both read off this
        # fold's search (training rows, 0 fits).
        if s['hybrids']:
            try:
                rec['hybrid_deltas_train_rows'] = hybrid_deltas(
                    s['search'], m, hybrid=s['hybrids'][0])
            except Exception as ex:
                rec['hybrid_deltas_train_rows'] = {
                    'error': f'{type(ex).__name__}: {ex}'}
        out[m] = rec
    return out


def _slot_score(fold_models, m, kind, slot_i):
    """Held-out score of one slot ('base' or 'hybrids', 0-based) or None."""
    rec = (fold_models or {}).get(m) or {}
    rows = rec.get(kind) or []
    if len(rows) <= slot_i:
        return None
    v = rows[slot_i].get('heldout_score')
    return v if v is not None and np.isfinite(v) else None


def run(data, config, output_dir, dataset_name, **kwargs):
    """Learning curves: hybrid advantage vs sample size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C05: LEARNING CURVES")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']

    # The dataset's two decided models (DECISIONS.md Step 1). knn is a model of
    # the distance search only, and c05 runs no distance search.
    model_names = [m for m in decided_models(dataset_name, 'general')
                   if m != 'knn']
    # Outer folds per draw, from the one setting (handoff 2.4 step 14). A draw
    # holding fewer scaffold groups than this gets one fold per group, and 1 is
    # a single 80/20 held-out split.
    n_folds = int(config['n_splits'])
    n_repeats = kwargs.get('n_repeats', 3)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = np.asarray(data['groups'])
    augmentations = data.get('augmentations') or None
    n_total = len(labels)

    # 2026-10-05, owner delegated, 2026-10-05: a dataset with fewer than 10
    # scaffold/sequence groups in total (the groups array every draw below
    # samples from) is skipped. abagym_Ang2/EGFR/VEGF and the four ginkgo
    # datasets have 5. Drawing whole groups for a smaller sample leaves too few
    # groups to cross-validate, and splitting inside a group would break the
    # no-random-split rule. The {'skipped': True} marker is what
    # classify_result_payload (_shared.py) reads as an intended skip.
    _n_groups_total = int(len(np.unique(groups)))
    if _n_groups_total < 10:
        print(f"  [skip] {_n_groups_total} groups < 10: learning curve not run")
        return {
            'skipped': True,
            'reason': (f'{_n_groups_total} scaffold/sequence groups; whole-group '
                       'draws/halves cannot be cross-validated and splitting '
                       'inside groups would break the no-random-split rule'),
            'n_groups': _n_groups_total,
            'n_samples': int(n_total),
            'result_version': RESULT_VERSION,
        }

    # Representations are the run's pool (run_all loads
    # DATASET_REPS[ds]['general']). Anything else is named, not removed: the
    # pool handed to the hybrid builder is not filtered here (handoff 2.4
    # step 9).
    try:
        from experiments.complexity.complexity_helpers import DATASET_REPS
        _general = set((DATASET_REPS.get(dataset_name) or {}).get('general')
                       or [])
    except Exception:
        _general = set()
    _outside = sorted(r for r in reps if _general and r not in _general)
    if _outside:
        print(f"  WARNING: representations outside DATASET_REPS['general'] "
              f"in this run's pool: {_outside}")

    # Augmentation: c05 compares nothing with and without added features. The
    # per-fold search adds a condition only where Step 8 keeps one for the
    # general search (decided_aug_conditions), which is no dataset today, so
    # every hybrid here is 'none'.
    _aug_kept = [c for c in decided_search(dataset_name, 'general')
                 ['aug_conditions'] if c != 'none']
    if augmentations and not _aug_kept:
        print("  Augmentation: not searched. DECISIONS.md Step 8 keeps no "
              "augmentation on this dataset's general search.")

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
        # Under KIRBY_LOCAL_GRID: three, at 0.10, 0.50 and 1.0. With three
        # points the curve is not fitted (CURVE_FIT_PARAMETERS above); the
        # points themselves are still written.
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

    _grid = decided_search(dataset_name, 'general')
    print(f"  Models: {model_names}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Decided budgets (features): {_grid['budgets']}")
    print(f"  Draws per sample size: {n_repeats} (1 at full size)")
    print(f"  Outer folds per draw: {n_folds}")

    search_failures = []
    results = {
        'result_version': RESULT_VERSION,
        'selection_scope': 'fold_train_rows',
        'selection_rule': (
            "per sample size, per draw of whole scaffold groups, per outer "
            "fold of the draw: one search on the fold's training rows "
            "(decided grid, pruning off); per decided model the 2 best single "
            "representations and the 2 best hybrids by that search, each "
            "fitted on the training rows and scored on the held-out rows. The "
            "curve is hybrid1 against base1 (owner, 2026-10-05, (B))."),
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'models_decided': list(model_names),
        'n_outer_folds_requested': n_folds,
        'split_seed': SPLIT_SEED,
        'sample_sizes': sample_sizes,
        'decided_grid_general': _grid,
        'representations': sorted(reps.keys()),
        'representations_outside_general': _outside,
        'augmentation_conditions_searched': ['none'] + _aug_kept,
        'models': {},
        # One entry per draw: the molecules asked for (n_asked), what the draw
        # returned, its scaffold groups, its seed, and per outer fold the
        # scores of every slot.
        'draws': [],
    }
    for m in model_names:
        results['models'][m] = {
            # Aligned with sample_sizes. Mean over draws of the mean over
            # folds of the held-out score, over the (draw, fold) cells where
            # BOTH hybrid1 and base1 scored, so the two sides are always
            # averaged over the same cells.
            'hybrid': [], 'best_single': [],
            'delta': [],
            # The same, for the second-ranked of each.
            'hybrid2': [], 'best_single2': [],
            # The training-row scores of the same slots (the search's own
            # mean over its inner folds), same cells.
            'hybrid_train_rows': [], 'best_single_train_rows': [],
            # How many draws and (draw, fold) cells went into each point, and
            # how many cells had only one side.
            'n_draws_paired': [], 'n_cells_paired': [],
            'n_cells_hybrid_only': [], 'n_cells_single_only': [],
            # Spread over the paired cells' folds, per draw, per point.
            'delta_by_draw': [],
        }

    for n_sub in sample_sizes:
        print(f"\n  n={n_sub}:")
        # Pre-compute group-level pool so subsampling preserves scaffold
        # integrity (subsampling individual rows shatters scaffold groups
        # across train/val of the same fold).
        _unique_groups = np.unique(groups)
        _group_to_idx = {g: np.where(groups == g)[0] for g in _unique_groups}
        # FIX 3h: at the full-data point the group sampling below picks EVERY
        # group regardless of seed, so all n_repeats seeds draw the identical
        # dataset. The mean of n_repeats identical runs is just the value, so
        # collapse to a single seed here. <100% fractions still use n_repeats.
        is_full = n_sub >= n_total
        n_repeats_eff = 1 if is_full else n_repeats
        draws_this_n = []
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
            sub_reps = {k: v[idx] for k, v in reps.items()}
            sub_labels = labels[idx]
            sub_groups = groups[idx]
            sub_augs = ({k: np.asarray(v)[idx] for k, v in augmentations.items()}
                        if augmentations else None)
            n_unique_groups = int(len(np.unique(sub_groups)))
            # Each draw's searches are filed under the real dataset name with
            # the draw in condition_suffix (handoff 2.4 step 3). The full-size
            # draw is every row, so it carries none and shares the dataset's
            # standard per-fold searches.
            cond = None if is_full else f'c05_n{n_sub}_s{seed}'
            draw = {'n_asked': int(n_sub), 'seed': int(seed),
                    'n_rows_drawn': int(len(idx)),
                    'n_groups': n_unique_groups,
                    'condition_suffix': cond, 'folds': []}
            draws_this_n.append(draw)
            results['draws'].append(draw)

            if n_unique_groups < 2:
                # Group-aware folds need at least 2 scaffold groups.
                draw['error'] = ('skipped: the draw holds 1 scaffold group and '
                                 'group-aware folds need at least 2')
                search_failures.append({
                    'n_sub': int(n_sub), 'seed': int(seed),
                    'n_groups': n_unique_groups, 'n_rows': int(len(idx)),
                    'stage': 'draw', 'error': draw['error']})
                print(f"    seed={seed}: skipped, the draw holds "
                      f"{n_unique_groups} scaffold group")
                continue

            try:
                folds = nested_cv_outer_folds(
                    sub_reps, sub_labels, sub_groups, n_folds, SPLIT_SEED,
                    task_type)
            except Exception as e:
                draw['error'] = f'outer split failed: {type(e).__name__}: {e}'
                search_failures.append({
                    'n_sub': int(n_sub), 'seed': int(seed),
                    'n_groups': n_unique_groups, 'n_rows': int(len(idx)),
                    'stage': 'outer_split', 'error': draw['error']})
                print(f"    seed={seed}: {draw['error']}")
                continue
            n_outer = len(folds)
            draw['n_outer_folds'] = int(n_outer)
            # n_outer stays the layout's count (namespaces); the folds run may
            # be the first few of it (OUTER_FOLDS_RUN, 2026-10-07).
            folds = first_outer_folds(folds, 'c05')
            draw['n_outer_folds_run'] = len(folds)
            if n_outer != n_folds:
                print(f"    seed={seed}: {n_outer} outer folds "
                      f"({n_unique_groups} scaffold groups in the draw, "
                      f"{n_folds} asked)")

            for k, (tr, va) in enumerate(folds):
                ns = nested_cv_cache_namespace(k, n_outer, SPLIT_SEED)
                frec = {'fold': int(k), 'n_train': int(len(tr)),
                        'n_val': int(len(va)), 'cache_namespace': ns,
                        'selection_scope': 'fold_train_rows'}
                try:
                    sel = fold_standard_selection(
                        sub_reps, sub_labels, sub_groups, train_idx=tr,
                        fold_i=k, n_outer_folds=n_outer,
                        dataset_name=dataset_name, models=model_names,
                        augmentations=sub_augs, task_type=task_type,
                        metric=metric_name, seed=SPLIT_SEED,
                        cache_namespace=ns, condition_suffix=cond)
                except (UndecidedSearch, StaleResultDB):
                    raise
                except Exception as e:
                    frec['error'] = f'{type(e).__name__}: {e}'
                    draw['folds'].append(frec)
                    search_failures.append({
                        'n_sub': int(n_sub), 'seed': int(seed), 'fold': int(k),
                        'n_groups': n_unique_groups, 'n_rows': int(len(idx)),
                        'stage': 'search', 'error': frec['error']})
                    print(f"    seed={seed} fold {k}: search failed - {e}")
                    continue
                for m, s in sel.items():
                    if s is None:
                        search_failures.append({
                            'n_sub': int(n_sub), 'seed': int(seed),
                            'fold': int(k), 'model': m,
                            'n_groups': n_unique_groups,
                            'n_rows': int(len(idx)), 'stage': 'search',
                            'error': ("the search on this fold's training "
                                      "rows failed")})
                frec['models'] = _score_fold(
                    sel, reps=sub_reps, labels=sub_labels, groups=sub_groups,
                    augmentations=sub_augs, tr=tr, va=va,
                    task_type=task_type, metric_fn=metric_fn,
                    lower_is_better=lower_is_better)
                draw['folds'].append(frec)
                for m in model_names:
                    h = _slot_score(frec['models'], m, 'hybrids', 0)
                    b = _slot_score(frec['models'], m, 'base', 0)
                    hs = f'{h:.4f}' if h is not None else '--'
                    bs = f'{b:.4f}' if b is not None else '--'
                    print(f"    seed={seed} fold {k} {m}: hybrid1={hs} "
                          f"base1={bs} ({metric_name}, held out)")

        # Aggregate this sample size. Only cells where both hybrid1 and base1
        # scored enter either side (dirty run: at 56 humanness molecules one
        # side was averaged over 2 draws and the other over 3, and nothing
        # said so).
        for m in model_names:
            mr = results['models'][m]
            per_draw = []
            only_h = only_s = 0
            for draw in draws_this_n:
                cells = []
                for frec in draw['folds']:
                    fm = frec.get('models')
                    h = _slot_score(fm, m, 'hybrids', 0)
                    b = _slot_score(fm, m, 'base', 0)
                    if h is not None and b is not None:
                        rec_m = fm[m]
                        cells.append({
                            'fold': frec['fold'], 'h': h, 'b': b,
                            'h2': _slot_score(fm, m, 'hybrids', 1),
                            'b2': _slot_score(fm, m, 'base', 1),
                            'h_tr': rec_m['hybrids'][0].get('train_rows_score'),
                            'b_tr': rec_m['base'][0].get('train_rows_score')})
                    elif h is not None:
                        only_h += 1
                    elif b is not None:
                        only_s += 1
                if not cells:
                    continue
                d = [(c['b'] - c['h']) if lower_is_better else (c['h'] - c['b'])
                     for c in cells]
                per_draw.append({
                    'seed': draw['seed'], 'n_rows_drawn': draw['n_rows_drawn'],
                    'cells': cells,
                    'h': float(np.mean([c['h'] for c in cells])),
                    'b': float(np.mean([c['b'] for c in cells])),
                    'delta_over_folds': _spread(d, lower_is_better),
                    'folds_scored': [c['fold'] for c in cells]})

            def _mean_of(key, draws=per_draw):
                vals = []
                for pd in draws:
                    v = [c[key] for c in pd['cells']
                         if c[key] is not None and np.isfinite(c[key])]
                    if v:
                        vals.append(float(np.mean(v)))
                return float(np.mean(vals)) if vals else float('nan')

            h_pt = (float(np.mean([pd['h'] for pd in per_draw]))
                    if per_draw else float('nan'))
            s_pt = (float(np.mean([pd['b'] for pd in per_draw]))
                    if per_draw else float('nan'))
            mr['hybrid'].append(h_pt)
            mr['best_single'].append(s_pt)
            mr['delta'].append(
                (s_pt - h_pt if lower_is_better else h_pt - s_pt)
                if per_draw else float('nan'))
            mr['hybrid2'].append(_mean_of('h2'))
            mr['best_single2'].append(_mean_of('b2'))
            mr['hybrid_train_rows'].append(_mean_of('h_tr'))
            mr['best_single_train_rows'].append(_mean_of('b_tr'))
            mr['n_draws_paired'].append(len(per_draw))
            mr['n_cells_paired'].append(sum(len(pd['cells']) for pd in per_draw))
            mr['n_cells_hybrid_only'].append(int(only_h))
            mr['n_cells_single_only'].append(int(only_s))
            mr['delta_by_draw'].append([
                {'seed': pd['seed'], 'n_rows_drawn': pd['n_rows_drawn'],
                 'hybrid': pd['h'], 'best_single': pd['b'],
                 'delta_over_folds': pd['delta_over_folds']}
                for pd in per_draw])
            if only_h or only_s:
                print(f"    {m}: {only_h} fold(s) scored the hybrid only and "
                      f"{only_s} the single representation only; left out of "
                      f"both sides at this point")
            print(f"    {m}: hybrid={h_pt:.4f}, best_single={s_pt:.4f} "
                  f"({metric_name}, held out; {len(per_draw)} of "
                  f"{len(draws_this_n)} draws)")

        results.setdefault('n_rows_drawn', []).append(
            sorted(d['n_rows_drawn'] for d in draws_this_n))
        results.setdefault('n_draws_attempted', []).append(len(draws_this_n))

    # Fit learning curves per model, only where there are more points than the
    # fit has parameters.
    for model_name in model_names:
        model_result = results['models'][model_name]
        valid = [i for i, h in enumerate(model_result['hybrid'])
                 if not np.isnan(h)]
        if len(valid) > CURVE_FIT_PARAMETERS:
            n_arr = np.array([sample_sizes[i] for i in valid])
            h_arr = np.array([model_result['hybrid'][i] for i in valid])
            model_result['hybrid_fit'] = fit_learning_curve(
                n_arr, h_arr, lower_is_better=lower_is_better)
        else:
            model_result['hybrid_fit'] = {
                'not_fitted': (f'{len(valid)} scored sample sizes; the curve '
                               f'has {CURVE_FIT_PARAMETERS} parameters and '
                               f'needs at least {CURVE_FIT_PARAMETERS + 1}')}

    print("\n" + "-" * 70)
    print("LEARNING CURVE SUMMARY")
    print("-" * 70)
    print(f"  {metric_name}, held-out rows; n = molecules asked for; drawn = "
          f"molecules the group draws returned; delta + means the hybrid won")
    header = f"  {'n':>6} {'drawn':>11}"
    for m in model_names:
        header += f"  {m+'_h':>10} {m+'_s':>10} {'delta':>8}"
    print(header)
    print("  " + "-" * (19 + (10 + 10 + 8 + 4) * len(model_names)))
    _drawn_all = results.get('n_rows_drawn') or []
    for i, n_sub in enumerate(sample_sizes):
        _dr = _drawn_all[i] if i < len(_drawn_all) else []
        _dr_s = (f"{min(_dr)}-{max(_dr)}" if _dr and min(_dr) != max(_dr)
                 else (str(_dr[0]) if _dr else '?'))
        row = f"  {n_sub:>6} {_dr_s:>11}"
        for m in model_names:
            mr = results['models'][m]
            row += (f"  {mr['hybrid'][i]:>10.4f} {mr['best_single'][i]:>10.4f} "
                    f"{mr['delta'][i]:>+8.4f}")
        print(row)

    print(f"\n  LEARNING CURVE FITS (r2_max = asymptotic {metric_name}):")
    for model_name in model_names:
        fit = results['models'][model_name].get('hybrid_fit', {})
        if 'not_fitted' in fit:
            print(f"    {model_name}: not fitted ({fit['not_fitted']})")
        elif 'r2_max' in fit and 'error' not in fit:
            print(f"    {model_name}: {metric_name}_max={fit['r2_max']:.4f}, "
                  f"a={fit['a']:.3f}, b={fit['b']:.3f}, fit_R2={fit['fit_r2']:.3f}")
        elif 'r2_max' in fit:
            print(f"    {model_name}: {metric_name}_max~{fit['r2_max']:.4f} "
                  f"(fit failed: {fit.get('error', '?')})")

    # Every draw, fold or model whose search or split failed, with the size and
    # the group count, so a missing point on the curve says why.
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
