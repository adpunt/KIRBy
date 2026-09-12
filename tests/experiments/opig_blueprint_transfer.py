#!/usr/bin/env python3
"""OPIG C5: Cross-Dataset Blueprint Transfer — Pretrained Hybrid Evaluation.

OPIG Category C: Models and Evaluation Protocols.
Conclusion: Hybrid blueprints discovered on a large source dataset can
    transfer to a smaller target dataset, enabling publishable "pretrained"
    hybrid configurations.

Dependencies: None (standalone). Uses find_best_hybrid with cache=True.

Design:
    Source dataset: large dataset (e.g., logd ~7K molecules)
    Target dataset: small dataset (e.g., herg_ki ~1.4K molecules)
    Shared reps: intersection of both datasets' DATASET_REPS general lists.

    Conditions evaluated on target (per downstream model, per CV fold):
        A. base_best:        Best individual rep (no hybrid) — baseline. The
                             representation is re-picked inside every held-out
                             fold, from a search over that fold's training rows.
        B. native:           Hybrid from a recipe searched inside that fold's
                             training rows — upper bound.
        C. recipe_transfer:  Source blueprint recipe (allocation, importance, budget,
                             evaluator) applied to target data. Feature selection
                             recomputed on target training data.
        D. full_transfer:    Source's exact feature indices applied directly to
                             target data. No recomputation — the "pretrained" hybrid.

    Also evaluates hybrid vs base reps on source (sanity check), with both arms
    re-picked inside each source fold.

    Changed 2026-09-04 (split audit): A and B used to take the representation
    and the recipe from a search over every target row, then score them on the
    folds that search had already scored — GroupKFold is deterministic on the
    same groups, so those were the same folds. On the source side the hybrid
    picked its winner from a search of about 600 recipes while its baseline
    picked from about 10 representations, both on the folds that scored them.
    Cost of the fix: one recipe search per held-out fold per dataset, so
    n_folds searches on the source and n_folds on the target instead of one
    each. C and D are unchanged — their recipe and feature indices come from
    the source dataset, whose rows are not in any target fold.

    Key questions:
        1. full_transfer > base_best?   -> Blueprints have direct transfer value.
        2. recipe_transfer ~ native?    -> Recipe generalises; no search needed.
        3. full_transfer ~ native?      -> Source feature indices reproduce the
                                            native blueprint. The downstream
                                            model is still trained fresh on the
                                            target's training fold.

Usage:
    python opig_blueprint_transfer.py --source logd --target herg_ki --mode full
    python opig_blueprint_transfer.py --source adme_hlm --target caco2 --mode intermediary
"""
import sys
import os

# Load xgboost/lightgbm native OpenMP runtimes BEFORE rdkit (imported during
# rep generation) to avoid a duplicate-libomp segfault on macOS when these
# models and threaded sklearn estimators run in sequence. Optional deps —
# guard so environments without them are unaffected.
try:
    import xgboost as _xgboost  # noqa: F401
except Exception:
    pass
try:
    import lightgbm as _lightgbm  # noqa: F401
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    load_data, get_config, save_experiment_results, convert_numpy,
    get_model_factories, _cv_split, build_fold_hybrid,
    get_domain_metric,
    get_best_for_downstream,
    DOWNSTREAM_MODELS,
    FULL_BUDGETS, INTERNAL_EVALUATORS,
)
from kirby.find_best_hybrid import find_best_hybrid
from kirby.hybrid import create_hybrid, apply_feature_selection


def _predict_for_metric(m, X, task_type, metric_name):
    """Return predictions appropriate for the metric.

    Classification metrics that need scores (auc/auroc/auprc) get
    predict_proba (binary -> positive column, multi-class -> full matrix),
    falling back to decision_function for models without predict_proba
    (e.g., SVC without probability=True, RidgeClassifierCV).
    All other cases use predict.
    """
    if task_type == 'classification' and metric_name in ('auc', 'auroc', 'auprc'):
        if hasattr(m, 'predict_proba'):
            p = m.predict_proba(X)
            return p[:, 1] if p.shape[1] == 2 else p
        if hasattr(m, 'decision_function'):
            return m.decision_function(X)
    return m.predict(X)


def run(source_data, target_data, config, output_dir, source_name, target_name,
        pruning_mode='default_pruning'):
    """OPIG C5: Cross-dataset blueprint transfer."""
    print("\n" + "=" * 70)
    print("OPIG C5: CROSS-DATASET BLUEPRINT TRANSFER")
    print(f"  Source: {source_name} ({len(source_data['labels'])} samples)")
    print(f"  Target: {target_name} ({len(target_data['labels'])} samples)")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(target_data)
    cmp = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)

    source_reps = source_data['reps']
    source_labels = source_data['labels']
    source_groups = source_data['groups']

    target_reps = target_data['reps']
    target_labels = target_data['labels']
    target_groups = target_data['groups']

    n_folds = config['n_splits']
    task_type = target_data['task_type']

    # ── Task / domain consistency check ────────────────────────────────
    if source_data['task_type'] != target_data['task_type']:
        raise ValueError(
            f"Source task_type ({source_data['task_type']}) != "
            f"target ({target_data['task_type']})"
        )
    src_domain = source_data.get('domain', 'molecular')
    tgt_domain = target_data.get('domain', 'molecular')
    if src_domain != tgt_domain:
        print(f"  WARNING: domain mismatch (source={src_domain}, target={tgt_domain}). "
              f"Metrics and feature semantics may not be comparable.")

    # ── Validate rep compatibility ──────────────────────────────────────
    shared_rep_names = sorted(set(source_reps.keys()) & set(target_reps.keys()))
    dim_mismatches = []
    for rn in shared_rep_names:
        if source_reps[rn].shape[1] != target_reps[rn].shape[1]:
            dim_mismatches.append(rn)
            print(f"  WARNING: {rn} dim mismatch: source={source_reps[rn].shape[1]}, "
                  f"target={target_reps[rn].shape[1]} -- excluded")
    shared_rep_names = [r for r in shared_rep_names if r not in dim_mismatches]

    if len(shared_rep_names) < 2:
        print("  ERROR: Need at least 2 shared reps for meaningful comparison")
        return {'error': 'insufficient_shared_reps'}

    src = {k: source_reps[k] for k in shared_rep_names}
    tgt = {k: target_reps[k] for k in shared_rep_names}

    print(f"\n  Shared reps ({len(shared_rep_names)}): {', '.join(shared_rep_names)}")
    print(f"  Metric: {metric_name} ({'lower' if lower_is_better else 'higher'} is better)")

    from experiments._shared import resolve_pruning_kwargs
    _src_pkw = resolve_pruning_kwargs(pruning_mode, len(source_labels),
                                       dataset_label=source_name)
    _tgt_pkw = resolve_pruning_kwargs(pruning_mode, len(target_labels),
                                       dataset_label=target_name)

    # ── Step 1: Hybrid search on source (single strategy, cached) ─────
    # Source/target are arbitrary OPIG datasets — no shared precompute.
    # Hardcode (tier=general, aug=none) per empirical winning trend; no
    # augmentations in this experiment.
    # Honour the CLI/config budget list (default FULL_BUDGETS). Previously the
    # two searches hardcoded FULL_BUDGETS, silently ignoring the --budgets flag
    # and making local smoke runs intractable. config['budgets'] defaults to
    # FULL_BUDGETS in get_config, so server runs are unchanged.
    search_budgets = config.get('budgets', FULL_BUDGETS)

    # Test-only: under KIRBY_SMOKE_N use a single rf evaluator + cheap
    # lightgbm_gain importance for both searches. No-op (full INTERNAL_EVALUATORS
    # + default importance tier) when the env var is unset, so server runs are
    # unchanged.
    _is_smoke = bool(os.environ.get('KIRBY_SMOKE_N', '').strip())
    _eval_list = ['rf'] if _is_smoke else INTERNAL_EVALUATORS
    _smoke_imp = ({'importance_methods': ['lightgbm_gain']} if _is_smoke else {})

    print(f"\n  Step 1: Hybrid search on source ({source_name})")
    source_search = find_best_hybrid(
        base_reps=src, labels=source_labels, groups=source_groups,
        budgets=search_budgets,
        augmentation_strategy='none',
        internal_evaluators=_eval_list,
        metric=metric_name, task_type=task_type,
        verbose=False, cache=True,
        dataset_name=source_name, tier='general',
        **_smoke_imp,
        **_src_pkw,
    )

    print(f"\n  Step 2: Hybrid search on target ({target_name})")
    target_search = find_best_hybrid(
        base_reps=tgt, labels=target_labels, groups=target_groups,
        budgets=search_budgets,
        augmentation_strategy='none',
        internal_evaluators=_eval_list,
        metric=metric_name, task_type=task_type,
        verbose=False, cache=True,
        dataset_name=target_name, tier='general',
        **_smoke_imp,
        **_tgt_pkw,
    )

    # ── Step 3: Per-model evaluation ────────────────────────────────────
    print(f"\n  Step 3: Per-model evaluation ({n_folds}-fold CV)")

    model_names = config.get('models', DOWNSTREAM_MODELS)

    def _search_models(sr):
        if hasattr(sr, 'best_by_model_budget') and sr.best_by_model_budget:
            return set(sr.best_by_model_budget.keys())
        return set(getattr(sr, 'best_by_model', {}).keys())

    available = _search_models(source_search) & _search_models(target_search)
    model_names = [m for m in model_names if m in available]
    print(f"  Models available from search: {model_names}")

    model_factories_tgt = get_model_factories(
        model_names, task_type=task_type, n_samples=len(target_labels))
    model_factories_src = get_model_factories(
        model_names, task_type=task_type, n_samples=len(source_labels))

    target_folds = list(_cv_split(
        target_groups, n_folds,
        next(iter(tgt.values())), target_labels))
    source_folds = list(_cv_split(
        source_groups, n_folds,
        next(iter(src.values())), source_labels))

    # ── Step 3 prelude: one recipe search per held-out fold ─────────────
    # The native arm, the base_best baseline, and the source-side check used to
    # take their recipe and their single representation from the whole-dataset
    # searches above, then score them on the folds those searches had already
    # scored. GroupKFold is deterministic on the same groups, so the evaluation
    # folds were the search folds. Now each held-out fold gets its own search
    # over its training rows only, and the recipe and the best single
    # representation for that fold come from it. Changed 2026-09-04.
    #
    # This is n_folds searches per dataset instead of one, on both source and
    # target. recipe_transfer and full_transfer are untouched: their recipe and
    # feature indices come from the SOURCE dataset, which shares no rows with
    # the target folds.
    def _fold_searches(reps_dict, labels_arr, groups_arr, folds, ds_name):
        """Search inside each fold's training rows. One entry per fold, None
        where the search raised.

        Pruning is resolved from the fold's own row count, not the dataset's,
        so a fold is pruned as hard as its size warrants.
        """
        out = []
        for fold_i, (train_idx, _val_idx) in enumerate(folds):
            tr_reps = {k: v[train_idx] for k, v in reps_dict.items()}
            tr_groups = (groups_arr[train_idx]
                         if groups_arr is not None else None)
            pkw = resolve_pruning_kwargs(
                pruning_mode, len(train_idx),
                dataset_label=f'{ds_name}/heldout_fold{fold_i}')
            try:
                out.append(find_best_hybrid(
                    base_reps=tr_reps, labels=labels_arr[train_idx],
                    groups=tr_groups,
                    budgets=search_budgets,
                    augmentation_strategy='none',
                    internal_evaluators=_eval_list,
                    metric=metric_name, task_type=task_type,
                    verbose=False, cache=True,
                    dataset_name=ds_name, tier='general',
                    condition_suffix=f'heldout_fold{fold_i}',
                    **_smoke_imp,
                    **pkw,
                ))
            except Exception as e:
                print(f"      fold {fold_i} search on {ds_name} failed: {e}")
                out.append(None)
        return out

    def _fold_best_rep(fold_search):
        """Best single representation for one model, from a fold's own search.

        Returns a {model: rep_name} map. Falls back to nothing when the fold's
        search is missing; the caller then holds that fold out of the arm.
        """
        picks = {}
        for rn, mscores in (getattr(fold_search, 'baselines', {}) or {}).items():
            if rn.startswith('_') or rn not in shared_rep_names:
                continue
            for mn, sc in (mscores or {}).items():
                if sc is None or np.isnan(sc):
                    continue
                cur = picks.get(mn)
                if cur is None or cmp(sc, cur[1]):
                    picks[mn] = (rn, sc)
        return {mn: rn for mn, (rn, _sc) in picks.items()}

    print(f"\n  Per-fold searches on target ({target_name}), "
          f"{len(target_folds)} folds")
    target_fold_searches = _fold_searches(
        tgt, target_labels, target_groups, target_folds, target_name)
    target_fold_reps = [_fold_best_rep(fs) if fs is not None else {}
                        for fs in target_fold_searches]

    print(f"  Per-fold searches on source ({source_name}), "
          f"{len(source_folds)} folds")
    source_fold_searches = _fold_searches(
        src, source_labels, source_groups, source_folds, source_name)
    source_fold_reps = [_fold_best_rep(fs) if fs is not None else {}
                        for fs in source_fold_searches]

    per_model = {}

    for model_name in model_names:
        if model_name not in model_factories_tgt or model_name not in model_factories_src:
            continue

        source_best = get_best_for_downstream(source_search, model_name)
        target_best = get_best_for_downstream(target_search, model_name)

        if source_best is None or target_best is None:
            print(f"\n    {model_name}: skipped (missing search config)")
            continue

        source_bp = source_best['blueprint']
        target_bp = target_best['blueprint']

        print(f"\n    {model_name}:")
        print(f"      Source BP: budget={source_bp.budget}, "
              f"alloc={source_bp.allocation_method}, "
              f"imp={source_bp.importance_method}, "
              f"eval={source_bp.internal_evaluator}")
        print(f"      Target BP: budget={target_bp.budget}, "
              f"alloc={target_bp.allocation_method}, "
              f"imp={target_bp.importance_method}, "
              f"eval={target_bp.internal_evaluator}")

        # Build source hybrid on FULL source data -> feature_info for full_transfer
        try:
            _, fi_source = create_hybrid(
                src, source_labels,
                allocation_method=source_bp.allocation_method,
                budget=source_bp.budget,
                importance_method=source_bp.importance_method,
                internal_evaluator=source_bp.internal_evaluator,
                # The accept/reject holdout inside create_hybrid is a GroupKFold
                # fold when groups are given and a RandomState(42) 20% random
                # split when they are not. These molecules are scaffold-grouped,
                # so without this the keep/drop decisions were judged on rows
                # sharing scaffolds with their own training rows.
                groups=source_groups,
                # What each representation scored on its own in the source
                # search, over these same rows. Reused so the budget split is
                # the one the search settled on, instead of every representation
                # being scored again here.
                precomputed_alloc_scores=getattr(source_bp, 'rep_scores', None),
                # Explicit: this experiment has no augmentations. Default of
                # 'uncapped' would silently mismatch the search recipe if
                # any future caller introduces augs without updating here.
                augmentations=None,
                augmentation_strategy='none',
            )
        except Exception as e:
            print(f"      Source hybrid build failed: {e}")
            continue

        # Report which features the source blueprint selected
        print(f"      Source blueprint feature selection:")
        for rn in sorted(fi_source.keys()):
            info = fi_source[rn]
            if isinstance(info, dict) and 'selected_indices' in info:
                n_sel = len(info['selected_indices'])
                total = src[rn].shape[1]
                print(f"        {rn}: {n_sel}/{total}")

        # The best single representation and the native recipe are re-picked
        # inside each fold below, from that fold's own training rows. The
        # whole-dataset picks printed here are context only, and no reported
        # score uses them.
        def _whole_data_best_rep(search_obj):
            best_rn, best_sc = None, None
            for rn, mscores in (getattr(search_obj, 'baselines', {}) or {}).items():
                if rn.startswith('_') or rn not in shared_rep_names:
                    continue
                sc_ = (mscores or {}).get(model_name)
                if sc_ is not None and not np.isnan(sc_):
                    if best_sc is None or cmp(sc_, best_sc):
                        best_sc, best_rn = sc_, rn
            return best_rn or shared_rep_names[0]

        best_tgt_rep = _whole_data_best_rep(target_search)
        best_src_rep = _whole_data_best_rep(source_search)

        print(f"      Best base rep on all target rows (context): {best_tgt_rep}")
        print(f"      Best base rep on all source rows (context): {best_src_rep}")

        # ── 3a: Evaluate on TARGET ──────────────────────────────────────
        tgt_fold_scores = {
            'base_best': [],
            'native': [], 'recipe_transfer': [], 'full_transfer': [],
        }
        tgt_fold_base_reps = []
        tgt_fold_native_bp = []

        print(f"      Target evaluation ({target_name}):")
        for fold_i, (train_idx, val_idx) in enumerate(target_folds):
            tr = {k: v[train_idx] for k, v in tgt.items()}
            va = {k: v[val_idx] for k, v in tgt.items()}
            y_tr = target_labels[train_idx]
            y_va = target_labels[val_idx]
            mk = model_factories_tgt[model_name]

            # This fold's own search, over its training rows only.
            fold_search = target_fold_searches[fold_i]
            fold_rep = target_fold_reps[fold_i].get(model_name)
            fold_best = (get_best_for_downstream(fold_search, model_name)
                         if fold_search is not None else None)
            fold_bp = fold_best['blueprint'] if fold_best else None
            tgt_fold_base_reps.append(fold_rep)
            tgt_fold_native_bp.append(
                None if fold_bp is None else {
                    'budget': fold_bp.budget,
                    'allocation': fold_bp.allocation_method,
                    'importance': fold_bp.importance_method,
                    'evaluator': fold_bp.internal_evaluator,
                })

            # A: Best single rep, picked on this fold's training rows
            if fold_rep is None:
                tgt_fold_scores['base_best'].append(float('nan'))
            else:
                sc = StandardScaler()
                X_tr = sc.fit_transform(tr[fold_rep])
                X_va = sc.transform(va[fold_rep])
                m = mk()
                m.fit(X_tr, y_tr)
                tgt_fold_scores['base_best'].append(float(metric_fn(
                    y_va, _predict_for_metric(m, X_va, task_type, metric_name))))

            # B: Native hybrid, recipe from this fold's training rows
            if fold_bp is None:
                print(f"        Fold {fold_i} native: no fold recipe")
                tgt_fold_scores['native'].append(float('nan'))
            else:
                try:
                    X_tr_h, X_va_h, _ = build_fold_hybrid(tr, va, y_tr, fold_bp,
                        groups=(target_groups[train_idx] if target_groups is not None else None),
                        model=mk(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better)
                    sc = StandardScaler()
                    X_tr_h = sc.fit_transform(X_tr_h)
                    X_va_h = sc.transform(X_va_h)
                    m = mk()
                    m.fit(X_tr_h, y_tr)
                    tgt_fold_scores['native'].append(float(metric_fn(
                        y_va, _predict_for_metric(m, X_va_h, task_type, metric_name))))
                except Exception as e:
                    print(f"        Fold {fold_i} native failed: {e}")
                    tgt_fold_scores['native'].append(float('nan'))

            # C: Recipe transfer (source recipe, target data)
            try:
                X_tr_h, X_va_h, _ = build_fold_hybrid(tr, va, y_tr, source_bp,
                    groups=(target_groups[train_idx] if target_groups is not None else None),
                    model=mk(),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better)
                sc = StandardScaler()
                X_tr_h = sc.fit_transform(X_tr_h)
                X_va_h = sc.transform(X_va_h)
                m = mk()
                m.fit(X_tr_h, y_tr)
                tgt_fold_scores['recipe_transfer'].append(float(metric_fn(
                    y_va, _predict_for_metric(m, X_va_h, task_type, metric_name))))
            except Exception as e:
                print(f"        Fold {fold_i} recipe failed: {e}")
                tgt_fold_scores['recipe_transfer'].append(float('nan'))

            # D: Full transfer (source feature indices on target)
            try:
                X_tr_f = apply_feature_selection(tr, fi_source)
                X_va_f = apply_feature_selection(va, fi_source)
                sc = StandardScaler()
                X_tr_f = sc.fit_transform(X_tr_f)
                X_va_f = sc.transform(X_va_f)
                m = mk()
                m.fit(X_tr_f, y_tr)
                tgt_fold_scores['full_transfer'].append(float(metric_fn(
                    y_va, _predict_for_metric(m, X_va_f, task_type, metric_name))))
            except Exception as e:
                print(f"        Fold {fold_i} full failed: {e}")
                tgt_fold_scores['full_transfer'].append(float('nan'))

            print(f"        Fold {fold_i}: base={tgt_fold_scores['base_best'][-1]:.4f} "
                  f"({fold_rep}), native={tgt_fold_scores['native'][-1]:.4f}, "
                  f"recipe={tgt_fold_scores['recipe_transfer'][-1]:.4f}, "
                  f"full={tgt_fold_scores['full_transfer'][-1]:.4f}")

        # ── 3b: Evaluate on SOURCE (context) ────────────────────────────
        # Both arms are re-picked inside each source fold. Previously the
        # hybrid recipe came from a search over every source row and the
        # baseline from that same search's ten representation scores, so the
        # hybrid drew its winner from a much larger set than the baseline did
        # and the gap between them leaned toward the hybrid.
        src_fold_scores = {'base_best': [], 'hybrid': []}
        src_fold_base_reps = []
        src_fold_hybrid_bp = []

        print(f"      Source evaluation ({source_name}):")
        for fold_i, (train_idx, val_idx) in enumerate(source_folds):
            tr = {k: v[train_idx] for k, v in src.items()}
            va = {k: v[val_idx] for k, v in src.items()}
            y_tr = source_labels[train_idx]
            y_va = source_labels[val_idx]
            mk = model_factories_src[model_name]

            fold_search = source_fold_searches[fold_i]
            fold_rep = source_fold_reps[fold_i].get(model_name)
            fold_best = (get_best_for_downstream(fold_search, model_name)
                         if fold_search is not None else None)
            fold_bp = fold_best['blueprint'] if fold_best else None
            src_fold_base_reps.append(fold_rep)
            src_fold_hybrid_bp.append(
                None if fold_bp is None else {
                    'budget': fold_bp.budget,
                    'allocation': fold_bp.allocation_method,
                    'importance': fold_bp.importance_method,
                    'evaluator': fold_bp.internal_evaluator,
                })

            # Best single rep, picked on this fold's training rows
            if fold_rep is None:
                src_fold_scores['base_best'].append(float('nan'))
            else:
                sc = StandardScaler()
                X_tr = sc.fit_transform(tr[fold_rep])
                X_va = sc.transform(va[fold_rep])
                m = mk()
                m.fit(X_tr, y_tr)
                src_fold_scores['base_best'].append(float(metric_fn(
                    y_va, _predict_for_metric(m, X_va, task_type, metric_name))))

            # Hybrid, recipe from this fold's training rows
            if fold_bp is None:
                print(f"        Fold {fold_i} source hybrid: no fold recipe")
                src_fold_scores['hybrid'].append(float('nan'))
            else:
                try:
                    X_tr_h, X_va_h, _ = build_fold_hybrid(tr, va, y_tr, fold_bp,
                        groups=(source_groups[train_idx] if source_groups is not None else None),
                        model=mk(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better)
                    sc = StandardScaler()
                    X_tr_h = sc.fit_transform(X_tr_h)
                    X_va_h = sc.transform(X_va_h)
                    m = mk()
                    m.fit(X_tr_h, y_tr)
                    src_fold_scores['hybrid'].append(float(metric_fn(
                        y_va, _predict_for_metric(m, X_va_h, task_type, metric_name))))
                except Exception as e:
                    print(f"        Fold {fold_i} source hybrid failed: {e}")
                    src_fold_scores['hybrid'].append(float('nan'))

            print(f"        Fold {fold_i}: base={src_fold_scores['base_best'][-1]:.4f} "
                  f"({fold_rep}), hybrid={src_fold_scores['hybrid'][-1]:.4f}")

        # ── Aggregate for this model ────────────────────────────────────
        def _agg(scores):
            return {'mean': float(np.nanmean(scores)),
                    'std': float(np.nanstd(scores)),
                    'per_fold': scores}

        model_res = {
            'target': {},
            'source': {
                'base_best': _agg(src_fold_scores['base_best']),
                'hybrid': _agg(src_fold_scores['hybrid']),
            },
            # The recipe transferred to the target. Chosen over every source
            # row, which is what recipe_transfer and full_transfer are meant to
            # test; the target folds share no rows with it.
            'source_blueprint': {
                'budget': source_bp.budget,
                'allocation': source_bp.allocation_method,
                'importance': source_bp.importance_method,
                'evaluator': source_bp.internal_evaluator,
            },
            # Context only. Chosen over every target row, so it is not what the
            # native arm was scored with — that is per_fold_native_blueprint.
            'target_blueprint_whole_data': {
                'budget': target_bp.budget,
                'allocation': target_bp.allocation_method,
                'importance': target_bp.importance_method,
                'evaluator': target_bp.internal_evaluator,
            },
        }
        for cond in ['base_best', 'native', 'recipe_transfer', 'full_transfer']:
            model_res['target'][cond] = _agg(tgt_fold_scores[cond])
        # What each held-out fold actually used, picked on its training rows.
        model_res['target']['per_fold_base_rep'] = tgt_fold_base_reps
        model_res['target']['per_fold_native_blueprint'] = tgt_fold_native_bp
        model_res['target']['base_rep_whole_data'] = best_tgt_rep
        model_res['source']['per_fold_base_rep'] = src_fold_base_reps
        model_res['source']['per_fold_hybrid_blueprint'] = src_fold_hybrid_bp
        model_res['source']['base_rep_whole_data'] = best_src_rep

        # Deltas vs target base, subtracted fold by fold. A fold whose arm
        # raised holds a nan and the two arms do not lose the same folds, so
        # differencing two nanmeans could take a 4-fold mean off a 5-fold one.
        # Only fold positions where BOTH arms are finite enter the difference.
        # delta_n_folds is how many folds are behind the number and
        # delta_std their spread — the win threshold below is 0.001 RMSE, far
        # under that spread, so both have to be readable next to the delta.
        base_folds = model_res['target']['base_best']['per_fold']
        for cond in ['native', 'recipe_transfer', 'full_transfer']:
            cond_folds = model_res['target'][cond]['per_fold']
            diffs = [((b - c) if lower_is_better else (c - b))
                     for b, c in zip(base_folds, cond_folds)
                     if np.isfinite(b) and np.isfinite(c)]
            model_res['target'][cond]['delta_vs_base'] = (
                float(np.mean(diffs)) if diffs else float('nan'))
            model_res['target'][cond]['delta_n_folds'] = len(diffs)
            model_res['target'][cond]['delta_std'] = (
                float(np.std(diffs)) if diffs else float('nan'))

        per_model[model_name] = model_res

    # ── Summary table ───────────────────────────────────────────────────
    if not per_model:
        print("\n  No models produced results.")
        return {'error': 'no_results'}

    arrow = '\u2193' if lower_is_better else '\u2191'
    print(f"\n  {'='*90}")
    print(f"  CROSS-DATASET BLUEPRINT TRANSFER SUMMARY")
    print(f"  Source: {source_name} -> Target: {target_name}")
    print(f"  Metric: {metric_name} ({arrow} better)")
    print(f"  {'='*90}")

    # Source context
    print(f"\n  Source ({source_name}):")
    print(f"  {'Model':<12} {'Base':>10} {'Hybrid':>10} {'Delta':>10}")
    print(f"  {'-'*42}")
    for mn, res in per_model.items():
        sb = res['source']['base_best']['mean']
        sh = res['source']['hybrid']['mean']
        d = (sb - sh) if lower_is_better else (sh - sb)
        print(f"  {mn:<12} {sb:>10.4f} {sh:>10.4f} {d:>+10.4f}")

    # Target
    # Each delta is the mean of the per-fold differences; folds is how many
    # folds had a finite score on both arms, and sd their spread.
    print(f"\n  Target ({target_name}):")
    print(f"  {'Model':<12} {'Base':>10} {'Native':>10} {'Recipe':>10} "
          f"{'Full':>10} {'d(Rec)':>8} {'folds':>6} {'sd':>8} "
          f"{'d(Full)':>8} {'folds':>6} {'sd':>8}")
    print(f"  {'-'*106}")
    for mn, res in per_model.items():
        t = res['target']
        print(f"  {mn:<12} "
              f"{t['base_best']['mean']:>10.4f} "
              f"{t['native']['mean']:>10.4f} "
              f"{t['recipe_transfer']['mean']:>10.4f} "
              f"{t['full_transfer']['mean']:>10.4f} "
              f"{t['recipe_transfer']['delta_vs_base']:>+8.4f} "
              f"{t['recipe_transfer']['delta_n_folds']:>6d} "
              f"{t['recipe_transfer']['delta_std']:>8.4f} "
              f"{t['full_transfer']['delta_vs_base']:>+8.4f} "
              f"{t['full_transfer']['delta_n_folds']:>6d} "
              f"{t['full_transfer']['delta_std']:>8.4f}")

    # Aggregate interpretation
    d_recipe = [r['target']['recipe_transfer']['delta_vs_base']
                for r in per_model.values()]
    d_full = [r['target']['full_transfer']['delta_vs_base']
              for r in per_model.values()]
    # A model whose two arms shared no finite fold has a nan delta; it drops out
    # of the mean over models instead of turning the whole mean into nan.
    mean_d_recipe = float(np.nanmean(d_recipe)) if np.any(np.isfinite(d_recipe)) else float('nan')
    mean_d_full = float(np.nanmean(d_full)) if np.any(np.isfinite(d_full)) else float('nan')
    n_recipe_win = sum(1 for d in d_recipe if d > 0.001)
    n_full_win = sum(1 for d in d_full if d > 0.001)
    n_models = len(per_model)

    print(f"\n  Mean delta recipe: {mean_d_recipe:+.4f}  "
          f"(wins {n_recipe_win}/{n_models} models)")
    print(f"  Mean delta full:   {mean_d_full:+.4f}  "
          f"(wins {n_full_win}/{n_models} models)")

    if n_full_win > n_models // 2:
        print(f"\n  RESULT: Blueprint FULLY TRANSFERS. "
              f"Exact feature indices from {source_name} improve "
              f"performance on {target_name}.")
    elif n_recipe_win > n_models // 2:
        print(f"\n  RESULT: Blueprint RECIPE transfers. "
              f"Rerun feature selection with the source recipe on target data.")
    else:
        print(f"\n  RESULT: Blueprint does NOT transfer reliably. "
              f"Hybrid is dataset-specific for this pair.")

    # ── Compile results ─────────────────────────────────────────────────
    results = {
        'source_dataset': source_name,
        'target_dataset': target_name,
        'source_n_samples': len(source_labels),
        'target_n_samples': len(target_labels),
        'shared_reps': shared_rep_names,
        'n_shared_reps': len(shared_rep_names),
        'n_folds': n_folds,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'per_model': convert_numpy(per_model),
        'aggregate': {
            'mean_delta_recipe': mean_d_recipe,
            'mean_delta_full': mean_d_full,
            'n_recipe_wins': n_recipe_win,
            'n_full_wins': n_full_win,
            'n_models': n_models,
        },
    }

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='OPIG C5: Cross-Dataset Blueprint Transfer')
    parser.add_argument('--source', type=str, default='logd',
                        help='Source (large) dataset name (default: logd)')
    parser.add_argument('--target', type=str, default='herg_ki',
                        help='Target (small) dataset name (default: herg_ki)')
    parser.add_argument('--mode', choices=['intermediary', 'full'],
                        default='full',
                        help='full (5-fold CV, default) or intermediary (single 80/20 split)')
    parser.add_argument('--reps', type=str, default=None,
                        help='Comma-separated rep list (default: intersection of DATASET_REPS)')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated downstream model list')
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated budget list')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Max folds to evaluate (stops early)')
    parser.add_argument('--pruning-mode', type=str, default='default_pruning',
                        help="Adaptive pruning mode (default 'default_pruning'). "
                             "'aggressive_pruning' = F16-validated multi-stage pruning "
                             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")
    parser = add_distance_metric_arg(parser)
    args = parser.parse_args()

    apply_distance_metric_arg(args)
    # Compute shared reps from DATASET_REPS intersection
    if args.reps:
        shared_rep_names = [r.strip() for r in args.reps.split(',')]
    else:
        try:
            from experiments.complexity.complexity_helpers import DATASET_REPS
            src_gen = DATASET_REPS.get(args.source, {}).get('general', [])
            tgt_gen = DATASET_REPS.get(args.target, {}).get('general', [])
            if src_gen and tgt_gen:
                shared_rep_names = sorted(set(src_gen) & set(tgt_gen))
                print(f"  Auto-intersected reps: {len(shared_rep_names)} shared "
                      f"(source={len(src_gen)}, target={len(tgt_gen)})")
            else:
                shared_rep_names = None
                print(f"  WARNING: DATASET_REPS not found for both datasets, "
                      f"using default preset")
        except ImportError:
            shared_rep_names = None

    # Load both datasets with the shared rep set (no augmentation reps —
    # this experiment evaluates blueprint transfer on base reps only).
    source_data = load_data(args.source, rep_names=shared_rep_names,
                            augment=False)
    target_data = load_data(args.target, rep_names=shared_rep_names,
                            augment=False)

    config = get_config(args)

    results = run(source_data, target_data, config, args.output,
                  args.source, args.target,
                  pruning_mode=args.pruning_mode)
    save_experiment_results(results, args.output,
                           'opig_blueprint_transfer',
                           f'{args.source}_to_{args.target}')
