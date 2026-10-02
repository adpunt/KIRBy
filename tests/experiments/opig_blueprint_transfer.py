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
    DOWNSTREAM_MODELS,
    FULL_BUDGETS, INTERNAL_EVALUATORS,
    select_model_rep_pairs, hybrid_deltas, top_hybrids,
    resolve_comparison_models,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS,
    RULE_N_GENERAL, RULE_N_DISTANCE, RULE_N_BASE,
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

    def _speed_kw(ds, distance=False):
        from experiments._shared import _apply_dataset_search_speed
        kw = _apply_dataset_search_speed(
            {'tier': 'general_distance' if distance else 'general'}, ds)
        kw.pop('tier', None)
        return kw

    # The owner's set (2026-10-02) also needs the DISTANCE lineage, the one the
    # precompute stores under tier 'general_distance' with the distance
    # evaluators, budgets and models. Every slot that could not be filled is
    # named in rule_missing and saved.
    rule_missing = []
    _dist_budgets = [b for b in search_budgets if b in DISTANCE_BUDGETS]
    if not _dist_budgets:
        _dist_budgets = list(DISTANCE_BUDGETS)
        rule_missing.append(
            f'the distance searches stop at {max(DISTANCE_BUDGETS)} features, '
            f'so they were run over {_dist_budgets} rather than the '
            f'{search_budgets} this run asked for')

    def _distance_kw(ds):
        kw = _speed_kw(ds, distance=True)
        kw.setdefault('models', list(DISTANCE_MODELS))
        kw['internal_evaluators'] = list(DISTANCE_EVALUATORS)
        return kw

    def _distance_search(reps_dict, labels_arr, groups_arr, ds_name, pkw,
                         suffix=None):
        try:
            return find_best_hybrid(
                base_reps=reps_dict, labels=labels_arr, groups=groups_arr,
                budgets=_dist_budgets,
                augmentation_strategy='none',
                metric=metric_name, task_type=task_type,
                verbose=False, cache=True,
                dataset_name=ds_name, tier='general_distance',
                **({'condition_suffix': suffix} if suffix else {}),
                **_smoke_imp,
                **pkw,
                **_distance_kw(ds_name),
            )
        except Exception as e:
            rule_missing.append(
                f'distance search on {ds_name}'
                f'{"/" + suffix if suffix else ""} raised '
                f'{type(e).__name__}: {e}')
            print(f"      distance search on {ds_name} failed: {e}")
            return None

    def _evals_for(ds):
        # An explicit evaluator list beats speed= inside find_best_hybrid, so
        # a fast-speed dataset gets the fast list by name.
        if _is_smoke:
            return _eval_list
        from experiments.dataset_policy import dataset_search_speed
        if dataset_search_speed(ds) == 'fast':
            from kirby.find_best_hybrid import FAST_INTERNAL_EVALUATORS
            return list(FAST_INTERNAL_EVALUATORS)
        return _eval_list

    print(f"\n  Step 1: Hybrid search on source ({source_name})")
    source_search = find_best_hybrid(
        base_reps=src, labels=source_labels, groups=source_groups,
        budgets=search_budgets,
        augmentation_strategy='none',
        internal_evaluators=_evals_for(source_name),
        metric=metric_name, task_type=task_type,
        verbose=False, cache=True,
        dataset_name=source_name, tier='general',
        **_smoke_imp,
        **_src_pkw,
        # The dataset's speed and model list, as the precompute stored them:
        # fast grid and the five models on the fast-speed datasets, + gp under
        # 1,000 molecules. This call went straight to find_best_hybrid and
        # missed the stored search on both (model-list audit, 2026-09-29).
        **_speed_kw(source_name),
    )

    print(f"\n  Step 2: Hybrid search on target ({target_name})")
    target_search = find_best_hybrid(
        base_reps=tgt, labels=target_labels, groups=target_groups,
        budgets=search_budgets,
        augmentation_strategy='none',
        internal_evaluators=_evals_for(target_name),
        metric=metric_name, task_type=task_type,
        verbose=False, cache=True,
        dataset_name=target_name, tier='general',
        **_smoke_imp,
        **_tgt_pkw,
        # The dataset's speed and model list, as the precompute stored them:
        # fast grid and the five models on the fast-speed datasets, + gp under
        # 1,000 molecules. This call went straight to find_best_hybrid and
        # missed the stored search on both (model-list audit, 2026-09-29).
        **_speed_kw(target_name),
    )

    print(f"\n  Step 2b: Distance-lineage searches on both datasets")
    source_distance_search = _distance_search(
        src, source_labels, source_groups, source_name, _src_pkw)
    target_distance_search = _distance_search(
        tgt, target_labels, target_groups, target_name, _tgt_pkw)

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
    def _fold_searches(reps_dict, labels_arr, groups_arr, folds, ds_name,
                       distance=False):
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
            if distance:
                # The distance counterpart of this fold's search (owner's set,
                # 2026-10-02): same training rows, the distance lineage.
                out.append(_distance_search(
                    tr_reps, labels_arr[train_idx], tr_groups, ds_name, pkw,
                    suffix=f'heldout_fold{fold_i}'))
                continue
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
        """The best single representations for one model, from a fold's own
        search, best first.

        Returns a {model: [rep_name, ...]} map, as many names as were scored.
        The owner's set (2026-10-02) takes the top RULE_N_BASE of them. Falls back to nothing when the fold's
        search is missing; the caller then holds that fold out of the arm.
        """
        # ⛔ TIE-BREAK ADDED 2026-09-18, owner's decision. `cmp` is strictly
        # better, so on an exact tie this kept whichever representation
        # `baselines` iterated first — the order they were built in, which is
        # arbitrary and recorded nowhere. Baseline scores are rounded to 4
        # decimals before storage (find_best_hybrid.py:5525), so exact ties
        # happen, and this pick names the representation a whole transfer arm
        # is then measured against.
        #
        # The score does not change. Only which of two equal names is picked,
        # and it is now the same name every run: the first by representation
        # name among those on the best score.
        picks = {}
        for rn, mscores in (getattr(fold_search, 'baselines', {}) or {}).items():
            if rn.startswith('_') or rn not in shared_rep_names:
                continue
            for mn, sc in (mscores or {}).items():
                if sc is None or np.isnan(sc):
                    continue
                picks.setdefault(mn, []).append((rn, float(sc)))
        out = {}
        for mn, rows in picks.items():
            rows.sort(key=lambda t: ((t[1] if lower_is_better else -t[1]),
                                     t[0]))
            out[mn] = [rn for rn, _sc in rows]
        return out

    print(f"\n  Per-fold searches on target ({target_name}), "
          f"{len(target_folds)} folds")
    target_fold_searches = _fold_searches(
        tgt, target_labels, target_groups, target_folds, target_name)
    target_fold_reps = [_fold_best_rep(fs) if fs is not None else {}
                        for fs in target_fold_searches]
    target_fold_distance_searches = _fold_searches(
        tgt, target_labels, target_groups, target_folds, target_name,
        distance=True)

    print(f"  Per-fold searches on source ({source_name}), "
          f"{len(source_folds)} folds")
    source_fold_searches = _fold_searches(
        src, source_labels, source_groups, source_folds, source_name)
    source_fold_reps = [_fold_best_rep(fs) if fs is not None else {}
                        for fs in source_fold_searches]
    source_fold_distance_searches = _fold_searches(
        src, source_labels, source_groups, source_folds, source_name,
        distance=True)

    # The owner's set, read off one general search and one distance search
    # (2026-10-02): the top 2 general recipes and the top 1 distance recipe for
    # one downstream model. Returns [(slot, blueprint, label)], general first.
    def _rule_slots(gen_search, dist_search, model_name, where):
        slots = []
        for lineage, obj, n in (('general', gen_search, RULE_N_GENERAL),
                                ('distance', dist_search, RULE_N_DISTANCE)):
            if obj is None:
                rule_missing.append(
                    f'{where}: no {lineage} search, so none of its {n} '
                    f'recipe(s) for {model_name} was used')
                continue
            try:
                picked = top_hybrids(obj, model_name, n, require_full=False,
                                     context=f'{where}/{lineage}')
            except Exception as e:
                rule_missing.append(f'{where} {lineage} for {model_name}: '
                                    f'{type(e).__name__}: {e}')
                continue
            for e in picked['selected']:
                slots.append((f"{lineage}{e['rank']}", e['blueprint'],
                              e['label']))
            if not picked['complete']:
                rule_missing.append(
                    f'{where} {lineage} for {model_name}: '
                    f"{picked['shortfall_reason']}")
        return slots

    # general1 keeps the key every reader already uses; the other two slots add
    # their own suffixed keys.
    def _slot_key(base, slot):
        return base if slot == 'general1' else f'{base}__{slot}'

    per_model = {}

    # Each downstream model's block is written to
    # opig_blueprint_transfer/<source>_to_<target>.partial.jsonl when it
    # finishes and read back on the next attempt (2026-09-25;
    # experiments/_checkpoint.py). The searches above come from the
    # find_best_hybrid cache and the folds are deterministic, so a block read
    # back holds what a fresh computation gives.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity, fingerprint_of)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 'opig_blueprint_transfer',
                           f'{source_name}_to_{target_name}'),
        run_identity('opig_blueprint_transfer',
                     f'{source_name}_to_{target_name}', target_data, config,
                     source_fingerprint=fingerprint_of(source_data, config),
                     pruning_mode=pruning_mode,
                     shared_rep_names=list(shared_rep_names)))

    for model_name in model_names:
        if model_name not in model_factories_tgt or model_name not in model_factories_src:
            continue
        if _ck.done(model=model_name):
            per_model[model_name] = _ck.get(model=model_name)
            print(f"\n    {model_name}: finished in an earlier attempt, "
                  f"read back")
            continue

        # The owner's set (2026-10-02): the source's top 2 general recipes
        # and its top 1 distance recipe, each transferred to the target in its
        # own pair of arms, instead of the single general winner.
        source_slots = _rule_slots(source_search, source_distance_search,
                                   model_name,
                                   f'source={source_name}')
        target_slots_whole = _rule_slots(target_search, target_distance_search,
                                         model_name,
                                         f'target={target_name}/context')

        if not source_slots:
            print(f"\n    {model_name}: skipped (no source recipe)")
            continue

        source_bp = source_slots[0][1]
        target_bp = target_slots_whole[0][1] if target_slots_whole else None

        print(f"\n    {model_name}:")
        for _slot, _bp, _lab in source_slots:
            print(f"      Source {_slot}: budget={_bp.budget}, "
                  f"alloc={_bp.allocation_method}, "
                  f"imp={_bp.importance_method}, "
                  f"eval={_bp.internal_evaluator}")
        for _slot, _bp, _lab in target_slots_whole:
            print(f"      Target {_slot} (context): budget={_bp.budget}, "
                  f"alloc={_bp.allocation_method}, "
                  f"imp={_bp.importance_method}, "
                  f"eval={_bp.internal_evaluator}")

        # One source hybrid per slot, built on FULL source data. Its feature
        # indices are what the full_transfer arm replays on the target.
        fi_source_by_slot = {}
        for _slot, _bp, _lab in source_slots:
            try:
                _, _fi = create_hybrid(
                    src, source_labels,
                    allocation_method=_bp.allocation_method,
                    budget=_bp.budget,
                    importance_method=_bp.importance_method,
                    internal_evaluator=_bp.internal_evaluator,
                    # The accept/reject holdout inside create_hybrid is a
                    # GroupKFold fold when groups are given and a
                    # RandomState(42) 20% random split when they are not. These
                    # molecules are scaffold-grouped, so without this the
                    # keep/drop decisions were judged on rows sharing scaffolds
                    # with their own training rows.
                    groups=source_groups,
                    # What each representation scored on its own in the source
                    # search, over these same rows. Reused so the budget split
                    # is the one the search settled on.
                    precomputed_alloc_scores=getattr(_bp, 'rep_scores', None),
                    # Explicit: this experiment has no augmentations.
                    augmentations=None,
                    augmentation_strategy='none',
                )
                fi_source_by_slot[_slot] = _fi
            except Exception as e:
                print(f"      Source hybrid build failed for {_slot}: {e}")
                rule_missing.append(
                    f'source={source_name} {_slot} for {model_name}: the '
                    f'hybrid build raised {type(e).__name__}: {e}')

        if not fi_source_by_slot:
            print(f"    {model_name}: skipped (no source hybrid built)")
            continue

        for _slot, _fi in fi_source_by_slot.items():
            print(f"      Source {_slot} feature selection:")
            for rn in sorted(_fi.keys()):
                info = _fi[rn]
                if isinstance(info, dict) and 'selected_indices' in info:
                    print(f"        {rn}: {len(info['selected_indices'])}"
                          f"/{src[rn].shape[1]}")

        # The best single representations and the native recipes are re-picked
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

        def _fit_score(X_tr, X_va, y_tr, y_va, mk):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)
            m = mk()
            m.fit(X_tr, y_tr)
            return float(metric_fn(
                y_va, _predict_for_metric(m, X_va, task_type, metric_name)))

        # ── 3a: Evaluate on TARGET ──────────────────────────────────────
        # Arm names: base_best is the fold's best single representation and
        # base_best__rank2 the second; native/recipe_transfer/full_transfer
        # carry one key per recipe slot, general1 under the original name.
        _base_keys = ['base_best'] + [f'base_best__rank{i}'
                                      for i in range(2, RULE_N_BASE + 1)]
        _tgt_slots = [f'general{i}' for i in range(1, RULE_N_GENERAL + 1)] \
            + [f'distance{i}' for i in range(1, RULE_N_DISTANCE + 1)]
        tgt_fold_scores = {k: [] for k in _base_keys}
        for _c in ('native', 'recipe_transfer', 'full_transfer'):
            for _slot in _tgt_slots:
                tgt_fold_scores[_slot_key(_c, _slot)] = []
        tgt_fold_base_reps = []
        tgt_fold_native_bp = []

        print(f"      Target evaluation ({target_name}):")
        for fold_i, (train_idx, val_idx) in enumerate(target_folds):
            tr = {k: v[train_idx] for k, v in tgt.items()}
            va = {k: v[val_idx] for k, v in tgt.items()}
            y_tr = target_labels[train_idx]
            y_va = target_labels[val_idx]
            mk = model_factories_tgt[model_name]
            _tr_groups = (target_groups[train_idx]
                          if target_groups is not None else None)

            # This fold's own searches, over its training rows only.
            fold_reps = target_fold_reps[fold_i].get(model_name) or []
            fold_slots = _rule_slots(
                target_fold_searches[fold_i],
                target_fold_distance_searches[fold_i], model_name,
                f'target={target_name}/heldout_fold{fold_i}')
            tgt_fold_base_reps.append(fold_reps[:RULE_N_BASE])
            tgt_fold_native_bp.append({
                _slot: {'budget': _bp.budget,
                        'allocation': _bp.allocation_method,
                        'importance': _bp.importance_method,
                        'evaluator': _bp.internal_evaluator,
                        'recipe': _lab}
                for _slot, _bp, _lab in fold_slots})

            # A: the fold's top single representations, picked on its
            # training rows.
            for _i, _key in enumerate(_base_keys):
                _rep = fold_reps[_i] if _i < len(fold_reps) else None
                if _rep is None:
                    tgt_fold_scores[_key].append(float('nan'))
                else:
                    tgt_fold_scores[_key].append(
                        _fit_score(tr[_rep], va[_rep], y_tr, y_va, mk))

            # B: native hybrid, one per recipe slot from this fold's rows.
            _have = {sl for sl, _b, _l in fold_slots}
            for _slot, _bp, _lab in fold_slots:
                _key = _slot_key('native', _slot)
                if _key not in tgt_fold_scores:
                    tgt_fold_scores[_key] = [float('nan')] * fold_i
                try:
                    X_tr_h, X_va_h, _ = build_fold_hybrid(
                        tr, va, y_tr, _bp, groups=_tr_groups, model=mk(),
                        metric_fn=metric_fn, lower_is_better=lower_is_better)
                    tgt_fold_scores[_key].append(
                        _fit_score(X_tr_h, X_va_h, y_tr, y_va, mk))
                except Exception as e:
                    print(f"        Fold {fold_i} native {_slot} failed: {e}")
                    tgt_fold_scores[_key].append(float('nan'))
            for _slot in _tgt_slots:
                if _slot not in _have:
                    tgt_fold_scores[_slot_key('native', _slot)].append(
                        float('nan'))

            # C and D: the SOURCE recipe (C) and the source's own feature
            # indices (D), one pair per source slot.
            for _slot, _bp, _lab in source_slots:
                _key = _slot_key('recipe_transfer', _slot)
                try:
                    X_tr_h, X_va_h, _ = build_fold_hybrid(
                        tr, va, y_tr, _bp, groups=_tr_groups, model=mk(),
                        metric_fn=metric_fn, lower_is_better=lower_is_better)
                    tgt_fold_scores[_key].append(
                        _fit_score(X_tr_h, X_va_h, y_tr, y_va, mk))
                except Exception as e:
                    print(f"        Fold {fold_i} recipe {_slot} failed: {e}")
                    tgt_fold_scores[_key].append(float('nan'))

                _fkey = _slot_key('full_transfer', _slot)
                _fi = fi_source_by_slot.get(_slot)
                if _fi is None:
                    tgt_fold_scores[_fkey].append(float('nan'))
                    continue
                try:
                    tgt_fold_scores[_fkey].append(_fit_score(
                        apply_feature_selection(tr, _fi),
                        apply_feature_selection(va, _fi), y_tr, y_va, mk))
                except Exception as e:
                    print(f"        Fold {fold_i} full {_slot} failed: {e}")
                    tgt_fold_scores[_fkey].append(float('nan'))
            _src_have = {sl for sl, _b, _l in source_slots}
            for _slot in _tgt_slots:
                if _slot not in _src_have:
                    for _c in ('recipe_transfer', 'full_transfer'):
                        tgt_fold_scores[_slot_key(_c, _slot)].append(
                            float('nan'))

            print(f"        Fold {fold_i}: "
                  f"base={tgt_fold_scores['base_best'][-1]:.4f} "
                  f"({fold_reps[0] if fold_reps else None}), "
                  f"native={tgt_fold_scores['native'][-1]:.4f}, "
                  f"recipe={tgt_fold_scores['recipe_transfer'][-1]:.4f}, "
                  f"full={tgt_fold_scores['full_transfer'][-1]:.4f}")

        # ── 3b: Evaluate on SOURCE (context) ────────────────────────────
        # Both arms are re-picked inside each source fold, and the hybrid arm
        # runs once per recipe slot.
        src_fold_scores = {k: [] for k in _base_keys}
        for _slot in _tgt_slots:
            src_fold_scores[_slot_key('hybrid', _slot)] = []
        src_fold_base_reps = []
        src_fold_hybrid_bp = []

        print(f"      Source evaluation ({source_name}):")
        for fold_i, (train_idx, val_idx) in enumerate(source_folds):
            tr = {k: v[train_idx] for k, v in src.items()}
            va = {k: v[val_idx] for k, v in src.items()}
            y_tr = source_labels[train_idx]
            y_va = source_labels[val_idx]
            mk = model_factories_src[model_name]
            _tr_groups = (source_groups[train_idx]
                          if source_groups is not None else None)

            fold_reps = source_fold_reps[fold_i].get(model_name) or []
            fold_slots = _rule_slots(
                source_fold_searches[fold_i],
                source_fold_distance_searches[fold_i], model_name,
                f'source={source_name}/heldout_fold{fold_i}')
            src_fold_base_reps.append(fold_reps[:RULE_N_BASE])
            src_fold_hybrid_bp.append({
                _slot: {'budget': _bp.budget,
                        'allocation': _bp.allocation_method,
                        'importance': _bp.importance_method,
                        'evaluator': _bp.internal_evaluator,
                        'recipe': _lab}
                for _slot, _bp, _lab in fold_slots})

            for _i, _key in enumerate(_base_keys):
                _rep = fold_reps[_i] if _i < len(fold_reps) else None
                if _rep is None:
                    src_fold_scores[_key].append(float('nan'))
                else:
                    src_fold_scores[_key].append(
                        _fit_score(tr[_rep], va[_rep], y_tr, y_va, mk))

            _have = {sl for sl, _b, _l in fold_slots}
            for _slot, _bp, _lab in fold_slots:
                _key = _slot_key('hybrid', _slot)
                try:
                    X_tr_h, X_va_h, _ = build_fold_hybrid(
                        tr, va, y_tr, _bp, groups=_tr_groups, model=mk(),
                        metric_fn=metric_fn, lower_is_better=lower_is_better)
                    src_fold_scores[_key].append(
                        _fit_score(X_tr_h, X_va_h, y_tr, y_va, mk))
                except Exception as e:
                    print(f"        Fold {fold_i} source hybrid {_slot} "
                          f"failed: {e}")
                    src_fold_scores[_key].append(float('nan'))
            for _slot in _tgt_slots:
                if _slot not in _have:
                    src_fold_scores[_slot_key('hybrid', _slot)].append(
                        float('nan'))

            print(f"        Fold {fold_i}: "
                  f"base={src_fold_scores['base_best'][-1]:.4f} "
                  f"({fold_reps[0] if fold_reps else None}), "
                  f"hybrid={src_fold_scores['hybrid'][-1]:.4f}")

        # ── Aggregate for this model ────────────────────────────────────
        def _agg(scores):
            return {'mean': float(np.nanmean(scores)),
                    'std': float(np.nanstd(scores)),
                    'per_fold': scores}

        model_res = {
            'target': {},
            'source': {k: _agg(v) for k, v in src_fold_scores.items()},
            # The recipes transferred to the target. Chosen over every source
            # row, which is what recipe_transfer and full_transfer are meant to
            # test; the target folds share no rows with them.
            'source_blueprint': {
                'budget': source_bp.budget,
                'allocation': source_bp.allocation_method,
                'importance': source_bp.importance_method,
                'evaluator': source_bp.internal_evaluator,
            },
            'source_blueprints_by_slot': {
                _slot: {'budget': _bp.budget,
                        'allocation': _bp.allocation_method,
                        'importance': _bp.importance_method,
                        'evaluator': _bp.internal_evaluator,
                        'recipe': _lab}
                for _slot, _bp, _lab in source_slots},
            # Context only. Chosen over every target row, so it is not what the
            # native arm was scored with — that is per_fold_native_blueprint.
            'target_blueprint_whole_data': (
                None if target_bp is None else {
                    'budget': target_bp.budget,
                    'allocation': target_bp.allocation_method,
                    'importance': target_bp.importance_method,
                    'evaluator': target_bp.internal_evaluator,
                }),
            'rule': (f'top {RULE_N_GENERAL} general + top {RULE_N_DISTANCE} '
                     f'distance recipes, top {RULE_N_BASE} single '
                     f'representations'),
        }
        for cond, folds in tgt_fold_scores.items():
            model_res['target'][cond] = _agg(folds)
        # What each held-out fold actually used, picked on its training rows.
        model_res['target']['per_fold_base_rep'] = tgt_fold_base_reps
        model_res['target']['per_fold_native_blueprint'] = tgt_fold_native_bp
        model_res['target']['base_rep_whole_data'] = best_tgt_rep
        model_res['source']['per_fold_base_rep'] = src_fold_base_reps
        model_res['source']['per_fold_hybrid_blueprint'] = src_fold_hybrid_bp
        model_res['source']['base_rep_whole_data'] = best_src_rep

        # Deltas vs the target's best single representation, subtracted fold by
        # fold. A fold whose arm raised holds a nan and the two arms do not
        # lose the same folds, so differencing two nanmeans could take a
        # 4-fold mean off a 5-fold one. Only fold positions where BOTH arms
        # are finite enter the difference. delta_n_folds is how many folds are
        # behind the number and delta_std their spread — the win threshold
        # below is 0.001 RMSE, far under that spread, so both have to be
        # readable next to the delta.
        base_folds = model_res['target']['base_best']['per_fold']
        for cond in list(tgt_fold_scores):
            if cond in _base_keys:
                continue
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
        _ck.put(model_res, model=model_name)

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

    # ── Top-2 rows and the two deltas, added 2026-09-18 ──────────────────
    # An extra beside every number above. Nothing here feeds base_best,
    # native, recipe_transfer, full_transfer, delta_vs_base, delta_n_folds,
    # delta_std, n_recipe_wins or n_full_wins, and no arm loses a
    # representation: both searches still run over all
    # len(shared_rep_names) shared representations and every fold arm above
    # still picks from all of them.
    #
    # Every call reads rows find_best_hybrid already stored, so this costs 0
    # new model fits on top of the 2 whole-dataset searches and the
    # 2 * n_folds per-fold searches this run performs.
    def _top2_for(search_obj, which, ds_name):
        """The top-2 grid and the two deltas for one of the two searches.

        Always returns a block. Until 2026-09-28 it returned None when only
        'dataset' and 'which_search' got in; it now always carries
        'top2_models_compared' and 'top2_models_dropped' as well, which name the
        downstream models asked for and why each missing one is missing, and
        those reasons used to exist only in the printed log.
        """
        block = {'dataset': ds_name, 'which_search': which}
        ctx = f'opig_blueprint_transfer/{which}={ds_name}'
        # WHICH DOWNSTREAM MODELS THIS BLOCK COVERS, resolved 2026-09-28.
        # `model_names` is the list both searches were given (line 224). What a
        # search on disk actually scored can be shorter: svm drops itself above
        # 10,000 rows, and a dataset's skip policy can drop a name -- and the two
        # searches here are on two different datasets, so one can be missing a
        # model the other has. The loops below then asked for a model with no
        # stored rows.
        #
        # `model_names` stays what this file ASKS for, so the set of models it
        # reports does not change: it goes in as `requested`, not None. A name
        # this search never scored is dropped and named, never swapped for
        # another model's rows.
        #
        # Both lists are SAVED, not only printed. Until now every reason below
        # went to the terminal alone, so a block holding three models read the
        # same as a run configured for three.
        cmp_models, cmp_dropped = resolve_comparison_models(
            search_obj, model_names)
        block['top2_models_compared'] = list(cmp_models)
        block['top2_models_dropped'] = {n: w for n, w in cmp_dropped}
        for n, w in cmp_dropped:
            print(f"    {which}: downstream model {n} not compared: {w}")
        try:
            # models=model_names, the list both searches scored (line 224),
            # not _shared.py's 5-name default: a downstream model the two
            # searches did not share is not evaluated anywhere in this file.
            # len(model_names) downstream models * (2 representations +
            # 2 hybrids) rows. per_model inside this holds each downstream
            # model's own top_base_reps and top_hybrids return, so the two
            # halves are recorded here and not as separate keys.
            block['top2_pairs_by_model'] = select_model_rep_pairs(
                search_obj, models=cmp_models, n_base=2, n_hybrids=2,
                require_full=False, context=ctx)
        except Exception as _e:
            print(f"    top2_pairs_by_model ({which}) not added: "
                  f"{type(_e).__name__}: {_e}")
            block['top2_pairs_by_model_absent_reason'] = \
                f"{type(_e).__name__}: {_e}"

        # The rank-1 hybrid against TWO reference numbers rather than one:
        #   delta_vs_source_mean — against the MEAN of the base
        #     representations the budget split actually gave features to,
        #     read off Blueprint.allocation_per_fold, so a representation
        #     given 0 features on every fold is out of that mean.
        #   delta_vs_best_rep — against the single best base representation
        #     for that same downstream model.
        # Positive means the hybrid is better, on either metric direction.
        # Both are in this run's metric, metric_name, which the two searches
        # were given; delta_vs_base above is the same metric measured a
        # different way — per held-out fold — so the two are not the same
        # number and carry different key names.
        #
        # hybrid= is passed rather than left to default. hybrid_deltas with
        # hybrid=None reads _picked['selected'][0]['row'] at
        # _shared.py:7432, and the entry top_hybrids builds
        # (_shared.py:7043-7066) carries no 'row' key — read in this session,
        # both ends. The rank-1 entry handed in carries 'score_mean' and
        # 'blueprint', the only two keys hybrid_deltas reads off it.
        #
        # One try per downstream model, so a model with no stored hybrid does
        # not cost the others their entry. Keys are downstream model names;
        # this dict is new and nothing iterates it.
        deltas = {}
        for mn in cmp_models:
            try:
                top1 = top_hybrids(search_obj, mn, 1, require_full=False,
                                   context=f'{ctx}/deltas/{mn}')
                if not top1['selected']:
                    print(f"    hybrid_deltas ({which}) for downstream model "
                          f"{mn} not added: {top1['shortfall_reason']}")
                    block['top2_models_dropped'][mn] = (
                        'no hybrid delta: %s' % top1['shortfall_reason'])
                    continue
                deltas[mn] = hybrid_deltas(
                    search_obj, mn, hybrid=top1['selected'][0],
                    context=f'{ctx}/deltas/{mn}')
            except Exception as _e:
                print(f"    hybrid_deltas ({which}) for downstream model {mn} "
                      f"not added: {type(_e).__name__}: {_e}")
                block['top2_models_dropped'][mn] = (
                    'no hybrid delta: %s: %s' % (type(_e).__name__, _e))
        if deltas:
            block['hybrid_deltas_by_model'] = deltas
        # The block is always returned since 2026-09-28. It now always carries
        # 'top2_models_compared' and 'top2_models_dropped', which name the
        # downstream models asked for and why each missing one is missing, so
        # returning None would throw away the only record of that.
        return block

    # Keyed 'source' and 'target', so the two searches' rows can never be read
    # as one set. This dict is new; nothing in this file or in the summary
    # tables above iterates it, and per_model is untouched — a non-model key
    # there would break the three loops at lines 663, 677 and 692-695, each of
    # which reads res['source'] and res['target'] off every value.
    #
    # The two distance searches are read here too (owner's set, 2026-10-02),
    # under their own keys, so the report rows cover the distance lineage and
    # not only the general one.
    top2_selection = {}
    for _which, _search_obj, _ds in (
            ('source', source_search, source_name),
            ('target', target_search, target_name),
            ('source_distance', source_distance_search, source_name),
            ('target_distance', target_distance_search, target_name)):
        if _search_obj is None:
            continue
        _blk = _top2_for(_search_obj, _which, _ds)
        if _blk is not None:
            top2_selection[_which] = _blk

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
    # Added, never overwritten: 'top2_selection' is not a key this dict
    # carried before, and the nine keys above are written exactly as they
    # were. convert_numpy runs over the whole dict inside
    # save_experiment_results (_shared.py:7966) and turns each stored
    # Blueprint into its to_dict form, so the block is JSON-serialisable.
    if top2_selection:
        results['top2_selection'] = convert_numpy(top2_selection)
    # Every slot the owner's set asked for and did not get, with the reason.
    results['rule'] = (
        f'per downstream model: top {RULE_N_GENERAL} general + top '
        f'{RULE_N_DISTANCE} distance recipes, both datasets, and the top '
        f'{RULE_N_BASE} single representations')
    results['rule_missing'] = rule_missing

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
    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard the models an earlier attempt finished '
             '(<output>/opig_blueprint_transfer/<pair>.partial.jsonl).')
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
    # Written first, then a failed run exits 1 (2026-09-25).
    if isinstance(results, dict) and 'error' in results:
        print(f"FAILED: {results['error']}")
        sys.exit(1)
