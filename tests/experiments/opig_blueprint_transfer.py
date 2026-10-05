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

    Changed 2026-10-04 (DECISIONS.md, the owner's rules of 2026-10-03):
      - Models: only the downstream models DECISIONS.md Step 1 gives BOTH
        datasets (logd dnn+svm, herg_ki svm+dnn), not the five
        DOWNSTREAM_MODELS.
      - Every search runs the dataset's decided grid (budgets, importance,
        allocation, evaluators, representation counts, augmentation 'none'),
        through _apply_dataset_search_speed. The source search over all
        source rows and the per-fold searches on both datasets now run the
        same grid; before, the first ran the fast grid and the second the
        full one.
      - A and B, and the source-side check, are the standard selection per
        model per fold (fold_standard_selection): the 2 best single
        representations and the 2 best hybrids, from that fold's own search.
        The distance search and its third hybrid are gone from both sides.
      - The two whole-target searches are gone. They fed only context fields
        (a target recipe and a target representation picked with every
        held-out fold in view), and nothing scored used them.
      - C and D still transfer the source's top 2 recipes from ONE search over
        every source row. That is the experiment, not a leak: no target row is
        in it. That search is built here (own_search=True), never refused,
        because its pool is the representations both datasets share and no
        precompute stores that pool.
      - D's source features are now built by build_fold_hybrid with the
        downstream model and the dataset metric, like C, and keep the
        recipe's representation count. Before, create_hybrid ran with neither
        and over the whole pool, so D differed from C in three ways at once.

    Changed 2026-10-05 (the owner's answers of 2026-10-05, handoff 2.3/2.4):
      - Pruning is off in every search: the search over every source row and
        every per-fold search. --pruning-mode is accepted and ignored with a
        printed line.
      - The source's 2 transferred recipes have distinct selected features:
        each candidate is built on every source row and one whose features
        equal a recipe already taken is skipped for the next (the per-fold
        hybrids already did this, fold_standard_selection).
      - Single representations and hybrids are scaled with FeatureScaler: 0/1
        features are left as they are, the rest standardised (answer 17).
      - Every held-out fold is written to the partial file as it finishes,
        not only each model at the end (dirty-run bug 12).
      - Every per-fold search that failed is saved with dataset, fold, model
        and error under 'search_failures', and its arms are skipped.
      - The training-row score (the fold search's own mean) is saved beside
        every held-out score.
      - One outer fold (--mode intermediary) is a single held-out split; every
        spread over folds then reads "one fold, no spread".
      - --max-folds now stops after that many held-out folds; it did nothing.
      - --models narrows the decided models only when it is given; before,
        get_config's five-model default stood in for it.

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

from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    load_data, get_config, save_experiment_results, convert_numpy,
    get_model_factories, build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_standard_selection, fold_searches, _apply_dataset_search_speed,
    get_domain_metric, hybrid_feature_signature,
    select_model_rep_pairs, hybrid_deltas, top_hybrids,
    resolve_comparison_models,
    STANDARD_N_BASE, STANDARD_N_HYBRIDS,
)
from experiments.decided_grid import (decided_models, decided_search,
                                      decided_aug_conditions)
from kirby.scaling import FeatureScaler
from kirby.find_best_hybrid import (find_best_hybrid, nested_cv_outer_folds,
                                    _BASELINE_NON_SCORE_KEYS)
from kirby.hybrid import apply_feature_selection

# The outer folds' seed. fold_standard_selection files each fold's search
# under (fold, n_folds, seed), so the split scored here and the split searched
# must be built with the same one.
FOLD_SEED = 42

# How many lower-ranked source recipes are drawn beyond the 2 transferred, so a
# recipe whose selected features equal one already taken can be replaced by the
# next one (2026-10-05). Same spare count fold_standard_selection draws.
SOURCE_SPARE_RECIPES = 4

# Written into the partial file's identity. Changing it moves every older
# partial file aside, so no fold computed under the earlier code is read back.
# '2026-10-05': pruning off, FeatureScaler, one unit per held-out fold.
# '2026-10-05b': each fold unit carries its own rule_missing entries and each
# model block its source-recipe ones, so a unit read back keeps the reason
# behind every nan arm.
ALIGNMENT_VERSION = '2026-10-05b'


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
        pruning_mode=None, requested_models=None):
    """OPIG C5: Cross-dataset blueprint transfer.

    pruning_mode: ignored since 2026-10-05; every search runs unpruned.
    requested_models: the --models list, or None. Narrows the models decided
        on both datasets; never adds one.
    """
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

    # Pruning is off in every search (owner, 2026-10-05; handoff 2.1). Before,
    # the source search and each per-fold search got a schedule from
    # resolve_pruning_kwargs(pruning_mode, ...).
    if pruning_mode not in (None, 'none', 'no_pruning'):
        print(f"  --pruning-mode {pruning_mode} ignored: every search runs "
              f"unpruned (2026-10-05).")

    # Every slot that could not be filled is named here, with its reason, and
    # saved.
    rule_missing = []
    # Every per-fold search that failed: dataset, fold, model, error. Its arms
    # are skipped (nan) for that fold, never filled from another search.
    search_failures = []

    # ── Downstream models: the ones DECISIONS.md gives both datasets ────
    # Step 1 names two models per dataset. A recipe is transferred for one
    # downstream model, so only a model both datasets carry can be compared.
    # A dataset with no Step 1 row raises UndecidedSearch here, before any
    # search. knn is not one of them: it is a distance-search model only
    # (Step 9), and this experiment runs no distance search.
    _src_models = decided_models(source_name, 'general')
    _tgt_models = decided_models(target_name, 'general')
    # --models narrows, when given (2026-10-05). config['models'] is not read:
    # get_config fills it with the five DOWNSTREAM_MODELS whether or not
    # --models was passed, so it filtered nothing.
    _requested = (list(requested_models) if requested_models
                  else list(dict.fromkeys(_tgt_models + _src_models)))
    model_names = [m for m in _tgt_models
                   if m in _src_models and m in _requested]
    for m in _tgt_models + [x for x in _src_models if x not in _tgt_models]:
        if m in model_names:
            continue
        if m not in _src_models or m not in _tgt_models:
            rule_missing.append(
                f'{m}: decided on only one of {source_name} ({_src_models}) '
                f'and {target_name} ({_tgt_models}), so no recipe can be '
                f'transferred for it')
        else:
            rule_missing.append(f'{m}: not in --models {_requested}')
    print(f"  Downstream models (decided on both datasets): {model_names}")
    if not model_names:
        return {'error': 'no_shared_decided_models',
                'rule_missing': rule_missing}

    # ── Step 1: Hybrid search on ALL source rows ───────────────────────
    # The recipes C and D transfer. Searched over every source row on
    # purpose: no target row is in it, and transferring what the source
    # learned is what this experiment tests.
    #
    # Its pool is the representations both datasets share, which no
    # precompute stores, so it is this experiment's own search and is built
    # here (own_search=True: KIRBY_NO_COLD_SEARCH does not refuse it). Its grid
    # is the source's decided grid: every axis left as None, filled by
    # _apply_dataset_search_speed. Pruning off: no schedule, no per-model
    # aggressiveness (find_best_hybrid's defaults are off). Augmentation is
    # the decided conditions' 'none'; a source whose decided general search
    # keeps another condition is named in rule_missing, since this experiment
    # loads no augmentation features.
    print(f"\n  Step 1: Hybrid search on source ({source_name}), all rows")
    _src_grid = _apply_dataset_search_speed({'tier': 'general'}, source_name)
    _src_grid.pop('tier', None)
    _src_augs = [c for c in decided_aug_conditions(source_name, 'general')
                 if c != 'none']
    if _src_augs:
        rule_missing.append(
            f'source={source_name}: decided augmentation {_src_augs} not run; '
            f'this experiment transfers base-representation features only')
    source_search = find_best_hybrid(
        base_reps=src, labels=source_labels, groups=source_groups,
        augmentation_strategy='none',
        metric=metric_name, task_type=task_type,
        verbose=False, cache=True,
        dataset_name=source_name, tier='general',
        own_search=True,
        **_src_grid,
    )

    # ── Step 2: Per-model evaluation ────────────────────────────────────
    print(f"\n  Step 2: Per-model evaluation ({n_folds}-fold CV)")

    model_factories_tgt = get_model_factories(
        model_names, task_type=task_type, n_samples=len(target_labels))
    model_factories_src = get_model_factories(
        model_names, task_type=task_type, n_samples=len(source_labels))

    target_folds = nested_cv_outer_folds(
        tgt, target_labels, target_groups, n_folds, FOLD_SEED, task_type)
    source_folds = nested_cv_outer_folds(
        src, source_labels, source_groups, n_folds, FOLD_SEED, task_type)
    # --max-folds: score only the first N held-out folds of the n_folds split
    # (2026-10-05; get_config stored it and nothing read it). The split and
    # each fold's search are still those of n_folds, so fold i here is fold i
    # of a full run.
    _max_folds = config.get('max_folds')
    if _max_folds:
        target_folds = target_folds[:int(_max_folds)]
        source_folds = source_folds[:int(_max_folds)]
        print(f"  --max-folds {_max_folds}: scoring {len(target_folds)} of "
              f"{n_folds} folds")

    # ── Step 2 prelude: the standard selection inside each held-out fold ──
    # One search per held-out fold, on that fold's training rows only, on the
    # dataset's decided grid (fold_standard_selection -> search_on_fold). From
    # it, per downstream model: the 2 best single representations and the 2
    # best hybrids. The native arm, the base_best baseline and the source-side
    # check all read it; nothing in them comes from a search over all rows.
    # A fold whose search does not exist yet is built here.
    def _fold_selections(reps_dict, labels_arr, groups_arr, folds, ds_name):
        """One {model: selection or None} per fold; {} where the call raised.

        Pruning is off (2026-10-05): no schedule is passed. A fold whose
        search failed is saved in search_failures with dataset, fold, model
        and error; its arms are skipped for that fold.
        """
        out = []
        for fold_i, (train_idx, _val_idx) in enumerate(folds):
            try:
                sel = fold_standard_selection(
                    reps_dict, labels_arr, groups_arr,
                    train_idx=train_idx, fold_i=fold_i,
                    n_outer_folds=n_folds, dataset_name=ds_name,
                    models=model_names, task_type=task_type,
                    metric=metric_name, seed=FOLD_SEED,
                    n_base=STANDARD_N_BASE, n_hybrids=STANDARD_N_HYBRIDS)
            except Exception as e:
                print(f"      fold {fold_i} search on {ds_name} failed: {e}")
                for m in model_names:
                    search_failures.append({
                        'dataset': ds_name, 'fold': int(fold_i), 'model': m,
                        'tier': 'general',
                        'error': f'{type(e).__name__}: {e}'})
                out.append({})
                continue
            # fold_standard_selection maps every model to None when the
            # fold's search failed and only prints why. The error is read
            # back from the same search, which fold_searches has memoised
            # under the same arguments, so nothing is searched twice.
            if any(v is None for v in sel.values()):
                errs = []
                try:
                    _s = fold_searches(
                        reps_dict, labels_arr, groups_arr,
                        train_idx=train_idx, fold_i=fold_i,
                        n_outer_folds=n_folds, dataset_name=ds_name,
                        task_type=task_type, metric=metric_name,
                        tier='general', seed=FOLD_SEED)
                    errs = [f.get('error') or f.get('error_type')
                            for f in (getattr(_s, 'failures', None) or [])]
                except Exception as e:
                    errs = [f'{type(e).__name__}: {e}']
                for m, v in sel.items():
                    if v is None:
                        search_failures.append({
                            'dataset': ds_name, 'fold': int(fold_i),
                            'model': m, 'tier': 'general',
                            'error': '; '.join(str(x) for x in errs)
                                     or 'the search returned no result'})
            out.append(sel)
        return out

    print(f"\n  Per-fold searches on target ({target_name}), "
          f"{len(target_folds)} folds")
    target_fold_sel = _fold_selections(
        tgt, target_labels, target_groups, target_folds, target_name)
    print(f"  Per-fold searches on source ({source_name}), "
          f"{len(source_folds)} folds")
    source_fold_sel = _fold_selections(
        src, source_labels, source_groups, source_folds, source_name)

    def _fold_alloc_kw(bp, fold_search, n_pool):
        """precomputed_alloc_scores from one fold's own search, when the
        recipe keeps fewer representations than the pool.

        build_fold_hybrid keeps the recipe's representation count by ranking
        the representations, and without this it ranks them by
        blueprint.rep_scores — the scores of the search the recipe came from.
        For a transferred recipe that is the search over every SOURCE row;
        recipe transfer means the target fold picks its own. The scores are
        built the way find_best_hybrid builds Blueprint.rep_scores (each
        representation's best baseline over the downstream models, higher is
        better), from the fold search's baselines.
        """
        rc = getattr(bp, 'rep_count', None)
        if rc is None or int(rc) >= n_pool:
            return {}
        if fold_search is None:
            raise ValueError(
                f'the recipe keeps {rc} of {n_pool} representations and this '
                f'fold has no search of its own to rank them')
        lib = fold_search.lower_is_better
        scores = {}
        for rn, ms in (getattr(fold_search, 'baselines', {}) or {}).items():
            vals = [v for k, v in (ms or {}).items()
                    if k not in _BASELINE_NON_SCORE_KEYS
                    and isinstance(v, (int, float)) and np.isfinite(v)]
            if vals:
                best = float(min(vals) if lib else max(vals))
                scores[rn] = -best if lib else best
        return {'precomputed_alloc_scores': scores}

    def _bp_record(bp, label, aug_condition, scope, search_score=None):
        """What one hybrid was, as the results file stores it.

        search_score: the recipe's mean score in the search that picked it
        (2026-10-05). For a per-fold hybrid that is the training-row score,
        over that search's inner folds; for a transferred recipe it is the
        score over every source row. Budget is in features.
        """
        return {'budget': bp.budget,
                'allocation': bp.allocation_method,
                'importance': bp.importance_method,
                'evaluator': bp.internal_evaluator,
                'rep_count': getattr(bp, 'rep_count', None),
                'augmentation': aug_condition,
                'recipe': label,
                'search_score': (float(search_score)
                                 if search_score is not None else None),
                'metric': metric_name,
                'lower_is_better': lower_is_better,
                'selection_scope': scope}

    def _sel_slots(sel, where, model_name):
        """[(slot, blueprint, label, aug_condition, train_score)] from one
        fold's standard selection, best first; the base representations; their
        training-row scores; and that fold's search (or None)."""
        if not sel:
            rule_missing.append(f'{where}: no search, so neither of its '
                                f'{STANDARD_N_HYBRIDS} hybrids for '
                                f'{model_name} was used')
            return [], [], [], None
        hyb = sel['hybrids']
        if len(hyb) < STANDARD_N_HYBRIDS:
            rule_missing.append(
                f'{where} for {model_name}: {len(hyb)} of '
                f'{STANDARD_N_HYBRIDS} hybrids')
        if len(sel['base']) < STANDARD_N_BASE:
            rule_missing.append(
                f'{where} for {model_name}: {len(sel["base"])} of '
                f'{STANDARD_N_BASE} single representations')
        slots = [(f'general{i + 1}', e['blueprint'], e['label'],
                  e.get('aug_condition', 'none'), e.get('score_mean'))
                 for i, e in enumerate(hyb)]
        base_train = [e.get('score') for e in
                      ((sel.get('base_detail') or {}).get('selected') or [])]
        return slots, list(sel['base']), base_train, sel['search']

    # general1 keeps the key every reader already uses; general2 adds its own
    # suffixed key.
    def _slot_key(base, slot):
        return base if slot == 'general1' else f'{base}__{slot}'

    per_model = {}

    # Each held-out fold is written to
    # opig_blueprint_transfer/<source>_to_<target>.partial.jsonl as it
    # finishes, keyed (model, side, fold), and each downstream model's block
    # when the model finishes; both are read back on the next attempt
    # (2026-09-25 per model; per fold since 2026-10-05, dirty-run bug 12).
    # The searches come from the find_best_hybrid cache and the folds are
    # deterministic, so a unit read back holds what a fresh computation
    # gives. The decided grids and ALIGNMENT_VERSION are part of the
    # identity, so a unit computed under another grid or under the pruned
    # code of 2026-10-04 is not read back.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity, fingerprint_of)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 'opig_blueprint_transfer',
                           f'{source_name}_to_{target_name}'),
        run_identity('opig_blueprint_transfer',
                     f'{source_name}_to_{target_name}', target_data, config,
                     source_fingerprint=fingerprint_of(source_data, config),
                     alignment=ALIGNMENT_VERSION,
                     pruning='off',
                     scaler='FeatureScaler',
                     n_outer_folds=int(n_folds),
                     shared_rep_names=list(shared_rep_names),
                     models=list(model_names),
                     selection='fold_standard_selection',
                     decided_grid={
                         'source': decided_search(source_name, 'general'),
                         'target': decided_search(target_name, 'general')}))

    _tgt_slots = [f'general{i}' for i in range(1, STANDARD_N_HYBRIDS + 1)]
    _base_keys = ['base_best'] + [f'base_best__rank{i}'
                                  for i in range(2, STANDARD_N_BASE + 1)]

    def _fit_score(X_tr, X_va, y_tr, y_va, mk):
        # FeatureScaler (2026-10-05, answer 17): 0/1 features are left as
        # they are, continuous ones standardised. StandardScaler scaled both.
        sc = FeatureScaler()
        X_tr = sc.fit_transform(X_tr)
        X_va = sc.transform(X_va)
        m = mk()
        m.fit(X_tr, y_tr)
        return float(metric_fn(
            y_va, _predict_for_metric(m, X_va, task_type, metric_name)))

    for model_name in model_names:
        if model_name not in model_factories_tgt or model_name not in model_factories_src:
            continue
        if _ck.done(model=model_name):
            per_model[model_name] = _ck.get(model=model_name)
            # The reasons behind its nan arms were saved with it (2026-10-05);
            # rule_missing lives in memory, so they are put back here.
            _mr = per_model[model_name]
            rule_missing.extend(_mr.get('source_rule_missing') or [])
            for _side in ('target', 'source'):
                for _r in (_mr.get(_side, {}).get('per_fold_records') or []):
                    rule_missing.extend(_r.get('rule_missing') or [])
            print(f"\n    {model_name}: finished in an earlier attempt, "
                  f"read back")
            continue

        # The source's top 2 recipes for this model, from the search over
        # every source row. Each is transferred to the target in its own pair
        # of arms (C and D). SOURCE_SPARE_RECIPES more are drawn so a recipe
        # whose selected features equal one already taken can be replaced by
        # the next (2026-10-05, owner: skip a second hybrid with identical
        # selected features).
        # Every rule_missing entry from here to the shortfall check is about
        # this model's source recipes; they are saved in its block
        # (source_rule_missing) so a block read back keeps them (2026-10-05).
        _n0_src = len(rule_missing)
        candidates = []
        try:
            picked = top_hybrids(source_search, model_name,
                                 STANDARD_N_HYBRIDS + SOURCE_SPARE_RECIPES,
                                 require_full=False,
                                 context=f'source={source_name}/all_rows')
            candidates = list(picked['selected'])
        except Exception as e:
            rule_missing.append(f'source={source_name} for {model_name}: '
                                f'{type(e).__name__}: {e}')

        # One source hybrid per transferred recipe, built on ALL source rows.
        # Its features are what the full_transfer arm replays on the target.
        # Built the way C builds its hybrid on a target fold — build_fold_hybrid,
        # the downstream model and the dataset metric judging each feature, the
        # recipe's representation count kept — so C and D differ only in
        # whether the features are chosen again on the target. The
        # representation ranking is the source search's own (blueprint.
        # rep_scores), measured over these same rows.
        mk_src = model_factories_src[model_name]
        source_slots = []          # [(slot, blueprint, label, search score)]
        fi_source_by_slot = {}
        source_skipped_identical = []
        _seen_sigs = {}
        for e in candidates:
            if len(source_slots) >= STANDARD_N_HYBRIDS:
                break
            _bp = e['blueprint']
            try:
                _, _, _fi = build_fold_hybrid(
                    src, src, source_labels, _bp,
                    # The accept/reject holdout inside create_hybrid is a
                    # GroupKFold fold when groups are given and a random 20%
                    # split when they are not; these molecules are
                    # scaffold-grouped.
                    groups=source_groups,
                    **fold_hybrid_scoring_kwargs(
                        _bp, model=mk_src(), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
            except Exception as ex:
                print(f"      Source hybrid build failed for {e['label']}: {ex}")
                rule_missing.append(
                    f'source={source_name} recipe {e["label"]} for '
                    f'{model_name}: the hybrid build raised '
                    f'{type(ex).__name__}: {ex}')
                continue
            _sig = hybrid_feature_signature(_fi)
            if _sig in _seen_sigs:
                source_skipped_identical.append({
                    'label': e['label'], 'score_mean': e.get('score_mean'),
                    'same_features_as': _seen_sigs[_sig]})
                print(f"      Source recipe {e['label']} selects the same "
                      f"features as {_seen_sigs[_sig]}; taking the next")
                continue
            _slot = f'general{len(source_slots) + 1}'
            _seen_sigs[_sig] = e['label']
            source_slots.append((_slot, _bp, e['label'], e.get('score_mean')))
            fi_source_by_slot[_slot] = _fi
        if len(source_slots) < STANDARD_N_HYBRIDS:
            rule_missing.append(
                f'source={source_name} for {model_name}: {len(source_slots)} '
                f'of {STANDARD_N_HYBRIDS} transferable recipes with distinct '
                f'selected features ({len(candidates)} candidates ranked)')
        source_rule_missing = list(rule_missing[_n0_src:])

        if not source_slots:
            print(f"\n    {model_name}: skipped (no source recipe built)")
            continue

        source_bp = source_slots[0][1]

        print(f"\n    {model_name}:")
        for _slot, _bp, _lab, _ss in source_slots:
            print(f"      Source {_slot}: budget={_bp.budget}, "
                  f"alloc={_bp.allocation_method}, "
                  f"imp={_bp.importance_method}, "
                  f"eval={_bp.internal_evaluator}, "
                  f"rep_count={getattr(_bp, 'rep_count', None)}")
            print(f"      Source {_slot} feature selection:")
            _fi = fi_source_by_slot[_slot]
            for rn in sorted(_fi.keys()):
                info = _fi[rn]
                if isinstance(info, dict) and 'selected_indices' in info:
                    print(f"        {rn}: {len(info['selected_indices'])}"
                          f"/{src[rn].shape[1]}")

        # ── 2a: Evaluate on TARGET ──────────────────────────────────────
        # Arm names: base_best is the fold's best single representation and
        # base_best__rank2 the second; native/recipe_transfer/full_transfer
        # carry one key per hybrid slot, general1 under the original name.
        def _target_fold(fold_i, train_idx, val_idx):
            """Every target arm on one held-out fold, as one saved unit."""
            # Entries added to rule_missing while this fold runs are saved in
            # its record, so a fold read back keeps why an arm is nan.
            _n0 = len(rule_missing)
            tr = {k: v[train_idx] for k, v in tgt.items()}
            va = {k: v[val_idx] for k, v in tgt.items()}
            y_tr = target_labels[train_idx]
            y_va = target_labels[val_idx]
            mk = model_factories_tgt[model_name]
            _tr_groups = (target_groups[train_idx]
                          if target_groups is not None else None)
            _score_kw = lambda bp: fold_hybrid_scoring_kwargs(
                bp, model=mk(), metric_fn=metric_fn,
                lower_is_better=lower_is_better)
            scores = {}

            # This fold's standard selection, from its training rows only.
            fold_slots, fold_reps, base_train, fold_search = _sel_slots(
                target_fold_sel[fold_i].get(model_name),
                f'target={target_name}/heldout_fold{fold_i}', model_name)

            # A: the fold's top single representations.
            for _i, _key in enumerate(_base_keys):
                _rep = fold_reps[_i] if _i < len(fold_reps) else None
                scores[_key] = (float('nan') if _rep is None else
                                _fit_score(tr[_rep], va[_rep], y_tr, y_va, mk))

            # B: native hybrid, one per slot of this fold's selection. The
            # representation count is ranked by this fold search's own
            # scores (_fold_alloc_kw), never a whole-dataset ranking.
            for _slot, _bp, _lab, _aug, _ts in fold_slots:
                _key = _slot_key('native', _slot)
                try:
                    X_tr_h, X_va_h, _ = build_fold_hybrid(
                        tr, va, y_tr, _bp, groups=_tr_groups,
                        **_fold_alloc_kw(_bp, fold_search, len(tr)),
                        **_score_kw(_bp))
                    scores[_key] = _fit_score(X_tr_h, X_va_h, y_tr, y_va, mk)
                except Exception as e:
                    print(f"        Fold {fold_i} native {_slot} failed: {e}")
                    rule_missing.append(
                        f'target={target_name}/heldout_fold{fold_i} native '
                        f'{_slot} for {model_name}: {type(e).__name__}: {e}')
                    scores[_key] = float('nan')

            # C and D: the SOURCE recipe (C), its features chosen again on
            # this fold's training rows, and the source's own features (D),
            # one pair per source slot. C ranks the recipe's representation
            # count by THIS target fold's search (dirty-run bug 1): the
            # recipe's own blueprint.rep_scores were measured on every source
            # row. A fold with no search of its own cannot rank them, and C
            # is then skipped for a recipe that keeps fewer than all.
            for _slot, _bp, _lab, _ss in source_slots:
                _key = _slot_key('recipe_transfer', _slot)
                try:
                    X_tr_h, X_va_h, _ = build_fold_hybrid(
                        tr, va, y_tr, _bp, groups=_tr_groups,
                        **_fold_alloc_kw(_bp, fold_search, len(tr)),
                        **_score_kw(_bp))
                    scores[_key] = _fit_score(X_tr_h, X_va_h, y_tr, y_va, mk)
                except Exception as e:
                    print(f"        Fold {fold_i} recipe {_slot} failed: {e}")
                    rule_missing.append(
                        f'target={target_name}/heldout_fold{fold_i} '
                        f'recipe_transfer {_slot} for {model_name}: '
                        f'{type(e).__name__}: {e}')
                    scores[_key] = float('nan')

                _fkey = _slot_key('full_transfer', _slot)
                try:
                    _fi = fi_source_by_slot[_slot]
                    scores[_fkey] = _fit_score(
                        apply_feature_selection(tr, _fi),
                        apply_feature_selection(va, _fi), y_tr, y_va, mk)
                except Exception as e:
                    print(f"        Fold {fold_i} full {_slot} failed: {e}")
                    rule_missing.append(
                        f'target={target_name}/heldout_fold{fold_i} '
                        f'full_transfer {_slot} for {model_name}: '
                        f'{type(e).__name__}: {e}')
                    scores[_fkey] = float('nan')
            for _c in ('native', 'recipe_transfer', 'full_transfer'):
                for _slot in _tgt_slots:
                    scores.setdefault(_slot_key(_c, _slot), float('nan'))

            return {
                'fold': int(fold_i),
                'n_train_rows': int(len(train_idx)),
                'n_held_out_rows': int(len(val_idx)),
                'held_out': scores,
                'base_reps': fold_reps[:STANDARD_N_BASE],
                # Each base representation's training-row score: its mean
                # over the fold search's inner folds.
                'base_train_scores': base_train[:STANDARD_N_BASE],
                'native_blueprints': {
                    _slot: _bp_record(_bp, _lab, _aug, 'fold_train_rows', _ts)
                    for _slot, _bp, _lab, _aug, _ts in fold_slots},
                # recipe_transfer and full_transfer have no training-row score
                # on this fold: no search on the target scores the source's
                # recipes. Their one search score is the source search's,
                # over every source row, stored under
                # source_blueprints_by_slot[*].search_score (2026-10-05).
                'transfer_train_scores': {
                    _slot: {'recipe_transfer': None, 'full_transfer': None}
                    for _slot, _bp, _lab, _ss in source_slots},
                'transfer_train_score_note': (
                    f'no target search scores the transferred recipes; their '
                    f'only search score is over every {source_name} row'),
                'metric': metric_name,
                'lower_is_better': lower_is_better,
                'selection_scope': 'fold_train_rows',
                'rule_missing': list(rule_missing[_n0:]),
            }

        # ── 2b: Evaluate on SOURCE (context) ────────────────────────────
        # The same standard selection, inside each source fold.
        def _source_fold(fold_i, train_idx, val_idx):
            """The source-side check on one held-out fold, as one saved unit."""
            _n0 = len(rule_missing)
            tr = {k: v[train_idx] for k, v in src.items()}
            va = {k: v[val_idx] for k, v in src.items()}
            y_tr = source_labels[train_idx]
            y_va = source_labels[val_idx]
            mk = model_factories_src[model_name]
            _tr_groups = (source_groups[train_idx]
                          if source_groups is not None else None)
            scores = {}

            fold_slots, fold_reps, base_train, fold_search = _sel_slots(
                source_fold_sel[fold_i].get(model_name),
                f'source={source_name}/heldout_fold{fold_i}', model_name)

            for _i, _key in enumerate(_base_keys):
                _rep = fold_reps[_i] if _i < len(fold_reps) else None
                scores[_key] = (float('nan') if _rep is None else
                                _fit_score(tr[_rep], va[_rep], y_tr, y_va, mk))

            for _slot, _bp, _lab, _aug, _ts in fold_slots:
                _key = _slot_key('hybrid', _slot)
                try:
                    X_tr_h, X_va_h, _ = build_fold_hybrid(
                        tr, va, y_tr, _bp, groups=_tr_groups,
                        **_fold_alloc_kw(_bp, fold_search, len(tr)),
                        **fold_hybrid_scoring_kwargs(
                            _bp, model=mk(), metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                    scores[_key] = _fit_score(X_tr_h, X_va_h, y_tr, y_va, mk)
                except Exception as e:
                    print(f"        Fold {fold_i} source hybrid {_slot} "
                          f"failed: {e}")
                    rule_missing.append(
                        f'source={source_name}/heldout_fold{fold_i} hybrid '
                        f'{_slot} for {model_name}: {type(e).__name__}: {e}')
                    scores[_key] = float('nan')
            for _slot in _tgt_slots:
                scores.setdefault(_slot_key('hybrid', _slot), float('nan'))

            return {
                'fold': int(fold_i),
                'n_train_rows': int(len(train_idx)),
                'n_held_out_rows': int(len(val_idx)),
                'held_out': scores,
                'base_reps': fold_reps[:STANDARD_N_BASE],
                'base_train_scores': base_train[:STANDARD_N_BASE],
                'hybrid_blueprints': {
                    _slot: _bp_record(_bp, _lab, _aug, 'fold_train_rows', _ts)
                    for _slot, _bp, _lab, _aug, _ts in fold_slots},
                'metric': metric_name,
                'lower_is_better': lower_is_better,
                'selection_scope': 'fold_train_rows',
                'rule_missing': list(rule_missing[_n0:]),
            }

        def _run_folds(side, folds, fold_fn, ds_label):
            """Every held-out fold of one side; each written when it finishes
            and read back when an earlier attempt finished it."""
            recs = []
            print(f"      {side.capitalize()} evaluation ({ds_label}):")
            for fold_i, (train_idx, val_idx) in enumerate(folds):
                if _ck.done(model=model_name, side=side, fold=fold_i):
                    rec = _ck.get(model=model_name, side=side, fold=fold_i)
                    # Its rule_missing entries were saved with it.
                    rule_missing.extend(rec.get('rule_missing') or [])
                    print(f"        Fold {fold_i}: read back")
                else:
                    rec = fold_fn(fold_i, train_idx, val_idx)
                    _ck.put(rec, model=model_name, side=side, fold=fold_i)
                recs.append(rec)
                s = rec['held_out']
                _b = rec['base_reps'][0] if rec['base_reps'] else None
                if side == 'target':
                    print(f"        Fold {fold_i}: "
                          f"base={s['base_best']:.4f} ({_b}), "
                          f"native={s['native']:.4f}, "
                          f"recipe={s['recipe_transfer']:.4f}, "
                          f"full={s['full_transfer']:.4f}")
                else:
                    print(f"        Fold {fold_i}: "
                          f"base={s['base_best']:.4f} ({_b}), "
                          f"hybrid={s['hybrid']:.4f}")
            return recs

        tgt_recs = _run_folds('target', target_folds, _target_fold,
                              target_name)
        src_recs = _run_folds('source', source_folds, _source_fold,
                              source_name)

        tgt_fold_scores = {k: [r['held_out'][k] for r in tgt_recs]
                           for k in (tgt_recs[0]['held_out'] if tgt_recs
                                     else {})}
        src_fold_scores = {k: [r['held_out'][k] for r in src_recs]
                           for k in (src_recs[0]['held_out'] if src_recs
                                     else {})}

        # ── Aggregate for this model ────────────────────────────────────
        # One fold (--mode intermediary, or --max-folds 1) is a single
        # held-out split: there is no spread, and 'std' is nan with a note
        # saying so instead of nanstd's 0.0.
        def _agg(scores):
            fin = [s for s in scores if np.isfinite(s)]
            out = {'mean': float(np.mean(fin)) if fin else float('nan'),
                   'n_folds_finite': len(fin),
                   'per_fold': scores}
            if len(fin) >= 2:
                out['std'] = float(np.std(fin))
            else:
                out['std'] = float('nan')
                out['spread_note'] = ('one fold, no spread' if len(fin) == 1
                                      else 'no finite fold')
            return out

        model_res = {
            'target': {},
            'source': {k: _agg(v) for k, v in src_fold_scores.items()},
            # The recipes transferred to the target. Chosen over every source
            # row, which is what recipe_transfer and full_transfer are meant to
            # test; no target row is in that search. search_score is each
            # recipe's mean over that search's inner folds.
            'source_blueprint': _bp_record(
                source_bp, source_slots[0][2], 'none', 'whole_source_dataset',
                source_slots[0][3]),
            'source_blueprints_by_slot': {
                _slot: _bp_record(_bp, _lab, 'none', 'whole_source_dataset',
                                  _ss)
                for _slot, _bp, _lab, _ss in source_slots},
            'source_recipes_skipped_identical_features':
                source_skipped_identical,
            # Why a source recipe is missing or failed to build, for this
            # model; read back with the block (2026-10-05).
            'source_rule_missing': source_rule_missing,
            'metric': metric_name,
            'lower_is_better': lower_is_better,
            'rule': (f'standard selection per fold: top {STANDARD_N_BASE} '
                     f'single representations and top {STANDARD_N_HYBRIDS} '
                     f'hybrids; top {STANDARD_N_HYBRIDS} source recipes '
                     f'with distinct selected features transferred'),
        }
        for cond, folds in tgt_fold_scores.items():
            model_res['target'][cond] = _agg(folds)
        # What each held-out fold actually used, picked on its training rows,
        # with its training-row and held-out scores (2026-10-05).
        model_res['target']['per_fold_records'] = tgt_recs
        model_res['target']['per_fold_base_rep'] = [
            r['base_reps'] for r in tgt_recs]
        model_res['target']['per_fold_native_blueprint'] = [
            r['native_blueprints'] for r in tgt_recs]
        model_res['target']['selection_scope'] = 'fold_train_rows'
        model_res['source']['per_fold_records'] = src_recs
        model_res['source']['per_fold_base_rep'] = [
            r['base_reps'] for r in src_recs]
        model_res['source']['per_fold_hybrid_blueprint'] = [
            r['hybrid_blueprints'] for r in src_recs]
        model_res['source']['selection_scope'] = 'fold_train_rows'

        # Deltas vs the target's best single representation, subtracted fold by
        # fold. A fold whose arm raised holds a nan and the two arms do not
        # lose the same folds, so differencing two nanmeans could take a
        # 4-fold mean off a 5-fold one. Only fold positions where BOTH arms
        # are finite enter the difference. delta_n_folds is how many folds are
        # behind the number and delta_std their spread — the win threshold
        # below is 0.001 RMSE, far under that spread, so both have to be
        # readable next to the delta. With one fold delta_std is nan and
        # delta_spread_note says "one fold, no spread".
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
                float(np.std(diffs)) if len(diffs) >= 2 else float('nan'))
            if len(diffs) < 2:
                model_res['target'][cond]['delta_spread_note'] = (
                    'one fold, no spread' if diffs else 'no shared finite fold')

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
    if len(target_folds) == 1:
        print(f"  One held-out fold: sd is one fold, no spread.")

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
    # delta_std, n_recipe_wins or n_full_wins.
    #
    # Since 2026-10-04 it is written for the SOURCE search only: the search
    # over every source row, the one the transferred recipes come from. The
    # same block for the target came from a search over every target row,
    # held-out folds included, and that search is gone.
    #
    # Every call reads rows find_best_hybrid already stored, so this costs 0
    # new model fits.
    def _top2_for(search_obj, which, ds_name):
        """The top-2 grid and the two deltas for one search.

        Always returns a block. Until 2026-09-28 it returned None when only
        'dataset' and 'which_search' got in; it now always carries
        'top2_models_compared' and 'top2_models_dropped' as well, which name the
        downstream models asked for and why each missing one is missing, and
        those reasons used to exist only in the printed log.
        """
        block = {'dataset': ds_name, 'which_search': which,
                 'selection_scope': 'whole_source_dataset'}
        ctx = f'opig_blueprint_transfer/{which}={ds_name}'
        # WHICH DOWNSTREAM MODELS THIS BLOCK COVERS, resolved 2026-09-28.
        # `model_names` is the decided models both datasets share. What a
        # search on disk actually scored can be shorter: svm drops itself above
        # 10,000 rows. The loops below then asked for a model with no stored
        # rows.
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
            # models=model_names, the decided models both datasets share,
            # not _shared.py's 5-name default.
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
        # Both are in this run's metric, metric_name, which the search was
        # given; delta_vs_base above is the same metric measured a
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

    # Keyed 'source'. Nothing in this file or in the summary tables above
    # iterates it, and per_model is untouched — a non-model key there would
    # break the summary loops, each of which reads res['source'] and
    # res['target'] off every value.
    top2_selection = {'source': _top2_for(source_search, 'source',
                                          source_name)}

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
        'models': list(model_names),
        # Every target arm and the source-side check were picked inside each
        # held-out fold; the transferred recipes over every source row.
        'selection_scope': {'target_arms': 'fold_train_rows',
                            'source_check_arms': 'fold_train_rows',
                            'transferred_recipes': 'whole_source_dataset'},
        'decided_grid': {
            'source': decided_search(source_name, 'general'),
            'target': decided_search(target_name, 'general')},
        'per_model': convert_numpy(per_model),
        'aggregate': {
            'mean_delta_recipe': mean_d_recipe,
            'mean_delta_full': mean_d_full,
            'n_recipe_wins': n_recipe_win,
            'n_full_wins': n_full_win,
            'n_models': n_models,
        },
    }
    # convert_numpy runs over the whole dict inside
    # save_experiment_results (_shared.py:7966) and turns each stored
    # Blueprint into its to_dict form, so the block is JSON-serialisable.
    if top2_selection:
        results['top2_selection'] = convert_numpy(top2_selection)
    # Every slot the owner's set asked for and did not get, with the reason.
    results['rule'] = (
        f'per downstream model, per held-out fold, both datasets: the top '
        f'{STANDARD_N_BASE} single representations and the top '
        f'{STANDARD_N_HYBRIDS} hybrids; the source\'s top '
        f'{STANDARD_N_HYBRIDS} recipes transferred to the target')
    results['rule_missing'] = rule_missing
    # Every per-fold search that failed (dataset, fold, model, error); the
    # arms that needed it are nan for that fold (2026-10-05).
    results['search_failures'] = search_failures
    results['pruning'] = 'off'
    results['n_folds_scored'] = {'target': len(target_folds),
                                 'source': len(source_folds)}

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
                        help='full (the configured outer fold count, default) or intermediary (one held-out split)')
    parser.add_argument('--reps', type=str, default=None,
                        help='Comma-separated rep list (default: intersection of DATASET_REPS)')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated downstream model list')
    parser.add_argument('--max-folds', type=int, default=None,
                        help='Score only the first N held-out folds of the '
                             'split (the split itself is unchanged)')
    # Accepted so older launch lines still parse; ignored since 2026-10-05,
    # when pruning was turned off in every search.
    parser.add_argument('--pruning-mode', type=str, default=None,
                        help="Ignored: every search runs unpruned (2026-10-05).")
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
                  pruning_mode=args.pruning_mode,
                  requested_models=([m.strip() for m in args.models.split(',')]
                                    if args.models else None))
    save_experiment_results(results, args.output,
                           'opig_blueprint_transfer',
                           f'{args.source}_to_{args.target}')
    # Written first, then a failed run exits 1 (2026-09-25).
    if isinstance(results, dict) and 'error' in results:
        print(f"FAILED: {results['error']}")
        sys.exit(1)
