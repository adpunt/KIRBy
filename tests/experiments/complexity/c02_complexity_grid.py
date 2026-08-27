#!/usr/bin/env python3
"""C02: Model-budget complexity grid -- tests optimal pairing hypothesis (CORE).

Paper: Complexity
Conclusion: Different models have different optimal representation budgets.
Dependencies: None (standalone).

Central experiment. Delegates core grid search to find_best_hybrid() with
configurable models, budgets, allocations, and importance methods.
Also computes PCA baselines, builds model x budget heatmap, and tracks
importance method wins.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split, _cv_split,
    find_best_hybrid, get_domain_metric,
    extract_top3_base_reps,
    build_honest_rescore_input,
    _build_rep_factories,
    INTERNAL_EVALUATORS, FULL_BUDGETS,
)
from kirby.find_best_hybrid import _predict_for_metric


def run(data, config, output_dir, dataset_name, **kwargs):
    """Run model x budget complexity grid via find_best_hybrid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C02: COMPLEXITY MATCHING GRID")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    models = get_model_factories(config['models'],
                                    task_type=data['task_type'],
                                    n_samples=len(data['labels']))
    # PCA baselines and the heatmap iterate the SAME full grid as the hybrid
    # search; using config['budgets'] (SCREENING_BUDGETS=[50,500,1000,2000])
    # left half the heatmap columns without PCA baseline entries.
    budgets = list(FULL_BUDGETS)
    allocation_methods = config['allocations']
    importance_methods = config['importance_methods']
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # Use FULL_BUDGETS so cache keys align with precompute (which always
    # runs with FULL_BUDGETS). config['budgets'] may be
    # SCREENING_BUDGETS for survey-style configs; we ignore it here.
    full_budgets = list(FULL_BUDGETS)

    # Model factories kept for PCA baselines + heatmap iteration below.
    # find_best_hybrid is called with models=None so it uses speed-tier
    # defaults (matches precompute's cache keys).
    model_instances = {name: factory() for name, factory in models.items()}

    augs = data.get('augmentations')
    from experiments.complexity.complexity_helpers import DATASET_REPS

    # --- All reps, no augmentation ---
    print(f"\n  --- All reps, no augmentation ({len(reps)} reps) ---")
    print(f"  {len(models)} models x {len(full_budgets)} budgets x "
          f"{len(allocation_methods)} allocations x {len(importance_methods)} importance")

    # Pass models=None so model_names normalizes to the speed-tier default
    # (8 models: gp, knn, lightgbm, mlp, rf, ridge, svm, xgboost). This is
    # what precompute uses — any other value causes a search-cache meta
    # drift on 'model_names' and triggers a full recompute. PCA baselines
    # and the model×budget heatmap below still use all 11 models via the
    # separate `models` / `model_instances` dicts.
    search_no_aug = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        models=None,
        budgets=full_budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        n_folds=n_folds,
        augmentations=None,
        augmentation_strategy='none',
        internal_evaluators=INTERNAL_EVALUATORS,
        metric=metric_name,
        task_type=data['task_type'],
        verbose=True,
        cache=True,
        dataset_name=dataset_name,
        tier='general',
    )

    # --- All reps, uncapped augmentation ---
    search_uncapped = search_no_aug  # fallback if no augmentations
    if not augs:
        print("  WARNING: No augmentation reps available — uncapped/capped "
              "conditions will be identical to no-augmentation.")
    if augs:
        print(f"\n  --- Uncapped augmentation ({len(augs)} aug reps) ---")
        search_uncapped = find_best_hybrid(
            base_reps=reps,
            labels=labels,
            groups=groups,
            models=None,
            budgets=full_budgets,
            allocation_methods=allocation_methods,
            importance_methods=importance_methods,
            n_folds=n_folds,
            augmentations=augs,
            augmentation_strategy='uncapped',
            internal_evaluators=INTERNAL_EVALUATORS,
            metric=metric_name,
            task_type=data['task_type'],
            verbose=True,
            cache=True,
            dataset_name=dataset_name,
            tier='general',
        )

    # --- Capped augmentation (on the 10-rep pool) ---
    # capped used to run on the retired 'top' subset; the pool is now that same
    # 10-rep set, so capped runs on it directly under tier='general'.
    # The 250-row skip was REMOVED 2026-08-27. It was added on 2026-06-01
    # because the OLD capped refitted a model and ran a permutation-importance
    # pass for every candidate feature. The rebuilt capped (2026-08-25) costs
    # one ranking fit, one base-ranking fit and five scoring fits per fold.
    # precompute_hybrids.py had the same rule and it went at the same time --
    # the two must stay in step, or the cache misses and this cold-builds.
    search_capped = None
    n_samples = len(labels)
    if augs and len(reps) >= 2:
        if True:
            print(f"\n  --- Capped augmentation "
                  f"({len(reps)} reps, {len(augs)} aug reps) ---")
            search_capped = find_best_hybrid(
                base_reps=reps,
                labels=labels,
                groups=groups,
                models=None,
                budgets=full_budgets,
                allocation_methods=allocation_methods,
                importance_methods=importance_methods,
                n_folds=n_folds,
                augmentations=augs,
                augmentation_strategy='capped',
                internal_evaluators=INTERNAL_EVALUATORS,
                metric=metric_name,
                task_type=data['task_type'],
                verbose=True,
                cache=True,
                dataset_name=dataset_name,
                tier='general',
            )

    # --- Frozen + FT comparison (fifth search pass) ---
    # Compares hybrid(frozen_only) vs hybrid(frozen+FT) by feeding the same
    # rep pool plus factory callables for the per-fold FT embeddings. The
    # factories hit the $KIRBY_CACHE_DIR/finetuned_reps/ cache populated by
    # precompute_ft_reps.py. Empty on datasets with no 'ft_reps' entry.
    search_ft = None
    ft_factories = _build_rep_factories(dataset_name, data)
    if ft_factories:
        # Stable id so the search cache for FT doesn't collide with non-FT
        # calls at the same tier. Encodes the set of FT rep names + base
        # models so adding/removing FT reps invalidates the cache.
        import hashlib
        ft_id_parts = sorted(
            f"{k}:{DATASET_REPS[dataset_name]['ft_reps'][k]['base_model']}"
            for k in ft_factories.keys()
        )
        ft_id = hashlib.sha256('|'.join(ft_id_parts).encode()).hexdigest()[:12]

        print(f"\n  --- Frozen + FT ({len(reps)} frozen + "
              f"{len(ft_factories)} FT: {', '.join(sorted(ft_factories))}) ---")
        search_ft = find_best_hybrid(
            base_reps=reps,
            labels=labels,
            groups=groups,
            models=None,
            budgets=full_budgets,
            allocation_methods=allocation_methods,
            importance_methods=importance_methods,
            n_folds=n_folds,
            augmentations=None,
            augmentation_strategy='none',
            internal_evaluators=INTERNAL_EVALUATORS,
            metric=metric_name,
            task_type=data['task_type'],
            verbose=True,
            cache=True,
            dataset_name=dataset_name,
            tier='general_ft',
            rep_factories=ft_factories,
            rep_factories_id=ft_id,
        )

    # Use uncapped as the primary search (best overall)
    search = search_uncapped

    top3 = extract_top3_base_reps(search)
    if top3:
        print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")

    # --- Reshape SearchResult to downstream format ---
    metric_label = search.metric_name

    # Configs: fields for Exp 9a ANOVA compatibility
    configs = []
    for r in search.all_results:
        configs.append({
            'model': r['model'],
            'allocation': r['allocation_method'],
            'importance': r['importance_method'],
            'budget': r['budget'],
            'score_mean': r['score_mean'],
            'score_std': r['score_std'],
            'fold_scores': r.get('fold_scores', []),
            'n_features': r['n_features'],
        })

    # Best by model — restrict to configured downstream models. The library's
    # search may include extra evaluators (e.g. knn, gp) via speed-tier
    # defaults; baselines/heatmap/per-model loops below all iterate `models`,
    # so keep this aligned to avoid KeyError downstream.
    best_by_model = {}
    for model_name in models:
        r = search.best_by_model.get(model_name)
        if r is None:
            continue
        best_by_model[model_name] = {
            'model': model_name,
            'allocation': r['allocation_method'],
            'importance': r['importance_method'],
            'budget': r['budget'],
            'score_mean': r['score_mean'],
            'score_std': r['score_std'],
            'n_features': r['n_features'],
        }

    # Baselines: reshape library format {rep: {model: score}} to Exp 2 format
    baselines = {}
    lib_baselines = search.baselines
    _lower = search.lower_is_better
    for model_name in models:
        best_single_rep = None
        best_single_score = np.inf if _lower else -np.inf
        for rep_name, rep_scores in lib_baselines.items():
            rep_score = rep_scores.get(model_name, np.inf if _lower else -np.inf)
            if (_lower and rep_score < best_single_score) or \
               (not _lower and rep_score > best_single_score):
                best_single_score = rep_score
                best_single_rep = rep_name
        # If no rep produced a finite score for this model, best_single_score is
        # still its ±inf starting sentinel and best_single_rep is None. Don't
        # write that as if it were a real baseline — a `-inf best score` polluted
        # the "top existing combos" ranking on the nucleotide datasets (e.g.
        # xgboost went all-NaN there). Record None + the reason instead.
        all_reps_failed = best_single_rep is None or not np.isfinite(best_single_score)
        baselines[model_name] = {
            'best_single_rep': None if all_reps_failed else best_single_rep,
            'best_single_score': None if all_reps_failed else float(best_single_score),
            'all_reps_failed': all_reps_failed,
        }
        if all_reps_failed:
            print(f"  {model_name}: best_rep=None (all reps failed — no finite score)")
        else:
            print(f"  {model_name}: best_rep={best_single_rep} {metric_label}={best_single_score:.4f}")

    # Per-(dataset, downstream model) hybrid at the model's best-baseline-rep
    # native dim. Direct apples-to-apples comparison vs the model's best
    # single-rep baseline (which _evaluate_single computes on the rep's full
    # native dim, e.g. ablang2 at 480D). For each model M:
    #   - find M's best baseline rep R and its dim D
    #   - run find_best_hybrid restricted to {M} at budget=[D]
    #   - allow_budget_exceeds_n=True so D > n_samples is permitted
    # Each per-model call has its own cache namespace via condition_suffix=
    # f'top_rep_dim_{model}'. Per-fold scores land in the standard
    # hybrid_eval cache, keyed by (subset_hash, alloc, imp, eval, budget,
    # fold, model, aug) — re-runs hit cache.
    search_per_model_top_rep = {}  # model_name -> SearchResult
    # Budgets the main search already evaluated for every model, so a per-model
    # top-rep-dim pass at one of these would be redundant. This is the explicit
    # full grid (full_budgets) PLUS len(labels): find_best_hybrid always appends
    # n_samples as an at-n boundary point, so budget==n is already covered. (Not
    # related to the allow_budget_exceeds_n path — that lets a SINGLE budget D
    # exceed n; this set just avoids re-running budgets the main grid covered.)
    skipped_already_tested = set(full_budgets) | {len(labels)}
    # GP and KNN are numerically unreliable at top_rep_dim budgets (≥1280D
    # ESM-style reps): GP's RBF kernel matrix degenerates and KNN hits curse
    # of dim, both producing all-NaN configs. Their best_single_rep pick at
    # that dim is itself noisy. Skip them entirely from this pass — the
    # remaining models (rf/xgboost/ridge/svm/mlp/lightgbm) cover the policy
    # question without false-positive empty caches.
    _PER_MODEL_TOP_REP_SKIP = {'gp', 'knn'}
    print("\n  --- Per-model hybrid at top-baseline-rep dim ---")
    for _model_name in models:
        if _model_name in _PER_MODEL_TOP_REP_SKIP:
            print(f"    {_model_name}: skipped (unreliable at high-dim budgets)")
            continue
        _info = baselines.get(_model_name)
        if not _info:
            continue
        _top_rep = _info.get('best_single_rep')
        if not _top_rep or _top_rep not in reps:
            continue
        _top_dim = reps[_top_rep].shape[1]
        if _top_dim in skipped_already_tested:
            # Main search already covers this budget for this model.
            continue
        try:
            _s = find_best_hybrid(
                base_reps=reps,
                labels=labels,
                groups=groups,
                models={_model_name: model_instances[_model_name]},
                budgets=[_top_dim],
                allocation_methods=allocation_methods,
                importance_methods=importance_methods,
                n_folds=n_folds,
                augmentations=None,
                augmentation_strategy='none',
                internal_evaluators=INTERNAL_EVALUATORS,
                metric=metric_name,
                task_type=data['task_type'],
                verbose=False,
                cache=True,
                dataset_name=dataset_name,
                tier='general',
                condition_suffix=f'top_rep_dim_{_model_name}',
                allow_budget_exceeds_n=True,
            )
            search_per_model_top_rep[_model_name] = _s
            if _model_name in _s.best_by_model:
                _r = _s.best_by_model[_model_name]
                _baseline = _info['best_single_score']
                _delta = _r['score_mean'] - _baseline
                print(f"    {_model_name}: hybrid@{_top_dim}D ({_top_rep}) "
                      f"-> {metric_label}={_r['score_mean']:.4f} "
                      f"(vs {_top_rep} {_baseline:.4f}, delta={_delta:+.4f})")
        except Exception as _e:
            print(f"    {_model_name}: failed — {_e}")

    # PCA baselines (library doesn't compute these -- keep manual)
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    # Some reps contain NaN (e.g. BLOSUM on sequences with unknown residues,
    # protein LM embeddings on edge cases). PCA / StandardScaler can't handle
    # them, so impute before the loop. Matches _evaluate_allocation's policy
    # in hybrid.py:1124 — nan->0, inf->clip.
    X_concat = np.clip(X_concat, -1e10, 1e10)
    X_concat = np.nan_to_num(X_concat, nan=0.0, posinf=1e10, neginf=-1e10)
    print("\n  Computing PCA baselines...")
    pca_baselines = {}
    for budget in budgets:
        pca_baselines[budget] = {}
        for model_name, model_factory in models.items():
            pca_scores = []
            for train_idx, val_idx in cv_split(groups, n_folds, X_concat, labels, mode=mode):
                # Cap n_components at per-fold train size, not the full dataset.
                # With CV splits, train is a strict subset and PCA's
                # n_components constraint is against the train slice.
                n_train = len(train_idx)
                n_comp = min(budget, X_concat.shape[1], n_train - 1)
                if n_comp < 1:
                    continue
                pca = PCA(n_components=n_comp)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_concat[train_idx])
                X_val_s = scaler.transform(X_concat[val_idx])
                X_train_p = pca.fit_transform(X_train_s)
                X_val_p = pca.transform(X_val_s)
                m = model_factory()
                m.fit(X_train_p, labels[train_idx])
                # Use _predict_for_metric so AUC/probabilistic metrics get
                # predict_proba — m.predict() returns class labels and gives a
                # meaningless AUROC on classification datasets.
                pca_scores.append(metric_fn(
                    labels[val_idx],
                    _predict_for_metric(m, X_val_p, metric_fn)))
            pca_baselines[budget][model_name] = (
                float(np.mean(pca_scores)) if pca_scores else float('nan'))

    # Heatmap: model x budget -> best score
    heatmap = {}
    for model_name in models:
        heatmap[model_name] = {}
        model_configs = [c for c in configs if c['model'] == model_name]
        for budget in full_budgets:
            budget_configs = [c for c in model_configs if c['budget'] == budget]
            if budget_configs:
                best = (min if lower_is_better else max)(budget_configs, key=lambda c: c['score_mean'])
                heatmap[model_name][budget] = best['score_mean']

    # Importance method wins
    importance_wins = {imp: 0 for imp in importance_methods}
    for model_name, best in best_by_model.items():
        imp = best['importance']
        if imp in importance_wins:
            importance_wins[imp] += 1

    # Print summary
    print(f"\n  BEST BY MODEL (metric: {metric_label}):")
    for model_name, best in best_by_model.items():
        baseline_score = baselines[model_name]['best_single_score']
        baseline_rep = baselines[model_name]['best_single_rep']
        # baseline_score is None when every rep failed for this model (see the
        # all_reps_failed guard above) — no valid delta to report in that case.
        if baseline_score is None:
            print(f"  >> {model_name}: {best['allocation']}/{best['importance']}"
                  f"/b={best['budget']} -> {metric_label}={best['score_mean']:.4f} "
                  f"(no valid single-rep baseline — all reps failed)")
            continue
        delta = best['score_mean'] - baseline_score
        print(f"  >> {model_name}: {best['allocation']}/{best['importance']}"
              f"/b={best['budget']} -> {metric_label}={best['score_mean']:.4f} "
              f"(vs {baseline_rep} {baseline_score:.4f}, delta={delta:+.4f})")

    print("\n  IMPORTANCE METHOD WINS:")
    for imp, count in sorted(importance_wins.items(), key=lambda x: -x[1]):
        print(f"    {imp}: {count}/{len(models)}")

    # Heatmap table
    print(f"\n  HEATMAP (model x budget -> best {metric_label}):")
    budget_strs = [str(b) for b in full_budgets]
    header = f"  {'Model':<12}" + "".join(f"{b:>8}" for b in budget_strs)
    print(header)
    print("  " + "-" * (12 + 8 * len(full_budgets)))
    for model_name in models:
        row = f"  {model_name:<12}"
        for budget in full_budgets:
            val = heatmap.get(model_name, {}).get(budget)
            row += f"{val:>8.4f}" if val is not None else f"{'--':>8}"
        print(row)

    # --- Condition comparison summary ---
    def _extract_best(search_obj):
        if search_obj is None:
            return {}
        out = {}
        for model_name in models:
            r = search_obj.best_by_model.get(model_name)
            if r is None:
                continue
            out[model_name] = {
                'model': model_name,
                'allocation': r['allocation_method'],
                'importance': r['importance_method'],
                'budget': r['budget'],
                'score_mean': r['score_mean'],
                'score_std': r['score_std'],
                'n_features': r['n_features'],
            }
        return out

    conditions = {
        'no_aug': _extract_best(search_no_aug),
        'uncapped': _extract_best(search_uncapped),
    }
    if search_capped is not None:
        conditions['capped'] = _extract_best(search_capped)
    if search_ft is not None:
        conditions['frozen_plus_ft'] = _extract_best(search_ft)
    # Per-model hybrid at top-rep dim is a distinct condition: each model
    # has its own SearchResult restricted to that single model + budget.
    # Stitch them into one model->config dict for the comparison table.
    if search_per_model_top_rep:
        per_model_at_top = {}
        for _mname, _s in search_per_model_top_rep.items():
            if _mname in _s.best_by_model:
                _r = _s.best_by_model[_mname]
                per_model_at_top[_mname] = {
                    'model': _mname,
                    'allocation': _r['allocation_method'],
                    'importance': _r['importance_method'],
                    'budget': _r['budget'],
                    'score_mean': _r['score_mean'],
                    'score_std': _r['score_std'],
                    'n_features': _r['n_features'],
                }
        if per_model_at_top:
            conditions['per_model_top_rep_dim'] = per_model_at_top

    # Print condition comparison
    print("\n  CONDITION COMPARISON (best score per model):")
    for model_name in models:
        parts = []
        for cond_name, cond_best in conditions.items():
            if model_name in cond_best:
                score = cond_best[model_name]['score_mean']
                parts.append(f"{cond_name}={score:.4f}")
        if parts:
            print(f"    {model_name}: {', '.join(parts)}")

    # Best per (dataset, model) across conditions: for each model, pick the
    # condition with the highest score (or lowest if lower_is_better). This
    # is the "best hybrid for this model on this dataset" answer the user
    # cares about — disambiguates from "best single rep baseline" and from
    # "best within one aug condition".
    best_per_model_across_conditions = {}
    print("\n  BEST PER MODEL ACROSS CONDITIONS:")
    for model_name in models:
        candidates = []
        for cond_name, cond_best in conditions.items():
            if model_name in cond_best:
                candidates.append((cond_name, cond_best[model_name]))
        if not candidates:
            continue
        if lower_is_better:
            cond_name, best_entry = min(candidates,
                                        key=lambda kv: kv[1]['score_mean'])
        else:
            cond_name, best_entry = max(candidates,
                                        key=lambda kv: kv[1]['score_mean'])
        best_per_model_across_conditions[model_name] = {
            'condition': cond_name,
            **best_entry,
        }
        print(f"    {model_name}: {cond_name} "
              f"({best_entry['allocation']}/{best_entry['importance']}"
              f"/b={best_entry['budget']}) "
              f"-> {metric_label}={best_entry['score_mean']:.4f}")

    # Native width of each base representation. Needed downstream to show
    # "hybrid beats the single rep while using fewer features" — the analysis
    # can't state that without each rep's own dimension. Widths are already
    # known from the loaded rep matrices, so this only affects what's written,
    # not what's computed (reuses all precomputed work). Key name matches the
    # `rep_dims` convention in make_structured_output (_shared.py).
    rep_dims = {name: int(mat.shape[1]) for name, mat in reps.items()}

    # Per-fold scores for BOTH arms (hybrid candidates + single-rep baselines),
    # keyed per model, so the reader can re-pick the per-fold winner and compute
    # an honest hybrid-vs-best-single delta instead of comparing a reported-mean
    # hybrid against an argmax-over-means baseline. build_honest_rescore_input
    # reads candidate fold_scores from search.all_results and per-fold baselines
    # from search.baseline_fold_scores (additive fields; empty on old caches ->
    # that model simply won't appear, tolerated by the helper). Built from the
    # primary (uncapped) search, matching best_by_model/baselines above.
    honest_rescore_block = build_honest_rescore_input(search)

    return {
        'configs': configs,
        'honest_rescore': honest_rescore_block,
        'best_by_model': best_by_model,
        'baselines': baselines,
        'rep_dims': rep_dims,
        'pca_baselines': pca_baselines,
        'heatmap': heatmap,
        'importance_wins': importance_wins,
        'conditions': conditions,
        'best_per_model_across_conditions': best_per_model_across_conditions,
        'has_augmentation': augs is not None,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
    }


if __name__ == '__main__':
    parser = create_experiment_parser('C02: Complexity Matching Grid',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Models: {config['models']}")
    print(f"Budgets: {config['budgets']}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    # Stamp the known metric direction so it survives even for datasets not in
    # DATASET_REGISTRY (save_experiment_results otherwise re-derives it via a
    # registry lookup that silently stamps nothing on a miss).
    _mname, _, _lower = get_domain_metric(data)
    save_experiment_results(results, args.output, 'c02_complexity_matching_grid',
                            args.dataset,
                            metric_name=_mname, lower_is_better=_lower)
