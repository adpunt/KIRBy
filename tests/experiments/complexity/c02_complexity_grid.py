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
    get_model_factories, cv_split, _cv_split, DOWNSTREAM_MODELS,
    find_best_hybrid, get_domain_metric,
    extract_top3_base_reps,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    resolve_comparison_models,
    build_fold_reselection_input, fold_reselection,
    _build_rep_factories, parse_rep_args,
    INTERNAL_EVALUATORS, FULL_BUDGETS,
    hr_exclude_folds,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS,
    RULE_N_GENERAL, RULE_N_DISTANCE,
)
from kirby.find_best_hybrid import _predict_for_metric


def _rule_from_searches(gen_search, dist_search, model, budget=None, *,
                        dist_error=None):
    """Owner's set (2026-10-02): top 2 general + top 1 distance hybrids for
    one downstream model, read off this file's own two searches.

    Returns (entries, missing). Each entry is a top_hybrids 'selected' entry
    plus 'slot' ('general1', 'general2', 'distance1'), 'lineage', 'tier',
    'objective' and 'budget_relaxed'. Every slot asked for and not filled is
    a sentence in `missing`.

    A distance search scores budgets up to max(DISTANCE_BUDGETS) features
    only. Asked for a larger budget, the distance read ranks over every budget
    it holds and the entry says budget_relaxed=True.
    """
    entries, missing = [], []
    for lineage, srch, n, tier in (
            ('general', gen_search, RULE_N_GENERAL, 'general'),
            ('distance', dist_search, RULE_N_DISTANCE, 'general_distance')):
        relaxed = False
        b = budget
        if lineage == 'distance':
            if model not in DISTANCE_MODELS:
                missing.append(f'distance for {model}: {model} is not one of '
                               f'the distance search models {DISTANCE_MODELS}, '
                               f'so it was never scored there')
                continue
            if srch is None:
                missing.append(f'distance for {model}: no distance search '
                               f'({dist_error or "not run"})')
                continue
            if b is not None and b > max(DISTANCE_BUDGETS):
                b, relaxed = None, True
        if srch is None:
            missing.append(f'{lineage} for {model}: no search')
            continue
        try:
            th = top_hybrids(srch, model, n, budget=b, require_full=False)
        except Exception as e:
            missing.append(f'{lineage} for {model}: top_hybrids raised '
                           f'{type(e).__name__}: {e}')
            continue
        if not th['complete']:
            missing.append(f'{lineage} for {model}: {th["shortfall_reason"]}')
        for e in th['selected']:
            e = dict(e)
            e['slot'] = f"{lineage}{e['rank']}"
            e['lineage'] = lineage
            e['tier'] = tier
            e['objective'] = 'predictive'
            e['budget_asked'] = budget
            e['budget_relaxed'] = relaxed
            entries.append(e)
    return entries, missing


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
    # search, so both read FULL_BUDGETS directly rather than config['budgets'].
    # A shorter budget list left half the heatmap columns without PCA
    # baseline entries.
    budgets = list(FULL_BUDGETS)
    allocation_methods = config['allocations']
    importance_methods = config['importance_methods']
    # THE 12 FAST-SPEED DATASETS (dataset_policy.FAST_SPEED_DATASETS) were
    # precomputed on the fast grid: 3 allocation methods, lightgbm_gain only,
    # the fast evaluators and representation counts, the five models named.
    # c02 passed the full lists itself, which beat speed= inside
    # find_best_hybrid, so on herg it asked for 8,160 configurations the
    # precompute never stored and was refused (server check 2026-09-29).
    # Owner's decision 2026-09-29: c02 follows the dataset's speed.
    from experiments.dataset_policy import dataset_search_speed as _ds_speed
    from kirby.find_best_hybrid import (
        FAST_ALLOCATION_METHODS as _FA, FAST_IMPORTANCE_METHODS as _FI,
        FAST_INTERNAL_EVALUATORS as _FE, FAST_REP_COUNTS as _FR)
    _c02_fast = _ds_speed(dataset_name) == 'fast'
    _search_models = list(DOWNSTREAM_MODELS) if _c02_fast else None
    # gp on the datasets under 1,000 molecules, as the precompute stores them
    # (dataset_policy.SMALL_DATASET_EXTRA_MODELS). c02 calls find_best_hybrid
    # directly, so _shared never added it and c02's search missed the stored
    # one on those datasets (model-list audit, 2026-09-29).
    from experiments._shared import _apply_small_dataset_models as _asm
    _search_models = _asm({'models': _search_models, 'tier': 'general'},
                          dataset_name)['models']
    _c02_evaluators = list(_FE) if _c02_fast else INTERNAL_EVALUATORS
    _c02_speed_kw = ({'speed': 'fast', 'rep_counts': list(_FR)}
                     if _c02_fast else {})
    if _c02_fast:
        allocation_methods = list(_FA)
        importance_methods = list(_FI)
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    # Use FULL_BUDGETS so cache keys align with precompute (which always
    # runs with FULL_BUDGETS). A caller may narrow config['budgets'];
    # we ignore it here.
    full_budgets = list(FULL_BUDGETS)

    # Model factories kept for PCA baselines + heatmap iteration below.
    # find_best_hybrid is called with models=None so it uses speed-tier
    # defaults (matches precompute's cache keys).
    model_instances = {name: factory() for name, factory in models.items()}

    augs = data.get('augmentations')
    # c02 builds its OWN searches, so KIRBY_LOCAL_AUG_CONDITIONS -- which cuts
    # the augmented conditions out of the shared precompute -- never reached it.
    # Measured 2026-09-20: c02 ran four searches and took 33,174 s on caco2, over
    # any per-experiment wall cap, so it could never write a result file and c08
    # and c09, which read c02's file, could never run either. Unset -- the
    # cluster -- nothing changes.
    from experiments._shared import _local_aug_conditions
    if augs and 'uncapped' not in _local_aug_conditions(['none', 'uncapped', 'capped']):
        print("  [local] KIRBY_LOCAL_AUG_CONDITIONS drops the uncapped and "
              "capped conditions; the augmentation-condition question has no "
              "answer in this run.")
        augs = None
    from experiments.complexity.complexity_helpers import DATASET_REPS

    # --- All reps, no augmentation ---
    print(f"\n  --- All reps, no augmentation ({len(reps)} reps) ---")
    print(f"  {len(models)} models x {len(full_budgets)} budgets x "
          f"{len(allocation_methods)} allocations x {len(importance_methods)} importance")

    # Pass models=None so model_names normalizes to whatever DEFAULT_MODELS
    # labels 'fast' or 'full' — rf, lightgbm, xgboost, svm, dnn since
    # 2026-09-04. This is what precompute uses — any other value causes a
    # search-cache meta drift on 'model_names' and triggers a full recompute.
    # PCA baselines and the model×budget heatmap below go through the separate
    # `models` / `model_instances` dicts, which get_config fills from
    # DOWNSTREAM_MODELS — the same five names, not the eleven this comment
    # used to claim (corrected 2026-09-10).
    search_no_aug = find_best_hybrid(
        base_reps=reps,
        labels=labels,
        groups=groups,
        models=_search_models,
        budgets=full_budgets,
        allocation_methods=allocation_methods,
        importance_methods=importance_methods,
        n_folds=n_folds,
        augmentations=None,
        augmentation_strategy='none',
        internal_evaluators=_c02_evaluators,
        **_c02_speed_kw,
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
            models=_search_models,
            budgets=full_budgets,
            allocation_methods=allocation_methods,
            importance_methods=importance_methods,
            n_folds=n_folds,
            augmentations=augs,
            augmentation_strategy='uncapped',
            internal_evaluators=_c02_evaluators,
            **_c02_speed_kw,
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
                models=_search_models,
                budgets=full_budgets,
                allocation_methods=allocation_methods,
                importance_methods=importance_methods,
                n_folds=n_folds,
                augmentations=augs,
                augmentation_strategy='capped',
                internal_evaluators=_c02_evaluators,
                **_c02_speed_kw,
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
    ft_search_error = None
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
        # A fine-tuned representation that will not build takes this arm down,
        # and nothing else. On caco2 lora_molformer raises ModuleNotFoundError:
        # No module named 'transformers.masking_utils' on all five folds, and
        # _precompute_fold_reps refuses the search rather than average folds
        # over different representation pools -- which is right. What was not
        # right is that the exception escaped run(), so caco2 lost the WHOLE of
        # c02: the no-augmentation, uncapped and capped searches had all
        # finished and not one of their numbers was written.
        try:
            search_ft = find_best_hybrid(
                base_reps=reps,
                labels=labels,
                groups=groups,
                models=_search_models,
                budgets=full_budgets,
                allocation_methods=allocation_methods,
                importance_methods=importance_methods,
                n_folds=n_folds,
                augmentations=None,
                augmentation_strategy='none',
                internal_evaluators=_c02_evaluators,
                **_c02_speed_kw,
                metric=metric_name,
                task_type=data['task_type'],
                verbose=True,
                cache=True,
                dataset_name=dataset_name,
                tier='general_ft',
                rep_factories=ft_factories,
                rep_factories_id=ft_id,
            )
        except Exception as _ft_exc:
            search_ft = None
            ft_search_error = f'{type(_ft_exc).__name__}: {_ft_exc}'
            print(f'  Frozen + FT arm FAILED, the other conditions stand: '
                  f'{ft_search_error}')

    # Use uncapped as the primary search (best overall)
    search = search_uncapped

    # Owner's set (2026-10-02): top 2 general + top 1 distance hybrids per
    # model, per family. c02 runs its own searches, so the distance
    # counterpart is its own search too: the primary condition above
    # (uncapped when augmentation representations exist, else none), run
    # with the distance settings precompute_hybrids.py uses for a distance
    # condition -- tier 'general_distance', DISTANCE_BUDGETS (10 to 500
    # features), DISTANCE_EVALUATORS, DISTANCE_MODELS. The rest of the
    # recipe grid is the same as the general searches here.
    _dist_cond = 'uncapped' if augs else 'none'
    search_distance = None
    distance_search_error = None
    _dist_models = _asm({'models': list(DISTANCE_MODELS),
                         'tier': 'general_distance'}, dataset_name)['models']
    print(f"\n  --- Distance counterpart ({_dist_cond}, "
          f"{len(DISTANCE_BUDGETS)} budgets up to {max(DISTANCE_BUDGETS)} "
          f"features) ---")
    try:
        search_distance = find_best_hybrid(
            base_reps=reps,
            labels=labels,
            groups=groups,
            models=_dist_models,
            budgets=list(DISTANCE_BUDGETS),
            allocation_methods=allocation_methods,
            importance_methods=importance_methods,
            n_folds=n_folds,
            augmentations=augs if _dist_cond != 'none' else None,
            augmentation_strategy=_dist_cond,
            internal_evaluators=list(DISTANCE_EVALUATORS),
            **_c02_speed_kw,
            metric=metric_name,
            task_type=data['task_type'],
            verbose=True,
            cache=True,
            dataset_name=dataset_name,
            tier='general_distance',
        )
    except Exception as _d_exc:
        distance_search_error = f'{type(_d_exc).__name__}: {_d_exc}'
        print(f'  Distance search FAILED, the general conditions stand: '
              f'{distance_search_error}')

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
        # The whole ranking, best first, by this model's own score in the
        # dataset's own metric. Added 2026-09-13 so the width-matched pass below
        # can run at the top TWO representations' widths, not just the best
        # one's — the owner's call: one width is one point, two is the start of
        # a curve.
        _ranked = [(rn, sc) for rn, sc in
                   ((r, v.get(model_name)) for r, v in lib_baselines.items())
                   if sc is not None and np.isfinite(sc)]
        # Score first, then representation name. ⛔ TIE-BREAK ADDED 2026-09-18,
        # owner's decision. Without the name this was `reverse=not _lower` on
        # the score alone, so two representations on the same score kept
        # whatever order lib_baselines iterated in — the order they were built
        # in, which is arbitrary and recorded nowhere. Baseline scores are
        # rounded to 4 decimals before storage (find_best_hybrid.py:5525), so
        # exact ties happen. The width pass below reads _ranked[:2], so a tie
        # decided which representation's width got measured.
        # The scores do not change. Only which of two equal names is used does,
        # and now it is the same name every run.
        _ranked.sort(key=lambda kv: ((kv[1] if _lower else -kv[1]), kv[0]))
        # Which representations tied with the rank-1 score, so a reader can see
        # that "the best base representation" was not a single thing here.
        _r1 = _ranked[0][1] if _ranked else None
        _tied_at_top = sorted(rn for rn, sc in _ranked
                              if _r1 is not None and sc == _r1)
        baselines[model_name] = {
            # Every representation on the rank-1 score, name-ordered. One entry
            # means no tie. Added 2026-09-18.
            'tied_at_best_single': _tied_at_top,
            'best_single_rep': None if all_reps_failed else best_single_rep,
            'best_single_score': None if all_reps_failed else float(best_single_score),
            'all_reps_failed': all_reps_failed,
            'ranked_reps': [rn for rn, _ in _ranked],
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
    # (model, rank) -> (representation, SearchResult). Added 2026-10-05: the
    # fold re-pick below used to merge every search in search_per_model_top_rep
    # by model name, so the rank-2 search replaced the rank-1 one for the same
    # model and only rank 2's held-out delta reached the file.
    search_per_model_top_rep_ranked = {}
    # Budgets the main search already evaluated for every model, so a per-model
    # top-rep-dim pass at one of these would be redundant: the explicit full
    # grid, and nothing else.
    #
    # len(labels) used to be in this set because find_best_hybrid appended
    # n_samples as an at-n boundary point whenever a requested budget exceeded
    # the row count. That stopped on 2026-09-10 (allow_budget_exceeds_n now
    # defaults to True and every budget asked for is searched), so a budget
    # equal to the molecule count is no longer covered by the main grid — and
    # keeping it here dropped any model whose best representation happens to be
    # exactly as wide as the dataset has rows, silently.
    skipped_already_tested = set(full_budgets)
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
        # THE TOP TWO representations, not just the best. Two of them give two
        # points on the budget curve instead of one, and the second-best is
        # often a very different width from the first. Owner's call 2026-09-13.
        # Ranked by this model's own score in the dataset's own metric.
        _top_reps = [r for r in (_info.get('ranked_reps') or []) if r in reps][:2]
        if not _top_reps:
            _fallback = _info.get('best_single_rep')
            _top_reps = [_fallback] if _fallback in reps else []
        # ── Two representations of the same WIDTH are one search ──────────
        # The call below is restricted to this one downstream model and to
        # budgets=[the representation's width]. Rank 1 and rank 2 differ in
        # exactly one argument, condition_suffix, and condition_suffix reaches
        # the cache filename, the tier label and the search meta and nothing
        # else (find_best_hybrid.py:8859-8860, 8974-8976). So when the two
        # representations are the same number of features wide, the second call
        # recomputes a bit-identical SearchResult under a second cache key.
        #
        # MEASURED on flab_thermo_tresanco (672 molecules, whole dataset, no row
        # dropped) on 2026-09-25, five representations, all 1000 features wide
        # under KIRBY_LOCAL_MAX_FEATURES_PER_REP=1000:
        #   rf  rank 1 saprot -> spearman 0.6258; rank 2 esmc -> spearman 0.6258
        #   svm rank 1 saprot -> spearman 0.6297; rank 2 esmc -> spearman 0.6297
        # Four searches here cost 198 s of c02's 298 s on that run and two of
        # the four were these duplicates.
        #
        # Not only a local-run effect: in the shipped ten-representation flab
        # pool, saprot, esm2_entropy_top20, esm2_cdr, esm2_early and
        # esm2_wt_delta are all 1280 features wide, so the cluster pays for the
        # same duplicate whenever two of those come first and second.
        #
        # ⛔ NOTHING IS CUT AND NO NUMBER CHANGES. The reused SearchResult is
        # the object the second call would have built. Only rank 2's cache entry
        # is not written, and the delta printed below is still taken against
        # rank 2's OWN baseline representation, which is the one quantity that
        # genuinely differs between the two ranks.
        _search_by_width = {}
        for _rank, _top_rep in enumerate(_top_reps, start=1):
            _top_dim = reps[_top_rep].shape[1]
            if _top_dim in skipped_already_tested:
                # Main search already covers this budget for this model.
                continue
            _reused = _search_by_width.get(_top_dim)
            try:
                _s = _reused if _reused is not None else find_best_hybrid(
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
                    internal_evaluators=_c02_evaluators,
                    **_c02_speed_kw,
                    metric=metric_name,
                    task_type=data['task_type'],
                    verbose=False,
                    cache=True,
                    dataset_name=dataset_name,
                    tier='general',
                    # The representation NAME, not just the model, so the two
                    # widths do not share a cache entry.
                    condition_suffix=f'top_rep_dim_{_model_name}_{_top_rep}',
                    allow_budget_exceeds_n=True,
                )
                _search_by_width.setdefault(_top_dim, _s)
                search_per_model_top_rep[f'{_model_name}__{_top_rep}'] = _s
                search_per_model_top_rep_ranked[(_model_name, _rank)] = (
                    _top_rep, _s)
                if _rank == 1:
                    # Keep the old key so every existing reader still finds the
                    # best representation's pass where it always was.
                    search_per_model_top_rep[_model_name] = _s
                if _model_name in _s.best_by_model:
                    _r = _s.best_by_model[_model_name]
                    # Against THIS representation's own score, not the best
                    # one's — at rank 2 the comparator is the rank-2
                    # representation, or the delta is between two different
                    # things.
                    _baseline = None
                    for _rn, _sc in ((r, v.get(_model_name))
                                     for r, v in search.baselines.items()):
                        if _rn == _top_rep and _sc is not None:
                            _baseline = float(_sc)
                    if _baseline is None:
                        _baseline = _info['best_single_score']
                    # Sign-normalised so + always means the hybrid won, on rmse
                    # as well as on r2/spearman/auc.
                    _delta = ((_baseline - _r['score_mean']) if lower_is_better
                              else (_r['score_mean'] - _baseline))
                    # Says so when the search was not run again, so a reader
                    # of the log can see why two ranks carry the same score.
                    _same = ('' if _reused is None else
                             f' [same {_top_dim}-feature search as an earlier '
                             f'representation of this model, not re-run]')
                    print(f"    {_model_name}: hybrid@{_top_dim}D "
                          f"(rank {_rank}: {_top_rep}) -> "
                          f"{metric_label}={_r['score_mean']:.4f} "
                          f"(vs {_top_rep} {_baseline:.4f}, "
                          f"delta={_delta:+.4f}){_same}")
            except Exception as _e:
                print(f"    {_model_name}: rank-{_rank} rep {_top_rep} "
                      f"failed — {_e}")

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
    # Added 2026-10-05, beside pca_baselines (whose {budget: {model: score}}
    # shape the analysis scripts read, so it is left as it was):
    #   pca_baseline_widths[budget][model] = PCA components fitted per fold.
    #     n_comp is min(budget, total features, training rows - 1), so every
    #     budget above the fold's training rows fits the same width and its
    #     score repeats; this shows which budgets those are.
    #   pca_baseline_fold_scores[budget][model] = the per-fold scores the mean
    #     is taken over, in the dataset's metric.
    pca_baseline_widths = {}
    pca_baseline_fold_scores = {}
    for budget in budgets:
        pca_baselines[budget] = {}
        pca_baseline_widths[budget] = {}
        pca_baseline_fold_scores[budget] = {}
        for model_name, model_factory in models.items():
            pca_scores = []
            pca_widths = []
            # task_type passed 2026-10-05 so these folds are built the way the
            # hybrid search builds its folds (StratifiedGroupKFold on a
            # classification dataset). Without it cv_split guessed the task
            # from whether the labels look like integers.
            for train_idx, val_idx in cv_split(groups, n_folds, X_concat, labels,
                                               mode=mode,
                                               task_type=data['task_type']):
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
                # 2026-10-05, owner delegated, 2026-10-05: each validation
                # feature is capped at the range that feature takes on the
                # scaled training rows, before the PCA projection. On adme_hlm
                # one validation molecule sat 83 million training standard
                # deviations out and the dnn's rmse ran to thousands.
                X_val_s = np.clip(X_val_s, X_train_s.min(0), X_train_s.max(0))
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
                pca_widths.append(int(n_comp))
            pca_baselines[budget][model_name] = (
                float(np.mean(pca_scores)) if pca_scores else float('nan'))
            pca_baseline_widths[budget][model_name] = pca_widths
            pca_baseline_fold_scores[budget][model_name] = [
                float(v) for v in pca_scores]

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

    # Print summary. The two sides are printed beside each other and NOT
    # subtracted here: the hybrid side is the best of 3,720 recipes and the
    # single-representation side the best of 10 representations, both maxima
    # taken on the same folds they are then reported on, so their difference
    # runs high. The delta to read is reselected_delta, printed below once the
    # per-fold re-pick has been done.
    print(f"\n  BEST BY MODEL (metric: {metric_label}):")
    for model_name, best in best_by_model.items():
        baseline_score = baselines[model_name]['best_single_score']
        baseline_rep = baselines[model_name]['best_single_rep']
        # baseline_score is None when every rep failed for this model (see the
        # all_reps_failed guard above).
        if baseline_score is None:
            print(f"  >> {model_name}: {best['allocation']}/{best['importance']}"
                  f"/b={best['budget']} -> {metric_label}={best['score_mean']:.4f} "
                  f"(no valid single-rep baseline — all reps failed)")
            continue
        print(f"  >> {model_name}: {best['allocation']}/{best['importance']}"
              f"/b={best['budget']} -> {metric_label}={best['score_mean']:.4f} "
              f"(best single rep {baseline_rep} {baseline_score:.4f})")

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
                # Per-fold scores of this condition's winning recipe. Without
                # them the argmax over conditions below is one number per
                # condition with nothing behind it — no error bar, and no way
                # to re-pick the winner on held-out folds.
                'fold_scores': list(r.get('fold_scores') or []),
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
                    'fold_scores': list(_r.get('fold_scores') or []),
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
    # hybrid against an argmax-over-means baseline. build_fold_reselection_input
    # reads candidate fold_scores from search.all_results and per-fold baselines
    # from search.baseline_fold_scores (additive fields; empty on old caches ->
    # that model simply won't appear, tolerated by the helper). Built from the
    # primary (uncapped) search, matching best_by_model/baselines above.
    fold_reselection_block = build_fold_reselection_input(search)

    # One block per condition as well, not the uncapped search alone. Each
    # condition ran its own search and so carries its own per-fold candidate and
    # baseline scores; building only the uncapped block left no-augmentation,
    # capped, frozen-plus-fine-tuned and per-model-top-rep-dim with a single
    # mean apiece and nothing to re-pick them on. Costs no fits.
    # per_model_top_rep_dim is one search per downstream model, each restricted
    # to that model, so their blocks merge by model key without overlapping.
    condition_searches = {'no_aug': search_no_aug, 'uncapped': search_uncapped}
    if search_capped is not None:
        condition_searches['capped'] = search_capped
    if search_ft is not None:
        condition_searches['frozen_plus_ft'] = search_ft

    fold_reselection_by_condition = {}
    for _cond, _s in condition_searches.items():
        if _s is None:
            continue
        try:
            fold_reselection_by_condition[_cond] = build_fold_reselection_input(_s)
        except Exception as e:
            print(f"  fold_reselection block for {_cond} unavailable: {e}")
    # One merged block per RANK, each keyed by model inside. Rank 1 keeps the
    # old condition name 'per_model_top_rep_dim' so existing readers find it;
    # rank 2 is 'per_model_top_rep_dim_rank2'. Before 2026-10-05 both ranks
    # went into one block keyed by model alone, and the rank-2 search, merged
    # last, overwrote rank 1 for every model that had two. Within a rank each
    # model has exactly one search, so the merge by model no longer collides.
    # 'representation_by_model' says which representation's width each model's
    # search was run at.
    for _rank_wanted, _cond_name in ((1, 'per_model_top_rep_dim'),
                                     (2, 'per_model_top_rep_dim_rank2')):
        _merged = None
        for (_mname, _rank), (_rep, _s) in search_per_model_top_rep_ranked.items():
            if _rank != _rank_wanted:
                continue
            try:
                _blk = build_fold_reselection_input(_s)
            except Exception as e:
                print(f"  fold_reselection block for {_cond_name}/"
                      f"{_mname} unavailable: {e}")
                continue
            if _merged is None:
                _merged = _blk
                _merged['representation_by_model'] = {_mname: _rep}
                continue
            _merged['candidates_by_model'].update(
                _blk.get('candidates_by_model', {}))
            _merged['baselines_by_model'].update(
                _blk.get('baselines_by_model', {}))
            _merged['representation_by_model'][_mname] = _rep
            # A fold excluded by any of these searches is excluded for all of
            # them: the blocks are merged into one, and fold_reselection drops
            # the listed folds from both arms together.
            _merged['exclude_folds'] = sorted(
                set(_merged.get('exclude_folds') or [])
                | set(_blk.get('exclude_folds') or []))
        if _merged is not None:
            fold_reselection_by_condition[_cond_name] = _merged

    # Compute the delta here rather than leaving it to whoever reads the file.
    # best_by_model and baselines below are both maxima taken on the folds they
    # are then reported on, and the hybrid side maximises over far more
    # candidates than the single-representation side, so their difference is
    # bigger than the same procedure would give on new molecules. This picks
    # each side's winner on the other folds and scores the held-out one, which
    # taxes both sides the same. It costs nothing: the per-fold scores are
    # already in the block above.
    reselected_delta = {}
    try:
        reselected_delta = fold_reselection(
            fold_reselection_block.get('candidates_by_model', {}),
            fold_reselection_block.get('baselines_by_model', {}),
            fold_reselection_block.get('lower_is_better'),
            metric_name=fold_reselection_block.get('metric_name'),
            exclude_folds=hr_exclude_folds(fold_reselection_block),
        )
    except Exception as e:
        print(f"  honest delta unavailable: {e}")

    # Same arithmetic on every condition's block.
    reselected_delta_by_condition = {}
    for _cond, _blk in fold_reselection_by_condition.items():
        try:
            reselected_delta_by_condition[_cond] = fold_reselection(
                _blk.get('candidates_by_model', {}),
                _blk.get('baselines_by_model', {}),
                _blk.get('lower_is_better'),
                metric_name=_blk.get('metric_name'),
                exclude_folds=hr_exclude_folds(_blk),
            )
        except Exception as e:
            print(f"  honest delta for {_cond} unavailable: {e}")

    # The delta to read: each held-out fold's hybrid recipe and its best single
    # representation are both picked on the other folds only.
    if reselected_delta:
        print(f"\n  HYBRID vs BEST SINGLE REPRESENTATION, held-out fold re-pick "
              f"({metric_label}, + means the hybrid won):")
        for model_name in models:
            _hd = reselected_delta.get(model_name)
            if not _hd:
                continue
            _d = _hd.get('delta_mean_of_folds')
            if _d is None or (isinstance(_d, float) and np.isnan(_d)):
                print(f"    {model_name}: no paired folds")
                continue
            print(f"    {model_name}: {_d:+.4f} over "
                  f"{_hd.get('n_paired_folds', 0)} folds")

    # --- The top TWO of each arm, and the two reference numbers -------------
    # Everything above reports one winner per downstream model: best_by_model
    # holds one hybrid configuration, baselines[model]['best_single_rep'] one
    # representation. Nothing said what came second or by how much. Baseline
    # scores are stored rounded to 4 decimals (find_best_hybrid.py:5525), so two
    # representations can hold the same score -- 'tied_at_best_single' above was
    # added on 2026-09-18 for exactly that, and it names the tie without saying
    # what the runner-up scored. These read rows the search already stored and
    # fit nothing: 0 new model fits.
    #
    # Built from `search` (the uncapped condition), the same SearchResult
    # best_by_model, baselines and fold_reselection are built from.
    #
    # require_full=False everywhere: a downstream model the search never scored
    # comes back as a short list carrying its own shortfall_reason, instead of
    # raising after every fit in the grid above has already been paid for.
    def _no_blueprint(obj):
        """Drop the pickled Blueprint from the blocks added below.

        It does serialise -- convert_numpy routes it through Blueprint.to_dict
        (find_best_hybrid.py:1438) -- but it carries kept_augmentations, one
        list of feature indices per fold per donor representation, and every
        recipe field on it (allocation method, importance method, budget,
        internal evaluator, step size, representation count) is already a key of
        the entry it hangs off. Which representations actually fed a hybrid is
        recorded by hybrid_deltas under 'source_reps'.
        """
        if isinstance(obj, dict):
            return {k: _no_blueprint(v) for k, v in obj.items()
                    if k != 'blueprint'}
        if isinstance(obj, list):
            return [_no_blueprint(v) for v in obj]
        return obj

    # `models` (line 45) is get_model_factories(config['models']) -- the model
    # names this file was asked for. The three searches above pass models=None,
    # so the search scores whatever the speed setting names, and under 'fast'
    # that is rf and lightgbm only. The two lists disagree, and the comparison
    # has to run over the models the search actually scored. Every name in
    # config['models'] the search did not score is written into the results
    # under 'models_dropped' with the reason; none is swapped for another
    # model's rows.
    comparison_models, models_dropped = resolve_comparison_models(
        search, list(models))
    if models_dropped:
        for _name, _why in models_dropped:
            print(f"  {_name} is not in the comparison: {_why}")

    top2_base_reps = {}
    top2_hybrids = {}
    hybrid_deltas_by_model = {}
    selection_failures = []
    for _m in comparison_models:
        try:
            top2_base_reps[_m] = top_base_reps(search, _m, 2,
                                               require_full=False)
        except Exception as _e:
            selection_failures.append(
                {'downstream_model': _m, 'half': 'base_reps',
                 'reason': f'{type(_e).__name__}: {_e}'})
            print(f"  top-2 base representations for {_m} unavailable: {_e}")
        _th = None
        try:
            _th = top_hybrids(search, _m, 2, require_full=False)
            top2_hybrids[_m] = _no_blueprint(_th)
        except Exception as _e:
            selection_failures.append(
                {'downstream_model': _m, 'half': 'hybrids',
                 'reason': f'{type(_e).__name__}: {_e}'})
            print(f"  top-2 hybrid configurations for {_m} unavailable: {_e}")
        # The rank-1 hybrid is handed to hybrid_deltas explicitly. Called
        # without one it picks its own and reads it as
        # _picked['selected'][0]['row'] (_shared.py:7432), and a selected entry
        # from top_hybrids has no 'row' key -- 24 keys, none of them 'row' --
        # so that path raises KeyError('row'). Checked by running both calls in
        # this session. The entry carries 'score_mean' and 'blueprint', which is
        # all hybrid_deltas reads off the hybrid it is given.
        if _th and _th['selected']:
            try:
                hybrid_deltas_by_model[_m] = hybrid_deltas(
                    search, _m, hybrid=_th['selected'][0])
            except Exception as _e:
                selection_failures.append(
                    {'downstream_model': _m, 'half': 'hybrid_deltas',
                     'reason': f'{type(_e).__name__}: {_e}'})
                print(f"  hybrid deltas for {_m} unavailable: {_e}")

    # The same two-by-two grid in one block: per downstream model, the 2 base
    # representations and the 2 hybrids, plus 'n_distinct_units' -- how many
    # separate things would have to be BUILT once one representation chosen by
    # three downstream models is counted once.
    top2_pairs_by_model = {}
    try:
        top2_pairs_by_model = _no_blueprint(
            select_model_rep_pairs(search, models=comparison_models,
                                   n_base=2, n_hybrids=2,
                                   require_full=False))
    except Exception as _e:
        selection_failures.append(
            {'downstream_model': None, 'half': 'model_rep_pairs',
             'reason': f'{type(_e).__name__}: {_e}'})
        print(f"  top-2 model-and-representation grid unavailable: {_e}")

    # Owner's set (2026-10-02): top 2 general + top 1 distance hybrids per
    # model. Every one of the three was scored by its search's own 5-fold
    # cross-validation; the scores, fold scores and both reference deltas are
    # written per slot. general1 is the same configuration as
    # top2_hybrids[model]['selected'][0]; the existing keys are unchanged.
    rule_hybrids_by_model = {}
    rule_deltas_by_model = {}
    rule_missing = []
    for _m in comparison_models:
        _entries, _miss = _rule_from_searches(
            search, search_distance, _m, dist_error=distance_search_error)
        rule_missing += _miss
        rule_hybrids_by_model[_m] = _no_blueprint(_entries)
        rule_deltas_by_model[_m] = {}
        for _e in _entries:
            _src = search if _e['lineage'] == 'general' else search_distance
            try:
                rule_deltas_by_model[_m][_e['slot']] = _no_blueprint(
                    hybrid_deltas(_src, _m, hybrid=_e))
            except Exception as _ex:
                rule_missing.append(
                    f"{_e['slot']} deltas for {_m}: "
                    f"{type(_ex).__name__}: {_ex}")
        for _e in _entries:
            print(f"    rule {_m} {_e['slot']}: {_e['allocation_method']}/"
                  f"{_e['importance_method']}/b={_e['budget']} -> "
                  f"{metric_label}={_e['score_mean']:.4f}")

    distance_fold_reselection = None
    if search_distance is not None:
        try:
            distance_fold_reselection = build_fold_reselection_input(
                search_distance)
        except Exception as _ex:
            rule_missing.append(f'distance fold_reselection block: '
                                f'{type(_ex).__name__}: {_ex}')

    # Printed because the two deltas answer different questions and a reader
    # given one number cannot tell which was asked. The first is the hybrid
    # against the mean of the representations the budget actually gave features
    # to, read off Blueprint.allocation_per_fold, so a representation given 0
    # features is not in the mean. The second is the hybrid against the single
    # best base representation for the same downstream model. A hybrid can win
    # the first and lose the second.
    if hybrid_deltas_by_model:
        print(f"\n  TOP HYBRID vs TWO REFERENCE NUMBERS ({metric_label}, "
              f"+ means the hybrid won):")
        for _m in comparison_models:
            _hd = hybrid_deltas_by_model.get(_m)
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

    return {
        'configs': configs,
        # None when the frozen-plus-fine-tuned arm ran or was never asked
        # for; otherwise the exception that stopped it, other conditions intact.
        'ft_search_error': ft_search_error,
        'fold_reselection': fold_reselection_block,
        # Report this, not best_by_model minus baselines.
        'reselected_delta': reselected_delta,
        # Per-condition blocks and deltas. 'fold_reselection'/'reselected_delta' above
        # are the uncapped condition, kept under their existing keys because
        # readers already fetch them there.
        'fold_reselection_by_condition': fold_reselection_by_condition,
        'reselected_delta_by_condition': reselected_delta_by_condition,
        'best_by_model': best_by_model,
        'baselines': baselines,
        'rep_dims': rep_dims,
        'pca_baselines': pca_baselines,
        'pca_baseline_widths': pca_baseline_widths,
        'pca_baseline_fold_scores': pca_baseline_fold_scores,
        # 2026-10-05, owner delegated, 2026-10-05: records that the PCA
        # baselines cap each scaled validation feature at its training range.
        'pca_validation_feature_cap': 'clip to training-row min/max after StandardScaler',
        'heatmap': heatmap,
        'importance_wins': importance_wins,
        'conditions': conditions,
        'best_per_model_across_conditions': best_per_model_across_conditions,
        'has_augmentation': augs is not None,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        # Added 2026-09-18, additive. New keys, no existing key touched:
        # 'baselines' still carries best_single_rep/best_single_score,
        # 'best_by_model' still carries one hybrid per downstream model, and
        # 'reselected_delta' is still the delta to read. 0 new model fits --
        # all four read rows the search above already stored.
        'top2_base_reps': top2_base_reps,
        'top2_hybrids': top2_hybrids,
        'top2_pairs_by_model': top2_pairs_by_model,
        'hybrid_deltas_by_model': hybrid_deltas_by_model,
        # Which downstream models the two-against-two comparison above ran over,
        # which of the ones asked for were not scored by the search, and any
        # model whose half of the comparison raised. A reader who finds three
        # models in 'top2_base_reps' and five in config['models'] gets the
        # reason from here.
        'comparison_models': comparison_models,
        'models_dropped': [{'downstream_model': _n, 'reason': _w}
                           for _n, _w in models_dropped],
        'comparison_failures': selection_failures,
        # Owner's set (2026-10-02): per model, the top 2 hybrids of the
        # general search and the top 1 of the distance search, each with its
        # slot, tier and cross-validated scores, plus every slot not filled.
        'rule_hybrids': rule_hybrids_by_model,
        'rule_hybrid_deltas': rule_deltas_by_model,
        'rule_missing': rule_missing,
        'rule_distance_condition': _dist_cond,
        'distance_search_error': distance_search_error,
        'distance_fold_reselection': distance_fold_reselection,
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

    # A comma-separated --reps is a list of representation names, not a preset.
    # Passed straight through it was read as a preset name, missed, and the ten
    # general representations were loaded instead.
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
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
