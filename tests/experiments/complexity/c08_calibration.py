#!/usr/bin/env python3
"""C08: Uncertainty Calibration at Optimal vs Suboptimal Complexity.

Paper: Complexity
Conclusion: Complexity matching improves calibration.
Dependencies: c02 (optional, reads JSON for optimal config).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_model_factories, _cv_split,
    create_hybrid, apply_feature_selection,
    find_best_hybrid, get_domain_metric, make_cv_splitter,
    get_best_search_for_downstream,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS, FULL_BUDGETS,
    get_rule_hybrids, MAIN_MODELS_WITH_KNN,
)
from kirby.hybrid import apply_augmentation_selection
from kirby.find_best_hybrid import _predict_for_metric


def _cal_split(train_groups, n_train, n_cal, fold_i):
    """Cut the conformal calibration rows out of a fold's training rows.

    Group-aware: a scaffold that appears in the proper-training half never
    appears in the calibration half, so the calibrated interval width comes
    from residuals as hard as the held-out fold's. Falls back to a permutation
    only when the fold has fewer than two distinct groups, where no group-aware
    cut exists.
    """
    if train_groups is not None and len(np.unique(train_groups)) >= 2:
        splitter, _ = make_cv_splitter(
            1, groups=train_groups, seed=42 + fold_i)
        dummy = np.zeros((n_train, 1))
        ptrain_idx, cal_idx = next(
            splitter.split(dummy, groups=train_groups))
        return ptrain_idx, cal_idx
    perm = np.random.RandomState(42 + fold_i).permutation(n_train)
    return perm[n_cal:], perm[:n_cal]


def run(data, config, output_dir, dataset_name, conditions=None, c02_results_dir=None):
    """Exp 8: Complexity matching produces better-calibrated uncertainty estimates.

    The 'optimal' condition uses find_best_hybrid() (or Exp 2 results) to
    discover the actual best config. Suboptimal conditions are deliberately
    constructed to show the contrast.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: UNCERTAINTY CALIBRATION AT OPTIMAL VS SUBOPTIMAL COMPLEXITY")
    print("=" * 70)

    # Domain-aware metric
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    task_type = data['task_type']
    is_classification = (task_type == 'classification')

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import Ridge, LogisticRegression

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    total_dim = sum(v.shape[1] for v in reps.values())

    all_conditions = conditions or ['optimal', 'underfit', 'overfit', 'mismatched']

    # Try to load Exp 2 results if a directory was provided
    exp2_results = None
    if c02_results_dir:
        exp2_results = load_experiment_results(c02_results_dir, 'c02_complexity_matching_grid',
                                               dataset_name)
        if exp2_results:
            print(f"  Loaded Exp 2 results from {c02_results_dir}")
        else:
            print(f"  No Exp 2 results found at {c02_results_dir}")

    # Discover optimal config via find_best_hybrid or Exp 2 results
    augs = data.get('augmentations')
    optimal_model_factory = None
    optimal_budget = 100
    # Fallback when neither find_best_hybrid nor a c02 result supplies a
    # recipe: the engine's own current defaults. optimal_alloc was 'greedy',
    # which is cut.
    optimal_alloc = 'performance_weighted'
    optimal_imp = 'treeshap'
    # 'match' is the canonical Blueprint default (downstream==evaluator);
    # 'ridge' would force ridge as internal evaluator regardless of task and
    # would not match precomputed cache keys.
    optimal_internal_eval = 'match'
    # Match the actual augmentation presence (collision-safe default).
    optimal_aug_strategy = 'uncapped' if augs else 'none'
    optimal_desc = 'RF at budget=100 (default)'
    # Defined here, not inside the else-branch: the fallback report below runs
    # on both paths.
    pool_lookup_errors = {}
    recipe_source = None
    # ⛔ DEFINED ON EVERY PATH. These were set only in the pool-lookup branch
    # below, and the c02 branch beside it does not set them -- so a run that
    # took the c02 path raised NameError on every fold, inside the try that
    # wraps the build, and all four conditions came back with no metrics.
    # Caught on flab_thermo_tresanco at 07:34 on 2026-09-24, by the
    # skipped-condition record added the same night.
    #
    # c02 stores model, allocation, importance and budget per model and stores
    # no rep_count, so on that path there is nothing to cut to and the hybrid
    # is built on the whole pool -- which is what it did before tonight.
    optimal_rep_count = None
    optimal_rep_scores = None
    # {condition: ['fold N: Error: message', ...]} for every fold that raised
    # inside the build. The print below it goes to the log and nowhere else, so
    # a saved file could say a condition produced nothing without saying why.
    _fold_errors = {}

    if exp2_results and exp2_results.get('best_by_model'):
        # Use best config from Exp 2
        all_best = exp2_results['best_by_model']
        overall_best = (min if lower_is_better else max)(all_best.values(), key=lambda c: c['score_mean'])
        optimal_budget = overall_best['budget']
        optimal_alloc = overall_best['allocation']
        optimal_imp = overall_best['importance']
        best_model = overall_best['model']
        optimal_model_factory = get_model_factories(
            [best_model], task_type=data['task_type'],
            n_samples=len(data['labels'])).get(best_model)
        optimal_desc = (f"{best_model} at budget={optimal_budget} "
                        f"[{optimal_imp}/{optimal_alloc}] (from Exp 2)")
        print(f"  Using Exp 2 optimal: {optimal_desc}")
        recipe_source = 'c02'
    else:
        # Pool helper picks the best (tier, aug_condition) cache file per
        # downstream model. We try across all common models and pick the
        # overall winner (preserves the prior "best across all models" semantics).
        print(f"  Using pool helper to find optimal ()...")
        try:
            best = None
            best_model = None
            best_search = None
            best_meta = None
            for candidate_model in ['rf', 'svm', 'lightgbm', 'dnn', 'knn']:
                try:
                    s, m = get_best_search_for_downstream(
                        reps, labels, groups,
                        downstream_model=candidate_model, budget=None,
                        augmentations=augs,
                        dataset_name=dataset_name,
                        task_type=data['task_type'],
                        data=data,
                    )
                except Exception as _e:
                    # ⛔ RECORD IT. Every one of the candidate models can throw
                    # here; with all of them gone `best` stays None and the
                    # whole experiment silently runs RandomForest at budget
                    # 100, performance_weighted, treeshap -- a default nobody
                    # chose -- while the results carry no sign that a discovery
                    # was ever attempted. The `except` further down only
                    # catches a throw OUTSIDE this loop, so nothing printed.
                    # c09 does the same lookup and records from_fallback.
                    pool_lookup_errors[candidate_model] = f'{type(_e).__name__}: {_e}'
                    continue
                if not m:
                    pool_lookup_errors[candidate_model] = 'no search metadata returned'
                    continue
                bc = m['best_config']
                score = bc.get('score_mean')
                if score is None:
                    continue
                is_better = (best is None) or (
                    score < best['score_mean'] if lower_is_better
                    else score > best['score_mean'])
                if is_better:
                    best = bc
                    best_model = candidate_model
                    best_search = s
                    best_meta = m
            if best:
                optimal_budget = best['blueprint'].budget
                optimal_alloc = best['blueprint'].allocation_method
                optimal_imp = best['blueprint'].importance_method
                optimal_internal_eval = best['blueprint'].internal_evaluator
                optimal_aug_strategy = best['blueprint'].augmentation_strategy
                # ⛔ rep_count AND THE SCORES THAT SAY WHICH ONES. rep_count is
                # a searched axis -- 2, 3, 5 or the whole pool -- so a
                # configuration that won on 2 of the 10 representations is
                # stored with rep_count=2. Lifting budget, allocation,
                # importance and evaluator but not this rebuilt every condition
                # on all 10, which is a different feature set and a different
                # budget split from the configuration that won, filed under the
                # winner's name. build_fold_hybrid was fixed for exactly this
                # on 2026-09-18 and RAISES rather than rebuild wide; c08 calls
                # create_hybrid directly and never reached that guard.
                optimal_rep_count = getattr(best['blueprint'], 'rep_count', None)
                optimal_rep_scores = getattr(best['blueprint'], 'rep_scores', None)
                best_model = best.get('model', best_model)
                optimal_model_factory = get_model_factories(
                    [best_model], task_type=data['task_type'],
                    n_samples=len(data['labels'])).get(best_model)
                optimal_desc = (f"{best_model} at budget={optimal_budget} "
                                f"[{optimal_imp}/{optimal_alloc}] "
                                f"(pool {best_meta['tier']}/{best_meta['aug_condition']})")
                print(f"  Discovered optimal: {optimal_desc}")
                top3 = extract_top3_base_reps(best_search)
                if top3:
                    print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        except Exception as e:
            print(f"  pool helper failed ({e}), using RF/budget=100 default")

    used_pool_fallback = optimal_model_factory is None
    if not used_pool_fallback and recipe_source is None:
        recipe_source = 'pool helper'
    if used_pool_fallback:
        recipe_source = 'hardcoded default (no recipe found)'
        print(f"  NO POOL RECIPE FOUND for any candidate model. Falling back "
              f"to RandomForest at budget 100. Why each candidate was "
              f"unusable: {pool_lookup_errors or 'no candidates tried'}")
    if optimal_model_factory is None:
        if is_classification:
            optimal_model_factory = lambda: RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1)
        else:
            optimal_model_factory = lambda: RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1)

    # Define conditions: (model_factory, budget, alloc, imp, description)
    if is_classification:
        mismatched_factory = lambda: LogisticRegression(C=100, max_iter=1000, random_state=42)
        mismatched_desc = 'LogReg at full dim (low-capacity model, high-dim input)'
    else:
        mismatched_factory = lambda: Ridge(alpha=0.01)
        mismatched_desc = 'Ridge at full dim (low-capacity model, high-dim input)'

    # 2026-10-05, owner delegated, 2026-10-05: the underfit condition must ask
    # for fewer features than the optimal condition. It was
    # min(25, max(5, total_dim // 4)), which on cdr asked for more features
    # than the optimal recipe's budget. Now: 25 features if 25 is below the
    # optimal budget; otherwise the largest budget in this experiment's budget
    # list (config['budgets']) below the optimal budget; if there is none, the
    # underfit condition is not run and the reason is stored. Because optimal
    # never asks for more than total_dim, underfit stays narrower than overfit
    # too, which is what the total_dim // 4 scaling was for.
    _c08_budget_list = sorted(int(b) for b in config.get('budgets', FULL_BUDGETS))

    def _underfit_for(opt_budget):
        """(underfit budget in features or None, reason when None)."""
        _opt = total_dim if opt_budget is None else int(opt_budget)
        if 25 < _opt:
            return 25, None
        _below = [b for b in _c08_budget_list if b < _opt]
        if _below:
            return _below[-1], None
        return None, (f'no budget below the optimal budget of {_opt} features '
                      f'in the budget list {_c08_budget_list}; underfit '
                      f'condition not run')

    underfit_budget, underfit_skip_reason = _underfit_for(optimal_budget)
    condition_specs = {
        'optimal': (optimal_model_factory, optimal_budget, optimal_alloc, optimal_imp,
                    optimal_desc),
        'underfit': (optimal_model_factory, underfit_budget, optimal_alloc, optimal_imp,
                     f'Same model at budget={underfit_budget} (too compressed)'),
        'overfit': (optimal_model_factory, total_dim, optimal_alloc, optimal_imp,
                    f'Same model at full dim={total_dim} (too many features)'),
        # The mismatch under test is the MODEL at full width, not the
        # allocation. Was 'greedy', which is cut and costs 20-100x for nothing.
        # ⛔ AND IT WAS STILL NOT THE MODEL ALONE. This line hardcoded
        # 'performance_weighted' and 'treeshap' while 'overfit' beside it --
        # the condition it is read against, same budget, same width -- uses the
        # DISCOVERED allocation and importance methods. The search picks the
        # allocation from five and the importance method from three, so the two
        # conditions commonly differed in the model, the allocation method and
        # the importance method together, and the comment above said the
        # opposite. Both now take the discovered values, so 'overfit' and
        # 'mismatched' differ in the model and nothing else.
        'mismatched': (mismatched_factory, total_dim, optimal_alloc,
                       optimal_imp, mismatched_desc),
    }
    if underfit_budget is None:
        # 2026-10-05, owner delegated: see _underfit_for above.
        condition_specs.pop('underfit')
        print(f"  underfit skipped: {underfit_skip_reason}")

    # Condition-distinctness diagnostic. The calibration comparison is only
    # meaningful if optimal / underfit / overfit use genuinely different budgets.
    # (Earlier analysis mis-stated that "optimal == full dim" on EGFR — in fact
    # optimal=659 vs total_dim=68948, so they ARE separated and the "underfit
    # calibrates best" result is a real finding, not a collapse. This field lets
    # the rerun confirm separation per dataset.) See
    # COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
    # {condition: [features built, per fold]}, filled in the fold loop.
    _built_widths = {}
    _cond_budgets = {'optimal': optimal_budget, 'underfit': underfit_budget,
                     'overfit': total_dim}
    results = {
        # Where the 'optimal' recipe came from, and why any candidate model
        # could not supply one. Without these, a run that found no recipe at
        # all and silently used RandomForest at budget 100 is indistinguishable
        # in the saved file from one that discovered a recipe.
        'recipe_source': recipe_source,
        'fold_errors': _fold_errors,
        'pool_lookup_errors': pool_lookup_errors,
        'used_hardcoded_fallback': bool(used_pool_fallback),
        'conditions': {},
        'task_type': task_type,
        'total_dim': total_dim,
        'condition_budgets': _cond_budgets,
        # {condition: [features built, per fold]}. Read beside
        # condition_budgets: a condition asking for total_dim and building a
        # third of it is not the condition its name says.
        'condition_widths_built': _built_widths,
        'conditions_distinct': len(set(_cond_budgets.values())) == len(_cond_budgets),
        # 2026-10-05, owner delegated, 2026-10-05: why the underfit condition
        # was not run (None when it ran), and the rule that picks its budget.
        'underfit_skip_reason': underfit_skip_reason,
        'underfit_budget_rule': ('25 features if below the optimal budget, '
                                 'else the largest budget in the budget list '
                                 'below it, else not run'),
    }
    if not results['conditions_distinct']:
        print(f"  WARNING: calibration conditions collapse (budgets "
              f"{_cond_budgets}) — optimal/underfit/overfit not separated.")

    # Owner's set (2026-10-02): the four conditions below are run once for
    # the recipe chosen as before AND once for each of every downstream
    # model's rule hybrids (top 2 general + top 1 distance). The fold loop is
    # this one function so every run is computed the same way.
    def _run_condition_set(condition_specs, optimal_internal_eval,
                           optimal_aug_strategy, optimal_rep_count,
                           optimal_rep_scores, out_conditions, _built_widths,
                           _fold_errors, recipe_objective=None):
        for cond_name in all_conditions:
            if cond_name not in condition_specs:
                continue
            model_factory, budget, alloc, imp, desc = condition_specs[cond_name]
            print(f"\n  {cond_name}: {desc}")

            sample_X = next(iter(reps.values()))

            fold_metrics = []
            # task_type: without it _should_stratify falls back to the integer-dtype
            # heuristic, which its own docstring calls a poor proxy -- a
            # classification target stored as 0.0/1.0 gets plain GroupKFold and a
            # fold can come out single-class.
            for fold_i, (train_idx, val_idx) in enumerate(
                    _cv_split(groups, n_folds, sample_X, labels,
                              task_type=task_type)):
                train_reps = {k: v[train_idx] for k, v in reps.items()}
                val_reps = {k: v[val_idx] for k, v in reps.items()}
                y_train, y_val = labels[train_idx], labels[val_idx]
                train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
                val_augs = {k: v[val_idx] for k, v in augs.items()} if augs else None

                try:
                    # model/metric_fn/lower_is_better/is_clf are passed so the
                    # budget split is decided by the estimator and the metric this
                    # fold is actually scored with. Without model, internal
                    # evaluator 'match' resolves to lightgbm for every condition,
                    # including the random-forest ones; without metric_fn the
                    # allocation ranking falls back to r2_score or accuracy instead
                    # of the dataset's own metric.
                    # Cut to the representations the recipe was searched over,
                    # the same way build_fold_hybrid does it (_shared.py:5096):
                    # rank by the blueprint's own rep_scores, break ties by name so
                    # two runs resolve identically.
                    _tr_reps, _va_reps = train_reps, val_reps
                    if (optimal_rep_count is not None
                            and int(optimal_rep_count) < len(train_reps)):
                        if not optimal_rep_scores:
                            raise ValueError(
                                f"c08: the discovered recipe was searched over "
                                f"{optimal_rep_count} of the pool's "
                                f"{len(train_reps)} representations and carries no "
                                f"rep_scores to say which. Rebuilding on all of "
                                f"them would score a different feature set under "
                                f"the winner's name.")
                        _ranked = sorted(
                            train_reps,
                            key=lambda r: (-optimal_rep_scores.get(r, float('-inf')), r))
                        _keep = set(_ranked[:int(optimal_rep_count)])
                        _tr_reps = {k: v for k, v in train_reps.items() if k in _keep}
                        _va_reps = {k: v for k, v in val_reps.items() if k in _keep}
                    X_train, feature_info = create_hybrid(
                        _tr_reps, y_train, groups=groups[train_idx],
                        allocation_method=alloc,
                        budget=budget,
                        importance_method=imp,
                        internal_evaluator=optimal_internal_eval,
                        augmentations=train_augs,
                        augmentation_strategy=optimal_aug_strategy,
                        model=model_factory(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better,
                        is_clf=is_classification,
                        **({'objective': recipe_objective}
                           if recipe_objective is not None else {}),
                    )
                    # ⛔ THE WIDTH BUILT, NOT THE BUDGET ASKED FOR. Two of c08's
                    # four conditions run at budget = total_dim and are read as
                    # "full dimensionality". create_hybrid stops each
                    # representation at its last feature scoring above zero, and
                    # allocate_weighted_global cuts the glued pool the same way
                    # without printing anything, so an "overfit at full dim"
                    # condition can be built on a fraction of the pool -- at which
                    # point it is not an overfit condition and the comparison it
                    # anchors cannot be read. condition_budgets stores the number
                    # asked for; this stores the number built.
                    _built_width = int(X_train.shape[1])
                    _built_widths.setdefault(cond_name, []).append(_built_width)
                    # _va_reps, not val_reps: the same representations the hybrid
                    # was built from, or the column order will not line up.
                    X_val = apply_feature_selection(_va_reps, feature_info)
                    aug_info = feature_info.get('augmentation_info')
                    if aug_info and val_augs:
                        X_val_aug = apply_augmentation_selection(val_augs, aug_info)
                        if X_val_aug is not None:
                            X_val = np.hstack([X_val, X_val_aug])
                except Exception as e:
                    _fold_errors.setdefault(cond_name, []).append(
                        f'fold {fold_i}: {type(e).__name__}: {e}')
                    print(f"    fold {fold_i}: hybrid creation failed - {e}")
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
                # 2026-10-05, owner delegated, 2026-10-05: each validation
                # feature is capped at the range it takes on the scaled
                # training rows. On adme_hlm one validation molecule sat 83
                # million training standard deviations out and the dnn's rmse
                # ran to thousands.
                X_val_s = np.clip(X_val_s, X_train_s.min(0), X_train_s.max(0))

                # Train base model
                model = model_factory()
                model.fit(X_train_s, y_train)

                if is_classification:
                    # Conformal classification: split train into proper-train + cal
                    n_train = len(y_train)
                    n_cal = max(20, int(n_train * 0.2))
                    # The calibration rows are cut on scaffold groups, not at
                    # random. A random 20% puts molecules from the same scaffold on
                    # both sides, so the residuals the interval is calibrated from
                    # are easier than the held-out fold the coverage is reported on.
                    ptrain_idx, cal_idx = _cal_split(
                        groups[train_idx], n_train, n_cal, fold_i)

                    X_ptrain, y_ptrain = X_train_s[ptrain_idx], y_train[ptrain_idx]
                    X_cal, y_cal = X_train_s[cal_idx], y_train[cal_idx]

                    model_conf = model_factory()
                    model_conf.fit(X_ptrain, y_ptrain)

                    # Compute val_score from model_conf so the performance number
                    # and the calibration metrics come from the SAME model (was a
                    # full-train vs proper-train mismatch).  For AUC metrics, use
                    # predict_proba (hard labels would give a meaningless AUROC).
                    _from_kfh = _predict_for_metric(model_conf, X_val_s, metric_fn)
                    val_score = float(metric_fn(y_val, _from_kfh))

                    # Conformal classification: nonconformity = 1 - P(true class)
                    if hasattr(model_conf, 'predict_proba'):
                        cal_probs = model_conf.predict_proba(X_cal)
                        # Map y_cal labels to column indices in model_conf.classes_.
                        # Calibration points whose true label was never seen in proper-train
                        # are reported and assigned worst-case nonconformity (1.0) rather
                        # than silently dropped (which would bias coverage).
                        class_to_col = {c: j for j, c in enumerate(model_conf.classes_)}
                        cal_scores = np.empty(len(y_cal), dtype=float)
                        n_unseen = 0
                        for i in range(len(y_cal)):
                            col = class_to_col.get(int(y_cal[i]))
                            if col is None:
                                cal_scores[i] = 1.0
                                n_unseen += 1
                            else:
                                cal_scores[i] = 1 - cal_probs[i, col]
                        if n_unseen:
                            print(f"    fold {fold_i}: {n_unseen}/{len(y_cal)} "
                                  f"calibration points have classes unseen in proper-train "
                                  f"(assigned worst-case nonconformity 1.0)")
                        if len(cal_scores) == 0:
                            print(f"    fold {fold_i}: no valid calibration scores")
                            continue

                        alpha = 0.1
                    # ⛔ A CALIBRATION HALF THIS SMALL CANNOT CARRY THE LEVEL
                    # ASKED FOR. n_cal above is computed and then IGNORED by the
                    # group-aware branch of _cal_split, which takes
                    # make_cv_splitter(1, ...) -- GroupShuffleSplit at test_size
                    # 0.2 -- so the real size is whatever the groups give. Split
                    # conformal needs at least (1-alpha)/alpha rows, 9 at
                    # alpha=0.1: below that ceil((n+1)(1-alpha))/n exceeds 1, the
                    # min() clamps it, and q becomes the LARGEST calibration
                    # residual. The interval is then the widest the calibration set
                    # can produce, not a 90% one, and the PICP printed beside it is
                    # not coverage at the stated level.
                        _n_cal_used = len(cal_scores)
                        _n_cal_actual = _n_cal_used
                        _raw_level = (np.ceil((_n_cal_actual + 1) * (1 - alpha))
                                      / _n_cal_actual)
                        conformal_level = min(1.0, _raw_level)
                        _level_clamped = bool(_raw_level > 1.0)
                        if _level_clamped:
                            print(f"      fold {fold_i}: calibration half has "
                                  f"{_n_cal_actual} rows, under the "
                                  f"{int(np.ceil((1 - alpha) / alpha))} that "
                                  f"alpha={alpha} needs; q is the largest "
                                  f"residual and the interval is NOT at the "
                                  f"{100 * (1 - alpha):.0f}% level")
                        q = np.quantile(cal_scores, conformal_level)

                        # Prediction sets on val
                        val_probs = model_conf.predict_proba(X_val_s)
                        classes = model_conf.classes_
                        pred_sets = []
                        for i in range(len(y_val)):
                            pset = [classes[j] for j in range(len(classes))
                                    if 1 - val_probs[i, j] <= q]
                            pred_sets.append(pset)

                        # Coverage: true label in prediction set
                        covered = np.array([y_val[i] in pred_sets[i] for i in range(len(y_val))])
                        picp = float(np.mean(covered))
                        avg_set_size = float(np.mean([len(ps) for ps in pred_sets]))
                        n_classes = len(classes)
                        # Sharpness: avg set size / n_classes (≈ 1/n_classes ⇒ sharp/decisive;
                        # 1 ⇒ trivial full-prediction-set). LOWER is better.
                        sharpness_ratio = avg_set_size / n_classes if n_classes > 0 else float('nan')
                        cal_error = abs(picp - (1 - alpha))

                        # Uncertainty-error: entropy of predicted probabilities vs correctness
                        pred_labels = model_conf.predict(X_val_s)
                        correct = (pred_labels == y_val).astype(float)
                        entropy = -np.sum(val_probs * np.log(val_probs + 1e-12), axis=1)
                        ue_corr = float('nan')
                        if len(entropy) >= 3:
                            ue_corr_val, _ = spearmanr(entropy, 1 - correct)
                            ue_corr = float(ue_corr_val)

                        fold_result = {
                            'val_score': val_score, 'picp': picp,
                            'avg_set_size': avg_set_size,
                            'sharpness_ratio': sharpness_ratio, 'cal_error': cal_error,
                            'conformal_q': float(q), 'n_val': len(y_val),
                            'n_cal': int(_n_cal_used),
                            'conformal_level_clamped': _level_clamped,
                            'n_classes': n_classes,
                            'ue_corr': ue_corr,
                        }
                    else:
                        # No predict_proba — report score only
                        fold_result = {
                            'val_score': val_score, 'picp': float('nan'),
                            'avg_set_size': float('nan'),
                            'sharpness_ratio': float('nan'), 'cal_error': float('nan'),
                            'conformal_q': float('nan'), 'n_val': len(y_val),
                            'ue_corr': float('nan'),
                        }
                    # The fold and the features its hybrid was built with,
                    # on the fold's own record (2026-10-05).
                    # condition_widths_built lists widths of the folds that
                    # built, so after a failed fold its positions no longer
                    # match fold numbers.
                    fold_result['fold'] = int(fold_i)
                    fold_result['n_features_built'] = _built_width
                    fold_metrics.append(fold_result)
                    # Read from fold_result (defined in both branches) — the locals
                    # picp/avg_set_size/cal_error are only assigned inside the
                    # predict_proba branch, so a classification model WITHOUT
                    # predict_proba (e.g. SVC without probability=True) used to crash
                    # this print with NameError. (OTHER-22)
                    print(f"    fold {fold_i}: {metric_name}="
                          f"{fold_result['val_score']:.4f}, "
                          f"PICP={fold_result['picp']:.3f}, "
                          f"set_size={fold_result['avg_set_size']:.2f}, "
                          f"cal_error={fold_result['cal_error']:.3f}")
                else:
                    # Regression: conformal prediction intervals.
                    # val_r2 must come from model_conf (not `model` trained on full
                    # y_train) so the reported performance number and the
                    # calibration metrics (PICP/MPIW) are from the same model.
                    n_train = len(y_train)
                    n_cal = max(20, int(n_train * 0.2))
                    # The calibration rows are cut on scaffold groups, not at
                    # random. A random 20% puts molecules from the same scaffold on
                    # both sides, so the residuals the interval is calibrated from
                    # are easier than the held-out fold the coverage is reported on.
                    ptrain_idx, cal_idx = _cal_split(
                        groups[train_idx], n_train, n_cal, fold_i)

                    X_ptrain, y_ptrain = X_train_s[ptrain_idx], y_train[ptrain_idx]
                    X_cal, y_cal = X_train_s[cal_idx], y_train[cal_idx]

                    model_conf = model_factory()
                    model_conf.fit(X_ptrain, y_ptrain)

                    y_pred_val = model_conf.predict(X_val_s)
                    val_r2 = metric_fn(y_val, y_pred_val)

                    y_pred_cal = model_conf.predict(X_cal)
                    cal_residuals = np.abs(y_cal - y_pred_cal)

                    # Conformal quantile with finite-sample correction (Romano et al. 2019)
                    alpha = 0.1
                    # ⛔ A CALIBRATION HALF THIS SMALL CANNOT CARRY THE LEVEL
                    # ASKED FOR. n_cal above is computed and then IGNORED by the
                    # group-aware branch of _cal_split, which takes
                    # make_cv_splitter(1, ...) -- GroupShuffleSplit at test_size
                    # 0.2 -- so the real size is whatever the groups give. Split
                    # conformal needs at least (1-alpha)/alpha rows, 9 at
                    # alpha=0.1: below that ceil((n+1)(1-alpha))/n exceeds 1, the
                    # min() clamps it, and q becomes the LARGEST calibration
                    # residual. The interval is then the widest the calibration set
                    # can produce, not a 90% one, and the PICP printed beside it is
                    # not coverage at the stated level.
                    n_cal = len(cal_residuals)
                    _n_cal_used = n_cal
                    _raw_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
                    conformal_level = min(1.0, _raw_level)
                    _level_clamped = bool(_raw_level > 1.0)
                    if _level_clamped:
                        print(f"      fold {fold_i}: calibration half has {n_cal} "
                              f"rows, under the "
                              f"{int(np.ceil((1 - alpha) / alpha))} that "
                              f"alpha={alpha} needs; q is the largest residual and "
                              f"the interval is NOT at the "
                              f"{100 * (1 - alpha):.0f}% level")
                    q = np.quantile(cal_residuals, conformal_level)

                    # Reuse y_pred_val above (same model, same X_val_s).
                    interval_lo = y_pred_val - q
                    interval_hi = y_pred_val + q

                    covered = (y_val >= interval_lo) & (y_val <= interval_hi)
                    picp = float(np.mean(covered))
                    mpiw = float(np.mean(interval_hi - interval_lo))
                    target_range = float(np.ptp(y_val))
                    sharpness_ratio = mpiw / target_range if target_range > 0 else float('nan')
                    cal_error = abs(picp - (1 - alpha))

                    errors = np.abs(y_val - y_pred_val)  # 2026-06-04 fix: y_pred_conf was undefined (NameError crashed c08 on every regression dataset); same model/X_val per L376
                    ue_corr = float('nan')
                    if hasattr(model_conf, 'estimators_') and len(errors) >= 3:
                        tree_preds = np.array([t.predict(X_val_s) for t in model_conf.estimators_])
                        per_sample_unc = tree_preds.std(axis=0)
                        ue_corr, _ = spearmanr(per_sample_unc, errors)
                        ue_corr = float(ue_corr)

                    # 'val_score' was 'val_r2' until 2026-10-05: it holds the
                    # dataset's metric, which is rmse or spearman, not R2.
                    fold_result = {
                        'val_score': val_r2, 'picp': picp, 'mpiw': mpiw,
                        'sharpness_ratio': sharpness_ratio, 'cal_error': cal_error,
                        'conformal_q': float(q), 'n_val': len(y_val),
                        'n_cal': int(_n_cal_used),
                        'conformal_level_clamped': _level_clamped,
                        'ue_corr': ue_corr,
                    }
                    # The fold and the features its hybrid was built with,
                    # on the fold's own record (2026-10-05).
                    # condition_widths_built lists widths of the folds that
                    # built, so after a failed fold its positions no longer
                    # match fold numbers.
                    fold_result['fold'] = int(fold_i)
                    fold_result['n_features_built'] = _built_width
                    fold_metrics.append(fold_result)
                    print(f"    fold {fold_i}: {metric_name}={val_r2:.4f}, PICP={picp:.3f}, "
                          f"MPIW={mpiw:.3f}, cal_error={cal_error:.3f}")

            # ⛔ A CONDITION WHOSE EVERY FOLD FAILED USED TO VANISH. `if
            # fold_metrics:` alone skipped writing it, so the summary table printed
            # three rows instead of four and said nothing about the fourth. It is
            # now written with a reason.
            if not fold_metrics:
                out_conditions[cond_name] = {
                    'description': desc,
                    'budget': budget,
                    'avg': {},
                    'n_folds_in_avg': {},
                    'n_folds_scored': 0,
                    'n_folds_requested': int(n_folds),
                    'folds': [],
                    'skipped_reason': 'every fold raised or produced no metrics',
                    'fold_errors': _fold_errors.get(cond_name, []),
                }
                print(f"    {cond_name}: NO FOLD PRODUCED METRICS — recorded as "
                      f"skipped, not dropped")
            if fold_metrics:
                avg = {}
                # Union of keys across folds, missing-key-tolerant: a classification
                # model without predict_proba emits fewer keys (no 'n_classes') than
                # one with it. If folds ever disagree on key sets, indexing
                # fold_metrics[0] keys with fm[key] would KeyError. (OTHER-22)
                all_keys = set()
                for fm in fold_metrics:
                    all_keys.update(fm.keys())
                # ⛔ HOW MANY FOLDS EACH AVERAGE IS OVER. avg[key] is the mean of
                # the folds where that key was finite, so a PICP over 1 fold of 5
                # and one over 5 of 5 were written identically and compared to each
                # other as if they were the same measurement.
                n_avg = {}
                # 'fold' is the fold number, not a measurement.
                all_keys.discard('fold')
                for key in all_keys:
                    vals = [fm[key] for fm in fold_metrics
                            if key in fm and not np.isnan(fm[key])]
                    avg[key] = float(np.mean(vals)) if vals else float('nan')
                    n_avg[key] = len(vals)
                _short = {k: v for k, v in n_avg.items() if v < len(fold_metrics)}
                if _short:
                    print(f"    {cond_name}: averaged over fewer than "
                          f"{len(fold_metrics)} folds for {_short}")
                out_conditions[cond_name] = {
                    'description': desc,
                    'budget': budget,
                    'avg': avg,
                    'n_folds_in_avg': n_avg,
                    'n_folds_scored': len(fold_metrics),
                    'n_folds_requested': int(n_folds),
                    'folds': fold_metrics,
                }
                score_key = 'val_score'
                picp_val = avg.get('picp', float('nan'))
                if is_classification:
                    set_key = avg.get('avg_set_size', float('nan'))
                    print(f"  >> {cond_name} avg: {metric_name}={avg.get(score_key, float('nan')):.4f}, "
                          f"PICP={picp_val:.3f}, set_size={set_key:.2f}, "
                          f"cal_error={avg.get('cal_error', float('nan')):.3f}")
                else:
                    print(f"  >> {cond_name} avg: {metric_name}={avg.get(score_key, float('nan')):.4f}, "
                          f"PICP={picp_val:.3f}, MPIW={avg.get('mpiw', float('nan')):.3f}, "
                          f"cal_error={avg.get('cal_error', float('nan')):.3f}")


    _run_condition_set(condition_specs, optimal_internal_eval,
                       optimal_aug_strategy, optimal_rep_count,
                       optimal_rep_scores, results['conditions'],
                       _built_widths, _fold_errors)

    # Owner's set (2026-10-02): top 2 general + top 1 distance hybrids per
    # model, per family. Until then c08 took ONE recipe -- the best across
    # five models -- and built every condition from it. Now each downstream
    # model's three rule hybrids each get all four conditions, built with that
    # model. 'conditions' above keeps the single cross-model recipe.
    results['rule_conditions'] = {}
    results['rule_missing'] = []
    results['rule_base_reps'] = {}
    for _rm in MAIN_MODELS_WITH_KNN:
        try:
            _rule = get_rule_hybrids(
                reps, labels, groups, downstream_model=_rm, budget=None,
                dataset_name=dataset_name, augmentations=augs,
                task_type=task_type, data=data, fold_i=None)
        except Exception as _e:
            results['rule_missing'].append(
                f'{_rm}: the owner\'s set could not be read '
                f'({type(_e).__name__}: {_e})')
            continue
        results['rule_missing'] += [f'{_rm}: {m}' for m in _rule['missing']]
        results['rule_base_reps'][_rm] = [
            (b.get('rep'), b.get('score_other_folds'))
            for b in _rule['base_reps']]
        _fac = get_model_factories([_rm], task_type=task_type,
                                   n_samples=len(labels)).get(_rm)
        if _fac is None:
            results['rule_missing'].append(
                f'{_rm}: no model factory on this dataset, so its '
                f'{len(_rule["hybrids"])} rule hybrid(s) were not built')
            continue
        for _h in _rule['hybrids']:
            _bp = _h['blueprint']
            _key = f"{_h.get('objective')}|{_h.get('slot')}"
            _desc = (f"{_rm} {_key} at budget={_bp.budget} "
                     f"[{_bp.importance_method}/{_bp.allocation_method}] "
                     f"({_h.get('tier')}/{_h.get('aug_condition')})")
            print(f"\n  === rule hybrid: {_desc} ===")
            # 2026-10-05, owner delegated, 2026-10-05: underfit is set below
            # THIS rule hybrid's budget by the same rule as above.
            _uf_budget, _uf_reason = _underfit_for(_bp.budget)
            _specs = {
                'optimal': (_fac, _bp.budget, _bp.allocation_method,
                            _bp.importance_method, _desc),
                'underfit': (_fac, _uf_budget, _bp.allocation_method,
                             _bp.importance_method,
                             f'Same model at budget={_uf_budget}'),
                'overfit': (_fac, total_dim, _bp.allocation_method,
                            _bp.importance_method,
                            f'Same model at full dim={total_dim}'),
                'mismatched': (mismatched_factory, total_dim,
                               _bp.allocation_method, _bp.importance_method,
                               mismatched_desc),
            }
            if _uf_budget is None:
                _specs.pop('underfit')
            _out, _bw, _fe = {}, {}, {}
            _run_condition_set(
                _specs, _bp.internal_evaluator,
                _h.get('aug_condition') or 'none',
                getattr(_bp, 'rep_count', None),
                getattr(_bp, 'rep_scores', None),
                _out, _bw, _fe, recipe_objective=_h.get('objective'))
            results['rule_conditions'].setdefault(_rm, {})[_key] = {
                'slot': _h.get('slot'), 'lineage': _h.get('lineage'),
                'objective': _h.get('objective'), 'tier': _h.get('tier'),
                'aug_condition': _h.get('aug_condition'),
                'recipe': _h.get('recipe'), 'budget': _h.get('budget'),
                'budget_relaxed': _h.get('budget_relaxed'),
                'score_in_search': _h.get('score_other_folds'),
                'conditions': _out,
                'condition_widths_built': _bw,
                'fold_errors': _fe,
                # 2026-10-05, owner delegated: underfit budget asked for, in
                # features, and why it was not run when it was not.
                'underfit_budget': _uf_budget,
                'underfit_skip_reason': _uf_reason,
            }

    # Summary table
    print("\n  CALIBRATION SUMMARY:")
    if is_classification:
        print(f"  {'Condition':<14} {metric_name:<8} {'PICP':<8} {'SetSize':<8} {'Cal Error':<10} {'Budget':<8}")
        print("  " + "-" * 56)
        for cond_name, cond_data in results['conditions'].items():
            a = cond_data['avg']
            print(f"  {cond_name:<14} {a.get('val_score', float('nan')):.4f}  "
                  f"{a.get('picp', float('nan')):.3f}  "
                  f"{a.get('avg_set_size', float('nan')):.2f}    "
                  f"{a.get('cal_error', float('nan')):.4f}    {cond_data['budget']}")
    else:
        print(f"  {'Condition':<14} {metric_name:<8} {'PICP':<8} {'MPIW':<8} {'Cal Error':<10} {'Budget':<8}")
        print("  " + "-" * 56)
        for cond_name, cond_data in results['conditions'].items():
            a = cond_data['avg']
            print(f"  {cond_name:<14} {a.get('val_score', float('nan')):.4f}  "
                  f"{a.get('picp', float('nan')):.3f}  "
                  f"{a.get('mpiw', float('nan')):.3f}  "
                  f"{a.get('cal_error', float('nan')):.4f}    {cond_data['budget']}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C08: Uncertainty Calibration',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--c02-results', type=str, default=None,
                        help='Path to c02 output directory (reads optimal config from JSON)')
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        choices=['optimal', 'underfit', 'overfit', 'mismatched'],
                        help='Which conditions to test (default: all)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  conditions=args.conditions,
                  c02_results_dir=args.c02_results)
    save_experiment_results(results, args.output, 'c08_calibration', args.dataset)
