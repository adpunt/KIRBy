#!/usr/bin/env python3
"""C18: Degrees of Freedom — Diversity vs Depth.

Paper: Complexity
Core question: Given a fixed total budget, is it better to draw features from many
    reps (diversity) or concentrate on few reps (depth)? Tests whether the hybrid's
    allocation across rep families adds value beyond selecting the best single rep.

Conditions at fixed budget:
  - depth_1: all features from top-1 rep
  - depth_2: all features split between top-2 reps
  - depth_3: all features from top-3 reps
  - uniform_all: equal allocation across all reps
  - natural: let the hybrid allocate freely (greedy)

Usage:
    python c18_diversity_vs_depth.py --dataset logd --mode intermediary
"""
import sys
import os
import time
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
# Ranking metrics need graded scores: an AUROC read off hard 0/1 labels
# collapses to the mean of sensitivity and specificity (hybrid.py:2661).
from kirby.find_best_hybrid import _predict_for_metric
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric,
    compute_feature_importance,
    build_fold_hybrid, get_precomputed_search, get_best_for_downstream,
    get_best_search_for_downstream,
    build_fold_reselection_input,
    top_base_reps, top_hybrids, hybrid_deltas,
)


# =========================================================================
# PER-FOLD HELPERS
# =========================================================================

def _importance_for_reps(rep_arrays, labels, method, cache=None):
    """Per-representation importance, re-used inside one fold instead of refitted.

    ⛔ NOTHING IS APPROXIMATED HERE AND NO ARM'S FEATURES CHANGE. This returns
    the same arrays compute_feature_importance returns; it just stops asking for
    the same one repeatedly.

    Why it is safe. compute_feature_importance (hybrid.py:4629) loops over the
    dict it is handed and fits one model per representation on that
    representation's own features -- every one of its seven methods does, and
    the lightgbm_gain branch this run uses scales, variance-pre-screens and fits
    inside that per-representation loop (hybrid.py:5104-5145). So the array it
    returns for a representation does not depend on which other representations
    were in the dict, and two calls with the same representation, the same rows
    and the same method return the same numbers.

    What it saves, measured on flab_thermo_tresanco (5 representations, 1000
    features each after KIRBY_LOCAL_MAX_FEATURES_PER_REP, 672 rows, 3 folds,
    budgets 100 and 250). c18 asked for importance 16 times per fold per budget:
    5 in the fold ranking, 1 + 2 + 3 in the three depth arms, and 5 one
    representation at a time in the uniform arm -- and then asked for the whole
    fold again at the next budget, because only how many features get taken
    depends on the budget. That is 96 LightGBM fits where 15 distinct ones
    exist.

    Measured 2026-09-25 on this laptop with nine to fifteen other experiments
    running beside it, so every wall time below is a share of one core, not a
    clean cost. Without this, the run had not finished the first of its two
    budgets after 20 minutes and was killed. With it the same run finished in
    36.6 minutes, and the two budgets took 1145s and 1053s -- the second of
    which does no importance work at all, because the first already did it.
    """
    if cache is None:
        return compute_feature_importance(rep_arrays, labels, method=method)
    out, missing = {}, {}
    for name, X in rep_arrays.items():
        if (method, name) in cache:
            out[name] = cache[(method, name)]
        else:
            missing[name] = X
    if missing:
        for name, scores in compute_feature_importance(
                missing, labels, method=method).items():
            cache[(method, name)] = scores
            out[name] = scores
    return out


def _rank_reps_by_importance(reps, labels, method='lightgbm_gain', cache=None):
    """Rank reps by aggregate importance. Returns sorted list of (name, score).

    `method` is the search's own importance method, passed in by the caller. The
    four hand-built arms and the search's hybrid have to rank features the same
    way, or a win for the hybrid could be the allocation or could be the
    importance method and the two cannot be separated.
    """
    imp_dict = _importance_for_reps(reps, labels, method, cache)
    rep_scores = {name: float(np.sum(scores)) for name, scores in imp_dict.items()}
    return sorted(rep_scores.items(), key=lambda x: x[1], reverse=True)


def _select_top_features(train_reps, val_reps, y_train, rep_subset, budget,
                         method='lightgbm_gain', cache=None):
    """Per-fold: select top features by train-only importance, apply to both.

    `method` is the search's own importance method — see _rank_reps_by_importance.
    """
    sub_tr = {k: train_reps[k] for k in rep_subset if k in train_reps}
    sub_va = {k: val_reps[k] for k in rep_subset if k in val_reps}
    if not sub_tr:
        # Three values, not two. The caller unpacks three
        # (X_tr, X_va, _wn = _select_top_features(...)), so the two-value return
        # that stood here raised ValueError instead of skipping the arm.
        return None, None, None

    rep_names = sorted(sub_tr.keys())
    imp_dict = _importance_for_reps(sub_tr, y_train, method, cache)
    imp_concat = np.concatenate([imp_dict[n] for n in rep_names])

    X_tr = np.hstack([sub_tr[n] for n in rep_names])
    X_va = np.hstack([sub_va[n] for n in rep_names])

    # ⛔ THIS ARM FILLS THE BUDGET, THE 'natural' ARM DOES NOT. Taking the top
    # n_select by argsort keeps features whatever they scored, and
    # lightgbm_gain writes exactly 0.0 for every feature no tree split on -- so
    # past the count that scored above zero this arm is padded from a block of
    # exact ties, broken by column position. The 'natural' arm goes through
    # build_fold_hybrid -> create_hybrid, which STOPS at the last positively
    # scored feature. On flab at budget 1,000 that is roughly 1,000 features
    # against roughly 400.
    #
    # c18 already aligns the importance METHOD across the arms and warns when
    # it cannot (see the fallback warning in run). It does not align the
    # stopping rule. Not changed here -- which rule both arms should use is the
    # open zero-gain decision -- but the realised width is now returned so the
    # difference is visible instead of implied.
    n_select = min(budget, X_tr.shape[1])
    top_idx = np.argsort(imp_concat)[::-1][:n_select]
    _n_positive = int(np.sum(np.asarray(imp_concat) > 0))
    _width_note = {'built': int(len(top_idx)),
                   'scored_above_zero': _n_positive,
                   'would_be_if_cut': int(min(len(top_idx), _n_positive))}
    return X_tr[:, top_idx], X_va[:, top_idx], _width_note


def _uniform_allocation(train_reps, val_reps, y_train, rep_subset, budget,
                        method='lightgbm_gain', cache=None):
    """Per-fold: equal features per rep, top by train-only importance.

    `method` is the search's own importance method — see _rank_reps_by_importance.
    """
    sub_names = sorted([k for k in rep_subset if k in train_reps])
    if not sub_names:
        return None, None

    # The equal share has to add up to the SAME number of features the depth
    # arms get, or a loss for it is a loss on width and not on spreading the
    # budget. budget // n_reps threw the remainder away: at 10 representations
    # and a budget of 25 this arm ran on 20 features against the depth arms'
    # 25, and at a budget of 10 on 10 against 10 only by accident. The
    # remainder now goes one feature at a time to the representations in the
    # order they are named, and a representation narrower than its share hands
    # what it cannot fill back to the others.
    quota = {n: 0 for n in sub_names}
    width = {n: int(train_reps[n].shape[1]) for n in sub_names}
    remaining = int(min(budget, sum(width.values())))
    while remaining > 0:
        open_names = [n for n in sub_names if quota[n] < width[n]]
        if not open_names:
            break
        share, extra = divmod(remaining, len(open_names))
        if share == 0:
            for n in open_names[:remaining]:
                quota[n] += 1
            break
        for i, n in enumerate(open_names):
            take = min(share + (1 if i < extra else 0), width[n] - quota[n])
            quota[n] += take
        remaining = int(min(budget, sum(width.values()))) - sum(quota.values())

    tr_parts, va_parts = [], []
    for name in sub_names:
        n_sel = quota[name]
        if n_sel <= 0:
            continue
        imp_dict = _importance_for_reps(
            {name: train_reps[name]}, y_train, method, cache)
        imp = imp_dict[name]
        top_idx = np.argsort(imp)[::-1][:n_sel]
        tr_parts.append(train_reps[name][:, top_idx])
        va_parts.append(val_reps[name][:, top_idx])

    if not tr_parts:
        return None, None
    return np.hstack(tr_parts), np.hstack(va_parts)


def _fold_heldout_rep_scores(train_reps, y_train, groups_train, blueprint,
                             metric_fn):
    """{rep: held-out score on this fold's training rows}, higher is better.

    Added 2026-09-28. The same call create_hybrid makes to size each
    representation's share (hybrid.py allocate_performance_weighted ->
    _score_rep_for_budget): the recipe's internal evaluator, fitted on the
    internal carve of this fold's TRAINING rows (internal_holdout_split) and
    scored with the dataset's metric on the other part of those rows. No
    held-out fold row is read.

    Two uses. (1) When the recipe's rep_count is smaller than the pool,
    build_fold_hybrid keeps the best rep_count representations, and without
    scores passed in it ranks them by blueprint.rep_scores, which the search
    measured over EVERY row -- the three outer held-out folds included. Under
    the local run logd and buchwald_hartwig hold 6 representations and
    rep_count is 5, so the natural arm there dropped a representation chosen
    with the held-out labels. Same fix as c15's _fold_alloc_scores. (2) The
    depth arms rank by summed gain; this is written beside that ranking so
    the two can be compared. Nothing else reads it.

    Returns None for a recipe this cannot reproduce (evaluator 'match', a
    geometry objective), and the caller then leaves build_fold_hybrid as it
    was.
    """
    evaluator = getattr(blueprint, 'internal_evaluator', None)
    objective = getattr(blueprint, 'objective', None)
    from kirby.find_best_hybrid import (OBJECTIVE_PREDICTIVE,
                                        _is_geometry_objective)
    if (not isinstance(evaluator, str) or evaluator == 'match'
            or (objective is not None and (isinstance(objective, dict)
                                           or _is_geometry_objective(objective)))):
        return None
    from kirby.hybrid import _score_rep_for_budget, internal_holdout_split
    from experiments._shared import object_direction
    lib = object_direction(blueprint, 'the c18 recipe')
    inner_tr, inner_va = internal_holdout_split(
        len(y_train), y_train, groups_train)
    out = {}
    for name in sorted(train_reps):
        X = train_reps[name]
        Xtr = np.nan_to_num(np.clip(X[inner_tr], -1e10, 1e10),
                            nan=0.0, posinf=1e10, neginf=-1e10)
        Xva = np.nan_to_num(np.clip(X[inner_va], -1e10, 1e10),
                            nan=0.0, posinf=1e10, neginf=-1e10)
        out[name] = float(_score_rep_for_budget(
            Xtr, Xva, y_train[inner_tr], y_train[inner_va],
            internal_evaluator=evaluator, metric_fn=metric_fn,
            lower_is_better=lib,
            objective=objective or OBJECTIVE_PREDICTIVE, is_clf=False))
    return out


# =========================================================================
# RUN
# =========================================================================

def _score_prediction(model, X_val, y_val, metric_fn, task_type):
    """Score model predictions appropriately for the metric.

    Classification + binary: AUROC needs class-1 probabilities, not labels.
    Classification + multi-class: F1 needs class labels, not probabilities.
    Regression: anything that has .predict() works.
    """
    # 2026-09-28: the probability branch that stood here handed class-1
    # probabilities to EVERY binary metric, not only auc, and relied on an
    # accuracy or f1 call raising on them to reach the line below.
    # _predict_for_metric gives probabilities to auc alone, so auc scores are
    # unchanged and no other metric can be fed probabilities.
    return metric_fn(y_val, _predict_for_metric(model, X_val, metric_fn))


def run(data, config, output_dir, dataset_name, **kwargs):
    """C18: Diversity vs Depth."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C18: DIVERSITY VS DEPTH")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, metric_lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    # The roster's rf (find_best_hybrid.DEFAULT_MODELS: 100 trees,
    # max_features='sqrt', random_state=42), the model the search below is
    # asked for and the one the top-2 rows and the two deltas are scored with.
    # Until 2026-09-28 every arm here fitted sklearn's default forest instead,
    # which for regression tries every feature at every split: the five arm
    # scores came from a different model than the numbers written beside them
    # under the name rf. Classification was unaffected (sklearn's classifier
    # already defaults to sqrt). Threads from KIRBY_RF_JOBS as in c15/c16/c22;
    # a seeded forest gives the same trees at any thread count.
    from kirby.find_best_hybrid import get_model as _get_model
    _rf_jobs = int(os.environ.get('KIRBY_RF_JOBS') or -1)

    def _RF():
        m = _get_model('rf', task_type)
        m.set_params(n_jobs=_rf_jobs)
        return m

    # Budget source priority: explicit kwargs override > config['budgets']
    # (set by get_config from FULL_BUDGETS, or --budgets) > legacy [50, 100]
    # fallback. CXA-14: previously this was hardcoded to kwargs.get(..., [50,100])
    # and __main__ never passed budgets, so the sweep was permanently locked to
    # [50, 100] regardless of --budgets / config. Filter to budgets that are
    # meaningful given the feature pool (a budget above total_dim collapses to
    # full-dim and adds no information to the diversity-vs-depth contrast).
    total_dim = sum(v.shape[1] for v in reps.values())
    budgets = kwargs.get('budgets') or config.get('budgets') or [50, 100]
    budgets = sorted({b for b in budgets if b <= total_dim} or {min(budgets)})
    all_rep_names = sorted(reps.keys())
    n_reps = len(all_rep_names)
    sample_X = next(iter(reps.values()))

    print(f"  {n_reps} reps, budgets: {budgets}")

    # Full-data ranking for reporting only (per-fold ranking used in evaluation)
    #
    # KIRBY_C18_FULL_DATA_RANKING: 1, the default, is what the server runs and
    # nothing about it changes. 0 skips this ranking.
    #
    # What it costs when skipped: 'rep_ranking_full_data' below comes back empty
    # and 'rep_ranking_full_data_skipped' is written beside it, so an empty list
    # cannot be read as "no representation scored". SELECTION_SHAPE_PLAN
    # 2026_09_18 has an open change to this key -- it is the only thing outside
    # this file that names it -- so the key itself stays.
    # What it buys: 5 LightGBM fits over all 672 rows. Two runs on 2026-09-25,
    # same grid and same dataset, gave identical scores for all ten arm x budget
    # cells: with this ranking 2694s, without it 2198s. Those two runs had
    # different numbers of other experiments beside them on this laptop, so 496s
    # is the gap between them and not a clean cost for the ranking alone. 2198s
    # is inside the 40-minute local cap and 2694s is not.
    # Why 0 loses nothing measurable: this ranks by summed importance over EVERY
    # row, the three held-out folds included, so no arm's score and no held-out
    # number can be read off it. Each fold ranks its representations again on
    # its own training rows, and that is what depth_1, depth_2 and depth_3 use.
    _do_full_ranking = os.environ.get('KIRBY_C18_FULL_DATA_RANKING', '1') != '0'
    _t0 = time.time()
    if _do_full_ranking:
        ranked_full = _rank_reps_by_importance(reps, labels)
        print(f"  Rep ranking (full data, reference only): "
              f"{', '.join(f'{n}({s:.2f})' for n, s in ranked_full[:5])}"
              f"... [{time.time() - _t0:.0f}s]")
    else:
        ranked_full = []
        print("  Rep ranking (full data, reference only): SKIPPED "
              "(KIRBY_C18_FULL_DATA_RANKING=0). Every fold still ranks its own "
              "representations; only the whole-row reference number is gone.")

    results = {
        'metric': metric_name,
        # Direction of every score below. Added 2026-09-28: the reader
        # (firstpass_notready.lower_is_better) reads this flag first and
        # otherwise falls back to the registry, so the file now says it.
        'lower_is_better': bool(metric_lower_is_better),
        'budgets': budgets,
        # Which importance method every arm at this budget ranked features with
        # (the search's own choice). Keyed by budget.
        'importance_method_by_budget': {},
        # 'search' or 'fallback', per budget. Without it a stored
        # 'lightgbm_gain' cannot be told apart from a failed pool lookup.
        'importance_method_source_by_budget': {},
        'n_reps': n_reps,
        'rep_ranking_full_data': [(n, float(s)) for n, s in ranked_full],
        # True means the list above is empty because the ranking was
        # skipped, not because nothing scored. See the knob above.
        'rep_ranking_full_data_skipped': not _do_full_ranking,
        'per_budget': {},
        # Honest-comparison block, keyed by budget (this experiment runs one
        # search per budget, so the single top-level "fold_reselection" block the
        # contract describes can't represent it -- one block per budget instead).
        # Each value is verbatim what build_fold_reselection_input(search) returns:
        # the SEARCH's own hybrid candidate recipes + per-rep baselines with
        # per-fold scores, which is where the natural arm's upstream selection
        # optimism lives. A reader picks a budget and calls fold_reselection(
        # block['candidates_by_model'], block['baselines_by_model'],
        # block['lower_is_better']).
        'fold_reselection_by_budget': {},
        # --- Added 2026-09-18, beside the five arms; nothing above changed. ---
        # This file reported one number per arm (depth_1, depth_2, depth_3,
        # uniform_all, natural) and nothing about the individual
        # representations. So a depth_1 arm that drew its features from a weak
        # representation and one that drew them from the best of the N in `reps`
        # produced the same-looking row, and rep_ranking_full_data above ranks by
        # summed lightgbm importance on the whole dataset, which is not a score
        # in the dataset's metric. These three keep the top 2 by MEASURED score.
        #
        # Keyed by budget, because this experiment runs one search per budget
        # and the winner can differ between budgets.
        #
        # They live at the top level and NOT inside per_budget[budget]:
        # firstpass_notready.py:686 reads per_budget and :691-693 treats every
        # key inside per_budget[budget] as an arm carrying a 'score', so a key
        # added there would be read as a sixth arm and enter the depth-vs-
        # diversity count.
        #
        # All three read rows the search already stored. 0 new model fits,
        # 0 new folds, 0 new representations built.
        # Top 2 base representations by the search's own mean over its folds,
        # downstream model 'rf'. {budget: top_base_reps(...) output}. Every
        # representation in the pool is eligible here, whatever its width: a
        # single representation is used whole and the budget does not cut it.
        'top2_base_reps_by_budget': {},
        # The same ranking with the representations wider than the budget barred
        # from selection, so this key and 'top2_hybrids_by_budget' are filtered
        # the same way. {budget: top_base_reps(..., max_features=budget)}.
        # Empty 'selected' means no representation in the pool is that narrow.
        'top2_base_reps_within_budget': {},
        # Top 2 hybrid recipes at that budget, downstream model 'rf'.
        # {budget: top_hybrids(...) output}.
        'top2_hybrids_by_budget': {},
        # The two reference numbers for the top hybrid at that budget:
        # delta_vs_source_mean and delta_vs_best_rep. {budget: hybrid_deltas(...)}.
        'hybrid_deltas_by_budget': {},
        # Added 2026-09-28. {budget: [per fold: {gain_sum_ranking,
        # heldout_scores, natural_reps_kept}]}. gain_sum_ranking is the order
        # the depth arms took representations in. heldout_scores is each
        # representation's score on the internal carve of that fold's
        # training rows, higher is better, the quantity the natural arm sizes
        # its shares by. natural_reps_kept is which representations the
        # natural arm was built from.
        'fold_rep_rankings_by_budget': {},
    }

    # Per-budget blueprint discovery via pool helper (best (tier, aug_cond)
    # for downstream_model='rf' at each budget; may differ across budgets).
    augmentations = data.get('augmentations')

    # Importance arrays already computed, keyed by which training rows they were
    # computed on. Two budgets walk the SAME folds and differ only in how many
    # features each arm takes, so every importance array the second budget needs
    # was computed under the first. See _importance_for_reps for why re-using
    # them changes no number.
    _imp_memo = {}

    for budget in budgets:
        # {arm: [{built, scored_above_zero, would_be_if_cut}, per fold]}. The
        # depth and uniform arms FILL the budget; the natural arm stops at the
        # last positively scored feature. Recorded so the comparison is not
        # read as width-matched when it is not.
        _arm_widths = {}
        print(f"\n  Budget={budget}:")
        try:
            search, meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model='rf', budget=budget,
                augmentations=augmentations,
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
        except Exception as e:
            print(f"  WARNING: pool search failed: {e}")
            search, meta = None, None

        # The importance method the search picked for this budget. The four
        # hand-built arms below rank features with it, so the only thing that
        # differs between them and the 'natural' arm is how the budget is split
        # across representations. Without this the hand-built arms ranked by
        # lightgbm_gain while the search may have chosen deeplift or treeshap,
        # and the two differences moved together.
        # ⛔ WHERE THE IMPORTANCE METHOD CAME FROM IS RECORDED SINCE 2026-09-18.
        #
        # This line read 'lightgbm_gain' whether the search had chosen
        # lightgbm_gain or the pool lookup above had failed, and the result file
        # stored the same string either way. So a c18 result showing
        # lightgbm_gain could be the arms agreeing with the search, or it could
        # be the run silently reinstating the exact confound the comment above
        # says this lookup exists to remove -- and nothing on disk told the two
        # apart. The failure printed a WARNING to the job log, which is not
        # where anyone reads a result from.
        #
        # 'search' means the search chose it. 'fallback' means the lookup
        # returned nothing or raised and the arms ran on the default anyway.
        fold_imp_method = 'lightgbm_gain'
        fold_imp_source = 'fallback'
        if meta is not None:
            try:
                fold_imp_method = meta['best_config']['blueprint'].importance_method
                fold_imp_source = 'search'
            except Exception as _e:
                print(f"    WARNING: the search's recipe carried no importance "
                      f"method ({_e}); falling back")
        if fold_imp_source == 'fallback':
            print(f"    ⛔ importance method for all arms: {fold_imp_method} "
                  f"— FALLBACK, not the search's choice. The depth arms and "
                  f"the natural arm may now differ in two ways, not one.")
        else:
            print(f"    importance method for all arms: {fold_imp_method} "
                  f"(from the search)")
        results['importance_method_by_budget'][budget] = fold_imp_method
        results['importance_method_source_by_budget'][budget] = fold_imp_source

        # Persist the fold-reselection input built from the SEARCH itself (all
        # hybrid candidate recipes + per-rep baseline per-fold scores). Pure
        # arithmetic, no model fitting; lets the reader re-pick the winner per
        # held-out fold and expose the natural arm's selection optimism.
        if search is not None:
            try:
                results['fold_reselection_by_budget'][budget] = convert_numpy(
                    build_fold_reselection_input(search))
            except Exception as e:
                print(f"  WARNING: fold_reselection build failed: {e}")

        # --- Top 2 rows and the two deltas (see the keys declared above) ---
        # downstream_model='rf': every arm below fits a RandomForest (_RF), and
        # the search above was asked for 'rf'. A top 2 taken under another
        # downstream model would not be on the same footing as the five arm
        # scores this file prints.
        # require_full=False: a search that scored fewer than 2 representations
        # or fewer than 2 recipes for 'rf' returns a short list with a
        # shortfall_reason instead of raising, which would otherwise end the
        # budget sweep partway through and leave the remaining budgets unrun.
        #
        # THE TWO HALVES AND THE BUDGET, settled 2026-09-28.
        # top_hybrids is given budget=budget, so it ranks only the recipes that
        # were allowed that many features. top_base_reps has no budget argument
        # at all -- a single representation is fed to the model whole, and the
        # budget never cuts it -- so the unfiltered ranking below is every
        # representation in the pool, at whatever width it has.
        #
        # The two are therefore filtered differently, and the fix is NOT to put
        # the budget on the base side and leave it there: top_base_reps'
        # max_features bars a representation wider than the limit from selection
        # outright, and the budgets swept here start at 10 features while the
        # narrowest representations in the pool are wider than that, so a single
        # max_features=budget call empties the base half at the small budgets
        # and the comparison loses one of its two sides.
        #
        # So both are recorded. 'top2_base_reps_by_budget' keeps every
        # representation eligible, which is the side that matches how the five
        # arms below use a representation. 'top2_base_reps_within_budget' is the
        # same ranking with max_features=budget, filtered exactly as the hybrid
        # side is, and its 'dropped' field names each representation barred and
        # its feature count. A reader comparing a base number against a hybrid
        # number at the same budget reads the second key.
        if search is not None:
            _top_hyb = None
            try:
                results['top2_base_reps_by_budget'][budget] = convert_numpy(
                    top_base_reps(search, 'rf', 2, require_full=False,
                                  context=f'c18 {dataset_name} budget={budget}'))
            except Exception as e:
                print(f"    WARNING: top 2 base representations failed: {e}")
            try:
                results['top2_base_reps_within_budget'][budget] = convert_numpy(
                    top_base_reps(
                        search, 'rf', 2, max_features=budget,
                        require_full=False,
                        context=f'c18 {dataset_name} budget={budget} '
                                f'(representations no wider than the budget)'))
            except Exception as e:
                print(f"    WARNING: top 2 base representations no wider than "
                      f"{budget} features failed: {e}")
            try:
                _top_hyb = top_hybrids(
                    search, 'rf', 2, budget=budget, require_full=False,
                    context=f'c18 {dataset_name} budget={budget}')
                results['top2_hybrids_by_budget'][budget] = convert_numpy(_top_hyb)
            except Exception as e:
                print(f"    WARNING: top 2 hybrid recipes failed: {e}")
            try:
                # `hybrid=` is passed rather than left to default. With
                # hybrid=None, hybrid_deltas picks its own top recipe and reads
                # ['row'] off the selected entry (_shared.py:7432), and the
                # entry top_hybrids builds (_shared.py:7043-7066) carries no
                # 'row' key, so that path raises KeyError. The entry itself
                # carries score_mean and blueprint, which is everything the two
                # deltas are computed from.
                if _top_hyb is not None and _top_hyb['selected']:
                    results['hybrid_deltas_by_budget'][budget] = convert_numpy(
                        hybrid_deltas(
                            search, 'rf', hybrid=_top_hyb['selected'][0],
                            budget=budget,
                            context=f'c18 {dataset_name} budget={budget}'))
            except Exception as e:
                print(f"    WARNING: the two hybrid deltas failed: {e}")

        fold_scores = {}  # {condition_name: [per_fold_scores]}
        # Per fold: the depth arms' ranking (summed gain) beside the held-out
        # ranking the natural arm uses, and which representations the natural
        # arm was restricted to. Top level, not in per_budget (see the note on
        # the top-2 keys above).
        _fold_rankings = []
        # Seconds spent inside build_fold_hybrid, summed over this
        # budget's folds, against the seconds the whole budget took.
        # Printed so a slow run says WHICH arm was slow instead of
        # leaving it to be guessed.
        _budget_t0 = time.time()
        _natural_secs = 0.0

        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                           labels, mode=mode):
            train_reps = {name: X[train_idx] for name, X in reps.items()}
            val_reps_fold = {name: X[val_idx] for name, X in reps.items()}
            y_train = labels[train_idx]
            y_val = labels[val_idx]

            # One importance store per set of training rows. The key is the
            # training row indices themselves, not a fold counter, so it cannot
            # hand one fold's importance to another if cv_split ever yields the
            # folds in a different order at the second budget.
            _imp_cache = _imp_memo.setdefault(
                hashlib.md5(np.ascontiguousarray(
                    np.asarray(train_idx, dtype=np.int64)).tobytes()).hexdigest(),
                {})

            # Rank reps on training data only
            fold_ranked = _rank_reps_by_importance(train_reps, y_train,
                                                   method=fold_imp_method,
                                                   cache=_imp_cache)

            # Depth conditions (top-1, top-2, top-3)
            for n_top in [1, 2, 3]:
                if n_top > n_reps:
                    continue
                cond_name = f'depth_{n_top}'
                top_rep_names = [name for name, _ in fold_ranked[:n_top]]
                X_tr, X_va, _wn = _select_top_features(
                    train_reps, val_reps_fold, y_train, top_rep_names, budget,
                    method=fold_imp_method, cache=_imp_cache)
                if _wn:
                    _arm_widths.setdefault(cond_name, []).append(_wn)
                if X_tr is not None:
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_va_s = scaler.transform(X_va)
                    rf = _RF()
                    rf.fit(X_tr_s, y_train)
                    fold_scores.setdefault(cond_name, []).append(
                        _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))

            # Uniform allocation across all reps
            X_tr, X_va = _uniform_allocation(
                train_reps, val_reps_fold, y_train, all_rep_names, budget,
                method=fold_imp_method, cache=_imp_cache)
            if X_tr is not None:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)
                rf = _RF()
                rf.fit(X_tr_s, y_train)
                fold_scores.setdefault('uniform_all', []).append(
                    _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))

            # Natural hybrid (per-fold via search blueprint).
            # depth_N and uniform_all operate on base reps only, so the natural
            # arm must also be base-reps-only to keep the comparison apples-to-
            # apples. Augmentations were inflating natural's feature pool.
            _rank_row = {'gain_sum_ranking': [n for n, _ in fold_ranked],
                         'heldout_scores': None,
                         'natural_reps_kept': None}
            _fold_rankings.append(_rank_row)
            if meta is not None:
                try:
                    best_config = meta['best_config']
                    if 'blueprint' in best_config:
                        bp = best_config['blueprint']
                        _nt0 = time.time()
                        _g_tr = (groups[train_idx] if groups is not None else None)
                        _ho = _fold_heldout_rep_scores(
                            train_reps, y_train, _g_tr, bp, metric_fn)
                        _rank_row['heldout_scores'] = _ho
                        # Passed on every fold since 2026-09-28 (second pass).
                        # They are the numbers create_hybrid computes for itself
                        # (same evaluator, carve, clipping, metric, direction),
                        # checked equal on synthetic data with 5 and 6
                        # representations, so a fold that keeps every
                        # representation builds the same features and skips one
                        # evaluator fit per representation. When rep_count cuts
                        # they also pick which representations are kept.
                        _alloc = _ho
                        X_tr_h, X_va_h, _fi = build_fold_hybrid(
                            train_reps, val_reps_fold, y_train, bp,
                            augmentations_train=None,
                            augmentations_val=None,
                            groups=_g_tr,
                            model=_RF(),
                            metric_fn=metric_fn,
                            precomputed_alloc_scores=_alloc)
                        _rank_row['natural_reps_kept'] = (
                            _fi.get('rep_count_kept') or sorted(train_reps))
                        _natural_secs += time.time() - _nt0
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr_h)
                        X_va_s = scaler.transform(X_va_h)
                        rf = _RF()
                        rf.fit(X_tr_s, y_train)
                        _arm_widths.setdefault('natural', []).append(
                            {'built': int(X_tr_h.shape[1]),
                             'scored_above_zero': None,
                             'would_be_if_cut': int(X_tr_h.shape[1])})
                        fold_scores.setdefault('natural', []).append(
                            _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))
                except Exception as e:
                    print(f"    natural fold FAILED: {e}")

        # Aggregate per-fold scores
        conditions = {}
        for cond_name, scores in fold_scores.items():
            score = float(np.mean(scores)) if scores else float('nan')
            _w = _arm_widths.get(cond_name) or []
            conditions[cond_name] = {
                'score': score,
                # What this arm actually built, per fold, against the budget.
                'width_built_per_fold': [x['built'] for x in _w],
                'width_scored_above_zero_per_fold': [
                    x['scored_above_zero'] for x in _w],
                # OTHER-28: this is the REQUESTED budget, not the realised
                # feature count (each arm selects min(budget, available); depth_1
                # may select fewer than budget when the top rep is small). Label
                # it accurately as 'budget' rather than the misleading 'n_features'.
                'budget': budget,
                # Honest-comparison: persist the raw per-fold scores (both the
                # depth/uniform baseline arms and the 'natural' hybrid arm) so
                # the mean is no longer the only signal downstream.
                'fold_scores': [float(s) for s in scores],
            }
            print(f"    {cond_name}: {metric_name}={score:.4f}")

        _budget_secs = time.time() - _budget_t0
        print(f"    [budget {budget}: {_budget_secs:.0f}s over {n_folds} "
              f"folds, of which {_natural_secs:.0f}s rebuilding the "
              f"natural arm's hybrid]")
        results['per_budget'][budget] = convert_numpy(conditions)
        results['fold_rep_rankings_by_budget'][budget] = convert_numpy(
            _fold_rankings)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C18: Diversity vs Depth')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c18_diversity_vs_depth',
                            args.dataset)
