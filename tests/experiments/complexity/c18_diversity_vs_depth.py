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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
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

def _rank_reps_by_importance(reps, labels, method='lightgbm_gain'):
    """Rank reps by aggregate importance. Returns sorted list of (name, score).

    `method` is the search's own importance method, passed in by the caller. The
    four hand-built arms and the search's hybrid have to rank features the same
    way, or a win for the hybrid could be the allocation or could be the
    importance method and the two cannot be separated.
    """
    imp_dict = compute_feature_importance(reps, labels, method=method)
    rep_scores = {name: float(np.sum(scores)) for name, scores in imp_dict.items()}
    return sorted(rep_scores.items(), key=lambda x: x[1], reverse=True)


def _select_top_features(train_reps, val_reps, y_train, rep_subset, budget,
                         method='lightgbm_gain'):
    """Per-fold: select top features by train-only importance, apply to both.

    `method` is the search's own importance method — see _rank_reps_by_importance.
    """
    sub_tr = {k: train_reps[k] for k in rep_subset if k in train_reps}
    sub_va = {k: val_reps[k] for k in rep_subset if k in val_reps}
    if not sub_tr:
        return None, None

    rep_names = sorted(sub_tr.keys())
    imp_dict = compute_feature_importance(sub_tr, y_train, method=method)
    imp_concat = np.concatenate([imp_dict[n] for n in rep_names])

    X_tr = np.hstack([sub_tr[n] for n in rep_names])
    X_va = np.hstack([sub_va[n] for n in rep_names])

    n_select = min(budget, X_tr.shape[1])
    top_idx = np.argsort(imp_concat)[::-1][:n_select]
    return X_tr[:, top_idx], X_va[:, top_idx]


def _uniform_allocation(train_reps, val_reps, y_train, rep_subset, budget,
                        method='lightgbm_gain'):
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
        imp_dict = compute_feature_importance(
            {name: train_reps[name]}, y_train, method=method)
        imp = imp_dict[name]
        top_idx = np.argsort(imp)[::-1][:n_sel]
        tr_parts.append(train_reps[name][:, top_idx])
        va_parts.append(val_reps[name][:, top_idx])

    if not tr_parts:
        return None, None
    return np.hstack(tr_parts), np.hstack(va_parts)


# =========================================================================
# RUN
# =========================================================================

def _score_prediction(model, X_val, y_val, metric_fn, task_type):
    """Score model predictions appropriately for the metric.

    Classification + binary: AUROC needs class-1 probabilities, not labels.
    Classification + multi-class: F1 needs class labels, not probabilities.
    Regression: anything that has .predict() works.
    """
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_val)
            if probs.shape[1] == 2:
                return metric_fn(y_val, probs[:, 1])
        except Exception:
            pass
    return metric_fn(y_val, model.predict(X_val))


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
    metric_name, metric_fn, _ = get_domain_metric(data)
    task_type = data['task_type']
    _RF = RandomForestClassifier if task_type == 'classification' else RandomForestRegressor

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
    ranked_full = _rank_reps_by_importance(reps, labels)
    print(f"  Rep ranking (full data, reference only): "
          f"{', '.join(f'{n}({s:.2f})' for n, s in ranked_full[:5])}...")

    results = {
        'metric': metric_name,
        'budgets': budgets,
        # Which importance method every arm at this budget ranked features with
        # (the search's own choice). Keyed by budget.
        'importance_method_by_budget': {},
        # 'search' or 'fallback', per budget. Without it a stored
        # 'lightgbm_gain' cannot be told apart from a failed pool lookup.
        'importance_method_source_by_budget': {},
        'n_reps': n_reps,
        'rep_ranking_full_data': [(n, float(s)) for n, s in ranked_full],
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
        # downstream model 'rf'. {budget: top_base_reps(...) output}.
        'top2_base_reps_by_budget': {},
        # Top 2 hybrid recipes at that budget, downstream model 'rf'.
        # {budget: top_hybrids(...) output}.
        'top2_hybrids_by_budget': {},
        # The two reference numbers for the top hybrid at that budget:
        # delta_vs_source_mean and delta_vs_best_rep. {budget: hybrid_deltas(...)}.
        'hybrid_deltas_by_budget': {},
    }

    # Per-budget blueprint discovery via pool helper (best (tier, aug_cond)
    # for downstream_model='rf' at each budget; may differ across budgets).
    augmentations = data.get('augmentations')

    for budget in budgets:
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
        if search is not None:
            _top_hyb = None
            try:
                results['top2_base_reps_by_budget'][budget] = convert_numpy(
                    top_base_reps(search, 'rf', 2, require_full=False,
                                  context=f'c18 {dataset_name} budget={budget}'))
            except Exception as e:
                print(f"    WARNING: top 2 base representations failed: {e}")
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

        for train_idx, val_idx in cv_split(groups, n_folds, sample_X,
                                           labels, mode=mode):
            train_reps = {name: X[train_idx] for name, X in reps.items()}
            val_reps_fold = {name: X[val_idx] for name, X in reps.items()}
            y_train = labels[train_idx]
            y_val = labels[val_idx]

            # Rank reps on training data only
            fold_ranked = _rank_reps_by_importance(train_reps, y_train,
                                                   method=fold_imp_method)

            # Depth conditions (top-1, top-2, top-3)
            for n_top in [1, 2, 3]:
                if n_top > n_reps:
                    continue
                cond_name = f'depth_{n_top}'
                top_rep_names = [name for name, _ in fold_ranked[:n_top]]
                X_tr, X_va = _select_top_features(
                    train_reps, val_reps_fold, y_train, top_rep_names, budget,
                    method=fold_imp_method)
                if X_tr is not None:
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_va_s = scaler.transform(X_va)
                    rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                    rf.fit(X_tr_s, y_train)
                    fold_scores.setdefault(cond_name, []).append(
                        _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))

            # Uniform allocation across all reps
            X_tr, X_va = _uniform_allocation(
                train_reps, val_reps_fold, y_train, all_rep_names, budget,
                method=fold_imp_method)
            if X_tr is not None:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)
                rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_tr_s, y_train)
                fold_scores.setdefault('uniform_all', []).append(
                    _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))

            # Natural hybrid (per-fold via search blueprint).
            # depth_N and uniform_all operate on base reps only, so the natural
            # arm must also be base-reps-only to keep the comparison apples-to-
            # apples. Augmentations were inflating natural's feature pool.
            if meta is not None:
                try:
                    best_config = meta['best_config']
                    if 'blueprint' in best_config:
                        bp = best_config['blueprint']
                        X_tr_h, X_va_h, _fi = build_fold_hybrid(
                            train_reps, val_reps_fold, y_train, bp,
                            augmentations_train=None,
                            augmentations_val=None,
                            groups=(groups[train_idx] if groups is not None else None),
                            model=_RF(n_estimators=100, random_state=42, n_jobs=-1),
                            metric_fn=metric_fn)
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr_h)
                        X_va_s = scaler.transform(X_va_h)
                        rf = _RF(n_estimators=100, random_state=42, n_jobs=-1)
                        rf.fit(X_tr_s, y_train)
                        fold_scores.setdefault('natural', []).append(
                            _score_prediction(rf, X_va_s, y_val, metric_fn, task_type))
                except Exception as e:
                    print(f"    natural fold FAILED: {e}")

        # Aggregate per-fold scores
        conditions = {}
        for cond_name, scores in fold_scores.items():
            score = float(np.mean(scores)) if scores else float('nan')
            conditions[cond_name] = {
                'score': score,
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

        results['per_budget'][budget] = convert_numpy(conditions)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C18: Diversity vs Depth')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c18_diversity_vs_depth',
                            args.dataset)
