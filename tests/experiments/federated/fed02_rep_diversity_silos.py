#!/usr/bin/env python3
"""Fed02: Representation Diversity as Federated Silos.

Tests whether KIRBy's feature-level hybrid outperforms FLuID's prediction-level
ensemble when silos contribute different molecular representations (not different
data). Each silo sees the same molecules but through a different computational
lens (e.g., ECFP4, Mordred, MolFormer, PDV, ChemBERTa).

6 conditions:
    1. Best single silo — best individual rep baseline
    2. FLuID similarity ensemble — 8-NN weighted mean of regressor predictions
    3. FLuID quality ensemble — inverse-RMSE weighted mean
    4. KIRBy hybrid — feature-level fusion across all 5 silo reps
    5. KIRBy top-tier hybrid — feature-level fusion on top 3 reps only
    6. FLuID→KIRBy pipeline — intra-silo clustering + cross-silo fusion (novel)

Datasets: herg (classification), logd, caco2 (regression).

Usage:
    python fed02_rep_diversity_silos.py --dataset herg --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import traceback
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split,
    get_precomputed_search, get_best_search_for_downstream, get_domain_metric,
)
from experiments.federated.federated_helpers import (
    compute_reg_metrics,
    compute_clf_metrics,
    predict_fluid_similarity_ensemble_regression,
    predict_fluid_quality_ensemble_regression,
    predict_fluid_similarity_ensemble_classification,
    predict_fluid_quality_ensemble_classification,
    fluid_within_kirby_across,
    smoke_n_folds, smoke_search_kwargs, federated_budgets,
    fit_fed_model,
    fed_to_kirby_model,
    make_fed_classifier,
    make_fed_regressor,
)


# ── Constants ────────────────────────────────────────────────────────────────

# 5 silo representations
SILO_REPS = {
    'A': 'ecfp4',        # structural fingerprints, 2048D
    'B': 'mordred',       # physicochemical descriptors, ~1600D
    'C': 'molformer',     # foundation model embeddings, 768D
    'D': 'pdv',           # RDKit descriptors, 200D
    'E': 'chemberta',     # learned SMILES embeddings, 768D
}

# Fallback for datasets where chemberta is unavailable
SILO_REPS_FALLBACK = {
    'caco2': {**SILO_REPS, 'E': 'avalon'},
}

K_INTRA = 3   # sub-clusters per silo for Condition 6
# Feature budgets the KIRBy arms sweep. KIRBY_FED_BUDGETS overrides it.
BUDGETS = federated_budgets()
HEADLINE_BUDGET = 100 if 100 in BUDGETS else BUDGETS[0]
# Folds of the inner split C5's three representations are chosen on. Three, not
# five: this runs inside every outer fold, so the cost is
# 5 representations x INNER_RANK_FOLDS fits per outer fold.
INNER_RANK_FOLDS = 3


def _rank_reps_train_only(silo_letters, silo_reps, train_reps, y_train,
                          groups, fed_model, fed_seed, task_type, is_clf,
                          compute_metrics, n_inner_folds=INNER_RANK_FOLDS):
    """Score each silo representation inside the training rows only.

    Returns {silo_letter: mean inner score} — MCC for classification, RMSE for
    regression, the same two numbers C5 was previously ranked on, but measured
    on rows the outer fold's test set never touches.

    The inner split is group-aware; with too few distinct scaffold groups in
    the training rows it raises rather than falling back to a random split
    (feedback_no_random_splits.md).
    """
    if groups is None:
        raise ValueError(
            "_rank_reps_train_only: no groups for the training rows — "
            "ranking representations on a random inner split is forbidden.")
    n_unique = len(np.unique(groups))
    if n_unique < n_inner_folds:
        raise ValueError(
            f"_rank_reps_train_only: {n_unique} distinct scaffold groups in "
            f"the training rows but n_inner_folds={n_inner_folds}.")

    per_rep = {s: [] for s in silo_letters}
    inner = list(cv_split(groups, n_inner_folds,
                          np.zeros(len(y_train)), y_train, seed=fed_seed,
                          task_type=task_type))
    for in_tr, in_va in inner:
        for s in silo_letters:
            X = train_reps[silo_reps[s]]
            mdl = fit_fed_model(fed_model, X[in_tr], y_train[in_tr],
                                seed=fed_seed, task_type=task_type)
            y_hat = mdl.predict(X[in_va])
            if is_clf:
                y_prob = (mdl.predict_proba(X[in_va])[:, 1]
                          if hasattr(mdl, 'predict_proba') else None)
                m = compute_metrics(y_train[in_va], y_hat, y_prob=y_prob)
                per_rep[s].append(m['mcc'])
            else:
                m = compute_metrics(y_train[in_va], y_hat)
                per_rep[s].append(m['rmse'])
    return {s: float(np.mean(v)) for s, v in per_rep.items()}


# Substitutes tried, in order, when a silo's own representation is not in the
# dataset's pool. A substitute already taken by another silo is skipped: the
# whole experiment is five silos holding five DIFFERENT representations, and
# two silos on the same one is not a weaker version of that question, it is a
# different question.
_SILO_FALLBACKS = ['chembert', 'avalon', 'graph_kernel', 'mordred', 'pdv',
                   'atom_pair', 'top_torsion', 'mhg_gnn', 'grover', 'smited']


def _get_silo_reps(dataset_name, available_reps):
    """Resolve silo reps for a dataset, using fallbacks if needed.

    No representation is ever given to two silos. Until 2026-09-12 the
    duplicate check ran only on the fallback path, so a silo whose OWN
    representation was in the pool took it even when an earlier silo had
    already been given the same one as a substitute. On caco2 that put avalon
    in silo B (standing in for mordred, which caco2's pool lacks) and again in
    silo E (avalon is E's own representation there), so two of the five silos
    held identical features and the run answered a four-silo question while
    reporting five.
    """
    import warnings
    base = SILO_REPS_FALLBACK.get(dataset_name, SILO_REPS)
    resolved = {}
    # Own representation first, so a silo that has its own is never displaced
    # by another silo's substitute.
    deferred = []
    for silo_letter, rep_name in base.items():
        if rep_name in available_reps:
            resolved[silo_letter] = rep_name
        else:
            deferred.append((silo_letter, rep_name))
    for silo_letter, rep_name in deferred:
        for fallback in _SILO_FALLBACKS:
            if fallback in available_reps and fallback not in resolved.values():
                resolved[silo_letter] = fallback
                warnings.warn(
                    f"Silo {silo_letter}: '{rep_name}' unavailable, "
                    f"using fallback '{fallback}'")
                break
        else:
            warnings.warn(
                f"Silo {silo_letter}: '{rep_name}' unavailable and "
                f"no unused fallback found — silo dropped")
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeError(
            f"_get_silo_reps({dataset_name}): two silos resolved to the same "
            f"representation — {resolved}. Every silo must hold a different "
            f"one or the experiment is not asking its own question.")
    return resolved


def _run_one(data, config, output_dir, dataset_name, **kwargs):
    """Fed02: Representation diversity as federated silos."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED02: REPRESENTATION DIVERSITY AS FEDERATED SILOS")
    print("=" * 70)

    fed_model = config.get('fed_model', 'rf')
    fed_seed = config.get('fed_seed', 0)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    is_clf = task_type == 'classification'

    # ── The recipe, read once per budget ──
    augs = data.get('augmentations')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # ⛔ THIS USED TO ASK FOR A BUDGET THAT DID NOT EXIST YET, FIXED 2026-09-28.
    # Two things disagreed. This lookup ran once, here, and asked for
    # `budget=BUDGET`. BUDGET is a local of this function and the only place it
    # is assigned is the `for BUDGET in BUDGETS:` loop 330 lines below, added by
    # commit c495942 on 2026-09-23 when the budget stopped being one value for
    # the whole file and became a swept one. So this line read a name that was
    # not bound yet, raised, and the except clause below set the recipe to
    # nothing -- which means C4 and C5 have used the hard defaults
    # (performance_weighted, treeshap, match) and not a searched recipe on every
    # run since that date. The line the file printed said the lookup "returned
    # no search", which is not what happened.
    #
    # What it does now: one lookup per budget, inside the loop, so each budget
    # gets the recipe searched for it. That is what a file whose whole point is
    # the budget sweep should do -- asking once with no budget would give every
    # budget one recipe chosen at a different width.
    #
    # The top-2 rows and the two hybrid deltas have no budget of their own, so
    # they still read one search, fetched with no budget named, which means the
    # best recipe across every budget the search holds.
    def _recipe_for(_budget):
        """(search, meta, blueprint, aug condition, allocation, importance,
        internal evaluator) for one budget. The last four fall back to the hard
        defaults when no recipe is stored, exactly as before, and the caller is
        told which happened."""
        try:
            _s, _m = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=fed_to_kirby_model(fed_model), budget=_budget,
                augmentations=augs,
                metric=metric_name,
                task_type=data['task_type'],
                domain=data.get('domain', 'molecular'),
                dataset_name=dataset_name,
                data=data,
                **smoke_search_kwargs(config),
            )
        except Exception as _e:
            # Print the reason. This branch used to swallow it.
            print(f"  the stored-search lookup for budget={_budget} raised "
                  f"{type(_e).__name__}: {_e} -- the hard defaults are used for "
                  f"this budget and it is recorded as unsearched")
            _s, _m = None, None
        _b = _m['best_config']['blueprint'] if _m else None
        # 01e309a cut this function off at the line above, so it returned None
        # and the unpack at the top of the budget loop raised "cannot unpack
        # non-iterable NoneType object" on the server (herg, 2026-09-29).
        # Restored from the resolution that commit replaced: meta's
        # aug_condition is the authoritative winning condition, then the
        # blueprint, then the hard defaults.
        if _m is not None:
            _aug = _m.get('aug_condition',
                          _b.augmentation_strategy if _b else
                          ('uncapped' if augs else 'none'))
        else:
            _aug = 'uncapped' if augs else 'none'
        _al = _b.allocation_method if _b else 'performance_weighted'
        _im = _b.importance_method if _b else 'treeshap'
        _ie = _b.internal_evaluator if _b else 'match'
        return _s, _m, _b, _aug, _al, _im, _ie

    recipe_source_by_budget = {}
    # No budget named: the best recipe across every budget the search holds.
    _top2_search, _ = _recipe_for(None)[:2]

    # ── Phase 1: Setup ───────────────────────────────────────────────────
    print("\n--- Phase 1: Setup ---")

    # Resolve silo reps
    silo_reps = _get_silo_reps(dataset_name, set(reps.keys()))
    silo_letters = sorted(silo_reps.keys())
    K = len(silo_reps)

    if K < 3:
        raise ValueError(
            f"Need at least 3 silo reps, got {K}. "
            f"Available: {sorted(reps.keys())}")

    print(f"  {dataset_name}: {len(labels)} samples, "
          f"{K} silos assigned:")
    for s in silo_letters:
        rep_name = silo_reps[s]
        dim = reps[rep_name].shape[1]
        print(f"    Silo {s}: {rep_name} ({dim}D)")

    # P0 fix: 5-fold scaffold-group CV (was single 80/20 → all 6 conditions
    # evaluated on one fold). Hard-fail when groups unavailable (per
    # feedback_no_random_splits.md — no random splits anywhere).
    n_samples = len(labels)
    N_FOLDS = smoke_n_folds(5)   # 2 under KIRBY_SMOKE_N, else 5
    if groups is None or len(np.unique(groups)) < N_FOLDS:
        n_groups = 0 if groups is None else len(np.unique(groups))
        raise ValueError(
            f"fed02 requires scaffold groups for {N_FOLDS}-fold CV. "
            f"Got n_groups={n_groups} (need >= {N_FOLDS}). "
            f"Per project rule (no random splits), refusing to fall back "
            f"to KFold.")

    fold_splits = list(cv_split(
        groups, N_FOLDS, np.zeros(n_samples), labels, seed=fed_seed))
    print(f"  Using {N_FOLDS}-fold GroupKFold CV")

    def _make_model():
        # Unfitted template of the swept FLuID model family (per-run fed_model
        # + fed_seed). Used only as create_hybrid's model= for 'match' → model
        # family resolution; all *fitted* models go through fit_fed_model.
        if is_clf:
            return make_fed_classifier(fed_model, seed=fed_seed)
        return make_fed_regressor(fed_model, seed=fed_seed)

    _compute_metrics = compute_clf_metrics if is_clf else compute_reg_metrics

    # Per-fold accumulators
    fold_single_silo = []   # list of {silo: metrics_dict} per fold
    fold_c2 = []
    fold_c3 = []
    fold_c4 = []
    fold_c5 = []
    fold_c6 = []
    fold_best_silo = []     # best silo letter per fold, picked on the TEST fold
    fold_best_silo_train = []   # best silo letter per fold, picked on the TRAIN rows
    fold_inner_scores = []  # {silo: inner train-only score} per fold
    # How far the inner ranking's #1 and #2 silos sit apart, in the metric's
    # own units, and which silo was #2. A margin near zero means the pick
    # between them is close to a coin flip -- MEASURED on caco2 fold 0: the
    # #1 pick flipped between two silos in 6 of 20 redraws of the same
    # training rows, and their scores sat 0.0010 apart against a run-to-run
    # wobble of 0.0006-0.0010 each (2026-09-18 check). Recorded per fold so a
    # reader can tell a real winner from a near-tie without rerunning that
    # check themselves.
    fold_rank0_margin = []
    fold_rank0_runnerup = []
    fold_budget_sweep = []  # {condition: {budget: metrics}} per fold
    fold_top_3_silos = []   # top-3 silo letters per fold
    fold_top_3_reps = []
    fold_n_train = []
    fold_n_test = []

    from kirby.hybrid import (
        create_hybrid, apply_feature_selection, apply_augmentation_selection)

    for fold_i, (train_idx, test_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 70}\nFOLD {fold_i + 1}/{N_FOLDS}\n{'=' * 70}")

        y_train = labels[train_idx]
        y_test = labels[test_idx]
        print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")
        fold_n_train.append(len(train_idx))
        fold_n_test.append(len(test_idx))

        # Pre-split reps (per fold)
        train_reps = {r: arr[train_idx] for r, arr in reps.items()}
        test_reps = {r: arr[test_idx] for r, arr in reps.items()}

        # P1 fix: compute train_augs/test_augs ONCE per fold (was computed
        # three times — phase 4 base + C4 + C5).
        train_augs = ({k: v[train_idx] for k, v in augs.items()}
                      if augs else None)
        test_augs = ({k: v[test_idx] for k, v in augs.items()}
                     if augs else None)

        def _append_test_augs(X_test, feature_info, _test_augs=test_augs):
            if not _test_augs:
                return X_test
            aug_info = feature_info.get('augmentation_info') if isinstance(feature_info, dict) else None
            if not aug_info:
                return X_test
            X_aug = apply_augmentation_selection(_test_augs, aug_info)
            return np.hstack([X_test, X_aug]) if X_aug is not None else X_test

        # P0/P1 fix: 'match' → 'rf' resolution. create_hybrid resolves
        # 'match' via type(model).__name__ — passing an instantiated RF
        # ensures 'rf' (not 'ridge' silent fallback).
        _match_model = _make_model()

        # ── Phase 1b: Rank the five representations inside the training
        # rows only ──────────────────────────────────────────────────
        # This ranking feeds two conditions. C5 takes its three from it. C1
        # takes its one from it. It used to live inside C5's try block, so a
        # C5 failure took C1's honest pick down with it; it runs on its own
        # now. Cost is 5 representations x INNER_RANK_FOLDS fits per fold,
        # paid once whether one condition reads it or both do.
        try:
            inner_scores = _rank_reps_train_only(
                silo_letters, silo_reps, train_reps, y_train,
                groups=(groups[train_idx] if groups is not None else None),
                fed_model=fed_model, fed_seed=fed_seed,
                task_type=task_type, is_clf=is_clf,
                compute_metrics=_compute_metrics,
                n_inner_folds=INNER_RANK_FOLDS)
            silo_ranked = sorted(inner_scores.items(), key=lambda x: x[1],
                                 reverse=bool(is_clf))
        except Exception as e:
            print(f"  Inner rep ranking FAILED: {e}")
            traceback.print_exc()
            inner_scores, silo_ranked = None, None
        fold_inner_scores.append(convert_numpy(inner_scores)
                                 if inner_scores else None)
        if silo_ranked and len(silo_ranked) >= 2:
            fold_rank0_margin.append(float(abs(silo_ranked[1][1] - silo_ranked[0][1])))
            fold_rank0_runnerup.append(silo_ranked[1][0])
        else:
            fold_rank0_margin.append(None)
            fold_rank0_runnerup.append(None)

        # ── Phase 2: Single-silo baselines (C1) ──────────────────────
        print(f"\n--- Phase 2: Single-silo baselines (C1) [task={task_type}] ---")
        single_silo = {}

        for s in silo_letters:
            rep_name = silo_reps[s]
            X_tr = train_reps[rep_name]
            X_te = test_reps[rep_name]

            mdl = fit_fed_model(fed_model, X_tr, y_train, seed=fed_seed,
                                task_type=task_type)
            y_pred = mdl.predict(X_te)

            if is_clf:
                y_prob = mdl.predict_proba(X_te)[:, 1] if hasattr(mdl, 'predict_proba') else None
                metrics = _compute_metrics(y_test, y_pred, y_prob=y_prob)
            else:
                metrics = _compute_metrics(y_test, y_pred)
            single_silo[s] = {
                'rep_name': rep_name,
                'n_features': int(X_tr.shape[1]),
                **metrics,
            }
            if is_clf:
                print(f"  Silo {s} ({rep_name}): "
                      f"MCC={metrics['mcc']:.4f}  "
                      f"BAC={metrics['bac']:.4f}  "
                      f"AUROC={metrics.get('auroc', float('nan')):.4f}")
            else:
                _rho = metrics.get('spearman_rho')
                _rho_s = ('undefined' if _rho is None else f"{_rho:.4f}")
                print(f"  Silo {s} ({rep_name}): "
                      f"RMSE={metrics['rmse']:.4f}  "
                      f"rho={_rho_s}  "
                      f"R2={metrics['r2']:.4f}")

        # Best silo (per fold), picked two ways. Both are reported; they
        # answer different questions and must never be merged into one row.
        #
        #  test-picked — the representation with the best score on THIS fold's
        #    held-out molecules. Nobody could have chosen it in advance, so it
        #    is a ceiling: the best single representation available to someone
        #    who already knew the answer.
        #  train-picked — the representation that won the inner ranking on the
        #    training rows, scored afterwards on the held-out molecules. This
        #    is the honest single-representation baseline, and it is the one
        #    C4's gain has to be measured against.
        if is_clf:
            best_silo_f = max(single_silo, key=lambda s: single_silo[s]['mcc'])
        else:
            best_silo_f = min(single_silo, key=lambda s: single_silo[s]['rmse'])
        best_silo_train_f = silo_ranked[0][0] if silo_ranked else None
        print(f"  Best on the held-out fold (ceiling): Silo {best_silo_f} "
              f"({silo_reps[best_silo_f]})")
        if best_silo_train_f is not None:
            print(f"  Best on the training rows (honest):  Silo "
                  f"{best_silo_train_f} ({silo_reps[best_silo_train_f]})")
        else:
            print("  Best on the training rows (honest):  unavailable "
                  "— the inner ranking failed on this fold")
        fold_single_silo.append(single_silo)
        fold_best_silo.append(best_silo_f)
        fold_best_silo_train.append(best_silo_train_f)

        # ── Phase 3: FLuID ensemble (C2, C3) ─────────────────────────
        print("\n--- Phase 3: FLuID ensemble conditions ---")
        teacher_models = [None]
        silo_train_features_sim = [None]
        silo_test_features_sim = [None]
        silo_train_features_qual = [None]
        silo_test_features_qual = [None]

        for s in silo_letters:
            rep_name = silo_reps[s]
            X_tr = train_reps[rep_name]
            X_te = test_reps[rep_name]

            mdl = fit_fed_model(fed_model, X_tr, y_train, seed=fed_seed,
                                task_type=task_type)

            teacher_models.append((mdl, len(X_tr), X_tr))
            silo_train_features_sim.append(X_tr)
            silo_test_features_sim.append(X_te)
            silo_train_features_qual.append((X_tr, y_train))
            silo_test_features_qual.append(X_te)

        print("  C2: FLuID similarity...", end=' ')
        try:
            if is_clf:
                y_pred_c2, y_prob_c2, tw_c2 = predict_fluid_similarity_ensemble_classification(
                    teacher_models, silo_train_features_sim,
                    silo_test_features_sim, K)
                c2 = _compute_metrics(y_test, y_pred_c2, y_prob=y_prob_c2)
                print(f"MCC={c2['mcc']:.4f}")
            else:
                y_pred_c2, tw_c2 = predict_fluid_similarity_ensemble_regression(
                    teacher_models, silo_train_features_sim,
                    silo_test_features_sim, K)
                c2 = _compute_metrics(y_test, y_pred_c2)
                print(f"RMSE={c2['rmse']:.4f}")
            c2['teacher_weights_mean'] = convert_numpy(
                {silo_letters[i]: float(tw_c2[:, i].mean())
                 for i in range(K)})
        except Exception as e:
            c2 = {'error': str(e)}
            print(f"FAILED: {e}")
        fold_c2.append(c2)

        print("  C3: FLuID quality...", end=' ')
        try:
            if is_clf:
                y_pred_c3, y_prob_c3, mcc_w_c3 = predict_fluid_quality_ensemble_classification(
                    teacher_models, silo_train_features_qual,
                    silo_test_features_qual, K)
                c3 = _compute_metrics(y_test, y_pred_c3, y_prob=y_prob_c3)
                c3['mcc_weights'] = convert_numpy(mcc_w_c3)
                print(f"MCC={c3['mcc']:.4f}")
            else:
                y_pred_c3, rmse_w_c3 = predict_fluid_quality_ensemble_regression(
                    teacher_models, silo_train_features_qual,
                    silo_test_features_qual, K)
                c3 = _compute_metrics(y_test, y_pred_c3)
                c3['rmse_weights'] = convert_numpy(rmse_w_c3)
                print(f"RMSE={c3['rmse']:.4f}")
        except Exception as e:
            c3 = {'error': str(e)}
            print(f"FAILED: {e}")
        fold_c3.append(c3)

        # C4, C5 and C6 all pick BUDGET features and are measured against
        # C2 and C3, which average five models over WHOLE representations
        # of 2,048, 1,613 and 768 columns. Width was the obvious reason
        # the selected arms lose and it had never been varied, so all
        # three sweep it together — comparing them at different widths
        # would mix the method with the size.
        c4_by_budget, c5_by_budget, c6_by_budget = {}, {}, {}
        for BUDGET in BUDGETS:
            # ── Phase 4: KIRBy hybrid (C4, C5) ───────────────────────────
            print("\n--- Phase 4: KIRBy hybrid conditions ---")

            # This budget's own recipe. Before 2026-09-28 every budget used one
            # recipe, and because the lookup asked for an unbound name that
            # recipe was the hard defaults.
            (_bsearch, _bmeta, _bbp, aug_cond, _alloc, _imp,
             _int_eval) = _recipe_for(BUDGET)
            recipe_source_by_budget[BUDGET] = (
                'searched' if _bmeta is not None else 'hard_defaults')
            print(f"  budget={BUDGET}: allocation={_alloc} importance={_imp} "
                  f"internal evaluator={_int_eval} augmentation={aug_cond} "
                  f"({recipe_source_by_budget[BUDGET]})")

            # Build silo-only rep dicts
            kirby_train = {silo_reps[s]: train_reps[silo_reps[s]]
                           for s in silo_letters}
            kirby_test = {silo_reps[s]: test_reps[silo_reps[s]]
                          for s in silo_letters}

            # Condition 4: KIRBy hybrid (all K reps)
            print("  C4: KIRBy hybrid (all)...", end=' ')
            try:
                X_h_train, fi_c4 = create_hybrid(
                    kirby_train, y_train,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=_match_model,  # P0/P1: enables 'match' → 'rf' resolution
                    metric_fn=metric_fn, lower_is_better=lower_is_better,
                    allocation_method=_alloc,
                    importance_method=_imp,
                    budget=BUDGET,
                    internal_evaluator=_int_eval,
                    augmentations=train_augs,
                    augmentation_strategy=aug_cond)
                X_h_test = _append_test_augs(
                    apply_feature_selection(kirby_test, fi_c4), fi_c4)

                mdl_c4 = fit_fed_model(fed_model, X_h_train, y_train, seed=fed_seed,
                                       task_type=task_type)
                y_pred_c4 = mdl_c4.predict(X_h_test)

                if is_clf:
                    y_prob_c4 = mdl_c4.predict_proba(X_h_test)[:, 1] if hasattr(mdl_c4, 'predict_proba') else None
                    c4 = _compute_metrics(y_test, y_pred_c4, y_prob=y_prob_c4)
                    print(f"MCC={c4['mcc']:.4f}  ({X_h_train.shape[1]}D)")
                else:
                    c4 = _compute_metrics(y_test, y_pred_c4)
                    print(f"RMSE={c4['rmse']:.4f}  ({X_h_train.shape[1]}D)")
                c4['n_features'] = int(X_h_train.shape[1])
                c4['allocation'] = convert_numpy(fi_c4.get('allocation', {}))
            except Exception as e:
                # Record traceback (not just str(e)) so a shape/import/NameError in
                # the create_hybrid path surfaces during the expensive run instead
                # of being reduced to a one-line message (FED-21).
                c4 = {'error': str(e), 'traceback': traceback.format_exc()}
                print(f"FAILED: {e}")
                traceback.print_exc()

            # Condition 5: KIRBy top-tier hybrid (top 3 reps by best silo metric)
            print("  C5: KIRBy top-tier (3)...", end=' ')
            try:
                # The five silo representations were ranked INSIDE the training
                # rows in Phase 1b, on an inner group-aware split. Take the three
                # that won there. Ranking them by their score on this fold's test
                # rows — which is what single_silo holds — picked C5's three
                # representations with the answers in view, and C5 was then scored
                # on those same rows.
                if silo_ranked is None:
                    raise RuntimeError(
                        "C5 needs the Phase 1b train-only ranking and it failed "
                        "on this fold.")
                top_3_silos = [s for s, _ in silo_ranked[:3]]
                top_3_names = [silo_reps[s] for s in top_3_silos]

                kirby_top3_train = {silo_reps[s]: train_reps[silo_reps[s]]
                                    for s in top_3_silos}
                kirby_top3_test = {silo_reps[s]: test_reps[silo_reps[s]]
                                   for s in top_3_silos}

                X_t3_train, fi_c5 = create_hybrid(
                    kirby_top3_train, y_train,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=_match_model,  # P0/P1: 'match' → 'rf' resolution
                    metric_fn=metric_fn, lower_is_better=lower_is_better,
                    allocation_method=_alloc,
                    importance_method=_imp,
                    budget=BUDGET,
                    internal_evaluator=_int_eval,
                    augmentations=train_augs,
                    augmentation_strategy=aug_cond)
                X_t3_test = _append_test_augs(
                    apply_feature_selection(kirby_top3_test, fi_c5), fi_c5)

                mdl_c5 = fit_fed_model(fed_model, X_t3_train, y_train, seed=fed_seed,
                                       task_type=task_type)
                y_pred_c5 = mdl_c5.predict(X_t3_test)

                if is_clf:
                    y_prob_c5 = mdl_c5.predict_proba(X_t3_test)[:, 1] if hasattr(mdl_c5, 'predict_proba') else None
                    c5 = _compute_metrics(y_test, y_pred_c5, y_prob=y_prob_c5)
                    print(f"MCC={c5['mcc']:.4f}  ({X_t3_train.shape[1]}D, "
                          f"reps={top_3_names})")
                else:
                    c5 = _compute_metrics(y_test, y_pred_c5)
                    print(f"RMSE={c5['rmse']:.4f}  ({X_t3_train.shape[1]}D, "
                          f"reps={top_3_names})")
                c5['n_features'] = int(X_t3_train.shape[1])
                c5['top_3_silos'] = top_3_silos
                c5['top_3_reps'] = top_3_names
                # The train-only scores the three were chosen on, so the choice can
                # be read back without re-running the fold.
                c5['inner_rank_scores'] = convert_numpy(inner_scores)
                c5['inner_rank_metric'] = 'mcc' if is_clf else 'rmse'
                c5['inner_rank_folds'] = INNER_RANK_FOLDS
                c5['allocation'] = convert_numpy(fi_c5.get('allocation', {}))
                if BUDGET == HEADLINE_BUDGET:
                    fold_top_3_silos.append(top_3_silos)
                    fold_top_3_reps.append(top_3_names)
            except Exception as e:
                c5 = {'error': str(e), 'traceback': traceback.format_exc()}
                print(f"FAILED: {e}")
                traceback.print_exc()
                if BUDGET == HEADLINE_BUDGET:
                    fold_top_3_silos.append(None)
                    fold_top_3_reps.append(None)

            # ── Phase 5: FLuID→KIRBy pipeline (C6) ───────────────────────
            print("\n--- Phase 5: FLuID-within-KIRBy-across pipeline (C6) ---")
            print("  C6: FLuID→KIRBy...", end=' ')
            try:
                train_groups_c6 = (groups[train_idx]
                                   if groups is not None else None)
                X_c6_train, X_c6_test, budgets_c6, scores_c6, meta_c6 = \
                    fluid_within_kirby_across(
                        kirby_train, kirby_test, y_train,
                        k_intra=K_INTRA,
                        allocation_method=_alloc, importance_method=_imp,
                        internal_evaluator=_int_eval,
                        downstream_model=fed_to_kirby_model(fed_model),  # 'match' resolution
                        groups=train_groups_c6,
                        seed=fed_seed,
                        task_type=task_type,
                        metric_fn=metric_fn, lower_is_better=lower_is_better)

                mdl_c6 = fit_fed_model(fed_model, X_c6_train, y_train, seed=fed_seed,
                                       task_type=task_type)
                y_pred_c6 = mdl_c6.predict(X_c6_test)

                if is_clf:
                    y_prob_c6 = mdl_c6.predict_proba(X_c6_test)[:, 1] if hasattr(mdl_c6, 'predict_proba') else None
                    c6 = _compute_metrics(y_test, y_pred_c6, y_prob=y_prob_c6)
                    _c6_label = f"MCC={c6['mcc']:.4f}"
                else:
                    c6 = _compute_metrics(y_test, y_pred_c6)
                    _c6_label = f"RMSE={c6['rmse']:.4f}"
                c6['n_features'] = int(X_c6_train.shape[1])
                c6['silo_budgets'] = convert_numpy(budgets_c6)
                c6['silo_scores'] = convert_numpy(scores_c6)
                c6['n_virtual_silos'] = len(meta_c6)
                c6['virtual_silo_metadata'] = convert_numpy(meta_c6)
                print(f"{_c6_label}  "
                      f"({c6['n_features']}D, {c6['n_virtual_silos']} virtual silos)")
            except Exception as e:
                c6 = {'error': str(e), 'traceback': traceback.format_exc()}
                print(f"FAILED: {e}")
                traceback.print_exc()
            c4_by_budget[str(BUDGET)] = c4
            c5_by_budget[str(BUDGET)] = c5
            c6_by_budget[str(BUDGET)] = c6
        # the width every summary row and cross-run comparison reads
        c4 = c4_by_budget[str(HEADLINE_BUDGET)]
        c5 = c5_by_budget[str(HEADLINE_BUDGET)]
        c6 = c6_by_budget[str(HEADLINE_BUDGET)]
        fold_c4.append(c4); fold_c5.append(c5); fold_c6.append(c6)
        # NOT c4['by_budget'] = c4_by_budget: c4 is an entry in that dict, so
        # the result would contain itself and json.dump would recurse for
        # ever. The sweep is kept beside the conditions, not inside them.
        fold_budget_sweep.append({
            'c4_kirby_hybrid': c4_by_budget,
            'c5_kirby_top_tier': c5_by_budget,
            'c6_fluid_within_kirby_across': c6_by_budget,
            # {budget: 'searched' or 'hard_defaults'}. Before 2026-09-28 every
            # budget on every fold would have read 'hard_defaults', because the
            # one recipe lookup asked for a budget that was not assigned yet and
            # raised. Recorded per fold so a run that finds no stored recipe for
            # some budgets cannot be read as one that found them all.
            'recipe_source_by_budget': dict(recipe_source_by_budget),
        })

    # ── Phase 6: Aggregate per-fold metrics → mean ± std ─────────────────
    print(f"\n{'=' * 70}\nAGGREGATING ACROSS {N_FOLDS} FOLDS\n{'=' * 70}")

    _scalar_keys_reg = ['rmse', 'spearman_rho', 'spearman_p',
                        'pearson_r', 'pearson_p', 'r2']
    _scalar_keys_clf = ['mcc', 'bac', 'f1', 'auroc', 'accuracy', 'f1_weighted']
    _scalar_keys = _scalar_keys_clf if is_clf else _scalar_keys_reg

    def _aggregate_folds(fold_list):
        """Mean±std of scalar metrics across folds; preserve last-fold extras."""
        out = {}
        # Collect scalars
        for k in _scalar_keys:
            vals = [f.get(k) for f in fold_list
                    if isinstance(f, dict) and isinstance(f.get(k), (int, float))
                    and not (isinstance(f.get(k), float) and np.isnan(f.get(k)))]
            if vals:
                out[k] = float(np.mean(vals))
                out[f'{k}_std'] = float(np.std(vals))
                out[f'{k}_per_fold'] = [float(v) for v in vals]
            else:
                out[k] = float('nan')
                out[f'{k}_std'] = float('nan')
                out[f'{k}_per_fold'] = []
        # Preserve non-scalar extras from the last successful fold (e.g.
        # allocation, n_features, teacher_weights_mean, silo_budgets).
        for f in reversed(fold_list):
            if isinstance(f, dict) and 'error' not in f:
                for k, v in f.items():
                    if k in _scalar_keys:
                        continue
                    if k.endswith('_std') or k.endswith('_per_fold'):
                        continue
                    out.setdefault(k, v)
                break
        # Record any per-fold errors
        errors = [f.get('error') for f in fold_list
                  if isinstance(f, dict) and 'error' in f]
        if errors:
            out['errors_per_fold'] = errors
        return out

    c2 = _aggregate_folds(fold_c2)
    c3 = _aggregate_folds(fold_c3)
    c4 = _aggregate_folds(fold_c4)
    c5 = _aggregate_folds(fold_c5)
    c6 = _aggregate_folds(fold_c6)

    # Aggregate single-silo per-letter, then pick overall best by mean primary
    _primary_key = 'mcc' if is_clf else 'rmse'
    silo_agg = {}
    for s in silo_letters:
        vals_by_metric = {k: [] for k in _scalar_keys}
        rep_name = silo_reps[s]
        n_feat = None
        for fold_ss in fold_single_silo:
            entry = fold_ss.get(s)
            if entry is None:
                continue
            for k in _scalar_keys:
                v = entry.get(k)
                if isinstance(v, (int, float)) and not (
                        isinstance(v, float) and np.isnan(v)):
                    vals_by_metric[k].append(float(v))
            if n_feat is None:
                n_feat = entry.get('n_features')
        agg = {'rep_name': rep_name, 'n_features': n_feat}
        for k, vals in vals_by_metric.items():
            if vals:
                agg[k] = float(np.mean(vals))
                agg[f'{k}_std'] = float(np.std(vals))
            else:
                agg[k] = float('nan')
                agg[f'{k}_std'] = float('nan')
        silo_agg[s] = agg

    if is_clf:
        best_silo = max(silo_agg, key=lambda s: silo_agg[s].get('mcc', -np.inf))
    else:
        best_silo = min(silo_agg, key=lambda s: silo_agg[s].get('rmse', np.inf))
    print(f"  Best silo on the held-out folds (ceiling): "
          f"{best_silo} ({silo_reps[best_silo]})")

    # The honest single-representation baseline. Each fold contributes the
    # held-out score of whichever representation won that fold's inner
    # ranking on the training rows — so the pick can change fold to fold, and
    # picked_per_fold records what each one chose. Averaging the chosen
    # representation's own fold scores is the whole arithmetic; no
    # representation is ever scored on rows that chose it.
    _trainpick_vals = {}
    for _f, _s in enumerate(fold_best_silo_train):
        if _s is None or _s not in fold_single_silo[_f]:
            continue
        for k, v in fold_single_silo[_f][_s].items():
            if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and np.isnan(v)):
                _trainpick_vals.setdefault(k, []).append(float(v))
    c1_trainpick = {
        'picked_per_fold': list(fold_best_silo_train),
        'picked_reps_per_fold': [silo_reps[s] if s in silo_reps else None
                                 for s in fold_best_silo_train],
        'n_folds_scored': len(_trainpick_vals.get(_primary_key, [])),
        # How close the #1 pick above was to the runner-up, per fold, in
        # inner_rank_metric's own units. A small margin next to a large
        # rank0_std/std elsewhere in this dict means the pick could easily
        # have gone the other way -- see the note on fold_rank0_margin above.
        'rank0_margin_per_fold': list(fold_rank0_margin),
        'rank0_runnerup_per_fold': list(fold_rank0_runnerup),
        'rank0_runnerup_reps_per_fold': [
            silo_reps[s] if s in silo_reps else None
            for s in fold_rank0_runnerup],
    }
    for k, vals in _trainpick_vals.items():
        c1_trainpick[k] = float(np.mean(vals))
        c1_trainpick[f'{k}_std'] = float(np.std(vals))
    if c1_trainpick['n_folds_scored'] == 0:
        print("  Best silo on the training rows (honest): none — the inner "
              "ranking failed on every fold")
    else:
        _picked = [s for s in fold_best_silo_train if s is not None]
        print(f"  Best silo on the training rows (honest): "
              f"{sorted(set(_picked))} across {len(_picked)} folds, "
              f"{_primary_key}={c1_trainpick[_primary_key]:.4f}")

    # ── Phase 6: Assemble results ─────────────────────────────────────────
    results = {
        'experiment': 'fed02_rep_diversity_silos',
        'dataset': dataset_name,
        'task_type': task_type,
        'config': {
            'silo_reps': {s: silo_reps[s] for s in silo_letters},
            'n_silos': K,
            'n_samples': len(labels),
            'n_folds': N_FOLDS,
            'n_train_per_fold': fold_n_train,
            'n_test_per_fold': fold_n_test,
            'k_intra_c6': K_INTRA,
            # BUDGET is the loop variable now, so its value after the folds is
            # whatever the sweep ended on. Report the list and the width every
            # headline row is taken at.
            'budgets_swept': BUDGETS,
            'budget': HEADLINE_BUDGET,
            'best_silo_per_fold': fold_best_silo,
            'best_silo_train_picked_per_fold': fold_best_silo_train,
            'inner_rank_scores_per_fold': fold_inner_scores,
            'inner_rank_metric': _primary_key,
            'inner_rank_folds': INNER_RANK_FOLDS,
            'top_3_silos_per_fold': fold_top_3_silos,
        },
        # C1 is reported two ways and there is no single 'C1 number'. Read
        # 'train_picked' against C4; read 'test_picked_ceiling' only as a
        # bound, and never put it in a table beside the other conditions
        # without saying it is one.
        'c1_best_single_silo': convert_numpy({
            'train_picked': c1_trainpick,
            'test_picked_ceiling': {
                'best_silo': best_silo,
                'best_rep': silo_reps[best_silo],
                f'best_{_primary_key}': silo_agg[best_silo][_primary_key],
                f'best_{_primary_key}_std': silo_agg[best_silo].get(
                    f'{_primary_key}_std', float('nan')),
                'picked_per_fold': fold_best_silo,
            },
            'per_silo': silo_agg,
        }),
        'c2_fluid_similarity': convert_numpy(c2),
        'c3_fluid_quality': convert_numpy(c3),
        'c4_kirby_hybrid': convert_numpy(c4),
        'c5_kirby_top_tier': convert_numpy(c5),
        'c6_fluid_within_kirby_across': convert_numpy(c6),
        # Every width, per fold, for the three arms that select features.
        # The aggregated conditions above are the headline width only.
        'budget_sweep_per_fold': convert_numpy(fold_budget_sweep),
        'privacy_notes': {
            'c1': 'each silo trains independently, no sharing',
            'c2': 'silos share predictions only',
            'c3': f'silos share predictions + one {"MCC" if is_clf else "RMSE"} scalar',
            'c4': 'centralized: all reps must be co-located',
            'c5': 'centralized: all reps must be co-located',
            'c6': 'intra-silo clustering local, cross-silo via federated FS',
        },
    }

    # ── Top-2 rows and the two hybrid deltas ──────────────────────
    # Owner's instruction, 2026-09-18: add the top 2 rows as an extra and
    # change nothing already reported. Every key written here is new; c1
    # through c6 above are untouched, and the six conditions still run on all
    # 5 silo representations.
    #
    # All four helpers read rows the precomputed search already stored. They
    # fit no model, so this block costs 0 new model fits on top of the
    # 5 representations x 3 inner folds x 5 outer folds ranking fits and the
    # six conditions.
    #
    # What was missing: this file read ONE configuration out of the search --
    # the blueprint at meta['best_config'], which sets _alloc, _imp and
    # _int_eval for C4 and C5 -- and wrote down neither the runner-up nor the
    # score of either. A reader could not tell a clear winner from two
    # configurations sitting a thousandth of an RMSE apart.
    if _top2_search is not None:
        # ⛔ ONE try PER CALL, changed 2026-09-28. Everything below used to sit
        # in a single try block. The two halves of the comparison are the top 2
        # base representations and the top 2 hybrid configurations, so a raise
        # in the base half took the hybrid half, the per-model grid and every
        # delta with it, and the one printed line named a single reason for four
        # missing keys.
        from experiments._shared import (
            top_base_reps, top_hybrids, select_model_rep_pairs,
            hybrid_deltas, resolve_comparison_models)
        _kirby_model = fed_to_kirby_model(fed_model)
        # require_full=False everywhere: a downstream model the search
        # never scored comes back as a short list carrying its own reason
        # instead of raising, so it cannot end a run that has already paid
        # for 5 folds of all six conditions.
        try:
            results['top2_base_reps'] = convert_numpy(top_base_reps(
                _top2_search, _kirby_model, n=2, require_full=False,
                context=f'fed02 {dataset_name} top-2 base representations'))
        except Exception as e:
            print(f"  top2_base_reps not added: {type(e).__name__}: {e}")
        try:
            # ⛔ NO budget= FILTER, changed 2026-09-28. It used to be
            # budget=BUDGET while top_base_reps above took no budget at all, so
            # the two halves of one comparison were filtered differently: the
            # hybrid half was cut to one number of features and the base half
            # ranked every representation at its own native width. Which way
            # they were made consistent, and why this way:
            #
            # A base representation has no budget to be filtered by -- its width
            # is whatever the builder produced. Cutting the base half to BUDGET
            # features would change which representations this file reports. So
            # the filter comes off the hybrid half, and both halves now rank
            # over every row the search stored. Every returned hybrid entry
            # carries its own 'budget' -- the number of features it was searched
            # at -- so the widths stay readable. Same as
            # s08_optimal_dimension.py:566-576 and
            # s10_importance_downstream.py:2353-2361.
            results['top2_hybrids'] = convert_numpy(top_hybrids(
                _top2_search, _kirby_model, n=2,
                require_full=False,
                context=f'fed02 {dataset_name} top-2 hybrids'))
        except Exception as e:
            print(f"  top2_hybrids not added: {type(e).__name__}: {e}")
        try:
            # The same two slots for each downstream model the search scored,
            # not only the one this run's FLuID model translates to.
            # select_model_rep_pairs calls resolve_comparison_models itself, so
            # models= is left off and it loops what the search scored.
            #
            # No budget= here either: select_model_rep_pairs passes budget only
            # to its top_hybrids half and never to its top_base_reps half
            # (_shared.py:8021-8027), so budget= filtered one half of every
            # model's row and not the other.
            results['top2_pairs_by_model'] = convert_numpy(
                select_model_rep_pairs(
                    _top2_search, n_base=2, n_hybrids=2,
                    require_full=False,
                    context=f'fed02 {dataset_name} top-2 grid'))
        except Exception as e:
            print(f"  top2_pairs_by_model not added: {type(e).__name__}: {e}")
        try:
            # Two reference numbers for the best hybrid, per downstream model:
            # against the mean of the representations the budget actually gave
            # features to, and against the best single representation. Keys
            # are downstream model names, strings; the only thing that reads
            # this dict is a JSON dump and aggregate_seed_runs, which walks it
            # by key and never by position or by int.
            #
            # The names to loop come off the search. This used to be
            # DOWNSTREAM_MODELS, a fixed list of five names in _shared.py, while
            # the search scores whatever list it was given -- which can be
            # shorter, since a dataset's skip policy drops svm and svm drops
            # itself above 10,000 rows. requested=None asks for every model the
            # search scored, and nothing is substituted for one it did not.
            _deltas = {}
            _delta_models, _delta_dropped = resolve_comparison_models(
                _top2_search, None)
            for _name, _why in _delta_dropped:
                _deltas[_name] = {'error': _why}
            for _m in _delta_models:
                try:
                    # The hybrid is passed by hand. Called without it,
                    # hybrid_deltas reads _picked['selected'][0]['row']
                    # (_shared.py:7432), and top_hybrids builds its entries
                    # without a 'row' key (_shared.py:7042-7066) -- score_mean
                    # and blueprint, the only two the deltas read, sit at the
                    # top level of the entry. So the defaulted call raises
                    # KeyError('row') for every downstream model. Read in this
                    # session, 2026-09-18; opig_humanness.py:363 and
                    # opig_graphinity_ddg.py:545 pass the hybrid for the same
                    # reason.
                    # No budget= filter, for the same reason top2_hybrids above
                    # dropped it. It also keeps the rank-1 hybrid the deltas are
                    # computed from and the rank-1 row in top2_hybrids the same
                    # configuration; with the filter on one and not the other
                    # they could be two different recipes.
                    _top1 = top_hybrids(
                        _top2_search, _m, 1,
                        require_full=False,
                        context=f'fed02 {dataset_name} deltas/{_m}')
                    if not _top1['selected']:
                        _deltas[_m] = {
                            'error': 'no hybrid configuration stored for this '
                                     'downstream model: '
                                     f"{_top1['shortfall_reason']}"}
                        continue
                    _deltas[_m] = hybrid_deltas(
                        _top2_search, _m, hybrid=_top1['selected'][0],
                        context=f'fed02 {dataset_name} hybrid deltas')
                except Exception as _e:
                    # One downstream model with no stored hybrid must not cost
                    # the others their deltas.
                    _deltas[_m] = {'error': f'{type(_e).__name__}: {_e}'}
            results['hybrid_deltas_by_model'] = convert_numpy(_deltas)
        except Exception as e:
            print(f"  hybrid deltas FAILED: {e}")
            traceback.print_exc()
    else:
        print("  top-2 rows / hybrid deltas skipped: "
              "get_best_search_for_downstream returned no search, so there "
              "are no stored rows to rank")

    # Summary table
    print(f"\n{'─' * 70}")
    def _summary_row(key):
        """The metrics dict one summary line prints.

        C1 has two rows and neither is stored at the top level, so they are
        resolved here; every other condition is its own top-level entry.
        """
        c1 = results['c1_best_single_silo']
        if key == 'c1_train_picked':
            return c1.get('train_picked', {})
        if key == 'c1_test_picked_ceiling':
            return c1.get('per_silo', {}).get(best_silo, {})
        return results[key]

    summary_items = [
        ('c1_train_picked',
         'C1: Best silo, picked on train'),
        ('c1_test_picked_ceiling',
         f'C1 CEILING: best on test ({silo_reps[best_silo]})'),
        ('c2_fluid_similarity', 'C2: FLuID similarity'),
        ('c3_fluid_quality', 'C3: FLuID quality'),
        ('c4_kirby_hybrid', 'C4: KIRBy hybrid (all)'),
        ('c5_kirby_top_tier', 'C5: KIRBy top-tier (3)'),
        ('c6_fluid_within_kirby_across', 'C6: FLuID→KIRBy'),
    ]

    if is_clf:
        print(f"  {'Condition':<35} {'MCC':>8} {'BAC':>8} {'AUROC':>8}")
        print(f"  {'─' * 61}")
        for key, name in summary_items:
            r = _summary_row(key)
            mcc = r.get('mcc', float('nan'))
            bac = r.get('bac', float('nan'))
            auroc = r.get('auroc', float('nan'))
            if np.isnan(mcc):
                print(f"  {name:<35} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            else:
                print(f"  {name:<35} {mcc:>8.4f} {bac:>8.4f} {auroc:>8.4f}")
    else:
        print(f"  {'Condition':<35} {'RMSE':>8} {'Spearman':>10} {'R2':>8}")
        print(f"  {'─' * 61}")
        for key, name in summary_items:
            r = _summary_row(key)
            rmse = r.get('rmse', float('nan'))
            rho = r.get('spearman_rho', float('nan'))
            r2 = r.get('r2', float('nan'))
            if np.isnan(rmse):
                print(f"  {name:<35} {'N/A':>8} {'N/A':>10} {'N/A':>8}")
            else:
                print(f"  {name:<35} {rmse:>8.4f} {rho:>10.4f} {r2:>8.4f}")

    return results


# ── One (FLuID model, seed) call saved as it finishes (2026-09-25) ──────────
# federated_helpers.run_all calls run() once per (fed_model, fed_seed) -- nine
# calls -- and wrote fed02_rep_diversity_silos/<dataset>.json only after the ninth. A kill or
# an exception in any call lost every call before it. Each call's return value
# is now written to fed02_rep_diversity_silos/<dataset>.partial.jsonl the moment it returns
# and read back on the next attempt (_checkpoint.resumable_run), keyed on the
# model, the seed and any keyword argument, under an identity that includes the
# input fingerprint, the plain config and the KIRBY_* settings. The returned
# dict is decoded to the same objects, so aggregate_seed_runs and the final
# JSON are unchanged. --force-rerun / KIRBY_FORCE_RERUN=1 discards the file.
def run(data, config, output_dir, dataset_name, **kwargs):
    from experiments._checkpoint import resumable_run
    _key = {'fed_model': str(config.get('fed_model', 'rf')),
            'fed_seed': int(config.get('fed_seed', 0)),
            **{k: v for k, v in kwargs.items()}}
    return resumable_run(
        'fed02_rep_diversity_silos',
        lambda: _run_one(data, config, output_dir, dataset_name, **kwargs),
        data, config, output_dir, dataset_name, key=_key)



if __name__ == '__main__':
    parser = create_experiment_parser('Fed02: Rep Diversity Silos')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, 'fed02_rep_diversity_silos', args.dataset)
