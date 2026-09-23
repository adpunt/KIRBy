#!/usr/bin/env python3
"""C24: Blueprint Transfer Test — Cross-Fold Generalization of Feature Selection.

Paper: Complexity / Discussion (feature selection overfitting defense)
Conclusion: Blueprints discovered on one fold generalize to other folds,
    demonstrating that feature selection is not overfit to training data.

Dependencies: None (standalone). Uses find_best_hybrid or precomputed search.

Design:
    For each CV fold k:
        1. Discover the optimal blueprint on fold k's training data.
        2. Apply fold k's blueprint to ALL other folds' test data.
        3. Measure performance vs each fold's own blueprint.

    If feature selection is NOT overfit:
        - Own-fold and cross-fold blueprints should perform similarly.
        - Small delta (own - cross) = generalizable selection.

    If feature selection IS overfit:
        - Cross-fold blueprints will degrade significantly.
        - Large delta = selection is training-data-specific.

    Also reports Nogueira stability index across fold blueprints.

Usage:
    python c24_blueprint_transfer.py --dataset logd --mode full
    python c24_blueprint_transfer.py --dataset caco2 --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    build_fold_hybrid,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, _cv_split,
    get_domain_metric,
    get_precomputed_search, get_best_for_downstream,
    INTERNAL_EVALUATORS, FULL_BUDGETS,
)
from kirby.find_best_hybrid import find_best_hybrid, _predict_for_metric


def nogueira_stability(feature_sets, total_features):
    """Compute the Nogueira et al. (2018) stability index.

    Args:
        feature_sets: list of sets, each containing (rep_name, feature_idx) tuples
            selected on each fold/resample.
        total_features: int, total number of features available.

    Returns:
        float in [-1, 1]. 1 = identical selection every time.
        Also returns standard error for confidence interval.

    Reference:
        Nogueira, Sechidis & Brown (2018). "On the Stability of Feature
        Selection Algorithms." JMLR 18(174), 1-54.
    """
    M = len(feature_sets)
    if M < 2 or total_features == 0:
        return float('nan'), float('nan')

    # Convert sets to binary indicator vectors
    # Build a unified feature index
    all_features = set()
    for fs in feature_sets:
        all_features.update(fs)
    all_features = sorted(all_features)
    feat_to_idx = {f: i for i, f in enumerate(all_features)}
    d = len(all_features)

    if d == 0:
        return float('nan'), float('nan')

    # Binary matrix Z: M x d
    Z = np.zeros((M, d), dtype=float)
    for i, fs in enumerate(feature_sets):
        for f in fs:
            if f in feat_to_idx:
                Z[i, feat_to_idx[f]] = 1.0

    # Average number of selected features
    k_bar = Z.sum(axis=1).mean()

    if k_bar == 0 or k_bar == d:
        return float('nan'), float('nan')

    # Per-feature selection frequency
    p_f = Z.mean(axis=0)  # (d,)

    # Nogueira stability (Eq. 4 from the paper)
    # phi = 1 - (1 / (k_bar * (1 - k_bar/d))) * (1/d) * sum_j( M/(M-1) * pj * (1-pj) )
    numerator = (M / (M - 1)) * np.sum(p_f * (1 - p_f))
    denominator = d * k_bar * (1 - k_bar / d)

    if denominator == 0:
        return float('nan'), float('nan')

    phi = 1 - numerator / denominator

    # Standard error (Eq. 7 from the paper, simplified)
    # For confidence intervals: phi ± z * se
    var_terms = p_f * (1 - p_f)
    se = np.sqrt((4 / (M * d**2 * (k_bar * (1 - k_bar / d))**2)) *
                 np.sum(var_terms**2)) if M > 2 else float('nan')

    return float(phi), float(se)


def run(data, config, output_dir, dataset_name):
    """C24: Blueprint transfer test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 24: BLUEPRINT TRANSFER TEST")
    print("=" * 70)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    cmp = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    # Use None (not {}) so downstream `if augs:` checks correctly skip the
    # aug-handling path when no augmentations exist for this dataset.
    augs = data.get('augmentations') or None
    n_folds = config['n_splits']

    sample_X = next(iter(reps.values()))
    total_features = sum(v.shape[1] for v in reps.values())

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    task_type = data['task_type']

    # cv_split RAISES (no random-split fallback, feedback_no_random_splits) when
    # the dataset has fewer distinct groups than the requested fold count. Small
    # antibody-regression datasets (e.g. abagym_Ang2, intra-family single-DMS,
    # and other few-group sets) trip this and propagate an uncaught ValueError
    # out of run() -> the driver serializes it as {'error': ...}, capping the
    # production wave at 19/20. Reduce the outer fold count to the number of
    # distinct groups (min 2), mirroring _make_cv_splitter's reduce-folds rule
    # and c05's fulldata-bp guard. If <2 distinct groups exist, a leak-free
    # grouped CV is impossible -> return a clean skip (no 'error' key) rather
    # than crash.
    n_distinct_groups = 0 if groups is None else len(np.unique(groups))
    if n_distinct_groups < 2:
        print(f"  Skipping: blueprint transfer needs >=2 distinct groups for a "
              f"leak-free grouped CV, but got {n_distinct_groups}.")
        return {'skipped': True, 'reason': 'insufficient_groups',
                'n_distinct_groups': n_distinct_groups}
    if n_distinct_groups < n_folds:
        print(f"  Only {n_distinct_groups} distinct groups for {n_folds}-fold CV; "
              f"reducing to {n_distinct_groups} folds (group-aware, no random fallback).")
        n_folds = n_distinct_groups

    # --- Step 1: Collect per-fold blueprints and own-fold scores ---
    print(f"\n  Step 1: Discover blueprints on each of {n_folds} folds")

    folds = list(_cv_split(groups, n_folds, sample_X, labels))
    fold_blueprints = {}  # fold_idx -> blueprint
    fold_best_models = {}  # fold_idx -> best downstream model name for that fold
    fold_feature_sets = {}  # fold_idx -> set of (rep, idx) tuples
    fold_feature_infos = {}  # fold_idx -> feature_info from that fold's build
    own_fold_scores = {}  # fold_idx -> score on own test data
    fold_errors = []  # (fold_idx, stage, error) — surfaced in the result so
    # an 'insufficient_folds' failure is self-diagnosing without the slurm log.

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_augs = {k: v[train_idx] for k, v in augs.items()} if augs else None
        val_augs = {k: v[val_idx] for k, v in augs.items()} if augs else None

        # Discover blueprint via find_best_hybrid
        try:
            search = find_best_hybrid(
                train_reps, y_train, groups=groups[train_idx],
                budgets=FULL_BUDGETS,
                augmentations=train_augs,
                augmentation_strategy='uncapped' if train_augs else 'none',
                internal_evaluators=INTERNAL_EVALUATORS,
                metric=metric_name,
                task_type=task_type,
                verbose=False,
                cache=True,
                # The real dataset name, so the per-dataset budget cap and
                # the per-dataset model skips match. It used to be
                # f"{dataset_name}_fold{fold_i}", and both are looked up BY
                # NAME, so c24 searched budgets its dataset is capped away from
                # and searched models the policy removes. The fold namespacing
                # moves to condition_suffix, which rides on `tier`. Each fold
                # trains on a different subset of molecules anyway, so their
                # rows already carried different label hashes and could not
                # have collided.
                dataset_name=dataset_name,
                condition_suffix=f"c24_fold{fold_i}",
                tier='general',
            )
            # Pick best across models
            best = None
            best_model = None
            for m_name, m_best in search.best_by_model.items():
                if best is None or cmp(m_best['score_mean'], best['score_mean']):
                    best = m_best
                    best_model = m_name
        except Exception as e:
            print(f"    Fold {fold_i}: search failed ({e}), skipping")
            fold_errors.append({'fold': fold_i, 'stage': 'search',
                                'exception': f'{type(e).__name__}: {e}'})
            continue

        if best is None:
            print(f"    Fold {fold_i}: no valid config found, skipping")
            continue

        bp = best['blueprint']
        fold_blueprints[fold_i] = bp
        fold_best_models[fold_i] = best_model
        print(f"    Fold {fold_i}: {best_model}, budget={bp.budget}, "
              f"alloc={bp.allocation_method}, imp={bp.importance_method}")

        # Create hybrid with this blueprint and evaluate on own test data.
        # build_fold_hybrid, not a bare create_hybrid: it forwards the fold's
        # groups (so the internal admission holdout is not a random 20% carve
        # of scaffold-sharing rows), the internal evaluator, the augmentation
        # strategy and budgets, the candidate ranking, and the estimator with
        # the dataset's own metric — a bare call scored accept/reject with
        # r2_score and silently took create_hybrid's defaults for the rest.
        try:
            model_factories = get_model_factories(
                [best_model], task_type=task_type,
                n_samples=len(y_train))
            X_train, X_val, fi = build_fold_hybrid(
                train_reps, val_reps, y_train, bp,
                augmentations_train=train_augs,
                augmentations_val=val_augs,
                groups=groups[train_idx],
                model=model_factories[best_model](),
                metric_fn=metric_fn,
                lower_is_better=lower_is_better)
            fold_feature_infos[fold_i] = fi

            # Collect selected features for stability analysis
            selected = set()
            for rep_name, rep_info in fi.items():
                if isinstance(rep_info, dict) and 'selected_indices' in rep_info:
                    for idx in rep_info['selected_indices']:
                        selected.add((rep_name, int(idx)))
            fold_feature_sets[fold_i] = selected

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_va = scaler.transform(X_val)

            model = model_factories[best_model]()
            model.fit(X_tr, y_train)
            own_fold_scores[fold_i] = float(metric_fn(
                y_val, _predict_for_metric(model, X_va, metric_fn)))
            print(f"      Own-fold {metric_name}: {own_fold_scores[fold_i]:.4f}")
        except Exception as e:
            print(f"      Own-fold eval failed: {e}")
            fold_errors.append({'fold': fold_i, 'stage': 'own_fold_eval',
                                'exception': f'{type(e).__name__}: {e}'})
            continue

    if len(fold_blueprints) < 2:
        # Fewer than 2 folds produced a scored blueprint (e.g. every outer
        # training fold collapsed to <2 groups inside find_best_hybrid, or no
        # config scored on this dataset). The transfer test genuinely cannot
        # run -> return a clean skip (no top-level 'error' key, which the
        # audit/wave counts as a hard failure). Diagnostics are kept under
        # 'fold_diagnostics' (inner key 'exception', not 'error') so the cause
        # is self-evident without the slurm log.
        print("  Skipping: need at least 2 successful folds for transfer test")
        return {'skipped': True, 'reason': 'insufficient_folds',
                'n_folds_succeeded': len(fold_blueprints),
                'fold_diagnostics': fold_errors}

    # --- Step 2: Cross-fold transfer ---
    print(f"\n  Step 2: Apply each source selection to every other fold's test rows")

    transfer_matrix = {}  # (source_fold, target_fold) -> score

    # Seed the diagonal with Step 1's own-fold scores so the diagonal is
    # consistent with the explicit own-fold eval (avoids stochastic divergence
    # from re-running the build + fit a second time on the same data).
    for fold_i, sc in own_fold_scores.items():
        transfer_matrix[(fold_i, fold_i)] = sc

    # Which rows each cell's recipe was searched on, so the size difference
    # between the diagonal and the off-diagonal is on the record.
    cell_recipe_provenance = {}

    for src_fold in list(fold_blueprints):
        for tgt_fold, (train_idx, val_idx) in enumerate(folds):
            if tgt_fold not in fold_blueprints:
                continue  # skip folds that failed
            if src_fold == tgt_fold:
                continue  # diagonal already populated from own-fold eval

            # Two changes here, together.
            #
            # First, WHICH ROWS. The source fold's recipe and features from
            # Step 1 were chosen on every fold except the source — which
            # INCLUDES this target fold's validation rows. Scored on them, the
            # cell measured a selection made with the answer in hand. The
            # recipe AND the features are chosen here on rows that leave out
            # both folds, so nothing that produced this cell has seen the rows
            # it is scored on. One confound is left and recorded per cell
            # below: the off-diagonal sees one fold less data than the
            # diagonal.
            #
            # Second, WHAT CROSSES. The features are carried over, not
            # re-selected on the target fold. Rebuilding on the target's own
            # rows let only the recipe cross, so two folds landing on the same
            # recipe gave a delta of exactly 0 and read as stable.
            _pair_mask = np.ones(len(labels), dtype=bool)
            _pair_mask[folds[src_fold][1]] = False
            _pair_mask[val_idx] = False
            _pair_idx = np.where(_pair_mask)[0]
            y_val_tgt = labels[val_idx]
            val_reps_tgt = {k: v[val_idx] for k, v in reps.items()}
            val_augs_tgt = ({k: v[val_idx] for k, v in augs.items()}
                            if augs else None)
            src_bp = None
            try:
                if len(np.unique(groups[_pair_idx])) < 2:
                    raise ValueError(
                        f"only {len(np.unique(groups[_pair_idx]))} distinct "
                        f"group(s) left after dropping folds "
                        f"{src_fold} and {tgt_fold}")
                _pair_reps = {k: v[_pair_idx] for k, v in reps.items()}
                _pair_augs = ({k: v[_pair_idx] for k, v in augs.items()}
                              if augs else None)
                _pair_search = find_best_hybrid(
                    _pair_reps, labels[_pair_idx], groups=groups[_pair_idx],
                    budgets=FULL_BUDGETS,
                    augmentations=_pair_augs,
                    augmentation_strategy='uncapped' if _pair_augs else 'none',
                    internal_evaluators=INTERNAL_EVALUATORS,
                    metric=metric_name,
                    task_type=task_type,
                    verbose=False,
                    cache=True,
                    dataset_name=dataset_name,
                    # SORTED, so the two ordered pairs that drop the SAME two
                    # folds share one cache key. _pair_idx is the rows left
                    # after dropping src_fold AND tgt_fold, which is identical
                    # for (A,B) and (B,A): same rows, same labels, same groups,
                    # same budgets, one search. condition_suffix rides on `tier`
                    # and so enters the key, so c24_drop1_3 and c24_drop3_1 were
                    # two keys for one search and each was computed cold. Ten of
                    # the twenty pair searches were duplicates -- flagged in the
                    # 2026-09-18 compute audit and never acted on. Sorting takes
                    # c24 from 25 searches per dataset to 15 with no axis cut
                    # and no reported number changed.
                    condition_suffix=(
                        "c24_drop%d_%d" % tuple(sorted((src_fold, tgt_fold)))),
                    tier='general',
                )
                _pair_best = None
                for _m, _mb in _pair_search.best_by_model.items():
                    if _pair_best is None or cmp(_mb['score_mean'],
                                                 _pair_best['score_mean']):
                        _pair_best = _mb
                if _pair_best is not None:
                    src_bp = _pair_best['blueprint']
            except Exception as exc:
                print(f"    recipe search excluding folds {src_fold},{tgt_fold} "
                      f"failed: {type(exc).__name__}: {exc}")
            if src_bp is None:
                fold_errors.append({
                    'fold': tgt_fold, 'stage': 'pair_recipe_search',
                    'source_fold': src_fold,
                    'exception': 'no blueprint from the pair-excluded search'})
                continue
            cell_recipe_provenance[f'{src_fold}->{tgt_fold}'] = {
                'rows_selection_saw': int(len(_pair_idx)),
                'folds_excluded_from_selection': [int(src_fold), int(tgt_fold)],
                'features_transferred': True,
            }

            try:
                # Use the TARGET fold's best model so the feature-selection
                # contrast (own-selection vs cross-selection) is not confounded
                # by model-type variation across folds.
                best_model_tgt = fold_best_models.get(tgt_fold)
                if best_model_tgt is not None:
                    factories = get_model_factories(
                        [best_model_tgt], task_type=task_type,
                        n_samples=len(_pair_idx))
                    model = factories[best_model_tgt]()
                elif task_type == 'classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

                # Features chosen on the pair-excluded rows, then APPLIED to
                # the target fold's held-out rows — build_fold_hybrid does both
                # halves, so the same columns come off both sides.
                y_pair = labels[_pair_idx]
                X_train, X_val, fi = build_fold_hybrid(
                    _pair_reps, val_reps_tgt, y_pair, src_bp,
                    augmentations_train=_pair_augs,
                    augmentations_val=val_augs_tgt,
                    groups=groups[_pair_idx],
                    model=model,
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better)

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_va = scaler.transform(X_val)

                model.fit(X_tr, y_pair)
                score = float(metric_fn(
                    y_val_tgt, _predict_for_metric(model, X_va, metric_fn)))
                transfer_matrix[(src_fold, tgt_fold)] = score
            except Exception as exc:
                # N6: surface the failed cross-fold transfer instead of
                # silently shrinking the transfer matrix. Without this a thrown
                # transfer is indistinguishable from "not attempted", and the
                # aggregate quietly degrades (per_target shrinks, aggregate->None)
                # with no diagnostic. The diagonal is seeded in Step 1, so this
                # degrades rather than errors.
                print(f"    transfer src={src_fold}->tgt={tgt_fold} failed: "
                      f"{type(exc).__name__}: {exc}")
                continue

    # --- Step 3: Compute the deltas ---
    print(f"\n  Step 3: Transfer analysis")
    print("    Diagonal: recipe and features chosen on every fold but the "
          "target's, scored on the target. Off-diagonal: chosen on every fold "
          "but the target's and one more, then those same features applied to "
          "the target's rows. Nothing that produced a cell has seen the rows "
          "that cell is scored on. What differs between the two is which rows "
          "the selection saw — one fold's worth.")

    # For each target fold: compare own blueprint score vs cross-fold blueprint scores
    per_target = {}
    n_nonfinite_targets = 0  # folds dropped because a score was NaN/inf
    for tgt_fold in fold_blueprints:
        own_score = transfer_matrix.get((tgt_fold, tgt_fold))
        # Drop non-finite (NaN/inf) fold scores before averaging. A Spearman
        # (or other metric) on near-constant predictions returns NaN; if that
        # NaN own-fold score is left in, this fold's delta is NaN and np.mean
        # turns the ENTIRE aggregate (mean_delta, max_abs_delta, mean_abs_delta)
        # into NaN — a run whose other folds transferred fine gets reported as a
        # total blow-up. Skip the poisoned fold instead and count it.
        cross_scores = [transfer_matrix[(src, tgt_fold)]
                        for src in fold_blueprints
                        if src != tgt_fold and (src, tgt_fold) in transfer_matrix
                        and np.isfinite(transfer_matrix[(src, tgt_fold)])]
        if own_score is None or not np.isfinite(own_score):
            n_nonfinite_targets += 1
            continue
        if own_score is not None and cross_scores:
            mean_cross = float(np.mean(cross_scores))
            if lower_is_better:
                delta = mean_cross - own_score  # positive = cross is worse (higher error)
            else:
                delta = own_score - mean_cross  # positive = own is better
            per_target[tgt_fold] = {
                'own_score': own_score,
                'mean_cross_score': mean_cross,
                'cross_scores': cross_scores,
                'delta': delta,
            }
            print(f"    Fold {tgt_fold}: own={own_score:.4f}, "
                  f"mean_cross={mean_cross:.4f}, delta={delta:+.4f}")

    # Aggregate
    deltas = [v['delta'] for v in per_target.values()]
    mean_delta = float(np.mean(deltas)) if deltas else float('nan')
    max_delta = float(np.max(np.abs(deltas))) if deltas else float('nan')

    print(f"\n  Transfer summary:")
    print(f"    Mean |delta|: {np.mean(np.abs(deltas)):.4f}" if deltas else "    No data")
    print(f"    Max  |delta|: {max_delta:.4f}" if deltas else "")
    if deltas:
        print(f"    Interpretation: {'STABLE (small delta)' if np.mean(np.abs(deltas)) < 0.02 else 'SOME DRIFT' if np.mean(np.abs(deltas)) < 0.05 else 'UNSTABLE (large delta)'}")
        print("    Reads as: how much the held-out score moves when the "
              "recipe AND the selected features come from one fold's worth "
              "less data. Two folds landing on the same recipe no longer "
              "give a delta of exactly zero — the features cross now.")

    # --- Step 4: Nogueira stability index ---
    print(f"\n  Step 4: Nogueira stability index")

    feature_set_list = [fold_feature_sets[k] for k in sorted(fold_feature_sets.keys())]
    phi, se = nogueira_stability(feature_set_list, total_features)
    print(f"    Nogueira phi: {phi:.4f} (SE={se:.4f})")
    if not np.isnan(phi):
        ci_lo = phi - 1.96 * se
        ci_hi = phi + 1.96 * se
        print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"    Interpretation: {'HIGH stability' if phi > 0.7 else 'MODERATE stability' if phi > 0.4 else 'LOW stability'}")

    # --- Compile results ---
    results = {
        'n_folds': len(fold_blueprints),
        'total_features': total_features,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'transfer_matrix': {f"{s},{t}": v for (s, t), v in transfer_matrix.items()},
        # Which rows each off-diagonal cell's selection saw. The diagonal saw
        # every fold but its own; each off-diagonal cell saw every fold but
        # its own and the source's, so the off-diagonal sees one fold less
        # data. No cell has seen the rows it is scored on.
        'cell_recipe_provenance': cell_recipe_provenance,
        'diagonal_selection_saw': 'every fold except the target',
        'features_transferred': True,
        'delta_meaning': (
            'how much the held-out score moves when the recipe and the '
            'selected features come from a different subset of folds. The '
            'features themselves cross now, so a delta of exactly zero no '
            'longer follows from two folds sharing a recipe'),
        'per_target': convert_numpy(per_target),
        'aggregate': {
            'mean_abs_delta': float(np.mean(np.abs(deltas))) if deltas else None,
            'max_abs_delta': max_delta if deltas else None,
            'mean_delta': mean_delta,
            'n_target_folds': len(deltas),
            'n_nonfinite_targets': n_nonfinite_targets,
        },
        'stability': {
            'nogueira_phi': phi,
            'nogueira_se': se,
            'ci_95': [phi - 1.96 * se, phi + 1.96 * se] if not np.isnan(se) else None,
            'n_feature_sets': len(feature_set_list),
            'mean_features_selected': float(np.mean([len(fs) for fs in feature_set_list])) if feature_set_list else None,
        },
        'fold_blueprints': {
            k: {
                'budget': bp.budget,
                'allocation': bp.allocation_method,
                'importance': bp.importance_method,
                'internal_evaluator': bp.internal_evaluator,
            } for k, bp in fold_blueprints.items()
        },
    }

    # Summary table
    print(f"\n  BLUEPRINT TRANSFER SUMMARY (score per source-selection x target fold):")
    print(f"  {'Source→Target':<16} " + " ".join(f"T{t:<4}" for t in sorted(fold_blueprints.keys())))
    print("  " + "-" * (16 + 6 * len(fold_blueprints)))
    for src in sorted(fold_blueprints.keys()):
        row = f"  S{src:<15} "
        for tgt in sorted(fold_blueprints.keys()):
            score = transfer_matrix.get((src, tgt))
            if score is not None:
                marker = " *" if src == tgt else "  "
                row += f"{score:.3f}{marker}"
            else:
                row += "  -    "
        print(row)
    print("  (* = own-fold blueprint)")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C24: Blueprint Transfer Test',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c24_blueprint_transfer', args.dataset)
