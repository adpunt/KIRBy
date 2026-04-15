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
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split,
    get_precomputed_search, get_domain_metric,
)
from experiments.federated.federated_helpers import (
    compute_reg_metrics,
    compute_clf_metrics,
    predict_fluid_similarity_ensemble_regression,
    predict_fluid_quality_ensemble_regression,
    predict_fluid_similarity_ensemble_classification,
    predict_fluid_quality_ensemble_classification,
    fluid_within_kirby_across,
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
BUDGET = 100  # feature budget for KIRBy hybrid


def _get_silo_reps(dataset_name, available_reps):
    """Resolve silo reps for a dataset, using fallbacks if needed."""
    import warnings
    base = SILO_REPS_FALLBACK.get(dataset_name, SILO_REPS)
    resolved = {}
    for silo_letter, rep_name in base.items():
        if rep_name in available_reps:
            resolved[silo_letter] = rep_name
        else:
            # Try common fallbacks
            for fallback in ['chembert', 'avalon', 'graph_kernel']:
                if fallback in available_reps and fallback not in resolved.values():
                    resolved[silo_letter] = fallback
                    warnings.warn(
                        f"Silo {silo_letter}: '{rep_name}' unavailable, "
                        f"using fallback '{fallback}'")
                    break
            else:
                warnings.warn(
                    f"Silo {silo_letter}: '{rep_name}' unavailable and "
                    f"no fallback found — silo dropped")
    return resolved


def run(data, config, output_dir, dataset_name, **kwargs):
    """Fed02: Representation diversity as federated silos."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED02: REPRESENTATION DIVERSITY AS FEDERATED SILOS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data.get('task_type', 'regression')
    is_clf = task_type == 'classification'

    # ── Get optimal hybrid config from precomputed search ──
    augs = data.get('augmentations')
    aug_cond = 'uncapped' if augs else 'none'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    try:
        _search = get_precomputed_search(
            reps, labels, groups,
            budget=BUDGET, aug_condition=aug_cond,
            augmentations=augs,
            speed=config.get('search_speed', 'full'),
            precomputed=data.get('precomputed_searches'),
            metric=metric_name,
            task_type=data.get('task_type', 'regression'),
            domain=data.get('domain', 'molecular'))
        bp_entry = _search.best_by_model.get('rf')
        bp = bp_entry['blueprint'] if bp_entry else None
    except Exception:
        bp = None
    _alloc = bp.allocation_method if bp else 'performance_weighted'
    _imp = bp.importance_method if bp else 'treeshap'
    _int_eval = bp.internal_evaluator if bp else 'match'

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

    # 80/20 scaffold-group split (same for all conditions)
    n_samples = len(labels)
    train_idx, test_idx = next(cv_split(
        groups, 1, np.zeros(n_samples), labels))

    y_train = labels[train_idx]
    y_test = labels[test_idx]
    print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")

    # Pre-split reps
    train_reps = {r: arr[train_idx] for r, arr in reps.items()}
    test_reps = {r: arr[test_idx] for r, arr in reps.items()}

    # ── Phase 2: Single-silo baselines (C1) ──────────────────────────────
    print(f"\n--- Phase 2: Single-silo baselines (C1) [task={task_type}] ---")
    single_silo = {}

    def _make_model():
        if is_clf:
            return RandomForestClassifier(
                n_estimators=150, max_depth=15,
                class_weight='balanced', random_state=0)
        return RandomForestRegressor(
            n_estimators=150, max_depth=15, random_state=0)

    _compute_metrics = compute_clf_metrics if is_clf else compute_reg_metrics

    for s in silo_letters:
        rep_name = silo_reps[s]
        X_tr = train_reps[rep_name]
        X_te = test_reps[rep_name]

        mdl = _make_model()
        mdl.fit(X_tr, y_train)
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
            print(f"  Silo {s} ({rep_name}): "
                  f"RMSE={metrics['rmse']:.4f}  "
                  f"rho={metrics['spearman_rho']:.4f}  "
                  f"R2={metrics['r2']:.4f}")

    # Best silo: highest MCC for clf, lowest RMSE for regression
    if is_clf:
        best_silo = max(single_silo, key=lambda s: single_silo[s]['mcc'])
    else:
        best_silo = min(single_silo, key=lambda s: single_silo[s]['rmse'])
    print(f"  Best: Silo {best_silo} ({silo_reps[best_silo]})")

    # ── Phase 3: FLuID ensemble (C2, C3) ─────────────────────────────────
    print("\n--- Phase 3: FLuID ensemble conditions ---")

    # Train one model per silo
    teacher_models = [None]  # index 0 placeholder
    silo_train_features_sim = [None]   # for similarity ensemble
    silo_test_features_sim = [None]
    silo_train_features_qual = [None]  # for quality ensemble (needs labels)
    silo_test_features_qual = [None]

    for s in silo_letters:
        rep_name = silo_reps[s]
        X_tr = train_reps[rep_name]
        X_te = test_reps[rep_name]

        mdl = _make_model()
        mdl.fit(X_tr, y_train)

        teacher_models.append((mdl, len(X_tr), X_tr))
        silo_train_features_sim.append(X_tr)
        silo_test_features_sim.append(X_te)
        silo_train_features_qual.append((X_tr, y_train))
        silo_test_features_qual.append(X_te)

    # Condition 2: FLuID similarity-weighted ensemble
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

    # Condition 3: FLuID quality-weighted ensemble
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

    # ── Phase 4: KIRBy hybrid (C4, C5) ───────────────────────────────────
    print("\n--- Phase 4: KIRBy hybrid conditions ---")
    from kirby.hybrid import create_hybrid, apply_feature_selection

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
            allocation_method=_alloc,
            importance_method=_imp,
            budget=BUDGET,
            internal_evaluator=_int_eval)
        X_h_test = apply_feature_selection(kirby_test, fi_c4)

        mdl_c4 = _make_model()
        mdl_c4.fit(X_h_train, y_train)
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
        c4 = {'error': str(e)}
        print(f"FAILED: {e}")

    # Condition 5: KIRBy top-tier hybrid (top 3 reps by best silo metric)
    print("  C5: KIRBy top-tier (3)...", end=' ')
    try:
        if is_clf:
            silo_ranked = sorted(single_silo.items(),
                                 key=lambda x: x[1]['mcc'], reverse=True)
        else:
            silo_ranked = sorted(single_silo.items(),
                                 key=lambda x: x[1]['rmse'])
        top_3_silos = [s for s, _ in silo_ranked[:3]]
        top_3_names = [silo_reps[s] for s in top_3_silos]

        kirby_top3_train = {silo_reps[s]: train_reps[silo_reps[s]]
                            for s in top_3_silos}
        kirby_top3_test = {silo_reps[s]: test_reps[silo_reps[s]]
                           for s in top_3_silos}

        X_t3_train, fi_c5 = create_hybrid(
            kirby_top3_train, y_train,
            allocation_method=_alloc,
            importance_method=_imp,
            budget=BUDGET,
            internal_evaluator=_int_eval)
        X_t3_test = apply_feature_selection(kirby_top3_test, fi_c5)

        mdl_c5 = _make_model()
        mdl_c5.fit(X_t3_train, y_train)
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
        c5['allocation'] = convert_numpy(fi_c5.get('allocation', {}))
    except Exception as e:
        c5 = {'error': str(e)}
        print(f"FAILED: {e}")

    # ── Phase 5: FLuID→KIRBy pipeline (C6) ───────────────────────────────
    print("\n--- Phase 5: FLuID-within-KIRBy-across pipeline (C6) ---")
    print("  C6: FLuID→KIRBy...", end=' ')
    try:
        X_c6_train, X_c6_test, budgets_c6, scores_c6, meta_c6 = \
            fluid_within_kirby_across(
                kirby_train, kirby_test, y_train,
                k_intra=K_INTRA, budget=BUDGET,
                allocation_method=_alloc, importance_method=_imp,
                internal_evaluator=_int_eval)

        mdl_c6 = _make_model()
        mdl_c6.fit(X_c6_train, y_train)
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
        c6 = {'error': str(e)}
        print(f"FAILED: {e}")

    # ── Phase 6: Assemble results ─────────────────────────────────────────
    _primary_key = 'mcc' if is_clf else 'rmse'
    results = {
        'experiment': 'fed02_rep_diversity_silos',
        'dataset': dataset_name,
        'task_type': task_type,
        'config': {
            'silo_reps': {s: silo_reps[s] for s in silo_letters},
            'n_silos': K,
            'n_samples': len(labels),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'k_intra_c6': K_INTRA,
            'budget': BUDGET,
        },
        'c1_best_single_silo': convert_numpy({
            'best_silo': best_silo,
            'best_rep': silo_reps[best_silo],
            f'best_{_primary_key}': single_silo[best_silo][_primary_key],
            'per_silo': single_silo,
        }),
        'c2_fluid_similarity': convert_numpy(c2),
        'c3_fluid_quality': convert_numpy(c3),
        'c4_kirby_hybrid': convert_numpy(c4),
        'c5_kirby_top_tier': convert_numpy(c5),
        'c6_fluid_within_kirby_across': convert_numpy(c6),
        'privacy_notes': {
            'c1': 'each silo trains independently, no sharing',
            'c2': 'silos share predictions only',
            'c3': f'silos share predictions + one {"MCC" if is_clf else "RMSE"} scalar',
            'c4': 'centralized: all reps must be co-located',
            'c5': 'centralized: all reps must be co-located',
            'c6': 'intra-silo clustering local, cross-silo via federated FS',
        },
    }

    # Summary table
    print(f"\n{'─' * 70}")
    summary_items = [
        ('c1_best_single_silo', f'C1: Best silo ({silo_reps[best_silo]})'),
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
            r = results[key]
            if key == 'c1_best_single_silo':
                r = r.get('per_silo', {}).get(best_silo, {})
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
            r = results[key]
            if key == 'c1_best_single_silo':
                r = r.get('per_silo', {}).get(best_silo, {})
            rmse = r.get('rmse', float('nan'))
            rho = r.get('spearman_rho', float('nan'))
            r2 = r.get('r2', float('nan'))
            if np.isnan(rmse):
                print(f"  {name:<35} {'N/A':>8} {'N/A':>10} {'N/A':>8}")
            else:
                print(f"  {name:<35} {rmse:>8.4f} {rho:>10.4f} {r2:>8.4f}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('Fed02: Rep Diversity Silos')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, 'fed02_rep_diversity_silos', args.dataset)
