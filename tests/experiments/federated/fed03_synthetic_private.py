#!/usr/bin/env python3
"""Fed03: Synthetic Private Datasets + Antibody Federated Extension.

Three sub-experiments:
    Part A (herg): Source-based hERG — Lhasa curated data + KIRBy ChEMBL
        as real federated silos with genuine source heterogeneity.
    Part B (herg_asymmetric): Asymmetric rep silos — each org contributes
        different representation types.
    Part C (ddg_combined / abagym_HER2): Antibody federated — natural silos
        from DMS, experimental DDG, synthetic DDG, and developability data.

Dispatches sub-experiment based on dataset_name argument.

Usage:
    python fed03_synthetic_private.py --dataset herg
    python fed03_synthetic_private.py --dataset herg_asymmetric
    python fed03_synthetic_private.py --dataset ddg_combined
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import traceback
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split,
    get_precomputed_search, get_best_search_for_downstream, get_domain_metric,
)
from experiments.federated.federated_helpers import (
    compute_clf_metrics, compute_reg_metrics,
    predict_fluid_similarity_ensemble,
    predict_fluid_quality_ensemble,
    federated_feature_selection,
    importance_profile_transfer,
    _resolve_match,
    smoke_n_folds,
    fit_fed_classifier, fit_fed_regressor, fit_fed_model,
    fed_to_kirby_model, make_fed_classifier, make_fed_regressor,
)


# ── Constants ────────────────────────────────────────────────────────────────

PKI_THRESHOLD = 6.0

# Strategy 4: asymmetric rep assignments per silo
ASYMMETRIC_SILO_REPS = {
    'structural': ['ecfp4', 'graph_kernel', 'atom_pair'],
    'physicochemical': ['mordred', 'pdv'],
    'learned': ['molformer', 'chemberta'],
}

# Antibody datasets that trigger Part C
ANTIBODY_DATASETS = {
    'ddg_combined', 'abagym_HER2', 'graphinity_foldx',
    'flab_thermo_tresanco',
}


# =============================================================================
# PART A: SOURCE-BASED hERG
# =============================================================================

def _run_herg_source_based(data, config, output_dir, dataset_name,
                           fed_model='rf', fed_seed=0):
    """Strategy 2: Real data sources as federated silos.

    Silo A: KIRBy ChEMBL hERG (1,415 mols, continuous pKi → binarized)
    Silo B: Lhasa training (7,772 mols, binary Active/Inactive)
    Silo C: Lhasa test (4,383 mols, binary Active/Inactive)
    """
    from experiments.federated.lhasa_herg_loader import (
        load_lhasa_herg_training, load_lhasa_herg_test,
        dedup_across_silos,
    )
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    print("\n" + "=" * 70)
    print("FED03 PART A: SOURCE-BASED hERG (Strategy 2)")
    print("=" * 70)

    reps_chembl = data['reps']
    labels_chembl = data['labels']
    smiles_chembl = data['smiles']

    # Get optimal hybrid config from precomputed search
    augs = data.get('augmentations')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    try:
        _search, _meta = get_best_search_for_downstream(
            reps_chembl, labels_chembl, data.get('groups'),
            downstream_model=fed_to_kirby_model(fed_model), budget=100,
            augmentations=augs,
            metric=metric_name,
            task_type=data['task_type'],
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            data=data,
        )
        _bp = _meta['best_config']['blueprint'] if _meta else None
    except Exception:
        _bp = None
    # Part A does not pass augmentations to create_hybrid (C6 uses 'none'
    # explicitly because Lhasa rows have no aligned aug arrays). The pool's
    # aug_condition is therefore not consumed here — removed dead aug_cond.
    _alloc = _bp.allocation_method if _bp else 'performance_weighted'
    _imp = _bp.importance_method if _bp else 'treeshap'
    _int_eval = _bp.internal_evaluator if _bp else 'match'

    # Binarize ChEMBL labels. When the calling dataset already has 0/1 labels
    # (e.g. load_data('herg') returns binary blocker labels — no continuous
    # pKi), the pKi>=6.0 threshold would collapse every sample to class 0.
    # Detect already-binary labels by value and pass them through.
    _uniq_chembl = set(np.unique(labels_chembl).tolist())
    if data.get('is_binary') or _uniq_chembl <= {0, 1, 0.0, 1.0}:
        labels_chembl_binary = labels_chembl.astype(int)
    else:
        labels_chembl_binary = (labels_chembl >= PKI_THRESHOLD).astype(int)

    # Compute InChIKeys for ChEMBL
    chembl_inchikeys = []
    for smi in smiles_chembl:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                chembl_inchikeys.append(
                    InchiToInchiKey(MolToInchi(mol)))
            else:
                chembl_inchikeys.append('UNKNOWN')
        except Exception:
            chembl_inchikeys.append('UNKNOWN')

    # Load Lhasa data
    print("\nLoading Lhasa hERG data...")
    lhasa_train_smiles, lhasa_train_labels, lhasa_train_keys, _ = \
        load_lhasa_herg_training()
    lhasa_test_smiles, lhasa_test_labels, lhasa_test_keys, _ = \
        load_lhasa_herg_test()

    # Cross-silo deduplication (ChEMBL first → keeps ChEMBL, removes from Lhasa)
    silo_smiles = {
        'chembl': smiles_chembl,
        'lhasa_train': lhasa_train_smiles,
        'lhasa_test': lhasa_test_smiles,
    }
    silo_labels = {
        'chembl': labels_chembl_binary,
        'lhasa_train': lhasa_train_labels,
        'lhasa_test': lhasa_test_labels,
    }
    silo_keys = {
        'chembl': chembl_inchikeys,
        'lhasa_train': lhasa_train_keys,
        'lhasa_test': lhasa_test_keys,
    }

    deduped_smiles, deduped_labels, deduped_keys, dedup_stats = \
        dedup_across_silos(silo_smiles, silo_labels, silo_keys)

    print("\nDeduplication:")
    for silo, stats in dedup_stats.items():
        print(f"  {silo}: {stats['before']} → {stats['after']} "
              f"({stats['removed']} removed)")

    # Pool all silos for evaluation
    all_smiles = []
    all_labels = []
    silo_assignments = []
    silo_names = ['chembl', 'lhasa_train', 'lhasa_test']

    for i, sname in enumerate(silo_names):
        sm = deduped_smiles[sname]
        lb = deduped_labels[sname]
        all_smiles.extend(sm)
        all_labels.extend(lb)
        silo_assignments.extend([i] * len(sm))

    all_labels = np.array(all_labels, dtype=int)
    silo_assignments = np.array(silo_assignments)
    n_total = len(all_labels)

    print(f"\nPooled: {n_total} molecules across 3 silos")
    for i, sname in enumerate(silo_names):
        n_s = int((silo_assignments == i).sum())
        n_act = int(all_labels[silo_assignments == i].sum())
        print(f"  Silo {i} ({sname}): {n_s} mols "
              f"({n_act} active, {n_s - n_act} inactive)")

    # Compute representations for ALL molecules (pooled)
    # For ChEMBL molecules, we already have reps; for Lhasa we need to compute
    print("\nComputing representations for Lhasa molecules...")
    from find_best_hybrid_molecular import get_representations as get_molecular_representations
    rep_names = list(reps_chembl.keys())

    # Filter ChEMBL reps to match deduped molecule count.
    # dedup_across_silos may have removed UNKNOWN-keyed ChEMBL mols.
    n_chembl_orig = len(smiles_chembl)
    n_chembl_deduped = len(deduped_smiles['chembl'])
    if n_chembl_deduped < n_chembl_orig:
        # Build keep mask: which original ChEMBL indices survived dedup
        deduped_chembl_set = set(deduped_keys['chembl'])
        chembl_keep = np.array([
            k in deduped_chembl_set
            for k in chembl_inchikeys], dtype=bool)
        reps_chembl = {r: arr[chembl_keep] for r, arr in reps_chembl.items()}
        print(f"  ChEMBL filtered: {n_chembl_orig} → {n_chembl_deduped} "
              f"(matched dedup)")

    n_chembl = n_chembl_deduped
    n_lhasa_tr = len(deduped_smiles['lhasa_train'])
    n_lhasa_te = len(deduped_smiles['lhasa_test'])

    # Get reps for Lhasa molecules. Use the SAME rep set as the ChEMBL silo
    # (rep_names) rather than the hardcoded 'all' preset: 'all' forces every
    # foundation-model rep even though only rep_names are consumed below (the
    # combine loop iterates rep_names), wasting compute and making this path
    # un-runnable where those models are unavailable. get_representations takes
    # a comma-separated rep list as its second argument.
    lhasa_all_smiles = (deduped_smiles['lhasa_train'] +
                        deduped_smiles['lhasa_test'])
    lhasa_reps = get_molecular_representations(lhasa_all_smiles, ','.join(rep_names))

    # Combine: ChEMBL reps + Lhasa reps
    combined_reps = {}
    for rname in rep_names:
        chembl_arr = reps_chembl[rname]
        if rname in lhasa_reps:
            lhasa_arr = lhasa_reps[rname]
            if chembl_arr.shape[1] != lhasa_arr.shape[1]:
                warnings.warn(
                    f"Rep {rname}: ChEMBL has {chembl_arr.shape[1]}D, "
                    f"Lhasa has {lhasa_arr.shape[1]}D — skipping")
                continue
            combined_reps[rname] = np.vstack([chembl_arr, lhasa_arr])
        else:
            warnings.warn(f"Rep {rname} not available for Lhasa molecules")

    print(f"  Combined reps: {len(combined_reps)} reps, "
          f"{n_total} molecules")

    # Train/test split: scaffold-group-aware for molecular datasets.
    # Per feedback_no_random_splits.md, a single 80/20 split is not enough;
    # use 5-fold scaffold-group CV and accumulate per-fold metrics.
    from find_best_hybrid_molecular import assign_scaffold_groups
    pooled_groups, _ = assign_scaffold_groups(all_smiles)
    if pooled_groups is None:
        raise RuntimeError(
            "Part A 5-fold CV requires scaffold groups; pooled_groups is None.")
    n_unique_groups = len(np.unique(pooled_groups))
    if n_unique_groups < 5:
        raise RuntimeError(
            f"Part A 5-fold CV requires >= 5 unique scaffold groups; "
            f"got {n_unique_groups}.")

    K = len(silo_names)

    # Build Morgan FPs for FLuID conditions (once, all rows)
    from rdkit.Chem import AllChem, DataStructs

    def _compute_morgan(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(2048, dtype=np.int8)
        arr = np.zeros(2048, dtype=np.int8)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048, useFeatures=True)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    all_fps = np.array([_compute_morgan(s) for s in all_smiles])

    # ── Per-fold accumulators ─────────────────────────────────────────────
    per_fold_results = []
    n_folds = smoke_n_folds(5)   # 2 under KIRBY_SMOKE_N, else 5
    folds = list(cv_split(
        pooled_groups, n_folds, np.zeros(n_total), all_labels, seed=fed_seed))
    print(f"\n{n_folds}-fold scaffold CV over {n_unique_groups} groups")

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n── Fold {fold_idx + 1}/{n_folds} "
              f"({len(train_idx)} train / {len(test_idx)} test) ──")
        y_train = all_labels[train_idx]
        y_test = all_labels[test_idx]
        train_silo = silo_assignments[train_idx]

        # Build silo indices relative to train set (per fold)
        train_silo_indices = {}
        for i, sname in enumerate(silo_names):
            train_silo_indices[i + 1] = np.where(train_silo == i)[0]

        train_fps = all_fps[train_idx]
        test_fps = all_fps[test_idx]

        # ── Conditions ────────────────────────────────────────────────
        results = {}

        # C1: Best single silo
        print("  C1: Single silo baselines...")
        single_silo = {}
        for i, sname in enumerate(silo_names):
            s_idx = train_silo_indices[i + 1]
            if len(s_idx) < 10:
                single_silo[sname] = {'error': 'too few samples'}
                continue
            clf = fit_fed_classifier(
                fed_model, train_fps[s_idx], y_train[s_idx], seed=fed_seed)
            pred = clf.predict(test_fps)
            pred_prob = clf.predict_proba(test_fps)[:, 1]
            single_silo[sname] = compute_clf_metrics(y_test, pred, pred_prob)
            print(f"    {sname}: MCC={single_silo[sname]['mcc']:.3f}")

        valid_silos = [s for s in single_silo if 'mcc' in single_silo[s]]
        best_silo = (max(valid_silos, key=lambda s: single_silo[s]['mcc'])
                     if valid_silos else None)
        results['c1_single_silo'] = convert_numpy({
            'per_silo': single_silo,
            'best_silo': best_silo,
        })

        # Hoisted: train teachers per silo BEFORE C2/C3 so both conditions
        # share the same state. Previously teacher_models/cluster_data_train
        # were defined inside C2's try block, causing NameError on C3 if C2
        # raised before completing (or silent stale-state reuse).
        teachers_ready = False
        teacher_models = [None]
        cluster_data_train = [None]
        try:
            for i, sname in enumerate(silo_names):
                s_idx = train_silo_indices[i + 1]
                X_s = train_fps[s_idx]
                y_s = y_train[s_idx]
                clf = fit_fed_classifier(fed_model, X_s, y_s, seed=fed_seed)

                # Create minimal DataFrame for similarity ensemble
                import pandas as pd_local
                df_s = pd_local.DataFrame({
                    'FP': list(X_s),
                    'CLASS': y_s,
                })
                teacher_models.append((clf, len(X_s), df_s))
                cluster_data_train.append(df_s)
            teachers_ready = True
        except Exception as e:
            print(f"    Teacher training FAILED: {e}")

        # C2: FLuID similarity ensemble
        print("  C2: FLuID similarity ensemble...")
        if not teachers_ready:
            results['c2_fluid_similarity'] = {'error': 'teachers not trained'}
            print(f"    SKIPPED: teachers not trained")
        else:
            try:
                pred_cls, pred_prob, tw = predict_fluid_similarity_ensemble(
                    teacher_models, cluster_data_train, test_fps, K)
                c2 = compute_clf_metrics(y_test, pred_cls, pred_prob)
                c2['teacher_weights_mean'] = convert_numpy(
                    {silo_names[i]: float(tw[:, i].mean())
                     for i in range(K)})
                results['c2_fluid_similarity'] = convert_numpy(c2)
                print(f"    MCC={c2['mcc']:.3f}")
            except Exception as e:
                results['c2_fluid_similarity'] = {'error': str(e)}
                print(f"    FAILED: {e}")

        # C3: FLuID quality ensemble
        print("  C3: FLuID quality ensemble...")
        if not teachers_ready:
            results['c3_fluid_quality'] = {'error': 'teachers not trained'}
            print(f"    SKIPPED: teachers not trained")
        else:
            try:
                pred_cls, pred_prob, mcc_w = predict_fluid_quality_ensemble(
                    teacher_models, cluster_data_train, test_fps, K)
                c3 = compute_clf_metrics(y_test, pred_cls, pred_prob)
                c3['mcc_weights'] = convert_numpy(mcc_w)
                results['c3_fluid_quality'] = convert_numpy(c3)
                print(f"    MCC={c3['mcc']:.3f}")
            except Exception as e:
                results['c3_fluid_quality'] = {'error': str(e)}
                print(f"    FAILED: {e}")

        # C4: KIRBy federated feature selection
        print("  C4: KIRBy federated FS...")
        try:
            train_reps = {r: arr[train_idx] for r, arr in combined_reps.items()}
            test_reps = {r: arr[test_idx] for r, arr in combined_reps.items()}

            X_h_train, silo_budgets, silo_scores, silo_fis = \
                federated_feature_selection(
                    train_reps, y_train, train_silo_indices,
                    K,
                    allocation_method=_alloc, importance_method=_imp,
                    internal_evaluator=_int_eval,
                    downstream_model=fed_to_kirby_model(fed_model),
                    metric_fn=metric_fn, lower_is_better=lower_is_better)

            from kirby.hybrid import apply_feature_selection
            h_test_parts = []
            for s in range(1, K + 1):
                fi = silo_fis.get(s)
                if fi is None:
                    continue
                try:
                    h_test_parts.append(
                        apply_feature_selection(test_reps, fi))
                except Exception:
                    continue

            if h_test_parts:
                X_h_test = np.hstack(h_test_parts)
                clf = fit_fed_classifier(
                    fed_model, X_h_train, y_train, seed=fed_seed)
                pred_cls = clf.predict(X_h_test)
                pred_prob = clf.predict_proba(X_h_test)[:, 1]
                c4 = compute_clf_metrics(y_test, pred_cls, pred_prob)
            else:
                c4 = {'error': 'no valid silo selections'}
            c4['silo_budgets'] = convert_numpy(silo_budgets)
            c4['silo_scores'] = convert_numpy(silo_scores)
            results['c4_kirby_federated'] = convert_numpy(c4)
            mcc_str = f"MCC={c4['mcc']:.3f}" if 'mcc' in c4 else 'N/A'
            print(f"    {mcc_str}  budgets={silo_budgets}")
        except Exception as e:
            # Surface the traceback (not just str(e)) so a shape/import/NameError
            # in the federated_feature_selection path is debuggable from the
            # expensive run's logs (FED-21).
            results['c4_kirby_federated'] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"    FAILED: {e}")
            traceback.print_exc()

        # C5: Naive pool (unrealistic baseline)
        print("  C5: Naive pool...")
        clf_pool = fit_fed_classifier(
            fed_model, train_fps, y_train, seed=fed_seed)
        pred_pool = clf_pool.predict(test_fps)
        pred_pool_prob = clf_pool.predict_proba(test_fps)[:, 1]
        c5 = compute_clf_metrics(y_test, pred_pool, pred_pool_prob)
        results['c5_naive_pool'] = convert_numpy(c5)
        print(f"    MCC={c5['mcc']:.3f}")

        # C6: KIRBy single-rep control
        print("  C6: KIRBy single-rep (ecfp4 only)...")
        try:
            from kirby.hybrid import create_hybrid, apply_feature_selection
            ecfp4_key = 'ecfp4' if 'ecfp4' in combined_reps else None
            if ecfp4_key is None:
                ecfp4_key = next(iter(combined_reps))
            single_reps = {ecfp4_key: combined_reps[ecfp4_key]}
            train_single = {ecfp4_key: single_reps[ecfp4_key][train_idx]}
            test_single = {ecfp4_key: single_reps[ecfp4_key][test_idx]}
            # Part A pools ChEMBL + Lhasa molecules, but Lhasa-only
            # augmentations don't align with the combined row count.
            # Explicit 'none' avoids cache-key collision.
            X_h_c6, fi_c6 = create_hybrid(
                train_single, y_train,
                allocation_method=_alloc,
                importance_method=_imp,
                internal_evaluator=_int_eval,
                model=make_fed_classifier(fed_model, seed=fed_seed),
                metric_fn=metric_fn, lower_is_better=lower_is_better,
                augmentation_strategy='none')
            X_h_c6_test = apply_feature_selection(test_single, fi_c6)
            clf_c6 = fit_fed_classifier(
                fed_model, X_h_c6, y_train, seed=fed_seed)
            pred_c6 = clf_c6.predict(X_h_c6_test)
            prob_c6 = clf_c6.predict_proba(X_h_c6_test)[:, 1]
            c6 = compute_clf_metrics(y_test, pred_c6, prob_c6)
            c6['rep_used'] = ecfp4_key
            results['c6_kirby_single_rep'] = convert_numpy(c6)
            print(f"    MCC={c6['mcc']:.3f}")
        except Exception as e:
            results['c6_kirby_single_rep'] = {'error': str(e)}
            print(f"    FAILED: {e}")

        # C7: FLuID multi-rep ensemble
        print("  C7: FLuID multi-rep ensemble...")
        try:
            rep_teachers = []
            rep_names_list = sorted(combined_reps.keys())
            for rname in rep_names_list:
                X_tr = combined_reps[rname][train_idx]
                X_te = combined_reps[rname][test_idx]
                clf_r = fit_fed_classifier(
                    fed_model, X_tr, y_train, seed=fed_seed)
                rep_teachers.append((clf_r, rname, X_tr, X_te))

            from sklearn.model_selection import cross_val_predict, KFold
            from sklearn.metrics import matthews_corrcoef
            # Default cv=3 is KFold(shuffle=False); on silo-concatenated
            # rows this puts fold 0 entirely in one silo. Use shuffled.
            cv_c7 = KFold(n_splits=3, shuffle=True, random_state=42)
            weights = []
            for clf_r, rname, X_tr, _ in rep_teachers:
                try:
                    cv_pred = cross_val_predict(
                        RandomForestClassifier(
                            n_estimators=100, max_depth=15,
                            class_weight='balanced', random_state=0),
                        X_tr, y_train, cv=cv_c7)
                    w = max(matthews_corrcoef(y_train, cv_pred), 0.01)
                except Exception:
                    w = 0.01
                weights.append(w)
            weights = np.array(weights)
            weights /= weights.sum()

            ensemble_probs = np.zeros(len(test_idx))
            for i, (clf_r, rname, _, X_te) in enumerate(rep_teachers):
                probs = clf_r.predict_proba(X_te)
                if probs.shape[1] == 2:
                    ensemble_probs += weights[i] * probs[:, 1]
                else:
                    ensemble_probs += weights[i] * probs.max(axis=1)
            pred_c7 = (ensemble_probs >= 0.5).astype(int)
            c7 = compute_clf_metrics(y_test, pred_c7, ensemble_probs)
            c7['n_rep_teachers'] = len(rep_teachers)
            c7['teacher_weights'] = convert_numpy(
                {rname: float(w) for (_, rname, _, _), w
                 in zip(rep_teachers, weights)})
            results['c7_fluid_multi_rep'] = convert_numpy(c7)
            print(f"    MCC={c7['mcc']:.3f}  ({len(rep_teachers)} teachers)")
        except Exception as e:
            results['c7_fluid_multi_rep'] = {'error': str(e)}
            print(f"    FAILED: {e}")

        per_fold_results.append(results)

    # ── Aggregate across folds ──
    def _agg_mcc(cond_key, fold_results):
        vals = []
        for fr in fold_results:
            r = fr.get(cond_key, {})
            if cond_key == 'c1_single_silo':
                # Use the per-fold best_silo's MCC
                bs = r.get('best_silo')
                if bs is None:
                    continue
                rr = r.get('per_silo', {}).get(bs, {})
                if 'mcc' in rr:
                    vals.append(rr['mcc'])
            else:
                if 'mcc' in r:
                    vals.append(r['mcc'])
        return vals

    aggregated = {}
    cond_keys = [
        'c1_single_silo', 'c2_fluid_similarity', 'c3_fluid_quality',
        'c4_kirby_federated', 'c5_naive_pool', 'c6_kirby_single_rep',
        'c7_fluid_multi_rep',
    ]
    for ck in cond_keys:
        vals = _agg_mcc(ck, per_fold_results)
        aggregated[ck] = {
            'mcc': float(np.mean(vals)) if vals else float('nan'),
            'mcc_mean': float(np.mean(vals)) if vals else float('nan'),
            'mcc_std': float(np.std(vals)) if vals else float('nan'),
            'mcc_per_fold': [float(v) for v in vals],
            'n_folds_completed': len(vals),
            'per_fold': [fr.get(ck, {}) for fr in per_fold_results],
        }

    # Summary
    print(f"\n  {'Condition':<30} {'MCC (mean ± std)':>20}")
    for key, name in [
        ('c1_single_silo', 'C1: Best silo (per-fold)'),
        ('c2_fluid_similarity', 'C2: FLuID similarity'),
        ('c3_fluid_quality', 'C3: FLuID quality'),
        ('c4_kirby_federated', 'C4: KIRBy federated'),
        ('c5_naive_pool', 'C5: Naive pool'),
        ('c6_kirby_single_rep', 'C6: KIRBy single-rep'),
        ('c7_fluid_multi_rep', 'C7: FLuID multi-rep'),
    ]:
        a = aggregated.get(key, {})
        m = a.get('mcc_mean', float('nan'))
        s = a.get('mcc_std', float('nan'))
        line = (f"{m:.3f} ± {s:.3f}" if not np.isnan(m) else 'N/A')
        print(f"  {name:<30} {line:>20}")

    return {
        'experiment': 'fed03_source_based_herg',
        'dataset': dataset_name,
        'config': {
            'silos': silo_names,
            'n_total': n_total,
            'dedup_stats': dedup_stats,
            'pki_threshold': PKI_THRESHOLD,
            'n_folds': n_folds,
        },
        **aggregated,
    }


# =============================================================================
# PART B: ASYMMETRIC REP SILOS
# =============================================================================

def _run_asymmetric_reps(data, config, output_dir, dataset_name,
                         fed_model='rf', fed_seed=0):
    """Strategy 4: Each org contributes different representation types.

    Silo A (structural): ecfp4, graph_kernel, atom_pair
    Silo B (physicochemical): mordred, pdv
    Silo C (learned): molformer, chemberta
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection

    print("\n" + "=" * 70)
    print("FED03 PART B: ASYMMETRIC REP SILOS (Strategy 4)")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')

    # Get optimal hybrid config from precomputed search
    augs = data.get('augmentations')
    metric_name_b, metric_fn_b, lower_is_better_b = get_domain_metric(data)
    try:
        _search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model=fed_to_kirby_model(fed_model), budget=100,
            augmentations=augs,
            metric=metric_name_b,
            task_type=data['task_type'],
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name,
            data=data,
        )
        _bp = _meta['best_config']['blueprint'] if _meta else None
    except Exception:
        _bp = None
    # 'capped' blueprints are tuned for the full top-tier rep set; using them
    # on Part B's silo subset (a small asymmetric slice) is a recipe
    # mismatch. Force fallback to non-capped defaults in that case.
    if _bp is not None and getattr(_bp, 'augmentation_strategy', None) == 'capped':
        warnings.warn(
            "Part B: ignoring 'capped' blueprint (tuned for top-tier rep set, "
            "not asymmetric silo subset). Falling back to uncapped/none.")
        _bp = None
    aug_cond = _bp.augmentation_strategy if _bp else ('uncapped' if augs else 'none')
    _alloc = _bp.allocation_method if _bp else 'performance_weighted'
    _imp = _bp.importance_method if _bp else 'treeshap'
    _int_eval = _bp.internal_evaluator if _bp else 'match'

    # Resolve available reps per silo
    silo_reps = {}
    for silo_name, rep_list in ASYMMETRIC_SILO_REPS.items():
        available = [r for r in rep_list if r in reps]
        if available:
            silo_reps[silo_name] = available
        else:
            warnings.warn(f"Silo {silo_name}: none of {rep_list} available")

    K = len(silo_reps)
    silo_names = sorted(silo_reps.keys())
    print(f"  {K} silos resolved:")
    for sname in silo_names:
        rep_list = silo_reps[sname]
        dims = [reps[r].shape[1] for r in rep_list]
        print(f"    {sname}: {rep_list} ({sum(dims)}D total)")

    # 5-fold scaffold-group-aware CV (per feedback_no_random_splits.md).
    n_samples = len(labels)
    if groups is None:
        raise RuntimeError(
            "Part B 5-fold CV requires groups (scaffold-aware); groups=None.")
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < 5:
        raise RuntimeError(
            f"Part B 5-fold CV requires >= 5 unique groups; "
            f"got {n_unique_groups}.")

    # Use the dataset's declared task_type (FED-6) — the cardinality heuristic
    # (len(unique)<=10) misroutes low-cardinality regression targets into the
    # classification path. Fall back to the heuristic only if task_type absent.
    _task_type_b = data.get('task_type')
    if _task_type_b in ('classification', 'regression'):
        is_clf = (_task_type_b == 'classification')
    else:
        is_clf = len(np.unique(labels)) <= 10
    metrics_fn = compute_clf_metrics if is_clf else compute_reg_metrics

    n_folds = smoke_n_folds(5)   # 2 under KIRBY_SMOKE_N, else 5
    folds = list(cv_split(groups, n_folds, np.zeros(n_samples), labels,
                          seed=fed_seed))
    print(f"  {n_folds}-fold CV over {n_unique_groups} groups "
          f"({'clf' if is_clf else 'reg'})")

    per_fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n── Fold {fold_idx + 1}/{n_folds} "
              f"({len(train_idx)} train / {len(test_idx)} test) ──")
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        train_reps = {r: arr[train_idx] for r, arr in reps.items()}
        test_reps = {r: arr[test_idx] for r, arr in reps.items()}

        results = {}

        # C1: Best single silo
        print("  C1: Single silo baselines...")
        single_silo = {}
        for sname in silo_names:
            rep_list = silo_reps[sname]
            X_tr = np.hstack([train_reps[r] for r in rep_list])
            X_te = np.hstack([test_reps[r] for r in rep_list])

            model = fit_fed_model(
                fed_model, X_tr, y_train, seed=fed_seed,
                task_type=('classification' if is_clf else 'regression'))
            y_pred = model.predict(X_te)
            if is_clf:
                y_prob = model.predict_proba(X_te)[:, 1]
                single_silo[sname] = metrics_fn(y_test, y_pred, y_prob)
            else:
                single_silo[sname] = metrics_fn(y_test, y_pred)
            single_silo[sname]['n_features'] = int(X_tr.shape[1])

            primary = single_silo[sname].get(
                'mcc', single_silo[sname].get('rmse'))
            print(f"    {sname}: {primary:.4f}  ({X_tr.shape[1]}D)")

        results['c1_single_silo'] = convert_numpy(single_silo)

        # C2: KIRBy hybrid across all silo reps
        print("  C2: KIRBy hybrid (all reps)...")
        try:
            all_silo_reps_tr = {}
            all_silo_reps_te = {}
            for sname in silo_names:
                for r in silo_reps[sname]:
                    all_silo_reps_tr[r] = train_reps[r]
                    all_silo_reps_te[r] = test_reps[r]
            train_augs_c2 = ({k: v[train_idx] for k, v in augs.items()}
                             if augs else None)
            test_augs_c2 = ({k: v[test_idx] for k, v in augs.items()}
                            if augs else None)

            _c2_model = (
                make_fed_classifier(fed_model, seed=fed_seed) if is_clf
                else make_fed_regressor(fed_model, seed=fed_seed))
            X_h_tr, fi = create_hybrid(
                all_silo_reps_tr, y_train,
                allocation_method=_alloc,
                importance_method=_imp,
                internal_evaluator=_int_eval,
                model=_c2_model,
                metric_fn=metric_fn_b, lower_is_better=lower_is_better_b,
                augmentations=train_augs_c2,
                augmentation_strategy=aug_cond)
            X_h_te = apply_feature_selection(all_silo_reps_te, fi)
            _aug_info = (fi.get('augmentation_info')
                         if isinstance(fi, dict) else None)
            if _aug_info and test_augs_c2:
                _X_te_aug = apply_augmentation_selection(
                    test_augs_c2, _aug_info)
                if _X_te_aug is not None:
                    X_h_te = np.hstack([X_h_te, _X_te_aug])

            model = fit_fed_model(
                fed_model, X_h_tr, y_train, seed=fed_seed,
                task_type=('classification' if is_clf else 'regression'))
            y_pred = model.predict(X_h_te)
            if is_clf:
                c2 = metrics_fn(y_test, y_pred,
                                model.predict_proba(X_h_te)[:, 1])
            else:
                c2 = metrics_fn(y_test, y_pred)
            c2['allocation'] = convert_numpy(fi.get('allocation', {}))
            c2['n_features'] = int(X_h_tr.shape[1])
            results['c2_kirby_hybrid'] = convert_numpy(c2)
            primary = c2.get('mcc', c2.get('rmse'))
            print(f"    {primary:.4f}  ({c2['n_features']}D)")
        except Exception as e:
            results['c2_kirby_hybrid'] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"    FAILED: {e}")
            traceback.print_exc()

        # C3: Federated feature selection (privacy-preserving)
        print("  C3: Federated FS (per-silo importance)...")
        try:
            from kirby.hybrid import create_hybrid as _ch
            from sklearn.model_selection import (
                GroupShuffleSplit as _GSS, ShuffleSplit as _SS)
            from experiments._shared import get_model_factories

            silo_task_type = 'classification' if is_clf else 'regression'
            train_groups = groups[train_idx] if groups is not None else None

            # Resolve 'match' → concrete evaluator ('rf' default) BEFORE the
            # create_hybrid call. Without this, _int_eval='match' + no model=
            # arg makes create_hybrid silently fall back to 'ridge'
            # (hybrid.py:3354-3355), which is forbidden (feedback_no_ridge.md).
            _resolved_eval = _resolve_match(_int_eval, 'rf')
            _silo_model_c3 = (
                make_fed_classifier(fed_model, seed=fed_seed) if is_clf
                else make_fed_regressor(fed_model, seed=fed_seed))

            silo_fis = {}
            silo_scores = {}
            for i, sname in enumerate(silo_names, 1):
                rep_list = silo_reps[sname]
                silo_rep_tr = {r: train_reps[r] for r in rep_list}
                try:
                    X_s, fi_s = _ch(
                        silo_rep_tr, y_train,
                        allocation_method=_alloc,
                        importance_method=_imp,
                        internal_evaluator=_resolved_eval,
                        model=_silo_model_c3)
                    silo_fis[i] = fi_s
                    if (train_groups is not None
                            and len(np.unique(train_groups)) >= 2):
                        splitter = _GSS(
                            n_splits=1, test_size=0.2,
                            random_state=42 + i + fold_idx)
                        tr, va = next(splitter.split(
                            X_s, y_train, groups=train_groups))
                    else:
                        splitter = _SS(
                            n_splits=1, test_size=0.2,
                            random_state=42 + i + fold_idx)
                        tr, va = next(splitter.split(X_s))
                    eval_factory = get_model_factories(
                        [_resolved_eval], task_type=silo_task_type,
                        n_samples=len(tr)).get(_resolved_eval)
                    if eval_factory is None:
                        silo_scores[i] = 0.0
                    else:
                        evaluator = eval_factory()
                        evaluator.fit(X_s[tr], y_train[tr])
                        silo_scores[i] = max(0.0, float(
                            evaluator.score(X_s[va], y_train[va])))
                except Exception:
                    silo_fis[i] = None
                    silo_scores[i] = 0.0

            h_parts_tr = []
            h_parts_te = []
            for i, sname in enumerate(silo_names, 1):
                fi = silo_fis.get(i)
                if fi is None:
                    continue
                rep_list = silo_reps[sname]
                silo_rep_tr = {r: train_reps[r] for r in rep_list}
                silo_rep_te = {r: test_reps[r] for r in rep_list}
                try:
                    h_parts_tr.append(
                        apply_feature_selection(silo_rep_tr, fi))
                    h_parts_te.append(
                        apply_feature_selection(silo_rep_te, fi))
                except Exception:
                    continue

            if h_parts_tr:
                X_fed_tr = np.hstack(h_parts_tr)
                X_fed_te = np.hstack(h_parts_te)
                model = fit_fed_model(
                    fed_model, X_fed_tr, y_train, seed=fed_seed,
                    task_type=silo_task_type)
                y_pred = model.predict(X_fed_te)
                if is_clf:
                    c3 = metrics_fn(
                        y_test, y_pred,
                        model.predict_proba(X_fed_te)[:, 1])
                else:
                    c3 = metrics_fn(y_test, y_pred)
            else:
                c3 = {'error': 'no valid silo selections'}
            c3['silo_scores'] = convert_numpy(silo_scores)
            results['c3_federated_fs'] = convert_numpy(c3)
            primary = c3.get('mcc', c3.get('rmse', float('nan')))
            print(f"    {primary:.4f}")
        except Exception as e:
            results['c3_federated_fs'] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"    FAILED: {e}")
            traceback.print_exc()

        per_fold_results.append(results)

    # ── Aggregate across folds ──
    primary_key = 'mcc' if is_clf else 'rmse'

    def _agg_primary(cond_key):
        vals = []
        for fr in per_fold_results:
            r = fr.get(cond_key, {})
            if cond_key == 'c1_single_silo':
                if not r:
                    continue
                if is_clf:
                    best = max(r, key=lambda s: r[s].get('mcc', -999))
                else:
                    best = min(r, key=lambda s: r[s].get('rmse', 999))
                v = r[best].get(primary_key)
            else:
                v = r.get(primary_key)
            if v is not None and not np.isnan(v):
                vals.append(float(v))
        return vals

    aggregated = {}
    for ck in ['c1_single_silo', 'c2_kirby_hybrid', 'c3_federated_fs']:
        vals = _agg_primary(ck)
        aggregated[ck] = {
            primary_key: float(np.mean(vals)) if vals else float('nan'),
            f'{primary_key}_mean': float(np.mean(vals)) if vals else float('nan'),
            f'{primary_key}_std': float(np.std(vals)) if vals else float('nan'),
            f'{primary_key}_per_fold': vals,
            'n_folds_completed': len(vals),
            'per_fold': [fr.get(ck, {}) for fr in per_fold_results],
        }

    # Summary
    print(f"\n  {'Condition':<35} {'Primary (mean ± std)':>22}")
    for key, name in [
        ('c1_single_silo', 'C1: Best single silo (per-fold)'),
        ('c2_kirby_hybrid', 'C2: KIRBy hybrid (all)'),
        ('c3_federated_fs', 'C3: Federated FS'),
    ]:
        a = aggregated.get(key, {})
        m = a.get(f'{primary_key}_mean', float('nan'))
        s = a.get(f'{primary_key}_std', float('nan'))
        line = (f"{m:.4f} ± {s:.4f}" if not np.isnan(m) else 'N/A')
        print(f"  {name:<35} {line:>22}")

    return {
        'experiment': 'fed03_asymmetric_reps',
        'dataset': dataset_name,
        'config': {
            'silo_reps': {s: silo_reps[s] for s in silo_names},
            'n_folds': n_folds,
            'primary_metric': primary_key,
            'is_classification': is_clf,
        },
        **aggregated,
    }


# =============================================================================
# PART C: ANTIBODY FEDERATED (F6)
# =============================================================================

def _run_antibody_federated(data, config, output_dir, dataset_name,
                            fed_model='rf', fed_seed=0):
    """F6: Natural antibody silos — DDG exp, DDG synth, DMS, developability.

    Tests importance profile transfer across tasks with different endpoints.
    """
    from kirby.hybrid import (
        create_hybrid, apply_feature_selection, apply_augmentation_selection,
    )
    from kirby.find_best_hybrid import find_best_hybrid
    from experiments._shared import FULL_BUDGETS

    print("\n" + "=" * 70)
    print("FED03 PART C: ANTIBODY FEDERATED (F6)")
    print("=" * 70)

    # The calling dataset is one silo's endpoint (lead task).
    # We load the other silo datasets for cross-task transfer.
    SILO_DATASETS = {
        'ddg_exp': 'ddg_combined',
        'ddg_synth': 'graphinity_foldx',
        'dms': 'abagym_HER2',
        'develop': 'flab_thermo_tresanco',
    }

    # Load all silos
    silo_data = {}
    for silo_name, ds_name in SILO_DATASETS.items():
        print(f"\n  Loading silo '{silo_name}' ({ds_name})...")
        try:
            sd = load_data(ds_name, rep_preset='all')
            silo_data[silo_name] = sd
            n = len(sd['labels'])
            print(f"    {n} samples, {len(sd['reps'])} reps")
        except Exception as e:
            print(f"    FAILED: {e}")

    if len(silo_data) < 2:
        return {'error': f'Only {len(silo_data)} silos loaded, need >= 2'}

    # Discover optimal hybrid config per silo via find_best_hybrid
    silo_blueprints = {}
    for silo_name, sd in silo_data.items():
        if len(sd['reps']) >= 2:
            try:
                _metric, _metric_fn, _lower = get_domain_metric(sd)
                _silo_ds_name = SILO_DATASETS.get(silo_name, silo_name)
                s_search = find_best_hybrid(
                    sd['reps'], sd['labels'],
                    groups=sd.get('groups'),
                    budgets=FULL_BUDGETS,
                    dataset_name=_silo_ds_name, tier='general',
                    metric=_metric, task_type=sd['task_type'],
                    adaptive_pruning=0.25,
                    augmentation_strategy='none')
                s_bp_entry = s_search.best_by_model.get(
                    fed_to_kirby_model(fed_model))
                if s_bp_entry:
                    silo_blueprints[silo_name] = s_bp_entry['blueprint']
            except Exception:
                pass
    # Use first available blueprint as default
    if not silo_blueprints:
        warnings.warn(
            "Part C: silo_blueprints empty — find_best_hybrid produced no "
            "blueprint for any silo. Falling back to hardcoded defaults "
            "(performance_weighted/treeshap/match). Investigate the cache / "
            "find_best_hybrid failures before trusting these results.")
    _default_bp = next(iter(silo_blueprints.values()), None) if silo_blueprints else None
    _alloc = _default_bp.allocation_method if _default_bp else 'performance_weighted'
    _imp = _default_bp.importance_method if _default_bp else 'treeshap'
    _int_eval = _default_bp.internal_evaluator if _default_bp else 'match'

    # Find shared reps across silos
    all_rep_sets = [set(sd['reps'].keys()) for sd in silo_data.values()]
    shared_reps = sorted(set.intersection(*all_rep_sets))
    print(f"\n  Shared reps across {len(silo_data)} silos: "
          f"{len(shared_reps)}")
    if len(shared_reps) < 3:
        # Fall back to pairwise shared reps
        warnings.warn(
            f"Only {len(shared_reps)} shared reps. "
            f"Using per-silo available reps instead.")

    results = {}

    # ── C1: Each silo independently (5-fold scaffold-group CV) ──
    print("\n  C1: Independent silo baselines (5-fold)...")
    independent = {}
    silo_fis = {}  # keep a representative fi per silo for cross-task transfer (C3)
    N_FOLDS_C = smoke_n_folds(5)   # 2 under KIRBY_SMOKE_N, else 5
    for silo_name, sd in silo_data.items():
        reps_s = sd['reps']
        labels_s = sd['labels']
        groups_s = sd.get('groups')

        # Use shared reps only (for fair comparison)
        rep_dict = {r: reps_s[r] for r in shared_reps if r in reps_s}
        if len(rep_dict) < 2:
            rep_dict = reps_s  # fallback to all

        n_s = len(labels_s)
        if groups_s is None:
            raise RuntimeError(
                f"Part C silo '{silo_name}': groups required for 5-fold CV.")
        n_ug = len(np.unique(groups_s))
        if n_ug < N_FOLDS_C:
            raise RuntimeError(
                f"Part C silo '{silo_name}': need >= {N_FOLDS_C} unique "
                f"groups, got {n_ug}.")
        silo_folds = list(cv_split(
            groups_s, N_FOLDS_C, np.zeros(n_s), labels_s, seed=fed_seed))

        silo_task_type = sd.get('task_type', 'regression')
        silo_is_clf = (silo_task_type == 'classification')
        # The metric the budget split is scored with. create_hybrid refuses a
        # model without one rather than silently giving every representation an
        # equal share of the budget.
        _silo_metric_name, _silo_metric_fn, _silo_lower = get_domain_metric(sd)
        sbp = silo_blueprints.get(silo_name)
        s_alloc = sbp.allocation_method if sbp else _alloc
        s_imp = sbp.importance_method if sbp else _imp
        s_int_eval = sbp.internal_evaluator if sbp else _int_eval
        s_augs = sd.get('augmentations')
        s_aug_cond = 'uncapped' if s_augs else 'none'

        per_fold_metrics = []
        last_fi = None
        for fold_idx, (tr, te) in enumerate(silo_folds):
            rep_tr = {r: arr[tr] for r, arr in rep_dict.items()}
            rep_te = {r: arr[te] for r, arr in rep_dict.items()}
            y_tr = labels_s[tr]
            y_te = labels_s[te]
            try:
                tr_augs = ({k: v[tr] for k, v in s_augs.items()}
                           if s_augs else None)
                te_augs = ({k: v[te] for k, v in s_augs.items()}
                           if s_augs else None)
                _silo_model = (
                    make_fed_classifier(fed_model, seed=fed_seed)
                    if silo_is_clf
                    else make_fed_regressor(fed_model, seed=fed_seed))
                X_h, fi = create_hybrid(
                    rep_tr, y_tr,
                    allocation_method=s_alloc,
                    importance_method=s_imp,
                    internal_evaluator=s_int_eval,
                    model=_silo_model,
                    metric_fn=_silo_metric_fn, lower_is_better=_silo_lower,
                    augmentations=tr_augs,
                    augmentation_strategy=s_aug_cond)
                X_h_te = apply_feature_selection(rep_te, fi)
                _aug_info_c = (fi.get('augmentation_info')
                               if isinstance(fi, dict) else None)
                if _aug_info_c and te_augs:
                    _X_te_aug_c = apply_augmentation_selection(
                        te_augs, _aug_info_c)
                    if _X_te_aug_c is not None:
                        X_h_te = np.hstack([X_h_te, _X_te_aug_c])
                last_fi = fi

                if silo_is_clf:
                    model = fit_fed_model(
                        fed_model, X_h, y_tr, seed=fed_seed,
                        task_type='classification')
                    y_pred = model.predict(X_h_te)
                    y_prob = (model.predict_proba(X_h_te)
                              if hasattr(model, 'predict_proba') else None)
                    fm = compute_clf_metrics(y_te, y_pred, y_prob)
                else:
                    reg = fit_fed_model(
                        fed_model, X_h, y_tr, seed=fed_seed,
                        task_type='regression')
                    y_pred = reg.predict(X_h_te)
                    fm = compute_reg_metrics(y_te, y_pred)
                fm['n_features'] = int(X_h.shape[1])
                per_fold_metrics.append(fm)
            except Exception as e:
                per_fold_metrics.append({'error': str(e)})
                print(f"    {silo_name} fold {fold_idx + 1}: FAILED: {e}")

        # Aggregate per-silo
        good = [m for m in per_fold_metrics if 'error' not in m]
        if not good:
            independent[silo_name] = {'error': 'all folds failed',
                                      'per_fold': per_fold_metrics}
            print(f"    {silo_name}: ALL FOLDS FAILED")
            continue

        if silo_is_clf:
            mcc_vals = [m['mcc'] for m in good if 'mcc' in m]
            independent[silo_name] = {
                'mcc_mean': float(np.mean(mcc_vals)) if mcc_vals else float('nan'),
                'mcc_std': float(np.std(mcc_vals)) if mcc_vals else float('nan'),
                'mcc_per_fold': [float(v) for v in mcc_vals],
                'mcc': (float(np.mean(mcc_vals)) if mcc_vals
                        else float('nan')),  # back-compat for C3 lookups
                'per_fold': per_fold_metrics,
                'n_folds_completed': len(good),
                'task_type': silo_task_type,
            }
            print(f"    {silo_name}: MCC={independent[silo_name]['mcc_mean']:.4f}"
                  f" ± {independent[silo_name]['mcc_std']:.4f}")
        else:
            rmse_vals = [m['rmse'] for m in good if 'rmse' in m]
            rho_vals = [m.get('spearman_rho') for m in good
                        if m.get('spearman_rho') is not None]
            independent[silo_name] = {
                'rmse_mean': float(np.mean(rmse_vals)) if rmse_vals else float('nan'),
                'rmse_std': float(np.std(rmse_vals)) if rmse_vals else float('nan'),
                'rmse_per_fold': [float(v) for v in rmse_vals],
                'rmse': (float(np.mean(rmse_vals)) if rmse_vals
                         else float('nan')),  # back-compat for C3 lookups
                'spearman_rho_mean': (float(np.mean(rho_vals))
                                      if rho_vals else float('nan')),
                'per_fold': per_fold_metrics,
                'n_folds_completed': len(good),
                'task_type': silo_task_type,
            }
            print(f"    {silo_name}: RMSE="
                  f"{independent[silo_name]['rmse_mean']:.4f} ± "
                  f"{independent[silo_name]['rmse_std']:.4f}  rho="
                  f"{independent[silo_name]['spearman_rho_mean']:.4f}")

        if last_fi is not None:
            silo_fis[silo_name] = last_fi

    results['c1_independent'] = convert_numpy(independent)

    # ── C2: DDG pooling (if both ddg silos loaded) ──
    if 'ddg_exp' in silo_data and 'ddg_synth' in silo_data:
        print("\n  C2: Pooled DDG (experimental + synthetic)...")
        try:
            sd_exp = silo_data['ddg_exp']
            sd_synth = silo_data['ddg_synth']
            # Scored with the experimental silo's own metric; the pooled arm
            # predicts the same quantity.
            _pool_metric, _pool_metric_fn, _pool_lower = get_domain_metric(sd_exp)

            # Find shared reps between these two
            shared_ddg = set(sd_exp['reps'].keys()) & set(sd_synth['reps'].keys())
            if len(shared_ddg) < 2:
                raise ValueError(f"Only {len(shared_ddg)} shared DDG reps")

            # Pool labels and reps (both kcal/mol DDG)
            n_exp = len(sd_exp['labels'])
            n_synth = len(sd_synth['labels'])

            pooled_reps = {}
            for r in shared_ddg:
                r_exp = sd_exp['reps'][r]
                r_synth = sd_synth['reps'][r]
                if r_exp.shape[1] == r_synth.shape[1]:
                    pooled_reps[r] = np.vstack([r_exp, r_synth])
            pooled_labels = np.concatenate(
                [sd_exp['labels'], sd_synth['labels']])

            # Concatenate groups with offset to avoid cross-silo collisions
            g_exp = sd_exp.get('groups')
            g_synth = sd_synth.get('groups')
            if g_exp is not None and g_synth is not None:
                offset = int(g_exp.max()) + 1
                pooled_groups = np.concatenate([g_exp, g_synth + offset])
            elif g_exp is not None:
                offset = int(g_exp.max()) + 1
                pooled_groups = np.concatenate([
                    g_exp, np.arange(n_synth) + offset])
            else:
                pooled_groups = None

            # 5-fold scaffold-group CV
            if pooled_groups is None:
                raise RuntimeError(
                    "Part C C2 pooled DDG: pooled_groups is None.")
            n_ug = len(np.unique(pooled_groups))
            if n_ug < 5:
                raise RuntimeError(
                    f"Part C C2 pooled DDG: need >= 5 unique pooled groups, "
                    f"got {n_ug}.")
            pooled_folds = list(cv_split(
                pooled_groups, 5,
                np.zeros(len(pooled_labels)), pooled_labels, seed=fed_seed))

            per_fold_c2 = []
            for fold_idx, (tr, te) in enumerate(pooled_folds):
                try:
                    rep_tr = {r: arr[tr] for r, arr in pooled_reps.items()}
                    rep_te = {r: arr[te] for r, arr in pooled_reps.items()}

                    # Pooled DDG augmentations would need to be concatenated the
                    # same way labels/reps are pooled. Skip for safety — 'none'
                    # avoids cache-key collision.
                    X_h, fi = create_hybrid(
                        rep_tr, pooled_labels[tr],
                        allocation_method=_alloc,
                        importance_method=_imp,
                        internal_evaluator=_int_eval,
                        model=make_fed_regressor(fed_model, seed=fed_seed),
                        metric_fn=_pool_metric_fn, lower_is_better=_pool_lower,
                        augmentation_strategy='none')
                    X_h_te = apply_feature_selection(rep_te, fi)

                    reg = fit_fed_regressor(
                        fed_model, X_h, pooled_labels[tr], seed=fed_seed)
                    y_pred = reg.predict(X_h_te)
                    fm = compute_reg_metrics(pooled_labels[te], y_pred)
                    per_fold_c2.append(fm)
                    print(f"    Fold {fold_idx + 1}: RMSE={fm['rmse']:.4f}  "
                          f"rho={fm['spearman_rho']:.4f}")
                except Exception as e:
                    per_fold_c2.append({'error': str(e)})
                    print(f"    Fold {fold_idx + 1}: FAILED: {e}")
                    continue

            rmse_vals = [m['rmse'] for m in per_fold_c2 if 'rmse' in m]
            rho_vals = [m['spearman_rho'] for m in per_fold_c2
                        if 'spearman_rho' in m]
            c2 = {
                'rmse_mean': float(np.mean(rmse_vals)) if rmse_vals else float('nan'),
                'rmse_std': float(np.std(rmse_vals)) if rmse_vals else float('nan'),
                'rmse_per_fold': [float(v) for v in rmse_vals],
                'spearman_rho_mean': (float(np.mean(rho_vals))
                                      if rho_vals else float('nan')),
                'per_fold': per_fold_c2,
                'n_exp': n_exp,
                'n_synth': n_synth,
                'n_folds_completed': len(rmse_vals),
            }
            results['c2_pooled_ddg'] = convert_numpy(c2)
            print(f"    RMSE={c2['rmse_mean']:.4f} ± {c2['rmse_std']:.4f}  "
                  f"rho={c2['spearman_rho_mean']:.4f}  "
                  f"(pooled {n_exp}+{n_synth}={n_exp + n_synth})")
        except Exception as e:
            results['c2_pooled_ddg'] = {'error': str(e)}
            print(f"    FAILED: {e}")

    # ── C3: Cross-task importance transfer (5-fold per pair) ──
    print("\n  C3: Cross-task importance transfer (5-fold)...")
    transfer_results = {}

    # For each source→target pair, transfer importance profile
    for source_name, source_fi in silo_fis.items():
        for target_name, target_sd in silo_data.items():
            if source_name == target_name:
                continue

            key = f"{source_name}→{target_name}"
            try:
                target_reps = target_sd['reps']
                target_labels = target_sd['labels']
                target_groups = target_sd.get('groups')
                target_task_type = target_sd.get('task_type', 'regression')
                target_is_clf = (target_task_type == 'classification')
                # FED-32: importance_profile_transfer is purely index-based, so
                # it does not crash on a source/target task_type mismatch — but
                # transferring importance learned on (e.g.) a regression endpoint
                # to a classification endpoint changes the scientific meaning.
                # Record both task types so the mismatch is visible in the
                # output rather than silently averaged into a transfer score.
                source_task_type = silo_data.get(
                    source_name, {}).get('task_type', 'regression')

                rep_dict = {r: target_reps[r]
                            for r in shared_reps if r in target_reps}
                if len(rep_dict) < 2:
                    rep_dict = target_reps

                n_target = len(target_labels)
                if target_groups is None:
                    raise RuntimeError(
                        f"Part C C3 ({key}): target groups None — required.")
                n_ug = len(np.unique(target_groups))
                if n_ug < 5:
                    raise RuntimeError(
                        f"Part C C3 ({key}): need >= 5 target groups, "
                        f"got {n_ug}.")
                t_folds = list(cv_split(
                    target_groups, 5, np.zeros(n_target), target_labels,
                    seed=fed_seed))

                per_fold_t = []
                for fold_idx, (tr, te) in enumerate(t_folds):
                    rep_tr = {r: arr[tr] for r, arr in rep_dict.items()}
                    rep_te = {r: arr[te] for r, arr in rep_dict.items()}

                    X_target, transferred_fi, stats = importance_profile_transfer(
                        source_fi, rep_tr, target_labels[tr], budget=100)

                    if X_target is None:
                        per_fold_t.append({
                            'error': 'no features transferred',
                            'stats': stats})
                        continue

                    X_te = apply_feature_selection(rep_te, transferred_fi)

                    if target_is_clf:
                        cls_model = fit_fed_model(
                            fed_model, X_target, target_labels[tr],
                            seed=fed_seed, task_type='classification')
                        y_pred = cls_model.predict(X_te)
                        y_prob = (cls_model.predict_proba(X_te)[:, 1]
                                  if hasattr(cls_model, 'predict_proba')
                                  else None)
                        fm = compute_clf_metrics(
                            target_labels[te], y_pred, y_prob)
                    else:
                        reg = fit_fed_model(
                            fed_model, X_target, target_labels[tr],
                            seed=fed_seed, task_type='regression')
                        y_pred = reg.predict(X_te)
                        fm = compute_reg_metrics(target_labels[te], y_pred)
                    fm['transfer_stats'] = stats
                    per_fold_t.append(fm)

                good = [m for m in per_fold_t if 'error' not in m]
                if not good:
                    transfer_results[key] = {
                        'error': 'all folds failed', 'per_fold': per_fold_t}
                    print(f"    {key}: ALL FOLDS FAILED")
                    continue

                ind = independent.get(target_name, {})
                if target_is_clf:
                    mcc_vals = [m['mcc'] for m in good if 'mcc' in m]
                    agg = {
                        'mcc_mean': float(np.mean(mcc_vals)) if mcc_vals else float('nan'),
                        'mcc_std': float(np.std(mcc_vals)) if mcc_vals else float('nan'),
                        'mcc_per_fold': [float(v) for v in mcc_vals],
                        'mcc': (float(np.mean(mcc_vals)) if mcc_vals
                                else float('nan')),
                        'per_fold': per_fold_t,
                        'n_folds_completed': len(good),
                        'source_task_type': source_task_type,
                        'target_task_type': target_task_type,
                        'task_type_mismatch': source_task_type != target_task_type,
                    }
                    ind_mcc = ind.get('mcc_mean', ind.get('mcc', float('nan')))
                    if not np.isnan(ind_mcc) and not np.isnan(agg['mcc_mean']):
                        agg['mcc_improvement'] = float(
                            agg['mcc_mean'] - ind_mcc)
                    transfer_results[key] = agg
                    print(f"    {key}: MCC={agg['mcc_mean']:.4f} ± "
                          f"{agg['mcc_std']:.4f}  (vs ind {ind_mcc:.4f})")
                else:
                    rmse_vals = [m['rmse'] for m in good if 'rmse' in m]
                    agg = {
                        'rmse_mean': float(np.mean(rmse_vals)) if rmse_vals else float('nan'),
                        'rmse_std': float(np.std(rmse_vals)) if rmse_vals else float('nan'),
                        'rmse_per_fold': [float(v) for v in rmse_vals],
                        'rmse': (float(np.mean(rmse_vals)) if rmse_vals
                                 else float('nan')),
                        'per_fold': per_fold_t,
                        'n_folds_completed': len(good),
                        'source_task_type': source_task_type,
                        'target_task_type': target_task_type,
                        'task_type_mismatch': source_task_type != target_task_type,
                    }
                    ind_rmse = ind.get('rmse_mean', ind.get('rmse', float('nan')))
                    if (not np.isnan(ind_rmse)
                            and not np.isnan(agg['rmse_mean'])):
                        agg['rmse_improvement'] = float(
                            ind_rmse - agg['rmse_mean'])
                    transfer_results[key] = agg
                    print(f"    {key}: RMSE={agg['rmse_mean']:.4f} ± "
                          f"{agg['rmse_std']:.4f}  (vs ind {ind_rmse:.4f})")
            except Exception as e:
                transfer_results[key] = {'error': str(e)}
                print(f"    {key}: FAILED: {e}")

    results['c3_importance_transfer'] = convert_numpy(transfer_results)

    # Summary
    print(f"\n  {'Silo':<15} {'Independent':>12} {'Best Transfer':>14}")
    for silo_name in silo_data:
        ind = independent.get(silo_name, {})
        # Independent is now {rmse_mean,...} or {mcc_mean,...} keyed by task.
        is_clf_silo = (ind.get('task_type') == 'classification')
        if is_clf_silo:
            ind_primary = ind.get('mcc_mean', ind.get('mcc', float('nan')))
            best_transfer = float('nan')
            for key, tr in transfer_results.items():
                if not key.endswith(f'→{silo_name}'):
                    continue
                v = tr.get('mcc_mean', tr.get('mcc'))
                if v is None or np.isnan(v):
                    continue
                if np.isnan(best_transfer) or v > best_transfer:
                    best_transfer = v
        else:
            ind_primary = ind.get('rmse_mean', ind.get('rmse', float('nan')))
            best_transfer = float('nan')
            for key, tr in transfer_results.items():
                if not key.endswith(f'→{silo_name}'):
                    continue
                v = tr.get('rmse_mean', tr.get('rmse'))
                if v is None or np.isnan(v):
                    continue
                if np.isnan(best_transfer) or v < best_transfer:
                    best_transfer = v

        print(f"  {silo_name:<15} {ind_primary:>12.4f} {best_transfer:>14.4f}")

    return {
        'experiment': 'fed03_antibody_federated',
        'dataset': dataset_name,
        'config': {
            'silos': list(SILO_DATASETS.keys()),
            'silo_datasets': SILO_DATASETS,
            'shared_reps': list(shared_reps),
            'n_folds': 5,
        },
        **results,
    }


# =============================================================================
# DISPATCHER
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Fed03: Synthetic private datasets.

    Dispatches to sub-experiment based on dataset_name and kwargs['strategy']:
        strategy='source'     → Part A (source-based hERG, hERG only)
        strategy='asymmetric' → Part B (asymmetric rep silos, any molecular)
        strategy='antibody'   → Part C (antibody federated, loads own data)
        default               → auto-detect from dataset_name
    """
    strategy = kwargs.get('strategy', None)
    fed_model = config.get('fed_model', 'rf')
    fed_seed = config.get('fed_seed', 0)

    if strategy == 'source' or (
            strategy is None and dataset_name in ('herg_ki', 'herg', 'lhasa_herg')):
        return _run_herg_source_based(
            data, config, output_dir, dataset_name,
            fed_model=fed_model, fed_seed=fed_seed)
    elif strategy == 'asymmetric':
        return _run_asymmetric_reps(
            data, config, output_dir, dataset_name,
            fed_model=fed_model, fed_seed=fed_seed)
    elif strategy == 'antibody' or dataset_name in ANTIBODY_DATASETS:
        return _run_antibody_federated(
            data, config, output_dir, dataset_name,
            fed_model=fed_model, fed_seed=fed_seed)
    else:
        # Default for non-hERG molecular datasets: asymmetric reps
        return _run_asymmetric_reps(
            data, config, output_dir, dataset_name,
            fed_model=fed_model, fed_seed=fed_seed)


if __name__ == '__main__':
    import argparse as _ap
    parser = _ap.ArgumentParser(description='Fed03: Synthetic Private Datasets')
    parser.add_argument('--dataset', type=str, default='herg_ki',
                        help='Dataset name (herg_ki, logd, caco2, ddg_combined, etc.)')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['source', 'asymmetric', 'antibody'],
                        help='Sub-experiment: source (Part A), '
                             'asymmetric (Part B), antibody (Part C)')
    parser.add_argument('--output', type=str, default='tests/results/federated')
    parser.add_argument('--reps', type=str, default='all')
    args = parser.parse_args()

    # For antibody strategy, Part C loads its own data — pass empty placeholder
    if args.strategy == 'antibody':
        data = {'reps': {}, 'labels': np.array([]), 'domain': 'antibody'}
    else:
        data = load_data(args.dataset, args.reps)

    results = run(data, {}, args.output, args.dataset, strategy=args.strategy)
    save_experiment_results(
        results, args.output, 'fed03_synthetic_private', args.dataset)
