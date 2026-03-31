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
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split,
    get_precomputed_search,
)
from experiments.federated.federated_helpers import (
    compute_clf_metrics, compute_reg_metrics,
    predict_fluid_similarity_ensemble,
    predict_fluid_quality_ensemble,
    federated_feature_selection,
    importance_profile_transfer,
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

def _run_herg_source_based(data, config, output_dir, dataset_name):
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
    aug_cond = 'uncapped' if augs else 'none'
    try:
        _search = get_precomputed_search(
            reps_chembl, labels_chembl, data.get('groups'),
            budget=100, aug_condition=aug_cond,
            augmentations=augs,
            speed=config.get('search_speed', 'fast'),
            precomputed=data.get('precomputed_searches'))
        _bp = _search.best_overall['blueprint'] if _search.best_overall else None
    except Exception:
        _bp = None
    _alloc = _bp.allocation_method if _bp else 'performance_weighted'
    _imp = _bp.importance_method if _bp else 'treeshap'
    _int_eval = _bp.internal_evaluator if _bp else 'ridge'

    # Binarize ChEMBL labels
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
            k in deduped_chembl_set and k != 'UNKNOWN'
            for k in chembl_inchikeys], dtype=bool)
        reps_chembl = {r: arr[chembl_keep] for r, arr in reps_chembl.items()}
        print(f"  ChEMBL filtered: {n_chembl_orig} → {n_chembl_deduped} "
              f"(matched dedup)")

    n_chembl = n_chembl_deduped
    n_lhasa_tr = len(deduped_smiles['lhasa_train'])
    n_lhasa_te = len(deduped_smiles['lhasa_test'])

    # Get reps for Lhasa molecules
    lhasa_all_smiles = (deduped_smiles['lhasa_train'] +
                        deduped_smiles['lhasa_test'])
    lhasa_reps = get_molecular_representations(lhasa_all_smiles, 'all')

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

    # Train/test split: scaffold-group-aware for molecular datasets
    from find_best_hybrid_molecular import assign_scaffold_groups
    pooled_groups, _ = assign_scaffold_groups(all_smiles)
    train_idx, test_idx = next(cv_split(
        pooled_groups, 1, np.zeros(n_total), all_labels))

    y_train = all_labels[train_idx]
    y_test = all_labels[test_idx]
    train_silo = silo_assignments[train_idx]
    print(f"\nSplit: {len(train_idx)} train / {len(test_idx)} test")

    # Build silo indices relative to train set
    train_silo_indices = {}
    for i, sname in enumerate(silo_names):
        train_silo_indices[i + 1] = np.where(train_silo == i)[0]

    K = len(silo_names)

    # Build Morgan FPs for FLuID conditions
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
    train_fps = all_fps[train_idx]
    test_fps = all_fps[test_idx]

    # ── Conditions ────────────────────────────────────────────────────────
    results = {}

    # C1: Best single silo
    print("\n  C1: Single silo baselines...")
    single_silo = {}
    for i, sname in enumerate(silo_names):
        s_idx = train_silo_indices[i + 1]
        if len(s_idx) < 10:
            single_silo[sname] = {'error': 'too few samples'}
            continue
        clf = RandomForestClassifier(
            n_estimators=150, max_depth=15,
            class_weight='balanced', random_state=0)
        clf.fit(train_fps[s_idx], y_train[s_idx])
        pred = clf.predict(test_fps)
        pred_prob = clf.predict_proba(test_fps)[:, 1]
        single_silo[sname] = compute_clf_metrics(y_test, pred, pred_prob)
        print(f"    {sname}: MCC={single_silo[sname]['mcc']:.3f}")

    best_silo = max(
        [s for s in single_silo if 'mcc' in single_silo[s]],
        key=lambda s: single_silo[s]['mcc'])
    results['c1_single_silo'] = convert_numpy({
        'per_silo': single_silo,
        'best_silo': best_silo,
    })

    # C2: FLuID similarity ensemble
    print("  C2: FLuID similarity ensemble...")
    try:
        # Train teachers per silo
        teacher_models = [None]
        cluster_data_train = [None]
        for i, sname in enumerate(silo_names):
            s_idx = train_silo_indices[i + 1]
            X_s = train_fps[s_idx]
            y_s = y_train[s_idx]
            clf = RandomForestClassifier(
                n_estimators=150, max_depth=15,
                class_weight='balanced', random_state=0)
            clf.fit(X_s, y_s)

            # Create minimal DataFrame for similarity ensemble
            import pandas as pd_local
            df_s = pd_local.DataFrame({
                'FP': list(X_s),
                'CLASS': y_s,
            })
            teacher_models.append((clf, len(X_s), df_s))
            cluster_data_train.append(df_s)

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
                K, budget=100,
                allocation_method=_alloc, importance_method=_imp,
                internal_evaluator=_int_eval)

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
            clf = RandomForestClassifier(
                n_estimators=150, max_depth=15,
                class_weight='balanced', random_state=0)
            clf.fit(X_h_train, y_train)
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
        results['c4_kirby_federated'] = {'error': str(e)}
        print(f"    FAILED: {e}")

    # C5: Naive pool (unrealistic baseline)
    print("  C5: Naive pool...")
    clf_pool = RandomForestClassifier(
        n_estimators=150, max_depth=15,
        class_weight='balanced', random_state=0)
    clf_pool.fit(train_fps, y_train)
    pred_pool = clf_pool.predict(test_fps)
    pred_pool_prob = clf_pool.predict_proba(test_fps)[:, 1]
    c5 = compute_clf_metrics(y_test, pred_pool, pred_pool_prob)
    results['c5_naive_pool'] = convert_numpy(c5)
    print(f"    MCC={c5['mcc']:.3f}")

    # C6: KIRBy single-rep control (Morgan FPs only through hybrid pipeline)
    # Controls for feature-space confound: if KIRBy wins with 1 rep, fusion
    # strategy itself adds value, not just more features.
    print("  C6: KIRBy single-rep (ecfp4 only)...")
    try:
        from kirby.hybrid import create_hybrid, apply_feature_selection
        ecfp4_key = 'ecfp4' if 'ecfp4' in combined_reps else None
        if ecfp4_key is None:
            # Fall back to first available rep
            ecfp4_key = next(iter(combined_reps))
        single_reps = {ecfp4_key: combined_reps[ecfp4_key]}
        train_single = {ecfp4_key: single_reps[ecfp4_key][train_idx]}
        test_single = {ecfp4_key: single_reps[ecfp4_key][test_idx]}
        X_h_c6, fi_c6 = create_hybrid(
            train_single, y_train,
            allocation_method=_alloc,
            importance_method=_imp, budget=100,
            internal_evaluator=_int_eval)
        X_h_c6_test = apply_feature_selection(test_single, fi_c6)
        clf_c6 = RandomForestClassifier(
            n_estimators=150, max_depth=15,
            class_weight='balanced', random_state=0)
        clf_c6.fit(X_h_c6, y_train)
        pred_c6 = clf_c6.predict(X_h_c6_test)
        prob_c6 = clf_c6.predict_proba(X_h_c6_test)[:, 1]
        c6 = compute_clf_metrics(y_test, pred_c6, prob_c6)
        c6['rep_used'] = ecfp4_key
        results['c6_kirby_single_rep'] = convert_numpy(c6)
        print(f"    MCC={c6['mcc']:.3f}")
    except Exception as e:
        results['c6_kirby_single_rep'] = {'error': str(e)}
        print(f"    FAILED: {e}")

    # C7: FLuID multi-rep ensemble (one teacher per rep, quality-weighted)
    # Controls for feature-space confound: gives FLuID the same feature
    # diversity as KIRBy. If KIRBy C4 still wins, it's the fusion strategy.
    print("  C7: FLuID multi-rep ensemble...")
    try:
        rep_teachers = []
        rep_names_list = sorted(combined_reps.keys())
        for rname in rep_names_list:
            X_tr = combined_reps[rname][train_idx]
            X_te = combined_reps[rname][test_idx]
            clf_r = RandomForestClassifier(
                n_estimators=150, max_depth=15,
                class_weight='balanced', random_state=0)
            clf_r.fit(X_tr, y_train)
            rep_teachers.append((clf_r, rname, X_tr, X_te))

        # Quality-weighted ensemble: weight by cross-val MCC per rep-teacher
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import matthews_corrcoef
        weights = []
        for clf_r, rname, X_tr, _ in rep_teachers:
            try:
                cv_pred = cross_val_predict(
                    RandomForestClassifier(
                        n_estimators=100, max_depth=15,
                        class_weight='balanced', random_state=0),
                    X_tr, y_train, cv=3)
                w = max(matthews_corrcoef(y_train, cv_pred), 0.01)
            except Exception:
                w = 0.01
            weights.append(w)
        weights = np.array(weights)
        weights /= weights.sum()

        # Weighted vote
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

    # Summary
    print(f"\n  {'Condition':<30} {'MCC':>8}")
    for key, name in [
        ('c1_single_silo', f'C1: Best silo ({best_silo})'),
        ('c2_fluid_similarity', 'C2: FLuID similarity'),
        ('c3_fluid_quality', 'C3: FLuID quality'),
        ('c4_kirby_federated', 'C4: KIRBy federated'),
        ('c5_naive_pool', 'C5: Naive pool'),
        ('c6_kirby_single_rep', 'C6: KIRBy single-rep'),
        ('c7_fluid_multi_rep', 'C7: FLuID multi-rep'),
    ]:
        r = results.get(key, {})
        if key == 'c1_single_silo':
            r = r.get('per_silo', {}).get(best_silo, {})
        mcc = r.get('mcc', float('nan'))
        print(f"  {name:<30} {mcc:>8.3f}" if not np.isnan(mcc)
              else f"  {name:<30} {'N/A':>8}")

    return {
        'experiment': 'fed03_source_based_herg',
        'dataset': dataset_name,
        'config': {
            'silos': silo_names,
            'n_total': n_total,
            'dedup_stats': dedup_stats,
            'pki_threshold': PKI_THRESHOLD,
        },
        **results,
    }


# =============================================================================
# PART B: ASYMMETRIC REP SILOS
# =============================================================================

def _run_asymmetric_reps(data, config, output_dir, dataset_name):
    """Strategy 4: Each org contributes different representation types.

    Silo A (structural): ecfp4, graph_kernel, atom_pair
    Silo B (physicochemical): mordred, pdv
    Silo C (learned): molformer, chemberta
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection

    print("\n" + "=" * 70)
    print("FED03 PART B: ASYMMETRIC REP SILOS (Strategy 4)")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')

    # Get optimal hybrid config from precomputed search
    augs = data.get('augmentations')
    aug_cond = 'uncapped' if augs else 'none'
    try:
        _search = get_precomputed_search(
            reps, labels, groups,
            budget=100, aug_condition=aug_cond,
            augmentations=augs,
            speed=config.get('search_speed', 'fast'),
            precomputed=data.get('precomputed_searches'))
        _bp = _search.best_overall['blueprint'] if _search.best_overall else None
    except Exception:
        _bp = None
    _alloc = _bp.allocation_method if _bp else 'performance_weighted'
    _imp = _bp.importance_method if _bp else 'treeshap'
    _int_eval = _bp.internal_evaluator if _bp else 'ridge'

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

    # 80/20 scaffold-group-aware split
    n_samples = len(labels)
    train_idx, test_idx = next(cv_split(
        groups, 1, np.zeros(n_samples), labels))

    y_train = labels[train_idx]
    y_test = labels[test_idx]
    is_clf = len(np.unique(labels)) <= 10
    metrics_fn = compute_clf_metrics if is_clf else compute_reg_metrics
    print(f"  Split: {len(train_idx)} train / {len(test_idx)} test "
          f"({'clf' if is_clf else 'reg'})")

    train_reps = {r: arr[train_idx] for r, arr in reps.items()}
    test_reps = {r: arr[test_idx] for r, arr in reps.items()}

    results = {}

    # C1: Best single silo
    print("\n  C1: Single silo baselines...")
    single_silo = {}
    for sname in silo_names:
        rep_list = silo_reps[sname]
        X_tr = np.hstack([train_reps[r] for r in rep_list])
        X_te = np.hstack([test_reps[r] for r in rep_list])

        if is_clf:
            model = RandomForestClassifier(
                n_estimators=150, max_depth=15,
                class_weight='balanced', random_state=0)
        else:
            model = RandomForestRegressor(
                n_estimators=150, max_depth=15, random_state=0)

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        if is_clf:
            y_prob = model.predict_proba(X_te)[:, 1]
            single_silo[sname] = metrics_fn(y_test, y_pred, y_prob)
        else:
            single_silo[sname] = metrics_fn(y_test, y_pred)
        single_silo[sname]['n_features'] = int(X_tr.shape[1])

        primary = single_silo[sname].get('mcc', single_silo[sname].get('rmse'))
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

        X_h_tr, fi = create_hybrid(
            all_silo_reps_tr, y_train,
            allocation_method=_alloc,
            importance_method=_imp, budget=100,
            internal_evaluator=_int_eval)
        X_h_te = apply_feature_selection(all_silo_reps_te, fi)

        if is_clf:
            model = RandomForestClassifier(
                n_estimators=150, max_depth=15,
                class_weight='balanced', random_state=0)
        else:
            model = RandomForestRegressor(
                n_estimators=150, max_depth=15, random_state=0)
        model.fit(X_h_tr, y_train)
        y_pred = model.predict(X_h_te)
        if is_clf:
            c2 = metrics_fn(y_test, y_pred, model.predict_proba(X_h_te)[:, 1])
        else:
            c2 = metrics_fn(y_test, y_pred)
        c2['allocation'] = convert_numpy(fi.get('allocation', {}))
        c2['n_features'] = int(X_h_tr.shape[1])
        results['c2_kirby_hybrid'] = convert_numpy(c2)
        primary = c2.get('mcc', c2.get('rmse'))
        print(f"    {primary:.4f}  ({c2['n_features']}D)")
    except Exception as e:
        results['c2_kirby_hybrid'] = {'error': str(e)}
        print(f"    FAILED: {e}")

    # C3: Federated feature selection (privacy-preserving)
    print("  C3: Federated FS (per-silo importance)...")
    try:
        # Each silo gets ALL training samples but only its OWN reps
        silo_indices_all = {}  # all silos see all molecules
        for i, sname in enumerate(silo_names, 1):
            silo_indices_all[i] = np.arange(len(y_train))

        # But restrict rep_dict per silo during federated FS
        # We run create_hybrid per silo with only that silo's reps
        from kirby.hybrid import create_hybrid as _ch
        from sklearn.linear_model import RidgeCV, RidgeClassifierCV
        from sklearn.model_selection import ShuffleSplit as _SS

        silo_fis = {}
        silo_scores = {}
        for i, sname in enumerate(silo_names, 1):
            rep_list = silo_reps[sname]
            silo_rep_tr = {r: train_reps[r] for r in rep_list}
            try:
                X_s, fi_s = _ch(
                    silo_rep_tr, y_train,
                    allocation_method=_alloc,
                    importance_method=_imp, budget=50,
                    internal_evaluator=_int_eval)
                silo_fis[i] = fi_s
                # Quick validation score
                ss = _SS(n_splits=1, test_size=0.2, random_state=42 + i)
                tr, va = next(ss.split(X_s))
                evaluator = (RidgeClassifierCV() if is_clf else RidgeCV())
                evaluator.fit(X_s[tr], y_train[tr])
                silo_scores[i] = max(0.0, float(
                    evaluator.score(X_s[va], y_train[va])))
            except Exception:
                silo_fis[i] = None
                silo_scores[i] = 0.0

        # Apply each silo's feature selection to test data
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
                h_parts_tr.append(apply_feature_selection(silo_rep_tr, fi))
                h_parts_te.append(apply_feature_selection(silo_rep_te, fi))
            except Exception:
                continue

        if h_parts_tr:
            X_fed_tr = np.hstack(h_parts_tr)
            X_fed_te = np.hstack(h_parts_te)
            if is_clf:
                model = RandomForestClassifier(
                    n_estimators=150, max_depth=15,
                    class_weight='balanced', random_state=0)
            else:
                model = RandomForestRegressor(
                    n_estimators=150, max_depth=15, random_state=0)
            model.fit(X_fed_tr, y_train)
            y_pred = model.predict(X_fed_te)
            if is_clf:
                c3 = metrics_fn(y_test, y_pred,
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
        results['c3_federated_fs'] = {'error': str(e)}
        print(f"    FAILED: {e}")

    # Summary
    print(f"\n  {'Condition':<35} {'Primary':>10}")
    for key, name in [
        ('c1_single_silo', 'C1: Best single silo'),
        ('c2_kirby_hybrid', 'C2: KIRBy hybrid (all)'),
        ('c3_federated_fs', 'C3: Federated FS'),
    ]:
        r = results.get(key, {})
        if key == 'c1_single_silo':
            best = min(r, key=lambda s: r[s].get('rmse', 999)) if not is_clf \
                else max(r, key=lambda s: r[s].get('mcc', -999))
            r = r[best]
        val = r.get('mcc', r.get('rmse', float('nan')))
        print(f"  {name:<35} {val:>10.4f}")

    return {
        'experiment': 'fed03_asymmetric_reps',
        'dataset': dataset_name,
        'config': {
            'silo_reps': {s: silo_reps[s] for s in silo_names},
            'is_classification': is_clf,
        },
        **results,
    }


# =============================================================================
# PART C: ANTIBODY FEDERATED (F6)
# =============================================================================

def _run_antibody_federated(data, config, output_dir, dataset_name):
    """F6: Natural antibody silos — DDG exp, DDG synth, DMS, developability.

    Tests importance profile transfer across tasks with different endpoints.
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection
    from kirby.find_best_hybrid import find_best_hybrid

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
                s_search = find_best_hybrid(
                    sd['reps'], sd['labels'],
                    groups=sd.get('groups'),
                    speed='fast', cache=True, verbose=False)
                if s_search.best_overall:
                    silo_blueprints[silo_name] = s_search.best_overall['blueprint']
            except Exception:
                pass
    # Use first available blueprint as default
    _default_bp = next(iter(silo_blueprints.values()), None) if silo_blueprints else None
    _alloc = _default_bp.allocation_method if _default_bp else 'performance_weighted'
    _imp = _default_bp.importance_method if _default_bp else 'treeshap'
    _int_eval = _default_bp.internal_evaluator if _default_bp else 'ridge'

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

    # ── C1: Each silo independently ──
    print("\n  C1: Independent silo baselines...")
    independent = {}
    silo_fis = {}
    for silo_name, sd in silo_data.items():
        reps_s = sd['reps']
        labels_s = sd['labels']
        groups_s = sd.get('groups')

        # Use shared reps only (for fair comparison)
        rep_dict = {r: reps_s[r] for r in shared_reps if r in reps_s}
        if len(rep_dict) < 2:
            rep_dict = reps_s  # fallback to all

        # Split (scaffold-group-aware via cv_split)
        n_s = len(labels_s)
        tr, te = next(cv_split(groups_s, 1, np.zeros(n_s), labels_s))

        rep_tr = {r: arr[tr] for r, arr in rep_dict.items()}
        rep_te = {r: arr[te] for r, arr in rep_dict.items()}
        y_tr = labels_s[tr]
        y_te = labels_s[te]

        try:
            sbp = silo_blueprints.get(silo_name)
            s_alloc = sbp.allocation_method if sbp else _alloc
            s_imp = sbp.importance_method if sbp else _imp
            s_int_eval = sbp.internal_evaluator if sbp else _int_eval
            X_h, fi = create_hybrid(
                rep_tr, y_tr,
                allocation_method=s_alloc,
                importance_method=s_imp, budget=100,
                internal_evaluator=s_int_eval)
            X_h_te = apply_feature_selection(rep_te, fi)
            silo_fis[silo_name] = fi

            reg = RandomForestRegressor(
                n_estimators=150, max_depth=15, random_state=0)
            reg.fit(X_h, y_tr)
            y_pred = reg.predict(X_h_te)
            metrics = compute_reg_metrics(y_te, y_pred)
            metrics['n_features'] = int(X_h.shape[1])
            metrics['allocation'] = convert_numpy(fi.get('allocation', {}))
            independent[silo_name] = metrics
            print(f"    {silo_name}: RMSE={metrics['rmse']:.4f}  "
                  f"rho={metrics['spearman_rho']:.4f}")
        except Exception as e:
            independent[silo_name] = {'error': str(e)}
            print(f"    {silo_name}: FAILED: {e}")

    results['c1_independent'] = convert_numpy(independent)

    # ── C2: DDG pooling (if both ddg silos loaded) ──
    if 'ddg_exp' in silo_data and 'ddg_synth' in silo_data:
        print("\n  C2: Pooled DDG (experimental + synthetic)...")
        try:
            sd_exp = silo_data['ddg_exp']
            sd_synth = silo_data['ddg_synth']

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

            # Split (scaffold-group-aware)
            tr, te = next(cv_split(
                pooled_groups, 1,
                np.zeros(len(pooled_labels)), pooled_labels))

            rep_tr = {r: arr[tr] for r, arr in pooled_reps.items()}
            rep_te = {r: arr[te] for r, arr in pooled_reps.items()}

            X_h, fi = create_hybrid(
                rep_tr, pooled_labels[tr],
                allocation_method=_alloc,
                importance_method=_imp, budget=100,
                internal_evaluator=_int_eval)
            X_h_te = apply_feature_selection(rep_te, fi)

            reg = RandomForestRegressor(
                n_estimators=150, max_depth=15, random_state=0)
            reg.fit(X_h, pooled_labels[tr])
            y_pred = reg.predict(X_h_te)
            c2 = compute_reg_metrics(pooled_labels[te], y_pred)
            c2['n_exp'] = n_exp
            c2['n_synth'] = n_synth
            results['c2_pooled_ddg'] = convert_numpy(c2)
            print(f"    RMSE={c2['rmse']:.4f}  rho={c2['spearman_rho']:.4f}  "
                  f"(pooled {n_exp}+{n_synth}={n_exp + n_synth})")
        except Exception as e:
            results['c2_pooled_ddg'] = {'error': str(e)}
            print(f"    FAILED: {e}")

    # ── C3: Cross-task importance transfer ──
    print("\n  C3: Cross-task importance transfer...")
    transfer_results = {}

    # For each source→target pair, transfer importance profile
    for source_name, source_fi in silo_fis.items():
        for target_name, target_sd in silo_data.items():
            if source_name == target_name:
                continue

            key = f"{source_name}→{target_name}"
            try:
                # Get target's test data
                target_reps = target_sd['reps']
                target_labels = target_sd['labels']
                target_groups = target_sd.get('groups')

                rep_dict = {r: target_reps[r]
                            for r in shared_reps if r in target_reps}
                if len(rep_dict) < 2:
                    rep_dict = target_reps

                n_target = len(target_labels)
                tr, te = next(cv_split(
                    target_groups, 1, np.zeros(n_target), target_labels))

                rep_tr = {r: arr[tr] for r, arr in rep_dict.items()}
                rep_te = {r: arr[te] for r, arr in rep_dict.items()}

                X_target, transferred_fi, stats = importance_profile_transfer(
                    source_fi, rep_tr, target_labels[tr], budget=100)

                if X_target is None:
                    transfer_results[key] = {
                        'error': 'no features transferred',
                        'stats': stats}
                    continue

                X_te = apply_feature_selection(rep_te, transferred_fi)

                reg = RandomForestRegressor(
                    n_estimators=150, max_depth=15, random_state=0)
                reg.fit(X_target, target_labels[tr])
                y_pred = reg.predict(X_te)

                metrics = compute_reg_metrics(target_labels[te], y_pred)
                metrics['transfer_stats'] = stats

                # Compare to independent baseline
                ind = independent.get(target_name, {})
                ind_rmse = ind.get('rmse', float('nan'))
                if not np.isnan(ind_rmse):
                    metrics['rmse_improvement'] = float(
                        ind_rmse - metrics['rmse'])

                transfer_results[key] = metrics
                print(f"    {key}: RMSE={metrics['rmse']:.4f}  "
                      f"(vs independent {ind_rmse:.4f})")
            except Exception as e:
                transfer_results[key] = {'error': str(e)}
                print(f"    {key}: FAILED: {e}")

    results['c3_importance_transfer'] = convert_numpy(transfer_results)

    # Summary
    print(f"\n  {'Silo':<15} {'Independent':>12} {'Best Transfer':>14}")
    for silo_name in silo_data:
        ind = independent.get(silo_name, {})
        ind_rmse = ind.get('rmse', float('nan'))

        best_transfer_rmse = float('nan')
        for key, tr in transfer_results.items():
            if key.endswith(f'→{silo_name}') and 'rmse' in tr:
                if np.isnan(best_transfer_rmse) or tr['rmse'] < best_transfer_rmse:
                    best_transfer_rmse = tr['rmse']

        print(f"  {silo_name:<15} {ind_rmse:>12.4f} {best_transfer_rmse:>14.4f}")

    return {
        'experiment': 'fed03_antibody_federated',
        'dataset': dataset_name,
        'config': {
            'silos': list(SILO_DATASETS.keys()),
            'silo_datasets': SILO_DATASETS,
            'shared_reps': list(shared_reps),
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

    if strategy == 'source' or (strategy is None and dataset_name == 'herg'):
        return _run_herg_source_based(data, config, output_dir, dataset_name)
    elif strategy == 'asymmetric':
        return _run_asymmetric_reps(data, config, output_dir, dataset_name)
    elif strategy == 'antibody' or dataset_name in ANTIBODY_DATASETS:
        return _run_antibody_federated(data, config, output_dir, dataset_name)
    else:
        # Default for non-hERG molecular datasets: asymmetric reps
        return _run_asymmetric_reps(data, config, output_dir, dataset_name)


if __name__ == '__main__':
    import argparse as _ap
    parser = _ap.ArgumentParser(description='Fed03: Synthetic Private Datasets')
    parser.add_argument('--dataset', type=str, default='herg',
                        help='Dataset name (herg, logd, caco2, ddg_combined, etc.)')
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
