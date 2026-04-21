#!/usr/bin/env python3
"""Fed05: Antibody Federated Extension.

Adapts FLuID's federated framework from small molecules to antibody sequences.
Uses ESM2 embedding clustering instead of Morgan FP clustering. Tests the same
core questions as fed01/02 in the antibody domain: does KIRBy's feature-level
fusion outperform prediction-level ensemble across heterogeneous antibody silos?

3 natural silos:
    A: Experimental DDG (ddg_combined, ~4500 mutations, kcal/mol)
    B: DMS binding (abagym_HER2, ~146 mutations, normalized score)
    C: Developability (flab_thermo_tresanco, Tm °C)

8 conditions:
    C1: Independent baselines per silo
    C2: Cross-silo importance transfer
    C3: Within-silo ESM2 clustering → federated feature selection
    C4: Noise heterogeneity (Gaussian noise on one silo)
    C5: Pooled binding (DDG + DMS, z-score normalized)
    C6: Rep diversity per silo (different rep subsets)
    C7: FLuID quality-weighted ensemble (inverse-RMSE)
    C8: FLuID similarity-weighted ensemble (ESM2 distance)

Usage:
    python fed05_antibody_federated.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import zscore

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy, cv_split,
    cap_dimensions, get_domain_metric, FULL_BUDGETS,
)
from experiments.federated.federated_helpers import (
    compute_reg_metrics, inject_noise_per_silo,
    federated_feature_selection, importance_profile_transfer,
    extract_importance_profile, compare_importance_profiles,
    cluster_antibody_silos,
    predict_fluid_quality_ensemble_regression,
)
from kirby.hybrid import create_hybrid, apply_feature_selection
from kirby.find_best_hybrid import find_best_hybrid


# ── Constants ────────────────────────────────────────────────────────────────

SILO_DATASETS = {
    'ddg_exp': 'ddg_combined',
    'dms': 'abagym_HER2',
    'develop': 'flab_thermo_tresanco',
}

# Rep subsets for Condition 6 (rep diversity per silo)
REP_DIVERSITY_SILOS = {
    'structural': ['aac', 'saprot', 'esm2'],
    'sequence_lm': ['ablang2', 'antiberty', 'prot_t5'],
    'cdr_focused': ['prot_t5_cdr', 'esm2_midlayers', 'igt5'],
}

K_INTRA = 3   # sub-clusters per silo for Condition 3
NOISE_SIGMA = 0.2  # noise level for Condition 4


def _load_silo(silo_name, ds_name):
    """Load one antibody silo dataset. Returns data dict or None."""
    try:
        sd = load_data(ds_name, rep_preset='all')
        n = len(sd['labels'])
        print(f"    {silo_name} ({ds_name}): {n} samples, "
              f"{len(sd['reps'])} reps")
        return sd
    except Exception as e:
        print(f"    {silo_name} ({ds_name}): FAILED — {e}")
        return None


def _split_silo(sd, seed=42):
    """80/20 group-aware split for one silo. Returns (train_idx, test_idx)."""
    groups = sd.get('groups')
    n = len(sd['labels'])
    return next(cv_split(groups, 1, np.zeros(n), sd['labels'], seed=seed))


def _silo_baseline(sd, shared_reps, seed=42,
                   alloc='performance_weighted', imp='treeshap',
                   int_eval='match'):
    """Train create_hybrid on one silo and evaluate. Returns metrics + fi."""
    reps_s = {r: sd['reps'][r] for r in shared_reps if r in sd['reps']}
    if len(reps_s) < 2:
        reps_s = sd['reps']

    tr, te = _split_silo(sd, seed)
    rep_tr = {r: arr[tr] for r, arr in reps_s.items()}
    rep_te = {r: arr[te] for r, arr in reps_s.items()}
    y_tr, y_te = sd['labels'][tr], sd['labels'][te]

    X_h, fi = create_hybrid(
        rep_tr, y_tr,
        allocation_method=alloc,
        importance_method=imp,
        internal_evaluator=int_eval)
    X_h_te = apply_feature_selection(rep_te, fi)

    reg = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=0)
    reg.fit(X_h, y_tr)
    y_pred = reg.predict(X_h_te)
    metrics = compute_reg_metrics(y_te, y_pred)
    metrics['n_features'] = int(X_h.shape[1])
    return metrics, fi, (tr, te)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Fed05: Antibody federated extension."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED05: ANTIBODY FEDERATED EXTENSION")
    print("=" * 70)

    # ── Phase 1: Load silos ──────────────────────────────────────────────
    print("\n--- Phase 1: Load silos ---")
    silo_data = {}
    for silo_name, ds_name in SILO_DATASETS.items():
        sd = _load_silo(silo_name, ds_name)
        if sd is not None:
            silo_data[silo_name] = sd

    if len(silo_data) < 2:
        return {'error': f'Only {len(silo_data)} silos loaded, need >= 2'}

    # Discover optimal hybrid config per silo via find_best_hybrid
    silo_blueprints = {}
    for sname, sd in silo_data.items():
        if len(sd['reps']) >= 2:
            try:
                _metric = get_domain_metric(sd)[0]
                _silo_ds_name = SILO_DATASETS.get(sname, sname)
                s_search = find_best_hybrid(
                    sd['reps'], sd['labels'],
                    groups=sd.get('groups'),
                    budgets=FULL_BUDGETS,
                    speed='full', cache=True, verbose=False,
                    dataset_name=_silo_ds_name, tier='general',
                    metric=_metric, adaptive_pruning=0.25,
                    augmentation_strategy='none')
                s_bp_entry = s_search.best_by_model.get('rf')
                if s_bp_entry:
                    silo_blueprints[sname] = s_bp_entry['blueprint']
            except Exception:
                pass
    _default_bp = next(iter(silo_blueprints.values()), None) if silo_blueprints else None
    _alloc = _default_bp.allocation_method if _default_bp else 'performance_weighted'
    _imp = _default_bp.importance_method if _default_bp else 'treeshap'
    _int_eval = _default_bp.internal_evaluator if _default_bp else 'match'

    # Find shared reps
    all_rep_sets = [set(sd['reps'].keys()) for sd in silo_data.values()]
    shared_reps = sorted(set.intersection(*all_rep_sets))
    print(f"\n  Shared reps ({len(shared_reps)}): {', '.join(shared_reps)}")

    if len(shared_reps) < 3:
        warnings.warn("Few shared reps — using per-silo available reps")

    results = {}
    silo_names = sorted(silo_data.keys())

    # ── Phase 2: C1 — Independent baselines ──────────────────────────────
    print("\n--- Phase 2: C1 — Independent baselines ---")
    independent = {}
    silo_fis = {}
    silo_splits = {}

    for sname in silo_names:
        sd = silo_data[sname]
        print(f"  {sname}...", end=' ')
        try:
            sbp = silo_blueprints.get(sname)
            s_alloc = sbp.allocation_method if sbp else _alloc
            s_imp = sbp.importance_method if sbp else _imp
            s_int_eval = sbp.internal_evaluator if sbp else _int_eval
            metrics, fi, split = _silo_baseline(
                sd, shared_reps,
                alloc=s_alloc, imp=s_imp, int_eval=s_int_eval)
            independent[sname] = metrics
            silo_fis[sname] = fi
            silo_splits[sname] = split
            print(f"RMSE={metrics['rmse']:.4f}  "
                  f"rho={metrics['spearman_rho']:.4f}")
        except Exception as e:
            independent[sname] = {'error': str(e)}
            print(f"FAILED: {e}")

    results['c1_independent'] = convert_numpy(independent)

    # ── Phase 3: C2 — Cross-silo importance transfer ─────────────────────
    print("\n--- Phase 3: C2 — Cross-silo importance transfer ---")
    transfer_results = {}

    for source_name in silo_names:
        source_fi = silo_fis.get(source_name)
        if source_fi is None:
            continue
        for target_name in silo_names:
            if source_name == target_name:
                continue
            key = f"{source_name}→{target_name}"
            print(f"  {key}...", end=' ')
            try:
                target_sd = silo_data[target_name]
                target_reps = {r: target_sd['reps'][r]
                               for r in shared_reps if r in target_sd['reps']}
                if len(target_reps) < 2:
                    target_reps = target_sd['reps']

                tr, te = silo_splits.get(target_name,
                                         _split_silo(target_sd))
                rep_tr = {r: arr[tr] for r, arr in target_reps.items()}
                rep_te = {r: arr[te] for r, arr in target_reps.items()}
                y_tr = target_sd['labels'][tr]
                y_te = target_sd['labels'][te]

                X_target, transferred_fi, stats = importance_profile_transfer(
                    source_fi, rep_tr, y_tr, budget=100)

                if X_target is None:
                    transfer_results[key] = {'error': 'no features transferred'}
                    print("no features")
                    continue

                X_te = apply_feature_selection(rep_te, transferred_fi)
                reg = RandomForestRegressor(
                    n_estimators=150, max_depth=15, random_state=0)
                reg.fit(X_target, y_tr)
                y_pred = reg.predict(X_te)
                metrics = compute_reg_metrics(y_te, y_pred)
                metrics['transfer_stats'] = stats

                ind_rmse = independent.get(target_name, {}).get('rmse', float('nan'))
                if not np.isnan(ind_rmse):
                    metrics['rmse_improvement'] = float(ind_rmse - metrics['rmse'])

                transfer_results[key] = metrics
                print(f"RMSE={metrics['rmse']:.4f}  "
                      f"(vs ind {ind_rmse:.4f})")
            except Exception as e:
                transfer_results[key] = {'error': str(e)}
                print(f"FAILED: {e}")

    results['c2_importance_transfer'] = convert_numpy(transfer_results)

    # ── Phase 4: C3 — Within-silo ESM2 clustering ────────────────────────
    print("\n--- Phase 4: C3 — Within-silo ESM2 clustering ---")
    clustering_results = {}

    for sname in silo_names:
        sd = silo_data[sname]
        silo_reps = {r: sd['reps'][r] for r in shared_reps if r in sd['reps']}
        if len(silo_reps) < 2:
            silo_reps = sd['reps']

        print(f"  {sname}: clustering...", end=' ')
        try:
            tr, te = silo_splits.get(sname, _split_silo(sd))
            y_tr = sd['labels'][tr]
            rep_tr = {r: arr[tr] for r, arr in silo_reps.items()}
            rep_te = {r: arr[te] for r, arr in silo_reps.items()}

            # Cluster on TRAIN data only (no test leakage into clusters)
            sub_silo_indices = cluster_antibody_silos(
                rep_tr, k=K_INTRA, seed=42)

            # sub_silo_indices are already train-relative (0..len(tr)-1)
            train_sub_indices = {}
            for c_id, idx in sub_silo_indices.items():
                if len(idx) >= 10:
                    train_sub_indices[c_id] = idx

            if len(train_sub_indices) < 2:
                clustering_results[sname] = {
                    'error': 'too few viable sub-clusters'}
                print("too few clusters")
                continue

            # Per-cluster importance profiles
            # Use lower budget for small clusters to avoid overfitting
            profiles = {}
            for c_id, c_idx in train_sub_indices.items():
                c_reps = {r: arr[c_idx] for r, arr in rep_tr.items()}
                c_labels = y_tr[c_idx]
                c_budget = min(30, len(c_idx) // 3)  # conservative for small clusters
                if c_budget < 5:
                    continue
                try:
                    _, fi_c = create_hybrid(
                        c_reps, c_labels,
                        allocation_method=_alloc,
                        importance_method=_imp,
                        internal_evaluator=_int_eval)
                    profiles[c_id] = extract_importance_profile(fi_c)
                except Exception:
                    pass

            mean_sp, pairwise = compare_importance_profiles(profiles)
            clustering_results[sname] = {
                'n_sub_clusters': len(train_sub_indices),
                'profile_stability': mean_sp,
                'profiles': profiles,
                'cluster_sizes': {
                    c: len(idx) for c, idx in train_sub_indices.items()},
            }
            print(f"{len(train_sub_indices)} clusters, "
                  f"stability={mean_sp:.3f}")

        except Exception as e:
            clustering_results[sname] = {'error': str(e)}
            print(f"FAILED: {e}")

    results['c3_within_silo_clustering'] = convert_numpy(clustering_results)

    # ── Phase 5: C4 — Noise heterogeneity ─────────────────────────────────
    print("\n--- Phase 5: C4 — Noise heterogeneity ---")
    noise_results = {}

    # Inject noise into the largest silo (ddg_exp), keep others clean
    noisy_silo = 'ddg_exp' if 'ddg_exp' in silo_data else silo_names[0]
    print(f"  Injecting Gaussian noise (sigma={NOISE_SIGMA}) into {noisy_silo}")

    # Build pooled data across silos for federated feature selection
    all_labels = []
    all_reps_pooled = {r: [] for r in shared_reps}
    silo_idx_map = {}  # silo_name → indices into pooled array
    offset = 0

    for sname in silo_names:
        sd = silo_data[sname]
        tr, _ = silo_splits.get(sname, _split_silo(sd))
        n_tr = len(tr)
        # Z-score normalize per silo before pooling (different endpoints)
        silo_y = sd['labels'][tr]
        silo_std = np.std(silo_y)
        silo_y_z = (zscore(silo_y) if silo_std > 1e-12
                     else np.zeros_like(silo_y))
        all_labels.append(silo_y_z)
        for r in shared_reps:
            if r in sd['reps']:
                all_reps_pooled[r].append(sd['reps'][r][tr])
        silo_idx_map[sname] = np.arange(offset, offset + n_tr)
        offset += n_tr

    # Check all reps have consistent sizes
    valid_reps = {}
    for r in shared_reps:
        parts = all_reps_pooled[r]
        if len(parts) == len(silo_names):
            valid_reps[r] = np.vstack(parts)
    pooled_labels = np.concatenate(all_labels)

    if len(valid_reps) >= 2:
        # Inject noise
        noise_silo_idx = {
            i + 1: silo_idx_map[sname]
            for i, sname in enumerate(silo_names)
        }
        noise_levels = {i + 1: 0.0 for i in range(len(silo_names))}
        noisy_silo_id = silo_names.index(noisy_silo) + 1
        noise_levels[noisy_silo_id] = NOISE_SIGMA

        labels_noisy, noise_mask, _ = inject_noise_per_silo(
            pooled_labels, noise_silo_idx, noise_levels,
            noise_type='gaussian')

        # Federated feature selection with noisy labels
        try:
            X_h_noisy, silo_budgets, silo_scores, _ = \
                federated_feature_selection(
                    valid_reps, labels_noisy, noise_silo_idx,
                    len(silo_names),
                    allocation_method=_alloc, importance_method=_imp,
                    internal_evaluator=_int_eval)

            # Check: does noisy silo get less budget?
            noise_results = {
                'noisy_silo': noisy_silo,
                'sigma': NOISE_SIGMA,
                'silo_budgets': silo_budgets,
                'silo_scores': silo_scores,
                'noisy_silo_budget': silo_budgets.get(noisy_silo_id, 0),
                'clean_silo_budgets': {
                    silo_names[i]: silo_budgets.get(i + 1, 0)
                    for i in range(len(silo_names))
                    if silo_names[i] != noisy_silo},
            }

            noisy_budget = silo_budgets.get(noisy_silo_id, 0)
            other_budgets = [silo_budgets.get(i + 1, 0)
                             for i in range(len(silo_names))
                             if i + 1 != noisy_silo_id]
            avg_clean = np.mean(other_budgets) if other_budgets else 0
            print(f"  Noisy silo budget: {noisy_budget}, "
                  f"avg clean: {avg_clean:.0f}")
            if noisy_budget < avg_clean:
                print("  --> KIRBy downweights noisy silo")
            else:
                print("  --> No implicit noise detection")

        except Exception as e:
            noise_results = {'error': str(e)}
            print(f"  FAILED: {e}")
    else:
        noise_results = {'error': f'only {len(valid_reps)} valid pooled reps'}
        print(f"  SKIP: insufficient pooled reps")

    results['c4_noise_heterogeneity'] = convert_numpy(noise_results)

    # ── Phase 6: C5 — Pooled binding ──────────────────────────────────────
    print("\n--- Phase 6: C5 — Pooled binding ---")
    pooled_binding = {}

    binding_silos = ['ddg_exp', 'dms']
    available_binding = [s for s in binding_silos if s in silo_data]

    if len(available_binding) >= 2:
        print(f"  Pooling: {available_binding} (z-score normalized)")
        try:
            # Z-score normalize each silo's labels before pooling
            pool_labels = []
            pool_reps = {r: [] for r in shared_reps}
            pool_source = []

            for sname in available_binding:
                sd = silo_data[sname]
                tr, _ = silo_splits.get(sname, _split_silo(sd))
                y_tr = sd['labels'][tr]
                y_z = (zscore(y_tr) if len(y_tr) > 1 and np.std(y_tr) > 1e-12
                       else np.zeros_like(y_tr))
                pool_labels.append(y_z)
                pool_source.extend([sname] * len(tr))
                for r in shared_reps:
                    if r in sd['reps']:
                        pool_reps[r].append(sd['reps'][r][tr])

            pool_labels = np.concatenate(pool_labels)
            pool_valid_reps = {}
            for r in shared_reps:
                parts = pool_reps[r]
                if len(parts) == len(available_binding):
                    pool_valid_reps[r] = np.vstack(parts)

            if len(pool_valid_reps) >= 2:
                # Concatenate groups with offset to avoid cross-silo collisions
                pool_groups_list = []
                offset = 0
                for sname in available_binding:
                    sd = silo_data[sname]
                    tr_s, _ = silo_splits.get(sname, _split_silo(sd))
                    g = sd.get('groups')
                    if g is not None:
                        pool_groups_list.append(g[tr_s] + offset)
                        offset = int(g[tr_s].max()) + 1
                    else:
                        pool_groups_list.append(np.arange(len(tr_s)) + offset)
                        offset += len(tr_s)
                pool_groups = np.concatenate(pool_groups_list)

                # 80/20 split (group-aware)
                p_tr, p_te = next(cv_split(
                    pool_groups, 1, np.zeros(len(pool_labels)),
                    pool_labels))

                rep_tr = {r: arr[p_tr] for r, arr in pool_valid_reps.items()}
                rep_te = {r: arr[p_te] for r, arr in pool_valid_reps.items()}

                X_h, fi = create_hybrid(
                    rep_tr, pool_labels[p_tr],
                    allocation_method=_alloc,
                    importance_method=_imp,
                    internal_evaluator=_int_eval)
                X_h_te = apply_feature_selection(rep_te, fi)

                reg = RandomForestRegressor(
                    n_estimators=150, max_depth=15, random_state=0)
                reg.fit(X_h, pool_labels[p_tr])
                y_pred = reg.predict(X_h_te)
                pooled_binding = compute_reg_metrics(pool_labels[p_te], y_pred)
                pooled_binding['n_pooled'] = len(pool_labels)
                pooled_binding['silos_pooled'] = available_binding
                print(f"  Pooled RMSE={pooled_binding['rmse']:.4f}  "
                      f"rho={pooled_binding['spearman_rho']:.4f}")
            else:
                pooled_binding = {'error': 'insufficient shared reps for pooling'}
                print("  SKIP: insufficient reps")

        except Exception as e:
            pooled_binding = {'error': str(e)}
            print(f"  FAILED: {e}")
    else:
        pooled_binding = {'error': f'need >= 2 binding silos, got {len(available_binding)}'}
        print(f"  SKIP: only {len(available_binding)} binding silos")

    results['c5_pooled_binding'] = convert_numpy(pooled_binding)

    # ── Phase 7: C6 — Rep diversity per silo ──────────────────────────────
    print("\n--- Phase 7: C6 — Rep diversity per silo ---")
    rep_diversity = {}

    # Use one silo's data (largest) for this condition
    lead_silo = max(silo_data, key=lambda s: len(silo_data[s]['labels']))
    sd_lead = silo_data[lead_silo]
    print(f"  Using {lead_silo} data ({len(sd_lead['labels'])} samples)")

    # Resolve available reps per diversity silo
    div_silos = {}
    for div_name, rep_list in REP_DIVERSITY_SILOS.items():
        available = [r for r in rep_list if r in sd_lead['reps']]
        if available:
            div_silos[div_name] = available

    K_div = len(div_silos)
    if K_div < 2:
        rep_diversity = {'error': f'only {K_div} rep diversity silos available'}
        print(f"  SKIP: need >= 2 diversity silos")
    else:
        print(f"  {K_div} diversity silos: "
              f"{', '.join(f'{k}={v}' for k, v in div_silos.items())}")

        tr, te = silo_splits.get(lead_silo, _split_silo(sd_lead))
        y_tr = sd_lead['labels'][tr]
        y_te = sd_lead['labels'][te]

        # Per-silo baselines
        div_baselines = {}
        teacher_models = [None]  # index 0 placeholder
        silo_train_sim = [None]
        silo_test_sim = [None]
        silo_train_qual = [None]
        silo_test_qual = [None]

        div_names = sorted(div_silos.keys())
        for dname in div_names:
            rlist = div_silos[dname]
            X_tr = np.hstack([sd_lead['reps'][r][tr] for r in rlist])
            X_te = np.hstack([sd_lead['reps'][r][te] for r in rlist])

            reg = RandomForestRegressor(
                n_estimators=150, max_depth=15, random_state=0)
            reg.fit(X_tr, y_tr)
            y_pred = reg.predict(X_te)
            div_baselines[dname] = compute_reg_metrics(y_te, y_pred)
            div_baselines[dname]['reps'] = rlist

            teacher_models.append((reg, len(X_tr), X_tr))
            silo_train_sim.append(X_tr)
            silo_test_sim.append(X_te)
            silo_train_qual.append((X_tr, y_tr))
            silo_test_qual.append(X_te)

            print(f"    {dname}: RMSE={div_baselines[dname]['rmse']:.4f}")

        # FLuID-style quality ensemble
        try:
            y_pred_ens, rmse_w = predict_fluid_quality_ensemble_regression(
                teacher_models, silo_train_qual, silo_test_qual, K_div)
            ensemble_metrics = compute_reg_metrics(y_te, y_pred_ens)
            ensemble_metrics['rmse_weights'] = rmse_w
            print(f"    Quality ensemble: RMSE={ensemble_metrics['rmse']:.4f}")
        except Exception as e:
            ensemble_metrics = {'error': str(e)}
            print(f"    Ensemble FAILED: {e}")

        # KIRBy hybrid across all diversity silo reps
        try:
            all_div_reps_tr = {}
            all_div_reps_te = {}
            for dname in div_names:
                for r in div_silos[dname]:
                    all_div_reps_tr[r] = sd_lead['reps'][r][tr]
                    all_div_reps_te[r] = sd_lead['reps'][r][te]

            X_h, fi_h = create_hybrid(
                all_div_reps_tr, y_tr,
                allocation_method=_alloc,
                importance_method=_imp,
                internal_evaluator=_int_eval)
            X_h_te = apply_feature_selection(all_div_reps_te, fi_h)

            reg_h = RandomForestRegressor(
                n_estimators=150, max_depth=15, random_state=0)
            reg_h.fit(X_h, y_tr)
            y_pred_h = reg_h.predict(X_h_te)
            hybrid_metrics = compute_reg_metrics(y_te, y_pred_h)
            hybrid_metrics['allocation'] = convert_numpy(
                fi_h.get('allocation', {}))
            print(f"    KIRBy hybrid: RMSE={hybrid_metrics['rmse']:.4f}")
        except Exception as e:
            hybrid_metrics = {'error': str(e)}
            print(f"    Hybrid FAILED: {e}")

        rep_diversity = {
            'lead_silo': lead_silo,
            'diversity_silos': dict(div_silos),
            'baselines': div_baselines,
            'quality_ensemble': ensemble_metrics,
            'kirby_hybrid': hybrid_metrics,
        }

    results['c6_rep_diversity'] = convert_numpy(rep_diversity)

    # ── Phase 8: C7/C8 — FLuID ensemble baselines ───────────────────────
    # Tests the central claim: KIRBy outperforms FLuID for antibodies.
    # C7: FLuID quality ensemble. C8: FLuID similarity ensemble (ESM2 dist).
    print("\n--- Phase 8: C7/C8 — FLuID ensemble baselines ---")

    # Train per-silo teachers on concatenated shared reps
    silo_teachers = {}
    silo_test_X = {}
    silo_test_y = {}
    for sname in silo_names:
        sd = silo_data[sname]
        tr, te = silo_splits.get(sname, _split_silo(sd))
        reps_s = {r: sd['reps'][r] for r in shared_reps if r in sd['reps']}
        if len(reps_s) < 2:
            continue
        X_tr = np.hstack([reps_s[r][tr] for r in sorted(reps_s)])
        X_te = np.hstack([reps_s[r][te] for r in sorted(reps_s)])
        X_tr, X_te = cap_dimensions(X_tr, X_te)
        y_tr = sd['labels'][tr]
        y_te = sd['labels'][te]

        reg = RandomForestRegressor(
            n_estimators=150, max_depth=15, random_state=0)
        reg.fit(X_tr, y_tr)
        silo_teachers[sname] = (reg, X_tr, y_tr)
        silo_test_X[sname] = X_te
        silo_test_y[sname] = y_te

    if len(silo_teachers) >= 2:
        # C7: FLuID quality ensemble — weight by inverse-RMSE per silo
        print("  C7: FLuID quality ensemble...")
        c7_per_silo = {}
        for target_name in silo_names:
            if target_name not in silo_test_X:
                continue
            X_te = silo_test_X[target_name]
            y_te = silo_test_y[target_name]

            # Each silo-teacher predicts on target's test set
            preds = {}
            weights = {}
            for sname, (reg, X_tr, y_tr) in silo_teachers.items():
                pred = reg.predict(X_te)
                preds[sname] = pred
                # Weight = inverse RMSE on own train set (cross-val)
                from sklearn.model_selection import cross_val_predict
                try:
                    cv_pred = cross_val_predict(reg, X_tr, y_tr, cv=3)
                    rmse = float(np.sqrt(np.mean((y_tr - cv_pred) ** 2)))
                    weights[sname] = 1.0 / max(rmse, 1e-6)
                except Exception:
                    weights[sname] = 1.0

            # Weighted average prediction
            w_arr = np.array([weights[s] for s in preds])
            w_arr /= w_arr.sum()
            ensemble_pred = sum(
                w * preds[s] for s, w in zip(preds, w_arr))
            c7_m = compute_reg_metrics(y_te, ensemble_pred)
            c7_m['weights'] = convert_numpy(
                {s: float(w) for s, w in zip(preds, w_arr)})
            c7_per_silo[target_name] = c7_m
            print(f"    {target_name}: RMSE={c7_m['rmse']:.4f}  "
                  f"rho={c7_m['spearman_rho']:.4f}")

        results['c7_fluid_quality'] = convert_numpy(c7_per_silo)

        # C8: FLuID similarity ensemble — weight by ESM2 embedding distance
        print("  C8: FLuID similarity ensemble (ESM2 distance)...")
        c8_per_silo = {}
        esm2_key = 'esm2' if 'esm2' in shared_reps else None
        if esm2_key is None:
            # Fall back to a known-good embedding rep for distance
            for _fb in ['prot_t5', 'ablang2', 'esmc', 'antiberty']:
                if _fb in shared_reps:
                    esm2_key = _fb
                    break
            else:
                esm2_key = shared_reps[0] if shared_reps else None

        if esm2_key is not None:
            for target_name in silo_names:
                sd_target = silo_data[target_name]
                if target_name not in silo_test_X:
                    continue
                _, te = silo_splits.get(
                    target_name, _split_silo(sd_target))
                y_te = silo_test_y[target_name]

                # Get ESM2 embeddings for KNN distance
                if esm2_key not in sd_target['reps']:
                    c8_per_silo[target_name] = {
                        'error': f'{esm2_key} not in target reps'}
                    continue
                test_emb = sd_target['reps'][esm2_key][te]

                # Build KNN from each silo's train ESM2
                from sklearn.neighbors import NearestNeighbors
                silo_preds = {}
                silo_dists = {}
                for sname, (reg, _, _) in silo_teachers.items():
                    sd_s = silo_data[sname]
                    tr_s, _ = silo_splits.get(sname, _split_silo(sd_s))
                    if esm2_key not in sd_s['reps']:
                        continue
                    train_emb = sd_s['reps'][esm2_key][tr_s]

                    # 8-NN distance to silo's train set
                    k_nn = min(8, len(train_emb) - 1)
                    if k_nn < 1:
                        continue
                    nn = NearestNeighbors(n_neighbors=k_nn)
                    nn.fit(train_emb)
                    dists, _ = nn.kneighbors(test_emb)
                    mean_dist = dists.mean(axis=1)

                    X_te = silo_test_X[target_name]
                    pred = reg.predict(X_te)
                    silo_preds[sname] = pred
                    silo_dists[sname] = mean_dist

                if len(silo_preds) >= 2:
                    # Similarity weight = inverse distance
                    # Per test sample: weight each silo by 1/dist
                    n_te = len(y_te)
                    ensemble_pred = np.zeros(n_te)
                    total_w = np.zeros(n_te)
                    for sname in silo_preds:
                        w = 1.0 / np.maximum(silo_dists[sname], 1e-10)
                        ensemble_pred += w * silo_preds[sname]
                        total_w += w
                    ensemble_pred /= total_w

                    c8_m = compute_reg_metrics(y_te, ensemble_pred)
                    c8_m['distance_rep'] = esm2_key
                    c8_per_silo[target_name] = c8_m
                    print(f"    {target_name}: RMSE={c8_m['rmse']:.4f}  "
                          f"rho={c8_m['spearman_rho']:.4f}")
                else:
                    c8_per_silo[target_name] = {
                        'error': 'insufficient silos with distance rep'}
        else:
            c8_per_silo = {'error': 'no shared rep for distance'}

        results['c8_fluid_similarity'] = convert_numpy(c8_per_silo)
    else:
        results['c7_fluid_quality'] = {
            'error': f'only {len(silo_teachers)} silos with teachers'}
        results['c8_fluid_similarity'] = {
            'error': f'only {len(silo_teachers)} silos with teachers'}
        print("  SKIP: insufficient silo teachers")

    # ── Phase 9: Save results ─────────────────────────────────────────────
    final = {
        'experiment': 'fed05_antibody_federated',
        'dataset': dataset_name,
        'config': {
            'silos': dict(SILO_DATASETS),
            'shared_reps': shared_reps,
            'k_intra': K_INTRA,
            'noise_sigma': NOISE_SIGMA,
        },
        **results,
    }

    # Summary table
    print(f"\n{'─' * 70}")
    print(f"  {'Silo':<12} {'C1 Ind':>10} {'Best C2':>10}")
    for sname in silo_names:
        ind = independent.get(sname, {})
        ind_rmse = ind.get('rmse', float('nan'))
        best_tr = float('nan')
        for k, v in transfer_results.items():
            if k.endswith(f'→{sname}') and 'rmse' in v:
                if np.isnan(best_tr) or v['rmse'] < best_tr:
                    best_tr = v['rmse']
        print(f"  {sname:<12} {ind_rmse:>10.4f} {best_tr:>10.4f}")

    return final


if __name__ == '__main__':
    # Fed05 loads its own silo data — the dataset arg is for naming only
    import argparse
    parser = argparse.ArgumentParser(
        description='Fed05: Antibody Federated Extension')
    parser.add_argument('--output', type=str,
                        default='tests/results/federated')
    args = parser.parse_args()

    # Pass placeholder data — run() loads silos internally
    data = {'reps': {}, 'labels': np.array([]), 'domain': 'antibody'}
    results = run(data, {}, args.output, 'antibody_federated')
    save_experiment_results(
        results, args.output, 'fed05_antibody_federated', 'antibody')
