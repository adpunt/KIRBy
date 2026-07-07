#!/usr/bin/env python3
"""C13: Forward Compatibility — do new reps benefit from hybridization (and vice versa)?

Paper: Complexity
Conclusion: Quantifies the symbiotic value of new representations within the
    KIRBy hybrid framework, WITH the selection-on-same-data bias controlled.
    A new rep is only credited when the hybrid WITH it beats the hybrid built
    around a random-Gaussian matrix of the SAME dimensionality (D - D_random),
    not merely when it beats the smaller hybrid without it (D - C, which is a
    selection-inflated upper bound).
Dependencies: None (standalone).

Condition ablation per new rep r, per dataset d:
  A: New rep r + ridge           (raw embedding quality, linear)
  B: New rep r + best model      (best-case single-rep)
  C: Best hybrid WITHOUT r       (current KIRBy ceiling; search space = existing)
  D: Best hybrid WITH r          (search space = existing + r)
  D_random: Best hybrid WITH a random-Gaussian matrix shaped like r
            (search space = existing + noise; the selection-bias null for D)

  (Condition E is gone. It was once a duplicate of D and carried no
   information; the "does the hybrid help r itself" question is now answered
   honestly by D - B, below. No E / E - B is emitted any more.)

Controls per new rep r:
  - Random Gaussian matrix (same dims as r): single-rep permutation null (cond A)
  - Row-shuffled r: tests content vs dimensionality

Key metrics (all sign-normalised: positive = improvement, any metric direction):
  1. D - C          : new-rep contribution, IN-SAMPLE UPPER BOUND ONLY. D searches
                      a superset of C on the same data, so this is biased positive
                      by construction. Never present it as an unbiased effect.
  2. D - D_random   : the DEBIASED new-rep contribution. Subtracts the selection
                      inflation that any same-dim matrix earns. If D - C is not
                      clearly above D_random - C (i.e. D - D_random <= 0), the
                      apparent gain is selection noise, not the rep.
  3. D - B          : does the hybrid improve the new rep beyond its own best
                      single-model score? (the honest "hybrid helps r" metric)
  4. Allocation share: features_from_r / total_budget
  5. Win rate       : fraction of datasets where the DEBIASED effect (D - D_random)
                      and D - B are positive.

No significance testing is reported. Wilcoxon on <=5 non-independent CV folds is
underpowered and Friedman/Nemenyi need many datasets per call; a p-value here would
be misleading. Present effect sizes (the deltas above) with fold spread, not p-values.

Usage:
    # Auto-detect new reps (anything not in LEGACY_REPS):
    python c13_forward_compatibility.py --dataset logd --reps all --mode full

    # Specify new reps explicitly:
    python c13_forward_compatibility.py --dataset logd --reps all --mode full \\
        --new-reps esm3,saprot

    # Intermediary (quick, 1-fold):
    python c13_forward_compatibility.py --dataset logd --reps all --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split,
    build_fold_hybrid,
    get_best_search_for_downstream,
    parse_rep_args,
    _is_tcga_dataset, _is_antibody_dataset,
)


# =============================================================================
# LEGACY REPS — anything NOT in this set is considered "new"
# =============================================================================

# Molecular legacy reps (available before Phase 4 forward-compat additions).
# Audited 2026-05-25 against current DATASET_REPS molecular `general` lists
# (logd, caco2, herg/herg_ki, adme_*, qm9, etc.).  Stale entries removed
# (rdkit_desc, map4, erg, molbert), missing-but-pre-Phase-4 added.
_LEGACY_MOLECULAR = {
    'ecfp4', 'maccs', 'rdkit_desc', 'mordred', 'mordred_pca50', 'avalon',
    'mol2vec',
    # Pre-Phase 4 fingerprints / kernels / descriptors / GNNs
    'atom_pair', 'top_torsion', 'rdkfp', 'pdv', 'graph_kernel', 'smited',
    'grover', 'mhg_gnn', 'graphmvp',
    # Pre-trained language models (pre-Phase 4)
    'molbert', 'chemberta', 'chembert', 'molformer', 'selformer',
}

# Antibody legacy reps
_LEGACY_ANTIBODY = {
    'onehot', 'blosum', 'aac', 'paratope', 'zscale', 'kidera',
    'esm2', 'ablang', 'protbert', 'antiberty',
    'esm2_concat', 'esm2_cdr3', 'esm2_layer24', 'esm2_early',
    'abroberta', 'abroberta_cdr',
    'antiberty_llr', 'esmc_llr',
}

# DTA / protein legacy reps
_LEGACY_DTA = {
    'ecfp4', 'maccs', 'rdkit_desc', 'mordred', 'mordred_pca50', 'avalon',
    'mol2vec', 'map4', 'erg',
    'molbert', 'chemberta',
    'esm2', 'protbert',
    'prot_albert', 'prot_xlnet', 'prot_electra', 'prot_t5',
    'contact_graph', 'usr_shape', 'pocket',
}

# TCGA legacy reps
_LEGACY_TCGA = {
    'gene_hvg2000', 'gene_pca100', 'gene_hallmark', 'gene_progeny',
    'gene_kegg', 'gene_dorothea', 'gene_reactome', 'gene_oncogenic',
    'gene_combined_pathway', 'gene_nmf50', 'gene_ica50',
    'gene_sparse_pca50', 'gene_autoencoder50', 'gene_corr_modules',
    'gene_quantile_rank', 'gene_immune', 'gene_stats',
    'gene_chromosome_arm',
    'methyl_pca50', 'methyl_hvg', 'promoter_meth',
    'mutation_matrix', 'mut_tmb', 'mut_driver', 'mut_pathway',
    'mut_pca50', 'sig_sbs', 'sig_dbs', 'sig_id',
    'cnv_gistic', 'cnv_pca50',
    'mirna_top200', 'rppa', 'clinical',
    'immune_cibersort', 'immune_xcell',
    'purity_consensus', 'stemness_mrnasi', 'heterogeneity_math',
    'msi_status', 'hrd_score', 'cimp_status',
    'wsi_ctranspath', 'wsi_phikon', 'wsi_hoptimus0', 'wsi_gigapath',
    'wsi_moco', 'wsi_titan', 'wsi_uni2h',
    'lncrna_pca50', 'splicing_psi50', 'gsva_hallmark',
}

# Reaction legacy reps
_LEGACY_REACTION = {
    'drfp', 'rxnfp', 'onehot_components', 'ecfp4_components',
    'maccs_components', 'mordred_components', 'mol2vec_components',
    'drug_sns',
}

# Union of all legacy reps across domains
LEGACY_REPS = (
    _LEGACY_MOLECULAR | _LEGACY_ANTIBODY | _LEGACY_DTA
    | _LEGACY_TCGA | _LEGACY_REACTION
)


def _detect_new_reps(reps, domain=None):
    """Auto-detect new reps: anything loaded that is NOT in LEGACY_REPS."""
    new = [name for name in sorted(reps.keys()) if name not in LEGACY_REPS]
    return new


def _split_reps(reps, new_rep_names):
    """Split reps dict into new and existing subsets."""
    new_reps = {k: v for k, v in reps.items() if k in new_rep_names}
    existing_reps = {k: v for k, v in reps.items() if k not in new_rep_names}
    return new_reps, existing_reps


# =============================================================================
# PER-FOLD EVALUATION HELPERS
# =============================================================================

def _eval_single_rep_per_fold(X, labels, groups, model_factory, n_folds, mode,
                               metric_fn):
    """Evaluate a single rep with a single model, returning per-fold scores."""
    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X, labels, mode=mode):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = model_factory()
        model.fit(X_train, labels[train_idx])
        scores.append(metric_fn(labels[val_idx], model.predict(X_val)))

    return scores


def _eval_hybrid_per_fold(reps, labels, groups, model_factory, n_folds, mode,
                           metric_fn, search_speed='full', augmentations=None,
                           metric_name='r2', dataset_name=None,
                           task_type='regression', condition_suffix=None):
    """Run find_best_hybrid and evaluate the best config per fold.

    Returns (per_fold_scores, search_result_dict).
    search_result_dict contains best_by_model results, baselines, allocation info.
    """
    if len(reps) < 2:
        # find_best_hybrid needs >= 2 reps; fall back to single-rep eval
        rep_name = list(reps.keys())[0]
        X = reps[rep_name]
        scores = _eval_single_rep_per_fold(
            X, labels, groups, model_factory, n_folds, mode, metric_fn)
        return scores, {
            'best_result': {'score_mean': float(np.mean(scores)) if scores else float('nan')},
            'baselines': {},
            'allocation': {rep_name: X.shape[1]},
            'n_features': X.shape[1],
        }

    # Try every common downstream model via the pool helper; keep the overall
    # winner. The pool helper picks the best (tier, aug_cond) cache file per
    # candidate model.
    best = None
    best_search = None
    best_meta = None
    for candidate_model in ['rf', 'svm', 'lightgbm', 'mlp', 'knn']:
        try:
            _extra = {'condition_suffix': condition_suffix} if condition_suffix else {}
            s, m = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=candidate_model, budget=None,
                augmentations=augmentations,
                speed=search_speed,
                dataset_name=dataset_name,
                task_type=task_type, metric=metric_name,
                return_fitted=True,
                **_extra,
            )
        except Exception:
            continue
        if not m:
            continue
        bc = m['best_config']
        score = bc.get('score_mean')
        if score is None:
            continue
        lib = s.lower_is_better
        is_better = (best is None) or (
            score < best['score_mean'] if lib else score > best['score_mean'])
        if is_better:
            best = bc
            best_search = s
            best_meta = m
    if best is None or best_search is None:
        print(f"      pool helper found no usable blueprint")
        return [], {'error': 'no blueprint found via pool'}
    search = best_search

    blueprint = best.get('blueprint')
    if blueprint is None:
        return [], {'error': 'no blueprint in best_config'}

    scores = []
    allocation_dicts = []
    last_feature_info = None
    sample_X = next(iter(reps.values()))

    for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
        train_reps = {k: v[train_idx] for k, v in reps.items()}
        val_reps = {k: v[val_idx] for k, v in reps.items()}
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_augs = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
        val_augs = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

        try:
            X_train, X_val, feature_info = build_fold_hybrid(
                train_reps, val_reps, y_train, blueprint,
                augmentations_train=train_augs, augmentations_val=val_augs)
        except Exception:
            continue

        # Extract allocation from feature_info
        alloc = feature_info.get('allocation', {})
        allocation_dicts.append(alloc)
        last_feature_info = feature_info

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train_s, y_train)
        scores.append(metric_fn(y_val, model.predict(X_val_s)))

    # Average allocation across folds
    avg_alloc = {}
    if allocation_dicts:
        all_keys = set()
        for ad in allocation_dicts:
            all_keys.update(ad.keys())
        for k in sorted(all_keys):
            avg_alloc[k] = float(np.mean([ad.get(k, 0) for ad in allocation_dicts]))

    result_info = {
        'best_result': convert_numpy(best),
        'baselines': convert_numpy(search.baselines),
        'allocation': avg_alloc,
        'n_features': best.get('n_features', sum(avg_alloc.values()) if avg_alloc else 0),
        'feature_info': last_feature_info or {},  # consumed by modality_novelty
        'blueprint': {
            'allocation_method': blueprint.allocation_method,
            'importance_method': blueprint.importance_method,
            'budget': blueprint.budget,
            'internal_evaluator': blueprint.internal_evaluator,
        },
    }

    return scores, result_info


# =============================================================================
# CONTROL CONDITIONS
# =============================================================================

def _make_random_gaussian(X, seed=42):
    """Random Gaussian matrix with same shape as X — permutation null."""
    rng = np.random.RandomState(seed)
    return rng.randn(*X.shape).astype(np.float32)


def _make_row_shuffled(X, seed=42):
    """Row-shuffled version of X — tests content vs dimensionality."""
    rng = np.random.RandomState(seed)
    X_shuffled = X.copy()
    for col in range(X_shuffled.shape[1]):
        rng.shuffle(X_shuffled[:, col])
    return X_shuffled


# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run(data, config, output_dir, dataset_name, new_rep_names=None, **kwargs):
    """C13: Forward Compatibility — 5-condition ablation per new rep.

    Args:
        data: Loaded dataset dict from load_data().
        config: Config dict (fast or full).
        output_dir: Output directory for results JSON.
        dataset_name: Name of the dataset.
        new_rep_names: List of rep names to treat as "new". If None, auto-detect.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C13: FORWARD COMPATIBILITY")
    print("=" * 70)

    # Smoke-test shrink (gated on KIRBY_SMOKE_N; no-op on server/full runs).
    # precompute_hybrid_searches reads the module-level _shared.FULL_BUDGETS and
    # IGNORES config['budgets'], so the only way to make the local hybrid search
    # tractable is to shrink FULL_BUDGETS here before any precompute fires.
    if os.environ.get('KIRBY_SMOKE_N', '').strip():
        import experiments._shared as _shared_mod
        if len(_shared_mod.FULL_BUDGETS) > 1:
            _shared_mod.FULL_BUDGETS = [25]
            print("  [SMOKE] FULL_BUDGETS shrunk to [25] for local hybrid search")

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    augs = data.get('augmentations')

    # Size-gate (2026-06-28): c13 runs uncached find_best_hybrid searches with
    # FULL budgets (it ignores config['budgets']), so it blows the wall on big
    # datasets. Above the cutoff, subsample (reps/labels/groups/augs jointly) to
    # K rows — same approach c17 already uses internally. Fixed seed = stable.
    from experiments._shared import LARGE_DATASET_N, LARGE_SUBSAMPLE_K
    if len(labels) > LARGE_DATASET_N:
        _rng = np.random.RandomState(42)
        _idx = np.sort(_rng.choice(len(labels), LARGE_SUBSAMPLE_K, replace=False))
        reps = {k: v[_idx] for k, v in reps.items()}
        labels = labels[_idx]
        groups = groups[_idx] if groups is not None else None
        if augs is not None:
            augs = {k: v[_idx] for k, v in augs.items()}
        print(f"  [size-gate] n>{LARGE_DATASET_N}: subsampled to "
              f"{LARGE_SUBSAMPLE_K} rows for c13")

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    search_speed = config['search_speed']

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    models = get_model_factories(
        config['models'],
        task_type=data['task_type'],
        n_samples=len(labels),
    )

    # Determine new vs existing reps
    if new_rep_names is None:
        new_rep_names = _detect_new_reps(reps, domain)
    else:
        # Filter to reps that are actually loaded
        new_rep_names = [r for r in new_rep_names if r in reps]

    new_reps, existing_reps = _split_reps(reps, new_rep_names)

    if not new_reps:
        print("\n  WARNING: No new reps detected. All loaded reps are in LEGACY_REPS.")
        print("  Use --new-reps to specify which reps to treat as new.")
        print(f"  Loaded reps: {sorted(reps.keys())}")
        return {'error': 'no_new_reps', 'loaded_reps': sorted(reps.keys())}

    print(f"  Domain:       {domain}")
    print(f"  Metric:       {metric_name} ({'lower is better' if lower_is_better else 'higher is better'})")
    print(f"  n_samples:    {len(labels)}")
    print(f"  n_folds:      {n_folds}")
    print(f"  Models:       {list(models.keys())}")
    print(f"  Search speed: {search_speed}")
    print(f"\n  New reps ({len(new_reps)}): "
          f"{', '.join(f'{k}({v.shape[1]}D)' for k, v in sorted(new_reps.items()))}")
    print(f"  Existing reps ({len(existing_reps)}): "
          f"{', '.join(f'{k}({v.shape[1]}D)' for k, v in sorted(existing_reps.items()))}")

    # Find best model for each condition via CV
    # We use all available models, then pick the best for conditions A/B
    per_rep_results = {}

    for rep_name, X_new in sorted(new_reps.items()):
        print(f"\n{'='*60}")
        print(f"  NEW REP: {rep_name} (dim={X_new.shape[1]})")
        print(f"{'='*60}")

        rep_result = {
            'dim': X_new.shape[1],
            'conditions': {},
            'controls': {},
            'deltas': {},
        }

        # =================================================================
        # Condition A: new rep r + ridge (raw linear quality)
        # =================================================================
        print(f"\n  Condition A: {rep_name} + ridge")
        ridge_factory = models.get('ridge')
        if ridge_factory is None:
            # Fallback to first available model
            first_model_name = list(models.keys())[0]
            ridge_factory = models[first_model_name]
            print(f"    (ridge not available, using {first_model_name})")

        a_scores = _eval_single_rep_per_fold(
            X_new, labels, groups, ridge_factory, n_folds, mode, metric_fn)
        a_mean = float(np.mean(a_scores)) if a_scores else float('nan')
        a_std = float(np.std(a_scores)) if a_scores else float('nan')
        rep_result['conditions']['A'] = {
            'description': f'{rep_name} + ridge',
            'score_mean': a_mean,
            'score_std': a_std,
            'fold_scores': [float(s) for s in a_scores],
        }
        print(f"    {metric_name} = {a_mean:.4f} +/- {a_std:.4f}")

        # =================================================================
        # Condition B: new rep r + best model
        # =================================================================
        print(f"\n  Condition B: {rep_name} + best model")
        best_b_mean = -np.inf if not lower_is_better else np.inf
        best_b_model = None
        best_b_scores = []

        for model_name, model_factory in models.items():
            b_scores = _eval_single_rep_per_fold(
                X_new, labels, groups, model_factory, n_folds, mode, metric_fn)
            b_mean = float(np.mean(b_scores)) if b_scores else float('nan')

            if np.isnan(b_mean):
                continue

            is_better = (b_mean > best_b_mean) if not lower_is_better else (b_mean < best_b_mean)
            if is_better:
                best_b_mean = b_mean
                best_b_model = model_name
                best_b_scores = b_scores

        b_std = float(np.std(best_b_scores)) if best_b_scores else float('nan')
        rep_result['conditions']['B'] = {
            'description': f'{rep_name} + {best_b_model}',
            'best_model': best_b_model,
            'score_mean': float(best_b_mean) if not np.isinf(best_b_mean) else float('nan'),
            'score_std': b_std,
            'fold_scores': [float(s) for s in best_b_scores],
        }
        print(f"    Best model: {best_b_model}")
        print(f"    {metric_name} = {best_b_mean:.4f} +/- {b_std:.4f}")

        # =================================================================
        # Condition C: best hybrid WITHOUT r (current KIRBy ceiling)
        # =================================================================
        print(f"\n  Condition C: best hybrid WITHOUT {rep_name}")
        if len(existing_reps) >= 2:
            # Use the best model from B for consistency
            c_model_factory = models.get(best_b_model, ridge_factory)
            c_scores, c_info = _eval_hybrid_per_fold(
                existing_reps, labels, groups, c_model_factory,
                n_folds, mode, metric_fn, search_speed=search_speed,
                augmentations=augs, metric_name=metric_name,
                # Pass real dataset_name + condition_suffix so the cache key
                # remains content-validated (the rep_shapes hash will already
                # distinguish "with r" vs "without r" automatically); synthetic
                # dataset names bypass the precompute cache and force a live
                # find_best_hybrid that sees all labels in its internal CV.
                dataset_name=dataset_name,
                condition_suffix=f"c13_without_{rep_name}",
                task_type=data['task_type'])
            c_mean = float(np.mean(c_scores)) if c_scores else float('nan')
            c_std = float(np.std(c_scores)) if c_scores else float('nan')
            rep_result['conditions']['C'] = {
                'description': f'hybrid WITHOUT {rep_name}',
                'score_mean': c_mean,
                'score_std': c_std,
                'fold_scores': [float(s) for s in c_scores],
                'hybrid_info': c_info,
            }
            print(f"    {metric_name} = {c_mean:.4f} +/- {c_std:.4f}")
            if c_info.get('allocation'):
                print(f"    Allocation: {c_info['allocation']}")
        else:
            print(f"    SKIPPED: need >= 2 existing reps (have {len(existing_reps)})")
            c_mean = float('nan')
            c_scores = []
            rep_result['conditions']['C'] = {
                'description': f'hybrid WITHOUT {rep_name} (skipped, < 2 existing reps)',
                'score_mean': float('nan'),
                'score_std': float('nan'),
                'fold_scores': [],
            }

        # =================================================================
        # Condition D: best hybrid WITH r (does r improve the hybrid?)
        # =================================================================
        print(f"\n  Condition D: best hybrid WITH {rep_name}")
        all_reps_for_d = {**existing_reps, rep_name: X_new}
        if len(all_reps_for_d) >= 2:
            d_model_factory = models.get(best_b_model, ridge_factory)
            d_scores, d_info = _eval_hybrid_per_fold(
                all_reps_for_d, labels, groups, d_model_factory,
                n_folds, mode, metric_fn, search_speed=search_speed,
                augmentations=augs, metric_name=metric_name,
                dataset_name=dataset_name,
                condition_suffix=f"c13_with_{rep_name}",
                task_type=data['task_type'])
            d_mean = float(np.mean(d_scores)) if d_scores else float('nan')
            d_std = float(np.std(d_scores)) if d_scores else float('nan')

            # Allocation share for the new rep
            alloc = d_info.get('allocation', {})
            total_budget = sum(alloc.values()) if alloc else 1
            alloc_share = alloc.get(rep_name, 0) / total_budget if total_budget > 0 else 0

            # Diversity vs depth: cosine distance between new rep and nearest
            # existing rep in the hybrid's selected feature space
            modality_novelty = float('nan')
            try:
                from scipy.spatial.distance import cosine as _cosine_dist
                feature_info = d_info.get('feature_info', {})
                if feature_info and rep_name in feature_info:
                    new_selected = feature_info[rep_name].get('selected_indices', [])
                    if new_selected:
                        new_vec = X_new[:, new_selected].mean(axis=0)
                        min_dist = float('inf')
                        for ename, edata in feature_info.items():
                            if ename == rep_name:
                                continue
                            e_sel = edata.get('selected_indices', [])
                            if e_sel and ename in existing_reps:
                                e_vec = existing_reps[ename][:, e_sel].mean(axis=0)
                                # Pad to same length for cosine
                                maxlen = max(len(new_vec), len(e_vec))
                                nv = np.zeros(maxlen); nv[:len(new_vec)] = new_vec
                                ev = np.zeros(maxlen); ev[:len(e_vec)] = e_vec
                                d = _cosine_dist(nv, ev)
                                if d < min_dist:
                                    min_dist = d
                        if min_dist < float('inf'):
                            modality_novelty = float(min_dist)
            except Exception:
                pass

            rep_result['conditions']['D'] = {
                'description': f'hybrid WITH {rep_name}',
                'score_mean': d_mean,
                'score_std': d_std,
                'fold_scores': [float(s) for s in d_scores],
                'hybrid_info': d_info,
                'allocation_share': float(alloc_share),
                'features_from_new_rep': float(alloc.get(rep_name, 0)),
                'modality_novelty': modality_novelty,
            }
            print(f"    {metric_name} = {d_mean:.4f} +/- {d_std:.4f}")
            if alloc:
                print(f"    Allocation: {alloc}")
                print(f"    Features from {rep_name}: {alloc.get(rep_name, 0)} "
                      f"({alloc_share:.1%} of budget)")
        else:
            print(f"    SKIPPED: need >= 2 total reps")
            d_mean = float('nan')
            d_scores = []
            rep_result['conditions']['D'] = {
                'description': f'hybrid WITH {rep_name} (skipped, < 2 total reps)',
                'score_mean': float('nan'),
                'score_std': float('nan'),
                'fold_scores': [],
            }

        # =================================================================
        # Condition D_random: best hybrid WITH a random-Gaussian matrix
        # shaped like r — the selection-bias null for D.
        #
        # D searches the superset {existing + r} on the SAME data C searched,
        # so D >= C almost always by selection alone. Replacing r with a random
        # matrix of identical dimensionality earns exactly that selection
        # inflation and nothing else, so D_random - C measures the bias. The
        # rep's real contribution is D - D_random. A distinct condition_suffix
        # gives this its own cache file (find_best_hybrid keys on
        # dataset_name + condition_suffix), so it is computed live on the
        # random content and never collides with condition D.
        # =================================================================
        print(f"\n  Condition D_random: best hybrid WITH random-Gaussian null "
              f"(same dim as {rep_name})")
        d_random_mean = float('nan')
        d_random_scores = []
        if len(existing_reps) >= 2:
            X_null = _make_random_gaussian(X_new, seed=1234)
            all_reps_for_d_random = {**existing_reps, rep_name: X_null}
            dr_model_factory = models.get(best_b_model, ridge_factory)
            d_random_scores, d_random_info = _eval_hybrid_per_fold(
                all_reps_for_d_random, labels, groups, dr_model_factory,
                n_folds, mode, metric_fn, search_speed=search_speed,
                augmentations=augs, metric_name=metric_name,
                dataset_name=dataset_name,
                condition_suffix=f"c13_randomnull_{rep_name}",
                task_type=data['task_type'])
            d_random_mean = float(np.mean(d_random_scores)) if d_random_scores else float('nan')
            d_random_std = float(np.std(d_random_scores)) if d_random_scores else float('nan')
            dr_alloc = d_random_info.get('allocation', {})
            dr_total = sum(dr_alloc.values()) if dr_alloc else 1
            dr_share = dr_alloc.get(rep_name, 0) / dr_total if dr_total > 0 else 0
            rep_result['conditions']['D_random'] = {
                'description': f'hybrid WITH random-Gaussian null ({X_new.shape[1]}D)',
                'score_mean': d_random_mean,
                'score_std': d_random_std,
                'fold_scores': [float(s) for s in d_random_scores],
                'allocation_share': float(dr_share),
            }
            print(f"    {metric_name} = {d_random_mean:.4f} +/- {d_random_std:.4f}")
            print(f"    Null allocation share: {dr_share:.1%} of budget")
        else:
            print(f"    SKIPPED: need >= 2 existing reps for the null "
                  f"(have {len(existing_reps)})")
            rep_result['conditions']['D_random'] = {
                'description': 'hybrid WITH random-Gaussian null (skipped, < 2 existing reps)',
                'score_mean': float('nan'),
                'score_std': float('nan'),
                'fold_scores': [],
            }

        # =================================================================
        # Compute deltas (all sign-normalised: positive = improvement)
        # =================================================================
        sign = -1 if lower_is_better else 1
        delta_dc = sign * (d_mean - c_mean)               # in-sample upper bound
        delta_drc = sign * (d_random_mean - c_mean)       # selection-bias null
        delta_ddr = sign * (d_mean - d_random_mean)       # DEBIASED contribution
        delta_db = sign * (d_mean - best_b_mean) if not np.isinf(best_b_mean) else float('nan')

        def _interp(delta, pos, neg):
            if np.isnan(delta):
                return 'insufficient data'
            return pos if delta > 0 else neg

        rep_result['deltas'] = {
            # Biased upper bound — keep for transparency, never present as effect.
            'D_minus_C': float(delta_dc) if not np.isnan(delta_dc) else float('nan'),
            # The selection inflation any same-dim matrix earns.
            'D_random_minus_C': float(delta_drc) if not np.isnan(delta_drc) else float('nan'),
            # The debiased new-rep contribution — THIS is the headline effect.
            'D_minus_D_random': float(delta_ddr) if not np.isnan(delta_ddr) else float('nan'),
            # Does the hybrid improve the new rep beyond its own best single model?
            'D_minus_B': float(delta_db) if not np.isnan(delta_db) else float('nan'),
            'D_minus_D_random_interpretation': _interp(
                delta_ddr,
                'new rep HELPS hybrid beyond a same-dim noise matrix',
                'apparent gain is selection noise (rep no better than random dims)'),
            'D_minus_B_interpretation': _interp(
                delta_db,
                'hybrid HELPS the new rep beyond its best single model',
                'hybrid does NOT beat the new rep alone'),
        }

        print(f"\n  --- DELTAS (positive = improvement) ---")
        print(f"    D - C        (upper bound, biased):      {delta_dc:+.4f}")
        print(f"    D_random - C (selection-bias null):      {delta_drc:+.4f}")
        print(f"    D - D_random (DEBIASED new-rep gain):    {delta_ddr:+.4f}  "
              f"({rep_result['deltas']['D_minus_D_random_interpretation']})")
        print(f"    D - B        (hybrid helps new rep?):    {delta_db:+.4f}  "
              f"({rep_result['deltas']['D_minus_B_interpretation']})")

        # =================================================================
        # Controls: random Gaussian and row-shuffled
        # =================================================================
        print(f"\n  Controls for {rep_name}:")

        # Random Gaussian (same dims)
        X_random = _make_random_gaussian(X_new)
        ctrl_random_scores = _eval_single_rep_per_fold(
            X_random, labels, groups, ridge_factory, n_folds, mode, metric_fn)
        ctrl_random_mean = float(np.mean(ctrl_random_scores)) if ctrl_random_scores else float('nan')
        rep_result['controls']['random_gaussian'] = {
            'description': f'random Gaussian ({X_new.shape[1]}D)',
            'score_mean': ctrl_random_mean,
            'score_std': float(np.std(ctrl_random_scores)) if ctrl_random_scores else float('nan'),
            'fold_scores': [float(s) for s in ctrl_random_scores],
        }
        print(f"    Random Gaussian ({X_new.shape[1]}D): {metric_name} = {ctrl_random_mean:.4f}")

        # Row-shuffled
        X_shuffled = _make_row_shuffled(X_new)
        ctrl_shuffled_scores = _eval_single_rep_per_fold(
            X_shuffled, labels, groups, ridge_factory, n_folds, mode, metric_fn)
        ctrl_shuffled_mean = float(np.mean(ctrl_shuffled_scores)) if ctrl_shuffled_scores else float('nan')
        rep_result['controls']['row_shuffled'] = {
            'description': f'row-shuffled {rep_name}',
            'score_mean': ctrl_shuffled_mean,
            'score_std': float(np.std(ctrl_shuffled_scores)) if ctrl_shuffled_scores else float('nan'),
            'fold_scores': [float(s) for s in ctrl_shuffled_scores],
        }
        print(f"    Row-shuffled {rep_name}: {metric_name} = {ctrl_shuffled_mean:.4f}")

        # Signal above null (positive = rep is better than noise)
        if lower_is_better:
            signal_above_random = ctrl_random_mean - a_mean
            signal_above_shuffled = ctrl_shuffled_mean - a_mean
        else:
            signal_above_random = a_mean - ctrl_random_mean
            signal_above_shuffled = a_mean - ctrl_shuffled_mean
        rep_result['controls']['signal_above_random'] = float(signal_above_random)
        rep_result['controls']['signal_above_shuffled'] = float(signal_above_shuffled)
        print(f"    Signal above random: {signal_above_random:+.4f}")
        print(f"    Signal above shuffled: {signal_above_shuffled:+.4f}")

        per_rep_results[rep_name] = rep_result

    # =====================================================================
    # Cross-rep summary
    # =====================================================================
    print(f"\n{'='*70}")
    print("  FORWARD COMPATIBILITY SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Dataset: {dataset_name} | Metric: {metric_name} | "
          f"Domain: {domain} | n_folds: {n_folds}")
    print(f"  {'lower is better' if lower_is_better else 'higher is better'}")

    print(f"\n  {'New Rep':<20} {'Dim':>5} {'B(best)':>9} {'C(no r)':>9} "
          f"{'D(+r)':>9} {'Drand':>9} {'D-Dr*':>8} {'D-B':>8} {'Alloc%':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")

    # * D-Dr = D - D_random, the DEBIASED new-rep contribution. D - C is
    #   deliberately NOT in the win-rate summary: it is selection-biased
    #   positive and would overstate the effect.
    win_count_ddr = 0
    win_count_db = 0
    total_reps_tested = 0

    for rep_name, rr in sorted(per_rep_results.items()):
        conds = rr['conditions']
        deltas = rr['deltas']
        b_val = conds['B']['score_mean']
        c_val = conds['C']['score_mean']
        d_val = conds['D']['score_mean']
        dr_val = conds.get('D_random', {}).get('score_mean', float('nan'))
        ddr = deltas['D_minus_D_random']
        db = deltas['D_minus_B']
        alloc_pct = conds['D'].get('allocation_share', 0) * 100

        # Win counting on the DEBIASED effect. `ddr` and `db` are already
        # sign-normalised (positive = improvement, any metric direction), so
        # re-applying lower_is_better would invert RMSE/MAE/logloss domains.
        total_reps_tested += 1
        if not np.isnan(ddr) and ddr > 0:
            win_count_ddr += 1
        if not np.isnan(db) and db > 0:
            win_count_db += 1

        print(f"  {rep_name:<20} {rr['dim']:>5} {b_val:>9.4f} {c_val:>9.4f} "
              f"{d_val:>9.4f} {dr_val:>9.4f} {ddr:>+8.4f} {db:>+8.4f} "
              f"{alloc_pct:>6.1f}%")

    if total_reps_tested > 0:
        print(f"\n  Win rates (on the DEBIASED effect, not the biased D-C):")
        print(f"    D > D_random (rep beats same-dim noise): {win_count_ddr}/{total_reps_tested} "
              f"({win_count_ddr/total_reps_tested:.0%})")
        print(f"    D > B (hybrid helps the new rep):        {win_count_db}/{total_reps_tested} "
              f"({win_count_db/total_reps_tested:.0%})")

    # Controls summary
    print(f"\n  Controls:")
    print(f"  {'New Rep':<20} {'A(real)':>9} {'Gaussian':>9} {'Shuffled':>9} "
          f"{'vs Random':>10} {'vs Shuf':>10}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")
    for rep_name, rr in sorted(per_rep_results.items()):
        a_val = rr['conditions']['A']['score_mean']
        rand_val = rr['controls']['random_gaussian']['score_mean']
        shuf_val = rr['controls']['row_shuffled']['score_mean']
        sig_r = rr['controls']['signal_above_random']
        sig_s = rr['controls']['signal_above_shuffled']
        print(f"  {rep_name:<20} {a_val:>9.4f} {rand_val:>9.4f} {shuf_val:>9.4f} "
              f"{sig_r:>+10.4f} {sig_s:>+10.4f}")

    # Build output
    output = make_structured_output(
        experiment_id='c13_forward_compatibility',
        dataset=dataset_name,
        data=data,
        config=config,
        results={
            'per_rep': per_rep_results,
            'summary': {
                'new_reps': sorted(new_reps.keys()),
                'existing_reps': sorted(existing_reps.keys()),
                # Headline win rates are on the DEBIASED effect. D-C is retained
                # per-rep in deltas as an in-sample upper bound only.
                'win_rate_D_vs_D_random': win_count_ddr / total_reps_tested if total_reps_tested > 0 else None,
                'win_rate_D_vs_B': win_count_db / total_reps_tested if total_reps_tested > 0 else None,
                'total_new_reps_tested': total_reps_tested,
                'metric_note': (
                    'D-C is a selection-biased upper bound; the unbiased new-rep '
                    'effect is D - D_random. No significance tests are reported '
                    '(underpowered on non-independent CV folds).'
                ),
            },
        },
        primary_metric=metric_name,
        lower_is_better=lower_is_better,
    )

    return output


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = create_experiment_parser(
        'C13: Forward Compatibility — do new reps benefit from hybridization?',
        has_delta=True,
        has_augment=True,
    )
    parser.add_argument(
        '--new-reps', type=str, default=None,
        help='Comma-separated list of rep names to treat as "new". '
             'If not specified, auto-detect by comparing against LEGACY_REPS.',
    )
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode:    {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps:    {args.reps}")
    print(f"Models:  {config['models']}")
    print(f"Folds:   {config['n_splits']}")

    # Parse new-reps argument
    new_rep_names = None
    if args.new_reps:
        new_rep_names = [r.strip() for r in args.new_reps.split(',') if r.strip()]
        print(f"New reps (user-specified): {new_rep_names}")

    # Honor a comma-separated --reps list (e.g. "aac,blosum,onehot") as an
    # explicit rep_names override. Without this, load_data treats the whole
    # string as a preset name, falls through to rep_names=None, and then
    # auto-fills the FULL DATASET_REPS 'general' list (foundation models
    # included) — which is unloadable locally and defeats --reps entirely.
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset,
                     rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  new_rep_names=new_rep_names)
    save_experiment_results(results, args.output, 'c13_forward_compatibility',
                            args.dataset)
