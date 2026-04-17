#!/usr/bin/env python3
"""C13: Forward Compatibility — do new reps benefit from hybridization (and vice versa)?

Paper: Complexity
Conclusion: Quantifies the symbiotic value of new representations within the
    KIRBy hybrid framework. New reps can improve the hybrid ceiling, and the
    hybrid can improve a new rep beyond its best single-model performance.
Dependencies: None (standalone).

5-condition ablation per new rep r, per dataset d:
  A: New rep r + ridge           (raw embedding quality, linear)
  B: New rep r + best model      (best-case single-rep)
  C: Best hybrid WITHOUT r       (current KIRBy ceiling)
  D: Best hybrid WITH r          (does r improve the hybrid? D - C)
  E: r + existing reps in hybrid (does hybrid improve r? E - B)

Controls per new rep r:
  - Random Gaussian matrix (same dims as r): permutation null
  - Row-shuffled r: tests content vs dimensionality

Key metrics:
  1. Delta performance: D - C (does new rep help hybrid?)
  2. Augmentation gain: E - B (does hybrid help new rep?)
  3. Allocation share: features_from_r / total_budget
  4. Win rate: fraction of datasets where condition improves

Statistical tests (cite Demsar JMLR 2006):
  - Per-dataset: Wilcoxon signed-rank if enough folds
  - Cross-dataset: Friedman + Nemenyi if enough datasets

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
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split,
    create_hybrid, apply_feature_selection,
    evaluate_single_rep_cv, build_fold_hybrid,
    INTERNAL_EVALUATORS, SCREENING_BUDGETS,
)
from kirby.find_best_hybrid import find_best_hybrid


# =============================================================================
# LEGACY REPS — anything NOT in this set is considered "new"
# =============================================================================

# Molecular legacy reps (available before Phase 4 forward-compat additions)
_LEGACY_MOLECULAR = {
    'ecfp4', 'maccs', 'rdkit_desc', 'mordred', 'mordred_pca50', 'avalon',
    'mol2vec', 'map4', 'erg',
    # Pre-trained language models (pre-Phase 4)
    'molbert', 'chemberta',
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
                           metric_name='r2', dataset_name=None):
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

    try:
        search = find_best_hybrid(
            base_reps=reps,
            labels=labels,
            groups=groups,
            speed=search_speed,
            budgets=FULL_BUDGETS,
            augmentations=augmentations,
            return_fitted=True,
            internal_evaluators=INTERNAL_EVALUATORS,
            metric=metric_name,
            verbose=False,
            cache=True,
            dataset_name=dataset_name,
            tier='general',
        )
    except Exception as e:
        print(f"      find_best_hybrid failed: {e}")
        return [], {'error': str(e)}

    # Extract the best blueprint: pick highest-scoring model from best_by_model
    if not search.best_by_model:
        return [], {'error': 'no best_by_model found'}

    _lib = search.lower_is_better
    best = (min if _lib else max)(
        search.best_by_model.values(), key=lambda r: r['score_mean'])

    blueprint = best.get('blueprint')
    if blueprint is None:
        return [], {'error': 'no blueprint in best_by_model'}

    scores = []
    allocation_dicts = []
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
        'blueprint': {
            'allocation_method': blueprint.allocation_method,
            'importance_method': blueprint.importance_method,
            'budget': blueprint.budget,
            'internal_evaluator': blueprint.internal_evaluator,
        },
    }

    return scores, result_info


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def _wilcoxon_test(scores_a, scores_b):
    """Wilcoxon signed-rank test. Returns (statistic, p_value) or None if insufficient data."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    # Need at least 5 non-zero differences
    if np.sum(diff != 0) < 5:
        return None
    try:
        stat, pval = wilcoxon(a, b)
        return {'statistic': float(stat), 'p_value': float(pval)}
    except Exception:
        return None


def _friedman_test(score_matrix):
    """Friedman test across conditions. score_matrix: list of lists (datasets x conditions).

    Returns (statistic, p_value) or None if insufficient data.
    Cite: Demsar JMLR 2006.
    """
    matrix = np.array(score_matrix)
    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        return None
    try:
        stat, pval = friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
        return {'statistic': float(stat), 'p_value': float(pval)}
    except Exception:
        return None


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

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    augs = data.get('augmentations')
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
            'statistical_tests': {},
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
                dataset_name=f"{dataset_name}_without_{rep_name}")
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
                dataset_name=f"{dataset_name}_with_{rep_name}")
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
        # Condition E: r + existing reps in hybrid (does hybrid improve r?)
        # This is the same as D conceptually, but we report the metric
        # differently: E - B tells us if hybridization helps the new rep.
        # E uses the same scores as D (the hybrid includes r).
        # =================================================================
        print(f"\n  Condition E: hybrid improves {rep_name}?")
        # E is the same evaluation as D: the full hybrid including r.
        # The key question is E - B (does hybrid improve r beyond its best single-model?)
        e_scores = d_scores
        e_mean = d_mean
        rep_result['conditions']['E'] = {
            'description': f'{rep_name} in hybrid (same as D)',
            'score_mean': e_mean,
            'score_std': float(np.std(e_scores)) if e_scores else float('nan'),
            'fold_scores': [float(s) for s in e_scores],
        }

        # =================================================================
        # Compute deltas
        # =================================================================
        delta_dc_raw = d_mean - c_mean  # does new rep help hybrid?
        delta_eb_raw = e_mean - best_b_mean if not np.isinf(best_b_mean) else float('nan')
        # Sign-normalize: positive = improvement regardless of metric direction
        sign = -1 if lower_is_better else 1
        delta_dc = sign * delta_dc_raw
        delta_eb = sign * delta_eb_raw

        rep_result['deltas'] = {
            'D_minus_C': float(delta_dc) if not np.isnan(delta_dc) else float('nan'),
            'E_minus_B': float(delta_eb) if not np.isnan(delta_eb) else float('nan'),
            'D_minus_C_interpretation': (
                'new rep HELPS hybrid' if delta_dc > 0
                else 'new rep HURTS or neutral for hybrid'
            ) if not np.isnan(delta_dc) else 'insufficient data',
            'E_minus_B_interpretation': (
                'hybrid HELPS new rep' if delta_eb > 0
                else 'hybrid HURTS or neutral for new rep'
            ) if not np.isnan(delta_eb) else 'insufficient data',
        }

        print(f"\n  --- DELTAS ---")
        print(f"    D - C (new rep helps hybrid?):   {delta_dc:+.4f}  "
              f"({rep_result['deltas']['D_minus_C_interpretation']})")
        print(f"    E - B (hybrid helps new rep?):   {delta_eb:+.4f}  "
              f"({rep_result['deltas']['E_minus_B_interpretation']})")

        # =================================================================
        # Statistical tests (Wilcoxon signed-rank, per-dataset)
        # =================================================================
        if len(d_scores) >= 5 and len(c_scores) >= 5:
            # Match fold count
            n_match = min(len(d_scores), len(c_scores))
            test_dc = _wilcoxon_test(d_scores[:n_match], c_scores[:n_match])
            if test_dc:
                rep_result['statistical_tests']['wilcoxon_D_vs_C'] = test_dc
                print(f"    Wilcoxon D vs C: p={test_dc['p_value']:.4f}")

        if len(e_scores) >= 5 and len(best_b_scores) >= 5:
            n_match = min(len(e_scores), len(best_b_scores))
            test_eb = _wilcoxon_test(e_scores[:n_match], best_b_scores[:n_match])
            if test_eb:
                rep_result['statistical_tests']['wilcoxon_E_vs_B'] = test_eb
                print(f"    Wilcoxon E vs B: p={test_eb['p_value']:.4f}")

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

    print(f"\n  {'New Rep':<20} {'Dim':>5} {'A(ridge)':>9} {'B(best)':>9} "
          f"{'C(no r)':>9} {'D(+r)':>9} {'D-C':>8} {'E-B':>8} {'Alloc%':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")

    win_count_dc = 0
    win_count_eb = 0
    total_reps_tested = 0

    for rep_name, rr in sorted(per_rep_results.items()):
        conds = rr['conditions']
        deltas = rr['deltas']
        a_val = conds['A']['score_mean']
        b_val = conds['B']['score_mean']
        c_val = conds['C']['score_mean']
        d_val = conds['D']['score_mean']
        dc = deltas['D_minus_C']
        eb = deltas['E_minus_B']
        alloc_pct = conds['D'].get('allocation_share', 0) * 100

        # Win counting
        total_reps_tested += 1
        if not np.isnan(dc):
            if (not lower_is_better and dc > 0) or (lower_is_better and dc < 0):
                win_count_dc += 1
        if not np.isnan(eb):
            if (not lower_is_better and eb > 0) or (lower_is_better and eb < 0):
                win_count_eb += 1

        print(f"  {rep_name:<20} {rr['dim']:>5} {a_val:>9.4f} {b_val:>9.4f} "
              f"{c_val:>9.4f} {d_val:>9.4f} {dc:>+8.4f} {eb:>+8.4f} "
              f"{alloc_pct:>6.1f}%")

    if total_reps_tested > 0:
        print(f"\n  Win rates:")
        print(f"    D > C (new rep helps hybrid):  {win_count_dc}/{total_reps_tested} "
              f"({win_count_dc/total_reps_tested:.0%})")
        print(f"    E > B (hybrid helps new rep):  {win_count_eb}/{total_reps_tested} "
              f"({win_count_eb/total_reps_tested:.0%})")

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
                'win_rate_D_vs_C': win_count_dc / total_reps_tested if total_reps_tested > 0 else None,
                'win_rate_E_vs_B': win_count_eb / total_reps_tested if total_reps_tested > 0 else None,
                'total_new_reps_tested': total_reps_tested,
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

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  new_rep_names=new_rep_names)
    save_experiment_results(results, args.output, 'c13_forward_compatibility',
                            args.dataset)
