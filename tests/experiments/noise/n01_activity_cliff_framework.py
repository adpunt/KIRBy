#!/usr/bin/env python3
"""N01: Activity Cliff Framework -- Dablander-inspired cliff analysis.

Paper: Noise / Robustness
Conclusion: Hybrid representations produce more consensus cliff identification,
    smoother SALI landscapes, fewer spurious cliffs, and smaller cliff vs
    non-cliff prediction gaps.

Key insight (Bajorath): activity cliffs are REPRESENTATION-DEPENDENT -- different
fingerprints identify different compounds as cliffs. KIRBy's hybrid should produce
more "consensus" cliff identification because it combines structural information
from multiple representations.

Four sub-analyses:
  1. Consensus cliff identification (Jaccard overlap of cliff sets across reps)
  2. SALI landscape smoothing (mean/max/std of per-sample max-SALI)
  3. Cliff mitigation by hybridization (cliff pair counts before vs after)
  4. Cliff-stratified prediction error (RMSE/R2 on cliff vs non-cliff subsets)

Usage:
    python n01_activity_cliff_framework.py --dataset logd --reps all --mode full
    python n01_activity_cliff_framework.py --dataset logd --sim-threshold 0.55 --activity-threshold 2.0
    python n01_activity_cliff_framework.py --dataset flab_thermo_tresanco --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split, make_cv_splitter,
)

from kirby.hybrid import create_hybrid, apply_feature_selection, compute_feature_importance
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# =============================================================================
# CLIFF DETECTION HELPERS
# =============================================================================

def _compute_ecfp4_tanimoto_sim(smiles_list):
    """Compute ECFP4 Tanimoto similarity matrix from SMILES.

    Returns (n, n) similarity matrix with values in [0, 1].
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Fallback: zero vector (will have Tc=0 with everything)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles('C'), 2, nBits=2048))
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    # Convert to numpy for fast Jaccard via scipy
    fp_array = np.zeros((len(fps), 2048), dtype=np.float32)
    for i, fp in enumerate(fps):
        arr = np.zeros(2048, dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        fp_array[i] = arr

    # Tanimoto = 1 - Jaccard distance for binary vectors
    jaccard_dist = scipy_cdist(fp_array, fp_array, metric='jaccard')
    tanimoto_sim = 1.0 - jaccard_dist
    np.fill_diagonal(tanimoto_sim, 0.0)  # exclude self-pairs
    return tanimoto_sim


def _compute_euclidean_sim(X, standardize=True):
    """Compute similarity matrix from Euclidean distance: sim = 1/(1+d).

    Used for non-molecular domains (antibody, DTA) where SMILES are unavailable.
    Returns (n, n) similarity matrix with values in (0, 1].
    """
    D = compute_distance_matrix(X, standardize=standardize)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    return sim


def identify_cliff_pairs(sim_matrix, y, sim_threshold, activity_k):
    """Identify activity cliff pairs from a similarity matrix.

    Args:
        sim_matrix: (n, n) similarity matrix (0 = dissimilar, 1 = identical).
        y: (n,) activity values.
        sim_threshold: minimum similarity to count as "structurally similar".
        activity_k: activity difference threshold in MAD units.

    Returns:
        cliff_pair_mask: (n, n) boolean, True where (i,j) is a cliff pair.
        info: dict with thresholds and counts.
    """
    n = len(y)
    mad_y = float(np.median(np.abs(y - np.median(y))))
    activity_threshold = activity_k * mad_y

    delta_y = np.abs(y[:, None] - y[None, :])
    cliff_pair_mask = (sim_matrix >= sim_threshold) & (delta_y > activity_threshold)
    np.fill_diagonal(cliff_pair_mask, False)

    cliff_mask = cliff_pair_mask.any(axis=1)
    n_cliff_pairs = int(cliff_pair_mask.sum()) // 2  # symmetric
    n_cliff_compounds = int(cliff_mask.sum())

    info = {
        'sim_threshold': sim_threshold,
        'activity_k': activity_k,
        'activity_threshold': float(activity_threshold),
        'mad_y': float(mad_y),
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / n) if n > 0 else 0.0,
    }
    return cliff_pair_mask, info


def _cliff_pair_set(cliff_pair_mask):
    """Convert boolean cliff pair mask to a set of frozensets for Jaccard."""
    pairs = set()
    ii, jj = np.where(np.triu(cliff_pair_mask, k=1))
    for i, j in zip(ii, jj):
        pairs.add(frozenset((int(i), int(j))))
    return pairs


def _jaccard_index(set_a, set_b):
    """Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# SALI COMPUTATION
# =============================================================================

def compute_sali_stats(D, y, eps=1e-6):
    """Compute SALI landscape statistics from a distance matrix.

    SALI(i,j) = |y_i - y_j| / (1 - sim(i,j)), where sim = 1/(1+d).

    Returns dict with mean, max, std, median of per-sample max-SALI.
    """
    n = len(y)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)

    delta_y = np.abs(y[:, None] - y[None, :])
    denom = 1.0 - sim + eps
    sali_matrix = delta_y / denom
    np.fill_diagonal(sali_matrix, 0.0)

    per_sample_max = sali_matrix.max(axis=1)

    return {
        'mean': float(np.mean(per_sample_max)),
        'max': float(np.max(per_sample_max)),
        'std': float(np.std(per_sample_max)),
        'median': float(np.median(per_sample_max)),
    }


# =============================================================================
# SUB-ANALYSIS 1: CONSENSUS CLIFF IDENTIFICATION
# =============================================================================

def _run_consensus(reps, labels, sim_threshold, activity_k, domain, smiles=None):
    """Compute cliff sets per rep and pairwise Jaccard overlap."""
    print("\n  --- Sub-analysis 1: Consensus Cliff Identification ---")

    cliff_sets = {}  # rep_name -> set of frozenset pairs
    cliff_infos = {}

    for rep_name, X_rep in reps.items():
        # Build similarity matrix for this rep (Euclidean-based)
        D = compute_distance_matrix(X_rep, standardize=True)
        sim = 1.0 / (1.0 + D)
        np.fill_diagonal(sim, 0.0)

        mask, info = identify_cliff_pairs(sim, labels, sim_threshold, activity_k)
        cliff_sets[rep_name] = _cliff_pair_set(mask)
        cliff_infos[rep_name] = info
        print(f"    {rep_name:<20} cliffs={info['n_cliff_pairs']:>5} "
              f"({info['pct_cliff_compounds']:.1f}% compounds)")

    # Pairwise Jaccard between all rep pairs
    rep_names = sorted(cliff_sets.keys())
    n_reps = len(rep_names)
    jaccard_matrix = np.zeros((n_reps, n_reps))
    pairwise_jaccards = []

    for i, j in combinations(range(n_reps), 2):
        jac = _jaccard_index(cliff_sets[rep_names[i]], cliff_sets[rep_names[j]])
        jaccard_matrix[i, j] = jac
        jaccard_matrix[j, i] = jac
        pairwise_jaccards.append({
            'rep_a': rep_names[i],
            'rep_b': rep_names[j],
            'jaccard': float(jac),
        })

    np.fill_diagonal(jaccard_matrix, 1.0)
    avg_jaccard = float(np.mean([p['jaccard'] for p in pairwise_jaccards])) if pairwise_jaccards else 0.0

    print(f"    Average pairwise Jaccard: {avg_jaccard:.4f}")
    print(f"    Min Jaccard: {min(p['jaccard'] for p in pairwise_jaccards):.4f}" if pairwise_jaccards else "")
    print(f"    Max Jaccard: {max(p['jaccard'] for p in pairwise_jaccards):.4f}" if pairwise_jaccards else "")

    return {
        'per_rep_cliff_counts': {k: v['n_cliff_pairs'] for k, v in cliff_infos.items()},
        'per_rep_cliff_pct': {k: v['pct_cliff_compounds'] for k, v in cliff_infos.items()},
        'avg_pairwise_jaccard': avg_jaccard,
        'pairwise_jaccards': pairwise_jaccards,
    }


# =============================================================================
# SUB-ANALYSIS 2: SALI LANDSCAPE SMOOTHING
# =============================================================================

def _run_sali_smoothing(reps, labels, X_hybrid):
    """Compare SALI landscape smoothness across reps and hybrid."""
    print("\n  --- Sub-analysis 2: SALI Landscape Smoothing ---")

    sali_results = {}

    for rep_name, X_rep in reps.items():
        D = compute_distance_matrix(X_rep, standardize=True)
        stats = compute_sali_stats(D, labels)
        sali_results[rep_name] = stats
        print(f"    {rep_name:<20} SALI mean={stats['mean']:.3f}  "
              f"max={stats['max']:.3f}  std={stats['std']:.3f}")

    # Hybrid SALI
    if X_hybrid is not None:
        D_hybrid = compute_distance_matrix(X_hybrid, standardize=True)
        stats_hybrid = compute_sali_stats(D_hybrid, labels)
        sali_results['_hybrid'] = stats_hybrid
        print(f"    {'_hybrid':<20} SALI mean={stats_hybrid['mean']:.3f}  "
              f"max={stats_hybrid['max']:.3f}  std={stats_hybrid['std']:.3f}")

        # Compare: hybrid vs average of individual reps
        rep_means = [v['mean'] for k, v in sali_results.items() if k != '_hybrid']
        rep_maxes = [v['max'] for k, v in sali_results.items() if k != '_hybrid']
        rep_stds = [v['std'] for k, v in sali_results.items() if k != '_hybrid']

        sali_results['_comparison'] = {
            'avg_rep_mean': float(np.mean(rep_means)),
            'avg_rep_max': float(np.mean(rep_maxes)),
            'avg_rep_std': float(np.mean(rep_stds)),
            'hybrid_mean': stats_hybrid['mean'],
            'hybrid_max': stats_hybrid['max'],
            'hybrid_std': stats_hybrid['std'],
            'delta_mean': float(stats_hybrid['mean'] - np.mean(rep_means)),
            'delta_max': float(stats_hybrid['max'] - np.mean(rep_maxes)),
        }
        print(f"    Hybrid vs avg-rep SALI mean: "
              f"{stats_hybrid['mean']:.3f} vs {np.mean(rep_means):.3f} "
              f"(delta={sali_results['_comparison']['delta_mean']:+.3f})")

    return sali_results


# =============================================================================
# SUB-ANALYSIS 3: CLIFF MITIGATION BY HYBRIDIZATION
# =============================================================================

def _run_cliff_mitigation(reps, labels, X_hybrid, sim_threshold, activity_k):
    """Count cliff pairs in each rep space vs hybrid space at same threshold."""
    print("\n  --- Sub-analysis 3: Cliff Mitigation by Hybridization ---")

    mitigation = {}

    for rep_name, X_rep in reps.items():
        D = compute_distance_matrix(X_rep, standardize=True)
        sim = 1.0 / (1.0 + D)
        np.fill_diagonal(sim, 0.0)
        mask, info = identify_cliff_pairs(sim, labels, sim_threshold, activity_k)
        mitigation[rep_name] = {
            'n_cliff_pairs': info['n_cliff_pairs'],
            'n_cliff_compounds': info['n_cliff_compounds'],
            'pct_cliff_compounds': info['pct_cliff_compounds'],
        }

    # Hybrid
    if X_hybrid is not None:
        D_hybrid = compute_distance_matrix(X_hybrid, standardize=True)
        sim_hybrid = 1.0 / (1.0 + D_hybrid)
        np.fill_diagonal(sim_hybrid, 0.0)
        mask_h, info_h = identify_cliff_pairs(sim_hybrid, labels, sim_threshold, activity_k)
        mitigation['_hybrid'] = {
            'n_cliff_pairs': info_h['n_cliff_pairs'],
            'n_cliff_compounds': info_h['n_cliff_compounds'],
            'pct_cliff_compounds': info_h['pct_cliff_compounds'],
        }

        # Summary
        rep_pair_counts = [v['n_cliff_pairs'] for k, v in mitigation.items()
                          if k != '_hybrid']
        avg_rep = float(np.mean(rep_pair_counts)) if rep_pair_counts else 0.0
        mitigation['_comparison'] = {
            'avg_rep_cliff_pairs': avg_rep,
            'hybrid_cliff_pairs': info_h['n_cliff_pairs'],
            'reduction_vs_avg': float(avg_rep - info_h['n_cliff_pairs']),
            'reduction_pct': float(100.0 * (avg_rep - info_h['n_cliff_pairs']) / avg_rep)
            if avg_rep > 0 else 0.0,
        }

        print(f"    Per-rep cliff pairs (avg): {avg_rep:.0f}")
        print(f"    Hybrid cliff pairs:        {info_h['n_cliff_pairs']}")
        print(f"    Reduction: {mitigation['_comparison']['reduction_vs_avg']:.0f} pairs "
              f"({mitigation['_comparison']['reduction_pct']:.1f}%)")
    else:
        print("    (hybrid not available, skipping comparison)")

    # Print per-rep
    for rep_name in sorted(k for k in mitigation if not k.startswith('_')):
        v = mitigation[rep_name]
        print(f"    {rep_name:<20} pairs={v['n_cliff_pairs']:>5}  "
              f"compounds={v['n_cliff_compounds']:>4} ({v['pct_cliff_compounds']:.1f}%)")

    return mitigation


# =============================================================================
# SUB-ANALYSIS 4: CLIFF-STRATIFIED PREDICTION ERROR
# =============================================================================

def _run_cliff_stratified(reps, labels, groups, config, data,
                          sim_threshold, activity_k,
                          hybrid_blueprint, smiles=None):
    """Compare prediction error on cliff vs non-cliff compounds."""
    print("\n  --- Sub-analysis 4: Cliff-Stratified Prediction Error ---")

    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    # Identify cliff compounds using a reference similarity.
    # For molecular: ECFP4 Tanimoto if SMILES available, else Euclidean on concat.
    # For antibody/DTA: Euclidean on concatenated reps.
    n = len(labels)
    if smiles is not None and domain == 'molecular':
        try:
            ref_sim = _compute_ecfp4_tanimoto_sim(smiles)
            sim_source = 'ecfp4_tanimoto'
        except Exception as e:
            print(f"    WARNING: ECFP4 Tanimoto failed ({e}), falling back to Euclidean")
            X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
            ref_sim = _compute_euclidean_sim(X_concat)
            sim_source = 'euclidean_concat'
    else:
        X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
        ref_sim = _compute_euclidean_sim(X_concat)
        sim_source = 'euclidean_concat'

    cliff_pair_mask, cliff_info = identify_cliff_pairs(
        ref_sim, labels, sim_threshold, activity_k)
    cliff_mask = cliff_pair_mask.any(axis=1)
    cliff_indices = set(np.where(cliff_mask)[0].tolist())

    print(f"    Cliff source: {sim_source}")
    print(f"    Cliff compounds: {cliff_info['n_cliff_compounds']} / {n} "
          f"({cliff_info['pct_cliff_compounds']:.1f}%)")
    print(f"    Cliff pairs: {cliff_info['n_cliff_pairs']}")

    if cliff_info['n_cliff_compounds'] < 5:
        print("    WARNING: too few cliff compounds for meaningful stratification")
        return {
            'cliff_source': sim_source,
            'cliff_info': cliff_info,
            'error': 'too_few_cliff_compounds',
        }

    splitter, use_groups = make_cv_splitter(n_folds, groups)
    sample_X = next(iter(reps.values()))

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    def _evaluate_condition(get_Xy_fn, label):
        """Run CV, return {metric_all, metric_cliff, metric_noncliff, rmse_*}."""
        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_all, fold_rmse_cliff, fold_rmse_noncliff = [], [], []
        fold_domain_all = []

        split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)
        for train_idx, val_idx in splitter.split(*split_args):
            y_train, y_val = labels[train_idx], labels[val_idx]
            try:
                X_tr, X_va = get_Xy_fn(train_idx, val_idx, y_train)
            except Exception:
                continue
            if X_tr is None:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            model = RandomForestRegressor(n_estimators=100, random_state=42,
                                          n_jobs=-1)
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_va_s)

            fold_all.append(r2_score(y_val, y_pred))
            fold_rmse_all.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
            fold_domain_all.append(metric_fn(y_val, y_pred))

            val_cliff = np.array([i for i, vi in enumerate(val_idx)
                                  if vi in cliff_indices])
            val_noncliff = np.array([i for i, vi in enumerate(val_idx)
                                     if vi not in cliff_indices])

            if len(val_cliff) > 1:
                fold_cliff.append(r2_score(y_val[val_cliff], y_pred[val_cliff]))
                fold_rmse_cliff.append(float(np.sqrt(
                    mean_squared_error(y_val[val_cliff], y_pred[val_cliff]))))
            if len(val_noncliff) > 1:
                fold_noncliff.append(r2_score(y_val[val_noncliff],
                                              y_pred[val_noncliff]))
                fold_rmse_noncliff.append(float(np.sqrt(
                    mean_squared_error(y_val[val_noncliff],
                                       y_pred[val_noncliff]))))

        if not fold_all:
            return None

        r2_c = float(np.mean(fold_cliff)) if fold_cliff else float('nan')
        r2_nc = float(np.mean(fold_noncliff)) if fold_noncliff else float('nan')
        rmse_c = float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan')
        rmse_nc = float(np.mean(fold_rmse_noncliff)) if fold_rmse_noncliff else float('nan')

        return {
            'r2_all': float(np.mean(fold_all)),
            'r2_cliff': r2_c,
            'r2_noncliff': r2_nc,
            'rmse_all': float(np.mean(fold_rmse_all)),
            'rmse_cliff': rmse_c,
            'rmse_noncliff': rmse_nc,
            f'{metric_name}_all': float(np.mean(fold_domain_all)) if fold_domain_all else float('nan'),
            'cliff_noncliff_gap_r2': float(r2_nc - r2_c)
            if not (np.isnan(r2_c) or np.isnan(r2_nc)) else float('nan'),
            'cliff_noncliff_gap_rmse': float(rmse_c - rmse_nc)
            if not (np.isnan(rmse_c) or np.isnan(rmse_nc)) else float('nan'),
        }

    strat_results = {}

    # Per-rep RF baselines
    print(f"\n    Per-rep RF baselines:")
    for rep_name, X_rep in reps.items():
        def _get_rep(tr, va, yt, _X=X_rep):
            return _X[tr], _X[va]
        res = _evaluate_condition(_get_rep, rep_name)
        if res:
            strat_results[rep_name] = res
            print(f"      {rep_name:<20} R2_all={res['r2_all']:.4f}  "
                  f"R2_cliff={res['r2_cliff']:.4f}  R2_noncliff={res['r2_noncliff']:.4f}  "
                  f"gap={res['cliff_noncliff_gap_r2']:.4f}")

    # Hybrid RF
    if hybrid_blueprint is not None:
        print(f"\n    Hybrid RF:")

        def _get_hybrid(tr, va, yt):
            train_reps = {name: rep[tr] for name, rep in reps.items()}
            val_reps = {name: rep[va] for name, rep in reps.items()}
            X_tr_h, feat_info = create_hybrid(
                train_reps, yt,
                allocation_method=hybrid_blueprint.allocation_method,
                budget=hybrid_blueprint.budget,
                importance_method=hybrid_blueprint.importance_method,
            )
            X_va_h = apply_feature_selection(val_reps, feat_info)
            return X_tr_h, X_va_h

        res = _evaluate_condition(_get_hybrid, 'hybrid')
        if res:
            strat_results['_hybrid'] = res
            print(f"      {'_hybrid':<20} R2_all={res['r2_all']:.4f}  "
                  f"R2_cliff={res['r2_cliff']:.4f}  R2_noncliff={res['r2_noncliff']:.4f}  "
                  f"gap={res['cliff_noncliff_gap_r2']:.4f}")

        # Comparison
        rep_gaps = [v['cliff_noncliff_gap_r2'] for k, v in strat_results.items()
                    if k != '_hybrid' and not np.isnan(v.get('cliff_noncliff_gap_r2', float('nan')))]
        if rep_gaps and '_hybrid' in strat_results:
            avg_rep_gap = float(np.mean(rep_gaps))
            hybrid_gap = strat_results['_hybrid']['cliff_noncliff_gap_r2']
            strat_results['_comparison'] = {
                'avg_rep_gap_r2': avg_rep_gap,
                'hybrid_gap_r2': hybrid_gap,
                'gap_reduction': float(avg_rep_gap - hybrid_gap),
            }
            print(f"      Gap reduction: {avg_rep_gap:.4f} -> {hybrid_gap:.4f} "
                  f"(delta={avg_rep_gap - hybrid_gap:+.4f})")

    return {
        'cliff_source': sim_source,
        'cliff_info': convert_numpy(cliff_info),
        'stratified': convert_numpy(strat_results),
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Run the full Dablander activity cliff framework.

    Four sub-analyses:
      1. Consensus cliff identification
      2. SALI landscape smoothing
      3. Cliff mitigation by hybridization
      4. Cliff-stratified prediction error
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N01: ACTIVITY CLIFF FRAMEWORK (Dablander)")
    print("=" * 70)

    sim_threshold = kwargs.get('sim_threshold', 0.55)
    activity_threshold = kwargs.get('activity_threshold', 2.0)
    search_speed = config.get('search_speed', 'fast')

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    smiles = data.get('smiles', None)

    n = len(labels)
    print(f"  Dataset: {dataset_name} ({domain}, {n} samples, {len(reps)} reps)")
    print(f"  Sim threshold: {sim_threshold}  Activity threshold: {activity_threshold} MAD")

    # --- Build hybrid for sub-analyses 2, 3, 4 ---
    hybrid_blueprint = None
    X_hybrid = None

    try:
        from experiments._shared import get_hybrid_features_and_weights
        X_hybrid, weights_flat, search = get_hybrid_features_and_weights(
            reps, labels, groups, search_speed, 'lightgbm_gain')
        if search.best_overall:
            hybrid_blueprint = search.best_overall['blueprint']
        print(f"  Hybrid: dim={X_hybrid.shape[1]}")
    except Exception as e:
        print(f"  WARNING: Hybrid search failed ({e}), hybrid comparisons skipped")

    results = {
        'config': {
            'sim_threshold': sim_threshold,
            'activity_threshold': activity_threshold,
            'domain': domain,
            'n_samples': n,
            'n_reps': len(reps),
        },
    }

    # --- Sub-analysis 1: Consensus cliff identification ---
    results['consensus'] = _run_consensus(
        reps, labels, sim_threshold, activity_threshold, domain, smiles)

    # Add hybrid consensus if available
    if X_hybrid is not None:
        # Build hybrid cliff set
        D_hybrid = compute_distance_matrix(X_hybrid, standardize=True)
        sim_hybrid = 1.0 / (1.0 + D_hybrid)
        np.fill_diagonal(sim_hybrid, 0.0)
        mask_h, _ = identify_cliff_pairs(sim_hybrid, labels, sim_threshold,
                                         activity_threshold)
        hybrid_cliff_set = _cliff_pair_set(mask_h)

        # Jaccard of hybrid vs each rep
        hybrid_jaccards = {}
        for rep_name, X_rep in reps.items():
            D_rep = compute_distance_matrix(X_rep, standardize=True)
            sim_rep = 1.0 / (1.0 + D_rep)
            np.fill_diagonal(sim_rep, 0.0)
            mask_r, _ = identify_cliff_pairs(sim_rep, labels, sim_threshold,
                                             activity_threshold)
            rep_cliff_set = _cliff_pair_set(mask_r)
            jac = _jaccard_index(hybrid_cliff_set, rep_cliff_set)
            hybrid_jaccards[rep_name] = float(jac)

        avg_hybrid_jaccard = float(np.mean(list(hybrid_jaccards.values()))) \
            if hybrid_jaccards else 0.0
        results['consensus']['hybrid_vs_rep_jaccards'] = hybrid_jaccards
        results['consensus']['avg_hybrid_vs_rep_jaccard'] = avg_hybrid_jaccard
        results['consensus']['avg_pairwise_jaccard_reps_only'] = \
            results['consensus']['avg_pairwise_jaccard']

        print(f"    Hybrid avg Jaccard vs individual reps: {avg_hybrid_jaccard:.4f}")
        print(f"    (vs rep-rep average: "
              f"{results['consensus']['avg_pairwise_jaccard']:.4f})")

    # --- Sub-analysis 2: SALI landscape smoothing ---
    results['sali'] = _run_sali_smoothing(reps, labels, X_hybrid)

    # --- Sub-analysis 3: Cliff mitigation by hybridization ---
    results['mitigation'] = _run_cliff_mitigation(
        reps, labels, X_hybrid, sim_threshold, activity_threshold)

    # --- Sub-analysis 4: Cliff-stratified prediction error ---
    results['stratified'] = _run_cliff_stratified(
        reps, labels, groups, config, data,
        sim_threshold, activity_threshold,
        hybrid_blueprint, smiles)

    # --- Grand summary ---
    print("\n" + "=" * 70)
    print("  GRAND SUMMARY")
    print("=" * 70)

    # Consensus
    if 'consensus' in results:
        c = results['consensus']
        print(f"  1. Consensus: avg rep-rep Jaccard = {c['avg_pairwise_jaccard']:.4f}")
        if 'avg_hybrid_vs_rep_jaccard' in c:
            print(f"     Hybrid vs reps Jaccard = {c['avg_hybrid_vs_rep_jaccard']:.4f}")

    # SALI
    if 'sali' in results and '_comparison' in results['sali']:
        s = results['sali']['_comparison']
        print(f"  2. SALI smoothing: hybrid mean={s['hybrid_mean']:.3f} vs "
              f"avg-rep={s['avg_rep_mean']:.3f} (delta={s['delta_mean']:+.3f})")

    # Mitigation
    if 'mitigation' in results and '_comparison' in results['mitigation']:
        m = results['mitigation']['_comparison']
        print(f"  3. Cliff mitigation: {m['avg_rep_cliff_pairs']:.0f} -> "
              f"{m['hybrid_cliff_pairs']} pairs "
              f"({m['reduction_pct']:.1f}% reduction)")

    # Stratified
    if 'stratified' in results:
        st = results['stratified']
        if isinstance(st, dict) and 'stratified' in st and '_comparison' in st['stratified']:
            comp = st['stratified']['_comparison']
            print(f"  4. Cliff-stratified gap: avg-rep={comp['avg_rep_gap_r2']:.4f}, "
                  f"hybrid={comp['hybrid_gap_r2']:.4f} "
                  f"(reduction={comp['gap_reduction']:+.4f})")

    return make_structured_output(
        experiment_id='n01',
        dataset=dataset_name,
        data=data,
        config=results.pop('config'),
        results=results,
        primary_metric='cliff_noncliff_gap_r2',
        lower_is_better=True,
    )


if __name__ == '__main__':
    parser = create_experiment_parser('N01: Activity Cliff Framework (Dablander)',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--sim-threshold', type=float, default=0.55,
                        help='Similarity threshold for cliff detection '
                             '(default: 0.55 for ECFP4 Tanimoto-equivalent)')
    parser.add_argument('--activity-threshold', type=float, default=2.0,
                        help='Activity difference threshold in MAD units '
                             '(default: 2.0)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Sim threshold: {args.sim_threshold}")
    print(f"Activity threshold: {args.activity_threshold}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  sim_threshold=args.sim_threshold,
                  activity_threshold=args.activity_threshold)
    save_experiment_results(results, args.output,
                            'n01_activity_cliff_framework', args.dataset)
