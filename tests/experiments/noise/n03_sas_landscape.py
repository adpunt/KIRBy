#!/usr/bin/env python3
"""N03: SAS Landscape Analysis -- 4-quadrant maps (Shanmugasundaram & Maggiora 2001).

Paper: Noise / Similarity bridge
Conclusion: Hybrid representations shift the structure-activity landscape from
    Zone I (activity cliffs) toward Zone IV (smooth SAR), improving GP-based
    methods that assume landscape smoothness.

References:
    - Shanmugasundaram & Maggiora (2001) J. Med. Chem. 44(21):3515-22
    - Guha (2012) Curr. Comput.-Aided Drug Des. 8(3):223-36
    - Moesser (2023) PhD thesis, Ch. 2 (Oxford)

Activity similarity: SimAct(A,B) = 1 - |Act(A)-Act(B)| / (Act_max - Act_min)
Structural similarity:
  - Canonical (molecular): ECFP4 Tanimoto
  - Per-rep / hybrid: cosine similarity rescaled to [0,1] = (1 + cos) / 2

Zone definitions (thresholds t_s=0.5, t_a=0.5):
  Zone I   (activity cliffs):  struct_sim >= t_s AND act_sim < t_a
  Zone II  (nondescript):      struct_sim < t_s  AND act_sim < t_a
  Zone III (scaffold hops):    struct_sim < t_s  AND act_sim >= t_a
  Zone IV  (smooth SAR):       struct_sim >= t_s AND act_sim >= t_a

4 sub-analyses:
  1. Per-rep SAS maps (cosine sim) + canonical ECFP4 baseline (molecular)
  2. Hybrid SAS map + zone population shift vs individual reps
  3. Cross-rep landscape: zone populations vs dim / MI / erank
  4. SAS-BO bridge: Zone I fraction predicts GP quality in that space

Usage:
    python n03_sas_landscape.py --dataset logd --reps all --mode full
    python n03_sas_landscape.py --dataset caco2 --reps all --mode intermediary
    python n03_sas_landscape.py --dataset flab_thermo_tresanco --reps all --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    compute_effective_rank, estimate_mutual_information_knn,
    get_domain_metric, get_hybrid_features_and_weights,
)

from kirby.hybrid import create_hybrid
# TODO: create_hybrid API not finalised — revisit after framework experiments.

# Maximum compounds for pairwise analysis (O(N^2) memory)
MAX_COMPOUNDS_SAS = 5000
# Maximum compounds for GP evaluation (O(N^3) compute)
MAX_COMPOUNDS_GP = 500


# =============================================================================
# CORE SAS COMPUTATION
# =============================================================================

def compute_activity_similarity(y):
    """Activity similarity per Shanmugasundaram & Maggiora (2001).

    SimAct(A,B) = 1 - |Act(A) - Act(B)| / (Act_max - Act_min)

    Returns (n, n) matrix with values in [0, 1].
    """
    y_range = float(np.max(y) - np.min(y))
    if y_range < 1e-10:
        return np.ones((len(y), len(y)), dtype=np.float32)
    delta_y = np.abs(y[:, None] - y[None, :])
    act_sim = 1.0 - delta_y / y_range
    np.fill_diagonal(act_sim, 1.0)
    return act_sim.astype(np.float32)


def compute_cosine_sim(X):
    """Cosine similarity rescaled to [0, 1]: (1 + cos) / 2.

    Threshold at 0.5 = orthogonal vectors (cos=0), a natural boundary.
    Returns (n, n) matrix with values in [0, 1].
    """
    cos_sim = cosine_similarity(X)
    sim = (1.0 + cos_sim) / 2.0
    sim = np.nan_to_num(sim, nan=0.5)  # handle zero-norm rows
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def compute_ecfp4_tanimoto_sim(smiles_list):
    """ECFP4 Tanimoto similarity (canonical structural similarity for molecular).

    Returns (n, n) matrix with values in [0, 1].
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles('C'), 2, nBits=2048))
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    fp_array = np.zeros((len(fps), 2048), dtype=np.float32)
    for i, fp in enumerate(fps):
        for bit in fp.GetOnBits():
            fp_array[i, bit] = 1.0

    jaccard_dist = scipy_cdist(fp_array, fp_array, metric='jaccard')
    tanimoto_sim = 1.0 - jaccard_dist
    np.fill_diagonal(tanimoto_sim, 1.0)
    return tanimoto_sim.astype(np.float32)


def compute_sas_zones(struct_sim, act_sim, struct_t=0.5, act_t=0.5):
    """Count pairs in each SAS zone (upper triangle only, no self-pairs).

    Returns dict with zone counts and percentages.
    """
    n = struct_sim.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)

    ss = struct_sim[triu_i, triu_j]
    aa = act_sim[triu_i, triu_j]
    n_pairs = len(ss)

    zone_I = int(np.sum((ss >= struct_t) & (aa < act_t)))
    zone_II = int(np.sum((ss < struct_t) & (aa < act_t)))
    zone_III = int(np.sum((ss < struct_t) & (aa >= act_t)))
    zone_IV = int(np.sum((ss >= struct_t) & (aa >= act_t)))

    return {
        'zone_I': zone_I,   'zone_II': zone_II,
        'zone_III': zone_III, 'zone_IV': zone_IV,
        'n_pairs': n_pairs,
        'pct_I':   float(100.0 * zone_I   / n_pairs) if n_pairs else 0.0,
        'pct_II':  float(100.0 * zone_II  / n_pairs) if n_pairs else 0.0,
        'pct_III': float(100.0 * zone_III / n_pairs) if n_pairs else 0.0,
        'pct_IV':  float(100.0 * zone_IV  / n_pairs) if n_pairs else 0.0,
    }


def _subsample_if_needed(reps, labels, smiles, max_n):
    """Subsample compounds if N > max_n for O(N^2) pairwise analysis."""
    n = len(labels)
    if n <= max_n:
        return reps, labels, smiles, np.arange(n)

    rng = np.random.RandomState(42)
    idx = rng.choice(n, max_n, replace=False)
    idx.sort()

    sub_reps = {k: v[idx] for k, v in reps.items()}
    sub_labels = labels[idx]
    sub_smiles = [smiles[i] for i in idx] if smiles is not None else None
    return sub_reps, sub_labels, sub_smiles, idx


def _print_zones(label, zones, dim=None):
    """Format a zone-population line."""
    dim_str = f"(d={dim:>5}) " if dim is not None else ""
    print(f"    {label:<22} {dim_str}"
          f"I={zones['pct_I']:5.2f}%  II={zones['pct_II']:5.2f}%  "
          f"III={zones['pct_III']:5.2f}%  IV={zones['pct_IV']:5.2f}%")


# =============================================================================
# SUB-ANALYSIS 1: PER-REP SAS MAPS
# =============================================================================

def _run_per_rep_sas(reps, labels, smiles, struct_t, act_t, domain):
    """SAS zones for each rep + canonical ECFP4 (molecular only)."""
    print("\n  --- Sub-analysis 1: Per-Rep SAS Maps ---")

    act_sim = compute_activity_similarity(labels)
    results = {}

    # Canonical ECFP4 Tanimoto (molecular only)
    if smiles is not None and domain == 'molecular':
        try:
            ecfp4_sim = compute_ecfp4_tanimoto_sim(smiles)
            zones = compute_sas_zones(ecfp4_sim, act_sim, struct_t, act_t)
            results['_ecfp4_canonical'] = {
                'sim_type': 'ecfp4_tanimoto',
                'zones': zones,
            }
            _print_zones('ECFP4 (canonical)', zones)
        except Exception as e:
            print(f"    WARNING: ECFP4 Tanimoto failed ({e})")

    # Per-rep cosine similarity
    for rep_name, X_rep in reps.items():
        X_s = StandardScaler().fit_transform(X_rep)
        struct_sim = compute_cosine_sim(X_s)
        zones = compute_sas_zones(struct_sim, act_sim, struct_t, act_t)
        results[rep_name] = {
            'dim': X_rep.shape[1],
            'sim_type': 'cosine',
            'zones': zones,
        }
        _print_zones(rep_name, zones, dim=X_rep.shape[1])

    return results, act_sim


# =============================================================================
# SUB-ANALYSIS 2: HYBRID SAS MAP + ZONE SHIFT
# =============================================================================

def _run_hybrid_sas(reps, labels, X_hybrid, act_sim, struct_t, act_t,
                    per_rep_results):
    """Hybrid SAS zones and comparison to individual reps."""
    print("\n  --- Sub-analysis 2: Hybrid SAS Map + Zone Shift ---")

    if X_hybrid is None:
        print("    (hybrid not available, skipping)")
        return None

    X_s = StandardScaler().fit_transform(X_hybrid)
    struct_sim = compute_cosine_sim(X_s)
    zones = compute_sas_zones(struct_sim, act_sim, struct_t, act_t)
    _print_zones('hybrid', zones, dim=X_hybrid.shape[1])

    # Per-rep averages (exclude canonical ECFP4 entry)
    rep_zone_I = [v['zones']['pct_I'] for k, v in per_rep_results.items()
                  if not k.startswith('_')]
    rep_zone_IV = [v['zones']['pct_IV'] for k, v in per_rep_results.items()
                   if not k.startswith('_')]
    avg_rep_I = float(np.mean(rep_zone_I)) if rep_zone_I else 0.0
    avg_rep_IV = float(np.mean(rep_zone_IV)) if rep_zone_IV else 0.0

    # Best individual rep (lowest Zone I)
    rep_keys = [k for k in per_rep_results if not k.startswith('_')]
    best_rep = min(rep_keys,
                   key=lambda k: per_rep_results[k]['zones']['pct_I'],
                   default=None) if rep_keys else None
    best_rep_I = per_rep_results[best_rep]['zones']['pct_I'] if best_rep else 0.0

    comparison = {
        'avg_rep_zone_I': avg_rep_I,
        'avg_rep_zone_IV': avg_rep_IV,
        'hybrid_zone_I': zones['pct_I'],
        'hybrid_zone_IV': zones['pct_IV'],
        'delta_zone_I_vs_avg': float(zones['pct_I'] - avg_rep_I),
        'delta_zone_IV_vs_avg': float(zones['pct_IV'] - avg_rep_IV),
        'delta_zone_I_vs_best': float(zones['pct_I'] - best_rep_I),
        'best_rep': best_rep,
        'best_rep_zone_I': best_rep_I,
    }

    print(f"\n    Comparison (Zone I = cliffs, Zone IV = smooth):")
    print(f"      Avg rep Zone I:    {avg_rep_I:5.2f}%")
    print(f"      Hybrid Zone I:     {zones['pct_I']:5.2f}% "
          f"(delta={comparison['delta_zone_I_vs_avg']:+.2f}%)")
    print(f"      Best rep ({best_rep}) Zone I: {best_rep_I:5.2f}%")
    print(f"      Hybrid vs best:    delta={comparison['delta_zone_I_vs_best']:+.2f}%")
    print(f"      Avg rep Zone IV:   {avg_rep_IV:5.2f}%")
    print(f"      Hybrid Zone IV:    {zones['pct_IV']:5.2f}% "
          f"(delta={comparison['delta_zone_IV_vs_avg']:+.2f}%)")

    return {
        'dim': X_hybrid.shape[1],
        'zones': zones,
        'comparison': comparison,
    }


# =============================================================================
# SUB-ANALYSIS 3: CROSS-REP LANDSCAPE COMPARISON
# =============================================================================

def _run_landscape_comparison(reps, labels, per_rep_results):
    """Correlate zone populations with rep properties (dim, MI, erank)."""
    print("\n  --- Sub-analysis 3: Cross-Rep Landscape Comparison ---")

    rep_names = [k for k in per_rep_results if not k.startswith('_')]
    if len(rep_names) < 3:
        print("    (need >= 3 reps for correlation analysis)")
        return None

    props = []
    for rname in rep_names:
        X = reps[rname]
        dim = X.shape[1]

        try:
            erank = compute_effective_rank(X)
        except Exception:
            erank = float('nan')
        try:
            mi = estimate_mutual_information_knn(X, labels)
        except Exception:
            mi = float('nan')

        zone_I = per_rep_results[rname]['zones']['pct_I']
        zone_IV = per_rep_results[rname]['zones']['pct_IV']

        props.append({
            'rep': rname, 'dim': dim, 'erank': erank, 'mi': mi,
            'zone_I_pct': zone_I, 'zone_IV_pct': zone_IV,
        })

    # Table
    print(f"    {'Rep':<18} {'Dim':>5} {'Erank':>7} {'MI':>7} "
          f"{'ZoneI%':>7} {'ZoneIV%':>8}")
    print(f"    {'-'*18} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for p in props:
        print(f"    {p['rep']:<18} {p['dim']:>5} {p['erank']:>7.1f} "
              f"{p['mi']:>7.3f} {p['zone_I_pct']:>7.2f} {p['zone_IV_pct']:>8.2f}")

    # Spearman correlations
    correlations = {}
    zone_Is = np.array([p['zone_I_pct'] for p in props])
    zone_IVs = np.array([p['zone_IV_pct'] for p in props])

    for prop_name in ('dim', 'erank', 'mi'):
        vals = np.array([p[prop_name] for p in props], dtype=float)
        valid = ~np.isnan(vals) & ~np.isnan(zone_Is)
        if valid.sum() >= 3:
            rho_I, p_I = spearmanr(vals[valid], zone_Is[valid])
            rho_IV, p_IV = spearmanr(vals[valid], zone_IVs[valid])
            correlations[prop_name] = {
                'rho_zone_I': float(rho_I), 'p_zone_I': float(p_I),
                'rho_zone_IV': float(rho_IV), 'p_zone_IV': float(p_IV),
            }

    if correlations:
        print(f"\n    Spearman correlations with zone populations:")
        for pn, c in correlations.items():
            print(f"      {pn:>6} -> Zone I: rho={c['rho_zone_I']:+.3f} (p={c['p_zone_I']:.3f})"
                  f"   Zone IV: rho={c['rho_zone_IV']:+.3f} (p={c['p_zone_IV']:.3f})")

    return {
        'per_rep_properties': props,
        'correlations': correlations,
    }


# =============================================================================
# SUB-ANALYSIS 4: SAS-BO BRIDGE
# =============================================================================

def _run_sas_bo_bridge(reps, labels, struct_t, act_t,
                       augmentations=None, augmentation_strategy='uncapped'):
    """Zone I fraction predicts GP quality -- explains why compact spaces help BO.

    Builds 4 representation conditions matching S11, computes SAS zones and
    GP quality in each. Shows correlation: fewer cliffs -> better GP fit.
    """
    print("\n  --- Sub-analysis 4: SAS-BO Bridge ---")

    n = len(labels)

    # Subsample for GP (O(N^3))
    rng = np.random.RandomState(42)
    if n > MAX_COMPOUNDS_GP:
        gp_idx = rng.choice(n, MAX_COMPOUNDS_GP, replace=False)
        gp_idx.sort()
        print(f"    GP subsample: {MAX_COMPOUNDS_GP} / {n}")
    else:
        gp_idx = np.arange(n)

    # Train/val split on subsample
    n_gp = len(gp_idx)
    perm = np.random.RandomState(123).permutation(n_gp)
    n_train = int(0.8 * n_gp)
    train_local, val_local = perm[:n_train], perm[n_train:]
    train_idx = gp_idx[train_local]
    val_idx = gp_idx[val_local]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # ---- Build 4 representation conditions (matching S11) ----
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    scaler_concat = StandardScaler()
    X_concat_s = scaler_concat.fit_transform(X_concat)

    conditions = {}

    # 1. High-D raw
    if 'ecfp4' in reps:
        conditions['ecfp4_raw'] = StandardScaler().fit_transform(reps['ecfp4'])
    else:
        highest = max(reps.keys(), key=lambda k: reps[k].shape[1])
        conditions[f'{highest}_raw'] = StandardScaler().fit_transform(
            reps[highest])

    # 2. PCA 50D
    n_comp = min(50, n - 1, X_concat_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    conditions['pca_50d'] = pca.fit_transform(X_concat_s)

    # 3. KIRBy hybrid 50D
    try:
        X_h, _ = create_hybrid(
            reps, labels,
            allocation_method='entropy_1.0',
            importance_method='lightgbm_gain',
            budget=50,
            augmentations=augmentations,
            augmentation_strategy=augmentation_strategy,
        )
        conditions['hybrid_50d'] = StandardScaler().fit_transform(X_h)
    except Exception as e:
        print(f"    WARNING: hybrid build failed ({e})")

    # 4. Random projection 50D
    d = X_concat.shape[1]
    proj = np.random.RandomState(42).randn(d, min(50, d)) / np.sqrt(50)
    conditions['random_50d'] = X_concat_s @ proj

    # ---- Evaluate each condition ----
    act_sim_sub = compute_activity_similarity(labels[gp_idx])

    results = {}
    for cond_name, X_full in conditions.items():
        X_sub = X_full[gp_idx]
        dim = X_sub.shape[1]

        # SAS zones
        struct_sim = compute_cosine_sim(X_sub)
        zones = compute_sas_zones(struct_sim, act_sim_sub, struct_t, act_t)

        # GP quality
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2,
            alpha=1e-6, normalize_y=True, random_state=42,
        )
        try:
            gp.fit(X_tr, y_train)
            mu, sigma = gp.predict(X_va, return_std=True)
            gp_r2 = float(r2_score(y_val, mu))
            gp_mean_std = float(np.mean(sigma))
            gp_log_ml = float(gp.log_marginal_likelihood_value_)
            gp_success = True
        except (np.linalg.LinAlgError, ValueError):
            gp_r2 = float('nan')
            gp_mean_std = float('nan')
            gp_log_ml = float('nan')
            gp_success = False

        results[cond_name] = {
            'dim': dim,
            'zones': zones,
            'gp_r2': gp_r2,
            'gp_mean_std': gp_mean_std,
            'gp_log_marginal_likelihood': gp_log_ml,
            'gp_success': gp_success,
        }

    # ---- Print table ----
    print(f"\n    {'Condition':<18} {'Dim':>5} {'ZoneI%':>7} {'ZoneIV%':>8} "
          f"{'GP R2':>8} {'GP std':>8} {'GP logML':>9}")
    print(f"    {'-'*18} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
    for cond_name in ['ecfp4_raw', 'pca_50d', 'hybrid_50d', 'random_50d']:
        if cond_name not in results:
            # Handle renamed high-D condition
            for k in results:
                if k.endswith('_raw') and k != 'ecfp4_raw':
                    cond_name = k
                    break
        if cond_name not in results:
            continue
        r = results[cond_name]
        z = r['zones']
        print(f"    {cond_name:<18} {r['dim']:>5} {z['pct_I']:>7.2f} "
              f"{z['pct_IV']:>8.2f} {r['gp_r2']:>8.4f} "
              f"{r['gp_mean_std']:>8.4f} {r['gp_log_ml']:>9.1f}")

    # ---- Correlation: Zone I vs GP R2 ----
    successful = {k: v for k, v in results.items()
                  if v['gp_success'] and not k.startswith('_')}
    if len(successful) >= 3:
        zone_Is = [v['zones']['pct_I'] for v in successful.values()]
        gp_r2s = [v['gp_r2'] for v in successful.values()]
        rho, p = spearmanr(zone_Is, gp_r2s)
        results['_correlation'] = {
            'zone_I_vs_gp_r2_rho': float(rho),
            'zone_I_vs_gp_r2_p': float(p),
        }
        print(f"\n    Zone I% -> GP R2: rho={rho:.3f} (p={p:.3f})")
        if rho < -0.3:
            print(f"    -> Fewer cliffs -> better GP fit -> better BO")

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """N03: SAS Landscape Analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N03: SAS LANDSCAPE ANALYSIS")
    print("  Shanmugasundaram & Maggiora (2001) 4-quadrant maps")
    print("=" * 70)

    struct_t = kwargs.get('struct_threshold', 0.5)
    act_t = kwargs.get('act_threshold', 0.5)
    search_speed = config.get('search_speed', 'fast')

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    smiles = data.get('smiles', None)

    n = len(labels)
    print(f"  Dataset: {dataset_name} ({domain}, {n} samples, {len(reps)} reps)")
    print(f"  Struct threshold: {struct_t}  Activity threshold: {act_t}")

    # Subsample for pairwise analysis if needed
    sub_reps, sub_labels, sub_smiles, sub_idx = _subsample_if_needed(
        reps, labels, smiles, MAX_COMPOUNDS_SAS)
    if len(sub_labels) < n:
        print(f"  Subsampled to {len(sub_labels)} compounds for pairwise analysis")

    results = {
        'config': {
            'struct_threshold': struct_t,
            'act_threshold': act_t,
            'domain': domain,
            'n_samples': n,
            'n_samples_sas': len(sub_labels),
            'n_reps': len(reps),
        },
    }

    # --- Sub-analysis 1: Per-rep SAS maps ---
    per_rep_results, act_sim = _run_per_rep_sas(
        sub_reps, sub_labels, sub_smiles, struct_t, act_t, domain)
    results['per_rep'] = per_rep_results

    # --- Build hybrid for sub-analyses 2-3 ---
    X_hybrid = None
    try:
        X_hybrid_full, _, search = get_hybrid_features_and_weights(
            reps, labels, groups, search_speed, 'lightgbm_gain')
        X_hybrid = X_hybrid_full[sub_idx] if len(sub_idx) < n else X_hybrid_full
        print(f"  Hybrid: dim={X_hybrid.shape[1]}")
    except Exception as e:
        print(f"  WARNING: Hybrid search failed ({e})")

    # --- Sub-analysis 2: Hybrid SAS map ---
    results['hybrid'] = _run_hybrid_sas(
        sub_reps, sub_labels, X_hybrid, act_sim,
        struct_t, act_t, per_rep_results)

    # --- Sub-analysis 3: Cross-rep landscape comparison ---
    results['landscape'] = _run_landscape_comparison(
        sub_reps, sub_labels, per_rep_results)

    # --- Sub-analysis 4: SAS-BO bridge ---
    augmentations = data.get('augmentations')
    augmentation_strategy = 'uncapped'
    results['bo_bridge'] = _run_sas_bo_bridge(
        reps, labels, struct_t, act_t,
        augmentations=augmentations,
        augmentation_strategy=augmentation_strategy)

    # --- Grand summary ---
    print("\n" + "=" * 70)
    print("  GRAND SUMMARY")
    print("=" * 70)

    if results.get('hybrid') and results['hybrid'] is not None:
        c = results['hybrid'].get('comparison', {})
        if c:
            print(f"  Zone I (cliffs):  avg rep {c['avg_rep_zone_I']:.2f}% -> "
                  f"hybrid {c['hybrid_zone_I']:.2f}% "
                  f"(delta={c['delta_zone_I_vs_avg']:+.2f}%)")
            print(f"  Zone IV (smooth): avg rep {c['avg_rep_zone_IV']:.2f}% -> "
                  f"hybrid {c['hybrid_zone_IV']:.2f}% "
                  f"(delta={c['delta_zone_IV_vs_avg']:+.2f}%)")

    if results.get('bo_bridge') and '_correlation' in results['bo_bridge']:
        corr = results['bo_bridge']['_correlation']
        print(f"  SAS-BO link: Zone I% -> GP R2 rho="
              f"{corr['zone_I_vs_gp_r2_rho']:.3f}")

    return make_structured_output(
        experiment_id='n03',
        dataset=dataset_name,
        data=data,
        config=results.pop('config'),
        results=results,
        primary_metric='zone_I_pct',
        lower_is_better=True,
    )


if __name__ == '__main__':
    parser = create_experiment_parser('N03: SAS Landscape Analysis',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--struct-threshold', type=float, default=0.5,
                        help='Structural similarity threshold for zone '
                             'boundaries (default: 0.5)')
    parser.add_argument('--act-threshold', type=float, default=0.5,
                        help='Activity similarity threshold for zone '
                             'boundaries (default: 0.5)')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"Struct threshold: {args.struct_threshold}")
    print(f"Activity threshold: {args.act_threshold}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  struct_threshold=args.struct_threshold,
                  act_threshold=args.act_threshold)
    save_experiment_results(results, args.output,
                            'n03_sas_landscape', args.dataset)
