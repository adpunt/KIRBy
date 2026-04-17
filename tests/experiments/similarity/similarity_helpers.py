"""Shared configuration and runner for per-dataset similarity experiments.

Each dataset gets its own script (e.g. herg.py, logd.py) that calls
run_all() with its rep list from the pipeline readiness tiers.
Cross-dataset experiment (S09) stays in its own file and consumes
the per-dataset JSON outputs.

Config: RF / LightGBM / MLP, 1-fold scaffold CV, fast search.
Rep lists reused from complexity_helpers.DATASET_REPS.
"""

import sys
import os
import time
from pathlib import Path

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))         # tests/experiments/

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist as scipy_cdist

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy,
    load_experiment_results,
    DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
)

# Reuse dataset rep configs — single source of truth
from experiments.complexity.complexity_helpers import DATASET_REPS


# =============================================================================
# SHARED SIMILARITY METRIC FUNCTIONS
# =============================================================================

def _is_nominal(y):
    """Detect if labels are nominal multi-class (>2 discrete classes).

    DEPRECATED for direct use — prefer passing is_clf=True explicitly.
    Only reliable on the FULL label array, not individual elements.
    """
    unique = np.unique(y)
    return len(unique) > 2 and np.allclose(y, y.astype(int))


def _pairwise_delta(y_i, y_j, is_clf=False):
    """Compute pairwise activity difference.

    Regression / binary: |y_i - y_j| (continuous difference).
    Classification (is_clf=True): 1{y_i != y_j} (binary indicator).
    """
    if is_clf:
        return (y_i != y_j).astype(float)
    return np.abs(y_i - y_j)


def _pairwise_delta_matrix(y, is_clf=False):
    """Compute full n×n pairwise activity difference matrix."""
    if is_clf:
        return (y[:, None] != y[None, :]).astype(float)
    return np.abs(y[:, None] - y[None, :])


def similarity_activity_correlation(D, y, n_sample=5000, is_clf=False):
    """Spearman correlation between pairwise distance and activity difference.

    Regression / binary: |delta_y| (continuous).
    Classification (is_clf): 1{y_i != y_j} — do nearby samples share a class?

    Positive = good (large distance -> large activity difference means the
    representation preserves the structure-activity relationship).
    Returns (rho, p_value).
    """
    n = len(y)
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = len(triu_i)

    if n_pairs > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_pairs, n_sample, replace=False)
        triu_i, triu_j = triu_i[idx], triu_j[idx]

    dists = D[triu_i, triu_j]
    delta_y = _pairwise_delta(y[triu_i], y[triu_j], is_clf=is_clf)

    rho, p = spearmanr(dists, delta_y)
    return float(rho), float(p)


def compute_sali(D, y, eps=1e-6, is_clf=False):
    """SALI: activity_diff / (1 - sim(i,j)), sim = 1/(1+d).

    Classification (is_clf): uses binary indicator (same/different class).
    Returns per-sample max SALI and summary stats.
    """
    n = len(y)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0)

    delta_y = _pairwise_delta_matrix(y, is_clf=is_clf)
    denom = 1.0 - sim + eps
    sali_matrix = delta_y / denom
    np.fill_diagonal(sali_matrix, 0)

    per_sample_max = sali_matrix.max(axis=1)
    per_sample_mean = sali_matrix.mean(axis=1)

    return {
        'mean': float(np.mean(per_sample_max)),
        'max': float(np.max(per_sample_max)),
        'std': float(np.std(per_sample_max)),
        'median': float(np.median(per_sample_max)),
        'per_sample_mean_avg': float(np.mean(per_sample_mean)),
    }


def violation_rate(D, y, k=5, is_clf=False):
    """Fraction of k-NN pairs with large activity difference.

    Regression / binary: |delta_y| > median(|delta_y|).
    Classification (is_clf): fraction of k-NN that are a different class.
    """
    from kirby.utils.noise_mitigation_methods import get_k_neighbors_from_distances

    n = len(y)
    k = min(k, n - 1)

    indices, _ = get_k_neighbors_from_distances(D, k)

    if is_clf:
        # Multi-class: fraction of k-NN that belong to a different class
        n_violations = 0
        n_total = 0
        for i in range(n):
            for j_idx in range(k):
                j = indices[i, j_idx]
                if y[i] != y[j]:
                    n_violations += 1
                n_total += 1
        return float(n_violations / n_total) if n_total > 0 else 0.0

    # Regression / binary: threshold-based
    triu_i, triu_j = np.triu_indices(n, k=1)
    if len(triu_i) > 50000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(triu_i), 50000, replace=False)
        all_delta = np.abs(y[triu_i[sample_idx]] - y[triu_j[sample_idx]])
    else:
        all_delta = np.abs(y[triu_i] - y[triu_j])
    median_delta = np.median(all_delta)

    n_violations = 0
    n_total = 0
    for i in range(n):
        for j_idx in range(k):
            j = indices[i, j_idx]
            if np.abs(y[i] - y[j]) > median_delta:
                n_violations += 1
            n_total += 1

    return float(n_violations / n_total) if n_total > 0 else 0.0


def identify_cliff_compounds_editdist(sequences, y, max_edits=5, activity_k=2.0,
                                      is_clf=False):
    """Identify activity cliff compounds using pairwise edit distance (Hamming).

    Representation-independent cliff definition for protein/antibody data.
    Uses sequence Hamming distance (on aligned sequences of equal length)
    or Levenshtein distance (if lengths differ).

    Definition: cliff = few mutations (edits <= max_edits) but large activity
    difference (|delta_y| > activity_k * MAD(y)).

    Args:
        sequences: list of str (aligned sequences, e.g., heavy chains).
        y: array of activity values.
        max_edits: Maximum edit distance to be considered "structurally similar".
        activity_k: Multiplier for MAD-based activity threshold.
        is_clf: If True, cliff = few edits + different class.

    Returns:
        cliff_mask: boolean array, True for cliff compounds.
        info: dict with thresholds, counts, per-compound cliff pair counts.
    """
    n = len(y)
    y = np.asarray(y)

    # Pairwise edit distance (Hamming if equal length, else Levenshtein)
    same_len = len(set(len(s) for s in sequences)) == 1
    if same_len:
        # Fast Hamming: vectorised character comparison
        arr = np.array([list(s) for s in sequences])
        edit_dist = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            edit_dist[i] = (arr != arr[i]).sum(axis=1)
    else:
        # True Levenshtein edit distance (slower)
        def _levenshtein(s1, s2):
            if len(s1) < len(s2):
                return _levenshtein(s2, s1)
            prev = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                curr = [i + 1]
                for j, c2 in enumerate(s2):
                    curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                                    prev[j] + (0 if c1 == c2 else 1)))
                prev = curr
            return prev[-1]

        edit_dist = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                d = _levenshtein(sequences[i], sequences[j])
                edit_dist[i, j] = d
                edit_dist[j, i] = d

    # Activity difference
    delta_y = _pairwise_delta_matrix(y, is_clf=is_clf)

    if is_clf:
        activity_threshold = 0.5
        mad_y = 0.0
    else:
        mad_y = float(np.median(np.abs(y - np.median(y))))
        activity_threshold = activity_k * mad_y

    # Cliff: few edits AND large activity difference
    cliff_pair_mask = (edit_dist <= max_edits) & (edit_dist > 0) & (delta_y > activity_threshold)
    cliff_mask = cliff_pair_mask.any(axis=1)
    cliff_pair_counts = cliff_pair_mask.sum(axis=1)
    n_cliff_pairs = int(cliff_pair_mask.sum()) // 2
    n_cliff_compounds = int(cliff_mask.sum())

    info = {
        'method': 'edit_distance',
        'max_edits': max_edits,
        'activity_k': activity_k,
        'activity_threshold': activity_threshold,
        'mad_y': mad_y,
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / n),
        'cliff_pair_counts': cliff_pair_counts,
        'mean_edit_dist': float(edit_dist[np.triu_indices(n, k=1)].mean()),
        'median_edit_dist': float(np.median(edit_dist[np.triu_indices(n, k=1)])),
    }

    return cliff_mask, info


def identify_cliff_compounds(ecfp4, y, tanimoto_threshold=0.55, activity_k=2.0,
                             is_clf=False):
    """Identify activity cliff compounds using ECFP4 Tanimoto.

    Definition based on Bajorath standard for ECFP4 (Stumpfe & Bajorath 2019):
      - Structural similarity: Tanimoto(ECFP4) >= tanimoto_threshold (default 0.55)
      - Activity difference: |delta_y| > activity_k * MAD(y)

    Classification (is_clf): cliff = similar structure but different class.

    Returns:
        cliff_mask: boolean array, True for cliff compounds
        info: dict with thresholds, counts, per-compound cliff pair counts
    """
    n = len(y)

    # Tanimoto similarity from binary ECFP4 (Tanimoto = 1 - Jaccard distance)
    jaccard_dist = scipy_cdist(ecfp4, ecfp4, metric='jaccard')
    tanimoto_sim = 1.0 - jaccard_dist
    np.fill_diagonal(tanimoto_sim, 0)

    # Activity difference matrix
    delta_y = _pairwise_delta_matrix(y, is_clf=is_clf)

    if is_clf:
        # Multi-class: cliff = high similarity AND different class
        activity_threshold = 0.5  # indicator is 0 or 1
        mad_y = 0.0
    else:
        # Regression: adaptive threshold k * MAD(y)
        mad_y = float(np.median(np.abs(y - np.median(y))))
        activity_threshold = activity_k * mad_y

    # Cliff pair mask: high similarity AND large activity difference
    cliff_pair_mask = (tanimoto_sim >= tanimoto_threshold) & (delta_y > activity_threshold)

    # Cliff compounds: involved in at least one cliff pair
    cliff_mask = cliff_pair_mask.any(axis=1)

    # Per-compound cliff pair count
    cliff_pair_counts = cliff_pair_mask.sum(axis=1)

    n_cliff_pairs = int(cliff_pair_mask.sum()) // 2  # symmetric
    n_cliff_compounds = int(cliff_mask.sum())

    info = {
        'tanimoto_threshold': tanimoto_threshold,
        'activity_k': activity_k,
        'activity_threshold': activity_threshold,
        'mad_y': mad_y,
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / n),
        'cliff_pair_counts': cliff_pair_counts,
    }

    return cliff_mask, info


# =============================================================================
# ANTIBODY SEQUENCE CLIFF ANALYSIS (classification-aware edit-distance sweep)
# =============================================================================
#
# References for threshold choices and precedent:
#   Mason et al. 2021, Nat Biomed Eng — trastuzumab CDR-H3 DMS, Hamming ≤ 1/2/3
#     neighborhoods with Hamming ≤ 2 as local-cliff default. Closest analog
#     for binary binder/non-binder flip analysis.
#   Adams et al. 2019, eLife — Ab-14 DMS, single/double mutant epistasis at
#     Hamming 1 and 2.
#   Phillips et al. 2023, eLife — CR9114/CR6261 DMS, pairs at Hamming 1 for
#     epistasis, ≤ 2 for local landscape.
#
# Caveat: classification cliffs are under-formalized for antibodies. Hamming
# on full V-region is length-biased; for variable-length CDR-H3 see Gupta 2017
# normalized Levenshtein. Cite Mason 2021 as the primary methodological anchor.

def _cdr_cache_path(dataset_name):
    """Path for cached IMGT CDR residue indices per antibody dataset."""
    from pathlib import Path
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return cache_root / 'cdr_indices' / f'{dataset_name}.pkl'


def _extract_cdr_sequences(heavy_seqs, dataset_name=None):
    """Extract CDR-only substrings from heavy chains using ANARCI (IMGT).

    Results are cached per dataset under ~/.cache/kirby/cdr_indices/ since
    ANARCI is slow (~seconds per sequence). Returns list of CDR-concatenated
    strings (CDR1+CDR2+CDR3). Returns None if ANARCI unavailable or fails.
    """
    import pickle
    cache_path = _cdr_cache_path(dataset_name) if dataset_name else None
    if cache_path and cache_path.exists():
        try:
            return pickle.loads(cache_path.read_bytes())
        except Exception:
            pass

    try:
        from kirby.representations.antibody import get_cdr_residue_indices
    except Exception:
        return None

    cdr_seqs = []
    failures = 0
    for seq in heavy_seqs:
        try:
            idxs = get_cdr_residue_indices(seq)
            if not idxs:
                failures += 1
                cdr_seqs.append(seq)  # fallback to full sequence for this one
                continue
            cdr_seqs.append(''.join(seq[i] for i in idxs))
        except Exception:
            failures += 1
            cdr_seqs.append(seq)

    if failures > 0.5 * len(heavy_seqs):
        # More than half failed — likely ANARCI not installed or broken.
        # Signal failure to caller so they can skip the CDR scope entirely.
        return None

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(pickle.dumps(cdr_seqs))
        except Exception:
            pass  # cache write failure is non-fatal

    return cdr_seqs


def antibody_sequence_cliff_sweep(sequences, labels, is_clf,
                                  max_edits_sweep=(1, 2, 3, 5),
                                  max_compounds=3000, scope_label='full_V',
                                  activity_k=2.0):
    """Run edit-distance cliff detection across multiple Hamming thresholds.

    For classification: cliff = pair within Hamming <= max_edits but with
    different class labels (binder/non-binder flip, polyreactive/specific
    flip, etc.).

    Args:
        sequences: list of str (heavy chains, or CDR-concatenated strings).
        labels: array of activity or class labels.
        is_clf: True for classification datasets.
        max_edits_sweep: Hamming thresholds to report (sensitivity sweep).
        max_compounds: subsample cap — n² Hamming matrix grows fast.
        scope_label: 'full_V' or 'cdr_only' — for reporting/logging only.
        activity_k: MAD multiplier for regression cliff threshold (unused
            for classification; see identify_cliff_compounds_editdist).

    Returns:
        dict: {
            'scope': scope_label,
            'n_analyzed': int,
            'thresholds': {max_edits: {
                'n_cliff_pairs': int,
                'n_cliff_compounds': int,
                'pct_cliff_compounds': float,
                'mean_edit_dist': float,
                'median_edit_dist': float,
            }},
            'notes': str,  # interpretation caveats
        }
    """
    n = len(sequences)
    y = np.asarray(labels)

    if n > max_compounds:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_compounds, replace=False)
        sequences = [sequences[i] for i in idx]
        y = y[idx]
        n = max_compounds

    result = {
        'scope': scope_label,
        'n_analyzed': n,
        'thresholds': {},
    }

    for max_edits in max_edits_sweep:
        try:
            _, info = identify_cliff_compounds_editdist(
                sequences, y, max_edits=max_edits, activity_k=activity_k,
                is_clf=is_clf)
            # Drop the per-compound count array (too large for JSON).
            info_clean = {k: v for k, v in info.items()
                          if k not in ('cliff_pair_counts',)}
            result['thresholds'][int(max_edits)] = info_clean
        except Exception as e:
            result['thresholds'][int(max_edits)] = {'error': str(e)}

    if is_clf:
        result['notes'] = (
            "Classification cliff = pair within Hamming <= max_edits but "
            "with different class labels. Binary indicator is coarser than "
            "continuous |delta_y|, so interpretation should be qualitative "
            "(cliff density, flip frequency) rather than quantitative. "
            "Primary anchor: Mason et al. 2021 (trastuzumab CDR-H3 DMS). "
            "See also Phillips 2023, Adams 2019."
        )
    else:
        result['notes'] = (
            "Regression cliff = pair within Hamming <= max_edits with "
            "|delta_y| > activity_k * MAD(y). Anchor: Mason 2021 DMS."
        )

    return result


def run_antibody_cliff_analysis(data, is_clf, dataset_name=None,
                                max_edits_sweep=(1, 2, 3, 5)):
    """Driver: run sequence cliff sweep at full V-region and CDR-only scopes.

    Args:
        data: dict from load_data(). Must contain 'labels' and either
            'heavy_seqs' (antibody) or 'sequences' (nucleotide).
        is_clf: True for classification datasets.
        dataset_name: Optional dataset name for CDR cache keying. If provided,
            extracted CDR indices are cached under ~/.cache/kirby/cdr_indices/
            so subsequent runs skip ANARCI. Falls back to `data.get('dataset_name')`
            if not passed.
        max_edits_sweep: Hamming thresholds to sweep.

    Returns a dict or None if the dataset has no sequence data (non-antibody,
    non-nucleotide, or sequences not exposed by the loader).
    """
    heavy_seqs = data.get('heavy_seqs')
    if heavy_seqs is None or len(heavy_seqs) == 0:
        # Antibody datasets expose heavy_seqs; nucleotide uses sequences key.
        seqs = data.get('sequences')
        if seqs is None or len(seqs) == 0:
            return None
        heavy_seqs = list(seqs)

    labels = data['labels']
    if dataset_name is None:
        dataset_name = data.get('dataset_name')
    results = {}

    # Scope 1: full V-region (fast, always available).
    print(f"  Sequence cliff sweep (full V-region, n={len(heavy_seqs)})...")
    results['full_V'] = antibody_sequence_cliff_sweep(
        list(heavy_seqs), labels, is_clf,
        max_edits_sweep=max_edits_sweep, scope_label='full_V')
    for thresh, info in results['full_V']['thresholds'].items():
        if 'error' not in info:
            print(f"    Hamming<={thresh}: {info['n_cliff_pairs']} pairs, "
                  f"{info['n_cliff_compounds']} cliff compounds "
                  f"({info.get('pct_cliff_compounds', 0):.1f}%)")

    # Scope 2: CDR-only (opt-in, slower due to ANARCI).
    # Caches CDR extraction per dataset so repeated runs are fast.
    cdr_seqs = _extract_cdr_sequences(list(heavy_seqs), dataset_name=dataset_name)
    if cdr_seqs is not None:
        print(f"  Sequence cliff sweep (CDR-only, from ANARCI/IMGT)...")
        results['cdr_only'] = antibody_sequence_cliff_sweep(
            cdr_seqs, labels, is_clf,
            max_edits_sweep=max_edits_sweep, scope_label='cdr_only')
        for thresh, info in results['cdr_only']['thresholds'].items():
            if 'error' not in info:
                print(f"    Hamming<={thresh}: {info['n_cliff_pairs']} pairs, "
                      f"{info['n_cliff_compounds']} cliff compounds "
                      f"({info.get('pct_cliff_compounds', 0):.1f}%)")
    else:
        results['cdr_only'] = {'skipped': True,
                               'reason': 'ANARCI unavailable or failed on >50% of sequences'}
        print(f"  Sequence cliff sweep (CDR-only): SKIPPED (ANARCI unavailable)")

    return results


# Similarity-specific extra params (layered on top of SPEED_TIERS config)
SIMILARITY_EXTRA = {
    'weight_methods': ['lightgbm_gain'],
    'k_values': [5, 10],
    'n_sample': 5000,
    'knn_conditions': ['hybrid_unweighted', 'hybrid_weighted'],
}


# ── Single-dataset experiment list ─────────────────────────────────────────
# S09 is cross-dataset — run separately.

SINGLE_DATASET_EXPERIMENTS = [
    's01',   # similarity baseline (standalone)
    's02',   # hybrid similarity
    's03',   # feature concentration
    's04',   # activity cliffs
    's05',   # kNN prediction
    's06',   # applicability domain
    's07',   # noise bridge
    's08',   # optimal dimension (standalone)
    's10',   # importance downstream (standalone)
    's11',   # bayesian optimization (standalone)
    's12',   # cross-domain cliffs (standalone, generic)
    's13',   # clustering + retrieval (standalone)
]


def _import_experiment(exp_id: str):
    """Dynamically import a similarity experiment module by ID (e.g. 's01')."""
    module_map = {
        's01': 's01_similarity_baseline',
        's02': 's02_hybrid_similarity',
        's03': 's03_feature_concentration',
        's04': 's04_activity_cliffs',
        's05': 's05_knn_prediction',
        's06': 's06_applicability_domain',
        's07': 's07_noise_bridge',
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.similarity.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name used by save_experiment_results."""
    name_map = {
        's01': 's01_similarity_baseline',
        's02': 's02_hybrid_similarity',
        's03': 's03_feature_concentration',
        's04': 's04_activity_cliffs',
        's05': 's05_knn_prediction',
        's06': 's06_applicability_domain',
        's07': 's07_noise_bridge',
        's08': 's08_optimal_dimension',
        's10': 's10_importance_downstream',
        's11': 's11_bayesian_optimization',
        's12': 's12_cross_domain_cliffs',
        's13': 's13_clustering_retrieval',
    }
    return name_map.get(exp_id, exp_id)


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None):
    """Load data once and run all single-dataset similarity experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: fast tier + similarity extras).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
    """
    if config is None:
        from kirby.find_best_hybrid import SPEED_TIERS
        from experiments._shared import SCREENING_BUDGETS
        tier = SPEED_TIERS['full']
        config = {
            'n_splits': tier['n_folds'],
            'search_speed': 'full',
            'allocations': tier['allocation_methods'],
            'importance_methods': tier['importance_methods'],
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': SCREENING_BUDGETS,
            'models': DOWNSTREAM_MODELS,
        }
        config.update(SIMILARITY_EXTRA)
    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)

    # Resolve rep list
    if rep_names is None:
        ds_reps = DATASET_REPS.get(dataset_name)
        if ds_reps is None:
            raise ValueError(
                f"No rep list for '{dataset_name}'. Pass rep_names explicitly "
                f"or add it to DATASET_REPS in complexity_helpers.py."
            )
        rep_names = ds_reps['general']

    print(f"\n{'=' * 70}")
    print(f"SIMILARITY EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Config:      {config['n_splits']} fold, {config['models']}, speed={config['search_speed']}")
    print(f"Reps ({len(rep_names)}): {', '.join(rep_names)}")
    print(f"Experiments: {experiments}")
    print(f"Output:      {output_dir}")

    # ── Load data ONCE ──
    t0 = time.time()
    print(f"\nLoading {dataset_name}...")
    data = load_data(dataset_name, rep_preset='all', rep_names=rep_names)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # NaN guard: some reps produce NaN for certain sequences. Replace with 0 and warn.
    import numpy as np
    reps_data = data.get('reps', {})
    for rname, X in list(reps_data.items()):
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            pct = 100 * nan_count / X.size
            print(f"  WARNING: {rname} has {nan_count} NaN values ({pct:.1f}%) — imputing 0")
            reps_data[rname] = np.nan_to_num(X, nan=0.0)

    # ── Load augmentation reps ──
    aug_rep_names = DATASET_REPS.get(dataset_name, {}).get('augmentation', [])
    if aug_rep_names:
        missing = [r for r in aug_rep_names if r not in reps_data]
        if missing:
            print(f"  Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing)
            aug_reps = {}
            for rn, arr in aug_data['reps'].items():
                if int(np.isnan(arr).sum()) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                aug_reps[rn] = arr
            data['augmentations'] = aug_reps
        else:
            data['augmentations'] = {r: reps_data[r] for r in aug_rep_names}
        print(f"  Augmentations ({len(data['augmentations'])}): "
              f"{', '.join(f'{k}={v.shape[1]}' for k, v in data['augmentations'].items())}")
    else:
        data['augmentations'] = None

    # ── Precompute hybrid searches (general reps) ──
    _task_type = data['task_type']
    if len(reps_data) >= 2:
        from experiments._shared import precompute_hybrid_searches
        print(f"\n  Precomputing hybrid searches (general, {len(reps_data)} reps)...")
        precompute_results = precompute_hybrid_searches(
            reps=reps_data,
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name, tier='general',
        )
        data['precomputed_searches'] = precompute_results
    else:
        data['precomputed_searches'] = {}

    # ── Precompute hybrid searches (top-tier reps) ──
    top_tier_names = DATASET_REPS.get(dataset_name, {}).get('top_tier', [])
    top_tier_reps = {k: v for k, v in reps_data.items() if k in top_tier_names}
    data['top_tier_reps'] = top_tier_reps
    if len(top_tier_reps) >= 2:
        print(f"\n  Precomputing hybrid searches (top-tier, {len(top_tier_reps)} reps)...")
        precompute_top = precompute_hybrid_searches(
            reps=top_tier_reps,
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name, tier='top',
        )
        data['precomputed_searches_top'] = precompute_top
    else:
        data['precomputed_searches_top'] = {}

    # ── Run experiments in order ──
    results = {}
    for exp_id in experiments:
        print(f"\n{'─' * 70}")
        t_exp = time.time()

        try:
            mod = _import_experiment(exp_id)
        except (ImportError, KeyError) as e:
            print(f"  SKIP {exp_id}: import failed ({e})")
            continue

        # Build kwargs — pass similarity-specific params
        kwargs = {}
        if exp_id in ('s01', 's02', 's08'):
            kwargs['n_sample'] = config.get('n_sample', 5000)
        if exp_id == 's02':
            kwargs['weight_methods'] = config.get('weight_methods', ['lightgbm_gain'])
        if exp_id == 's05':
            kwargs['k_values'] = config.get('k_values', [5, 10])
            kwargs['conditions'] = config.get('knn_conditions')
        if exp_id == 's08':
            kwargs['budgets'] = config.get('budgets', [25, 50, 100, 200])

        try:
            exp_result = mod.run(data, config, output_dir, dataset_name, **kwargs)
            results[exp_id] = exp_result

            # Save per-experiment JSON
            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(exp_result, output_dir, exp_full_name, dataset_name)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': str(e)}

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE: {dataset_name} — {len(results)} experiments in {total:.0f}s")
    print(f"{'=' * 70}")

    return results
