"""Shared configuration and runner for federated experiments.

Bridges FLuID (Lhasa Limited) and KIRBy for the federated experiment category.
FLuID uses RDKit Mol objects + DataFrames; KIRBy uses SMILES + numpy arrays.

Privacy principles:
    - No silo shares raw data with other silos or the coordinator
    - Silos may share: predictions, feature indices, importance weights, validation scores
    - The "naive pool" condition is explicitly labeled as privacy-violating baseline

Config: RF / LightGBM / MLP, 1-fold scaffold CV, fast search.
Rep lists reused from complexity_helpers.DATASET_REPS.
"""

import sys
import os
import time
import warnings
from pathlib import Path

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))         # tests/experiments/

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score, f1_score,
)

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy,
    DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

# ── FLuID import ─────────────────────────────────────────────────────────────
# FLuID calls IPython display() and Plotly showPlot() inside its functions.
# Monkey-patch before import to suppress in non-notebook environments.

import builtins
_original_display = getattr(builtins, 'display', None)
builtins.display = lambda *a, **kw: None

_fluid_default = '/Users/apunt/repos/lhasa/FLuID_POC/preprocessing'
_fluid_server = '/data/stat-cadd/scat9264/FLuID_POC/preprocessing'
if os.path.isdir(_fluid_server):
    _fluid_default = _fluid_server
_fluid_path = os.environ.get('FLUID_PATH', _fluid_default)
sys.path.insert(0, _fluid_path)

try:
    import FLuID as fluid
    # Patch FLuID's module-level display binding (imported from IPython)
    fluid.display = lambda *a, **kw: None
    fluid.showPlot = lambda *a, **kw: None
    _FLUID_AVAILABLE = True
except ImportError:
    _FLUID_AVAILABLE = False
    warnings.warn("FLuID not available — federated experiments will fail.")

# Restore display if it existed before
if _original_display is not None:
    builtins.display = _original_display


# ── Lhasa hERG loader (returns data in load_data format) ─────────────────────

def load_lhasa_herg_data(rep_names=None):
    """Load Lhasa's curated hERG training data in the same format as load_data().

    Returns a data dict compatible with fed01/02/04 with:
    - Binary labels (0/1) already binarized
    - task_type='classification'
    - Scaffold groups for CV splitting
    """
    from experiments.federated.lhasa_herg_loader import load_lhasa_herg_training

    smiles, labels, inchikeys, _ = load_lhasa_herg_training()
    print(f"  Lhasa hERG: {len(labels)} molecules, "
          f"{int(labels.sum())} active, {len(labels) - int(labels.sum())} inactive")

    # Compute representations (same path as fed03)
    from find_best_hybrid_molecular import (
        get_representations as get_molecular_representations,
        assign_scaffold_groups,
    )
    if rep_names is None:
        rep_names = 'all'
    reps = get_molecular_representations(smiles, rep_names)

    # Scaffold groups
    try:
        groups, _ = assign_scaffold_groups(smiles)
    except Exception:
        groups = None

    return {
        'reps': reps,
        'labels': labels,
        'smiles': smiles,
        'groups': groups,
        'task_type': 'classification',
        'domain': 'molecular',
        'is_binary': True,  # Signal to experiments: labels already binarized
        'augmentations': None,
    }


# ── Federated-specific config ────────────────────────────────────────────────

_FEDERATED_CONFIG_CACHE = None


def _get_federated_config():
    global _FEDERATED_CONFIG_CACHE
    if _FEDERATED_CONFIG_CACHE is None:
        from kirby.find_best_hybrid import SPEED_TIERS
        tier = SPEED_TIERS['fast']
        _FEDERATED_CONFIG_CACHE = {
            'n_splits': tier['n_folds'],
            'search_speed': 'fast',
            'allocations': tier['allocation_methods'],
            'importance_methods': tier['importance_methods'],
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': [100],
            'models': DOWNSTREAM_MODELS,
        }
    return _FEDERATED_CONFIG_CACHE


# ── Experiment list ──────────────────────────────────────────────────────────

SINGLE_DATASET_EXPERIMENTS = [
    'fed01',   # noise heterogeneity across silos + federated noise detection
    'fed02',   # rep diversity as federated silos
    'fed03',   # synthetic private datasets + antibody federated
    'fed04',   # cluster-aware importance
    'fed05',   # antibody federated extension
]

# Experiments that load their own silo data internally — run_all skips
# load_data for these and passes a placeholder data dict instead.
SELF_LOADING_EXPERIMENTS = {
    'fed05',   # loads 3 antibody silo datasets internally
}


# =============================================================================
# FORMAT BRIDGE: KIRBy <-> FLuID
# =============================================================================

def smiles_to_fluid_df(smiles_list, labels, pki_threshold=6.0, source='KIRBy'):
    """Convert KIRBy SMILES + continuous labels into FLuID-format DataFrame.

    Args:
        smiles_list: List of SMILES strings.
        labels: Continuous labels (e.g. pKi values).
        pki_threshold: Threshold for binarization (>= threshold = Active).
        source: Source name for the SOURCE column.

    Returns:
        (fluid_df, valid_mask): DataFrame with FLuID columns and boolean mask
        of valid molecules (some SMILES may fail RDKit parsing).
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    mols = []
    fps = []
    inchis = []
    valid_mask = np.ones(len(smiles_list), dtype=bool)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask[i] = False
            mols.append(None)
            fps.append(None)
            inchis.append(None)
            continue
        mols.append(mol)
        fps.append(fluid.compute_fp_array(mol))
        try:
            inchi = MolToInchi(mol)
            inchis.append(InchiToInchiKey(inchi) if inchi else 'UNKNOWN')
        except Exception:
            inchis.append('UNKNOWN')

    # Filter to valid only
    valid_idx = np.where(valid_mask)[0]
    labels_valid = np.asarray(labels)[valid_idx]

    # If labels are already binary (0/1), skip thresholding
    unique_vals = set(np.unique(labels_valid))
    if unique_vals <= {0, 1, 0.0, 1.0}:
        binary_labels = labels_valid.astype(int)
    else:
        binary_labels = (labels_valid >= pki_threshold).astype(int)
    activity_strs = np.where(binary_labels == 1, 'Active', 'Inactive')

    df = pd.DataFrame({
        'MOLECULE': [mols[i] for i in valid_idx],
        'FP': [fps[i] for i in valid_idx],
        'CLASS': binary_labels,
        'ACTIVITY': activity_strs,
        'INCHI': [inchis[i] for i in valid_idx],
        'SOURCE': source,
        'PKI': labels_valid,
    })
    df.index = range(len(df))

    return df, valid_mask


def cluster_into_silos(smiles_list, labels, k=5, smooth_factor=0.05,
                       seed=42, pki_threshold=6.0):
    """Cluster molecules into K silos using FLuID's K-Means on Morgan FPs.

    Args:
        smiles_list: List of SMILES strings.
        labels: Continuous labels (e.g. pKi values).
        k: Number of clusters (silos).
        smooth_factor: Fraction of samples to randomly reassign across clusters.
        seed: Random seed for reproducibility.
        pki_threshold: Threshold for binarization.

    Returns:
        (silo_indices, fluid_df, cluster_data, valid_mask):
        - silo_indices: {silo_id: np.array of row indices in fluid_df}
          silo 0 = all samples, silos 1..k = clusters
        - fluid_df: FLuID-format DataFrame
        - cluster_data: list of DataFrames [full, cluster_1, ..., cluster_k]
        - valid_mask: boolean mask of valid molecules
    """
    fluid_df, valid_mask = smiles_to_fluid_df(
        smiles_list, labels, pki_threshold=pki_threshold)

    params = {
        'k': k,
        'smooth_factor': smooth_factor,
        'random_state': seed,
        # FLuID's cluster_data_space accesses these for plotting (even though
        # we patched showPlot, fig.update_layout runs before showPlot):
        'figure_font': dict(family="Arial", size=12),
    }

    # FLuID's cluster_data_space modifies df in place, adds CLUSTER/CLUSTER_ID
    cluster_data = fluid.cluster_data_space(
        fluid_df, 'Teacher', 'T', k, smooth_factor, params)

    # Extract index mappings
    silo_indices = {0: np.arange(len(fluid_df))}
    for s in range(1, k + 1):
        silo_indices[s] = np.array(
            cluster_data[s].index.tolist(), dtype=int)

    return silo_indices, fluid_df, cluster_data, valid_mask


def cluster_training_only(train_fluid_df, k=5, smooth_factor=0.05, seed=42):
    """Cluster TRAINING molecules only, matching FLuID's pattern.

    FLuID clusters trainingData only — test set never touches K-Means.
    This function takes a pre-split training DataFrame and clusters it.

    Args:
        train_fluid_df: FLuID-format DataFrame of TRAINING molecules only.
            Must have 'FP' column (Morgan fingerprints).
        k: Number of clusters (silos).
        smooth_factor: Fraction of samples to randomly reassign.
        seed: Random seed.

    Returns:
        (silo_indices, cluster_data, kmeans_model):
        - silo_indices: {silo_id: np.array of row indices} (training-relative)
        - cluster_data: list of DataFrames [full_train, cluster_1, ..., cluster_k]
        - kmeans_model: fitted KMeans for assigning new molecules to clusters
    """
    if not _FLUID_AVAILABLE:
        raise RuntimeError("FLuID not available — cannot cluster")

    from sklearn.cluster import KMeans as _KMeans

    params = {
        'k': k,
        'smooth_factor': smooth_factor,
        'random_state': seed,
        'figure_font': dict(family="Arial", size=12),
    }

    # Reset index to 0..N_train-1 so cluster indices are training-relative
    train_df = train_fluid_df.reset_index(drop=True)

    # FLuID's cluster_data_space: K-Means + smooth_factor reassignment
    # Modifies train_df in place, adds CLUSTER/CLUSTER_ID columns
    cluster_data = fluid.cluster_data_space(
        train_df, 'Teacher', 'T', k, smooth_factor, params)

    # Extract index mappings (training-relative)
    silo_indices = {}
    for s in range(1, k + 1):
        silo_indices[s] = np.array(
            cluster_data[s].index.tolist(), dtype=int)

    # Fit KMeans on training FPs for test molecule assignment.
    # FLuID's cluster_data_space applies smoothing AFTER K-Means, so the
    # CLUSTER column may differ from raw K-Means labels. We fit a fresh
    # K-Means on the SMOOTHED assignments to match.
    X_train = np.asarray(list(train_df.FP))
    smoothed_labels = np.asarray(train_df['CLUSTER']) - 1  # 0-indexed for sklearn
    kmeans_model = _KMeans(n_clusters=k, random_state=seed)
    kmeans_model.fit(X_train)
    # Override cluster_centers_ with centroids of smoothed assignments
    for c in range(k):
        mask = smoothed_labels == c
        if mask.any():
            kmeans_model.cluster_centers_[c] = X_train[mask].mean(axis=0)

    return silo_indices, cluster_data, kmeans_model


# =============================================================================
# NOISE INJECTION (PER-SILO)
# =============================================================================

def inject_noise_per_silo(labels, silo_indices, noise_levels,
                          noise_type='label_flip', seed=42):
    """Inject heterogeneous noise across silos.

    Args:
        labels: 1D array of binary labels (0/1) for label_flip,
                or continuous for gaussian.
        silo_indices: {silo_id: np.array of indices}. Use silos 1..k only.
        noise_levels: {silo_id: fraction} e.g. {1: 0.0, 2: 0.05, ..., 5: 0.3}.
        noise_type: 'label_flip' (binary) or 'gaussian' (continuous).
        seed: Random seed.

    Returns:
        (labels_noisy, noise_mask_global, noise_mask_per_silo)
    """
    rng = np.random.RandomState(seed)
    labels_noisy = labels.copy()
    noise_mask_global = np.zeros(len(labels), dtype=bool)
    noise_mask_per_silo = {}

    for silo_id, fraction in noise_levels.items():
        idx = silo_indices[silo_id]
        silo_mask = np.zeros(len(labels), dtype=bool)

        if fraction <= 0:
            noise_mask_per_silo[silo_id] = silo_mask
            continue

        if noise_type == 'label_flip':
            n_flip = int(fraction * len(idx))
            if n_flip == 0:
                noise_mask_per_silo[silo_id] = silo_mask
                continue
            flip_idx = rng.choice(idx, size=n_flip, replace=False)
            labels_noisy[flip_idx] = 1 - labels_noisy[flip_idx]
            silo_mask[flip_idx] = True

        elif noise_type == 'gaussian':
            # Use SILO-LOCAL label range for calibrated noise scale
            silo_labels = labels[idx]
            silo_range = np.max(silo_labels) - np.min(silo_labels)
            if silo_range < 1e-12:
                silo_range = 1.0  # fallback for constant labels
            noise = rng.normal(0, fraction * silo_range, len(idx))
            labels_noisy[idx] += noise
            # Ground truth mask: use silo-local MAD
            silo_mad = float(np.median(np.abs(
                silo_labels - np.median(silo_labels))))
            if silo_mad < 1e-12:
                silo_mad = float(np.std(silo_labels)) or 1.0
            silo_mask[idx[np.abs(noise) > silo_mad]] = True

        noise_mask_global |= silo_mask
        noise_mask_per_silo[silo_id] = silo_mask

    return labels_noisy, noise_mask_global, noise_mask_per_silo


# =============================================================================
# FLUID TEACHER ENSEMBLE (CONDITIONS 2-4)
# =============================================================================

def train_teachers_per_silo(fluid_df, cluster_data, noisy_labels, k,
                            algorithm='rf'):
    """Train one FLuID classifier per silo using noisy labels.

    Updates fluid_df CLASS/ACTIVITY in place, then wraps build_teacher_models.
    Privacy: each silo trains locally on its own data.

    Returns:
        teacher_models: list of (classifier, size, df) tuples, length k+1
    """
    # Update labels in the DataFrame
    fluid_df['CLASS'] = noisy_labels
    fluid_df['ACTIVITY'] = np.where(noisy_labels == 1, 'Active', 'Inactive')

    # Rebuild cluster_data views with updated labels
    cluster_data_updated = [fluid_df]  # Teacher 0 = full dataset
    for s in range(1, k + 1):
        cluster_data_updated.append(
            fluid_df[fluid_df['CLUSTER'] == s])

    params = {'k': k, 'teacher_algorithm': algorithm}
    teacher_models = fluid.build_teacher_models(cluster_data_updated, params)
    return teacher_models


def predict_fluid_similarity_ensemble(teacher_models, cluster_data_train,
                                      test_fps, k, seed=42):
    """FLuID's 8-NN similarity-weighted ensemble prediction.

    Reimplements the core math from FLuID.annotate_transfer_data() and
    federate_labels() without I/O, plotting, or pickle side effects.

    Privacy: each teacher runs predict locally, shares only predictions
    and similarity weights to coordinator.

    Args:
        teacher_models: list of (clf, size, df) tuples from training.
        cluster_data_train: list of TRAIN-ONLY DataFrames per cluster.
            Must NOT include test samples (to avoid data leakage in KNN).
        test_fps: Morgan FP array for test molecules.
        k: number of silos.
        seed: random seed.

    Returns:
        (pred_classes, pred_probs_positive, teacher_weights_per_sample)
    """
    n_test = len(test_fps)

    # Collect per-teacher predictions and weights
    all_probs = {}   # {teacher_id: (n_test, 2) probabilities}
    all_weights = {} # {teacher_id: (n_test,) similarity weights}

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            continue  # Skip clusters with < 2 samples (no valid teacher)
        teacher_df = cluster_data_train[t]

        # Teacher predictions on test set
        probs = clf.predict_proba(test_fps)  # (n_test, 2)
        all_probs[t] = probs

        # 8-NN distance weighting using TRAIN-ONLY teacher data
        teacher_fps = np.asarray(list(teacher_df.FP))
        n_sample = max(1, int(0.2 * len(teacher_fps)))
        rng = np.random.RandomState(seed + t)
        sample_idx = rng.choice(len(teacher_fps), size=n_sample, replace=False)
        teacher_fps_sample = teacher_fps[sample_idx]

        n_neighbors = min(8, len(teacher_fps_sample))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(teacher_fps_sample)
        distances, _ = knn.kneighbors(test_fps)
        all_weights[t] = distances.mean(axis=1)  # mean distance to 8-NN

    if not all_probs:
        # All teachers were None — return uniform predictions
        return np.zeros(n_test, dtype=int), np.full(n_test, 0.5), {}

    # Normalize distances → similarities per silo (consistent with regression variant)
    active_teachers = sorted(all_probs.keys())
    for t in active_teachers:
        raw = all_weights[t]
        min_t, max_t = raw.min(), raw.max()
        delta_t = max_t - min_t if max_t > min_t else 1.0
        normalized = (raw - min_t) / delta_t
        all_weights[t] = (1.0 - normalized) ** 2

    # Normalize weights to sum to 1 per sample (active teachers only)
    weight_sum = sum(all_weights[t] for t in active_teachers)
    weight_sum = np.maximum(weight_sum, 1e-10)
    for t in active_teachers:
        all_weights[t] = all_weights[t] / weight_sum

    # Weighted average of probability distributions
    fed_probs = np.zeros((n_test, 2))
    for t in active_teachers:
        fed_probs += all_probs[t] * all_weights[t][:, np.newaxis]

    pred_classes = (fed_probs[:, 1] > fed_probs[:, 0]).astype(int)
    pred_probs_positive = fed_probs[:, 1]  # P(Active) for AUROC

    teacher_weights = np.column_stack(
        [all_weights.get(t, np.zeros(n_test)) for t in range(1, k + 1)])

    return pred_classes, pred_probs_positive, teacher_weights


def predict_fluid_quality_ensemble(teacher_models, cluster_data_train,
                                   test_fps, k):
    """Quality-weighted ensemble: weight teachers by cross-val MCC.

    Privacy: each silo shares one scalar (its MCC), not data.

    Args:
        teacher_models: list of (clf, size, df) tuples.
        cluster_data_train: list of TRAIN-ONLY DataFrames per cluster.
        test_fps: Morgan FP array for test molecules.
        k: number of silos.

    Returns:
        (pred_classes, pred_probs_positive, teacher_mcc_weights)
    """
    from sklearn.model_selection import KFold

    n_test = len(test_fps)
    mccs = {}
    all_probs = {}

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            mccs[t] = 0.0
            continue  # Skip clusters with < 2 samples (no valid teacher)
        teacher_df = cluster_data_train[t]

        # Teacher predictions on test
        all_probs[t] = clf.predict_proba(test_fps)

        # Cross-val MCC on own TRAINING data only (no test leakage)
        # FLuID uses cv=5 always. We adapt to tiny silos where 5-fold may
        # fail (e.g. <5 samples in minority class). Equivalent when viable.
        X_t = np.asarray(list(teacher_df.FP))
        y_t = np.asarray(list(teacher_df.CLASS))
        n_classes = len(np.unique(y_t))
        min_class_count = min(np.bincount(y_t.astype(int)))
        n_cv = min(5, min_class_count, len(y_t))
        if n_classes < 2 or n_cv < 2:
            mccs[t] = 0.0
            continue
        try:
            y_pred_cv = cross_val_predict(
                clf, X_t, y_t, cv=KFold(n_splits=n_cv, shuffle=False))
            mccs[t] = max(0.0, matthews_corrcoef(y_t, y_pred_cv))
        except Exception:
            mccs[t] = 0.0

    if not all_probs:
        # All teachers were None — return uniform predictions
        return np.zeros(n_test, dtype=int), np.full(n_test, 0.5), {}

    active_teachers = sorted(all_probs.keys())

    # Normalize MCC weights (active teachers only)
    total_mcc = sum(mccs.get(t, 0.0) for t in active_teachers)
    if total_mcc <= 0:
        weights = {t: 1.0 / len(active_teachers) for t in active_teachers}
    else:
        weights = {t: mccs.get(t, 0.0) / total_mcc for t in active_teachers}

    # Weighted average
    fed_probs = np.zeros((n_test, 2))
    for t in active_teachers:
        fed_probs += all_probs[t] * weights[t]

    pred_classes = (fed_probs[:, 1] > fed_probs[:, 0]).astype(int)
    pred_probs_positive = fed_probs[:, 1]  # P(Active) for AUROC

    return pred_classes, pred_probs_positive, weights


def federate_labels_and_train_student(teacher_models, cluster_data, fluid_df,
                                      k, test_fps, algorithm='rf',
                                      student_size=None,
                                      train_indices=None, seed=42):
    """FLuID-faithful student pipeline.

    Matches FLuID's published protocol:
    1. Each teacher predicts on public pool with similarity-weighted vote
       (FLuID's federate_labels uses W-T weights, not equal weights)
    2. Student data selected via FLuID's 'balanced' mode: sort by confidence
       descending, pick student_size/2 per class
    3. Student RF trained on federated pseudo-labels

    Privacy: only predictions on shared public molecules are exchanged.
    Test molecules are excluded from the student training pool.

    Simulation note: KIRBy uses training data as the public pool (no separate
    transfer pool like FLuID's 50K ChEMBL). This is a known simulation-level
    compromise documented in the federated audit.

    Args:
        train_indices: If provided, only these indices of fluid_df are
            used as the public pool. This prevents test data leakage.
        student_size: Number of federated pseudo-labels to select for
            the student. Default: min(10000, pool_size). FLuID uses 10000.

    Returns:
        (pred_classes, pred_probs, n_federated)
    """
    from sklearn.ensemble import RandomForestClassifier

    all_fps = np.asarray(list(fluid_df.FP))

    # Use only train indices as public pool (exclude test to avoid leakage)
    if train_indices is not None:
        pool_fps = all_fps[train_indices]
    else:
        pool_fps = all_fps
    n_pool = len(pool_fps)

    # Default student_size (FLuID: 10000; scale down for small pools)
    if student_size is None:
        student_size = min(10000, n_pool)

    # ── Step 1: Similarity-weighted federation (matches FLuID's federate_labels)
    # Each teacher predicts probabilities on the pool, weighted by 8-NN similarity
    all_probs = {}  # {t: (n_pool, 2)}
    all_weights = {}  # {t: (n_pool,)} similarity weights

    rng = np.random.RandomState(seed)

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            continue
        probs = clf.predict_proba(pool_fps)
        all_probs[t] = probs

        # 8-NN distance weighting using 20% sample of teacher's training FPs
        # (matches FLuID's annotate_transfer_data: data.sample(frac=0.2))
        teacher_df = cluster_data[t] if t < len(cluster_data) else None
        if teacher_df is not None and len(teacher_df) > 0:
            teacher_fps = np.asarray(list(teacher_df.FP))
            n_sample = max(1, int(0.2 * len(teacher_fps)))
            sample_idx = rng.choice(len(teacher_fps), size=n_sample, replace=False)
            teacher_fps_sample = teacher_fps[sample_idx]

            n_neighbors = min(8, len(teacher_fps_sample))
            knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            knn.fit(teacher_fps_sample)
            distances, _ = knn.kneighbors(pool_fps)
            all_weights[t] = distances.mean(axis=1)
        else:
            all_weights[t] = np.ones(n_pool)

    active_teachers = sorted(all_probs.keys())
    if not active_teachers:
        raise ValueError("No active teachers available for student pipeline")

    # Normalize distances → similarities: (1 - normalized_distance)^2
    # (matches FLuID's annotate_transfer_data normalization)
    all_raw = np.column_stack([all_weights[t] for t in active_teachers])
    min_w = all_raw.min()
    max_w = all_raw.max()
    delta = max_w - min_w if max_w > min_w else 1.0

    for t in active_teachers:
        normalized = (all_weights[t] - min_w) / delta
        all_weights[t] = (1.0 - normalized) ** 2

    # Normalize weights to sum to 1 per sample
    weight_sum = sum(all_weights[t] for t in active_teachers)
    weight_sum = np.maximum(weight_sum, 1e-10)
    for t in active_teachers:
        all_weights[t] = all_weights[t] / weight_sum

    # Weighted average of probability distributions
    fed_probs = np.zeros((n_pool, 2))
    for t in active_teachers:
        fed_probs += all_probs[t] * all_weights[t][:, np.newaxis]

    fed_classes = (fed_probs[:, 1] > fed_probs[:, 0]).astype(int)
    fed_confidence = np.maximum(fed_probs[:, 0], fed_probs[:, 1])

    # ── Step 2: Balanced selection by confidence (matches FLuID's 'balanced' mode)
    # FLuID's select_student_data: shuffle, then pick student_size/2 per class
    # sorted by federated confidence (highest first)
    half = student_size // 2
    active_idx = np.where(fed_classes == 1)[0]
    inactive_idx = np.where(fed_classes == 0)[0]

    # Sort each class by confidence descending, take top half
    active_order = np.argsort(fed_confidence[active_idx])[::-1]
    inactive_order = np.argsort(fed_confidence[inactive_idx])[::-1]

    n_active = min(half, len(active_idx))
    n_inactive = min(half, len(inactive_idx))
    selected_active = active_idx[active_order[:n_active]]
    selected_inactive = inactive_idx[inactive_order[:n_inactive]]

    selected = np.concatenate([selected_active, selected_inactive])
    student_fps = pool_fps[selected]
    student_classes = fed_classes[selected]

    # ── Step 3: Train student on federated pseudo-labels
    student = RandomForestClassifier(
        n_estimators=150, max_depth=15, class_weight='balanced',
        random_state=0)
    student.fit(student_fps, student_classes)

    # Predict on test set
    test_probs = student.predict_proba(test_fps)
    pred_classes = (test_probs[:, 1] > test_probs[:, 0]).astype(int)
    pred_probs_positive = test_probs[:, 1]  # P(Active) for AUROC

    return pred_classes, pred_probs_positive, len(fed_classes)


# =============================================================================
# KIRBY FEDERATED FEATURE SELECTION (CONDITION 5)
# =============================================================================

def federated_feature_selection(rep_dict, labels, silo_indices, k,
                                budget=100, allocation_method='performance_weighted',
                                importance_method='treeshap',
                                internal_evaluator='rf', seed=42):
    """Privacy-preserving federated feature selection.

    Protocol:
        1. Each silo runs create_hybrid locally on its own subset
        2. Each silo shares: feature_info (indices + importance), validation score
        3. Coordinator allocates total budget across silos proportional to scores
        4. Each silo selects its allocated features (top by importance)
        5. At inference: features selected using same indices per silo

    Privacy: no raw features leave any silo. Only feature indices,
    importance weights, and validation scores are shared.

    Args:
        rep_dict: {rep_name: np.ndarray (n_total, d)} — full feature arrays.
        labels: Labels for all samples (binary or continuous).
        silo_indices: {silo_id: np.array} for silos 1..k.
        k: Number of silos.
        budget: Total feature budget across all silos.
        allocation_method: KIRBy allocation method for within-silo selection.
        importance_method: KIRBy importance method.
        seed: Random seed.

    Returns:
        (X_hybrid, silo_budgets, silo_scores, silo_feature_infos)
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection

    from sklearn.linear_model import RidgeClassifierCV, RidgeCV
    from sklearn.model_selection import ShuffleSplit

    is_classification = len(np.unique(labels)) <= 10

    silo_scores = {}
    silo_feature_infos = {}

    # Phase 1: Each silo computes importance locally and shares validation score
    for s in range(1, k + 1):
        idx = silo_indices[s]
        if len(idx) < 20:
            silo_scores[s] = 0.0
            silo_feature_infos[s] = None
            continue

        # Silo's local data
        silo_reps = {r: arr[idx] for r, arr in rep_dict.items()}
        silo_labels = labels[idx]

        # Local importance computation (generous budget — will be trimmed)
        n_silo = len(idx)
        per_silo_budget = min(max(10, budget // k), n_silo)
        try:
            X_silo, fi = create_hybrid(
                silo_reps, silo_labels,
                allocation_method=allocation_method,
                importance_method=importance_method,
                budget=per_silo_budget,
                internal_evaluator=internal_evaluator,
            )
            silo_feature_infos[s] = fi

            # Silo shares its validation score (one scalar).
            # Quick internal eval: Ridge on 80/20 split of silo data.
            if len(idx) >= 30:
                ss = ShuffleSplit(n_splits=1, test_size=0.2,
                                 random_state=seed + s)
                tr, va = next(ss.split(X_silo))
                evaluator = (RidgeClassifierCV() if is_classification
                             else RidgeCV())
                evaluator.fit(X_silo[tr], silo_labels[tr])
                silo_scores[s] = max(0.0, float(
                    evaluator.score(X_silo[va], silo_labels[va])))
            else:
                silo_scores[s] = 0.5  # default for tiny silos

        except Exception as e:
            warnings.warn(f"Silo {s} hybrid failed: {e}")
            silo_scores[s] = 0.0
            silo_feature_infos[s] = None

    # Phase 2: Coordinator allocates budget across silos proportional to scores
    total_score = sum(silo_scores.values())
    if total_score <= 0:
        silo_budgets = {s: budget // k for s in range(1, k + 1)}
    else:
        silo_budgets = {}
        for s in range(1, k + 1):
            silo_budgets[s] = max(1, int(
                budget * silo_scores[s] / total_score))

    # Phase 3: Apply budget allocation — trim each silo's feature_info to
    # its allocated budget, then deduplicate features across silos.
    seen_features = set()  # (rep_name, feature_index) pairs already selected
    hybrid_parts = []

    for s in range(1, k + 1):
        fi = silo_feature_infos[s]
        if fi is None:
            continue

        allocated = silo_budgets.get(s, 0)
        trimmed_fi = {}
        total_selected = 0

        for rep_name, rep_info in fi.items():
            if not isinstance(rep_info, dict) or 'selected_indices' not in rep_info:
                trimmed_fi[rep_name] = rep_info
                continue

            indices = rep_info['selected_indices']
            importances = rep_info.get('importance_scores', np.ones(len(indices)))

            # Deduplicate: skip features already selected by another silo
            new_indices = []
            new_importances = []
            for idx_i, feat_idx in enumerate(indices):
                key = (rep_name, int(feat_idx))
                if key not in seen_features:
                    new_indices.append(feat_idx)
                    new_importances.append(
                        importances[idx_i] if idx_i < len(importances) else 0)

            # Trim to budget
            remaining = allocated - total_selected
            if remaining <= 0:
                trimmed_fi[rep_name] = {
                    'selected_indices': np.array([], dtype=int),
                    'importance_scores': np.array([]),
                    'n_features': 0,
                }
                continue

            if len(new_indices) > remaining:
                # Keep top features by importance
                order = np.argsort(new_importances)[::-1][:remaining]
                new_indices = [new_indices[o] for o in order]
                new_importances = [new_importances[o] for o in order]

            # Mark as seen
            for feat_idx in new_indices:
                seen_features.add((rep_name, int(feat_idx)))

            total_selected += len(new_indices)
            trimmed_fi[rep_name] = {
                'selected_indices': np.array(new_indices, dtype=int),
                'importance_scores': np.array(new_importances),
                'n_features': len(new_indices),
            }

        silo_feature_infos[s] = trimmed_fi

        try:
            X_silo = apply_feature_selection(rep_dict, trimmed_fi)
            if X_silo.shape[1] > 0:
                hybrid_parts.append(X_silo)
            else:
                silo_feature_infos[s] = None  # mark failed
        except Exception:
            silo_feature_infos[s] = None  # mark failed
            continue

    if not hybrid_parts:
        raise ValueError("No silo produced valid feature selections")

    X_hybrid = np.hstack(hybrid_parts)

    return X_hybrid, silo_budgets, silo_scores, silo_feature_infos


# =============================================================================
# FEDERATED NOISE DETECTION (CONDITION 6)
# =============================================================================

def federated_noise_detection(rep_dict, labels, silo_indices, noise_mask_true,
                              k, detector_names=None):
    """Run noise detectors in three privacy settings.

    Settings:
        - centralized: all features pooled (unrealistic baseline)
        - per_silo: each silo runs detectors locally (privacy-preserving)
        - federated_hybrid: detectors on federated feature-selected data

    Privacy: per_silo and federated_hybrid never share raw features.

    Returns:
        {setting: {detector: {precision, recall, f1, n_flagged}}}
    """
    from sklearn.preprocessing import StandardScaler
    from kirby.utils.noise_mitigation_methods import (
        get_mitigation_method, compute_distance_matrix,
        DISTANCE_AWARE_METHODS,
    )

    if detector_names is None:
        detector_names = [
            'knn_k5', 'knn_k10', 'lof', 'distance_weighted',
            'cv_disagreement', 'bootstrap_ensemble',
            'zscore', 'prediction_error',
        ]

    n_total = len(labels)
    n_noisy = int(noise_mask_true.sum())
    results = {}

    def _eval_detectors(X, y, mask_true, setting_name, D=None):
        setting_results = {}
        X_s = StandardScaler().fit_transform(X)
        for det_name in detector_names:
            try:
                kwargs = {}
                if det_name in DISTANCE_AWARE_METHODS and D is not None:
                    kwargs['precomputed_distances'] = D
                detector = get_mitigation_method(det_name, **kwargs)
                flagged_idx = detector.identify_noise(X_s, y)
                flagged_mask = np.zeros(len(y), dtype=bool)
                flagged_mask[flagged_idx] = True

                tp = int((flagged_mask & mask_true).sum())
                n_flagged = int(flagged_mask.sum())
                precision = tp / max(n_flagged, 1)
                recall = tp / max(mask_true.sum(), 1)
                f1 = (2 * precision * recall /
                      max(precision + recall, 1e-10))

                setting_results[det_name] = {
                    'n_flagged': n_flagged,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                }
            except Exception as e:
                setting_results[det_name] = {'error': str(e)}
        return setting_results

    # Setting 1: Centralized (unrealistic)
    X_concat = np.hstack([rep_dict[r] for r in sorted(rep_dict.keys())])
    D_concat = compute_distance_matrix(X_concat, standardize=True)
    results['centralized'] = _eval_detectors(
        X_concat, labels, noise_mask_true, 'centralized', D_concat)

    # Setting 2: Per-silo (privacy-preserving)
    # Each silo runs detectors locally, reports flagged global indices
    per_silo_flagged = np.zeros(n_total, dtype=bool)
    per_silo_detail = {}
    for s in range(1, k + 1):
        idx = silo_indices[s]
        if len(idx) < 20:
            continue
        X_silo = np.hstack([rep_dict[r][idx] for r in sorted(rep_dict.keys())])
        silo_noise_mask = noise_mask_true[idx]
        silo_results = _eval_detectors(
            X_silo, labels[idx], silo_noise_mask, f'silo_{s}')
        per_silo_detail[f'silo_{s}'] = silo_results

    results['per_silo'] = per_silo_detail

    # Setting 3: Federated hybrid (privacy-preserving feature selection)
    try:
        X_fed, _budgets, _scores, _fis = federated_feature_selection(
            rep_dict, labels, silo_indices, k, budget=100,
            allocation_method='performance_weighted',
            importance_method='treeshap',
            internal_evaluator='rf', seed=42)
        D_fed = compute_distance_matrix(X_fed, standardize=True)
        results['federated_hybrid'] = _eval_detectors(
            X_fed, labels, noise_mask_true, 'federated_hybrid', D_fed)
    except Exception as e:
        results['federated_hybrid'] = {'_error': str(e)}

    return results


# =============================================================================
# CLASSIFICATION METRICS HELPER
# =============================================================================

def compute_clf_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics for federated experiment evaluation.

    Returns dict with MCC, BAC, AUROC, F1.
    """
    metrics = {
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'bac': float(balanced_accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auroc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics['auroc'] = float('nan')
    return metrics


# =============================================================================
# REGRESSION METRICS AND ENSEMBLE HELPERS (for fed02+)
# =============================================================================

def compute_reg_metrics(y_true, y_pred):
    """Compute regression metrics for federated experiment evaluation.

    Returns dict with rmse, spearman_rho, spearman_p, pearson_r, pearson_p, r2.
    """
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import spearmanr as _spearmanr, pearsonr as _pearsonr

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    try:
        sp_r, sp_p = _spearmanr(y_true, y_pred)
        sp_r, sp_p = float(sp_r), float(sp_p)
    except Exception:
        sp_r, sp_p = float('nan'), float('nan')

    try:
        pe_r, pe_p = _pearsonr(y_true, y_pred)
        pe_r, pe_p = float(pe_r), float(pe_p)
    except Exception:
        pe_r, pe_p = float('nan'), float('nan')

    return {
        'rmse': rmse,
        'spearman_rho': sp_r,
        'spearman_p': sp_p,
        'pearson_r': pe_r,
        'pearson_p': pe_p,
        'r2': r2,
    }


def predict_fluid_similarity_ensemble_regression(teacher_models,
                                                 silo_train_features,
                                                 silo_test_features, k,
                                                 seed=42):
    """FLuID-style 8-NN similarity-weighted regression ensemble.

    Each silo trains a regressor on its own representation. Predictions are
    combined via weighted mean, with weights derived from 8-NN distances in
    each silo's OWN feature space.

    Privacy: each silo shares only predictions and similarity weights.

    Args:
        teacher_models: list (length k+1). Index 0 is None placeholder.
            Indices 1..k are (regressor, n_train, X_train) tuples.
        silo_train_features: list (length k+1). Index 0 is None.
            Indices 1..k are training feature arrays for each silo.
        silo_test_features: list (length k+1). Index 0 is None.
            Indices 1..k are test feature arrays for each silo.
        k: number of silos.
        seed: random seed.

    Returns:
        (predictions, teacher_weights_per_sample)
    """
    n_test = silo_test_features[1].shape[0]

    all_preds = {}
    all_weights = {}

    for t in range(1, k + 1):
        reg = teacher_models[t][0]
        if reg is None:
            continue  # Skip clusters with < 2 samples (no valid teacher)
        X_train_t = silo_train_features[t]
        X_test_t = silo_test_features[t]

        # Predict in this silo's feature space
        all_preds[t] = reg.predict(X_test_t)

        # 8-NN distance weighting in this silo's OWN feature space
        n_sample = max(1, int(0.2 * len(X_train_t)))
        rng = np.random.RandomState(seed + t)
        sample_idx = rng.choice(len(X_train_t), size=n_sample, replace=False)
        X_ref = X_train_t[sample_idx]

        n_neighbors = min(8, len(X_ref))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X_ref)
        distances, _ = knn.kneighbors(X_test_t)
        all_weights[t] = distances.mean(axis=1)

    if not all_preds:
        # All teachers were None — return zeros
        return np.zeros(n_test), np.zeros((n_test, k))

    active_teachers = sorted(all_preds.keys())

    # Normalize distances → similarities PER SILO (each silo's feature
    # space has different scale, so global normalization would be meaningless)
    for t in active_teachers:
        raw = all_weights[t]
        min_t, max_t = raw.min(), raw.max()
        delta_t = max_t - min_t if max_t > min_t else 1.0
        normalized = (raw - min_t) / delta_t
        all_weights[t] = (1.0 - normalized) ** 2

    # Normalize weights to sum to 1 per sample (active teachers only)
    weight_sum = sum(all_weights[t] for t in active_teachers)
    weight_sum = np.maximum(weight_sum, 1e-10)
    for t in active_teachers:
        all_weights[t] = all_weights[t] / weight_sum

    # Weighted mean of continuous predictions
    predictions = np.zeros(n_test)
    for t in active_teachers:
        predictions += all_preds[t] * all_weights[t]

    teacher_weights = np.column_stack(
        [all_weights.get(t, np.zeros(n_test)) for t in range(1, k + 1)])

    return predictions, teacher_weights


def predict_fluid_quality_ensemble_regression(teacher_models,
                                              silo_train_features,
                                              silo_test_features, k):
    """Quality-weighted regression ensemble: weight by inverse cross-val RMSE.

    Privacy: each silo shares one scalar (its RMSE), not data.

    Args:
        teacher_models: list (length k+1). Index 0 is None placeholder.
        silo_train_features: list (length k+1) of (X_train, y_train) tuples.
            Index 0 is None. Each entry is (features, labels) for that silo.
        silo_test_features: list (length k+1) of X_test arrays.
        k: number of silos.

    Returns:
        (predictions, rmse_weights_dict)
    """
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import mean_squared_error

    n_test = silo_test_features[1].shape[0]
    rmses = {}
    all_preds = {}

    for t in range(1, k + 1):
        reg = teacher_models[t][0]
        if reg is None:
            rmses[t] = float('inf')
            continue  # Skip clusters with < 2 samples (no valid teacher)
        X_train_t, y_train_t = silo_train_features[t]
        X_test_t = silo_test_features[t]

        all_preds[t] = reg.predict(X_test_t)

        # Cross-val RMSE on own training data
        n_cv = min(5, max(2, len(y_train_t) // 5))
        try:
            y_cv = cross_val_predict(
                reg, X_train_t, y_train_t,
                cv=KFold(n_splits=n_cv, shuffle=False))
            rmses[t] = float(np.sqrt(mean_squared_error(y_train_t, y_cv)))
        except Exception:
            rmses[t] = float('inf')

    if not all_preds:
        # All teachers were None — return zeros
        return np.zeros(n_test), {}

    active_teachers = sorted(all_preds.keys())

    # Weights: inverse RMSE (lower RMSE = higher weight), active teachers only
    inv_rmses = {t: 1.0 / max(rmses.get(t, float('inf')), 1e-6)
                 for t in active_teachers}
    total = sum(inv_rmses.values())
    if total <= 0:
        weights = {t: 1.0 / len(active_teachers) for t in active_teachers}
    else:
        weights = {t: v / total for t, v in inv_rmses.items()}

    # Quality-weighted mean
    predictions = np.zeros(n_test)
    for t in active_teachers:
        predictions += all_preds[t] * weights[t]

    return predictions, weights


def predict_fluid_similarity_ensemble_classification(
        teacher_models, silo_train_features, silo_test_features, k, seed=42):
    """FLuID-style 8-NN similarity-weighted classification ensemble.

    Mirrors predict_fluid_similarity_ensemble_regression but uses
    predict_proba and weighted probability averaging for classification.

    Args:
        teacher_models: list (length k+1). Index 0 is None placeholder.
            Indices 1..k are (classifier, n_train, X_train) tuples.
        silo_train_features: list (length k+1). Index 0 is None.
            Indices 1..k are training feature arrays for each silo.
        silo_test_features: list (length k+1). Index 0 is None.
            Indices 1..k are test feature arrays for each silo.
        k: number of silos.
        seed: random seed.

    Returns:
        (pred_classes, pred_probs_positive, teacher_weights_per_sample)
    """
    n_test = silo_test_features[1].shape[0]

    all_probs = {}
    all_weights = {}

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            continue  # Skip clusters with < 2 samples (no valid teacher)
        X_train_t = silo_train_features[t]
        X_test_t = silo_test_features[t]

        all_probs[t] = clf.predict_proba(X_test_t)

        # 8-NN distance weighting in this silo's OWN feature space
        n_sample = max(1, int(0.2 * len(X_train_t)))
        rng = np.random.RandomState(seed + t)
        sample_idx = rng.choice(len(X_train_t), size=n_sample, replace=False)
        X_ref = X_train_t[sample_idx]

        n_neighbors = min(8, len(X_ref))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X_ref)
        distances, _ = knn.kneighbors(X_test_t)
        all_weights[t] = distances.mean(axis=1)

    if not all_probs:
        return np.zeros(n_test, dtype=int), np.full(n_test, 0.5), np.zeros((n_test, k))

    active_teachers = sorted(all_probs.keys())

    # Normalize distances → similarities PER SILO
    for t in active_teachers:
        raw = all_weights[t]
        min_t, max_t = raw.min(), raw.max()
        delta_t = max_t - min_t if max_t > min_t else 1.0
        normalized = (raw - min_t) / delta_t
        all_weights[t] = (1.0 - normalized) ** 2

    # Normalize weights to sum to 1 per sample (active teachers only)
    weight_sum = sum(all_weights[t] for t in active_teachers)
    weight_sum = np.maximum(weight_sum, 1e-10)
    for t in active_teachers:
        all_weights[t] = all_weights[t] / weight_sum

    # Weighted average of probability distributions
    n_classes = all_probs[active_teachers[0]].shape[1]
    fed_probs = np.zeros((n_test, n_classes))
    for t in active_teachers:
        fed_probs += all_probs[t] * all_weights[t][:, np.newaxis]

    pred_classes = fed_probs.argmax(axis=1)
    pred_probs_positive = fed_probs[:, 1] if n_classes == 2 else None

    teacher_weights = np.column_stack(
        [all_weights.get(t, np.zeros(n_test)) for t in range(1, k + 1)])

    return pred_classes, pred_probs_positive, teacher_weights


def predict_fluid_quality_ensemble_classification(
        teacher_models, silo_train_features, silo_test_features, k):
    """Quality-weighted classification ensemble: weight by cross-val MCC.

    Mirrors predict_fluid_quality_ensemble_regression but uses classifiers.

    Args:
        teacher_models: list (length k+1). Index 0 is None placeholder.
        silo_train_features: list (length k+1) of (X_train, y_train) tuples.
            Index 0 is None.
        silo_test_features: list (length k+1) of X_test arrays.
        k: number of silos.

    Returns:
        (pred_classes, pred_probs_positive, mcc_weights_dict)
    """
    from sklearn.model_selection import KFold, cross_val_predict

    n_test = silo_test_features[1].shape[0]
    mccs = {}
    all_probs = {}

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            mccs[t] = 0.0
            continue  # Skip clusters with < 2 samples (no valid teacher)
        X_train_t, y_train_t = silo_train_features[t]
        X_test_t = silo_test_features[t]

        all_probs[t] = clf.predict_proba(X_test_t)

        # Cross-val MCC on own training data
        n_classes = len(np.unique(y_train_t))
        min_class = min(np.bincount(y_train_t.astype(int)))
        n_cv = min(5, min_class, len(y_train_t))
        if n_classes < 2 or n_cv < 2:
            mccs[t] = 0.0
            continue
        try:
            y_cv = cross_val_predict(
                clf, X_train_t, y_train_t,
                cv=KFold(n_splits=n_cv, shuffle=False))
            mccs[t] = max(0.0, float(matthews_corrcoef(y_train_t, y_cv)))
        except Exception:
            mccs[t] = 0.0

    if not all_probs:
        return np.zeros(n_test, dtype=int), np.full(n_test, 0.5), {}

    active_teachers = sorted(all_probs.keys())

    # Normalize MCC weights (active teachers only)
    total_mcc = sum(mccs.get(t, 0.0) for t in active_teachers)
    if total_mcc <= 0:
        weights = {t: 1.0 / len(active_teachers) for t in active_teachers}
    else:
        weights = {t: mccs.get(t, 0.0) / total_mcc for t in active_teachers}

    # Weighted average of probabilities
    n_classes = all_probs[active_teachers[0]].shape[1]
    fed_probs = np.zeros((n_test, n_classes))
    for t in active_teachers:
        fed_probs += all_probs[t] * weights[t]

    pred_classes = fed_probs.argmax(axis=1)
    pred_probs_positive = fed_probs[:, 1] if n_classes == 2 else None

    return pred_classes, pred_probs_positive, weights


def fluid_within_kirby_across(rep_dict_train, rep_dict_test, labels,
                              k_intra=3, budget=100, seed=42,
                              allocation_method='performance_weighted',
                              importance_method='treeshap',
                              internal_evaluator='rf'):
    """Condition 6: FLuID clustering within each silo, KIRBy across silos.

    For each representation (silo):
        1. PCA to reduce dimensionality
        2. K-Means to cluster molecules within that silo's feature space
    Then create virtual silo indices (one per rep × sub-cluster) and run
    federated_feature_selection across the expanded set.

    Privacy: intra-silo clustering is local. Cross-silo allocation uses
    the federated_feature_selection protocol.

    Args:
        rep_dict_train: {rep_name: X_train_array} for all silo reps.
        rep_dict_test: {rep_name: X_test_array} for all silo reps.
        labels: training labels (continuous).
        k_intra: number of sub-clusters per silo.
        budget: total feature budget.
        seed: random seed.

    Returns:
        (X_hybrid_train, X_hybrid_test, silo_budgets, silo_scores, metadata)
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from kirby.hybrid import apply_feature_selection

    virtual_silo_indices = {}
    metadata = {}
    silo_counter = 1

    for rep_name, X_train in rep_dict_train.items():
        # PCA to reduce high-dim reps before clustering
        n_components = max(1, min(50, X_train.shape[1], X_train.shape[0] - 1))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        if n_components < X_scaled.shape[1]:
            X_reduced = PCA(n_components=n_components,
                            random_state=seed).fit_transform(X_scaled)
        else:
            X_reduced = X_scaled

        km = KMeans(n_clusters=k_intra, random_state=seed, n_init=10)
        assignments = km.fit_predict(X_reduced)

        for c in range(k_intra):
            indices = np.where(assignments == c)[0]
            if len(indices) >= 20:
                virtual_silo_indices[silo_counter] = indices
                metadata[silo_counter] = {
                    'parent_rep': rep_name,
                    'cluster': c,
                    'n_samples': len(indices),
                }
                silo_counter += 1

    k_expanded = silo_counter - 1
    if k_expanded == 0:
        raise ValueError("No virtual silos with >= 20 samples")

    # Run per-silo feature selection, restricting each silo to its parent rep.
    # This enforces the privacy boundary: silo from rep A cannot select from rep B.
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import ShuffleSplit as _SS

    silo_fis = {}
    silo_scores = {}
    silo_budgets_raw = {}
    hybrid_train_parts = []

    for s in range(1, k_expanded + 1):
        idx = virtual_silo_indices[s]
        parent_rep = metadata[s]['parent_rep']
        # Only this silo's parent rep
        silo_rep_tr = {parent_rep: rep_dict_train[parent_rep][idx]}
        silo_labels = labels[idx]

        n_silo = len(idx)
        per_silo_budget = min(max(5, budget // k_expanded), n_silo)
        try:
            X_s, fi = create_hybrid(
                silo_rep_tr, silo_labels,
                allocation_method=allocation_method,
                importance_method=importance_method,
                budget=per_silo_budget,
                internal_evaluator=internal_evaluator)
            silo_fis[s] = fi

            # Validation score
            if len(idx) >= 30:
                ss = _SS(n_splits=1, test_size=0.2, random_state=seed + s)
                tr_v, va_v = next(ss.split(X_s))
                ev = RidgeCV()
                ev.fit(X_s[tr_v], silo_labels[tr_v])
                silo_scores[s] = max(0.0, float(
                    ev.score(X_s[va_v], silo_labels[va_v])))
            else:
                silo_scores[s] = 0.5
        except Exception:
            silo_fis[s] = None
            silo_scores[s] = 0.0

    # Budget allocation proportional to scores
    total_score = sum(silo_scores.values())
    for s in range(1, k_expanded + 1):
        if total_score > 0:
            silo_budgets_raw[s] = max(1, int(
                budget * silo_scores[s] / total_score))
        else:
            silo_budgets_raw[s] = budget // k_expanded
    silo_budgets = silo_budgets_raw

    # Build training hybrid: apply each silo's feature selection to full train
    # using only the parent rep (privacy: each silo only exposes its own rep)
    valid_silos = []
    for s in range(1, k_expanded + 1):
        fi = silo_fis.get(s)
        if fi is None:
            continue
        parent_rep = metadata[s]['parent_rep']
        parent_dict = {parent_rep: rep_dict_train[parent_rep]}
        try:
            X_part = apply_feature_selection(parent_dict, fi)
            if X_part.shape[1] > 0:
                hybrid_train_parts.append(X_part)
                valid_silos.append(s)
        except Exception:
            continue

    if not hybrid_train_parts:
        raise ValueError("No virtual silo produced valid features")

    X_hybrid_train = np.hstack(hybrid_train_parts)

    # Apply same selections to test data (same parent-rep restriction)
    hybrid_test_parts = []
    for s in valid_silos:
        fi = silo_fis[s]
        parent_rep = metadata[s]['parent_rep']
        parent_dict_te = {parent_rep: rep_dict_test[parent_rep]}
        X_part = apply_feature_selection(parent_dict_te, fi)
        if X_part.shape[1] > 0:
            hybrid_test_parts.append(X_part)

    if not hybrid_test_parts:
        raise ValueError("No virtual silo produced valid test features")

    X_hybrid_test = np.hstack(hybrid_test_parts)

    return X_hybrid_train, X_hybrid_test, silo_budgets, silo_scores, metadata


# =============================================================================
# IMPORTANCE PROFILE ANALYSIS (for fed04 cluster-aware importance)
# =============================================================================

def extract_importance_profile(feature_info):
    """Extract per-rep importance summary from a feature_info dict.

    Returns {rep_name: total_importance_score} — one scalar per rep
    summarizing how important that rep is in this hybrid.
    """
    profile = {}
    for rep_name, rep_info in feature_info.items():
        if not isinstance(rep_info, dict) or 'selected_indices' not in rep_info:
            continue
        n_features = rep_info.get('n_features', len(rep_info['selected_indices']))
        importance = rep_info.get('importance_scores', np.ones(n_features))
        profile[rep_name] = float(np.sum(importance)) if len(importance) > 0 else 0.0
    return profile


def compare_importance_profiles(profiles_dict):
    """Compare importance profiles across clusters via Spearman correlation.

    Args:
        profiles_dict: {cluster_id: {rep_name: importance_score}}

    Returns:
        (mean_spearman, pairwise_dict) where pairwise_dict maps
        '(c1,c2)' → spearman_r for each cluster pair.
    """
    from scipy.stats import spearmanr as _spearmanr
    from itertools import combinations

    cluster_ids = sorted(profiles_dict.keys())
    if len(cluster_ids) < 2:
        return 1.0, {}

    # Get union of all rep names
    all_reps = sorted(set().union(
        *[set(p.keys()) for p in profiles_dict.values()]))
    if len(all_reps) < 3:
        return float('nan'), {}

    # Build importance vectors (missing reps get 0)
    vectors = {}
    for cid in cluster_ids:
        vectors[cid] = np.array(
            [profiles_dict[cid].get(r, 0.0) for r in all_reps])

    # Pairwise Spearman
    pairwise = {}
    correlations = []
    for c1, c2 in combinations(cluster_ids, 2):
        try:
            r, _ = _spearmanr(vectors[c1], vectors[c2])
            r = float(r)
        except Exception:
            r = float('nan')
        pairwise[f'({c1},{c2})'] = r
        if not np.isnan(r):
            correlations.append(r)

    mean_r = float(np.mean(correlations)) if correlations else float('nan')
    return mean_r, pairwise


# =============================================================================
# ANTIBODY CLUSTERING (for fed05 — replaces FLuID's Morgan FP K-Means)
# =============================================================================

def cluster_antibody_silos(rep_dict, k=3, seed=42, cluster_rep='esm2'):
    """K-Means clustering on protein embeddings for antibody sequences.

    Replaces FLuID's Morgan FP K-Means for the antibody domain.
    Uses ESM2 embeddings (1280D) by default, with PCA reduction.

    Args:
        rep_dict: {rep_name: np.ndarray} from load_data.
        k: number of clusters.
        seed: random seed.
        cluster_rep: which rep to cluster on (default 'esm2').

    Returns:
        silo_indices: {cluster_id: np.array of indices} (1-indexed, 1..k)
    """
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Find clustering rep
    if cluster_rep in rep_dict:
        X = rep_dict[cluster_rep]
    else:
        # Fallback: try other large reps
        for fallback in ['esmc', 'prot_t5', 'ablang2', 'antiberty']:
            if fallback in rep_dict:
                X = rep_dict[fallback]
                cluster_rep = fallback
                break
        else:
            # Last resort: concatenate all reps
            X = np.hstack([rep_dict[r] for r in sorted(rep_dict.keys())])
            cluster_rep = 'concat'

    # PCA reduction if high-dimensional
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = max(1, min(50, X_scaled.shape[1], X_scaled.shape[0] - 1))
    if n_components < X_scaled.shape[1]:
        X_reduced = PCA(n_components=n_components,
                        random_state=seed).fit_transform(X_scaled)
    else:
        X_reduced = X_scaled

    km = _KMeans(n_clusters=k, random_state=seed, n_init=10)
    assignments = km.fit_predict(X_reduced)

    silo_indices = {}
    for c in range(k):
        silo_indices[c + 1] = np.where(assignments == c)[0]

    return silo_indices


# =============================================================================
# IMPORTANCE PROFILE TRANSFER (for fed03 cross-task transfer)
# =============================================================================

def importance_profile_transfer(source_feature_info, target_rep_dict,
                                target_labels, budget=100):
    """Transfer importance profile from source silo to target silo.

    Uses source's per-rep importance rankings to guide feature selection on
    target's data. Only feature INDEX preferences (which features in which
    reps) are transferred — no raw data crosses silo boundaries.

    Protocol:
        1. Source silo shares feature_info (selected_indices + importance per rep)
        2. Target silo selects the same feature indices from its own reps
        3. If a rep in source doesn't exist in target, it's skipped
        4. Budget is distributed proportional to source importance

    Args:
        source_feature_info: feature_info dict from create_hybrid on source data.
        target_rep_dict: {rep_name: np.ndarray} for target silo.
        target_labels: labels for target silo (for optional re-ranking).
        budget: max features to select.

    Returns:
        (X_target_hybrid, transferred_fi, transfer_stats)
    """
    from kirby.hybrid import apply_feature_selection

    transferred_fi = {}
    transfer_stats = {'matched_reps': 0, 'skipped_reps': 0, 'total_features': 0}

    for rep_name, rep_info in source_feature_info.items():
        if not isinstance(rep_info, dict) or 'selected_indices' not in rep_info:
            continue

        if rep_name not in target_rep_dict:
            transfer_stats['skipped_reps'] += 1
            continue

        source_indices = rep_info['selected_indices']
        source_importance = rep_info.get('importance_scores',
                                         np.ones(len(source_indices)))
        target_n_features = target_rep_dict[rep_name].shape[1]

        # Only keep indices that are valid for target's feature dimensions
        valid = source_indices < target_n_features
        indices = source_indices[valid]
        importance = source_importance[valid] if len(source_importance) == len(source_indices) else np.ones(len(indices))

        # Trim to budget share
        remaining = budget - transfer_stats['total_features']
        if remaining <= 0:
            break
        if len(indices) > remaining:
            order = np.argsort(importance)[::-1][:remaining]
            indices = indices[order]
            importance = importance[order]

        transferred_fi[rep_name] = {
            'selected_indices': np.array(indices, dtype=int),
            'importance_scores': np.array(importance),
            'n_features': len(indices),
        }
        transfer_stats['matched_reps'] += 1
        transfer_stats['total_features'] += len(indices)

    if transfer_stats['total_features'] == 0:
        return None, transferred_fi, transfer_stats

    X_target = apply_feature_selection(target_rep_dict, transferred_fi)
    return X_target, transferred_fi, transfer_stats


# =============================================================================
# RUN INFRASTRUCTURE (follows noise_helpers.run_all pattern)
# =============================================================================

def _import_experiment(exp_id: str):
    """Dynamically import a federated experiment module by ID."""
    module_map = {
        'fed01': 'fed01_noise_heterogeneity',
        'fed02': 'fed02_rep_diversity_silos',
        'fed03': 'fed03_synthetic_private',
        'fed04': 'fed04_cluster_aware_importance',
        'fed05': 'fed05_antibody_federated',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.federated.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name."""
    name_map = {
        'fed01': 'fed01_noise_heterogeneity',
        'fed02': 'fed02_rep_diversity_silos',
        'fed03': 'fed03_synthetic_private',
        'fed04': 'fed04_cluster_aware_importance',
        'fed05': 'fed05_antibody_federated',
    }
    return name_map.get(exp_id, exp_id)


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None):
    """Load data once and run all federated experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: all).
        config: Config dict override (default: FEDERATED_CONFIG).
        rep_names: Explicit rep list override.
    """
    if not _FLUID_AVAILABLE:
        print("ERROR: FLuID not available. Federated experiments require FLuID.")
        print(f"Set FLUID_PATH env var (current: {_fluid_path}) or install FLuID.")
        sys.exit(1)

    if config is None:
        config = dict(_get_federated_config())
    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)

    # Check if ALL requested experiments are self-loading (skip data load)
    all_self_loading = all(e in SELF_LOADING_EXPERIMENTS for e in experiments)

    # Resolve rep list (skip if all experiments load their own data or Lhasa)
    if not all_self_loading and dataset_name not in ('herg', 'lhasa_herg'):
        if rep_names is None:
            ds_reps = DATASET_REPS.get(dataset_name)
            if ds_reps is None:
                raise ValueError(
                    f"No rep list for '{dataset_name}'. Pass rep_names "
                    f"explicitly or add it to DATASET_REPS in "
                    f"complexity_helpers.py."
                )
            rep_names = ds_reps['general']
    else:
        rep_names = rep_names or []

    print(f"\n{'=' * 70}")
    print(f"FEDERATED EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Config:      {config['n_splits']} fold, {config['models']}, "
          f"speed={config['search_speed']}")
    print(f"Reps ({len(rep_names)}): {', '.join(rep_names)}")
    print(f"Experiments: {experiments}")
    print(f"Output:      {output_dir}")

    # ── Load data ONCE (skip for self-loading experiments) ──
    t0 = time.time()
    if all_self_loading:
        print(f"\nSkipping load_data — all experiments load their own data")
        data = {'reps': {}, 'labels': np.array([]),
                'domain': 'federated', 'augmentations': None}
    elif dataset_name in ('herg', 'lhasa_herg'):
        print(f"\nLoading Lhasa hERG data...")
        data = load_lhasa_herg_data(rep_names=rep_names)
        print(f"  Loaded in {time.time() - t0:.1f}s")
    else:
        print(f"\nLoading {dataset_name}...")
        data = load_data(dataset_name, rep_preset='all', rep_names=rep_names)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # NaN guard
        reps_data = data.get('reps', {})
        for rname, X in list(reps_data.items()):
            nan_count = int(np.isnan(X).sum())
            if nan_count > 0:
                pct = 100 * nan_count / X.size
                print(f"  WARNING: {rname} has {nan_count} NaN values "
                      f"({pct:.1f}%) -- imputing 0")
                reps_data[rname] = np.nan_to_num(X, nan=0.0)

        # ── Load augmentation reps ──
        aug_rep_names = DATASET_REPS.get(dataset_name, {}).get(
            'augmentation', [])
        if aug_rep_names:
            missing = [r for r in aug_rep_names if r not in reps_data]
            if missing:
                print(f"  Loading {len(missing)} augmentation reps: "
                      f"{missing}")
                aug_data = load_data(dataset_name, rep_preset='all',
                                     rep_names=missing)
                aug_reps = {}
                for rn, arr in aug_data['reps'].items():
                    if int(np.isnan(arr).sum()) > 0:
                        arr = np.nan_to_num(arr, nan=0.0)
                    aug_reps[rn] = arr
                data['augmentations'] = aug_reps
            else:
                data['augmentations'] = {
                    r: reps_data[r] for r in aug_rep_names}
            print(f"  Augmentations ({len(data['augmentations'])}): "
                  f"{', '.join(f'{k}={v.shape[1]}' for k, v in data['augmentations'].items())}")
        else:
            data['augmentations'] = None

        # ── Precompute hybrid searches (general reps) ──
        _task_type = data.get('task_type', 'regression')
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
            )
            data['precomputed_searches'] = precompute_results
        else:
            data['precomputed_searches'] = {}

        # ── Precompute hybrid searches (top-tier reps) ──
        top_tier_names = DATASET_REPS.get(dataset_name, {}).get('top_tier', [])
        top_tier_reps = {k: v for k, v in reps_data.items()
                         if k in top_tier_names}
        data['top_tier_reps'] = top_tier_reps
        if len(top_tier_reps) >= 2:
            print(f"\n  Precomputing hybrid searches "
                  f"(top-tier, {len(top_tier_reps)} reps)...")
            precompute_top = precompute_hybrid_searches(
                reps=top_tier_reps,
                labels=data['labels'],
                groups=data.get('groups'),
                config=config,
                augmentations=data.get('augmentations'),
                task_type=_task_type,
                domain=data.get('domain', 'molecular'),
            )
            data['precomputed_searches_top'] = precompute_top
        else:
            data['precomputed_searches_top'] = {}

    # ── Run experiments ──
    results = {}
    for exp_id in experiments:
        print(f"\n{'~' * 70}")
        t_exp = time.time()

        try:
            mod = _import_experiment(exp_id)
        except (ImportError, KeyError) as e:
            print(f"  SKIP {exp_id}: import failed ({e})")
            continue

        try:
            exp_result = mod.run(data, config, output_dir, dataset_name)
            results[exp_id] = exp_result

            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(
                exp_result, output_dir, exp_full_name, dataset_name)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': str(e)}

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE: {dataset_name} -- {len(results)} experiments "
          f"in {total:.0f}s")
    print(f"{'=' * 70}")

    return results
