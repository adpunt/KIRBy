#!/usr/bin/env python3
"""N14: Activity-Cliff Landscape Survey + Stage-0 Noise Diagnostics.

Paper: Noise (the GATE experiment — trains NO downstream models).

WHY THIS EXISTS. The housing `noisy_housing` study rests on one contract:

    A large residual is NOT presumed to be a wrong label.

In housing, a "genuine outlier" is a correct-but-atypical price (a hidden gem) —
identified, never corrected. Its chemistry analogue is an ACTIVITY CLIFF: a real
structure-activity discontinuity, not a mislabel. Before porting the housing
correction / weighting / triage families into KIRBy, we must first MEASURE the
activity-cliff landscape, because those methods only earn their keep where cliffs
actually exist and are distinguishable from assay noise. This experiment emits
the go/no-go numbers that decide (a) which method arms are worth building and
(b) whether any dataset warrants a standalone OPIG experiment with real ground
truth.

WHAT IT OUTPUTS, per molecular regression dataset:
  1. A high-confidence activity-cliff list, built from EXTERNAL criteria (the
     chemistry analogue of a chemist hand-marking real cliffs):
       - MMP-based where the loader exposes matched-molecular pairs (chemically
         interpretable, no fingerprint threshold), else ECFP4-Tanimoto structural
         similarity (Bajorath standard, reusing n01's reference);
       - >= 100-fold potency gap (2 log units, |Delta activity| > 2.0) — ~4x the
         Kramer et al. 2012 assay noise SD (~0.54 pKi), so unlikely to be noise.
     A PROVENANCE dict records exactly which filters were actually applied, so
     we never overclaim "high-confidence" when only structure + potency were
     checkable (same-assay and ChEMBL duplicate/transcription-flag filters are
     pre-baked into some cleaned loaders and not separately checkable here).
  2. Cliff PREVALENCE (count + fraction of cliff compounds/pairs).
  3. Cliff GEOMETRY — clustered vs isolated: mean cliff-neighbour degree and
     connected-component sizes over the cliff graph. This decides whether the
     deferred suite should favour neighbour-based methods (clustered) or
     per-point methods (isolated singletons).
  4. FEASIBILITY flags: >= 5 high-confidence cliffs (can the Elkan-Noto PU
     classifier calibrate); has_replicates (reuse n06's builder); has_multi_assay
     (can cross_source_consistency run).
  5. Housing STAGE-0 diagnostics (via from_linear): log-residual kurtosis,
     U-shape heteroscedasticity, segment bias, residual SD vs the Kramer floor.
     from_linear divides labels by their own standard deviation, so everything
     the diagnostics return is in standard deviations of the label. The Kramer
     floor is 0.54 pKi. Every residual quantity therefore also appears in label
     units, under a *_label_units name, and only those may be compared with it.

SCOPE: molecular regression only. Cliffs are a molecular concept, and the
log-price <-> pActivity mapping breaks on nominal labels (n01 already found ~0
cliffs off-molecular). Classification / non-molecular datasets get a clean skip.

2026-10-06 (Adelaide delegated): on a dataset with one row per measurement
(herg_ic50) a molecule's own repeat measurements are left out of its neighbour
lists and are never a cliff edge, in each fold's graph and in the ECFP4/MMP
reference graph. The cliff graph still covers every molecule, the forest is
still one forest on every representation side by side, and the LightGBM
residual fit is kept.

Usage:
    python n14_activity_cliff_landscape.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, cv_split, LARGE_N_SKIP_THRESHOLD,
)
# Reuse the representation-INDEPENDENT cliff reference already in the suite.
from experiments.noise.n01_activity_cliff_framework import (
    _compute_ecfp4_tanimoto_sim, SIMILAR_PERCENTILE,
)
# Reuse the (drug, target) replicate-group builder for the has_replicates flag.
from experiments.noise.n06_bindingdb_ground_truth import (
    build_replicate_groups, compute_replicate_stds,
)
from kirby.utils.noisy_labels import from_linear, stage0_diagnostics

# High-confidence activity-cliff potency threshold: >= 100-fold = 2 log units.
# This is the literature AC definition (Bajorath / MoleculeACE strict / Dablander)
# and is ~4x the ~0.54 pKi Kramer noise SD, so a pair clearing it is very
# unlikely to be a noise artefact.
AC_POTENCY_THRESHOLD = 2.0

# Structural-similarity cutoff for the ECFP4-Tanimoto reference (Bajorath 0.55).
ECFP4_SIM_THRESHOLD = 0.55

# A dataset needs at least this many high-confidence cliffs for the deferred
# Elkan-Noto PU classifier (calibrate_gate) to have a positive class to learn.
PU_MIN_CLIFFS = 5

# Kramer et al. 2012 inter-laboratory assay noise, standard deviation ~0.54 pKi.
# It is quoted in pKi, so anything compared against it must be in label units,
# not in standard deviations of the label.
KRAMER_NOISE_FLOOR_PKI = 0.54

# 2026-10-05 (chat 5): stage0_diagnostics also returns a one-sample t-test
# p-value on the residual mean ('bias_pvalue') and a flag built on it
# ('bias_flag'). Significance tests are excluded everywhere (Adelaide), so
# both are dropped from what this file records, and the label-units bias flag
# that used the same p-value (with a 0.005 cutoff) is gone. The residual mean
# itself is still recorded, in both scales.
STAGE0_SIGNIFICANCE_KEYS = ('bias_pvalue', 'bias_flag')


def _rescale_stage0(stage0, params):
    """Add label-units copies of every residual quantity stage0_diagnostics
    returns, and say which scale each one is on.

    from_linear returns (mu, label_sd) and maps y -> exp((y - mu) / label_sd),
    so log(y_pos) is the label divided by its own standard deviation. Every
    residual the diagnostics compute is therefore in standard deviations of the
    label. Multiplying by label_sd puts them back into the label's own units,
    which is the only scale the Kramer floor can be compared against.

    Spread and bias rescale; kurtosis, the fraction beyond 3 SD and the two
    residual correlations are scale-free and are left alone. The library's
    p-value and the flag built on it are dropped (no significance tests).
    """
    label_sd = float(params[1])
    out = {k: v for k, v in stage0.items()
           if k not in STAGE0_SIGNIFICANCE_KEYS}
    out['label_sd'] = label_sd
    out['residual_scale'] = 'standardized_label_sd'
    out['residual_scale_label_units'] = 'label'
    out['kramer_noise_floor_pki'] = KRAMER_NOISE_FLOOR_PKI
    out['kramer_floor_scale'] = 'label (pKi); compare against the *_label_units fields'

    out['resid_sd_label_units'] = float(stage0['log_resid_sd']) * label_sd
    out['tier_sd_label_units'] = {
        k: (float(v) * label_sd if np.isfinite(v) else float('nan'))
        for k, v in stage0['tier_sd'].items()}
    out['tier_mean_bias_label_units'] = {
        k: (float(v) * label_sd if np.isfinite(v) else float('nan'))
        for k, v in stage0['tier_mean_bias'].items()}
    out['overall_bias_label_units'] = float(stage0['overall_log_bias']) * label_sd
    out['resid_sd_over_kramer_floor'] = (
        out['resid_sd_label_units'] / KRAMER_NOISE_FLOOR_PKI)
    out['resid_sd_at_or_below_kramer_floor'] = bool(
        out['resid_sd_label_units'] <= KRAMER_NOISE_FLOOR_PKI)
    return out


def _cliff_mask_from_pairs(n, pair_iter, potency_threshold,
                           molecule_of_row=None):
    """Build a symmetric (n, n) boolean cliff adjacency from (i, j, delta) MMP
    pairs, marking pairs whose |delta activity| exceeds the potency threshold.
    A pair of two rows of the same molecule (molecule_of_row) is skipped."""
    adj = np.zeros((n, n), dtype=bool)
    n_considered = 0
    for i, j, delta in pair_iter:
        i, j = int(i), int(j)
        if i < 0 or j < 0 or i >= n or j >= n:
            continue
        if (molecule_of_row is not None
                and molecule_of_row[i] == molecule_of_row[j]):
            continue
        n_considered += 1
        if abs(float(delta)) > potency_threshold:
            adj[i, j] = True
            adj[j, i] = True
    np.fill_diagonal(adj, False)
    return adj, n_considered


def _cliff_mask_from_similarity(sim_matrix, labels, sim_threshold,
                                potency_threshold, similar_percentile=None):
    """Cliff adjacency from a similarity matrix + ABSOLUTE potency threshold.

    Unlike n01.identify_cliff_pairs (which uses a MAD-relative activity cutoff),
    the high-confidence AC list uses the literature ABSOLUTE 2-log potency gap,
    so this builds the mask directly. The similarity cutoff is either absolute
    (sim_threshold) or, when a percentile is given, distribution-relative (the
    top `similar_percentile`% most-similar off-diagonal pairs) — mirroring n01's
    handling for reps whose similarity never reaches an absolute 0.55.
    """
    n = len(labels)
    off = sim_matrix[~np.eye(n, dtype=bool)]
    if similar_percentile is not None and off.size:
        sim_cutoff = float(np.percentile(off, 100.0 - similar_percentile))
    else:
        sim_cutoff = float(sim_threshold)
    delta = np.abs(labels[:, None] - labels[None, :])
    adj = (sim_matrix >= sim_cutoff) & (delta > potency_threshold)
    np.fill_diagonal(adj, False)
    return adj, sim_cutoff


def _geometry(adj):
    """Cliff-graph geometry: prevalence, neighbour degree, component structure.

    'Clustered' cliffs (congeneric cliff-rich series) live in large connected
    components with high degree — neighbour-based methods apply. 'Isolated'
    cliffs are singleton compounds (degree-1 in a size-2 component) — only
    per-point / cross-source methods can confirm them.
    """
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    cliff_mask = degree > 0
    n_cliff_compounds = int(cliff_mask.sum())
    n_cliff_pairs = int(adj.sum()) // 2

    out = {
        'n_compounds': int(n),
        'n_cliff_compounds': n_cliff_compounds,
        'n_cliff_pairs': n_cliff_pairs,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / n) if n else 0.0,
        'mean_cliff_degree': float(degree[cliff_mask].mean()) if n_cliff_compounds else 0.0,
        'max_cliff_degree': int(degree.max()) if n else 0,
    }

    if n_cliff_compounds == 0:
        out.update({
            'n_cliff_components': 0, 'largest_component_size': 0,
            'n_singleton_cliff_pairs': 0, 'frac_cliffs_in_singleton_pairs': 0.0,
            'geometry_lean': 'no_cliffs',
        })
        return out, cliff_mask

    # Connected components over the SUBGRAPH induced by cliff compounds.
    idx = np.where(cliff_mask)[0]
    sub = adj[np.ix_(idx, idx)]
    n_comp, comp_labels = connected_components(csr_matrix(sub), directed=False)
    comp_sizes = np.bincount(comp_labels)
    largest = int(comp_sizes.max())
    # A "singleton cliff pair" = an isolated 2-compound component (the isolated
    # hidden-gem geometry: two similar compounds, no wider cliff cluster).
    n_singleton_pairs = int((comp_sizes == 2).sum())
    frac_singletons = float(
        2 * n_singleton_pairs / n_cliff_compounds) if n_cliff_compounds else 0.0

    out.update({
        'n_cliff_components': int(n_comp),
        'largest_component_size': largest,
        'component_size_histogram': {int(s): int(c) for s, c in
                                     zip(*np.unique(comp_sizes, return_counts=True))},
        'n_singleton_cliff_pairs': n_singleton_pairs,
        'frac_cliffs_in_singleton_pairs': frac_singletons,
        # Clustered if the largest component holds a meaningful share and mean
        # degree exceeds 1 (compounds participate in multiple cliffs); else the
        # landscape is dominated by isolated pairs.
        'geometry_lean': (
            'clustered' if (largest >= 3 and out['mean_cliff_degree'] > 1.0)
            else 'isolated'),
    })
    return out, cliff_mask


def _molecule_of_row(data, dataset_name, n):
    """Which rows are the same molecule, or None when every row is its own
    molecule.

    DECIDED 2026-10-06 (Adelaide delegated): on a dataset with one row per
    measurement (herg_ic50, 12,008 rows for 9,802 molecules) a molecule's own
    repeat measurements are left out of its neighbour lists and are never a
    cliff edge. Two measurements of one molecule have ECFP4 Tanimoto 1 and
    sit next to each other in every space, so before this two repeats more
    than AC_POTENCY_THRESHOLD apart counted as a cliff. The ids are
    data['replicate_groups'], the grouping n01 and n22 use; herg_ic50 falls
    back to its loader when the run's data does not carry them, as n22 does.
    """
    m = data.get('replicate_groups')
    if m is None and dataset_name == 'herg_ic50':
        try:
            from kirby.datasets.herg import load_herg_ic50_measurements
            m = load_herg_ic50_measurements()['replicate_groups']
        except Exception as exc:
            print(f"  WARNING: herg_ic50 molecule ids could not be loaded "
                  f"({type(exc).__name__}: {exc}); repeats are not excluded")
            return None
    if m is None:
        return None
    m = np.asarray(m)
    if len(m) != n:
        print(f"  WARNING: replicate_groups has {len(m)} entries for {n} "
              f"rows; repeats are not excluded")
        return None
    if len(np.unique(m)) == n:
        return None
    return m


def _build_ac_reference(data, dataset_name, labels, molecule_of_row=None):
    """Return (cliff_adjacency, provenance) for the high-confidence AC list.

    MMP mode fires when the loader exposes matched-molecular pairs aligned to the
    data rows (`data['mmp_pairs']`, as chembl_qsar produces). Otherwise falls
    back to the ECFP4-Tanimoto structural reference. Provenance lists exactly
    which filters were applied so downstream readers can gauge confidence.
    With molecule_of_row, two rows of the same molecule are never a cliff
    pair (2026-10-06).
    """
    n = len(labels)
    n_repeat_pairs = 0
    if molecule_of_row is not None:
        _, _cnt = np.unique(molecule_of_row, return_counts=True)
        n_repeat_pairs = int((_cnt * (_cnt - 1) // 2).sum())
    provenance = {
        'repeat_measurements_excluded': molecule_of_row is not None,
        'n_repeat_pairs_excluded': n_repeat_pairs,
        'potency_threshold_log_units': AC_POTENCY_THRESHOLD,
        'potency_criterion': '>=100-fold (2 log units)',
        # These two are honestly unavailable to this survey: same-assay and
        # ChEMBL dup/transcription flags are pre-applied inside cleaned loaders
        # (e.g. Dablander CSVs) and are NOT re-checkable from `data` here.
        'same_assay_filter_applied': False,
        'chembl_flag_filter_applied': False,
        'note_unchecked_filters': (
            'same-assay and ChEMBL duplicate/transcription-error filters are not '
            'checkable from the loaded data; treat "high-confidence" as MMP/'
            'structure + potency only unless the source loader pre-applied them.'),
    }

    mmp_pairs = data.get('mmp_pairs')
    if mmp_pairs:
        adj, n_considered = _cliff_mask_from_pairs(
            n, mmp_pairs, AC_POTENCY_THRESHOLD, molecule_of_row)
        provenance.update({
            'similarity_criterion': 'matched_molecular_pair',
            'similarity_source': 'loader_mmp_pairs',
            'mmp_pairs_considered': int(n_considered),
        })
        return adj, provenance

    # Structural fallback: ECFP4-Tanimoto (reuses n01). Requires SMILES.
    smiles = data.get('smiles')
    if smiles is None:
        raise ValueError(
            "No MMP pairs and no SMILES available; cannot build an activity-"
            "cliff reference for this dataset.")
    sim = _compute_ecfp4_tanimoto_sim(smiles)
    if molecule_of_row is not None:
        # Similarity 0 between two rows of one molecule: never a cliff pair,
        # and counted as dissimilar by the percentile fallback below (as in
        # n01).
        sim = np.where(molecule_of_row[:, None] == molecule_of_row[None, :],
                       0.0, sim)
        np.fill_diagonal(sim, 0.0)
    # Absolute 0.55 first; if it flags nothing (rep-dependent), fall back to the
    # distribution-relative top-percentile cutoff n01 uses.
    adj, sim_cutoff = _cliff_mask_from_similarity(
        sim, labels, ECFP4_SIM_THRESHOLD, AC_POTENCY_THRESHOLD)
    similar_percentile = None
    if adj.sum() == 0:
        adj, sim_cutoff = _cliff_mask_from_similarity(
            sim, labels, ECFP4_SIM_THRESHOLD, AC_POTENCY_THRESHOLD,
            similar_percentile=SIMILAR_PERCENTILE)
        similar_percentile = SIMILAR_PERCENTILE
    provenance.update({
        'similarity_criterion': 'ecfp4_tanimoto',
        'similarity_source': 'computed_from_smiles',
        'similarity_cutoff': float(sim_cutoff),
        'similar_percentile': similar_percentile,
    })
    return adj, provenance


# What the similarity part of this file writes, since chat 5 (2026-10-05,
# HANDOFF_EXPERIMENT_ALIGNMENT section 2.4). Written into every checkpoint
# fingerprint, so a geometry line from the whole-dataset design before that
# date (rf's top 2 from the stored search, every molecule in one graph) is
# recomputed, never read back.
# 2026-10-06: a molecule's own repeat measurements left out of its
# neighbour lists and of the cliff edges (changes herg_ic50 only).
RESULT_LAYOUT = 'fold_aligned_every_molecule_repeats_excluded_2026-10-06'

# The cliff-graph numbers summarised over folds (noise_helpers.fold_spread).
GEOMETRY_SUMMARY_FIELDS = ('n_cliff_compounds', 'n_cliff_pairs',
                           'pct_cliff_compounds', 'mean_cliff_degree',
                           'largest_component_size',
                           'frac_cliffs_in_singleton_pairs')


def _knn_cliff_geometry(idx, y, molecule_of_row=None):
    """_geometry of the graph in which two molecules are joined when one is
    among the other's k nearest (idx: (n, k) neighbour indices into y) and
    their labels differ by more than AC_POTENCY_THRESHOLD. Two rows of the
    same molecule (molecule_of_row, aligned with y) are never joined; the
    lists already leave them out, so this only guards against a list built
    without them."""
    n = len(y)
    adj = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = np.asarray(idx).ravel()
    gap = np.abs(y[rows] - y[cols]) > AC_POTENCY_THRESHOLD
    if molecule_of_row is not None:
        gap &= molecule_of_row[rows] != molecule_of_row[cols]
    adj[rows[gap], cols[gap]] = True
    adj |= adj.T
    np.fill_diagonal(adj, False)
    return _geometry(adj)[0]


def _geometry_per_setting(nbrs, settings, y, unavailable,
                          molecule_of_row=None):
    """{setting name: {str(k): geometry}} for every setting and every k its
    lists hold. A setting with no lists goes into `unavailable` with why."""
    from experiments.noise.n07_remediation import usable_ks
    from experiments.noise.similarity_space import neighbours_at
    out = {}
    for st in settings:
        L = nbrs.get(st.name)
        if L is None or not L.get('available'):
            unavailable.setdefault(st.name, (L or {}).get('reason')
                                   or 'no neighbour lists were built')
            continue
        for k in usable_ks(L):
            out.setdefault(st.name, {})[str(k)] = _knn_cliff_geometry(
                neighbours_at(L, k)[0], y, molecule_of_row)
    return out


def _forest_lists_every_molecule(dataset_name, setting, X_all, train_idx,
                                 val_idx, y_train, exclude_groups=None):
    """forest_proximity neighbour lists over every molecule of one fold.

    The forest is fitted on the fold's training rows and training labels
    (noise_cache.cached_forest_leaf_ids, the forest prox_smooth fits), on
    X_all scaled with kirby.scaling.FeatureScaler fitted on the training rows
    (0/1 features unscaled). The held-out rows are put through the same
    forest. Lists then run over every molecule, training rows first and
    held-out rows after, each molecule's own row left out, the same
    distance and sort as similarity_space.neighbour_lists uses for this
    space. Added 2026-10-05: neighbour_lists itself gives held-out rows
    neighbours among the training rows only, not among each other.
    exclude_groups (one molecule id per row, training rows first): a row
    never takes another row of its own molecule (2026-10-06)."""
    from kirby.scaling import FeatureScaler
    from experiments.noise import noise_cache as nc
    from experiments.noise import similarity_space as ss
    if X_all is None:
        return {'available': False, 'setting_name': setting.name,
                'reason': 'no representations were loaded for the forest'}
    try:
        sc = FeatureScaler().fit(X_all[train_idx])
        L_tr, L_va, O_tr = nc.cached_forest_leaf_ids(
            dataset_name, sc.transform(X_all[train_idx]), y_train, False,
            X_apply=sc.transform(X_all[val_idx]))
        L = np.concatenate([L_tr, L_va], axis=0)
        # Validation molecules were never in any bootstrap: out of bag for
        # every tree (2026-10-26, out-of-bag proximity).
        O = np.concatenate([O_tr, np.ones(L_va.shape, dtype=bool)], axis=0)
        n = len(L)
        k = ss._k_eff(n, int(setting.k_max), exclude_groups)
        idx, dist = ss._lists_from_distance(
            lambda a, b: ss._forest_distance(L[a:b], L, O[a:b], O), n, n, k,
            self_rows=True, exclude_groups=exclude_groups, itemsize=4,
            sorter='numpy')
    except Exception as exc:
        return {'available': False, 'setting_name': setting.name,
                'reason': f'{type(exc).__name__}: {exc}'}
    return {'setting': setting.to_dict(), 'setting_name': setting.name,
            'available': True, 'reason': None, 'k_eff': k,
            'idx_train': idx, 'dist_train': dist,
            'idx_val': None, 'dist_val': None}


def _cliff_geometry_by_similarity(data, dataset_name, labels, output_dir,
                                  config, molecule_of_row=None):
    """The cliff graph's geometry under every similarity setting and k, per
    outer fold, over every molecule, in spaces built on that fold's
    training rows.

    Added 2026-09-25 (plan sections 1-3). Rewritten 2026-10-05 (chat 5,
    HANDOFF_EXPERIMENT_ALIGNMENT 2.4; n14 is in group A of 2.3 "Still open":
    no models, keep its design, but its similarity spaces follow the kNN
    record).

    Until 2026-10-05 this ran once over every molecule, with rf's top 2
    single representations from the whole-dataset stored search as base_1
    and base_2, and no hybrid or augmented space (a recipe chooses features
    from labels, and there was no fold to choose them on). rf is not a
    decided model on any dataset n14 reaches, and no whole-dataset search may
    be read (R1, R2).

    Now, per outer fold (noise_helpers.noise_outer_folds):
      - noise_helpers.fold_configurations runs this fold's searches on its
        training rows and gives each of the dataset's two decided models its
        2 best single representations (base_1, base_2), its 2 best general
        hybrids, and the recipes of the distance hybrid and the similarity
        hybrid.
      - The settings are n07.sweep_settings for this dataset's group
        (KNN_SEARCH_SPACE_RECORD): no model_input setting, no importance
        weighting, the group's distances, Tanimoto only with SMILES.
        Scaffold-mates stay in the neighbour lists (axis 6).
      - Per model: every setting whose space depends on the model (the base,
        augmented and hybrid spaces), built on the fold's training rows from
        the fold's training labels (n07.fold_similarity_spaces).
      - Once per fold, under 'all_models': tanimoto_ecfp4 and
        forest_proximity, which do not depend on the model.
      - The graph is over every molecule (training rows first, held-out
        rows passed through the spaces built on the training rows), as the
        reference graph is: an edge joins two molecules when one is among the other's k nearest and their labels
        differ by more than AC_POTENCY_THRESHOLD, for k in 5, 10, 20.
      - On a dataset with one row per measurement (herg_ic50) a molecule's
        own repeat measurements are left out of its neighbour lists and are
        never an edge (2026-10-06).
    The counting scheme (rank, inverse_square) is not used: the graph reads
    which molecules are neighbours, never a weighted neighbour label. Nothing
    is refitted per setting, so no screen is needed; every setting is scored.

    Each (fold, unit, setting) whose lists were built is written to a
    checkpoint as it finishes and read back on resume; an unavailable
    setting is not written, so the next run retries it.
    """
    from experiments.decided_grid import decided_models
    from experiments._shared import (get_domain_metric, get_model_factories,
                                     input_fingerprint, REP_DTYPE)
    from experiments.noise.noise_helpers import (
        noise_outer_folds, fold_configurations, configuration_record,
        fold_spread, dataset_knn_group)
    from experiments.noise.n07_remediation import (
        sweep_settings, sweep_ks, fold_neighbours, fold_similarity_spaces,
        FoldCheckpoint, similarity_record)
    from experiments.noise import similarity_space as _ss

    task_type = data['task_type']
    reps = data.get('reps') or {}
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    knn_grp = dataset_knn_group(data, dataset_name)
    settings = sweep_settings(dataset_name, n_molecules, has_smiles)
    # tanimoto_ecfp4 and forest_proximity read no model's representations, so
    # they are built once per fold; every other setting once per model.
    shared_spaces = (_ss.TANIMOTO_SPACE, _ss.FOREST_SPACE)
    shared_settings = [s for s in settings if s.space in shared_spaces]
    model_settings = [s for s in settings if s.space not in shared_spaces]
    metric = get_domain_metric(data)
    metric_name, _metric_fn, lower_is_better = metric
    # R1: the dataset's two decided models (DECISIONS.md Step 1). n14 fits
    # none of them; they decide which representations and hybrids are the
    # similarity spaces, and each hybrid rebuild's accept-or-reject step
    # scores with that model, as in every other noise experiment.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))

    # forest_proximity's forest is fitted on this matrix: every loaded
    # representation side by side, as before 2026-10-05. Kept in float32
    # (handoff 2.3 item 18).
    # DECIDED 2026-10-06 (Adelaide delegated): one forest per fold, fitted on
    # every representation side by side, as now. In the experiments that fit
    # models forest_proximity's forest is fitted on the configuration's own
    # matrix; n14 has no configuration, so there is no per-space forest.
    X_all = None
    if reps:
        X_all = np.hstack([np.nan_to_num(
            np.asarray(reps[k]).astype(REP_DTYPE, copy=False), nan=0.0)
            for k in sorted(reps)])

    # Which molecules the graph covers: every molecule, as before 2026-10-05,
    # so each fold's graph covers the same molecules as the ECFP4/MMP
    # reference graph above. The spaces are built on the fold's training rows
    # (recipes, scalers and the forest are fitted there) and the held-out
    # molecules are passed through the same build; the graph then joins any
    # two molecules, training or held-out. (Changed 2026-10-05 after review:
    # the first chat-5 version covered only the training molecules, which
    # changed the survey without a decision.)
    # DECIDED 2026-10-06 (Adelaide delegated): the graph covers every
    # molecule, as now (not each fold's training molecules only).
    # DECIDED 2026-10-06 (Adelaide delegated): herg_ic50 keeps one row per
    # measurement, and a molecule's own repeat measurements are left out of
    # its neighbour lists (exclude_groups = molecule ids) and are never a
    # cliff edge, here and in the ECFP4/MMP reference graph. Before this a
    # molecule's repeats were its nearest neighbours, and two repeats more
    # than AC_POTENCY_THRESHOLD apart counted as a cliff edge. None on a
    # dataset with one row per molecule, which keeps its lists as they were.
    folds, n_folds = noise_outer_folds(data, config)
    print(f"  Similarity geometry: {len(settings)} settings ({knn_grp}), "
          f"k {list(sweep_ks())}, {len(folds)} of {n_folds} folds, spaces "
          f"from models {models}")

    # One checkpoint line per (fold, unit, setting) whose neighbour lists were
    # built, plus one per (fold, model) for the configurations. A setting that
    # came back unavailable is NOT stored, so the next run retries it (a
    # drive dropout or a failed build does not freeze it).
    ckpt = FoldCheckpoint(output_dir, 'n14_activity_cliff_landscape',
                          dataset_name, tag='geometry', fingerprint={
                              'layout': RESULT_LAYOUT,
                              'input': input_fingerprint(data, config),
                              'potency': AC_POTENCY_THRESHOLD,
                              'n_folds': int(n_folds), 'models': models,
                              'settings': [s.name for s in settings],
                              'ks': list(sweep_ks())})

    def _from_ckpt(fold_i, unit, unit_settings):
        geo, todo = {}, []
        for st in unit_settings:
            v = ckpt.get([int(fold_i), unit, st.name])
            if v is None:
                todo.append(st)
            else:
                geo[st.name] = v
        return geo, todo

    def _store(fold_i, unit, geo_new, geo):
        for sname, by_k in geo_new.items():
            if by_k:
                ckpt.put([int(fold_i), unit, sname], by_k)
            geo[sname] = by_k

    per_fold = {}
    for fold_i, train_idx, val_idx in folds:
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        # Graph rows: the training molecules first, then the held-out ones.
        # Neighbour indices point into this order; y_graph matches it.
        order = np.concatenate([train_idx, val_idx])
        y_tr = labels[train_idx]
        y_graph = labels[order]
        mol_graph = (None if molecule_of_row is None
                     else molecule_of_row[order])
        g_tr = None if groups_arr is None else groups_arr[train_idx]
        print(f"    Fold {fold_i} of {n_folds}: spaces from {len(train_idx)} "
              f"training molecules, graph over {len(order)} molecules")
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'n_graph': int(len(order)), 'units': {}}

        # --- tanimoto_ecfp4 and forest_proximity, once per fold ---------
        geo, todo = _from_ckpt(fold_i, 'all_models', shared_settings)
        unav = {}
        for st in todo:
            if st.space == _ss.FOREST_SPACE:
                nb = {st.name: _forest_lists_every_molecule(
                    dataset_name, st, X_all, train_idx, val_idx, y_tr,
                    exclude_groups=mol_graph)}
            else:
                smi = [smiles[i] for i in order] if has_smiles else None
                nb = fold_neighbours(dataset_name, [st], {}, None, y_graph,
                                     exclude_groups=mol_graph,
                                     smiles_train=smi, is_clf=False)
            _store(fold_i, 'all_models',
                   _geometry_per_setting(nb, [st], y_graph, unav, mol_graph),
                   geo)
        frec['units']['all_models'] = {'geometry': geo, 'unavailable': unav}

        # --- every other setting, once per decided model ----------------
        entries = None
        for m in models:
            geo, todo = _from_ckpt(fold_i, m, model_settings)
            rec = ckpt.get([int(fold_i), m, 'configurations'])
            unav = {}
            if todo or rec is None:
                if entries is None:
                    try:
                        entries = fold_configurations(
                            data, dataset_name, fold_i=fold_i,
                            train_idx=train_idx, n_folds=n_folds)
                    except Exception as exc:
                        entries = {}
                        frec['error'] = (f'per-fold searches failed: '
                                         f'{type(exc).__name__}: {exc}')
                        print(f"      {frec['error']}")
                entry = entries.get(m)
                if entry is None:
                    # Not checkpointed: a failed search is retried next run.
                    frec['units'][m] = {
                        'configurations': {'available': False},
                        'geometry': geo,
                        'why': frec.get('error') or
                        'the per-fold search failed for this model'}
                    continue
                if rec is None:
                    rec = configuration_record(entry)
                    rec['metric'] = metric_name
                    rec['lower_is_better'] = bool(lower_is_better)
                    ckpt.put([int(fold_i), m, 'configurations'], rec)
            if todo:
                # Only the settings not yet on disk are built.
                spaces_tr, spaces_va, unav_b = fold_similarity_spaces(
                    dataset_name, todo, model_input_tr=None,
                    model_input_va=None, reps=reps, train_idx=train_idx,
                    val_idx=val_idx, y_train=y_tr, groups_train=g_tr,
                    base_reps=entry.get('base') or [],
                    recipes=entry.get('recipes') or {},
                    augmentations=data.get('augmentations'), model_name=m,
                    model_factory=factories.get(m), metric=metric,
                    task_type=task_type)
                unav.update(unav_b)
                # Every molecule in one matrix per space: training rows
                # first, held-out rows after, as in `order`.
                spaces_graph = {
                    sp: np.concatenate([np.asarray(Z_tr),
                                        np.asarray(spaces_va[sp])], axis=0)
                    for sp, Z_tr in spaces_tr.items()
                    if spaces_va.get(sp) is not None}
                nb = fold_neighbours(dataset_name, todo, spaces_graph, None,
                                     y_graph, exclude_groups=mol_graph,
                                     is_clf=False)
                _store(fold_i, m,
                       _geometry_per_setting(nb, todo, y_graph, unav,
                                             mol_graph), geo)
            else:
                print(f"      {m}: read back from {ckpt.path}")
            frec['units'][m] = {'configurations': rec, 'geometry': geo,
                                'unavailable': unav}
            print(f"      {m}: base {rec.get('base')}, {len(geo)} settings "
                  f"scored, {len(unav)} unavailable")
        per_fold[str(fold_i)] = frec

    # Summaries over folds: mean and spread, or "one fold, no spread". No
    # significance test.
    summary = {}
    for unit in ['all_models'] + models:
        by_setting = {}
        for frec in per_fold.values():
            geo = (frec['units'].get(unit) or {}).get('geometry') or {}
            for sname, by_k in geo.items():
                for k, g in by_k.items():
                    by_setting.setdefault(sname, {}).setdefault(k, []).append(g)
        summary[unit] = {
            sname: {k: {f: fold_spread([g.get(f) for g in gs])
                        for f in GEOMETRY_SUMMARY_FIELDS}
                    for k, gs in by_k.items()}
            for sname, by_k in by_setting.items()}

    meta = similarity_record(
        settings, None, knn_group=knn_grp, models=models,
        n_folds=int(n_folds), folds_run=[int(f) for f, _, _ in folds],
        selection_scope='fold_train_rows', metric=metric_name,
        lower_is_better=bool(lower_is_better),
        settings_per_model=[s.name for s in model_settings],
        settings_once_per_fold=[s.name for s in shared_settings],
        neighbour_schemes=('not used: the cliff graph reads which molecules '
                           'are neighbours, not a weighted label'),
        scaffold_mates_excluded=False,
        repeat_measurements_excluded=molecule_of_row is not None,
        graph_rows=('every molecule; spaces fitted on the fold\'s training '
                    'rows, held-out molecules passed through them'),
        cliff_rule=('an edge joins two molecules when one is among the '
                    "other's k nearest in the setting and |delta label| > "
                    f'{AC_POTENCY_THRESHOLD}'),
        checkpoint_file=ckpt.path, read_back=ckpt.n_resumed,
        result_layout=RESULT_LAYOUT)
    return {'per_fold': per_fold, 'summary': summary}, meta


def run(data, config, output_dir, dataset_name, **kwargs):
    """N14: activity-cliff landscape survey. Trains no downstream models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N14: ACTIVITY-CLIFF LANDSCAPE SURVEY")
    print("=" * 70)

    domain = data.get('domain', 'molecular')
    task_type = data['task_type']
    labels = np.asarray(data['labels'], dtype=float).ravel()
    groups = data.get('groups')
    n = len(labels)

    # --- Scope guards (clean skip markers, not errors) ---
    if task_type == 'classification':
        print("  SKIP: classification — the AC/potency mapping needs a "
              "continuous activity endpoint.")
        return {'skipped': True, 'reason': 'classification out of scope',
                'policy': 'regression_only', 'domain': domain, 'n_samples': n}
    if domain != 'molecular':
        print(f"  SKIP: domain={domain} — activity cliffs are a molecular "
              "concept (n01 found ~0 cliffs off-molecular).")
        return {'skipped': True, 'reason': f'non-molecular domain ({domain})',
                'policy': 'molecular_only', 'domain': domain, 'n_samples': n}
    if n > LARGE_N_SKIP_THRESHOLD:
        print(f"  SKIP: n={n} > {LARGE_N_SKIP_THRESHOLD} — builds an n x n "
              "similarity matrix.")
        return {'skipped': True, 'reason': f'large-n OOM guard n={n}',
                'policy': 'large_n_oom', 'domain': domain, 'n_samples': n}

    print(f"  Dataset: {dataset_name} ({domain}, n={n})")
    print(f"  Label range [{labels.min():.2f}, {labels.max():.2f}], "
          f"MAD={np.median(np.abs(labels - np.median(labels))):.2f}")

    # --- 1-2-3. High-confidence AC list, prevalence, geometry ---
    # 2026-10-06: which rows are the same molecule (herg_ic50); None when
    # every row is its own molecule.
    molecule_of_row = _molecule_of_row(data, dataset_name, n)
    if molecule_of_row is not None:
        print(f"  {n} rows for {len(np.unique(molecule_of_row))} molecules: "
              f"a molecule's own repeats are left out of its neighbours and "
              f"are never a cliff edge")
    adj, provenance = _build_ac_reference(data, dataset_name, labels,
                                          molecule_of_row)
    geometry, cliff_mask = _geometry(adj)
    n_cliffs = geometry['n_cliff_compounds']
    print(f"  Cliff reference: {provenance['similarity_criterion']} + "
          f"{provenance['potency_criterion']}")
    print(f"  Cliffs: {n_cliffs} compounds ({geometry['pct_cliff_compounds']:.1f}%), "
          f"{geometry['n_cliff_pairs']} pairs, lean={geometry['geometry_lean']}")

    # Label-spread context so a reader can judge whether the absolute 2-log
    # threshold is meaningful for THIS endpoint (it is literal 100-fold only for
    # genuine potency endpoints, pIC50/pKi/pKd).
    label_mad = float(np.median(np.abs(labels - np.median(labels))))
    label_context = {
        'label_mad': label_mad,
        'label_std': float(labels.std()),
        'label_range': [float(labels.min()), float(labels.max())],
        'potency_threshold_in_mad_units': (
            float(AC_POTENCY_THRESHOLD / label_mad) if label_mad > 0 else None),
        'note': ('the 2-log threshold is literal 100-fold potency only for '
                 'genuine activity endpoints (pIC50/pKi/pKd); for other '
                 'regression targets it is a fixed 2-unit label gap.'),
    }

    # --- 3b. The same cliff graph under every similarity setting ---------
    # Added 2026-09-25 (plan sections 1 and 6: the cliff definitions in n01,
    # n14 and n25 take a similarity setting). The graph above is one
    # similarity -- MMP pairs, or ECFP4 Tanimoto at 0.55 -- and is unchanged.
    # Here an edge joins two molecules when one is among the other's k nearest
    # in the setting's features and their labels differ by more than
    # AC_POTENCY_THRESHOLD, for every k in SWEPT_KS, and _geometry reads the
    # graph the same way. Since 2026-10-05 this runs per outer fold, over
    # every molecule, with each decided model's spaces built from the fold's
    # own searches on its training rows (see _cliff_geometry_by_similarity).
    geometry_by_similarity, sim_meta = _cliff_geometry_by_similarity(
        data, dataset_name, labels, output_dir, config,
        molecule_of_row=molecule_of_row)

    # --- 4. Feasibility flags for the deferred method families ---
    rep_groups = build_replicate_groups(data)
    has_replicates = False
    n_replicate_samples = 0
    if rep_groups is not None:
        _stds, has_rep = compute_replicate_stds(labels, rep_groups)
        has_replicates = bool(has_rep.any())
        n_replicate_samples = int(has_rep.sum())
    meta = data.get('metadata') or {}
    has_multi_assay = bool(meta.get('assay_ids') is not None or
                           meta.get('n_assays', 0) and meta.get('n_assays', 0) > 1)

    feasibility = {
        'n_high_conf_cliffs': int(n_cliffs),
        'pu_calibration_feasible': bool(n_cliffs >= PU_MIN_CLIFFS),
        'pu_min_cliffs': PU_MIN_CLIFFS,
        'has_replicates': has_replicates,
        'n_replicate_samples': n_replicate_samples,
        'has_multi_assay': has_multi_assay,
    }
    print(f"  Feasibility: PU-calibratable={feasibility['pu_calibration_feasible']}, "
          f"replicates={has_replicates} (n={n_replicate_samples}), "
          f"multi_assay={has_multi_assay}")

    # --- 5. Housing stage-0 diagnostics, rescaled back to label units ---
    # Feature space = concatenation of all base reps (the model-agnostic view;
    # matches n01's Euclidean-concat fallback).
    #
    # from_linear divides the labels by their own standard deviation before
    # exponentiating, so every residual quantity stage0_diagnostics returns is
    # in units of "one label standard deviation", NOT pKi. The Kramer et al.
    # 2012 assay floor of 0.54 is in pKi. The two only coincide when a dataset's
    # label standard deviation is 1.0. On herg_ki it is about 1.2, so a returned
    # 0.45 is 0.54 pKi — exactly at the floor, while reading as well under it.
    # _rescale_stage0 multiplies the residual spread and the two bias terms by
    # that standard deviation and records which scale each number is on.
    reps = data.get('reps', {})
    stage0 = None
    stage0_error = None
    if reps and groups is not None:
        try:
            # float32, as every representation is loaded (handoff 2.3 item
            # 18; was dtype=float, i.e. float64, until 2026-10-05).
            # stage0_diagnostics casts to float64 itself (diagnostics.py), so
            # the number it returns does not change.
            from experiments._shared import REP_DTYPE
            from experiments.noise.noise_helpers import noise_n_folds
            X_concat = np.hstack([np.asarray(reps[k]).astype(REP_DTYPE,
                                                             copy=False)
                                  for k in sorted(reps.keys())])
            X_concat = np.nan_to_num(X_concat, nan=0.0)
            y_pos, _params = from_linear(labels)
            _label_sd = float(_params[1])
            # The residual fit's own folds: every molecule is predicted once,
            # by a LightGBM fitted on the other folds (core.oof_predict needs
            # that full coverage, so it keeps cv_split and needs at least 2).
            # These are not the outer folds of the similarity part above. The
            # count is the one fold setting (config['n_splits'], or
            # KIRBY_NOISE_N_FOLDS), no longer a literal 5 (2026-10-05).
            n_splits = max(2, noise_n_folds(config))
            # ⛔ DO NOT RAISE THE THREAD COUNT HERE. Tried on 2026-09-18 and
            # it SEGFAULTS: torch links libiomp5 and LightGBM links libomp,
            # and a multi-threaded LightGBM fit in a process that has already
            # started torch's pool kills it with no traceback
            # (project_fold_hook_and_lgbm_segfault_2026_09_02). This process
            # has imported torch long before it gets here. KIRBY_LGBM_THREADS
            # is the switch, it is set to 1 locally for that reason, and this
            # fit inherits it.
            #
            # What that costs, measured here: this is the widest matrix
            # anything in the suite fits -- every loaded representation glued
            # together, 16,405 features on herg_ki against 1,412 molecules --
            # and one LightGBM at 600 trees over 5 folds took more than 14
            # minutes at 1.5 GB resident on one thread. The fix is the
            # dimension cap NOISE_AUDIT_2026_09_10.md asks for at n14:399,
            # NOT more threads, and it is a change to what the residual spread
            # is measured on, so it is not made here.
            #
            # DECIDED 2026-10-06 (Adelaide delegated): the LightGBM residual
            # fit is kept as it is, on every representation concatenated,
            # with no dimension cap.
            #
            # 11 features per molecule is also a regime where the surrogate can
            # memorise, which biases the residual spread DOWN and makes the
            # labels read cleaner than they are. The width is printed so that
            # is visible beside the number.
            _w = int(X_concat.shape[1])
            print(f"  Stage-0 feature space: {_w} features over "
                  f"{len(labels)} molecules (every representation "
                  f"concatenated, no dimension cap)")
            # Written to a checkpoint when it finishes and read back on resume
            # (2026-09-25): the widest fit in the suite, more than 14 minutes
            # on herg_ki, and the only part of n14 that fits anything.
            from experiments.noise.n07_remediation import FoldCheckpoint
            from experiments._shared import input_fingerprint
            _s0ck = FoldCheckpoint(output_dir, 'n14_activity_cliff_landscape',
                                   dataset_name, tag='stage0', fingerprint={
                                       'layout': RESULT_LAYOUT,
                                       'input': input_fingerprint(data, config),
                                       'n_splits': int(n_splits),
                                       'width': _w})
            stage0 = _s0ck.cached(['stage0'], lambda: _rescale_stage0(
                stage0_diagnostics(X_concat, y_pos, groups, n_splits=n_splits,
                                   splitter=cv_split), _params))
            print(f"  Stage-0: resid_sd={stage0['resid_sd_label_units']:.3f} "
                  f"label units (={stage0['log_resid_sd']:.3f} SDs of a "
                  f"{_label_sd:.3f}-SD label), "
                  f"Kramer floor {KRAMER_NOISE_FLOOR_PKI} pKi, "
                  f"kurtosis={stage0['excess_kurtosis']:.2f}, "
                  f"U-shape={stage0['u_shaped_heteroscedasticity']}, "
                  f"residual mean={stage0['overall_bias_label_units']:.3f} "
                  f"label units")
        except Exception as exc:
            stage0_error = f"{type(exc).__name__}: {exc}"
            print(f"  Stage-0 diagnostics failed: {stage0_error}")
    else:
        stage0_error = 'no reps or no groups available'

    # --- Decision hints (derived, for the go/no-go on n15-n17 + OPIG) ---
    decision_hint = {
        # Enough cliff structure to make gated_shrinkage differ from plain
        # shrinkage and to give triage something to separate.
        'cliffs_present': bool(n_cliffs > 0),
        'cliff_prevalence_pct': geometry['pct_cliff_compounds'],
        'geometry_lean': geometry['geometry_lean'],
        # Warrants a standalone OPIG experiment with REAL ground truth: high
        # prevalence + clustered + replicate-verifiable cliffs.
        'opig_standalone_candidate': bool(
            geometry['pct_cliff_compounds'] >= 5.0
            and geometry['geometry_lean'] == 'clustered'
            and has_replicates and n_cliffs >= PU_MIN_CLIFFS),
    }

    results = {
        'experiment': 'n14_activity_cliff_landscape',
        'dataset': dataset_name,
        'domain': domain,
        'n_samples': n,
        'trains_downstream_models': False,
        'ac_reference_provenance': provenance,
        'label_context': label_context,
        'cliff_geometry': geometry,
        'feasibility': feasibility,
        'stage0_diagnostics': stage0,
        'stage0_error': stage0_error,
        'decision_hint': decision_hint,
        # 2026-10-05 (chat 5): {'per_fold': {fold: {'n_train', 'n_val',
        # 'units': {'all_models' | model: {'configurations' (models only),
        # 'geometry': {setting: {k: _geometry}}, 'unavailable'}}}},
        # 'summary': {unit: {setting: {k: {field: fold_spread}}}}}.
        # Until then {setting: {k: geometry}} over every molecule.
        'cliff_geometry_by_similarity': geometry_by_similarity,
        'similarity': sim_meta,
        'n_folds': sim_meta.get('n_folds'),
        'selection_scope': 'fold_train_rows',
        'metric': sim_meta.get('metric'),
        'lower_is_better': sim_meta.get('lower_is_better'),
        'result_layout': RESULT_LAYOUT,
        'repeat_measurements_excluded': molecule_of_row is not None,
    }
    print(f"  OPIG standalone candidate: {decision_hint['opig_standalone_candidate']}")
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N14: Activity-Cliff Landscape Survey')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output,
                            'n14_activity_cliff_landscape', args.dataset)
