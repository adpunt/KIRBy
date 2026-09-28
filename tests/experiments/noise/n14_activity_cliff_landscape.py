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
     U-shape heteroscedasticity, segment bias, residual SD vs the dataset's
     published assay error (assay_noise_floor: herg_ki 0.54, caco2 0.35,
     logd 0.48, in label units; none elsewhere).
     from_linear divides labels by their own standard deviation, so everything
     the diagnostics return is in standard deviations of the label. The assay
     floor is in label units. Every residual quantity therefore also appears in
     label units, under a *_label_units name, and only those may be compared
     with it.

SCOPE: molecular regression only. Cliffs are a molecular concept, and the
log-price <-> pActivity mapping breaks on nominal labels (n01 already found ~0
cliffs off-molecular). Classification / non-molecular datasets get a clean skip.

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

# Added to the threshold in every "gap > threshold" test, so a gap that is
# exactly 2 in the labels' decimals is never a cliff. Labels such as logd's
# are given to one decimal, and in binary 3.1 - 1.1 comes out a hair above
# 2.0 while 3.6 - 1.6 does not. Until 2026-09-28 that rounding decided the
# pair: on logd 16 of the 1,285 similar pairs whose gap is exactly 2.0 were
# counted as cliffs (7,336 against 7,320) and the other 1,269 were not.
GAP_TOLERANCE = 1e-9

# Structural-similarity cutoff for the ECFP4-Tanimoto reference (Bajorath 0.55).
ECFP4_SIM_THRESHOLD = 0.55

# A dataset needs at least this many high-confidence cliffs for the deferred
# Elkan-Noto PU classifier (calibrate_gate) to have a positive class to learn.
PU_MIN_CLIFFS = 5

# Kramer et al. 2012 inter-laboratory assay noise, standard deviation ~0.54 pKi.
# It is quoted in pKi, so anything compared against it must be in label units,
# not in standard deviations of the label.
KRAMER_NOISE_FLOOR_PKI = 0.54


def assay_noise_floor(dataset_name):
    """(floor in the label's own units, source) for this dataset, or (None, why).

    Until 2026-09-28 every dataset was compared with Kramer's 0.54 pKi, so logd
    and caco2 were held to a hERG-binding figure in the wrong units. The
    per-dataset figures are the settled table in
    dataset_policy.REFERENCE_ERROR_LABEL_UNITS (herg_ki 0.54 Kramer 2012,
    caco2 0.35 Bentz 2013, logd 0.48 Niu 2024). A dataset with no published
    assay error gets no floor and no comparison.
    """
    from experiments.dataset_policy import REFERENCE_ERROR_LABEL_UNITS
    tau = REFERENCE_ERROR_LABEL_UNITS.get(dataset_name)
    if tau is None:
        return None, ('no published assay error for this dataset in '
                      'dataset_policy.REFERENCE_ERROR_LABEL_UNITS')
    return float(tau), (f'dataset_policy.REFERENCE_ERROR_LABEL_UNITS'
                        f'[{dataset_name!r}]')

# stage0_diagnostics' bias cutoff, applied to the residual mean. The library
# applies it to the standardized residual; the label-units flag below applies it
# to the same quantity after rescaling.
BIAS_MAGNITUDE_CUTOFF = 0.005


def _rescale_stage0(stage0, params, dataset_name=None):
    """Add label-units copies of every residual quantity stage0_diagnostics
    returns, and say which scale each one is on.

    from_linear returns (mu, label_sd) and maps y -> exp((y - mu) / label_sd),
    so log(y_pos) is the label divided by its own standard deviation. Every
    residual the diagnostics compute is therefore in standard deviations of the
    label. Multiplying by label_sd puts them back into the label's own units,
    which is the only scale the assay floor can be compared against.

    Spread and bias rescale; kurtosis, the fraction beyond 3 SD, the two
    residual correlations and the p-value are scale-free and are left alone.
    """
    label_sd = float(params[1])
    floor, floor_source = assay_noise_floor(dataset_name)
    out = dict(stage0)
    out['label_sd'] = label_sd
    out['residual_scale'] = 'standardized_label_sd'
    out['residual_scale_label_units'] = 'label'
    out['assay_noise_floor_label_units'] = floor
    out['assay_noise_floor_source'] = floor_source
    out['assay_floor_scale'] = ("the label's own units; compare against the "
                                "*_label_units fields")

    out['resid_sd_label_units'] = float(stage0['log_resid_sd']) * label_sd
    out['tier_sd_label_units'] = {
        k: (float(v) * label_sd if np.isfinite(v) else float('nan'))
        for k, v in stage0['tier_sd'].items()}
    out['tier_mean_bias_label_units'] = {
        k: (float(v) * label_sd if np.isfinite(v) else float('nan'))
        for k, v in stage0['tier_mean_bias'].items()}
    out['overall_bias_label_units'] = float(stage0['overall_log_bias']) * label_sd
    out['bias_flag_label_units'] = bool(
        stage0['bias_pvalue'] < 0.01
        and abs(out['overall_bias_label_units']) > BIAS_MAGNITUDE_CUTOFF)
    # The residual spread is model error plus label noise, so it is an upper
    # bound on the noise: above the floor says nothing about the labels alone.
    out['resid_sd_over_assay_floor'] = (
        None if floor is None else out['resid_sd_label_units'] / floor)
    out['resid_sd_at_or_below_assay_floor'] = (
        None if floor is None else bool(out['resid_sd_label_units'] <= floor))
    return out


def _cliff_mask_from_pairs(n, pair_iter, potency_threshold):
    """Build a symmetric (n, n) boolean cliff adjacency from (i, j, delta) MMP
    pairs, marking pairs whose |delta activity| exceeds the potency threshold."""
    adj = np.zeros((n, n), dtype=bool)
    n_considered = 0
    for i, j, delta in pair_iter:
        i, j = int(i), int(j)
        if i < 0 or j < 0 or i >= n or j >= n:
            continue
        n_considered += 1
        if abs(float(delta)) > potency_threshold + GAP_TOLERANCE:
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
    # A pair with similarity 0 is never similar, as in n01 since 02847b1. A
    # percentile cutoff that lands on 0 made `>= 0` true for every pair, and
    # n01 gives a molecule RDKit refuses similarity 0 with everything.
    adj = ((sim_matrix >= sim_cutoff) & (sim_matrix > 0)
           & (delta > potency_threshold + GAP_TOLERANCE))
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


# Chance baseline for every cliff count (2026-09-28, review 5 of
# n14__herg_ki). A cliff count alone cannot say whether similar molecules
# differ by 2 log units more or less often than random molecules do. On
# herg_ki the 10,809 pairs with Tanimoto at least 0.55 hold 106 cliff pairs;
# the same pairs with the labels permuted across molecules hold a median of
# 1,270. So the bare 106 reads as "cliffs are common" when it means similar
# molecules differ far less often than chance. Written beside each count, never
# in its place. Gated on KIRBY_NOISE_CONTROLS like the suite's other controls.
CHANCE_SHUFFLES = 200
CHANCE_SEED = 0


def _unique_pairs(rows, cols, n, same=None):
    """Unordered (i, j) pairs, i < j, each once, no self pair and no pair of
    rows that are the same molecule."""
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    keep = rows != cols
    a = np.minimum(rows[keep], cols[keep])
    b = np.maximum(rows[keep], cols[keep])
    key = np.unique(a * n + b)
    a, b = key // n, key % n
    if same is not None and a.size:
        ok = ~same[a, b]
        a, b = a[ok], b[ok]
    return a, b


def _chance_baseline(a, b, labels, n_shuffles=CHANCE_SHUFFLES,
                     seed=CHANCE_SEED):
    """Cliff pairs among the candidate pairs (a[t], b[t]) with the real labels
    and with the labels permuted across all molecules, n_shuffles times.

    The candidate pairs are fixed (the similar pairs, or the k-nearest pairs);
    only which label sits on which molecule changes. A count below the
    shuffled ones means similar molecules differ by more than the threshold
    less often than random molecules do.
    """
    labels = np.asarray(labels, dtype=float)
    cut = AC_POTENCY_THRESHOLD + GAP_TOLERANCE
    n_cand = int(len(a))
    observed = int((np.abs(labels[a] - labels[b]) > cut).sum()) if n_cand else 0
    rng = np.random.default_rng(seed)
    sh = np.empty(n_shuffles, dtype=np.int64)
    for s in range(n_shuffles):
        p = labels[rng.permutation(len(labels))]
        sh[s] = int((np.abs(p[a] - p[b]) > cut).sum()) if n_cand else 0
    med = float(np.median(sh))
    return {
        'n_candidate_pairs': n_cand,
        'cliff_pairs': observed,
        'cliff_share_of_candidate_pairs': (observed / n_cand) if n_cand else None,
        'shuffled_cliff_pairs_median': med,
        'shuffled_cliff_pairs_p05': float(np.percentile(sh, 5)),
        'shuffled_cliff_pairs_p95': float(np.percentile(sh, 95)),
        'cliff_pairs_over_shuffled_median': (observed / med) if med > 0 else None,
        'share_of_shuffles_at_or_below_observed': float((sh <= observed).mean()),
        'n_shuffles': int(n_shuffles),
        'seed': int(seed),
        'what_is_shuffled': ('labels permuted across all molecules; the '
                             'candidate pairs are unchanged'),
    }


def _share_all_pairs_over_threshold(labels):
    """Share of all unordered molecule pairs whose labels differ by more than
    the threshold: the cliff rate if similarity told nothing about the label.
    Sorted counting, so no n x n matrix."""
    y = np.sort(np.asarray(labels, dtype=float))
    n = len(y)
    if n < 2:
        return None
    cut = AC_POTENCY_THRESHOLD + GAP_TOLERANCE
    # for each i, molecules j > i (in sorted order) with y[j] - y[i] > cut
    j0 = np.searchsorted(y, y + cut, side='right')
    over = int(np.maximum(n - j0, 0).sum())
    return over / (n * (n - 1) / 2)


def _same_molecule(data, n):
    """(n, n) bool, True where two rows are the same molecule, or None.

    A loader that keeps one row per measurement (herg_ic50 today) says which
    rows are one molecule in data['replicate_groups']. Two such rows have
    Tanimoto 1 and, if their measurements differ by more than 2 log units,
    passed as an activity cliff until 2026-09-28. A cliff is two different
    molecules; two measurements of one molecule disagreeing is a repeat
    measurement, which has_replicates reports.
    """
    rg = data.get('replicate_groups')
    if rg is None:
        return None
    rg = np.asarray(rg)
    if len(rg) != n:
        raise ValueError(f"replicate_groups has {len(rg)} rows for {n} labels")
    same = rg[:, None] == rg[None, :]
    np.fill_diagonal(same, False)
    return same


def _drop_same_molecule(adj, same):
    """adj with every same-molecule pair removed, and how many pairs that was."""
    if same is None:
        return adj, 0
    n_dropped = int((adj & same).sum()) // 2
    return adj & ~same, n_dropped


def _build_ac_reference(data, dataset_name, labels):
    """Return (cliff_adjacency, provenance) for the high-confidence AC list.

    MMP mode fires when the loader exposes matched-molecular pairs aligned to the
    data rows (`data['mmp_pairs']`, as chembl_qsar produces). Otherwise falls
    back to the ECFP4-Tanimoto structural reference. Provenance lists exactly
    which filters were applied so downstream readers can gauge confidence.
    """
    n = len(labels)
    same = _same_molecule(data, n)
    provenance = {
        'potency_threshold_log_units': AC_POTENCY_THRESHOLD,
        # Rows that are the same molecule are never a cliff pair (2026-09-28).
        'same_molecule_rows_known': same is not None,
        # Strict: a gap of exactly 2.0 log units is not a cliff (the mask
        # uses >). The label said >= until 2026-09-28.
        'potency_criterion': '>100-fold (|delta label| > 2 log units)',
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
            n, mmp_pairs, AC_POTENCY_THRESHOLD)
        adj, n_same = _drop_same_molecule(adj, same)
        provenance.update({
            'same_molecule_pairs_dropped': n_same,
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
    # The list is always the absolute Tanimoto 0.55. Until 2026-09-28 a zero
    # count here was replaced by the top-2% most-similar pairs, which on a
    # dataset with no cliff at 0.55 turned "none" into a positive count under
    # the same high-confidence name and fed it to the feasibility flag. n01
    # keeps 0.55 for Tanimoto for the reason that applies here (0.55 is
    # meaningful for Tanimoto; n01:86-88). The relaxed count is still
    # computed when the strict one is zero, and written beside it under its
    # own name, never in its place.
    adj, sim_cutoff = _cliff_mask_from_similarity(
        sim, labels, ECFP4_SIM_THRESHOLD, AC_POTENCY_THRESHOLD)
    adj, n_same = _drop_same_molecule(adj, same)
    provenance.update({
        'same_molecule_pairs_dropped': n_same,
        'similarity_criterion': 'ecfp4_tanimoto',
        'similarity_source': 'computed_from_smiles',
        'similarity_cutoff': float(sim_cutoff),
        'similar_percentile': None,
    })
    from experiments.noise.controls import controls_enabled
    if controls_enabled():
        _ci, _cj = np.nonzero(np.triu(
            (sim >= ECFP4_SIM_THRESHOLD) & (sim > 0), k=1))
        _a, _b = _unique_pairs(_ci, _cj, n, same)
        provenance['chance_baseline'] = _chance_baseline(_a, _b, labels)
    if adj.sum() == 0:
        adj_p, cut_p = _cliff_mask_from_similarity(
            sim, labels, ECFP4_SIM_THRESHOLD, AC_POTENCY_THRESHOLD,
            similar_percentile=SIMILAR_PERCENTILE)
        adj_p, _ = _drop_same_molecule(adj_p, same)
        provenance['relaxed_top_percentile_not_high_confidence'] = {
            'similar_percentile': SIMILAR_PERCENTILE,
            'similarity_cutoff': float(cut_p),
            'geometry': _geometry(adj_p)[0],
        }
    return adj, provenance


def _cliff_geometry_by_similarity(data, dataset_name, labels, output_dir,
                                  config):
    """The cliff graph's geometry under every similarity setting and k.

    Added 2026-09-25 (plan sections 1-3). Spaces, over every molecule:
      model_input        every representation concatenated and standardised,
                         the stage-0 feature space below
      base_1, base_2     rf's top-2 single representations, ranked by rf's own
                         score in the stored search (noise_base_reps_by_model)
                         -- rf because n14 fits no model of its own and rf is
                         the reference model of the suite's pool reads
      tanimoto_ecfp4     from the SMILES
      forest_proximity   the forest fitted on model_input and the labels
    Neighbour lists through n07.fold_neighbours (noise_cache). Each (setting,
    k) geometry is written to a checkpoint as it finishes and read back on
    resume.

    TODO(similarity-sweep 2026-09-25): the augmented spaces and the two
    hybrids are recipes built from a fold's TRAINING labels; n14 has no fold,
    so building them here means choosing features from every molecule's
    label. Whether a whole-dataset build is the right space for a survey that
    fits nothing is the owner's call; until then those settings are recorded
    as unavailable, with this reason, and skipped.
    """
    from sklearn.preprocessing import StandardScaler
    from experiments.noise.n07_remediation import (
        sweep_settings, sweep_ks, usable_ks, fold_neighbours, FoldCheckpoint,
        similarity_record)
    from experiments.noise.similarity_space import neighbours_at
    from experiments._shared import input_fingerprint
    settings = sweep_settings()
    reps = data.get('reps') or {}
    n = len(labels)
    unavailable = {}
    spaces = {}
    if reps:
        spaces['model_input'] = StandardScaler().fit_transform(np.hstack(
            [np.nan_to_num(np.asarray(reps[k], dtype=float), nan=0.0)
             for k in sorted(reps)]))
    need = {st.space for st in settings}
    base = []
    if need & {'base_1', 'base_2'}:
        try:
            from experiments._shared import (get_best_search_for_downstream,
                                             noise_base_reps_by_model)
            _s, _m = get_best_search_for_downstream(
                reps, data['labels'], data.get('groups'), downstream_model='rf',
                budget=None, augmentations=data.get('augmentations'),
                dataset_name=dataset_name, task_type=data['task_type'],
                data=data)
            _b, _miss = noise_base_reps_by_model({'rf': _s}, ['rf'], reps=reps,
                                                 dataset_name=dataset_name,
                                                 verbose=False)
            base = _b.get('rf', [])
            if not base:
                unavailable['base_1'] = unavailable['base_2'] = _miss.get('rf')
        except Exception as exc:
            unavailable['base_1'] = unavailable['base_2'] = (
                f'{type(exc).__name__}: {exc}')
    for i, rn in enumerate(base[:2], start=1):
        spaces[f'base_{i}'] = StandardScaler().fit_transform(
            np.asarray(reps[rn], dtype=float))
    for sp in ('hybrid', 'hybrid_similarity', 'base_1_aug_capped',
               'base_1_aug_uncapped', 'base_2_aug_capped',
               'base_2_aug_uncapped'):
        if sp in need:
            unavailable[sp] = ('a recipe build chooses features from training '
                               'labels and n14 has no fold; see the TODO in '
                               '_cliff_geometry_by_similarity')
    # Rows that are the same molecule are never joined (2026-09-28). The key
    # is in the fingerprint only where such rows exist, so a dataset with one
    # row per molecule, whose graphs this cannot change, keeps its checkpoint.
    same = _same_molecule(data, n)
    from experiments.noise.controls import controls_enabled
    _chance = bool(controls_enabled())
    ckpt = FoldCheckpoint(output_dir, 'n14_activity_cliff_landscape',
                          dataset_name, tag='geometry', fingerprint={
                              'input': input_fingerprint(data, config),
                              'potency': AC_POTENCY_THRESHOLD,
                              'gap_tolerance': GAP_TOLERANCE,
                              'base': base,
                              **({'same_molecule_excluded': True}
                                 if same is not None else {}),
                              # A stored graph without the chance baseline is
                              # not served to a run that asks for one.
                              **({'chance_baseline': [CHANCE_SHUFFLES,
                                                      CHANCE_SEED]}
                                 if _chance else {})})
    nb = fold_neighbours(dataset_name, settings, spaces, None, labels,
                         smiles_train=data.get('smiles'), is_clf=False)
    out = {}
    for st in settings:
        L = nb.get(st.name)
        if L is None or not L.get('available'):
            unavailable.setdefault(st.name, (L or {}).get('reason')
                                   or unavailable.get(st.space))
            continue
        for k in usable_ks(L):
            def _geo(L=L, k=k):
                idx = neighbours_at(L, k)[0]
                adj = np.zeros((n, n), dtype=bool)
                rows = np.repeat(np.arange(n), idx.shape[1])
                cols = idx.ravel()
                gap = (np.abs(labels[rows] - labels[cols])
                       > AC_POTENCY_THRESHOLD + GAP_TOLERANCE)
                adj[rows[gap], cols[gap]] = True
                adj |= adj.T
                np.fill_diagonal(adj, False)
                adj, _ = _drop_same_molecule(adj, same)
                geo = _geometry(adj)[0]
                if _chance:
                    # candidate pairs: every distinct k-nearest pair
                    _a, _b = _unique_pairs(rows, cols, n, same)
                    geo['chance_baseline'] = _chance_baseline(_a, _b, labels)
                return geo
            out.setdefault(st.name, {})[str(k)] = ckpt.cached(
                [st.name, int(k)], _geo)
    meta = similarity_record(
        settings, None, base_reps_rf=base, spaces_unavailable=unavailable,
        cliff_rule=('an edge joins two molecules when one is among the '
                    "other's k nearest in the setting and |delta label| > "
                    f'{AC_POTENCY_THRESHOLD}'),
        checkpoint_file=ckpt.path, read_back=ckpt.n_resumed)
    return out, meta


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
    adj, provenance = _build_ac_reference(data, dataset_name, labels)
    geometry, cliff_mask = _geometry(adj)
    n_cliffs = geometry['n_cliff_compounds']
    print(f"  Cliff reference: {provenance['similarity_criterion']} + "
          f"{provenance['potency_criterion']}")
    print(f"  Cliffs: {n_cliffs} compounds ({geometry['pct_cliff_compounds']:.1f}%), "
          f"{geometry['n_cliff_pairs']} pairs, lean={geometry['geometry_lean']}")
    _cb = provenance.get('chance_baseline')
    if _cb:
        print(f"  Chance baseline: {_cb['cliff_pairs']} cliff pairs among "
              f"{_cb['n_candidate_pairs']} similar pairs; with the labels "
              f"shuffled {_cb['n_shuffles']} times the median is "
              f"{_cb['shuffled_cliff_pairs_median']:.0f} "
              f"({_cb['shuffled_cliff_pairs_p05']:.0f} to "
              f"{_cb['shuffled_cliff_pairs_p95']:.0f})")

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
        # The cliff rate if similarity said nothing about the label: the share
        # of ALL molecule pairs whose labels differ by more than the threshold
        # (2026-09-28). 0 means no cliff can exist under this threshold.
        'share_of_all_pairs_over_threshold': _share_all_pairs_over_threshold(
            labels),
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
    # graph the same way. Every molecule is its own training row: n14 fits no
    # model and has no held-out fold.
    geometry_by_similarity, sim_meta = _cliff_geometry_by_similarity(
        data, dataset_name, labels, output_dir, config)

    # --- 4. Feasibility flags for the deferred method families ---
    rep_groups = build_replicate_groups(data)
    has_replicates = False
    n_replicate_samples = 0
    if rep_groups is not None:
        _stds, has_rep = compute_replicate_stds(labels, rep_groups)
        has_replicates = bool(has_rep.any())
        n_replicate_samples = int(has_rep.sum())
    meta = data.get('metadata') or {}
    # The molecular loader never sets data['metadata']; it hands the per-row
    # ChEMBL assay ids out as data['record_assays'] (herg_ic50 today,
    # _shared.py load_data). Before 2026-09-28 only the metadata keys were
    # read, so a dataset that has assay ids reported none.
    _rec_assays = data.get('record_assays')
    n_record_assays = (0 if _rec_assays is None else
                       int(len(np.unique(np.asarray(_rec_assays).astype(str)))))
    has_multi_assay = bool(meta.get('assay_ids') is not None or
                           meta.get('n_assays', 0) and meta.get('n_assays', 0) > 1
                           or n_record_assays > 1)

    feasibility = {
        'n_high_conf_cliffs': int(n_cliffs),
        'pu_calibration_feasible': bool(n_cliffs >= PU_MIN_CLIFFS),
        'pu_min_cliffs': PU_MIN_CLIFFS,
        'has_replicates': has_replicates,
        'n_replicate_samples': n_replicate_samples,
        'has_multi_assay': has_multi_assay,
        'n_record_assays': n_record_assays,
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
    # label standard deviation is 1.0. On herg_ki it is 0.915 (1,412 molecules,
    # measured 2026-09-28), so a returned 0.59 is 0.54 pKi — exactly at the
    # floor, while reading as above it.
    # _rescale_stage0 multiplies the residual spread and the two bias terms by
    # that standard deviation and records which scale each number is on.
    reps = data.get('reps', {})
    stage0 = None
    stage0_error = None
    if reps and groups is not None:
        try:
            X_concat = np.hstack([np.asarray(reps[k], dtype=float)
                                  for k in sorted(reps.keys())])
            X_concat = np.nan_to_num(X_concat, nan=0.0)
            y_pos, _params = from_linear(labels)
            _label_sd = float(_params[1])
            n_splits = config.get('n_splits', 5) if config else 5
            n_splits = max(2, n_splits)
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
            # together, 16,405 features on herg_ki against 1,412 molecules
            # when this was written (15,081 over ten representations in the
            # 2026-09-28 local run) -- and one LightGBM at 600 trees over 5
            # folds took more than 14 minutes at 1.5 GB resident on one
            # thread. In the 2026-09-28 local run, 3 folds on a loaded
            # machine, this fit took about 2 h 18 min of wall time on herg_ki. The fix is the
            # dimension cap NOISE_AUDIT_2026_09_10.md asks for at n14:399,
            # NOT more threads, and it is a change to what the residual spread
            # is measured on, so it is not made here.
            #
            # About 11 features per molecule (15,081 features over 1,412
            # molecules in the 2026-09-28 local run, ten representations) is a
            # regime where the surrogate overfits its training folds. The
            # residuals are out of fold, so overfitting cannot hide the
            # held-out labels' noise; it makes the out-of-fold error LARGER,
            # which biases the residual spread UP and makes the labels read
            # noisier than they are. (Until 2026-09-28 this comment said the
            # opposite.) The width is printed so that is visible beside the
            # number.
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
                                       'input': input_fingerprint(data, config),
                                       'n_splits': int(n_splits),
                                       'width': _w})
            # The checkpoint holds the raw diagnostics only (key 'stage0_raw',
            # 2026-09-28). The rescale and the assay floor are applied after,
            # so a change to either never serves a stale cached value. Lines
            # under the old key 'stage0' held the rescaled dict and are ignored.
            _raw = _s0ck.cached(['stage0_raw'], lambda: stage0_diagnostics(
                X_concat, y_pos, groups, n_splits=n_splits,
                splitter=cv_split))
            stage0 = _rescale_stage0(_raw, _params, dataset_name)
            _floor = stage0['assay_noise_floor_label_units']
            print(f"  Stage-0: resid_sd={stage0['resid_sd_label_units']:.3f} "
                  f"label units (={stage0['log_resid_sd']:.3f} SDs of a "
                  f"{_label_sd:.3f}-SD label), "
                  f"assay floor {_floor} label units, "
                  f"kurtosis={stage0['excess_kurtosis']:.2f}, "
                  f"U-shape={stage0['u_shaped_heteroscedasticity']}, "
                  f"bias_flag={stage0['bias_flag_label_units']}")
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
        # 2026-09-25: {setting: {k: _geometry of the k-nearest cliff graph}}.
        'cliff_geometry_by_similarity': geometry_by_similarity,
        'similarity': sim_meta,
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
