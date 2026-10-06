#!/usr/bin/env python3
"""N01: Activity Cliff Framework -- cliff-stratified prediction error.

Paper: Noise / Robustness
Scope: MOLECULAR datasets.
Conclusion: Hybrid representations produce smaller cliff vs non-cliff
    prediction gaps than individual representations.

Single sub-analysis: Cliff-stratified prediction error (RMSE/R2 on cliff vs
non-cliff subsets). Cliffs come from a reference built outside the
representations being graded — ECFP4 Tanimoto for molecular, edit distance on
heavy chains for antibody, Tanimoto on reactant and product Morgan bits for
reaction, Tanimoto on the drug within one protein for drug-target affinity —
and error is measured with held-out CV.

The exception is Euclidean on the concatenated representations, which is what
runs when no such reference exists (TCGA, and any molecular dataset where RDKit
refuses the SMILES). There the cliffs ARE a function of the representations
being graded, and whichever contributes the most features defines most of them.
Every result records `cliff_reference_independent`; when it is False the
cliff-versus-non-cliff comparison between representations is circular and must
not be reported as one.

The three earlier sub-analyses (consensus cliff identification, SALI landscape
smoothing, cliff mitigation by hybridization) were REMOVED as scientifically
circular -- they derived cliffs from each representation's own embedding, so a
supervised hybrid wins by construction.

2026-09-12, decided: n01 searches its recipe inside each fold, on the fast
speed tier, with no augmentation flag — donor features are not in n01's pool.
One search per outer fold, on that fold's training rows; the recipe the hybrid
arm uses on a fold's held-out rows never saw those rows' labels. It used to
read one search run over every row. Donor representations loaded by --augment
are not searched over and are not rebuilt into the hybrid.

2026-10-05, chat 5 of HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.0. What
changed, against the 2026-09-12 to 2026-10-02 version:
  models     the dataset's two decided models (DECISIONS.md Step 1), not the
             five of DOWNSTREAM_MODELS_NOISE.
  fitted     per outer fold and model, the 2 best base representations and
             the 2 best hybrids, ranked on that fold's training rows
             (noise_helpers.fold_configurations): 2 models x 4 = 8 fitted
             configurations per fold. It used to fit every representation
             plus 3 hybrids (2 general + 1 distance) for each of 5 models.
  searches   the decided grid per dataset (budgets, importance methods,
             allocation methods, evaluators, representation counts,
             augmentation), pruning off. It used to run speed='fast' with
             fixed lists on every dataset.
  distance   the distance search's top hybrid and the similarity search's top
             hybrid are neighbour spaces in the cliff sweep only, never fitted.
  folds      noise_helpers.noise_outer_folds: the count comes from
             config['n_splits'] (KIRBY_NOISE_N_FOLDS overrides), any count
             including 1.
  comparison the hybrid's cliff gap is read beside the gaps of the same
             model's 2 best base representations. It used to be compared with
             the 3 representations whose gap was closest to 0, which were the
             3 weakest (dirty run, 2026-09-28 to 2026-10-02).

2026-10-06 (Adelaide delegated): on a dataset with one row per measurement
(herg_ic50) a molecule's own repeat measurements are left out of its reference
cliffs and of its neighbour lists. The any-neighbour cliff rule is kept, with
no weighted version.

Usage:
    python n01_activity_cliff_framework.py --dataset logd --reps all --mode full
    python n01_activity_cliff_framework.py --dataset logd --sim-threshold 0.55 --activity-threshold 2.0
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist


# 2026-09-12, decided: n01 searches its recipe inside each fold.
# get_fitted_pool_hybrid and get_best_search_for_downstream both read a recipe
# searched over every row of the dataset, so neither is imported any more.
# 2026-10-05 (chat 5): the five-model lists (DOWNSTREAM_MODELS_NOISE), the
# fast-tier search lists (FULL_BUDGETS, DISTANCE_*), StandardScaler, the
# rf-ranked extract_top3_base_reps and get_best_for_downstream are gone with
# the code that used them; the configurations come from
# noise_helpers.fold_configurations.
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric,
    # Added 2026-09-18. Both read rows a search already stored and fit
    # nothing: the training-row scores of the base representations, and the
    # rank-1 hybrid against its sources and against the best representation.
    top_base_reps, hybrid_deltas,
)

from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# =============================================================================
# CLIFF DETECTION HELPERS
# =============================================================================

# Distribution-relative "structurally similar" cutoff. On standardised high-D
# embeddings (antibody / reaction / TCGA) the pairwise Euclidean distance is
# large, so sim = 1/(1+d) never reaches an absolute 0.55 and the experiment
# found ~0 cliffs on every non-molecular dataset (ginkgo_hic all-zero, tcga
# stratified too_few_cliff_compounds, buchwald 0 cliffs). The similarity suite
# fixed the identical bug by switching to a percentile: "structurally similar"
# = the closest SIMILAR_PERCENTILE% of off-diagonal pairs, regardless of the
# absolute distance scale. See COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md and
# s12.identify_cliff_pairs_euclidean. The molecular ECFP4-Tanimoto branch keeps
# the absolute threshold (0.55 is meaningful for Tanimoto).
SIMILAR_PERCENTILE = 2.0


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


def _morgan_bits(smiles_list, n_bits=2048):
    """(n, n_bits) float array of Morgan (ECFP4) bits, zeros where RDKit
    refuses the string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    arr = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        for bit in fp.GetOnBits():
            arr[i, bit] = 1.0
    return arr


def _compute_reaction_tanimoto_sim(rxn_smiles, n_bits=2048):
    """Tanimoto similarity between reactions, from the reaction SMILES.

    Each reaction becomes two Morgan bit blocks — one for the union of its
    reactants, one for its product — so a pair is similar when it shares both
    starting materials and product, and reversing a reaction is not similar to
    itself. Built from the chemistry, never from the representations being
    graded, which is the whole point: the concatenated-representation fallback
    let whichever representation contributed the most features define most of
    the cliffs it was then graded on.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    n = len(rxn_smiles)
    arr = np.zeros((n, 2 * n_bits), dtype=np.float32)
    for i, rxn in enumerate(rxn_smiles):
        txt = str(rxn)
        left, _, right = txt.partition('>>')
        # Some dumps carry the three-part reactants>agents>products form.
        if '>' in left:
            left = left.split('>')[0]
        for offset, side in ((0, left), (n_bits, right)):
            for part in side.split('.'):
                part = part.strip()
                if not part:
                    continue
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                for bit in fp.GetOnBits():
                    arr[i, offset + bit] = 1.0

    sim = 1.0 - scipy_cdist(arr, arr, metric='jaccard')
    np.fill_diagonal(sim, 0.0)
    return sim


def _compute_dta_sim(drug_smiles, protein_ids=None, n_bits=2048):
    """Tanimoto similarity between drug-target pairs, from the drug SMILES.

    A row is one (drug, protein) measurement, so two rows are only comparable
    when they share the protein: the same drug against two different proteins
    legitimately has different affinity, and calling that a cliff would flag
    every repeated drug. Pairs on different proteins are set to zero
    similarity. Built from the drug structures, not from the representations
    being graded.
    """
    arr = _morgan_bits(drug_smiles, n_bits=n_bits)
    sim = 1.0 - scipy_cdist(arr, arr, metric='jaccard')
    np.fill_diagonal(sim, 0.0)
    if protein_ids is not None:
        pid = np.asarray(protein_ids)
        if len(pid) == len(drug_smiles):
            sim = np.where(pid[:, None] == pid[None, :], sim, 0.0)
            np.fill_diagonal(sim, 0.0)
    return sim


def identify_cliff_pairs(sim_matrix, y, sim_threshold, activity_k, is_clf=False,
                         similar_percentile=None):
    """Identify activity cliff pairs from a similarity matrix.

    Args:
        sim_matrix: (n, n) similarity matrix (0 = dissimilar, 1 = identical).
        y: (n,) activity values.
        sim_threshold: minimum similarity to count as "structurally similar"
            (used only when ``similar_percentile`` is None).
        activity_k: activity difference threshold in MAD units.
        is_clf: if True, use binary indicator (different class = cliff).
        similar_percentile: if set (e.g. 2.0), define "structurally similar" as
            the top ``similar_percentile``% most-similar off-diagonal pairs
            instead of the absolute ``sim_threshold``. Distribution-relative —
            required for Euclidean similarity on standardised high-D reps where
            sim = 1/(1+d) never reaches an absolute 0.55 (see SIMILAR_PERCENTILE
            note above). Mirrors s12.identify_cliff_pairs_euclidean.

    Returns:
        cliff_pair_mask: (n, n) boolean, True where (i,j) is a cliff pair.
        info: dict with thresholds and counts.
    """
    n = len(y)

    if is_clf:
        # Classification: cliff = similar structure, different class
        delta_y = (y[:, None] != y[None, :]).astype(float)
        activity_threshold = 0.5  # any class difference
        mad_y = 0.0
    else:
        mad_y = float(np.median(np.abs(y - np.median(y))))
        activity_threshold = activity_k * mad_y
        delta_y = np.abs(y[:, None] - y[None, :])

    # "Structurally similar" cutoff: absolute, or distribution-relative (top
    # similar_percentile% most-similar off-diagonal pairs).
    if similar_percentile is not None:
        _off = sim_matrix[~np.eye(n, dtype=bool)]
        sim_cutoff = float(np.percentile(_off, 100.0 - similar_percentile)) \
            if _off.size else sim_threshold
    else:
        sim_cutoff = sim_threshold

    cliff_pair_mask = (sim_matrix >= sim_cutoff) & (delta_y > activity_threshold)
    np.fill_diagonal(cliff_pair_mask, False)

    cliff_mask = cliff_pair_mask.any(axis=1)
    n_cliff_pairs = int(cliff_pair_mask.sum()) // 2  # symmetric
    n_cliff_compounds = int(cliff_mask.sum())

    info = {
        'sim_threshold': sim_threshold,
        'similar_percentile': similar_percentile,
        'sim_cutoff': float(sim_cutoff),
        'activity_k': activity_k,
        'activity_threshold': float(activity_threshold),
        'mad_y': float(mad_y),
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': float(100.0 * n_cliff_compounds / n) if n > 0 else 0.0,
    }
    return cliff_pair_mask, info


# =============================================================================
# CLIFF-STRATIFIED PREDICTION ERROR
# =============================================================================

def _fold_general_search(data, dataset_name, *, fold_i, train_idx, n_folds):
    """This fold's general search (the 'none' condition), read back from
    _shared.fold_searches's in-process memo with the arguments
    noise_helpers.fold_configurations passed it, so it is the search the
    configurations were ranked on and costs no new fit. Used only for the
    training-row scores in the result file. None when it cannot be read."""
    from experiments._shared import fold_searches
    try:
        s = fold_searches(
            data['reps'], np.asarray(data['labels']), data.get('groups'),
            train_idx=np.asarray(train_idx), fold_i=int(fold_i),
            n_outer_folds=int(n_folds), dataset_name=dataset_name,
            augmentations=data.get('augmentations'),
            task_type=data.get('task_type') or 'auto', tier='general')
        return s.get('none')
    except Exception as e:
        print(f"      fold {fold_i}: general search not readable for the "
              f"training-row scores ({type(e).__name__}: {e})")
        return None


def _training_row_scores(search, model_name, entry, context):
    """What the fold's search scored on its training rows, for one model: the
    2 best base representations' scores and the rank-1 hybrid's two deltas
    (against the mean of the representations that fed it, and against the
    best single representation). The hybrids' own training-row scores are in
    configuration_record ('score_mean'). 0 new model fits."""
    out = {}
    if search is None:
        out['note'] = 'the fold\'s general search could not be read back'
        return out
    try:
        tb = top_base_reps(search, model_name, 2, require_full=False,
                           context=context)
        out['base'] = [{'rep': e.get('rep'), 'score': e.get('score')}
                       for e in (tb.get('selected') or [])]
        if tb.get('shortfall_reason'):
            out['base_shortfall'] = tb['shortfall_reason']
    except Exception as e:
        out['base_error'] = f'{type(e).__name__}: {str(e)[:200]}'
    hyb = (entry or {}).get('hybrids') or []
    if hyb:
        try:
            out['hybrid_deltas'] = convert_numpy(hybrid_deltas(
                search, model_name, hybrid=hyb[0], context=context))
        except Exception as e:
            out['hybrid_deltas_error'] = f'{type(e).__name__}: {str(e)[:200]}'
    return out


def _run_cliff_stratified(reps, labels, groups, config, data,
                          sim_threshold, activity_k,
                          smiles=None, is_clf=False,
                          dataset_name=None, models_to_run=None,
                          search_verbose=False, output_dir=None):
    """Compare prediction error on cliff vs non-cliff compounds.

    2026-10-05 (chat 5): per outer fold, each of the dataset's two decided
    models is fitted on its 4 configurations -- its 2 best base
    representations and its 2 best hybrids, all ranked by searches on that
    fold's training rows (noise_helpers.fold_configurations) -- and every
    prediction on the fold's held-out rows is graded cliff against non-cliff,
    first on the reference cliffs, then under every neighbour setting.
    """
    print("\n  --- Sub-analysis 4: Cliff-Stratified Prediction Error ---")

    from experiments.noise.noise_helpers import (
        noise_outer_folds, fold_spread, fold_configurations,
        fold_model_inputs, configuration_record, CONFIGURATION_SPACES)
    from experiments.decided_grid import decided_models

    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric_triple = (metric_name, metric_fn, lower_is_better)
    task_type = data['task_type']

    if dataset_name is None:
        # The decided grid and the per-fold search store are both keyed on
        # the dataset name, so there is nothing to search without one.
        return {'error': 'no_dataset_name'}

    # R1: the dataset's two models from DECISIONS.md Step 1. knn is a model of
    # the distance search only, and n01 never fits it.
    decided = [m for m in decided_models(dataset_name, 'general')
               if m != 'knn']
    if models_to_run and list(models_to_run) != decided:
        print(f"    Models asked for by the caller ({', '.join(models_to_run)}) "
              f"are ignored; the decided models are {', '.join(decided)}")
    model_factories = get_model_factories(
        decided, task_type=task_type, n_samples=len(labels))
    if not model_factories:
        return {'error': 'no_model_factories', 'models_decided': decided}

    # Identify cliff compounds using a REPRESENTATION-INDEPENDENT reference.
    # For molecular: ECFP4 Tanimoto (Bajorath standard).
    # For antibody: edit distance on heavy-chain sequences.
    # For reaction: Tanimoto on reactant and product Morgan bits.
    # For drug-target affinity: Tanimoto on the drug, within one protein.
    # Fallback: Euclidean on concatenated reps — the ONE branch that is not
    # representation-independent, so it records that it is not.
    #
    # cliff_reference_independent says which of those two happened. Until
    # 2026-09-04 reaction, TCGA and drug-target affinity all took the fallback,
    # and every representation was graded on cliffs found by concatenating all
    # ten of them: whichever contributed the most features defined most of the
    # cliffs. Reaction and drug-target affinity now have their own reference.
    # TCGA still has none — its representations are all derived from the same
    # expression matrix, so there is nothing outside them to build one from,
    # and the flag says so rather than the docstring claiming otherwise.
    #
    # DECIDED 2026-10-06 (Adelaide delegated): on a dataset with one row per
    # measurement (herg_ic50, 12,008 rows for 9,802 molecules), a molecule's
    # own repeat measurements are left out of its reference cliffs and of its
    # neighbour lists. Two measurements of one molecule have Tanimoto 1, so
    # before this two repeats more than activity_k label MADs apart counted as
    # a cliff pair. The molecule ids are data['replicate_groups'], the same
    # grouping n22 passes as exclude_groups.
    n = len(labels)
    molecule_of_row = data.get('replicate_groups')
    if molecule_of_row is not None:
        molecule_of_row = np.asarray(molecule_of_row)
        if len(molecule_of_row) != n:
            print(f"    WARNING: replicate_groups has {len(molecule_of_row)} "
                  f"entries for {n} rows; repeats are not excluded")
            molecule_of_row = None
    n_repeat_pairs_excluded = 0
    repeats_dropped_from_reference = False

    def _drop_repeat_pairs(sim):
        """Similarity 0 between two rows of the same molecule, so a
        molecule's repeat measurements are never its reference cliffs."""
        nonlocal n_repeat_pairs_excluded, repeats_dropped_from_reference
        if molecule_of_row is None:
            return sim
        same = molecule_of_row[:, None] == molecule_of_row[None, :]
        np.fill_diagonal(same, False)
        n_repeat_pairs_excluded = int(same.sum()) // 2
        repeats_dropped_from_reference = True
        sim = np.where(same, 0.0, sim)
        np.fill_diagonal(sim, 0.0)
        return sim
    heavy_seqs = data.get('heavy_seqs')
    cliff_reference_independent = True
    cliff_reference_note = None
    rxn_smiles = data.get('rxn_smiles')
    drug_smiles = data.get('drug_smiles')

    def _concat_reference():
        """Euclidean on every representation glued together. Records which
        representation contributed how many features, because that is what
        decides the cliffs when this branch runs."""
        names = sorted(reps.keys())
        X_cat = np.hstack([reps[k] for k in names])
        note = {
            'reason': 'no reference outside the representations being graded',
            'n_features_per_rep': {k: int(np.asarray(reps[k]).shape[1])
                                   for k in names},
        }
        return _compute_euclidean_sim(X_cat), note

    if heavy_seqs is not None and domain == 'antibody':
        # Edit-distance ground truth for antibodies
        from experiments.similarity.similarity_helpers import identify_cliff_compounds_editdist
        # Distribution-relative similar set (closest SIMILAR_PERCENTILE% of
        # sequence-pair edit distances) rather than an absolute max_edits=5.
        # On deep-mutational-scan antibody data max_edits=5 either flags a huge
        # fraction (s07 contamination saturated) or — on therapeutic panels
        # with high mutual edit distance (median 54-75 here) — flags ~none,
        # which is why ginkgo_hic yielded too_few_cliff_compounds.
        #
        # OPEN (chat 5): dirty-run bug 10. On flab_expression_koenig every
        # antibody came out a cliff in the dirty run: edit distances are whole
        # numbers, the helper keeps every pair AT the percentile cutoff
        # (similarity_helpers.identify_cliff_compounds_editdist, `edit_dist <=
        # edit_cutoff`), and the ties took in 18% of all pairs instead of 2%.
        # The helper is shared with n25 and is not changed here. The share of
        # antibodies marked as cliffs is in cliff_info['pct_cliff_compounds'].
        cliff_mask, cliff_info = identify_cliff_compounds_editdist(
            heavy_seqs, labels, activity_k=activity_k,
            is_clf=is_clf, similar_percentile=SIMILAR_PERCENTILE)
        sim_source = f'edit_distance(similar_pct={SIMILAR_PERCENTILE})'
    elif smiles is not None and domain == 'molecular':
        try:
            ref_sim = _drop_repeat_pairs(_compute_ecfp4_tanimoto_sim(smiles))
            sim_source = 'ecfp4_tanimoto'
            cliff_pair_mask, cliff_info = identify_cliff_pairs(
                ref_sim, labels, sim_threshold, activity_k, is_clf=is_clf)
            cliff_mask = cliff_pair_mask.any(axis=1)
        except Exception as e:
            print(f"    WARNING: ECFP4 Tanimoto failed ({e}), falling back to Euclidean")
            ref_sim, cliff_reference_note = _concat_reference()
            ref_sim = _drop_repeat_pairs(ref_sim)
            cliff_reference_independent = False
            sim_source = 'euclidean_concat'
            cliff_pair_mask, cliff_info = identify_cliff_pairs(
                ref_sim, labels, sim_threshold, activity_k, is_clf=is_clf,
                similar_percentile=SIMILAR_PERCENTILE)
            cliff_mask = cliff_pair_mask.any(axis=1)
    else:
        ref_sim = None
        if domain == 'reaction' and rxn_smiles is not None:
            try:
                ref_sim = _compute_reaction_tanimoto_sim(rxn_smiles)
                sim_source = 'reaction_ecfp4_tanimoto'
            except Exception as e:
                print(f"    WARNING: reaction Tanimoto failed ({e})")
                ref_sim = None
        elif domain == 'dta' and drug_smiles is not None:
            try:
                _meta = data.get('metadata') or {}
                ref_sim = _compute_dta_sim(drug_smiles,
                                           _meta.get('protein_ids'))
                sim_source = 'drug_ecfp4_tanimoto_within_protein'
            except Exception as e:
                print(f"    WARNING: drug Tanimoto failed ({e})")
                ref_sim = None

        if ref_sim is None:
            ref_sim, cliff_reference_note = _concat_reference()
            cliff_reference_independent = False
            sim_source = 'euclidean_concat'
        ref_sim = _drop_repeat_pairs(ref_sim)

        cliff_pair_mask, cliff_info = identify_cliff_pairs(
            ref_sim, labels, sim_threshold, activity_k, is_clf=is_clf,
            similar_percentile=SIMILAR_PERCENTILE)
        cliff_mask = cliff_pair_mask.any(axis=1)

    cliff_mask = np.asarray(cliff_mask, dtype=bool)
    # True only when the reference similarity had the molecule's repeat
    # pairs set to 0 (the edit-distance branch builds its own pairs).
    cliff_info['repeat_measurements_excluded'] = bool(
        repeats_dropped_from_reference)
    cliff_info['n_repeat_pairs_excluded'] = int(n_repeat_pairs_excluded)

    print(f"    Cliff source: {sim_source}")
    if not cliff_reference_independent:
        print(f"    Cliff reference is built from the representations being "
              f"graded — the cliff/non-cliff gap below is circular for this "
              f"dataset ({cliff_reference_note['reason']})")
    print(f"    Cliff compounds: {cliff_info['n_cliff_compounds']} / {n} "
          f"({cliff_info['pct_cliff_compounds']:.1f}%)")
    print(f"    Cliff pairs: {cliff_info['n_cliff_pairs']}")

    if cliff_info['n_cliff_compounds'] < 5:
        print("    WARNING: too few cliff compounds for meaningful stratification")
        return {
            'cliff_source': sim_source,
            'cliff_info': convert_numpy(cliff_info),
            'error': 'too_few_cliff_compounds',
        }

    # --- Folds (handoff 2.4 item 14) ---------------------------------------
    # nested_cv_outer_folds through noise_outer_folds: the folds the per-fold
    # searches are filed under, so a fold's configurations are ranked on its
    # own training rows and scored on its held-out rows. The count is
    # config['n_splits'] (KIRBY_NOISE_N_FOLDS overrides); 1 is one held-out
    # split. KIRBY_NOISE_ONLY_FOLDS runs a subset of the same split.
    folds, n_folds = noise_outer_folds(data, config)
    print(f"    {len(folds)} of {n_folds} outer folds; models "
          f"{', '.join(model_factories)}; per fold and model "
          f"{len(CONFIGURATION_SPACES)} fitted configurations "
          f"({', '.join(CONFIGURATION_SPACES)})")

    # --- The neighbour settings (KNN_SEARCH_SPACE_RECORD) -------------------
    # The dataset's group decides the spaces and distances: full_knn 5000
    # molecules or fewer, fast_knn more. No importance weighting, k in
    # (5, 10, 20), scaffold-mates kept in every list (axis 6). On a dataset
    # with one row per measurement fold_neighbours gets the molecule ids as
    # exclude_groups (DECIDED 2026-10-06), which is not the scaffold
    # exclusion axis 6 cut. The distance search's top hybrid
    # ('hybrid__distance1') and the similarity search's top hybrid
    # ('hybrid_similarity') are neighbour spaces here and are never fitted
    # (handoff 2.3 item 1: n01 measures something with neighbours).
    #
    # This is a scoring pass: each configuration is fitted once per fold, and
    # its predictions are graded under every setting with no further fit, so
    # every setting is covered and nothing is screened.
    #
    # How much each neighbour counts (axis 5: rank, inverse_square) does not
    # enter: a held-out molecule is a cliff when ANY of its k nearest training
    # molecules differs from it by more than the activity threshold, and no
    # neighbour label is averaged.
    # DECIDED 2026-10-06 (Adelaide delegated): the any-neighbour cliff rule
    # is kept, and there is no weighted version, so the counting schemes do
    # not apply to n01.
    #
    # TODO(similarity-sweep 2026-09-25): translating 'structurally similar'
    # from a Tanimoto threshold (0.55) or a top-2%-of-pairs cutoff to "one of
    # the k nearest training molecules" is this change's choice, not the
    # owner's. The two agree in spirit and differ in count: a k-nearest
    # definition gives every molecule k candidate partners whatever their
    # distance. Owner to confirm or name another rule.
    from experiments.noise.n07_remediation import (
        sweep_settings, sweep_ks, fold_similarity_spaces, fold_neighbours,
        FoldCheckpoint, similarity_record)
    from experiments.noise.similarity_space import neighbours_at
    from experiments._shared import input_fingerprint
    # Molecule count for the group: distinct SMILES when loaded (herg_ic50
    # has one row per measurement), else rows -- the rule
    # noise_helpers.dataset_knn_group uses.
    n_molecules = (len(set(smiles)) if smiles is not None and len(smiles)
                   else n)
    sim_settings = sweep_settings(dataset_name, n_molecules,
                                  has_smiles=smiles is not None)
    sim_ks = sweep_ks()
    sim_acc = {}
    sim_unavailable = {}
    if is_clf:
        _act_thr = None
    else:
        _act_thr = float(activity_k) * float(
            np.median(np.abs(labels - np.median(labels))))
    # One JSON line per finished fit, and one per finished fold, written as
    # each finishes and read back on resume (dirty-run bug 12). The
    # 'configurations' entry was added 2026-10-05 so lines written by the
    # five-model, every-representation version are not read back.
    ckpt = FoldCheckpoint(output_dir, 'n01_activity_cliff_framework',
                          dataset_name, tag='folds',
                          fingerprint={
                              'input': input_fingerprint(data, config),
                              'models': list(model_factories),
                              'sim_threshold': sim_threshold,
                              'activity_k': activity_k,
                              'n_folds': n_folds,
                              'configurations':
                                  'fold_configurations_2base_2hybrid_2026_10_05',
                              # 2026-10-06: repeats of one molecule left out
                              # of the reference cliffs and neighbour lists.
                              'repeat_exclusion': 'molecule_ids_2026_10_06'})

    def _sim_strata(model_name, space, entry, train_idx, val_idx, X_tr_s,
                    X_va_s, y_val, y_pred):
        """Cliff against non-cliff under every (setting, k), for one fold's
        predictions of one configuration."""
        spaces_tr, spaces_va, unav = fold_similarity_spaces(
            dataset_name, sim_settings, model_input_tr=X_tr_s,
            model_input_va=X_va_s, reps=reps, train_idx=train_idx,
            val_idx=val_idx, y_train=labels[train_idx],
            groups_train=(np.asarray(groups)[train_idx]
                          if groups is not None else None),
            base_reps=(entry or {}).get('base') or [],
            recipes=(entry or {}).get('recipes') or {},
            # 2026-10-05: the decided augmentation (Step 8) replaces the
            # 2026-09-12 "no augmentation flag". The base-with-augmentation
            # neighbour spaces (KNN record axis 1, numbers 3 and 4) are built
            # from whatever donors the run loaded; none loaded, they are
            # reported unavailable.
            augmentations=data.get('augmentations'), model_name=model_name,
            model_factory=model_factories.get(model_name),
            metric=metric_triple, task_type=task_type)
        nb = fold_neighbours(
            dataset_name, sim_settings, spaces_tr, spaces_va,
            labels[train_idx],
            # DECIDED 2026-10-06: a molecule's own repeat measurements are
            # never its neighbours (as n22). None on one-row-per-molecule
            # datasets, which keeps the lists as they were.
            exclude_groups=(None if molecule_of_row is None
                            else molecule_of_row[train_idx]),
            smiles_train=(None if smiles is None else
                          [smiles[i] for i in train_idx]),
            smiles_val=(None if smiles is None else
                        [smiles[i] for i in val_idx]),
            is_clf=is_clf)
        y_tr = labels[train_idx]
        for st in sim_settings:
            L = nb.get(st.name)
            if L is None or not L.get('available') or L.get('idx_val') is None:
                sim_unavailable.setdefault(f'{model_name}|{space}|{st.name}',
                                           unav.get(st.space)
                                           or (L or {}).get('reason'))
                continue
            for k in sim_ks:
                if L['idx_val'].shape[1] < k:
                    continue
                idx = neighbours_at(L, k, which='val')[0]
                if is_clf:
                    differs = y_tr[idx] != y_val[:, None]
                else:
                    differs = np.abs(y_tr[idx] - y_val[:, None]) > _act_thr
                if molecule_of_row is not None:
                    # exclude_groups covers the training rows' own lists only.
                    # The folds are scaffold-grouped, so a held-out molecule's
                    # repeats should never be training rows; any that are do
                    # not count as its cliff partner.
                    _same = (molecule_of_row[train_idx][idx]
                             == molecule_of_row[val_idx][:, None])
                    differs = differs & ~_same
                cl = np.any(differs, axis=1)
                row = {'n_val': int(len(y_val)), 'n_cliff': int(cl.sum()),
                       'cliff': None, 'noncliff': None}
                for key_, m_ in (('cliff', cl), ('noncliff', ~cl)):
                    if int(m_.sum()) > 1:
                        try:
                            row[key_] = float(metric_fn(y_val[m_], y_pred[m_]))
                        except Exception:
                            pass
                (sim_acc.setdefault(model_name, {}).setdefault(space, {})
                 .setdefault(st.name, {}).setdefault(str(k), [])
                 .append(row))

    def _fit_predict(model_factory, X_tr_s, X_va_s, y_train):
        model = model_factory()
        model.fit(X_tr_s, y_train)
        if metric_name == 'auc' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_va_s)
            if getattr(proba, 'ndim', 1) == 2 and proba.shape[1] == 2:
                _p = proba[:, 1]
            else:
                _p = proba
        else:
            _p = model.predict(X_va_s)
        return {'y_pred': np.asarray(_p).tolist(),
                'dtype': str(np.asarray(_p).dtype)}

    def _gap(score_c, score_nc):
        """Cliff minus non-cliff error, sign-normalised so positive = worse on
        cliffs whatever the metric's direction."""
        if score_c is None or score_nc is None or \
                np.isnan(score_c) or np.isnan(score_nc):
            return float('nan')
        return (float(score_c - score_nc) if lower_is_better
                else float(score_nc - score_c))

    # --- Per fold: configurations, fits, grading ---------------------------
    # per_fold_scores[model][space] = [{fold, name, all, cliff, noncliff}]
    per_fold_scores = {m: {sp: [] for sp in CONFIGURATION_SPACES}
                       for m in model_factories}
    fold_records = {}
    search_errors = []
    for fold_i, train_idx, val_idx in folds:
        y_train, y_val = labels[train_idx], labels[val_idx]
        val_is_cliff = cliff_mask[val_idx]
        print(f"\n    Fold {fold_i}: {len(train_idx)} training rows, "
              f"{len(val_idx)} held-out rows ({int(val_is_cliff.sum())} "
              f"reference cliffs)")
        try:
            cfg = fold_configurations(data, dataset_name, fold_i=fold_i,
                                      train_idx=train_idx, n_folds=n_folds,
                                      verbose=search_verbose)
        except Exception as e:
            search_errors.append({'fold': int(fold_i),
                                  'error': f'{type(e).__name__}: {e}'})
            print(f"      per-fold searches failed "
                  f"({type(e).__name__}: {e}); fold skipped")
            continue
        general_search = _fold_general_search(
            data, dataset_name, fold_i=fold_i, train_idx=train_idx,
            n_folds=n_folds)

        fold_rec = {'fold': int(fold_i), 'n_train': int(len(train_idx)),
                    'n_val': int(len(val_idx)),
                    'n_val_reference_cliffs': int(val_is_cliff.sum()),
                    'models': {}}
        for model_name, model_factory in model_factories.items():
            entry = cfg.get(model_name)
            ctx = f'n01 {dataset_name} fold {fold_i} {model_name}'
            inputs, unavail, names = fold_model_inputs(
                data, dataset_name, model_name, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_train, factory=model_factory,
                metric=metric_triple)
            m_rec = configuration_record(entry, names)
            # configuration_record reads 'evaluator' and 'n_reps' off each
            # hybrid, but a top_hybrids entry names them 'internal_evaluator'
            # and 'rep_count' (_shared.top_hybrids), so both come back None.
            # Filled here from the entry's own names, with the hybrid's width
            # in features (2026-10-05).
            for _h_rec, _h in zip(m_rec.get('hybrids') or [],
                                  (entry or {}).get('hybrids') or []):
                if _h_rec.get('evaluator') is None:
                    _h_rec['evaluator'] = _h.get('internal_evaluator')
                if _h_rec.get('n_reps') is None:
                    _h_rec['n_reps'] = _h.get('rep_count')
                _h_rec['n_features'] = _h.get('n_features')
            m_rec['configurations_unavailable'] = {
                sp: why for sp, why in unavail.items()
                if sp in CONFIGURATION_SPACES}
            m_rec['training_row_scores'] = _training_row_scores(
                general_search, model_name, entry, ctx)
            m_rec['held_out'] = {}
            for sp in CONFIGURATION_SPACES:
                if sp not in inputs:
                    print(f"      {model_name:<9} {sp:<17} unavailable: "
                          f"{unavail.get(sp)}")
                    continue
                X_tr_s, X_va_s = inputs[sp]
                name = names.get(sp)
                # The fit, written to the checkpoint as it finishes and read
                # back on resume. The prediction is stored at full precision
                # and rebuilt in its own dtype, so a resumed fold scores the
                # same numbers. The key names the representation or recipe,
                # because base_1 is a different representation on different
                # folds.
                _rec = ckpt.cached(
                    [model_name, sp, str(name), int(fold_i)],
                    lambda _f=model_factory, _a=X_tr_s, _b=X_va_s:
                        _fit_predict(_f, _a, _b, y_train))
                y_pred = np.asarray(_rec['y_pred'], dtype=_rec['dtype'])

                row = {'fold': int(fold_i), 'name': name,
                       'all': float(metric_fn(y_val, y_pred)),
                       'cliff': None, 'noncliff': None,
                       'n_cliff': int(val_is_cliff.sum())}
                for key_, m_ in (('cliff', val_is_cliff),
                                 ('noncliff', ~val_is_cliff)):
                    if int(m_.sum()) > 1:
                        try:
                            row[key_] = float(metric_fn(y_val[m_], y_pred[m_]))
                        except Exception:
                            pass
                row['gap'] = _gap(row['cliff'], row['noncliff'])
                per_fold_scores[model_name][sp].append(row)
                m_rec['held_out'][sp] = dict(row)
                print(f"      {model_name:<9} {sp:<17} {str(name)[:28]:<28} "
                      f"{metric_name}_all={row['all']:.4f}  gap="
                      f"{row['gap']:.4f}")

                _sim_strata(model_name, sp, entry, train_idx, val_idx,
                            X_tr_s, X_va_s, y_val, y_pred)
            fold_rec['models'][model_name] = m_rec
        fold_rec = convert_numpy(fold_rec)
        fold_records[int(fold_i)] = fold_rec
        # The fold's record, written once as the fold finishes.
        _fk = ['fold_record', int(fold_i)]
        if ckpt._k(_fk) not in ckpt.entries:
            ckpt.put(_fk, fold_rec)

    # --- Summaries over folds (fold_spread: mean and SD, or 'one fold, no
    # spread'; no significance test) -----------------------------------------
    strat_results_by_model = {}
    for model_name in model_factories:
        per_model = {}
        for sp in CONFIGURATION_SPACES:
            rows = per_fold_scores[model_name][sp]
            if not rows:
                continue
            s_all = fold_spread([r['all'] for r in rows])
            s_c = fold_spread([r['cliff'] for r in rows])
            s_nc = fold_spread([r['noncliff'] for r in rows])
            s_gap = fold_spread([r['gap'] for r in rows])
            score_c = s_c['mean'] if s_c['mean'] is not None else float('nan')
            score_nc = (s_nc['mean'] if s_nc['mean'] is not None
                        else float('nan'))
            per_model[sp] = {
                f'{metric_name}_all': s_all['mean'],
                f'{metric_name}_cliff': score_c,
                f'{metric_name}_noncliff': score_nc,
                # Difference of the fold means, as before 2026-10-05. Each
                # cliff and non-cliff mean is over the folds whose held-out
                # rows had more than 1 molecule in that stratum.
                'cliff_noncliff_gap': _gap(score_c, score_nc),
                'metric': metric_name,
                'lower_is_better': lower_is_better,
                'n_cliff_folds': s_c['n_folds'],
                'n_noncliff_folds': s_nc['n_folds'],
                'spread_all': s_all, 'spread_cliff': s_c,
                'spread_noncliff': s_nc,
                # The gap per fold, on the folds where both strata scored.
                'spread_gap': s_gap,
                # Which representation or recipe this configuration was on
                # each fold.
                'names_by_fold': {str(r['fold']): r['name'] for r in rows},
            }
        if per_model:
            # dirty-run fix 2026-10-05: the hybrid's gap is read beside the
            # gaps of the same model's 2 best base representations (ranked on
            # each fold's training rows), and each gap beside its score on all
            # held-out molecules, so a small gap from a representation that
            # predicts badly everywhere is visible as such. It used to be
            # compared with the 3 representations whose gap was closest to 0.
            per_model['_comparison'] = {
                'gap_by_configuration': {
                    sp: v['cliff_noncliff_gap'] for sp, v in per_model.items()
                    if sp in CONFIGURATION_SPACES},
                'score_all_by_configuration': {
                    sp: v[f'{metric_name}_all'] for sp, v in per_model.items()
                    if sp in CONFIGURATION_SPACES},
                'hybrid_gap': (per_model.get('hybrid') or {}).get(
                    'cliff_noncliff_gap'),
                'metric': metric_name,
                'lower_is_better': lower_is_better,
                'gap_direction': 'positive = worse on cliffs',
            }
        strat_results_by_model[model_name] = per_model

    # The similarity axis, 2026-09-25: {model: {configuration: {setting: {k:
    # {cliff, noncliff, gap, n_folds_cliff, n_folds_noncliff, mean n_cliff}}}}},
    # in the dataset's own metric, the gap sign-normalised the way
    # cliff_noncliff_gap above is (positive = worse on cliffs).
    stratified_by_similarity = {}
    for _m, _by_l in sim_acc.items():
        for _l, _by_s in _by_l.items():
            for _sn, _by_k in _by_s.items():
                for _k, _rows in _by_k.items():
                    _sc = fold_spread([r['cliff'] for r in _rows])
                    _sn_ = fold_spread([r['noncliff'] for r in _rows])
                    sc_c = (_sc['mean'] if _sc['mean'] is not None
                            else float('nan'))
                    sc_n = (_sn_['mean'] if _sn_['mean'] is not None
                            else float('nan'))
                    (stratified_by_similarity.setdefault(_m, {})
                     .setdefault(_l, {}).setdefault(_sn, {}))[_k] = {
                        f'{metric_name}_cliff': sc_c,
                        f'{metric_name}_noncliff': sc_n,
                        'cliff_noncliff_gap': _gap(sc_c, sc_n),
                        'n_cliff_folds': _sc['n_folds'],
                        'n_noncliff_folds': _sn_['n_folds'],
                        'mean_n_cliff_val': float(np.mean(
                            [r['n_cliff'] for r in _rows])),
                        'mean_n_val': float(np.mean(
                            [r['n_val'] for r in _rows])),
                    }

    return {
        'stratified_by_similarity': convert_numpy(stratified_by_similarity),
        'similarity': convert_numpy(similarity_record(
            sim_settings, None, cliff_rule=(
                'a held-out molecule is a cliff when one of its k nearest '
                'training molecules in the setting differs by more than '
                f'{_act_thr} label units' if not is_clf else
                'a held-out molecule is a cliff when one of its k nearest '
                'training molecules in the setting has another class'),
            counting_scheme=('not used: the cliff rule asks whether any of '
                             'the k neighbours differs, and averages no '
                             'neighbour label'),
            n_molecules_for_group=int(n_molecules),
            spaces_unavailable=sim_unavailable,
            checkpoint_file=ckpt.path, units_read_back=ckpt.n_resumed)),
        'cliff_source': sim_source,
        # False means the cliffs were found by concatenating the very
        # representations the cliff/non-cliff gap then grades. Any comparison
        # between representations on such a dataset is circular; the note says
        # how many features each one contributed.
        'cliff_reference_independent': bool(cliff_reference_independent),
        'cliff_reference_note': convert_numpy(cliff_reference_note),
        'cliff_info': convert_numpy(cliff_info),
        # {model: {configuration: summary over folds, '_comparison': ...}}.
        # Configurations are base_1, base_2 (the fold's 2 best single
        # representations for that model), hybrid and hybrid__general2 (its 2
        # best hybrids with distinct selected features).
        'stratified_by_model': convert_numpy(strat_results_by_model),
        'models': list(model_factories),
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in folds],
        'configurations_per_fold': (f'{len(model_factories)} models x '
                                    f'{len(CONFIGURATION_SPACES)} = '
                                    f'{len(model_factories) * len(CONFIGURATION_SPACES)}'),
        # Per fold: per model, the configurations (configuration_record:
        # base representations, hybrids with allocation, importance, budget
        # in features, evaluator, representation count, augmentation
        # condition, neighbour spaces), the training-row scores and the
        # held-out scores. selection_scope 'fold_train_rows'.
        'selection': {
            'scope': 'fold_train_rows',
            'rule': ('per outer fold and decided model: 2 best base '
                     'representations + 2 best hybrids from searches on '
                     "that fold's training rows; distance and similarity "
                     'hybrids are neighbour spaces only'),
            'grid': 'decided (DECISIONS.md), pruning off',
            'per_fold': fold_records,
            'search_errors': search_errors,
        },
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Run the cliff-stratified prediction-error analysis (molecular scope).

    Single sub-analysis: cliff-stratified prediction error. Cliffs are
    identified from a representation-INDEPENDENT reference (ECFP4 Tanimoto for
    molecular; edit distance for antibody; Euclidean concat fallback) and error
    is measured per model x rep with held-out CV, so the cliff vs non-cliff gap
    comparison is not circular.

    2026-09-12, decided: the hybrid's recipe is searched inside each fold.
    2026-10-05 (chat 5): per fold, the dataset's two decided models, each on
    its 2 best base representations and 2 best hybrids (8 fitted
    configurations), on the decided search grid.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N01: CLIFF-STRATIFIED PREDICTION ERROR")
    print("=" * 70)

    sim_threshold = kwargs.get('sim_threshold', 0.55)
    activity_threshold = kwargs.get('activity_threshold', 2.0)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    smiles = data.get('smiles', None)

    task_type = data['task_type']
    is_clf = task_type == 'classification'

    n = len(labels)
    print(f"  Dataset: {dataset_name} ({domain}, {n} samples, {len(reps)} reps)")
    print(f"  Sim threshold: {sim_threshold}  Activity threshold: {activity_threshold} MAD")

    # --- The recipe ---------------------------------------------------------
    # 2026-09-12, decided: n01 searches its recipe inside each fold, so run()
    # reads no whole-dataset recipe before the fold loop. Every configuration
    # comes from inside _run_cliff_stratified, from searches on each fold's
    # training rows (noise_helpers.fold_configurations, 2026-10-05).
    results = {
        'config': {
            'sim_threshold': sim_threshold,
            'similar_percentile': SIMILAR_PERCENTILE,
            'activity_threshold': activity_threshold,
            'domain': domain,
            'n_samples': n,
            'n_reps': len(reps),
            'recipe_search': 'per_fold',
            # 2026-10-05: the decided grid replaces speed='fast'; the
            # configurations replace every-representation + 3 hybrids.
            'search_grid': 'decided (DECISIONS.md), pruning off',
            'configurations': ('per fold and decided model: 2 base '
                               'representations + 2 hybrids'),
        },
    }

    # --- Cliff-stratified prediction error ---
    # The cliff mask inside _run_cliff_stratified uses the module
    # SIMILAR_PERCENTILE for the rep-independent reference.
    results['stratified'] = _run_cliff_stratified(
        reps, labels, groups, config, data,
        sim_threshold, activity_threshold,
        smiles=smiles,
        is_clf=is_clf,
        dataset_name=dataset_name,
        # 2026-10-05: a model list from the caller is printed and ignored;
        # the dataset's two decided models are used (R1).
        models_to_run=(kwargs.get('models') or config.get('models') or None),
        output_dir=output_dir)

    # --- Grand summary ---
    print("\n" + "=" * 70)
    print("  GRAND SUMMARY")
    print("=" * 70)

    st = results['stratified']
    if isinstance(st, dict) and st.get('stratified_by_model'):
        print(f"  Cliff minus non-cliff {st.get('metric')} over "
              f"{len(st.get('folds_run') or [])} folds, positive = worse on "
              f"cliffs; {st.get('metric')}_all beside it:")
        for _m, _pm in st['stratified_by_model'].items():
            comp = (_pm or {}).get('_comparison') or {}
            for _sp, _g in (comp.get('gap_by_configuration') or {}).items():
                _a = (comp.get('score_all_by_configuration') or {}).get(_sp)
                print(f"     {_m:<9} {_sp:<17} gap="
                      f"{_g if _g is None else round(_g, 4)}  all="
                      f"{_a if _a is None else round(_a, 4)}")

    return make_structured_output(
        experiment_id='n01',
        dataset=dataset_name,
        data=data,
        config=results.pop('config'),
        results=results,
        primary_metric='cliff_noncliff_gap',
        # The GAP is smaller-is-better; the dataset's own metric may not be
        # (spearman on the antibody sets, ci on DTA). make_structured_output
        # records this under primary_metric_lower_is_better and resolves the
        # top-level lower_is_better from the dataset itself.
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
