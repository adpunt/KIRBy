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

Usage:
    python n01_activity_cliff_framework.py --dataset logd --reps all --mode full
    python n01_activity_cliff_framework.py --dataset logd --sim-threshold 0.55 --activity-threshold 2.0
"""
import sys
import os
import copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.preprocessing import StandardScaler


# 2026-09-12, decided: n01 searches its recipe inside each fold, on the fast
# speed tier, with no augmentation flag. get_fitted_pool_hybrid and
# get_best_search_for_downstream both read a recipe searched over every row of
# the dataset, so neither is imported any more.
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split, make_cv_splitter,
    get_fitted_hybrid, flatten_fold_weights, build_fold_hybrid,
    get_best_for_downstream,
    extract_top3_base_reps,
    # Added 2026-09-18. All four read rows a search already stored and fit
    # nothing, so the block that calls them costs 0 new model fits.
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    DOWNSTREAM_MODELS_NOISE, FULL_BUDGETS,
    _resolve_search_metric,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS,
)

from kirby.hybrid import apply_feature_selection, compute_feature_importance
from kirby.hybrid import create_hybrid
from kirby.utils.noise_mitigation_methods import compute_distance_matrix, ActivityCliffDetector


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

# Seed for the fold split and for the cache namespace each fold's recipe search
# is filed under. 42 is the seed find_best_hybrid.nested_cv_outer_folds uses, so
# fold i here holds out the same rows as fold i there and the two can share a
# cached search. Added 2026-09-12 with the per-fold recipe search.
_FOLD_SEARCH_SEED = 42


def _run_cliff_stratified(reps, labels, groups, config, data,
                          sim_threshold, activity_k,
                          smiles=None, is_clf=False,
                          dataset_name=None, models_to_run=None,
                          search_verbose=False, output_dir=None):
    """Compare prediction error on cliff vs non-cliff compounds.

    2026-09-12, decided: n01 searches its recipe inside each fold, on the fast
    speed tier, with no augmentation flag — donor features are not in n01's
    pool. So this function no longer takes a blueprint from its caller. Each
    outer fold runs one search on that fold's training rows, and every
    downstream model reads its recipe out of that fold's search.

    The five downstream models (DOWNSTREAM_MODELS_NOISE) each get their own
    blueprint and their own per-representation baselines, as before.
    """
    print("\n  --- Sub-analysis 4: Cliff-Stratified Prediction Error ---")

    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']

    if models_to_run is None:
        models_to_run = list(DOWNSTREAM_MODELS_NOISE)

    model_factories = get_model_factories(
        list(models_to_run), task_type=task_type,
        n_samples=len(labels))
    if not model_factories:
        return {'error': 'no_model_factories'}

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

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
    n = len(labels)
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
        cliff_mask, cliff_info = identify_cliff_compounds_editdist(
            heavy_seqs, labels, activity_k=activity_k,
            is_clf=is_clf, similar_percentile=SIMILAR_PERCENTILE)
        cliff_indices = set(np.where(cliff_mask)[0].tolist())
        sim_source = f'edit_distance(similar_pct={SIMILAR_PERCENTILE})'
    elif smiles is not None and domain == 'molecular':
        try:
            ref_sim = _compute_ecfp4_tanimoto_sim(smiles)
            sim_source = 'ecfp4_tanimoto'
            cliff_pair_mask, cliff_info = identify_cliff_pairs(
                ref_sim, labels, sim_threshold, activity_k, is_clf=is_clf)
            cliff_mask = cliff_pair_mask.any(axis=1)
            cliff_indices = set(np.where(cliff_mask)[0].tolist())
        except Exception as e:
            print(f"    WARNING: ECFP4 Tanimoto failed ({e}), falling back to Euclidean")
            ref_sim, cliff_reference_note = _concat_reference()
            cliff_reference_independent = False
            sim_source = 'euclidean_concat'
            cliff_pair_mask, cliff_info = identify_cliff_pairs(
                ref_sim, labels, sim_threshold, activity_k, is_clf=is_clf,
                similar_percentile=SIMILAR_PERCENTILE)
            cliff_mask = cliff_pair_mask.any(axis=1)
            cliff_indices = set(np.where(cliff_mask)[0].tolist())
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

        cliff_pair_mask, cliff_info = identify_cliff_pairs(
            ref_sim, labels, sim_threshold, activity_k, is_clf=is_clf,
            similar_percentile=SIMILAR_PERCENTILE)
        cliff_mask = cliff_pair_mask.any(axis=1)
        cliff_indices = set(np.where(cliff_mask)[0].tolist())

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
            'cliff_info': cliff_info,
            'error': 'too_few_cliff_compounds',
        }

    # The folds are built once and held, because the recipe search now runs on
    # each fold's training rows and the same fold must be the one scored.
    # make_cv_splitter at seed 42 builds the same splitter as
    # find_best_hybrid.nested_cv_outer_folds, so fold i here is fold i of the
    # nested-CV cache namespace used below.
    splitter, use_groups = make_cv_splitter(n_folds, groups, _FOLD_SEARCH_SEED,
                                            task_type=task_type)
    sample_X = next(iter(reps.values()))
    split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)
    folds = list(splitter.split(*split_args))

    # --- The recipe, searched inside each fold ------------------------------
    # 2026-09-12, decided: n01 searches its recipe inside each fold, on the
    # fast speed tier, with no augmentation flag. What each part of that
    # changes here:
    #   inside each fold — one find_best_hybrid per outer fold, on that fold's
    #     training rows only, filed under that fold's nested-CV cache
    #     namespace. n01 used to read one search run over every row, so the
    #     held-out rows' labels had been seen when the recipe was picked.
    #   fast speed tier — speed='fast' takes find_best_hybrid's fast lists:
    #     3 allocation methods, 2 importance methods, 2 internal evaluators,
    #     instead of 5, 3 and 4.
    #   no augmentation flag — augmentations=None and augmentation_strategy
    #     'none', so features donated by other representations are not in the
    #     pool the search picks from. The 'uncapped' and 'capped' conditions
    #     are not searched and the per-fold rebuild below passes no donors
    #     either.
    # The downstream model list is passed by name rather than left to
    # speed='fast', which would cut it to rf and lightgbm. n01 reports a row
    # for xgboost, svm and dnn as well, and get_best_for_downstream answers a
    # request for a model the search never scored with another model's
    # blueprint without saying so — those three rows would silently carry rf's
    # recipe. Every row records model_scored_by_name so this stays visible.
    from kirby.find_best_hybrid import (find_best_hybrid,
                                        nested_cv_cache_namespace)

    search_metric, search_task = _resolve_search_metric(
        labels, task_type=task_type, metric=metric_name, domain=domain,
        dataset_name=dataset_name)

    search_models = list(model_factories.keys())
    print(f"    Recipe: searched inside each of {len(folds)} folds, on that "
          f"fold's training rows, speed='fast', no augmentation flag")
    print(f"    Search asks {len(search_models)} downstream models "
          f"({', '.join(search_models)}) over {len(FULL_BUDGETS)} budgets, "
          f"metric {search_metric}")

    fold_searches = {}
    fold_search_errors = []
    # TODO(similarity-sweep 2026-09-25): the owner's run settings for the
    # prelim noise runs are KIRBY_NO_COLD_SEARCH=1 -- experiments read the
    # precompute and never start a search. n01 searches inside each fold
    # (decided 2026-09-12, above), so under that setting a fold whose search
    # is not already in the nested-CV cache namespace is refused and the
    # hybrid arm drops that fold (fold_search_errors records it). Which of the
    # two decisions wins for n01 is the owner's call; nothing here changes it.
    for fold_i, (train_idx, _val_idx) in enumerate(folds):
        try:
            fold_searches[fold_i] = find_best_hybrid(
                base_reps={k: np.asarray(v)[train_idx] for k, v in reps.items()},
                labels=labels[train_idx],
                groups=(np.asarray(groups)[train_idx]
                        if groups is not None else None),
                budgets=FULL_BUDGETS,
                models=search_models,
                augmentations=None,
                augmentation_strategy='none',
                speed='fast',
                metric=search_metric,
                task_type=search_task,
                # find_best_hybrid(cache=True) raises without a dataset_name —
                # the cache key is the filename. A caller with no dataset name
                # gets the same search, unstored, instead of the raise.
                cache=dataset_name is not None,
                verbose=search_verbose,
                dataset_name=dataset_name,
                tier='general',
                cache_namespace=nested_cv_cache_namespace(
                    fold_i, len(folds), _FOLD_SEARCH_SEED),
            )
        except Exception as e:
            fold_searches[fold_i] = None
            fold_search_errors.append({'fold': int(fold_i),
                                       'error': f'{type(e).__name__}: {e}'})
            print(f"      fold {fold_i}: recipe search failed "
                  f"({type(e).__name__}: {e})")

    # The distance counterpart of each fold's search (the owner's set,
    # 2026-10-02: top 2 general + top 1 distance hybrids per model). Same
    # fold, same training rows, same speed and no augmentation flag; the
    # distance settings are the ones the precompute's distance searches use:
    # tier 'general_distance', DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
    # DISTANCE_MODELS.
    fold_dist_searches = {}
    fold_dist_search_errors = []
    for fold_i, (train_idx, _val_idx) in enumerate(folds):
        try:
            fold_dist_searches[fold_i] = find_best_hybrid(
                base_reps={k: np.asarray(v)[train_idx] for k, v in reps.items()},
                labels=labels[train_idx],
                groups=(np.asarray(groups)[train_idx]
                        if groups is not None else None),
                budgets=DISTANCE_BUDGETS,
                internal_evaluators=DISTANCE_EVALUATORS,
                models=list(DISTANCE_MODELS),
                augmentations=None,
                augmentation_strategy='none',
                speed='fast',
                metric=search_metric,
                task_type=search_task,
                cache=dataset_name is not None,
                verbose=search_verbose,
                dataset_name=dataset_name,
                tier='general_distance',
                cache_namespace=nested_cv_cache_namespace(
                    fold_i, len(folds), _FOLD_SEARCH_SEED),
            )
        except Exception as e:
            fold_dist_searches[fold_i] = None
            fold_dist_search_errors.append({'fold': int(fold_i),
                                            'error': f'{type(e).__name__}: {e}'})
            print(f"      fold {fold_i}: distance search failed "
                  f"({type(e).__name__}: {e})")

    # One blueprint per (model, fold), read out of that fold's search.
    bp_by_model_fold = {m: {} for m in model_factories}
    recipe_rows = []
    for fold_i, search in fold_searches.items():
        if search is None:
            continue
        for model_name in model_factories:
            best = get_best_for_downstream(search, model_name, budget=None)
            if not best or best.get('blueprint') is None:
                continue
            bp = copy.deepcopy(best['blueprint'])
            # Donor features are out of the pool by decision, so the rebuild
            # must not switch them back on.
            try:
                bp.augmentation_strategy = 'none'
            except Exception:
                pass
            bp_by_model_fold[model_name][fold_i] = bp
            scored_by_name = bool(
                (getattr(search, 'best_by_model_budget', None) or {}).get(model_name)
                or (getattr(search, 'best_by_model', None) or {}).get(model_name))
            recipe_rows.append({
                'fold': int(fold_i),
                'model': model_name,
                # False = this fold's search scored no configuration under this
                # model's name, so the blueprint came from another model.
                'model_scored_by_name': scored_by_name,
                'budget': best.get('budget', getattr(bp, 'budget', None)),
                'allocation_method': getattr(bp, 'allocation_method', None),
                'importance_method': getattr(bp, 'importance_method', None),
                'internal_evaluator': getattr(bp, 'internal_evaluator', None),
                'score_mean': best.get('score_mean'),
                'aug_condition': 'none',
                'selection_scope': 'fold_train_rows',
            })

    # The owner's set (2026-10-02), per (model, fold): the top 2 hybrids of
    # that fold's general search and the top 1 of its distance search, each
    # built and scored as its own arm. top_hybrids reads only the asked
    # model's rows, so a slot the search did not fill stays empty and its
    # reason goes to rule_missing. general1 falls back to the
    # get_best_for_downstream pick above when top_hybrids returns nothing,
    # which is what this arm used before.
    RULE_SLOTS = ('general1', 'general2', 'distance1')
    rule_bp = {m: {sl: {} for sl in RULE_SLOTS} for m in model_factories}
    rule_rows, rule_missing = [], []
    for fold_i in range(len(folds)):
        for model_name in model_factories:
            for lineage, srch, n_take in (
                    ('general', fold_searches.get(fold_i), 2),
                    ('distance', fold_dist_searches.get(fold_i), 1)):
                sel = []
                if srch is None:
                    why = (f'the {lineage} search on this fold failed or was '
                           f'not run')
                else:
                    try:
                        _t = top_hybrids(srch, model_name, n_take,
                                         require_full=False,
                                         context=f'n01 {dataset_name} fold '
                                                 f'{fold_i} {lineage}')
                        sel = list(_t.get('selected') or [])
                        why = _t.get('shortfall_reason') or (
                            f'only {len(sel)} configuration(s) scored')
                    except Exception as e:
                        why = f'top_hybrids raised {type(e).__name__}: {e}'
                for r in range(1, n_take + 1):
                    slot = f'{lineage}{r}'
                    e = sel[r - 1] if len(sel) >= r else None
                    bp = e.get('blueprint') if e else None
                    if bp is None and slot == 'general1':
                        bp = bp_by_model_fold[model_name].get(fold_i)
                    if bp is None:
                        rule_missing.append(
                            f'{model_name} fold {fold_i} {slot}: {why}')
                        continue
                    bp = copy.deepcopy(bp)
                    try:
                        bp.augmentation_strategy = 'none'
                    except Exception:
                        pass
                    rule_bp[model_name][slot][fold_i] = bp
                    rule_rows.append({
                        'fold': int(fold_i), 'model': model_name,
                        'slot': slot, 'lineage': lineage,
                        'tier': ('general_distance' if lineage == 'distance'
                                 else 'general'),
                        'label': e.get('label') if e else None,
                        'score_mean': e.get('score_mean') if e else None,
                        'budget': getattr(bp, 'budget', None),
                        'allocation_method': getattr(bp, 'allocation_method',
                                                     None),
                        'importance_method': getattr(bp, 'importance_method',
                                                     None),
                        'internal_evaluator': getattr(bp, 'internal_evaluator',
                                                      None),
                        'from_top_hybrids': e is not None,
                    })
    for _m in model_factories:
        if rule_bp[_m]['general1']:
            bp_by_model_fold[_m] = dict(rule_bp[_m]['general1'])

    for row in recipe_rows:
        flag = '' if row['model_scored_by_name'] else '  [blueprint from another model]'
        print(f"      fold {row['fold']} {row['model']:<9} "
              f"budget={row['budget']} {row['allocation_method']}/"
              f"{row['importance_method']}/{row['internal_evaluator']}{flag}")

    # Top-3 base representations. They used to be ranked by one whole-dataset
    # search; now each fold's own search ranks them and the printed three are
    # ranked by the mean of a representation's rf baseline over the folds that
    # scored it. The per-fold lists are kept beside it.
    top3_by_fold = {}
    _rank_scores = {}
    for fold_i, search in fold_searches.items():
        if search is None:
            continue
        _t3 = extract_top3_base_reps(search)
        if _t3:
            top3_by_fold[int(fold_i)] = [(str(nm), float(sc)) for nm, sc in _t3]
        for rep_name, model_scores in (getattr(search, 'baselines', {}) or {}).items():
            if str(rep_name).startswith('_'):
                continue
            sc = (model_scores or {}).get('rf')
            if sc is None or not np.isfinite(sc):
                continue
            _rank_scores.setdefault(str(rep_name), []).append(float(sc))
    top3_base_reps = sorted(
        ((k, float(np.mean(v))) for k, v in _rank_scores.items() if v),
        key=lambda kv: kv[1], reverse=not lower_is_better)[:3]

    # --- Top-2 rows and the two hybrid deltas, added beside the above -------
    # 2026-09-18, owner: add the top 2 rows as an extra and change nothing that
    # is already reported. Nothing above this line moves. top3_base_reps,
    # top3_base_reps_by_fold and recipe_rows keep the numbers they already had,
    # and the sweep is untouched: each fold's search still scores every
    # representation in `reps` over all 8 budgets in FULL_BUDGETS.
    #
    # What a result file could not say before this block existed. recipe_rows
    # holds ONE configuration per (fold, downstream model) -- the argmax
    # get_best_for_downstream returns -- so the second-best configuration was
    # never written down and the gap to it could not be read off the file. The
    # representation ranking above is rf's alone: extract_top3_base_reps
    # (_shared.py:809) reads the 'rf' baseline and _rank_scores reads 'rf' at
    # line 568, so the other 4 downstream models had no representation ranking
    # stored at all.
    #
    # All four helpers read rows the per-fold searches already stored. 0 new
    # model fits, 0 new searches.
    #
    # require_full=False everywhere: a downstream model this fold's search
    # never scored yields a short list with 'shortfall_reason' set, instead of
    # raising and killing a run whose per-representation baselines are fine.
    _HYB_KEEP = ('rank', 'label', 'score_mean', 'score_std', 'n_folds_scored',
                 'n_folds_asked', 'n_features', 'model', 'allocation_method',
                 'importance_method', 'budget', 'internal_evaluator',
                 'step_size', 'rep_count', 'rep_count_source', 'rep_subset',
                 'tune', 'tune_source', 'tied_with')

    def _trim_hybrid_sel(sel):
        """The hybrid rows without the pickled Blueprint. convert_numpy walks a
        dataclass through asdict (_shared.py:7615-7617), so a Blueprint left in
        here writes its per-fold feature allocation into the result JSON."""
        out = {k: v for k, v in sel.items() if k != 'selected'}
        out['selected'] = [{k: e.get(k) for k in _HYB_KEEP}
                           for e in sel['selected']]
        return out

    top2_base_reps_by_fold = {}
    top2_hybrids_by_fold = {}
    top2_pairs_by_model_by_fold = {}
    hybrid_deltas_by_model_by_fold = {}
    # WHICH DOWNSTREAM MODELS THE FOUR BLOCKS BELOW COVER, and why a name is
    # missing from one. Added 2026-09-28. Every reason below was printed to the
    # terminal and saved nowhere, so a result file holding three models read the
    # same as a run configured for three.
    #
    # One dict of {downstream model: reason}, and the fold is named inside each
    # reason, because the blocks are per fold and a model can come back on one
    # fold and not another. The last reason written for a model wins; the fold
    # numbers in the text say how many folds it covers.
    top2_models_compared = list(search_models)
    top2_models_dropped = {}
    for fold_i, search in fold_searches.items():
        if search is None:
            continue
        _ctx = f'n01 {dataset_name} fold {fold_i}'
        _base_f, _hyb_f, _delta_f = {}, {}, {}
        for model_name in search_models:
            try:
                _base_f[model_name] = top_base_reps(
                    search, model_name, 2, require_full=False, context=_ctx)
            except Exception as e:
                print(f"      [warn] top-2 base representations "
                      f"(fold {fold_i}, {model_name}): "
                      f"{type(e).__name__}: {str(e)[:150]}")
                top2_models_dropped[model_name] = (
                    'no top-2 base representations on fold %s: %s: %s'
                    % (fold_i, type(e).__name__, str(e)[:150]))
            _h = None
            try:
                _h = top_hybrids(search, model_name, 2, require_full=False,
                                 context=_ctx)
                _hyb_f[model_name] = _trim_hybrid_sel(_h)
            except Exception as e:
                print(f"      [warn] top-2 hybrid configurations "
                      f"(fold {fold_i}, {model_name}): "
                      f"{type(e).__name__}: {str(e)[:150]}")
                top2_models_dropped[model_name] = (
                    'no top-2 hybrid configurations on fold %s: %s: %s'
                    % (fold_i, type(e).__name__, str(e)[:150]))
            # The rank-1 hybrid is handed over explicitly. Called with
            # hybrid=None, hybrid_deltas reads _picked['selected'][0]['row']
            # (_shared.py:7424) and top_hybrids builds no 'row' key in its
            # entries (:7043-7066), so that path raises KeyError. The rank-1
            # entry itself carries both fields hybrid_deltas reads off it,
            # score_mean and blueprint.
            if _h and _h.get('selected'):
                try:
                    _delta_f[model_name] = hybrid_deltas(
                        search, model_name, hybrid=_h['selected'][0],
                        context=_ctx)
                except Exception as e:
                    print(f"      [warn] hybrid deltas "
                          f"(fold {fold_i}, {model_name}): "
                          f"{type(e).__name__}: {str(e)[:150]}")
                    top2_models_dropped[model_name] = (
                        'no hybrid deltas on fold %s: %s: %s'
                        % (fold_i, type(e).__name__, str(e)[:150]))
        if _base_f:
            top2_base_reps_by_fold[int(fold_i)] = _base_f
        if _hyb_f:
            top2_hybrids_by_fold[int(fold_i)] = _hyb_f
        if _delta_f:
            hybrid_deltas_by_model_by_fold[int(fold_i)] = _delta_f

        # models=search_models, not the default: DOWNSTREAM_MODELS is 5 names
        # (_shared.py:539) and models_to_run can be shorter, so the default
        # would count a downstream model this run never asked for as missing.
        try:
            _pairs = select_model_rep_pairs(
                search, models=search_models, n_base=2, n_hybrids=2,
                require_full=False, context=_ctx)
            _row = {k: _pairs[k] for k in (
                'models', 'n_base', 'n_hybrids', 'n_pairs_possible', 'n_pairs',
                'n_missing', 'complete', 'n_distinct_units', 'metric_name',
                'shortfalls')}
            # 'per_model' holds the same two selections again, each hybrid with
            # its Blueprint attached, and 'distinct_units' is keyed by tuples.
            # Both are dropped: the per-model detail is already above under
            # top2_base_reps_by_fold and top2_hybrids_by_fold.
            _row['pairs'] = [
                {'downstream_model': p['downstream_model'], 'kind': p['kind'],
                 'rank_within_model': p['rank_within_model'],
                 'label': p['label'], 'score': p['score'],
                 'unit_id': str(p['unit_id'])}
                for p in _pairs['pairs']]
            _row['collapsed'] = [
                {'unit_id': str(c['unit_id']), 'label': c['label'],
                 'chosen_by': [list(t) for t in c['chosen_by']]}
                for c in _pairs['collapsed']]
            top2_pairs_by_model_by_fold[int(fold_i)] = _row
        except Exception as e:
            print(f"      [warn] top-2 model-and-representation pairs "
                  f"(fold {fold_i}): {type(e).__name__}: {str(e)[:150]}")
            for _m in search_models:
                top2_models_dropped.setdefault(_m, (
                    'no model-and-representation pairs on fold %s: %s: %s'
                    % (fold_i, type(e).__name__, str(e)[:150])))

    if top2_pairs_by_model_by_fold:
        _n_pairs = sum(r['n_pairs'] for r in top2_pairs_by_model_by_fold.values())
        _n_poss = sum(r['n_pairs_possible']
                      for r in top2_pairs_by_model_by_fold.values())
        print(f"    Top-2 rows: {_n_pairs} of {_n_poss} "
              f"model-and-representation pairs filled across "
              f"{len(top2_pairs_by_model_by_fold)} folds "
              f"({len(search_models)} downstream models x "
              f"(2 base representations + 2 hybrids) per fold), "
              f"0 new model fits")

    # ── The similarity axis and the per-fold checkpoint, 2026-09-25 ────────
    # Plan sections 1-3. The cliffs above are ONE similarity: ECFP4 Tanimoto
    # at 0.55 over every pair of molecules (or the fallbacks), and every key
    # this function returned before 2026-09-25 is graded on them, unchanged.
    #
    # The sweep adds the cliff definition under every similarity setting at
    # every k in SWEPT_KS: a held-out molecule is a cliff when one of its k
    # nearest TRAINING molecules in that setting's features differs from it by
    # more than the activity threshold above (activity_k x the label MAD; a
    # different class on a classification set), and the same predictions are
    # re-graded cliff against non-cliff. Neighbours come from
    # n07.fold_neighbours (noise_cache), validation rows against training
    # rows, so the held-out labels define the stratum and never a feature.
    #
    # TODO(similarity-sweep 2026-09-25): translating 'structurally similar'
    # from a Tanimoto threshold (0.55) or a top-2%-of-pairs cutoff to "one of
    # the k nearest training molecules" is this change's choice, not the
    # owner's. The two agree in spirit and differ in count: a k-nearest
    # definition gives every molecule k candidate partners whatever their
    # distance. Owner to confirm or name another rule.
    #
    # n01 has no noise and changes no training data, so it writes no
    # D-versus-D' rows; each (model, model input, fold) fit is checkpointed
    # instead and read back on resume.
    from experiments.noise.n07_remediation import (
        sweep_settings, sweep_ks, usable_ks, fold_similarity_spaces,
        fold_neighbours, FoldCheckpoint, similarity_record,
        cached_build_fold_hybrid)
    from experiments.noise.similarity_space import neighbours_at
    from experiments._shared import input_fingerprint
    sim_settings = sweep_settings()
    sim_ks = sweep_ks()
    sim_acc = {}
    sim_unavailable = {}
    if is_clf:
        _act_thr = None
    else:
        _act_thr = float(activity_k) * float(
            np.median(np.abs(labels - np.median(labels))))
    ckpt = FoldCheckpoint(output_dir, 'n01_activity_cliff_framework',
                          dataset_name or 'unnamed', tag='folds',
                          fingerprint={
                              'input': input_fingerprint(data, config),
                              'models': list(model_factories),
                              'sim_threshold': sim_threshold,
                              'activity_k': activity_k,
                              'n_folds': len(folds)})

    def _fold_top2(model_name, fold_i):
        """This fold's search's top 2 single representations for the model
        (top2_base_reps_by_fold, filled above). [] when it has none."""
        sel = ((top2_base_reps_by_fold.get(int(fold_i)) or {})
               .get(model_name) or {}).get('selected') or []
        return [e.get('rep') for e in sel if e.get('rep') in reps][:2]

    def _sim_recipes_fold(model_name, fold_i, bp_f):
        """The fold's hybrid recipes for the similarity spaces: 'hybrid' as
        before, and 'hybrid__general2' (owner decision 2026-10-03) from this
        fold's general rank-2 recipe (rule_bp, filled above) when the search
        held one; otherwise fold_similarity_spaces reports it unavailable."""
        rec = {'hybrid': (bp_f, 'none')} if bp_f is not None else {}
        bp_g2 = (rule_bp.get(model_name) or {}).get('general2', {}).get(fold_i)
        if bp_g2 is not None:
            rec['hybrid__general2'] = (bp_g2, 'none')
        return rec

    def _sim_strata(model_name, label, fold_i, train_idx, val_idx, X_tr_s,
                    X_va_s, y_val, y_pred):
        """Cliff against non-cliff under every (setting, k), for one fold's
        predictions."""
        bp_f = (bp_by_model_fold.get(model_name) or {}).get(fold_i)
        spaces_tr, spaces_va, unav = fold_similarity_spaces(
            dataset_name, sim_settings, model_input_tr=X_tr_s,
            model_input_va=X_va_s, reps=reps, train_idx=train_idx,
            val_idx=val_idx, y_train=labels[train_idx],
            groups_train=(np.asarray(groups)[train_idx]
                          if groups is not None else None),
            base_reps=_fold_top2(model_name, fold_i),
            recipes=_sim_recipes_fold(model_name, fold_i, bp_f),
            augmentations=None, model_name=model_name,
            model_factory=model_factories.get(model_name),
            metric=(metric_name, metric_fn, lower_is_better),
            task_type=task_type)
        nb = fold_neighbours(
            dataset_name, sim_settings, spaces_tr, spaces_va,
            labels[train_idx],
            smiles_train=(None if smiles is None else
                          [smiles[i] for i in train_idx]),
            smiles_val=(None if smiles is None else
                        [smiles[i] for i in val_idx]),
            is_clf=is_clf)
        y_tr = labels[train_idx]
        for st in sim_settings:
            L = nb.get(st.name)
            if L is None or not L.get('available') or L.get('idx_val') is None:
                sim_unavailable.setdefault(f'{model_name}|{label}|{st.name}',
                                           unav.get(st.space)
                                           or (L or {}).get('reason'))
                continue
            for k in sim_ks:
                if L['idx_val'].shape[1] < k:
                    continue
                idx = neighbours_at(L, k, which='val')[0]
                if is_clf:
                    cl = np.any(y_tr[idx] != y_val[:, None], axis=1)
                else:
                    cl = np.any(np.abs(y_tr[idx] - y_val[:, None]) > _act_thr,
                                axis=1)
                row = {'n_val': int(len(y_val)), 'n_cliff': int(cl.sum()),
                       'cliff': None, 'noncliff': None}
                for key_, m_ in (('cliff', cl), ('noncliff', ~cl)):
                    if int(m_.sum()) > 1:
                        try:
                            row[key_] = float(metric_fn(y_val[m_], y_pred[m_]))
                        except Exception:
                            pass
                (sim_acc.setdefault(model_name, {}).setdefault(label, {})
                 .setdefault(st.name, {}).setdefault(str(k), [])
                 .append(row))

    def _evaluate_condition(get_Xy_fn, label, model_factory, model_name=None):
        """Run CV, return {metric_all, metric_cliff, metric_noncliff, gap, ...}."""
        fold_all, fold_cliff, fold_noncliff = [], [], []

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            y_train, y_val = labels[train_idx], labels[val_idx]
            try:
                X_tr, X_va = get_Xy_fn(train_idx, val_idx, y_train, fold_i)
            except Exception:
                continue
            if X_tr is None:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            def _fit_predict(X_tr_s=X_tr_s, X_va_s=X_va_s, y_train=y_train):
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
            # The fit, written to the checkpoint as the fold finishes and read
            # back on resume (2026-09-25). The prediction is stored at full
            # precision and rebuilt in its own dtype, so a resumed fold scores
            # the same numbers.
            _rec = ckpt.cached([model_name, label, int(fold_i)], _fit_predict)
            y_pred = np.asarray(_rec['y_pred'], dtype=_rec['dtype'])

            fold_all.append(metric_fn(y_val, y_pred))
            _sim_strata(model_name, label, fold_i, train_idx, val_idx,
                        X_tr_s, X_va_s, y_val, y_pred)

            val_cliff = np.array([i for i, vi in enumerate(val_idx)
                                  if vi in cliff_indices])
            val_noncliff = np.array([i for i, vi in enumerate(val_idx)
                                     if vi not in cliff_indices])

            if len(val_cliff) > 1:
                try:
                    fold_cliff.append(metric_fn(y_val[val_cliff], y_pred[val_cliff]))
                except Exception:
                    pass
            if len(val_noncliff) > 1:
                try:
                    fold_noncliff.append(metric_fn(y_val[val_noncliff],
                                                   y_pred[val_noncliff]))
                except Exception:
                    pass

        if not fold_all:
            return None

        score_c = float(np.mean(fold_cliff)) if fold_cliff else float('nan')
        score_nc = float(np.mean(fold_noncliff)) if fold_noncliff else float('nan')

        gap = float('nan')
        if not (np.isnan(score_c) or np.isnan(score_nc)):
            if lower_is_better:
                gap = float(score_c - score_nc)
            else:
                gap = float(score_nc - score_c)

        return {
            f'{metric_name}_all': float(np.mean(fold_all)),
            f'{metric_name}_cliff': score_c,
            f'{metric_name}_noncliff': score_nc,
            'cliff_noncliff_gap': gap,
            'metric': metric_name,
            'lower_is_better': lower_is_better,
            # Sample-support behind the gap so degenerate small-sample gaps can
            # be filtered downstream: each cliff/non-cliff score is a mean over
            # only the folds that had >1 cliff (resp. non-cliff) compound in the
            # held-out set. On tiny antibody/reaction sets this can be 1-2 folds,
            # which is what produces the wild negative gaps.
            'n_cliff_folds': len(fold_cliff),
            'n_noncliff_folds': len(fold_noncliff),
        }

    # Per-model loop (matched evaluator). Each downstream model uses its own
    # blueprint, and per-rep baselines are also reported per-model so the
    # cliff-noncliff gap claim isn't single-model-biased (audit P1 finding).
    #
    # 2026-09-12: the blueprint is now per (model, fold), so a model whose
    # search failed on every fold still gets its per-representation baselines —
    # the hybrid arm is the only one that drops out.
    strat_results_by_model = {}
    for model_name, model_factory in model_factories.items():
        model_bps = bp_by_model_fold.get(model_name) or {}
        per_model = {}

        print(f"\n    [{model_name}] Per-rep baselines:")
        for rep_name, X_rep in reps.items():
            def _get_rep(tr, va, yt, fold_i, _X=X_rep):
                return _X[tr], _X[va]
            res = _evaluate_condition(_get_rep, rep_name, model_factory,
                                      model_name=model_name)
            if res:
                per_model[rep_name] = res
                mn = res['metric']
                print(f"      {rep_name:<20} {mn}_all={res[f'{mn}_all']:.4f}  "
                      f"{mn}_cliff={res[f'{mn}_cliff']:.4f}  "
                      f"{mn}_noncliff={res[f'{mn}_noncliff']:.4f}  "
                      f"gap={res['cliff_noncliff_gap']:.4f}")

        # The owner's set (2026-10-02): one arm per slot. general1 keeps the
        # existing '_hybrid' key; the others are '_hybrid__general2' and
        # '_hybrid__distance1'.
        for _slot in RULE_SLOTS:
            _slot_bps = rule_bp[model_name][_slot]
            if not _slot_bps:
                continue
            if _slot == 'general1':
                _label, _key = 'hybrid', '_hybrid'
            else:
                _label, _key = f'hybrid__{_slot}', f'_hybrid__{_slot}'
            model_bps = _slot_bps
            print(f"    [{model_name}] Hybrid ({_slot}):")

            # _mf binds the loop's model_factory at definition time, the same
            # way _bps already does — a late-bound free variable would hand the
            # LAST model in the loop to every arm.
            #
            # 2026-09-12: the blueprint is looked up by fold, because each fold
            # searched its own. A fold whose search failed returns (None, None)
            # and _evaluate_condition skips it. No augmentation dictionaries
            # are passed: donor features are not in n01's pool by decision.
            def _get_hybrid(tr, va, yt, fold_i, _bps=model_bps,
                            _mf=model_factory):
                _bp = _bps.get(fold_i)
                if _bp is None:
                    return None, None
                train_reps = {name: rep[tr] for name, rep in reps.items()}
                val_reps = {name: rep[va] for name, rep in reps.items()}
                # Stored through noise_cache since 2026-09-25
                # (n07.cached_build_fold_hybrid): the same call, keyed on its
                # inputs, so a resumed run does not rebuild it.
                X_tr_h, X_va_h, _ = cached_build_fold_hybrid(
                    dataset_name or 'unnamed',
                    train_reps, val_reps, yt, _bp,
                    augmentations_train=None, augmentations_val=None,
                    groups=(groups[tr] if groups is not None else None),
                    model=_mf(),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better)
                return X_tr_h, X_va_h

            res = _evaluate_condition(_get_hybrid, _label, model_factory,
                                      model_name=model_name)
            if res:
                per_model[_key] = res
                mn = res['metric']
                print(f"      {_key:<20} {mn}_all={res[f'{mn}_all']:.4f}  "
                      f"{mn}_cliff={res[f'{mn}_cliff']:.4f}  "
                      f"{mn}_noncliff={res[f'{mn}_noncliff']:.4f}  "
                      f"gap={res['cliff_noncliff_gap']:.4f}")

        # Every key starting '_' is a hybrid arm or this comparison, never a
        # representation.
        rep_gap_list = [(k, v['cliff_noncliff_gap'])
                        for k, v in per_model.items()
                        if not k.startswith('_')
                        and not np.isnan(v.get('cliff_noncliff_gap', float('nan')))]
        if rep_gap_list and '_hybrid' in per_model:
            top3_gap = sorted(rep_gap_list, key=lambda x: abs(x[1]))[:3]
            hybrid_gap = per_model['_hybrid']['cliff_noncliff_gap']
            per_model['_comparison'] = {
                'top3_smallest_gap': top3_gap,
                'hybrid_gap': hybrid_gap,
                # The gap of every arm of the owner's set, by slot.
                'hybrid_gap_by_slot': {
                    sl: per_model[k]['cliff_noncliff_gap']
                    for sl, k in (('general1', '_hybrid'),
                                  ('general2', '_hybrid__general2'),
                                  ('distance1', '_hybrid__distance1'))
                    if k in per_model},
            }

        strat_results_by_model[model_name] = per_model

    # Legacy flat structure: RF view (or first available) for backwards
    # compatibility. Downstream readers preferring multi-model should use
    # strat_results_by_model.
    _legacy_model = 'rf' if 'rf' in strat_results_by_model else (
        next(iter(strat_results_by_model)) if strat_results_by_model else None)
    strat_results = (dict(strat_results_by_model[_legacy_model])
                     if _legacy_model else {})

    # The similarity axis, 2026-09-25: {model: {model input: {setting: {k:
    # {cliff, noncliff, gap, n_folds_cliff, n_folds_noncliff, mean n_cliff}}}}},
    # in the dataset's own metric, the gap sign-normalised the way
    # cliff_noncliff_gap above is (positive = worse on cliffs).
    stratified_by_similarity = {}
    for _m, _by_l in sim_acc.items():
        for _l, _by_s in _by_l.items():
            for _sn, _by_k in _by_s.items():
                for _k, _rows in _by_k.items():
                    _c = [r['cliff'] for r in _rows if r['cliff'] is not None]
                    _n = [r['noncliff'] for r in _rows
                          if r['noncliff'] is not None]
                    sc_c = float(np.mean(_c)) if _c else float('nan')
                    sc_n = float(np.mean(_n)) if _n else float('nan')
                    gap = (float('nan') if (np.isnan(sc_c) or np.isnan(sc_n))
                           else float(sc_c - sc_n) if lower_is_better
                           else float(sc_n - sc_c))
                    (stratified_by_similarity.setdefault(_m, {})
                     .setdefault(_l, {}).setdefault(_sn, {}))[_k] = {
                        f'{metric_name}_cliff': sc_c,
                        f'{metric_name}_noncliff': sc_n,
                        'cliff_noncliff_gap': gap,
                        'n_cliff_folds': len(_c),
                        'n_noncliff_folds': len(_n),
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
            spaces_unavailable=sim_unavailable,
            checkpoint_file=ckpt.path, folds_read_back=ckpt.n_resumed)),
        'cliff_source': sim_source,
        # False means the cliffs were found by concatenating the very
        # representations the cliff/non-cliff gap then grades. Any comparison
        # between representations on such a dataset is circular; the note says
        # how many features each one contributed.
        'cliff_reference_independent': bool(cliff_reference_independent),
        'cliff_reference_note': convert_numpy(cliff_reference_note),
        'cliff_info': convert_numpy(cliff_info),
        'stratified': convert_numpy(strat_results),
        'stratified_by_model': convert_numpy(strat_results_by_model),
        'models': list(strat_results_by_model.keys()),
        # 2026-09-12, decided: recipe searched inside each fold, fast speed
        # tier, no augmentation flag. One row per (fold, model) saying which
        # recipe that fold's search returned and whether the search scored
        # that model by name.
        'recipe_selection': convert_numpy({
            'scope': 'fold_train_rows',
            'speed': 'fast',
            'aug_condition': 'none',
            'augmentation_flag': 'off',
            'n_folds': len(folds),
            'seed': _FOLD_SEARCH_SEED,
            'budgets': list(FULL_BUDGETS),
            'models_searched': search_models,
            'search_metric': search_metric,
            'per_fold': recipe_rows,
            'errors': fold_search_errors,
            'distance_search': {
                'tier': 'general_distance',
                'budgets': list(DISTANCE_BUDGETS),
                'internal_evaluators': list(DISTANCE_EVALUATORS),
                'models_searched': list(DISTANCE_MODELS),
                'errors': fold_dist_search_errors,
            },
        }),
        # The owner's set (2026-10-02): per (fold, model), the top 2 hybrids
        # of the fold's general search and the top 1 of its distance search.
        # Each was built and scored as its own arm in stratified_by_model
        # ('_hybrid', '_hybrid__general2', '_hybrid__distance1').
        'rule': ('top 2 general + top 1 distance hybrids per model, each '
                 'fold read from its own searches'),
        'rule_recipes': convert_numpy(rule_rows),
        'rule_missing': list(rule_missing),
        'top3_base_reps': [(nm, sc) for nm, sc in top3_base_reps],
        'top3_base_reps_by_fold': convert_numpy(top3_by_fold),
        # Added 2026-09-18, beside the keys above and overwriting none of them.
        # Each is keyed by fold number first, then by downstream model. They are
        # NOT inside 'stratified_by_model': firstpass_notready.py:1592-1593
        # loops that dict as `for model, mv in bym.items()` and reads
        # mv['_comparison'], so a fold number landing in it would be read as a
        # downstream model. Nothing loops this dict's top level -- the readers
        # are firstpass_notready.py:1589-1592 (.get('error'),
        # .get('stratified_by_model')) and run() below (.get('top3_base_reps')),
        # all key lookups.
        'top2_base_reps_by_fold': convert_numpy(top2_base_reps_by_fold),
        'top2_hybrids_by_fold': convert_numpy(top2_hybrids_by_fold),
        'top2_pairs_by_model_by_fold': convert_numpy(top2_pairs_by_model_by_fold),
        # The downstream models the four top-2 blocks were asked for, and
        # {downstream model: why it is missing from one of them, on which fold}.
        # Added 2026-09-28: these reasons were printed and saved nowhere.
        'top2_models_compared': top2_models_compared,
        'top2_models_dropped': top2_models_dropped,
        # delta_vs_source_mean is the rank-1 hybrid against the mean of the base
        # representations that actually fed it, read off the per-fold budget
        # split so a representation given 0 features is out of the mean.
        # delta_vs_best_rep is the same hybrid against the single best base
        # representation for that downstream model. Positive = hybrid better,
        # on either metric direction.
        'hybrid_deltas_by_model_by_fold': convert_numpy(
            hybrid_deltas_by_model_by_fold),
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

    2026-09-12, decided: the hybrid's recipe is searched inside each fold, on
    the fast speed tier, with no augmentation flag. n01 reads no whole-dataset
    search any more.
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
    # 2026-09-12, decided: n01 searches its recipe inside each fold, on the
    # fast speed tier, with no augmentation flag — donor features are not in
    # n01's pool. So run() no longer reads a whole-dataset recipe before the
    # fold loop. The two calls that did — get_fitted_pool_hybrid for rf and
    # get_best_search_for_downstream for the other four downstream models —
    # are gone, along with the whole-dataset hybrid they fitted, which was
    # only printed. Every recipe now comes from inside _run_cliff_stratified,
    # one search per fold on that fold's training rows.
    #
    # data['augmentations'] is deliberately not read here. The --augment flag
    # still loads donor representations, and n01 now ignores them: they are
    # neither searched over nor rebuilt into the hybrid.
    results = {
        'config': {
            'sim_threshold': sim_threshold,
            'similar_percentile': SIMILAR_PERCENTILE,
            'activity_threshold': activity_threshold,
            'domain': domain,
            'n_samples': n,
            'n_reps': len(reps),
            # 2026-09-12 decision, recorded on every result file.
            'recipe_search': 'per_fold',
            'search_speed': 'fast',
            'augmentation_flag': 'off',
        },
    }

    # --- Cliff-stratified prediction error ---
    # Trains models per model×rep. The cliff mask inside _run_cliff_stratified
    # uses the module SIMILAR_PERCENTILE for the rep-independent reference.
    results['stratified'] = _run_cliff_stratified(
        reps, labels, groups, config, data,
        sim_threshold, activity_threshold,
        smiles=smiles,
        is_clf=is_clf,
        dataset_name=dataset_name,
        # 2026-09-20: `noise_helpers.run_all` puts the caller's model list in
        # config['models'] and passes no `models` kwarg. Without this the
        # configured list was dropped and every run scored all five models.
        models_to_run=(kwargs.get('models') or config.get('models') or None),
        output_dir=output_dir)

    # Top-3 base representations now come from the per-fold searches (mean of
    # each representation's rf baseline over the folds that scored it), not
    # from a whole-dataset search.
    _strat = results['stratified']
    top3 = (_strat.get('top3_base_reps') or []) if isinstance(_strat, dict) else []
    results['top3_base_reps'] = [(nm, sc) for nm, sc in top3]
    if top3:
        print(f"  Top-3 base reps (rf baseline, mean over the per-fold "
              f"searches): "
              f"{', '.join(f'{nm}={sc:.4f}' for nm, sc in top3)}")

    # --- Grand summary ---
    print("\n" + "=" * 70)
    print("  GRAND SUMMARY")
    print("=" * 70)

    # Cliff-stratified prediction error
    if 'stratified' in results:
        st = results['stratified']
        if isinstance(st, dict) and 'stratified' in st and '_comparison' in st['stratified']:
            comp = st['stratified']['_comparison']
            print(f"  Cliff-stratified gap (top-3 smallest):")
            for rn, rg in comp.get('top3_smallest_gap', []):
                print(f"     {rn}: {rg:.4f}")
            print(f"     Hybrid: {comp['hybrid_gap']:.4f}")

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
