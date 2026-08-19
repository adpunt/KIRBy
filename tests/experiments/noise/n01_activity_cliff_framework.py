#!/usr/bin/env python3
"""N01: Activity Cliff Framework -- cliff-stratified prediction error.

Paper: Noise / Robustness
Scope: MOLECULAR datasets.
Conclusion: Hybrid representations produce smaller cliff vs non-cliff
    prediction gaps than individual representations.

Single sub-analysis: Cliff-stratified prediction error (RMSE/R2 on cliff vs
non-cliff subsets). Cliffs are identified from a REPRESENTATION-INDEPENDENT
reference (ECFP4 Tanimoto for molecular; edit distance for antibody; Euclidean
on concatenated reps as a fallback) and error is measured with held-out CV, so
the comparison is not circular: no representation is scored against cliffs
derived from its own embedding.

The three earlier sub-analyses (consensus cliff identification, SALI landscape
smoothing, cliff mitigation by hybridization) were REMOVED as scientifically
circular -- they derived cliffs from each representation's own embedding, so a
supervised hybrid wins by construction.

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
from sklearn.preprocessing import StandardScaler


from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, make_structured_output, convert_numpy,
    get_model_factories, get_domain_metric, cv_split, make_cv_splitter,
    get_fitted_hybrid, get_fitted_pool_hybrid, flatten_fold_weights, build_fold_hybrid,
    get_best_for_downstream, get_best_search_for_downstream,
    extract_top3_base_reps,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
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

def _run_cliff_stratified(reps, labels, groups, config, data,
                          sim_threshold, activity_k,
                          hybrid_blueprint, smiles=None,
                          augmentations=None, augmentation_strategy='uncapped',
                          is_clf=False, bp_by_model=None, aug_by_model=None):
    """Compare prediction error on cliff vs non-cliff compounds.

    bp_by_model / aug_by_model (optional): pre-fetched per-model blueprints
    and aug_conditions for the matched-evaluator expansion. When provided,
    the evaluation runs across DOWNSTREAM_MODELS_NOISE; otherwise falls back
    to RF-only with the legacy hybrid_blueprint (the audit's P1 was that
    only RF was tested even though the noise suite convention is multi-model).
    """
    from experiments._shared import DOWNSTREAM_MODELS_NOISE

    print("\n  --- Sub-analysis 4: Cliff-Stratified Prediction Error ---")

    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # Multi-model factory dict (matched evaluator). Falls back to RF only if
    # the caller didn't pre-fetch per-model blueprints.
    if bp_by_model is None:
        models_to_run = ['rf']
        bp_by_model = {'rf': hybrid_blueprint}
        aug_by_model = {'rf': augmentation_strategy}
    else:
        models_to_run = list(bp_by_model.keys())

    model_factories = get_model_factories(
        models_to_run, task_type=data['task_type'],
        n_samples=len(labels))
    if not model_factories:
        return {'error': 'no_model_factories'}

    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    # Identify cliff compounds using a REPRESENTATION-INDEPENDENT reference.
    # For molecular: ECFP4 Tanimoto (Bajorath standard).
    # For antibody: edit distance on heavy-chain sequences.
    # Fallback: Euclidean on concatenated reps.
    n = len(labels)
    heavy_seqs = data.get('heavy_seqs')

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
            X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
            ref_sim = _compute_euclidean_sim(X_concat)
            sim_source = 'euclidean_concat'
            cliff_pair_mask, cliff_info = identify_cliff_pairs(
                ref_sim, labels, sim_threshold, activity_k, is_clf=is_clf,
                similar_percentile=SIMILAR_PERCENTILE)
            cliff_mask = cliff_pair_mask.any(axis=1)
            cliff_indices = set(np.where(cliff_mask)[0].tolist())
    else:
        X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
        ref_sim = _compute_euclidean_sim(X_concat)
        sim_source = 'euclidean_concat'
        cliff_pair_mask, cliff_info = identify_cliff_pairs(
            ref_sim, labels, sim_threshold, activity_k, is_clf=is_clf,
            similar_percentile=SIMILAR_PERCENTILE)
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

    def _evaluate_condition(get_Xy_fn, label, model_factory):
        """Run CV, return {metric_all, metric_cliff, metric_noncliff, gap, ...}."""
        fold_all, fold_cliff, fold_noncliff = [], [], []

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

            model = model_factory()
            model.fit(X_tr_s, y_train)
            if metric_name == 'auc' and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_va_s)
                if getattr(proba, 'ndim', 1) == 2 and proba.shape[1] == 2:
                    y_pred = proba[:, 1]
                else:
                    y_pred = proba
            else:
                y_pred = model.predict(X_va_s)

            fold_all.append(metric_fn(y_val, y_pred))

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
    strat_results_by_model = {}
    for model_name, model_factory in model_factories.items():
        model_bp = bp_by_model.get(model_name)
        model_aug = aug_by_model.get(model_name,
                                     augmentation_strategy) if aug_by_model else augmentation_strategy
        model_use_augs = augmentations if model_aug != 'none' else None
        per_model = {}

        print(f"\n    [{model_name}] Per-rep baselines:")
        for rep_name, X_rep in reps.items():
            def _get_rep(tr, va, yt, _X=X_rep):
                return _X[tr], _X[va]
            res = _evaluate_condition(_get_rep, rep_name, model_factory)
            if res:
                per_model[rep_name] = res
                mn = res['metric']
                print(f"      {rep_name:<20} {mn}_all={res[f'{mn}_all']:.4f}  "
                      f"{mn}_cliff={res[f'{mn}_cliff']:.4f}  "
                      f"{mn}_noncliff={res[f'{mn}_noncliff']:.4f}  "
                      f"gap={res['cliff_noncliff_gap']:.4f}")

        if model_bp is not None:
            print(f"    [{model_name}] Hybrid:")

            def _get_hybrid(tr, va, yt, _bp=model_bp, _augs=model_use_augs):
                train_reps = {name: rep[tr] for name, rep in reps.items()}
                val_reps = {name: rep[va] for name, rep in reps.items()}
                train_augs = {k: v[tr] for k, v in _augs.items()} if _augs else None
                val_augs = {k: v[va] for k, v in _augs.items()} if _augs else None
                X_tr_h, X_va_h, _ = build_fold_hybrid(
                    train_reps, val_reps, yt, _bp,
                    augmentations_train=train_augs, augmentations_val=val_augs)
                return X_tr_h, X_va_h

            res = _evaluate_condition(_get_hybrid, 'hybrid', model_factory)
            if res:
                per_model['_hybrid'] = res
                mn = res['metric']
                print(f"      {'_hybrid':<20} {mn}_all={res[f'{mn}_all']:.4f}  "
                      f"{mn}_cliff={res[f'{mn}_cliff']:.4f}  "
                      f"{mn}_noncliff={res[f'{mn}_noncliff']:.4f}  "
                      f"gap={res['cliff_noncliff_gap']:.4f}")

            rep_gap_list = [(k, v['cliff_noncliff_gap'])
                            for k, v in per_model.items()
                            if k != '_hybrid'
                            and not np.isnan(v.get('cliff_noncliff_gap', float('nan')))]
            if rep_gap_list and '_hybrid' in per_model:
                top3_gap = sorted(rep_gap_list, key=lambda x: abs(x[1]))[:3]
                hybrid_gap = per_model['_hybrid']['cliff_noncliff_gap']
                per_model['_comparison'] = {
                    'top3_smallest_gap': top3_gap,
                    'hybrid_gap': hybrid_gap,
                }

        strat_results_by_model[model_name] = per_model

    # Legacy flat structure: RF view (or first available) for backwards
    # compatibility. Downstream readers preferring multi-model should use
    # strat_results_by_model.
    _legacy_model = 'rf' if 'rf' in strat_results_by_model else (
        next(iter(strat_results_by_model)) if strat_results_by_model else None)
    strat_results = (dict(strat_results_by_model[_legacy_model])
                     if _legacy_model else {})

    return {
        'cliff_source': sim_source,
        'cliff_info': convert_numpy(cliff_info),
        'stratified': convert_numpy(strat_results),
        'stratified_by_model': convert_numpy(strat_results_by_model),
        'models': list(strat_results_by_model.keys()),
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
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N01: CLIFF-STRATIFIED PREDICTION ERROR")
    print("=" * 70)

    sim_threshold = kwargs.get('sim_threshold', 0.55)
    activity_threshold = kwargs.get('activity_threshold', 2.0)
    search_speed = config.get('search_speed', 'full')

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

    # --- Build the RF hybrid (anchor for the cliff-stratified analysis) ---
    augmentations = data.get('augmentations')
    hybrid_blueprint = None
    X_hybrid = None
    meta = None
    search = None
    top3 = None

    try:
        # Matched-evaluator rule: the cliff-stratified analysis scores the
        # hybrid with a model and must use that model's blueprint. RF is built
        # first because it is the canonical top-3 anchor and the RF blueprint
        # feeds the per-model prefetch below.
        X_hybrid, fi_hybrid, search, meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=augmentations,
            speed=search_speed, data=data,
            dataset_name=dataset_name, task_type=task_type)
        if meta:
            hybrid_blueprint = meta['best_config']['blueprint']
        print(f"  Hybrid: dim={X_hybrid.shape[1]}")
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (ranked by RF baseline): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        print(f"  WARNING: Hybrid search failed ({e}), hybrid comparisons skipped")
        # All four (X_hybrid, hybrid_blueprint, meta, search, top3) stay
        # None — they were initialized to None above the try.

    # Pre-fetch per-model blueprints for the cliff-stratified analysis. Each
    # call is cached so re-fetching is cheap when the pool already has the
    # entry. RF is already loaded above; just reuse its bp/aug_condition.
    from experiments._shared import DOWNSTREAM_MODELS_NOISE
    bp_by_model_s4 = {}
    aug_by_model_s4 = {}
    if hybrid_blueprint is not None:
        bp_by_model_s4['rf'] = hybrid_blueprint
        aug_by_model_s4['rf'] = (meta.get('aug_condition', 'uncapped')
                                 if meta else 'uncapped')
    for _mname in DOWNSTREAM_MODELS_NOISE:
        if _mname == 'rf':
            continue
        try:
            _search, _meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=_mname, budget=None,
                augmentations=augmentations,
                speed=search_speed,
                dataset_name=dataset_name, task_type=task_type, data=data,
            )
            if _meta is not None:
                _bp = _meta['best_config']['blueprint']
                _aug = _meta.get('aug_condition', 'uncapped')
                if getattr(_bp, 'augmentation_strategy', None) != _aug:
                    try:
                        _bp.augmentation_strategy = _aug
                    except Exception:
                        pass
                bp_by_model_s4[_mname] = _bp
                aug_by_model_s4[_mname] = _aug
        except Exception:
            # If a model's blueprint can't be fetched, skip it.
            pass

    results = {
        'config': {
            'sim_threshold': sim_threshold,
            'similar_percentile': SIMILAR_PERCENTILE,
            'activity_threshold': activity_threshold,
            'domain': domain,
            'n_samples': n,
            'n_reps': len(reps),
        },
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
    }

    # --- Cliff-stratified prediction error ---
    # Trains models per model×rep. The cliff mask inside _run_cliff_stratified
    # uses the module SIMILAR_PERCENTILE for the rep-independent reference.
    # Use the augmentation condition the pool actually selected, not a
    # hardcoded 'uncapped' — the latter is the 2026-04-21 cache-collision
    # pattern (see project_cache_collision_fix). meta['aug_condition'] holds
    # the slot the winning blueprint came from.
    aug_strategy_from_pool = (meta.get('aug_condition', 'uncapped')
                              if meta else 'uncapped')
    augs_for_cliff = augmentations if aug_strategy_from_pool != 'none' else None
    results['stratified'] = _run_cliff_stratified(
        reps, labels, groups, config, data,
        sim_threshold, activity_threshold,
        hybrid_blueprint, smiles,
        augmentations=augmentations,
        augmentation_strategy=aug_strategy_from_pool,
        is_clf=is_clf,
        bp_by_model=bp_by_model_s4,
        aug_by_model=aug_by_model_s4)

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
