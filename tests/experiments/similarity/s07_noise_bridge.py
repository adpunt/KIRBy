#!/usr/bin/env python3
"""S07: Noise Detection Bridge (Supplementary) -- cliff/noise FP analysis.

Paper: Similarity
Conclusion: Hybrid distances reduce false-positive cliff-as-noise rate.
Dependencies: None.

Uses REPRESENTATION-INDEPENDENT ground-truth cliffs:
  - Antibody data: edit distance (Hamming) on heavy-chain sequences
  - Molecular data: ECFP4 Tanimoto (Bajorath standard)
Then measures kNN noise detection FP rate (ground-truth cliffs flagged as noise)
in hybrid space vs top-3 individual rep baselines.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_fitted_hybrid, get_fitted_pool_hybrid, flatten_fold_weights, extract_top3_base_reps,
    get_domain_metric,
    honest_rescore,
    cv_split,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
    hr_exclude_folds,
)

from kirby.utils.noise_mitigation_methods import (
    compute_distance_matrix, get_k_neighbors_from_distances,
    ActivityCliffDetector, KNNConsistencyFilter, NeighborhoodConsensusFilter,
)
from kirby.utils.noise_estimation import (
    ActivityCliffAwareEstimator, ConformalEstimator,
)
from experiments.similarity.similarity_helpers import (
    identify_cliff_compounds, identify_cliff_compounds_editdist,
)


def _knn_noise_fp_rate(X, labels, ground_truth_cliffs, k=5, standardize=True, **kwargs):
    """Compute kNN noise detection cliff-contamination rate against ground-truth cliffs.

    Returns:
        dict with n_noise_flagged, n_fp_cliff_in_noise, cliff_contamination_rate.
        Note: cliff_contamination_rate = |cliffs flagged as noise| / |noise flagged|,
        i.e. the precision of cliff-as-noise (NOT a classical FPR which would divide
        by |ground-truth cliffs|).
    """
    k = kwargs.get('k', k)
    standardize = kwargs.get('standardize', standardize)
    D = compute_distance_matrix(X, standardize=standardize)
    knn_filter = KNNConsistencyFilter(k=k, precomputed_distances=D)
    noise_flagged = set(knn_filter.identify_noise(X, labels).tolist())
    fp = noise_flagged & ground_truth_cliffs
    # When NOTHING is flagged as noise the rate is 0/0 = UNDEFINED, not 0.0.
    # Returning 0.0 here was a divide-by-empty artifact that handed a perfect
    # (lowest-possible) score to a detector that never fires, letting a
    # do-nothing representation win the "best rep" baseline slot. Report NaN so
    # such reps are treated as undefined rather than best-in-class.
    cliff_contamination_rate = len(fp) / len(noise_flagged) if noise_flagged else float('nan')
    return {
        'n_noise_flagged': len(noise_flagged),
        'n_fp_cliff_in_noise': len(fp),
        'cliff_contamination_rate': cliff_contamination_rate,
    }


# ── CV-honest redesign of the cliff-contamination statistic (decision B) ──────
# The whole-data `_knn_noise_fp_rate` above is TRANSDUCTIVE: the 90th-percentile
# noise threshold and every point's neighbours are read off the SAME matrix the
# score is reported on, so a rep can win the "best rep" slot just by fitting the
# whole dataset's geometry — the same optimism the honest-comparison fix removes
# from the model arms. `_knn_noise_contamination_cv` recomputes the SAME
# statistic under cross-validation on group-aware folds supplied by the caller:
# the scaler stats AND the noise threshold are FIT on the TRAIN
# folds only, then contamination is MEASURED on the HELD-OUT fold, where each
# held-out point's k neighbours are drawn from TRAIN points ONLY (never from
# other held-out points). One held-out rate per fold is returned; the shared
# `honest_rescore` helper then applies the identical leave-one-fold-out winner
# re-selection to both arms. This is an UNSUPERVISED geometry statistic (not a
# downstream-model CV score), so it needs a LOCAL fold loop — `find_best_hybrid`
# / `build_honest_rescore_input` do not produce these numbers.
def _leave_one_group_out_splits(groups):
    """The fold scheme s07 has used since it was written: ONE fold per distinct
    value of `groups`. On a scaffold-grouped dataset that is one fold per
    scaffold — the fold count is the number of scaffolds, not the configured
    `config['n_splits']`, and a held-out fold holding one or two molecules can
    only produce a contamination rate of 0.0, 1.0 or NaN.

    Yields (train_idx, test_idx) index arrays.
    """
    groups = np.asarray(groups)
    for g in np.unique(groups):
        test_mask = groups == g
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


def _configured_fold_splits(groups, labels, n_splits, task_type=None, seed=42):
    """The suite's shared group-aware splitter (`_shared.cv_split`) at the
    configured fold count — the same folds every other experiment in the suite
    scores on. n_splits=1 gives one grouped 80/20 split; n_splits>1 gives
    StratifiedGroupKFold for classification and GroupKFold otherwise.

    Returns a list of (train_idx, test_idx). Propagates cv_split's ValueError
    when `groups` has fewer than max(2, n_splits) distinct values.
    """
    n = len(np.asarray(labels))
    X_idx = np.zeros((n, 1))          # cv_split uses X for length only
    return list(cv_split(groups, n_splits, X_idx, labels, seed=seed,
                         task_type=task_type))


def _knn_noise_contamination_cv(X, labels, ground_truth_cliffs, splits,
                                k=5, standardize=True, matrix_for_fold=None):
    """Per-fold, held-out kNN cliff-contamination rate (fit on TRAIN, measure on
    the held-out fold). Returns a list with one rate per fold; a fold that flags
    no held-out point as noise contributes NaN (0/0 is undefined — same
    convention as the whole-data version), which the honest re-scorer skips.

    `splits` is an iterable of (train_idx, test_idx) index arrays, so the fold
    definition is the CALLER's choice: `_leave_one_group_out_splits` reproduces
    the original one-fold-per-group scheme, `_configured_fold_splits` uses the
    shared splitter at `config['n_splits']`.

    matrix_for_fold: called as f(train_idx, test_idx) -> (X, feature_weights or
    None) to supply a matrix whose FEATURES were chosen without this fold's
    labels. The hybrid arms pass it; a single representation needs no selection,
    so it passes `X` directly and leaves this None. A fold whose call returns
    (None, None) contributes NaN.

    Neighbour label-consistency and the 90th-percentile threshold mirror
    `KNNConsistencyFilter.identify_noise`, but the held-out points take their k
    neighbours exclusively from the train split so nothing leaks across the
    fold boundary.
    """
    from scipy.spatial.distance import cdist
    if X is not None:
        X = np.asarray(X)
    labels = np.asarray(labels)
    fold_rates = []
    for train_idx, test_idx in splits:
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        n_train = int(train_idx.size)
        n_test = int(test_idx.size)
        kk = min(k, n_train - 1)
        # Need >= k+1 train points to define k neighbours + a train threshold;
        # degenerate folds contribute NaN (skipped by honest_rescore).
        if kk < 1 or n_test == 0:
            fold_rates.append(float('nan'))
            continue
        weights = None
        if matrix_for_fold is not None:
            X_fold, weights = matrix_for_fold(train_idx, test_idx)
            if X_fold is None:
                fold_rates.append(float('nan'))
                continue
            X_fold = np.asarray(X_fold)
        else:
            X_fold = X
        X_train = X_fold[train_idx]
        X_test = X_fold[test_idx]
        if standardize:
            scaler = StandardScaler().fit(X_train)   # TRAIN stats only
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        if weights is not None:
            # Scale-then-weight, the convention every other similarity
            # experiment uses. Weights come from this fold's own training rows.
            w = np.sqrt(np.maximum(np.nan_to_num(weights, nan=0.0), 0))
            if w.shape[0] == X_train.shape[1]:
                X_train = X_train * w
                X_test = X_test * w
        y_train = labels[train_idx]
        # Train-side threshold: each train point's k nearest OTHER train
        # neighbours' label std, 90th percentile (== the filter's auto-threshold
        # but fit strictly on TRAIN).
        D_tr = cdist(X_train, X_train)
        np.fill_diagonal(D_tr, np.inf)
        tr_nbr = np.argsort(D_tr, axis=1)[:, :kk]
        tr_std = np.std(y_train[tr_nbr], axis=1)
        threshold = np.percentile(tr_std, 90)
        # Held-out points: neighbours from TRAIN only.
        D_te = cdist(X_test, X_train)
        te_nbr = np.argsort(D_te, axis=1)[:, :kk]
        te_std = np.std(y_train[te_nbr], axis=1)
        flagged_global = set(test_idx[te_std > threshold].tolist())
        fp = flagged_global & ground_truth_cliffs
        rate = len(fp) / len(flagged_global) if flagged_global else float('nan')
        fold_rates.append(rate)
    return fold_rates


# The cliff-contamination statistic is a single fixed-geometry kNN filter, not a
# sweep over downstream models, so the honest_rescore input is keyed by one
# model name ('knn') by design — the model set is NOT assumed from a factory.
_S07_MODEL_KEY = 'knn'


# The seed that names each outer fold's search cache. It must match the seed the
# outer folds are built with, or one fold would read another fold's search.
_FOLD_SEARCH_SEED = 42


def _outer_fold_hybrids(reps, labels, groups, augmentations, n_splits,
                        task_type, metric_name, domain, dataset_name):
    """One hybrid per outer fold, selected without that fold's held-out rows.

    For each outer fold the recipe is searched on the fold's TRAINING rows
    (get_fold_best_search) and the features are then chosen from the same rows
    (build_fold_hybrid). The selection is applied to every row, so the caller can
    read out any row subset; what matters is that fold i's matrix never used the
    labels of the rows fold i holds out.

    Two contamination fold schemes read these. The k-fold scheme's folds ARE the
    outer folds. The one-fold-per-group scheme holds out one group at a time, and
    a group lives entirely inside one outer test fold, so that outer fold's
    matrix is the one whose selection never saw the group.

    Returns:
        (matrices, weights, row_outer_fold, failures)
        matrices[i]        full-row matrix under outer fold i's selection
        weights[i]         per-feature importance from fold i's training rows
        row_outer_fold[r]  the outer fold that holds row r out
        failures           [{fold, error_type, error}] for folds that threw
    """
    from experiments._shared import get_fold_best_search, build_fold_hybrid
    from kirby.find_best_hybrid import nested_cv_outer_folds

    labels = np.asarray(labels)
    outer = nested_cv_outer_folds(reps, labels, groups, n_splits,
                                  _FOLD_SEARCH_SEED, task_type)
    row_outer_fold = np.full(len(labels), -1, dtype=int)
    matrices, weights, failures = {}, {}, []

    for fold_i, (tr, te) in enumerate(outer):
        row_outer_fold[te] = fold_i
        try:
            _search, meta = get_fold_best_search(
                reps, labels, groups,
                train_idx=tr, fold_i=fold_i, n_outer_folds=len(outer),
                downstream_model='knn', budget=None,
                augmentations=augmentations,
                metric=metric_name, task_type=task_type, domain=domain,
                dataset_name=dataset_name, distance=True,
                seed=_FOLD_SEARCH_SEED)
            if meta is None or 'blueprint' not in meta['best_config']:
                raise RuntimeError('no kNN config with a blueprint')
            bp = meta['best_config']['blueprint']
            bp.augmentation_strategy = meta['aug_condition']
            augs = (augmentations
                    if augmentations and meta['aug_condition'] != 'none' else None)
            # val_reps is every row, so the returned matrix carries this fold's
            # selection applied across the dataset.
            _, X_all, feat_info = build_fold_hybrid(
                {k: X[tr] for k, X in reps.items()},
                dict(reps),
                labels[tr], bp,
                augmentations_train=({k: v[tr] for k, v in augs.items()}
                                     if augs else None),
                augmentations_val=(dict(augs) if augs else None),
                groups=(np.asarray(groups)[tr] if groups is not None else None))
            matrices[fold_i] = X_all
            try:
                w = flatten_fold_weights(feat_info)
                weights[fold_i] = w if w.shape[0] == X_all.shape[1] else None
            except Exception:
                weights[fold_i] = None
        except Exception as e:
            print(f"  [warn] outer fold {fold_i} hybrid failed: "
                  f"{type(e).__name__}: {str(e)[:100]}")
            failures.append({'fold': fold_i, 'error_type': type(e).__name__,
                             'error': str(e)})
    return matrices, weights, row_outer_fold, failures


def _make_hybrid_arms(matrices, weights, row_outer_fold):
    """{arm name: matrix_for_fold} for _contamination_blocks.

    A contamination fold's held-out rows all sit in one outer fold, so its
    matrix is that outer fold's. A fold whose rows straddle outer folds, or
    whose outer fold failed to build, returns (None, None) and scores NaN.
    """
    def _pick(test_idx):
        f = np.unique(row_outer_fold[np.asarray(test_idx)])
        if len(f) != 1 or int(f[0]) not in matrices:
            return None
        return int(f[0])

    def _unweighted(train_idx, test_idx):
        f = _pick(test_idx)
        return (None, None) if f is None else (matrices[f], None)

    def _weighted(train_idx, test_idx):
        f = _pick(test_idx)
        if f is None or weights.get(f) is None:
            return None, None
        return matrices[f], weights[f]

    arms = {'hybrid_unweighted': _unweighted}
    if any(w is not None for w in weights.values()):
        arms['hybrid_weighted'] = _weighted
    return arms


def _contamination_blocks(reps, hybrid_arms, labels, ground_truth_cliffs,
                          groups, n_splits, task_type):
    """Build the SAME honest_rescore contract under TWO fold definitions.

    Returns (block_per_group, block_kfold):
      - block_per_group: one fold per distinct group value
        (`_leave_one_group_out_splits`) — the scheme the stored
        `honest_rescore` key has always carried.
      - block_kfold: the shared group-aware splitter at `n_splits`
        (`_configured_fold_splits`).
    Each block records `fold_scheme` and the fold count actually used. If the
    shared splitter rejects the dataset's groups, block_kfold carries the
    error text instead of scores.

    hybrid_arms: {candidate_name: matrix_for_fold}, where matrix_for_fold is
    called as f(train_idx, test_idx) -> (X, feature_weights or None). Each call
    returns a matrix whose features were selected without the held-out rows'
    labels — see _outer_fold_hybrids. It used to be {name: one matrix built on
    every row}, which handed the hybrid every held-out label before scoring.
    """
    def _block(splits, extra):
        splits = list(splits)
        baselines = {_S07_MODEL_KEY: {}}
        for rep_name, X_rep in reps.items():
            baselines[_S07_MODEL_KEY][rep_name] = _knn_noise_contamination_cv(
                X_rep, labels, ground_truth_cliffs, splits)
        cands = []
        for cand_name, matrix_for_fold in hybrid_arms.items():
            cands.append({
                'name': cand_name,
                'fold_scores': _knn_noise_contamination_cv(
                    None, labels, ground_truth_cliffs, splits,
                    matrix_for_fold=matrix_for_fold),
            })
        block = {
            'lower_is_better': True,
            # Each fold's hybrid was recipe-searched and feature-selected on its
            # own outer-fold training rows, so nothing here is a whole-dataset
            # pick.
            'recipe_selection': 'fold_train_rows',
            # Every fold counts. The candidates here are the arms this file built itself,
            # not a search's recipe list, so no fold filtered them.
            'exclude_folds': [],
            'candidates_by_model': {_S07_MODEL_KEY: cands} if cands else {},
            'baselines_by_model': baselines,
        }
        block.update(extra)
        return block

    per_group_splits = list(_leave_one_group_out_splits(groups))
    block_per_group = _block(per_group_splits, {
        'fold_scheme': 'one_fold_per_group',
        'n_folds': len(per_group_splits),
        'fold_scheme_note': ('one fold per distinct group value; the fold count '
                             'is the number of groups, not config["n_splits"]'),
    })

    try:
        kfold_splits = _configured_fold_splits(groups, labels, n_splits, task_type)
    except Exception as e:
        return block_per_group, {
            'fold_scheme': 'shared_group_splitter',
            'n_folds_requested': int(n_splits),
            'error': f"{type(e).__name__}: {e}",
        }
    block_kfold = _block(kfold_splits, {
        'fold_scheme': 'shared_group_splitter',
        'n_folds_requested': int(n_splits),
        'n_folds': len(kfold_splits),
        'fold_scheme_note': ('experiments._shared.cv_split at config["n_splits"] '
                             '— the folds the rest of the suite scores on'),
    })
    return block_per_group, block_kfold


def run(data, config, output_dir, dataset_name, **kwargs):
    """Noise detectors flag activity cliffs as noise (FP).
    Hybrid + per-rep baselines compared against ground-truth cliffs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S07: NOISE DETECTION BRIDGE (Supplementary)")
    print("=" * 70)


    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    # Fold count for the shared group-aware splitter (1 in intermediary mode).
    n_splits = int((config or {}).get('n_splits', 5) or 1)

    results = {}

    # ── Step 1: Identify ground-truth cliffs (representation-independent) ──
    ground_truth_cliffs = set()
    cliff_info = {}

    # Antibody: edit distance on heavy-chain sequences
    if 'heavy_seqs' in data and data['heavy_seqs'] is not None:
        heavy_seqs = data['heavy_seqs']
        max_edits = kwargs.get('max_edits', 5)
        # Distribution-relative structural threshold. Absolute max_edits=5 was
        # degenerate: ~0% cliffs on distinct-antibody datasets (flab) and ~100%
        # on mutational-scan datasets (abagym). See
        # COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
        # The percentile is now auto-selected in-run from the prevalence sweep
        # (closest to a 10% cliff-compound target) — no rerun/retune needed; the
        # full sweep is recorded in cliff_info['percentile_sweep'].
        similar_pct = kwargs.get('similar_percentile', [0.5, 1.0, 2.0, 5.0, 10.0])
        cliff_mask, cliff_info = identify_cliff_compounds_editdist(
            heavy_seqs, labels, max_edits=max_edits, is_clf=is_clf,
            similar_percentile=similar_pct)
        ground_truth_cliffs = set(np.where(cliff_mask)[0].tolist())
        _sel_pct = cliff_info.get('selected_percentile', similar_pct)
        print(f"\n  Ground-truth cliffs (edit distance, selected_pct={_sel_pct}, "
              f"cutoff={cliff_info['edit_cutoff']:.0f} edits):")
        print(f"    {cliff_info['n_cliff_compounds']} compounds "
              f"({cliff_info['pct_cliff_compounds']:.1f}%), "
              f"{cliff_info['n_cliff_pairs']} pairs")
        print(f"    Median pairwise edit dist: {cliff_info['median_edit_dist']:.0f}")
        results['cliff_method'] = 'edit_distance'
        results['cliff_info'] = {k: v for k, v in cliff_info.items()
                                 if k != 'cliff_pair_counts'}

    # Molecular: ECFP4 Tanimoto
    elif 'ecfp4' in reps:
        cliff_mask, cliff_info = identify_cliff_compounds(
            reps['ecfp4'], labels, is_clf=is_clf)
        ground_truth_cliffs = set(np.where(cliff_mask)[0].tolist())
        print(f"\n  Ground-truth cliffs (ECFP4 Tanimoto >= {cliff_info['tanimoto_threshold']}):")
        print(f"    {cliff_info['n_cliff_compounds']} compounds "
              f"({cliff_info['pct_cliff_compounds']:.1f}%), "
              f"{cliff_info['n_cliff_pairs']} pairs")
        results['cliff_method'] = 'ecfp4_tanimoto'
        results['cliff_info'] = {k: v for k, v in cliff_info.items()
                                 if k != 'cliff_pair_counts'}
    else:
        print("\n  WARNING: No ground-truth cliff source available "
              "(no heavy_seqs or ecfp4). Using representation-dependent fallback.")
        results['cliff_method'] = 'none'

    if not ground_truth_cliffs:
        print("  No ground-truth cliffs found. Skipping FP analysis.")
        results.setdefault('summary', {
            'n_ground_truth_cliffs': 0,
            'best_rep': None,
            'best_rep_cliff_contamination_rate': float('nan'),
            'top3_reps': [],
            'delta_vs_best_rep': float('nan'),
        })
        results.setdefault('per_rep_baselines', {})
        results.setdefault('hybrid_unweighted', {})
        results.setdefault('hybrid_weighted', {'skipped': 'no_ground_truth_cliffs'})
        # Empty honest_rescore block: nothing to score without ground-truth
        # cliffs; the reader treats an empty block as "no honest arms".
        results.setdefault('honest_rescore', {
            'lower_is_better': True,
            # Every fold counts. The candidates here are the arms this file built itself,
            # not a search's recipe list, so no fold filtered them.
            'exclude_folds': [],
            'candidates_by_model': {},
            'baselines_by_model': {},
            'fold_scheme': 'one_fold_per_group',
        })
        results.setdefault('honest_rescore_kfold', {
            'lower_is_better': True,
            # Every fold counts. The candidates here are the arms this file built itself,
            # not a search's recipe list, so no fold filtered them.
            'exclude_folds': [],
            'candidates_by_model': {},
            'baselines_by_model': {},
            'fold_scheme': 'shared_group_splitter',
            'n_folds_requested': n_splits,
        })
        return results

    # ── Step 2: Get hybrid ──
    augmentations = data.get('augmentations')
    results['hybrid_search'] = {'status': 'ok'}
    try:
        X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=None,
            augmentations=augmentations,
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, distance=True)
        if fi:
            weights_flat = flatten_fold_weights(fi)
            if len(weights_flat) != X_hybrid.shape[1]:
                weights_flat = None
        else:
            weights_flat = None
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            results['top3_base_reps'] = [(n, s) for n, s in top3]
    except Exception as e:
        # Narrow-ish catch: the hybrid arm is what failed, NOT the whole
        # experiment. Record an EXPLICIT failure flag so s09 aggregation can
        # drop the hybrid arm for this dataset instead of silently reading the
        # per-rep baselines as if the hybrid was evaluated (the bare-except
        # bug — the JSON had no skip marker).
        import traceback
        print(f"  Hybrid search FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        X_hybrid = None
        weights_flat = None
        results['hybrid_search'] = {
            'status': 'failed',
            'error': f"{type(e).__name__}: {e}",
        }
        results['hybrid_unweighted'] = {'failed': True, 'error': f"{type(e).__name__}: {e}"}
        results['hybrid_weighted'] = {'failed': True, 'error': f"{type(e).__name__}: {e}"}

    # ── Step 3: Compute FP rates for hybrid ──
    # X_hybrid below is the WHOLE-DATASET matrix and feeds the whole-dataset
    # (transductive) rates only. The held-out blocks in Step 6 use hybrid_arms,
    # one matrix per outer fold whose features and recipe were chosen without
    # that fold's held-out rows.
    hybrid_embeddings = {}
    if X_hybrid is not None:
        # Hybrid unweighted
        hybrid_uw = _knn_noise_fp_rate(X_hybrid, labels, ground_truth_cliffs)
        results['hybrid_unweighted'] = hybrid_uw
        hybrid_embeddings['hybrid_unweighted'] = X_hybrid
        print(f"\n  Hybrid (unweighted): "
              f"cliff contamination={hybrid_uw['cliff_contamination_rate']:.3f} "
              f"({hybrid_uw['n_fp_cliff_in_noise']}/{hybrid_uw['n_noise_flagged']})")

        # Hybrid weighted — only if importance weights are available.
        # Standardize first, then apply weights; pass standardize=False to avoid
        # re-standardizing after weighting.
        if weights_flat is None:
            results['hybrid_weighted'] = {'skipped': 'no_importance_available'}
            hybrid_w = None
            print("  Hybrid (weighted):   skipped (no importance weights available)")
        else:
            w_clean = np.nan_to_num(weights_flat, nan=0.0)
            n_nan = int(np.isnan(weights_flat).sum())
            n_neg = int((w_clean < 0).sum())
            if n_nan or n_neg:
                print(f"  WARN: weights had {n_nan} NaN and {n_neg} negative entries; "
                      f"clamped to 0.")
            w_clean = np.maximum(w_clean, 0)
            X_hybrid_w = StandardScaler().fit_transform(X_hybrid) * np.sqrt(w_clean)
            hybrid_w = _knn_noise_fp_rate(X_hybrid_w, labels, ground_truth_cliffs,
                                          standardize=False)
            results['hybrid_weighted'] = hybrid_w
            hybrid_embeddings['hybrid_weighted'] = X_hybrid_w
            print(f"  Hybrid (weighted):   "
                  f"cliff contamination={hybrid_w['cliff_contamination_rate']:.3f} "
                  f"({hybrid_w['n_fp_cliff_in_noise']}/{hybrid_w['n_noise_flagged']})")
    else:
        hybrid_uw = None
        hybrid_w = None

    # ── Step 3b: One hybrid per outer fold, for the held-out blocks ──
    # The recipe and the features both come from each outer fold's TRAINING rows.
    # The held-out contamination rate used to be measured on X_hybrid above,
    # whose features were picked with every row's label, including the labels of
    # the rows each fold then scores.
    hybrid_arms = {}
    results['heldout_hybrid_status'] = {'computed': False,
                                        'reason': 'not attempted'}
    try:
        _mats, _wts, _row_fold, _fails = _outer_fold_hybrids(
            reps, labels, groups, augmentations, n_splits, task_type,
            metric_name, domain, dataset_name)
        hybrid_arms = _make_hybrid_arms(_mats, _wts, _row_fold)
        results['heldout_hybrid_status'] = {
            'computed': bool(hybrid_arms),
            'n_outer_folds_built': len(_mats),
            'arms': sorted(hybrid_arms.keys()),
            'failures': _fails,
            'reason': (None if hybrid_arms
                       else 'no outer fold produced a hybrid'),
        }
        print(f"\n  Held-out hybrid arms: {sorted(hybrid_arms.keys()) or 'none'} "
              f"({len(_mats)}/{n_splits} outer folds built)")
    except Exception as e:
        results['heldout_hybrid_status'] = {
            'computed': False,
            'error_type': type(e).__name__,
            'error': str(e),
            'reason': f"per-fold hybrid build raised {type(e).__name__}",
        }
        print(f"\n  Held-out hybrid arms: none ({type(e).__name__}: {e})")

    # ── Step 4: Per-rep baselines — top 3 by lowest cliff contamination rate ──
    print(f"\n  Per-rep baselines (kNN cliff-as-noise contamination vs ground-truth cliffs):")
    per_rep_results = {}
    for rep_name, X_rep in reps.items():
        rep_fp = _knn_noise_fp_rate(X_rep, labels, ground_truth_cliffs)
        per_rep_results[rep_name] = rep_fp

    # Sort by cliff contamination rate (lower is better). Reps that flagged NO
    # noise have an UNDEFINED (NaN) rate and are pushed to the end — a detector
    # that never fires cannot legitimately be the "best" baseline.
    def _rate_key(rep_fp):
        r = rep_fp['cliff_contamination_rate']
        return float('inf') if (r is None or not np.isfinite(r)) else r
    sorted_reps = sorted(per_rep_results.items(), key=lambda x: _rate_key(x[1]))
    for rep_name, rep_fp in sorted_reps[:5]:
        print(f"    {rep_name}: cliff contamination={rep_fp['cliff_contamination_rate']:.3f} "
              f"({rep_fp['n_fp_cliff_in_noise']}/{rep_fp['n_noise_flagged']})")
    if len(sorted_reps) > 5:
        print(f"    ... ({len(sorted_reps) - 5} more)")

    results['per_rep_baselines'] = per_rep_results

    # ── Step 5: Summary ──
    # Only reps with a DEFINED (finite) contamination rate are eligible as the
    # baseline; reps that flagged no noise (NaN) are excluded so a do-nothing
    # detector cannot win the baseline slot.
    defined_reps = [(n, r) for n, r in sorted_reps
                    if np.isfinite(r['cliff_contamination_rate'])]
    print(f"\n  Summary:")
    print(f"    Ground-truth cliffs: {len(ground_truth_cliffs)}")
    if not defined_reps:
        # No rep flagged any noise -> no valid baseline. Record explicitly.
        print("    Best individual rep: none (no rep flagged any noise; "
              "baseline undefined)")
        results['summary'] = {
            'n_ground_truth_cliffs': len(ground_truth_cliffs),
            'best_rep': None,
            'best_rep_cliff_contamination_rate': float('nan'),
            'top3_reps': [],
            'delta_vs_best_rep': float('nan'),
            'delta_vs_best_rep_weighted': float('nan'),
            'hybrid_failed': bool(X_hybrid is None),
            'no_defined_baseline': True,
        }
        # No whole-data baseline flagged any noise. The CV pass may still yield
        # held-out flags on some folds, so persist the honest blocks regardless
        # rather than emitting empty ones here.
        block_pg, block_kf = _contamination_blocks(
            reps, hybrid_arms, labels, ground_truth_cliffs,
            groups, n_splits, task_type)
        results['honest_rescore'] = block_pg
        results['honest_rescore_kfold'] = block_kf
        return results
    top3 = defined_reps[:3]
    best_rep_name, best_rep_fp = top3[0]
    print(f"    Best individual rep: {best_rep_name} "
          f"(cliff contamination={best_rep_fp['cliff_contamination_rate']:.3f})")
    delta_uw = None
    delta_w = None
    if X_hybrid is not None:
        delta_uw = (hybrid_uw['cliff_contamination_rate']
                    - best_rep_fp['cliff_contamination_rate'])
        print(f"    Hybrid unweighted:  cliff contamination="
              f"{hybrid_uw['cliff_contamination_rate']:.3f} "
              f"(delta vs {best_rep_name}: {delta_uw:+.3f})")
        if hybrid_w is not None:
            delta_w = (hybrid_w['cliff_contamination_rate']
                       - best_rep_fp['cliff_contamination_rate'])
            print(f"    Hybrid weighted:    cliff contamination="
                  f"{hybrid_w['cliff_contamination_rate']:.3f} "
                  f"(delta vs {best_rep_name}: {delta_w:+.3f})")

    results['summary'] = {
        'n_ground_truth_cliffs': len(ground_truth_cliffs),
        'best_rep': best_rep_name,
        'best_rep_cliff_contamination_rate': best_rep_fp['cliff_contamination_rate'],
        'top3_reps': [(name, r['cliff_contamination_rate']) for name, r in top3],
        'delta_vs_best_rep': delta_uw if delta_uw is not None else float('nan'),
        'delta_vs_best_rep_weighted': delta_w if delta_w is not None else float('nan'),
        # Explicit flag: when the hybrid search failed, the deltas are NaN
        # because the hybrid arm did not run (NOT because the hybrid tied the
        # baseline). s09 should drop the hybrid delta for this dataset.
        'hybrid_failed': bool(X_hybrid is None),
    }

    # ── Step 6: CV-honest per-fold rescoring (decision B) ──
    # Recompute the cliff-contamination statistic under cross-validation and
    # persist PER-FOLD held-out scores for BOTH arms in the shared
    # `honest_rescore` JSON contract, so the reader can apply the identical
    # leave-one-fold-out winner re-selection to the hybrid candidates and the
    # per-rep baselines. lower_is_better=True: contamination is minimised.
    #
    # TWO fold definitions are written, side by side and labelled:
    #   'honest_rescore'       — one fold per distinct group value (the scheme
    #                            this key has always carried; on a scaffold-
    #                            grouped dataset the fold count is the number of
    #                            scaffolds, not config['n_splits']).
    #   'honest_rescore_kfold' — experiments._shared.cv_split at
    #                            config['n_splits'], the splitter the rest of
    #                            the suite uses. Each block records the fold
    #                            count it actually ran.
    honest_block, honest_block_kfold = _contamination_blocks(
        reps, hybrid_arms, labels, ground_truth_cliffs,
        groups, n_splits, task_type)
    results['honest_rescore'] = honest_block
    results['honest_rescore_kfold'] = honest_block_kfold

    # Convenience: also store the re-scored summaries (pure arithmetic; the
    # reader can recompute each from the block above via honest_rescore()).
    for _sum_key, _blk, _label in (
            ('honest_summary', honest_block, 'one fold per group'),
            ('honest_summary_kfold', honest_block_kfold, 'shared group splitter')):
        if 'error' in _blk:
            results[_sum_key] = {'error': _blk['error']}
            print(f"\n  CV-honest (held-out) cliff contamination "
                  f"[{_label}]: not computed ({_blk['error']})")
            continue
        try:
            rescored = honest_rescore(_blk['candidates_by_model'],
                                      _blk['baselines_by_model'],
                                      lower_is_better=True,
                                      exclude_folds=hr_exclude_folds(_blk))
            results[_sum_key] = rescored
            r = rescored.get(_S07_MODEL_KEY)
            if r is not None:
                print(f"\n  CV-honest (held-out) cliff contamination "
                      f"[{_label}, n_folds={_blk['n_folds']}]:")
                print(f"    hybrid={r['hybrid_honest']:.3f}  "
                      f"best_single={r['best_single_honest']:.3f}  "
                      f"delta={r['delta']:+.3f}  "
                      f"(n_folds_used={r['n_folds_used']}, single_fold={r['single_fold']})")
        except Exception as e:
            # Arithmetic-only helper; never let a rescore hiccup lose the raw block.
            results[_sum_key] = {'error': f"{type(e).__name__}: {e}"}

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S07: Noise Detection Bridge (Supplementary)',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's07_noise_bridge', args.dataset)
