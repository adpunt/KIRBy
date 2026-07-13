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
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
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
# statistic under cross-validation using the dataset's own scaffold folds
# (`groups`): the scaler stats AND the noise threshold are FIT on the TRAIN
# folds only, then contamination is MEASURED on the HELD-OUT fold, where each
# held-out point's k neighbours are drawn from TRAIN points ONLY (never from
# other held-out points). One held-out rate per fold is returned; the shared
# `honest_rescore` helper then applies the identical leave-one-fold-out winner
# re-selection to both arms. This is an UNSUPERVISED geometry statistic (not a
# downstream-model CV score), so it needs a LOCAL fold loop — `find_best_hybrid`
# / `build_honest_rescore_input` do not produce these numbers.
def _knn_noise_contamination_cv(X, labels, ground_truth_cliffs, groups,
                                k=5, standardize=True):
    """Per-fold, held-out kNN cliff-contamination rate (fit on TRAIN, measure on
    the held-out fold). Returns a list with one rate per fold; a fold that flags
    no held-out point as noise contributes NaN (0/0 is undefined — same
    convention as the whole-data version), which the honest re-scorer skips.

    Neighbour label-consistency and the 90th-percentile threshold mirror
    `KNNConsistencyFilter.identify_noise`, but the held-out points take their k
    neighbours exclusively from the train split so nothing leaks across the
    fold boundary.
    """
    from scipy.spatial.distance import cdist
    X = np.asarray(X)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    fold_rates = []
    for f in np.unique(groups):
        test_mask = groups == f
        train_mask = ~test_mask
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        kk = min(k, n_train - 1)
        # Need >= k+1 train points to define k neighbours + a train threshold;
        # degenerate folds contribute NaN (skipped by honest_rescore).
        if kk < 1 or n_test == 0:
            fold_rates.append(float('nan'))
            continue
        X_train = X[train_mask]
        X_test = X[test_mask]
        if standardize:
            scaler = StandardScaler().fit(X_train)   # TRAIN stats only
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        y_train = labels[train_mask]
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
        test_global = np.where(test_mask)[0]
        flagged_global = set(test_global[te_std > threshold].tolist())
        fp = flagged_global & ground_truth_cliffs
        rate = len(fp) / len(flagged_global) if flagged_global else float('nan')
        fold_rates.append(rate)
    return fold_rates


# The cliff-contamination statistic is a single fixed-geometry kNN filter, not a
# sweep over downstream models, so the honest_rescore input is keyed by one
# model name ('knn') by design — the model set is NOT assumed from a factory.
_S07_MODEL_KEY = 'knn'


def run(data, config, output_dir, dataset_name, **kwargs):
    """Noise detectors flag activity cliffs as noise (FP).
    Hybrid + per-rep baselines compared against ground-truth cliffs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S07: NOISE DETECTION BRIDGE (Supplementary)")
    print("=" * 70)

    search_speed = config['search_speed']

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

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
            'candidates_by_model': {},
            'baselines_by_model': {},
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
            speed=search_speed,
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
    # hybrid_embeddings collects each hybrid candidate's fitted matrix so Step 6
    # can score them under CV as honest_rescore candidate recipes.
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
        # held-out flags on some folds, so persist the honest block regardless
        # rather than emitting an empty one here.
        baselines_by_model = {_S07_MODEL_KEY: {}}
        for rep_name, X_rep in reps.items():
            baselines_by_model[_S07_MODEL_KEY][rep_name] = _knn_noise_contamination_cv(
                X_rep, labels, ground_truth_cliffs, groups)
        candidates = []
        for cand_name, X_cand in hybrid_embeddings.items():
            std_flag = (cand_name != 'hybrid_weighted')
            candidates.append({
                'name': cand_name,
                'fold_scores': _knn_noise_contamination_cv(
                    X_cand, labels, ground_truth_cliffs, groups,
                    standardize=std_flag),
            })
        results['honest_rescore'] = {
            'lower_is_better': True,
            'candidates_by_model': {_S07_MODEL_KEY: candidates} if candidates else {},
            'baselines_by_model': baselines_by_model,
        }
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
    # Per-rep held-out baselines.
    baselines_by_model = {}
    for rep_name, X_rep in reps.items():
        fold_rates = _knn_noise_contamination_cv(
            X_rep, labels, ground_truth_cliffs, groups)
        baselines_by_model.setdefault(_S07_MODEL_KEY, {})[rep_name] = fold_rates

    # Hybrid candidate recipes (unweighted / weighted). The weighted embedding is
    # already globally standardized+weighted, so it is NOT re-standardized per
    # fold (re-standardizing would divide the sqrt(importance) scaling back out
    # and collapse it to the unweighted candidate) — the train-fit noise
    # threshold + train-only neighbours still give the held-out honesty.
    candidates = []
    for cand_name, X_cand in hybrid_embeddings.items():
        std_flag = (cand_name != 'hybrid_weighted')
        fold_rates = _knn_noise_contamination_cv(
            X_cand, labels, ground_truth_cliffs, groups, standardize=std_flag)
        candidates.append({'name': cand_name, 'fold_scores': fold_rates})
    candidates_by_model = {_S07_MODEL_KEY: candidates} if candidates else {}

    honest_block = {
        'lower_is_better': True,
        'candidates_by_model': candidates_by_model,
        'baselines_by_model': baselines_by_model,
    }
    results['honest_rescore'] = honest_block

    # Convenience: also store the re-scored summary (pure arithmetic; the reader
    # can recompute this from the block above via honest_rescore()).
    try:
        rescored = honest_rescore(candidates_by_model, baselines_by_model,
                                  lower_is_better=True)
        results['honest_summary'] = rescored
        r = rescored.get(_S07_MODEL_KEY)
        if r is not None:
            print(f"\n  CV-honest (held-out) cliff contamination:")
            print(f"    hybrid={r['hybrid_honest']:.3f}  "
                  f"best_single={r['best_single_honest']:.3f}  "
                  f"delta={r['delta']:+.3f}  "
                  f"(n_folds_used={r['n_folds_used']}, single_fold={r['single_fold']})")
    except Exception as e:
        # Arithmetic-only helper; never let a rescore hiccup lose the raw block.
        results['honest_summary'] = {'error': f"{type(e).__name__}: {e}"}

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
