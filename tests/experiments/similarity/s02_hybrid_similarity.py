#!/usr/bin/env python3
"""S02: Hybrid Property-Aware Similarity -- the fix via precomputed pool best + weighting.

Paper: Similarity
Conclusion: Hybrid distance improves SAC by 100%+.
Dependencies: None.

Computes per-rep SAC baselines, finds best individual rep, retrieves the
precomputed pool best hybrid via get_fitted_pool_hybrid, then for each weight
method computes importance-weighted distance matrix and compares SAC: hybrid
weighted vs best individual rep.
Saves blueprint and search details for downstream experiments (S03, S04, S05).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_fitted_hybrid, get_fitted_pool_hybrid,
    flatten_fold_weights, extract_top3_base_reps,
    cv_split, honest_rescore, build_fold_hybrid,
    hr_exclude_folds,
)
from experiments.similarity.similarity_helpers import (
    similarity_activity_correlation, compute_sali, violation_rate,
    run_antibody_cliff_analysis,
)

from kirby.hybrid import compute_feature_importance
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# =============================================================================
# CROSS-VALIDATED, SELECTION-HONEST SAC (redesign — decision B, user-approved)
# =============================================================================
# ORIGINAL (whole-dataset) design: the best individual rep was argmax'd over the
# SAC of every rep on the FULL dataset (run() best_rep loop), the hybrid was
# built on the FULL dataset via get_fitted_pool_hybrid, and its weighted SAC was
# computed on the FULL dataset too. Both arms therefore SELECTED and SCORED on
# the same samples — pure selection optimism, exactly the leak the honest-rescore
# fix targets.
#
# REDESIGN: split by the dataset groups (group-aware CV). SELECT the winning rep
# and the winning hybrid recipe on TRAIN folds only, then COMPUTE the SAC
# statistic on the HELD-OUT fold, and average held-out SAC across folds. Because
# SAC is an UNSUPERVISED distance-metric statistic (not a model fold-score from
# find_best_hybrid), the SearchResult-based build_honest_rescore_input helper
# does not fit here; instead this file computes per-fold HELD-OUT SAC for every
# baseline rep and every hybrid recipe locally (below) and writes them into the
# CONTRACT's "honest_rescore" JSON block. The shared honest_rescore() helper then
# supplies the identical held-out arithmetic used everywhere else: for each
# held-out fold it re-picks the winner by the mean over the OTHER folds and reads
# that winner's held-out score. Any feature SELECTION that depends on labels
# (the importance WEIGHTING) is fit on the TRAIN rows of each fold only; the
# unweighted arms need no fitting so their held-out SAC is leak-free by
# construction and the honesty comes purely from the per-fold winner re-pick.
#
# WHAT IS AND IS NOT REFIT PER FOLD. The hybrid's FEATURES are refit per fold
# (2026-09-04): each held-out fold rebuilds the hybrid with build_fold_hybrid on
# its own training rows, so no feature is chosen using a label the fold then
# scores against. The importance WEIGHTS are refit per fold as well. What is
# still chosen over every row is the RECIPE — allocation method, budget,
# importance method, internal evaluator — which comes from the whole-dataset
# pool search (get_fitted_pool_hybrid). That is a separate open finding against
# get_best_search_for_downstream, not something this file decides. The
# recipe-level choice s02 itself makes (which weight method, weighted vs
# unweighted) is made per held-out fold by honest_rescore().
#
# SAC is a single-model statistic (the distance metric; downstream_model='knn'),
# so everything is keyed under one model name, 'knn'. lower_is_better is False
# (higher SAC = distance better preserves the structure-activity relationship).

_SAC_MODEL_KEY = 'knn'  # SAC is a distance-metric statistic; single "model".


def _hybrid_weights_from_fi(fi, wm_importance, n_cols):
    """Build the per-column hybrid weight vector from a feature_info layout and
    a (train-computed) per-rep importance dict.

    Mirrors run()'s full-data weight construction exactly, but the importance
    source is passed in (so callers can supply TRAIN-only importance). Raises if
    the flattened length does not match the hybrid column count.
    """
    feat_info = {}
    for rep_name, info in fi.items():
        if rep_name == 'augmentation_info' or not isinstance(info, dict):
            continue
        idx = info.get('selected_indices')
        if idx is None:
            continue
        entry = {'selected_indices': idx,
                 'n_features': info.get('n_features', len(idx))}
        if rep_name in wm_importance:
            entry['importance_scores'] = wm_importance[rep_name][idx]
        feat_info[rep_name] = entry
    if fi.get('augmentation_info'):
        # Augmentation importance stays as the global blueprint's (see LIMITATION
        # above); it keeps weights_flat aligned with the hybrid columns.
        feat_info['augmentation_info'] = fi['augmentation_info']
    weights_flat = flatten_fold_weights(feat_info)
    if len(weights_flat) != n_cols:
        raise ValueError(
            f"weights_flat length {len(weights_flat)} != hybrid columns {n_cols}")
    return weights_flat


def _cv_honest_sac(reps, labels, groups, blueprint, aug_condition,
                   augmentations, weight_methods,
                   n_splits, n_sample, is_clf, task_type, seed=42):
    """Cross-validated, selection-honest SAC (the redesign described above).

    Computes, for every held-out CV fold, the SAC of each BASELINE rep
    (unweighted + train-weighted) and each HYBRID recipe (unweighted +
    train-weighted per weight method) on that held-out fold's samples. Label-
    dependent importance weighting is fit on TRAIN rows only; unweighted arms
    use no labels for the distance.

    The hybrid's FEATURES are now selected per fold too (2026-09-04): each fold
    rebuilds the hybrid from `blueprint` with build_fold_hybrid on that fold's
    training rows. Before that the whole-dataset matrix was sliced, so the
    features had been chosen with the held-out rows' labels in hand, while the
    single representations it is compared against had no selection at all.

    Returns the CONTRACT "honest_rescore" block ready to json.dump:
        {"lower_is_better": False,
         "candidates_by_model": {"knn": [{"name", "fold_scores"}, ...]},
         "baselines_by_model":  {"knn": {rep_key: [fold_scores], ...}}}
    A reader loads it and calls honest_rescore(...) to get the honest deltas.
    """
    labels = np.asarray(labels)
    baseline_folds = {}   # rep_key   -> [per-fold held-out SAC]
    cand_folds = {}       # recipe id -> [per-fold held-out SAC]

    def _push(store, key, val):
        store.setdefault(key, []).append(float(val))

    def _sac(X_sub, y_sub, weights=None):
        try:
            if weights is None:
                D = compute_distance_matrix(X_sub, standardize=True)
            else:
                D = compute_distance_matrix(
                    X_sub, feature_weights=np.maximum(weights, 0),
                    standardize=True)
            rho, _ = similarity_activity_correlation(
                D, y_sub, n_sample, is_clf=is_clf)
            return rho
        except Exception:
            return float('nan')

    _sample_X = next(iter(reps.values()))
    _augs = augmentations if (augmentations and aug_condition != 'none') else None
    for train_idx, test_idx in cv_split(groups, n_splits, _sample_X, labels,
                                        seed=seed, task_type=task_type):
        y_test = labels[test_idx]
        y_train = labels[train_idx]

        # This fold's hybrid: features chosen from the training rows only, then
        # applied to the held-out rows. A build that throws drops the hybrid arm
        # for this fold; the helper tolerates a ragged fold list.
        try:
            _, Xh_test, fi = build_fold_hybrid(
                {k: X[train_idx] for k, X in reps.items()},
                {k: X[test_idx] for k, X in reps.items()},
                y_train, blueprint,
                augmentations_train=({k: v[train_idx] for k, v in _augs.items()}
                                     if _augs else None),
                augmentations_val=({k: v[test_idx] for k, v in _augs.items()}
                                   if _augs else None),
                groups=(np.asarray(groups)[train_idx]
                        if groups is not None else None))
        except Exception:
            Xh_test, fi = None, None

        # Train-only importance for the WEIGHTED arms (lightgbm_gain matches the
        # full-data weighted baselines). Failure -> weighted arms simply absent
        # this fold (helper tolerates ragged/missing folds).
        reps_train = {k: X[train_idx] for k, X in reps.items()}
        try:
            base_imp_train = compute_feature_importance(
                reps_train, y_train, method='lightgbm_gain')
        except Exception:
            base_imp_train = None

        # --- baseline single-rep arm ---
        for rep_name, X in reps.items():
            Xt = X[test_idx]
            _push(baseline_folds, rep_name, _sac(Xt, y_test))
            if base_imp_train is not None and rep_name in base_imp_train:
                _push(baseline_folds, f'{rep_name}|weighted',
                      _sac(Xt, y_test, weights=base_imp_train[rep_name]))

        # --- hybrid candidate arm ---
        if Xh_test is None:
            continue
        _push(cand_folds, 'hybrid_unweighted', _sac(Xh_test, y_test))
        for wm in weight_methods:
            try:
                if wm == 'lightgbm_gain' and base_imp_train is not None:
                    wm_imp = base_imp_train
                else:
                    wm_imp = compute_feature_importance(
                        reps_train, y_train, method=wm)
                weights_flat = _hybrid_weights_from_fi(
                    fi, wm_imp, Xh_test.shape[1])
                sac_hw = _sac(Xh_test, y_test, weights=weights_flat)
            except Exception:
                sac_hw = float('nan')
            _push(cand_folds, f'hybrid_weighted|{wm}', sac_hw)

    return {
        'lower_is_better': False,  # SAC higher = better (see redesign note).
        # Every fold counts. The candidates here are the arms this file built itself,
        # not a search's recipe list, so no fold filtered them.
        'exclude_folds': [],
        'candidates_by_model': {
            _SAC_MODEL_KEY: [{'name': k, 'fold_scores': v}
                             for k, v in cand_folds.items()],
        },
        'baselines_by_model': {
            _SAC_MODEL_KEY: dict(baseline_folds),
        },
    }


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """The fix: find_best_hybrid + importance-weighted distance improves SAC."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S02: HYBRID PROPERTY-AWARE SIMILARITY")
    print("=" * 70)

    weight_methods = kwargs.get('weight_methods', ['lightgbm_gain'])
    n_sample = kwargs.get('n_sample', 5000)
    task_type = data['task_type']
    is_classification = task_type == 'classification'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    results = {'weight_methods': {}, 'per_rep': {}}

    # Compute importance for weighted baselines (once, used for all baselines)
    print("  Computing feature importance (lgbm_gain) for weighted baselines...")
    try:
        importance = compute_feature_importance(reps, labels, method='lightgbm_gain')
    except Exception as e:
        print(f"  WARNING: importance failed ({e}), weighted baselines unavailable")
        importance = None

    # Best individual rep baseline (the real-world comparison)
    best_rep_sac = -np.inf  # SAC > 0 is good; we want highest
    best_rep_name = None
    best_rep_sac_weighted = -np.inf
    best_rep_name_weighted = None
    for rep_name, X in reps.items():
        D_rep = compute_distance_matrix(X, standardize=True)
        sac_rep, _ = similarity_activity_correlation(D_rep, labels, n_sample, is_clf=is_classification)
        viol_rep = violation_rate(D_rep, labels, k=5, is_clf=is_classification)
        rep_result = {
            'dim': X.shape[1], 'sac_rho': sac_rep, 'violation_rate': viol_rep,
        }

        # Weighted version. Clamp importance to >=0 before passing: the helper
        # applies sqrt(feature_weights) with no guard, so any signed importance
        # (permutation / SHAP, which weight_methods may request) would inject
        # NaN. lightgbm_gain is non-negative so this is a no-op there; the
        # clamp matches every other similarity experiment (s04/s05/s06/s07).
        if importance is not None and rep_name in importance:
            D_rep_w = compute_distance_matrix(
                X, feature_weights=np.maximum(importance[rep_name], 0),
                standardize=True)
            sac_rep_w, _ = similarity_activity_correlation(D_rep_w, labels, n_sample, is_clf=is_classification)
            viol_rep_w = violation_rate(D_rep_w, labels, k=5, is_clf=is_classification)
            rep_result['sac_rho_weighted'] = sac_rep_w
            rep_result['violation_rate_weighted'] = viol_rep_w
            if sac_rep_w > best_rep_sac_weighted:
                best_rep_sac_weighted = sac_rep_w
                best_rep_name_weighted = rep_name
            print(f"  {rep_name}: SAC={sac_rep:.4f}, SAC_w={sac_rep_w:.4f}, dim={X.shape[1]}")
        else:
            print(f"  {rep_name}: SAC={sac_rep:.4f}, viol={viol_rep:.3f}, dim={X.shape[1]}")

        results['per_rep'][rep_name] = rep_result
        if best_rep_name is None or sac_rep > best_rep_sac:
            best_rep_sac = sac_rep
            best_rep_name = rep_name

    results['best_individual_rep'] = {
        'name': best_rep_name, 'sac_rho': best_rep_sac,
        'dim': reps[best_rep_name].shape[1],
    }
    if best_rep_name_weighted is not None:
        results['best_individual_rep_weighted'] = {
            'name': best_rep_name_weighted, 'sac_rho': best_rep_sac_weighted,
            'dim': reps[best_rep_name_weighted].shape[1],
        }
    print(f"\n  Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f})")
    if best_rep_name_weighted:
        print(f"  Best individual rep (weighted): {best_rep_name_weighted} "
              f"(SAC_w={best_rep_sac_weighted:.4f})")

    # Concat baseline (unweighted + weighted)
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    D_concat = compute_distance_matrix(X_concat, standardize=True)
    sac_concat, _ = similarity_activity_correlation(D_concat, labels, n_sample, is_clf=is_classification)
    viol_concat = violation_rate(D_concat, labels, k=5, is_clf=is_classification)

    concat_result = {
        'dim': X_concat.shape[1],
        'sac_rho': sac_concat,
        'violation_rate': viol_concat,
    }

    if importance is not None:
        w_concat = np.concatenate([importance[k] for k in sorted(reps.keys())])
        # Clamp to >=0 (sqrt-weighting in the helper has no guard; see above).
        D_concat_w = compute_distance_matrix(
            X_concat, feature_weights=np.maximum(w_concat, 0), standardize=True)
        sac_concat_w, _ = similarity_activity_correlation(D_concat_w, labels, n_sample, is_clf=is_classification)
        viol_concat_w = violation_rate(D_concat_w, labels, k=5, is_clf=is_classification)
        concat_result['sac_rho_weighted'] = sac_concat_w
        concat_result['violation_rate_weighted'] = viol_concat_w
        print(f"  Concat: SAC={sac_concat:.4f}, SAC_w={sac_concat_w:.4f}, "
              f"dim={X_concat.shape[1]}")
    else:
        print(f"  Concat (reference only): SAC={sac_concat:.4f}, viol={viol_concat:.3f}, "
              f"dim={X_concat.shape[1]}")

    results['concat'] = concat_result

    # best_sac is the best SAC any HYBRID arm reached, and nothing else. It used
    # to be seeded with best_rep_sac, so it could only ever rise above the best
    # single representation. s09 reads this key and subtracts best_rep_sac from
    # it, so every dataset where the hybrid LOST contributed exactly 0.0000 to
    # the cross-dataset SAC delta instead of its real negative value. Seeded
    # with None so a losing hybrid reports a negative delta and a hybrid that
    # never ran reports nothing at all.
    best_sac = None
    best_method = None

    # Run search once (selection is weight_method-independent), then vary weights
    t0 = time.time()
    budget = kwargs.get('budget', 100)
    augmentations = data.get('augmentations')
    print(f"  Running hybrid search (budget={budget})...")
    try:
        X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=budget,
            augmentations=augmentations,
            dataset_name=dataset_name, task_type=task_type)
    except Exception as e:
        print(f"    Search FAILED: {e}")
        # No hybrid exists, so there is no hybrid SAC. Reporting best_rep_sac
        # here would have read downstream as a hybrid that exactly tied.
        results['best_weight_method'] = None
        results['best_sac'] = None
        results['hybrid_status'] = f'search failed: {e}'
        return results

    search_time = time.time() - t0
    hybrid_dim = X_hybrid.shape[1]

    # The recipe the held-out fold loop rebuilds from. It comes from the
    # whole-dataset pool search, same as the matrix above; what changes below is
    # that each fold re-selects its FEATURES from that recipe on its own
    # training rows instead of slicing this matrix.
    _blueprint_for_folds = (
        search.best_by_model['knn']['blueprint']
        if search.best_by_model.get('knn', {}).get('blueprint') else None)

    top3 = extract_top3_base_reps(search)
    if top3:
        print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        results['top3_base_reps'] = [(n, s) for n, s in top3]

    # Hybrid unweighted (same for all weight methods -- only compute once)
    D_hybrid = compute_distance_matrix(X_hybrid, standardize=True)
    sac_hybrid, _ = similarity_activity_correlation(D_hybrid, labels, n_sample, is_clf=is_classification)
    viol_hybrid = violation_rate(D_hybrid, labels, k=5, is_clf=is_classification)
    sali_hybrid = compute_sali(D_hybrid, labels, is_clf=is_classification)

    print(f"    Search done ({search_time:.1f}s), dim={hybrid_dim}")
    print(f"    Hybrid unweighted: SAC={sac_hybrid:.4f}, viol={viol_hybrid:.3f}")

    for wm in weight_methods:
        print(f"\n  Weight method: {wm}")

        # Compute importance with this weight method, apply to selected features.
        # Use `fi` (feature_info dict from get_fitted_pool_hybrid) as the column
        # layout source — it matches X_hybrid exactly. Override per-rep
        # importance_scores with the current weight method's scores.
        try:
            if wm == 'lightgbm_gain' and importance is not None:
                wm_importance = importance
            else:
                wm_importance = compute_feature_importance(reps, labels, method=wm)
            feat_info = {}
            for rep_name, info in fi.items():
                if rep_name == 'augmentation_info' or not isinstance(info, dict):
                    continue
                idx = info.get('selected_indices')
                if idx is None:
                    continue
                entry = {'selected_indices': idx,
                         'n_features': info.get('n_features', len(idx))}
                if rep_name in wm_importance:
                    entry['importance_scores'] = wm_importance[rep_name][idx]
                feat_info[rep_name] = entry
            if fi.get('augmentation_info'):
                feat_info['augmentation_info'] = fi['augmentation_info']
            weights_flat = flatten_fold_weights(feat_info)
            assert len(weights_flat) == X_hybrid.shape[1], (
                f"weights_flat length {len(weights_flat)} != X_hybrid columns {X_hybrid.shape[1]}"
            )
        except Exception as e:
            print(f"    Weight computation failed: {e}")
            results['weight_methods'][wm] = {'error': str(e)}
            continue

        # Hybrid weighted. Clamp to >=0 (sqrt-weighting in the helper has no
        # guard; wm may be a signed importance method).
        D_weighted = compute_distance_matrix(
            X_hybrid, feature_weights=np.maximum(weights_flat, 0), standardize=True)
        sac_weighted, _ = similarity_activity_correlation(D_weighted, labels, n_sample, is_clf=is_classification)
        viol_weighted = violation_rate(D_weighted, labels, k=5, is_clf=is_classification)
        sali_weighted = compute_sali(D_weighted, labels, is_clf=is_classification)

        wm_result = {
            'hybrid_unweighted': {
                'dim': hybrid_dim,
                'sac_rho': sac_hybrid,
                'violation_rate': viol_hybrid,
                'sali_mean': sali_hybrid['mean'],
            },
            'hybrid_weighted': {
                'dim': hybrid_dim,
                'sac_rho': sac_weighted,
                'violation_rate': viol_weighted,
                'sali_mean': sali_weighted['mean'],
            },
            'sac_delta_from_best_rep': sac_weighted - best_rep_sac,
            'search_time_seconds': search_time,
            'blueprint': (search.best_by_model['knn']['blueprint'].to_dict()
                          if search.best_by_model.get('knn', {}).get('blueprint') else None),
        }

        results['weight_methods'][wm] = wm_result

        # higher = better distance-activity correlation
        if best_sac is None or sac_weighted > best_sac:
            best_sac = sac_weighted
            best_method = wm

        print(f"    Hybrid weighted:   SAC={sac_weighted:.4f}, viol={viol_weighted:.3f}")
        print(f"    Delta SAC from best rep ({best_rep_name}): "
              f"{sac_weighted - best_rep_sac:+.4f}")

    results['best_weight_method'] = best_method
    results['best_sac'] = best_sac

    # Summary
    print(f"\n  Summary:")
    print(f"    Best individual rep: {best_rep_name} (SAC={best_rep_sac:.4f}, "
          f"dim={reps[best_rep_name].shape[1]})")
    if best_sac is None:
        results['hybrid_status'] = 'no weight method produced a hybrid SAC'
        print(f"    Best hybrid:        NONE (every weight method failed)")
    else:
        print(f"    Best hybrid:        {best_method} (SAC={best_sac:.4f}, "
              f"dim={hybrid_dim})")
        delta_sac = best_sac - best_rep_sac
        print(f"    Delta vs best rep:  {delta_sac:+.4f}")

    # -------------------------------------------------------------------------
    # Cross-validated, selection-honest SAC (redesign — decision B). The block
    # above is the ORIGINAL whole-dataset statistic (kept for back-compat /
    # reference); the numbers reported as the honest result come from held-out
    # CV folds with per-fold winner re-selection. See _cv_honest_sac docstring.
    # Persisted under the CONTRACT key "honest_rescore" (per-fold scores for BOTH
    # arms) so any reader can call honest_rescore(...) without re-running.
    # -------------------------------------------------------------------------
    n_splits = int(config.get('n_splits', 5))
    if _blueprint_for_folds is None:
        results['honest_rescore_error'] = (
            'pool search returned no kNN blueprint, so no per-fold hybrid '
            'could be rebuilt')
        print("  Honest CV SAC skipped: no kNN blueprint from the pool search")
    try:
        if _blueprint_for_folds is None:
            raise RuntimeError('no kNN blueprint')
        hr_block = _cv_honest_sac(
            reps, labels, groups, _blueprint_for_folds,
            _meta.get('aug_condition', 'none') if _meta else 'none',
            augmentations, weight_methods,
            n_splits=n_splits, n_sample=n_sample, is_clf=is_classification,
            task_type=task_type)
        hr_block['recipe_selection'] = (_meta or {}).get('selection_scope')
        results['honest_rescore'] = hr_block
        # Reuse the shared honest arithmetic to also store the resolved summary
        # (held-out hybrid vs best-single, direction-corrected delta) inline.
        hr_summary = honest_rescore(
            hr_block['candidates_by_model'], hr_block['baselines_by_model'],
            hr_block['lower_is_better'],
            exclude_folds=hr_exclude_folds(hr_block),
            # The fold loop re-selects the FEATURES on each fold's training
            # rows, but the allocation method, budget, importance method and
            # evaluator all came from the pool search over every row. The
            # number below is held out for the weights and not for the recipe;
            # s09 prints this label next to it.
            recipe_selection=(_meta or {}).get('selection_scope'))
        results['honest_rescore_result'] = hr_summary
        knn = hr_summary.get(_SAC_MODEL_KEY, {})
        if knn:
            print(f"\n  Honest CV SAC ({n_splits}-fold held-out): "
                  f"hybrid={knn.get('hybrid_honest'):+.4f}, "
                  f"best_single={knn.get('best_single_honest'):+.4f}, "
                  f"delta={knn.get('delta'):+.4f}"
                  + ("  [single-fold: OPTIMISTIC, not honest]"
                     if knn.get('single_fold') else ""))
    except Exception as e:
        # Group-aware CV can legitimately fail (too few distinct groups for
        # n_splits); never let it crash the whole experiment. Record and move on.
        print(f"  Honest CV SAC skipped: {e}")
        results['honest_rescore_error'] = str(e)

    # Sequence-based cliff sweep (antibody datasets + nucleotide) — independent
    # of any representation choice. Same sweep as s01 for cross-experiment
    # comparability; CDR indices cached per-dataset for repeat runs.
    cliff_results = run_antibody_cliff_analysis(
        data, is_clf=is_classification, dataset_name=dataset_name)
    if cliff_results is not None:
        results['_sequence_cliffs'] = cliff_results

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S02: Hybrid Property-Aware Similarity',
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
    save_experiment_results(results, args.output, 's02_hybrid_similarity', args.dataset)
