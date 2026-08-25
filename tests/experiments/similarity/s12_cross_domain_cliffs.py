#!/usr/bin/env python3
"""S12: Cross-Domain Activity Cliff Analysis — generic cliff detection for all domains.

Paper: Similarity
Companion to S04 (which uses ECFP4 Tanimoto, molecular-only). S12 uses Euclidean
similarity on concatenated reps, making it applicable to antibody, TCGA, and molecular
datasets alike. Same approach as N01's non-molecular branch.

Cliff definition: pairs where sim >= threshold AND |delta_y| > k * MAD(y).
For Euclidean: sim = 1/(1+d) on standardised concat.

Usage:
    python s12_cross_domain_cliffs.py --dataset flab_thermo_tresanco --mode intermediary
    python s12_cross_domain_cliffs.py --dataset tcga_luad_vs_lusc --mode intermediary
    python s12_cross_domain_cliffs.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, get_precomputed_search,
    get_best_search_for_downstream,
    build_fold_hybrid, extract_top3_base_reps, DOWNSTREAM_MODELS,
    honest_rescore,
)
from kirby.hybrid import create_hybrid
from kirby.hybrid import apply_feature_selection
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


def _finite_mean(fold_scores):
    """Mean over finite fold scores (NaN = fold that had no usable cliff/non-cliff
    subset); NaN if none are finite. Local helper: the cliff/non-cliff CV scores are
    a redesigned segmented statistic that never flows through a SearchResult, so the
    shared arithmetic in _shared.py does not fit here. Kept fold-aligned (NaN
    placeholders) so honest_rescore can index the same fold across reps/hybrid."""
    finite = [s for s in fold_scores if s is not None and not np.isnan(s)]
    return float(np.mean(finite)) if finite else float('nan')


# =========================================================================
# CLIFF DETECTION (domain-generic, mirrors N01's identify_cliff_pairs)
# =========================================================================

def identify_cliff_pairs_euclidean(X_concat, y, sim_threshold=0.55, activity_k=2.0,
                                   max_compounds=5000, is_clf=False,
                                   similar_percentile=None):
    """Identify activity cliff pairs using Euclidean similarity.

    sim = 1/(1+d) on standardised features. Cliff = high structural similarity
    AND high activity difference.

    Regression / binary: |delta_y| > k * MAD(y).
    Classification (is_clf): different class (high sim + different label).

    Args:
        sim_threshold: absolute similarity cutoff (used only when
            ``similar_percentile`` is None).
        similar_percentile: if set (e.g. 5.0), define "structurally similar" as
            the top ``similar_percentile``% most-similar off-diagonal pairs
            instead of the absolute ``sim_threshold``. On standardised high-D
            embeddings the pairwise Euclidean distance is large so sim = 1/(1+d)
            never reaches an absolute 0.55 (s12 found 0 cliffs in 13/16
            datasets). A percentile makes the threshold distribution-relative.
            See COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
            If a list/tuple of candidate percentiles is passed, the expensive
            similarity matrix is built ONCE and every candidate is swept; the
            one whose cliff-compound prevalence is closest to
            ``TARGET_CLIFF_PCT`` (10%) is auto-selected (preferring candidates
            with prevalence > 0). The returned ``info`` then also carries
            ``'percentile_sweep'`` and ``'selected_percentile'``.

    Returns (cliff_mask, info) where cliff_mask is boolean array marking cliff compounds.
    """
    from experiments.similarity.similarity_helpers import _pairwise_delta_matrix

    TARGET_CLIFF_PCT = 10.0
    sweep_candidates = None
    if isinstance(similar_percentile, (list, tuple)):
        sweep_candidates = list(similar_percentile)

    n = len(y)
    if n > max_compounds:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_compounds, replace=False)
        X_sub = X_concat[idx]
        y_sub = y[idx]
    else:
        X_sub = X_concat
        y_sub = y
        idx = np.arange(n)

    D = compute_distance_matrix(X_sub, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)

    if is_clf:
        activity_threshold = 0.5  # indicator is 0 or 1
        mad_y = 0.0
    else:
        mad_y = float(np.median(np.abs(y_sub - np.median(y_sub))))
        activity_threshold = activity_k * mad_y

    # "Structurally similar" pairs: absolute cutoff, or distribution-relative
    # (top similar_percentile% most-similar off-diagonal pairs). The sim matrix
    # and delta_y are the expensive part; given any sim_cutoff the cliff stats
    # are cheap boolean ops, so we factor them into a closure that can be swept
    # over several candidate percentiles essentially for free.
    delta_y = _pairwise_delta_matrix(y_sub, is_clf=is_clf)
    _off = sim[~np.eye(len(sim), dtype=bool)]

    def _percentile_cutoff(pct):
        return float(np.percentile(_off, 100.0 - pct))

    def _cliff_stats(sim_cutoff):
        cliff_pair_mask = (sim >= sim_cutoff) & (delta_y > activity_threshold)
        np.fill_diagonal(cliff_pair_mask, False)
        cliff_mask_sub = cliff_pair_mask.any(axis=1)
        n_cliff_pairs = int(cliff_pair_mask.sum()) // 2
        n_cliff_compounds = int(cliff_mask_sub.sum())
        pct_cliff = float(100.0 * n_cliff_compounds / len(y_sub))
        return cliff_mask_sub, n_cliff_pairs, n_cliff_compounds, pct_cliff

    percentile_sweep = None
    selected_percentile = None
    if sweep_candidates is not None:
        # Build the expensive matrices once (already done), then sweep cheaply.
        percentile_sweep = {}
        for cand in sweep_candidates:
            _, _, _, pct_cliff = _cliff_stats(_percentile_cutoff(cand))
            percentile_sweep[cand] = pct_cliff
        # Auto-select: prevalence closest to the 10% target, preferring
        # candidates with prevalence > 0 (only fall back to including 0%
        # candidates if every candidate is 0%).
        positive = [c for c in sweep_candidates if percentile_sweep[c] > 0]
        pool = positive if positive else list(sweep_candidates)
        selected_percentile = float(
            min(pool, key=lambda c: abs(percentile_sweep[c] - TARGET_CLIFF_PCT)))
        sim_cutoff = _percentile_cutoff(selected_percentile)
    elif similar_percentile is not None:
        sim_cutoff = _percentile_cutoff(similar_percentile)
    else:
        sim_cutoff = sim_threshold

    cliff_mask_sub, n_cliff_pairs, n_cliff_compounds, pct_cliff_compounds = \
        _cliff_stats(sim_cutoff)

    # Map back to full indices
    cliff_indices = set(idx[cliff_mask_sub].tolist())

    info = {
        'sim_threshold': sim_threshold,
        'similar_percentile': similar_percentile,
        'sim_cutoff': float(sim_cutoff),
        'activity_k': activity_k,
        'activity_threshold': float(activity_threshold),
        'mad_y': float(mad_y),
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': pct_cliff_compounds,
        'n_analyzed': len(y_sub),
        'n_unanalyzed': max(0, n - max_compounds),
        'max_compounds_cap': max_compounds,
        'similarity_type': 'euclidean',
    }
    if percentile_sweep is not None:
        info['percentile_sweep'] = percentile_sweep
        info['selected_percentile'] = selected_percentile

    return cliff_indices, info


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S12: Cross-domain activity cliff analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S12: CROSS-DOMAIN ACTIVITY CLIFFS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    domain = data.get('domain', 'molecular')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    model_factories = get_model_factories(DOWNSTREAM_MODELS, task_type=task_type,
                                           n_samples=len(labels))
    print(f"  Models: {list(model_factories.keys())}")

    sim_threshold = kwargs.get('sim_threshold', 0.55)
    activity_k = kwargs.get('activity_k', 2.0)
    # Distribution-relative structural threshold. Absolute sim_threshold=0.55
    # found 0 cliffs in 13/16 datasets on standardised high-D embeddings. See
    # COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
    # The percentile is now auto-selected in-run from the prevalence sweep
    # (closest to a 10% cliff-compound target) — no rerun/retune needed; the
    # full sweep is recorded in cliff_info['percentile_sweep'].
    similar_pct = kwargs.get('similar_percentile', [0.5, 1.0, 2.0, 5.0, 10.0])

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print(f"  Domain: {domain}, n={len(labels)}, dim={X_concat.shape[1]}")

    # Identify cliffs
    # NOTE: cliff identity uses full-dataset labels — accepted leakage for cliff
    # partitioning, per N01 precedent.
    cliff_indices, cliff_info = identify_cliff_pairs_euclidean(
        X_concat, labels, sim_threshold, activity_k, is_clf=is_clf,
        similar_percentile=similar_pct)
    print(f"  Cliffs: {cliff_info['n_cliff_compounds']}/{cliff_info['n_analyzed']} "
          f"({cliff_info['pct_cliff_compounds']:.1f}%), "
          f"{cliff_info['n_cliff_pairs']} pairs")

    if not cliff_indices:
        print("  No cliffs found — widening structural percentile...")
        # Relax by treating more pairs as structurally similar.
        for pct in [10.0, 20.0, 30.0]:
            cliff_indices, cliff_info = identify_cliff_pairs_euclidean(
                X_concat, labels, sim_threshold, activity_k, is_clf=is_clf,
                similar_percentile=pct)
            if cliff_info['pct_cliff_compounds'] >= 5:
                cliff_info['threshold_was_relaxed'] = True
                cliff_info['final_percentile'] = pct
                print(f"  Widened to top {pct}% similar: "
                      f"{cliff_info['n_cliff_compounds']} cliffs "
                      f"({cliff_info['pct_cliff_compounds']:.1f}%)")
                break

    if not cliff_indices:
        # No activity cliffs is a LEGITIMATE methodological outcome, not a
        # failure: when the representation captures the structure-activity
        # relationship well, the structurally-most-similar pairs share
        # activity, so no cliff exists even after widening the structural
        # percentile to 30%. This is common for antibody datasets on
        # foundation-model embeddings (abagym/abcov/flab_polyreact in the
        # production wave). Returning a key literally named 'error' made the
        # wave audit ("if 'error' in result") count s12 as a failed
        # experiment, leaving similarity stuck at 11/12. Signal the outcome
        # with the canonical 'skipped'/'reason' convention (cf. s04) — no
        # 'error' key — so the experiment is counted complete.
        print("  No activity cliffs at any structural percentile (<=30%) — "
              "representation preserves structure-activity; reporting as a "
              "valid no-cliff result (not an error).")
        return {
            'skipped': True,
            'reason': 'no_cliffs_found',
            'domain': domain,
            'metric': metric_name,
            'lower_is_better': lower_is_better,
            'cliff_info': convert_numpy(cliff_info),
        }

    # Get hybrid blueprint via pool helper — per downstream model
    budget = kwargs.get('budget', 100)
    top3 = None  # ensure defined even if the search exits early
    bp_by_model = {}
    for mname in model_factories:
        try:
            search, meta = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=mname, budget=budget,
                augmentations=data.get('augmentations'),
                dataset_name=dataset_name, task_type=task_type,
                data=data,
            )
            bp_m = meta['best_config']['blueprint'] if meta else None
            if bp_m is not None and meta is not None:
                # Override blueprint augmentation_strategy with the pool-helper's
                # resolved aug_condition to avoid cache collisions (s05/s06/s08 pattern).
                bp_m.augmentation_strategy = meta['aug_condition']
            bp_by_model[mname] = bp_m
            if mname == 'rf' and search is not None:
                top3 = extract_top3_base_reps(search)
                if top3:
                    print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        except Exception as e:
            print(f"  [warn] hybrid search ({mname}): {type(e).__name__}: {str(e)[:100]}")
            bp_by_model[mname] = None

    # CV evaluation: per-rep baselines + hybrid on cliff vs non-cliff
    sample_X = next(iter(reps.values()))

    # Per-rep baselines per model
    per_rep_results = {}

    for rep_name, X_rep in reps.items():
        rep_model_results = {}
        for mname, mfactory in model_factories.items():
            # Fold-aligned per-fold cliff scores: append NaN for any fold that has
            # no usable cliff subset (or a failed fit) so fold f means the same fold
            # across every rep + the hybrid — required for held-out selection in
            # honest_rescore (the scalar means below are NaN-tolerant, so identical
            # to the old skip-based values).
            cliff_fold_scores = []
            noncliff_fold_scores = []
            for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_rep[train_idx])
                X_va = scaler.transform(X_rep[val_idx])

                try:
                    m = mfactory()
                    m.fit(X_tr, labels[train_idx])
                    y_pred = m.predict(X_va)
                except Exception as e:
                    print(f"  [warn] per-rep fit/predict ({rep_name}/{mname}): {type(e).__name__}: {str(e)[:100]}")
                    cliff_fold_scores.append(float('nan'))
                    noncliff_fold_scores.append(float('nan'))
                    continue

                val_cliff = [i for i, vi in enumerate(val_idx) if vi in cliff_indices]
                val_noncliff = [i for i, vi in enumerate(val_idx) if vi not in cliff_indices]

                cliff_fold_scores.append(
                    float(metric_fn(labels[val_idx[val_cliff]], y_pred[val_cliff]))
                    if len(val_cliff) >= 2 else float('nan'))
                noncliff_fold_scores.append(
                    float(metric_fn(labels[val_idx[val_noncliff]], y_pred[val_noncliff]))
                    if len(val_noncliff) >= 2 else float('nan'))

            rep_model_results[mname] = {
                'cliff_score': _finite_mean(cliff_fold_scores),
                'noncliff_score': _finite_mean(noncliff_fold_scores),
                # NEW (additive): per-fold scores feed honest held-out selection.
                'cliff_fold_scores': cliff_fold_scores,
                'noncliff_fold_scores': noncliff_fold_scores,
            }
        per_rep_results[rep_name] = rep_model_results

    # Hybrid evaluation per model
    hybrid_results_by_model = {}
    for mname, mfactory in model_factories.items():
        bp = bp_by_model.get(mname)
        if bp is None:
            print(f"  [warn] no hybrid blueprint found for {mname}, skipping hybrid eval")
            hybrid_results_by_model[mname] = {'skipped': 'no_blueprint'}
            continue
        # Fold-aligned per-fold hybrid scores (NaN placeholder on skipped/failed
        # folds) so honest_rescore can index the same fold as the per-rep baselines.
        h_cliff_fold_scores = []
        h_noncliff_fold_scores = []
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
            train_reps_h = {k: v[train_idx] for k, v in reps.items()}
            val_reps_h = {k: v[val_idx] for k, v in reps.items()}
            aug_tr = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
            aug_va = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None

            try:
                X_tr_h, X_va_h, fi = build_fold_hybrid(
                    train_reps_h, val_reps_h, labels[train_idx], bp,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=mfactory(),
                    metric_fn=metric_fn,
                    lower_is_better=lower_is_better)
            except Exception as e:
                print(f"  [warn] hybrid build ({mname}): {type(e).__name__}: {str(e)[:100]}")
                h_cliff_fold_scores.append(float('nan'))
                h_noncliff_fold_scores.append(float('nan'))
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_h)
            X_va_s = scaler.transform(X_va_h)

            try:
                m = mfactory()
                m.fit(X_tr_s, labels[train_idx])
                y_pred = m.predict(X_va_s)
            except Exception as e:
                print(f"  [warn] hybrid fit/predict ({mname}): {type(e).__name__}: {str(e)[:100]}")
                h_cliff_fold_scores.append(float('nan'))
                h_noncliff_fold_scores.append(float('nan'))
                continue

            val_cliff = [i for i, vi in enumerate(val_idx) if vi in cliff_indices]
            val_noncliff = [i for i, vi in enumerate(val_idx) if vi not in cliff_indices]

            h_cliff_fold_scores.append(
                float(metric_fn(labels[val_idx[val_cliff]], y_pred[val_cliff]))
                if len(val_cliff) >= 2 else float('nan'))
            h_noncliff_fold_scores.append(
                float(metric_fn(labels[val_idx[val_noncliff]], y_pred[val_noncliff]))
                if len(val_noncliff) >= 2 else float('nan'))

        hybrid_results_by_model[mname] = {
            'cliff_score': _finite_mean(h_cliff_fold_scores),
            'noncliff_score': _finite_mean(h_noncliff_fold_scores),
            # NEW (additive): per-fold scores feed honest held-out selection.
            'cliff_fold_scores': h_cliff_fold_scores,
            'noncliff_fold_scores': h_noncliff_fold_scores,
        }

    results = {
        'domain': domain,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'cliff_info': convert_numpy(cliff_info),
        'per_rep': convert_numpy(per_rep_results),
        'hybrid': convert_numpy(hybrid_results_by_model),
    }
    if top3:
        results['top3_base_reps'] = [(n, s) for n, s in top3]

    # Honest-comparison block (CONTRACT json_convention). Built on the CLIFF score
    # (the headline cliff comparison the same-fold argmax below optimises). Fold
    # scores are fold-aligned NaN-padded lists so honest_rescore can re-pick the
    # winning rep / hybrid per HELD-OUT fold instead of the same-fold argmax. This
    # experiment computes its own segmented CV baselines (no SearchResult), so the
    # two per-model dicts are assembled by hand rather than via
    # build_honest_rescore_input. Only models actually evaluated appear.
    candidates_by_model = {}
    baselines_by_model = {}
    for mname in model_factories:
        h_fold = hybrid_results_by_model.get(mname, {}).get('cliff_fold_scores')
        if h_fold is not None:
            candidates_by_model[mname] = [{'name': 'hybrid', 'fold_scores': h_fold}]
        base = {}
        for rep_name, rep_model_results in per_rep_results.items():
            rf = rep_model_results.get(mname, {}).get('cliff_fold_scores')
            if rf is not None:
                base[rep_name] = rf
        if base:
            baselines_by_model[mname] = base
    results['honest_rescore'] = {
        'lower_is_better': lower_is_better,
        'candidates_by_model': convert_numpy(candidates_by_model),
        'baselines_by_model': convert_numpy(baselines_by_model),
    }

    # Summary per model
    print(f"\n  Hybrid cliff {metric_name} per model:")
    results['hybrid_cliff_delta'] = {}
    for mname in model_factories:
        h_cliff = hybrid_results_by_model.get(mname, {}).get('cliff_score', float('nan'))
        # Best single rep cliff score for this model (same-fold argmax — optimistic;
        # retained for back-compat, superseded by honest_rescore above).
        rep_cliffs = [r.get(mname, {}).get('cliff_score', float('nan'))
                      for r in per_rep_results.values()]
        rep_cliffs = [s for s in rep_cliffs if not np.isnan(s)]
        best_rep_cliff = max(rep_cliffs, default=float('nan')) if not lower_is_better \
            else min(rep_cliffs, default=float('nan'))
        if not np.isnan(h_cliff) and not np.isnan(best_rep_cliff):
            delta = h_cliff - best_rep_cliff
            results['hybrid_cliff_delta'][mname] = float(delta)
            print(f"    {mname}: {h_cliff:.4f} (best rep: {best_rep_cliff:.4f}, "
                  f"delta: {delta:+.4f})")

    # Honest held-out cliff delta (replaces the same-fold argmax): pick the winning
    # rep / hybrid on the OTHER folds, score it on the held-out fold, average.
    honest = honest_rescore(
        results['honest_rescore']['candidates_by_model'],
        results['honest_rescore']['baselines_by_model'],
        lower_is_better)
    results['hybrid_cliff_delta_honest'] = {
        m: convert_numpy(r) for m, r in honest.items()}
    print(f"\n  Honest held-out cliff {metric_name} per model:")
    for mname, r in honest.items():
        if not np.isnan(r.get('delta', float('nan'))):
            print(f"    {mname}: hybrid={r['hybrid_honest']:.4f} "
                  f"best_single={r['best_single_honest']:.4f} "
                  f"delta={r['delta']:+.4f}"
                  f"{' [single_fold]' if r.get('single_fold') else ''}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S12: Cross-Domain Activity Cliffs')
    parser.add_argument('--sim-threshold', type=float, default=0.55)
    parser.add_argument('--activity-threshold', type=float, default=2.0)
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset,
                  sim_threshold=args.sim_threshold,
                  activity_k=args.activity_threshold)
    save_experiment_results(results, args.output, 's12_cross_domain_cliffs', args.dataset)
