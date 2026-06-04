#!/usr/bin/env python3
"""N04: GP Leave-One-Out Cliff-Aware Noise Detection.

Paper: Noise (KILLER EXPERIMENT)
Core thesis: In hybrid space, GP-LOO predictive probability distinguishes genuine
    noise from activity cliffs. In standard fingerprint/concat space, it cannot —
    because distances don't track activity, cliffs and noise look identical.

Steps:
  1. Build hybrid representation
  2. Fit GP in hybrid space
  3. LOO predictions → flag samples with low predictive probability
  4. Validate via synthetic noise injection: are injected-noise samples detected
     more often than genuine cliff compounds?

Comparison conditions:
  - GP-LOO in hybrid space (MAIN — should have lowest cliff FP rate).
    Hybrid is built using downstream_model='rf' (the noise precompute pool
    excludes GP); GP-LOO is then applied downstream to the RF-optimised
    feature space.
  - GP-LOO in full concat space (baseline)
  - GP-LOO in PCA-dim-matched space (n_components = budget, NOT 50 — earlier
    docstring said "PCA-50" but the code uses min(budget, total_dim, n-1)
    which matches the hybrid budget rather than a fixed 50)
  - GP-LOO in each individual base rep space (per-rep baselines)
  - ActivityCliffAwareEstimator (library, non-GP)
  - Prediction error filter (naive baseline)

Usage:
    python n04_gp_loo_detection.py --dataset logd --mode intermediary
    python n04_gp_loo_detection.py --dataset flab_thermo_tresanco --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold, KFold

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_fitted_hybrid, get_fitted_pool_hybrid, extract_top3_base_reps,
    cap_dimensions,
)
from kirby.utils.noise_estimation import ActivityCliffAwareEstimator
from kirby.utils.noise_mitigation_methods import compute_distance_matrix


# Maximum samples for GP (O(n³))
GP_MAX_SAMPLES = 1000

# Multi-seed noise averaging. Single-seed numbers are point estimates with
# no variance over noise realizations — easy to misread when a method's
# detection rate happens to be sensitive to which subset of samples got
# corrupted. Aggregate across NOISE_SEEDS and report mean ± std.
NOISE_SEEDS = [0, 1, 2]

# Activity-cliff percentile knob sweep. "Structurally similar" = the closest
# `similar_percentile`% of off-diagonal Euclidean pairs. The main analysis
# uses PRIMARY_AC_PERCENTILE; the additive `ac_percentile_sweep` output
# reports cliff prevalence + the cliff false-positive win-rate at EACH knob
# so the user can pick a prevalence after the ARC rerun. The cliff mask is a
# fixed structural property of (X, y) and is INDEPENDENT of the noise-injection
# loop, so the sweep reuses the already-computed per-(fraction,seed,space)
# `flagged` masks rather than re-running any GP-LOO scoring.
AC_PERCENTILES = [1.0, 2.0, 5.0, 10.0]
PRIMARY_AC_PERCENTILE = 2.0


def _subsample_if_needed(X, y, max_n, seed=42):
    """Subsample to max_n if needed. Returns (X_sub, y_sub, idx)."""
    if len(y) <= max_n:
        return X, y, np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), max_n, replace=False)
    return X[idx], y[idx], idx


def _identify_cliffs_euclidean(X, y, similar_percentile=PRIMARY_AC_PERCENTILE,
                               activity_k=2.0, is_clf=False):
    """Identify activity cliff compounds using Euclidean distance.

    A cliff compound is one that has at least one neighbour that is
    structurally SIMILAR (small Euclidean distance) but has a very DIFFERENT
    activity. Similarity is sim = 1/(1+D).

    BUG FIX (2026-06-03): the previous version used an absolute similarity
    cutoff of 0.55. On standardised high-dimensional reps, sim = 1/(1+D)
    never approaches 0.55 (observed max ≈ 0.14 on buchwald_hartwig concat),
    so the cutoff matched zero pairs and n_cliffs was always 0 — which made
    the whole cliff-false-positive thesis (and the win-rate summary) degenerate.

    Fix mirrors n01.identify_cliff_pairs / s12.identify_cliff_pairs_euclidean:
    define "structurally similar" as the top ``similar_percentile``% most-
    similar off-diagonal pairs (distribution-relative), not an absolute cutoff.
    """
    D = compute_distance_matrix(X, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    if is_clf:
        delta_y = (y[:, None] != y[None, :]).astype(float)
        activity_threshold = 0.5
    else:
        mad_y = float(np.median(np.abs(y - np.median(y))))
        activity_threshold = activity_k * mad_y
        delta_y = np.abs(y[:, None] - y[None, :])
    # Distribution-relative similarity cutoff: the top similar_percentile%
    # most-similar off-diagonal pairs count as "structurally similar".
    n = len(y)
    off = sim[~np.eye(n, dtype=bool)]
    sim_cutoff = float(np.percentile(off, 100.0 - similar_percentile)) \
        if off.size else 1.0
    cliff_mask = ((sim >= sim_cutoff) & (delta_y > activity_threshold)).any(axis=1)
    return cliff_mask


def _gp_loo_scores(X, y):
    """Compute GP LOO noise scores via closed-form LOO (Rasmussen & Williams §5.4.2).

    Closed-form LOO avoids refitting n times. Given the trained GP kernel matrix
    K and its inverse K_inv:
        LOO_residual_i = (K_inv @ y)[i] / K_inv[i,i]
        LOO_variance_i = 1 / K_inv[i,i]
        score_i = |LOO_residual_i| / sqrt(LOO_variance_i)

    Score = standardised LOO residual. Higher = more likely noise.
    """
    n = len(y)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    kernel = ConstantKernel() * RBF() + WhiteKernel()
    scores = np.zeros(n)
    gp_failures = 0

    # Fit GP to learn hyperparameters
    try:
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                      random_state=42, alpha=1e-6)
        gp.fit(X_s, y)
        learned_kernel = gp.kernel_
    except Exception:
        learned_kernel = kernel
        gp_failures += 1

    # Closed-form LOO via kernel matrix inverse
    try:
        K = learned_kernel(X_s) + 1e-6 * np.eye(n)
        K_inv = np.linalg.inv(K)
        alpha_vec = K_inv @ y  # K_inv @ y, not the regularization alpha

        K_inv_diag = np.diag(K_inv)
        K_inv_diag = np.maximum(K_inv_diag, 1e-10)  # numerical safety

        # LOO residual and variance (Rasmussen & Williams eq. 5.12)
        loo_residual = alpha_vec / K_inv_diag
        loo_variance = 1.0 / K_inv_diag
        loo_std = np.sqrt(np.maximum(loo_variance, 1e-10))

        scores = np.abs(loo_residual) / loo_std
    except (np.linalg.LinAlgError, Exception) as e:
        print(f"    GP closed-form LOO failed: {e}")
        gp_failures += 1
        # Fallback: RF out-of-bag predictions (proper held-out estimate)
        rf = RandomForestRegressor(n_estimators=200, oob_score=True,
                                   random_state=42, n_jobs=-1)
        rf.fit(X_s, y)
        oob_pred = rf.oob_prediction_
        oob_residual = np.abs(y - oob_pred)
        scores = oob_residual / (np.std(oob_residual) + 1e-10)

    return scores, gp_failures


def _inject_noise(y, fraction, seed=42):
    """Inject noise into fraction of labels. Returns (y_noisy, noise_mask)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    n_noisy = max(1, int(n * fraction))
    noise_idx = rng.choice(n, n_noisy, replace=False)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[noise_idx] = True

    y_noisy = y.copy()
    y_range = np.max(y) - np.min(y)
    y_noisy[noise_idx] += rng.normal(0, y_range * 0.3, n_noisy)

    return y_noisy, noise_mask


def run(data, config, output_dir, dataset_name, **kwargs):
    """N04: GP-LOO cliff-aware noise detection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N04: GP-LOO CLIFF-AWARE NOISE DETECTION")
    print("=" * 70)

    task_type = data['task_type']
    is_clf = task_type == 'classification'

    if is_clf:
        print("  SKIP: N04 is GP-based (regression-only). Classification not supported.")
        return {'skipped': True, 'reason': 'GP-LOO is regression-only'}

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    # Get the domain metric NAME for the results header; the fn/direction
    # aren't used here (this experiment reports precision/recall/cliff_FP,
    # not the domain metric). Expose metric_name in the returned summary
    # so downstream readers know which dataset metric the run targeted.
    metric_name, _metric_fn, lower_is_better = get_domain_metric(data)

    noise_fractions = kwargs.get('noise_fractions', [0.05, 0.1, 0.2])
    budget = kwargs.get('budget', 100)

    # Build feature spaces
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    total_dim = X_concat.shape[1]

    spaces = {}

    # 1. Hybrid. We previously requested downstream_model='gp', but the noise
    # precompute pool builds blueprints only for DOWNSTREAM_MODELS_NOISE
    # (rf, xgboost, ridge, svm, mlp, lightgbm) — gp was never present, so the
    # pool returned no candidate and the hybrid was silently absent on every
    # dataset. GP-LOO is downstream of the hybrid feature space, not part of
    # it; an RF-optimised hybrid is the correct feature space to score with
    # GP. Track whether the hybrid actually built so the summary can flag
    # missing-hybrid runs instead of defaulting to the 1.0 sentinel.
    _search = None
    hybrid_available = False
    try:
        X_hybrid, fi, _search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='rf', budget=budget,
            augmentations=augmentations,
            speed=config.get('search_speed', 'full'),
            data=data,
            dataset_name=dataset_name,
            task_type=task_type)
        spaces['hybrid'] = X_hybrid
        hybrid_available = True
        top3 = extract_top3_base_reps(_search)
        if top3:
            print(f"  Top-3 base reps (ranked by RF baseline): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        print(f"  Hybrid creation failed: {e}")

    # 2. Concat
    spaces['concat'] = X_concat

    # 3. PCA-matched
    pca_dim = min(budget, total_dim, len(labels) - 1)
    if pca_dim >= 2:
        spaces['pca_matched'] = PCA(n_components=pca_dim, random_state=42).fit_transform(
            StandardScaler().fit_transform(X_concat))

    # 4. Per-rep spaces (individual base reps)
    for rep_name, X_rep in reps.items():
        spaces[rep_name] = cap_dimensions(StandardScaler().fit_transform(X_rep))[0]

    print(f"  Spaces: {list(spaces.keys())}")
    print(f"  Noise fractions: {noise_fractions}")

    # Identify cliffs in the original data (using concat Euclidean). The cliff
    # mask is a fixed structural property of (X_sub, y_sub) and does NOT depend
    # on the noise-injection loop below, so we compute one mask PER percentile
    # knob up front and reuse them in the additive sweep. The PRIMARY knob
    # drives the headline `cliff_mask` / `n_cliffs` used by the legacy summary.
    X_sub, y_sub, sub_idx = _subsample_if_needed(X_concat, labels, GP_MAX_SAMPLES)
    cliff_masks_by_pct = {
        pct: _identify_cliffs_euclidean(X_sub, y_sub, similar_percentile=pct)
        for pct in AC_PERCENTILES
    }
    # Ensure the primary knob has a mask even if it isn't in AC_PERCENTILES.
    if PRIMARY_AC_PERCENTILE not in cliff_masks_by_pct:
        cliff_masks_by_pct[PRIMARY_AC_PERCENTILE] = _identify_cliffs_euclidean(
            X_sub, y_sub, similar_percentile=PRIMARY_AC_PERCENTILE)
    cliff_mask = cliff_masks_by_pct[PRIMARY_AC_PERCENTILE]
    n_cliffs = int(cliff_mask.sum())
    print(f"  Cliff compounds (primary {PRIMARY_AC_PERCENTILE}%): "
          f"{n_cliffs}/{len(y_sub)} ({100*n_cliffs/len(y_sub):.1f}%)")
    for pct in AC_PERCENTILES:
        nc = int(cliff_masks_by_pct[pct].sum())
        print(f"    cliff knob {pct:>5.1f}%: {nc}/{len(y_sub)} "
              f"({100*nc/len(y_sub):.1f}%)")

    top3 = extract_top3_base_reps(_search) if _search is not None else None
    results = {
        'metric': metric_name,
        # Direction of the dataset's domain metric. This experiment reports
        # detection precision/recall/cliff_FP (not the domain metric itself),
        # but downstream readers that key on a standard envelope expect a
        # lower_is_better flag; surface it here. n04 deliberately returns a flat
        # dict (matching n05/n07-n13, which also do NOT use
        # make_structured_output) rather than the structured wrapper used only
        # by n01/n02/n03.
        'lower_is_better': bool(lower_is_better),
        'n_samples': len(labels),
        'n_subsampled': len(y_sub),
        'n_cliffs': n_cliffs,
        'pct_cliffs': float(100 * n_cliffs / len(y_sub)),
        'ac_percentiles': AC_PERCENTILES,
        'primary_ac_percentile': PRIMARY_AC_PERCENTILE,
        'noise_fractions': noise_fractions,
        'spaces': list(spaces.keys()),
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        'per_noise_level': {},
    }

    # Retained per-(fraction, seed, condition) `flagged` + `noise_mask` so the
    # additive cliff-percentile sweep can recompute cliff_fp_rate per knob
    # WITHOUT re-running GP-LOO. Structure: flagged_records[(frac, cond)] is a
    # list (one per seed) of {'flagged': bool[n_sub], 'noise_mask': bool[n_sub]}.
    flagged_records = {}

    for frac in noise_fractions:
        print(f"\n  === Noise fraction: {frac:.0%} ===")
        # Per-seed accumulators: {space_name_or_method: [list_of_metric_dicts]}
        per_seed_acc = {}

        for noise_seed in NOISE_SEEDS:
            y_noisy, noise_mask_full = _inject_noise(labels, frac, seed=noise_seed)

            # Subsample noise mask
            noise_mask = noise_mask_full[sub_idx]
            y_noisy_sub = y_noisy[sub_idx]
            n_injected = int(noise_mask.sum())
            print(f"    [seed={noise_seed}] Injected: {n_injected} noisy samples")

            level_results = {}

            # GP-LOO conditions
            for space_name, X_full in spaces.items():
                X_space_sub = X_full[sub_idx] if len(X_full) > len(sub_idx) else X_full
                if X_space_sub.shape[0] != len(y_noisy_sub):
                    continue

                print(f"      {space_name} (dim={X_space_sub.shape[1]})...", end='', flush=True)
                scores, gp_failures = _gp_loo_scores(X_space_sub, y_noisy_sub)

                threshold_pct = min(frac * 2, 0.3)
                threshold = np.percentile(scores, (1 - threshold_pct) * 100)
                flagged = scores >= threshold

                # Retain for the cliff-percentile sweep (cliff-independent).
                flagged_records.setdefault((frac, space_name), []).append(
                    {'flagged': flagged.copy(), 'noise_mask': noise_mask.copy()})

                tp = int((flagged & noise_mask).sum())
                fp_cliff = int((flagged & cliff_mask & ~noise_mask).sum())
                fp_other = int((flagged & ~noise_mask & ~cliff_mask).sum())
                fn = int((~flagged & noise_mask).sum())

                precision = tp / max(flagged.sum(), 1)
                recall = tp / max(noise_mask.sum(), 1)
                cliff_fp_rate = fp_cliff / max(cliff_mask.sum(), 1)

                level_results[space_name] = {
                    'gp_failures': gp_failures,
                    'n_flagged': int(flagged.sum()),
                    'true_positives': tp,
                    'false_positives_cliff': fp_cliff,
                    'false_positives_other': fp_other,
                    'false_negatives': fn,
                    'precision': float(precision),
                    'recall': float(recall),
                    'cliff_fp_rate': float(cliff_fp_rate),
                }
                print(f" precision={precision:.3f}, recall={recall:.3f}, "
                      f"cliff_FP={cliff_fp_rate:.3f}")

            # ActivityCliffAwareEstimator condition (non-GP)
            if 'hybrid' in spaces:
                X_h_sub = spaces['hybrid'][sub_idx] if len(spaces['hybrid']) > len(sub_idx) else spaces['hybrid']
                if X_h_sub.shape[0] == len(y_noisy_sub):
                    print(f"      cliff_aware_estimator...", end='', flush=True)
                    try:
                        estimator = ActivityCliffAwareEstimator(k=10, metric='cosine')
                        estimator.fit(StandardScaler().fit_transform(X_h_sub), y_noisy_sub)
                        scores_ca = estimator.noise_scores_
                        threshold_ca = np.percentile(scores_ca, (1 - threshold_pct) * 100)
                        flagged_ca = scores_ca >= threshold_ca

                        flagged_records.setdefault(
                            (frac, 'cliff_aware_estimator'), []).append(
                            {'flagged': flagged_ca.copy(),
                             'noise_mask': noise_mask.copy()})

                        tp_ca = int((flagged_ca & noise_mask).sum())
                        fp_cliff_ca = int((flagged_ca & cliff_mask & ~noise_mask).sum())
                        precision_ca = tp_ca / max(flagged_ca.sum(), 1)
                        recall_ca = tp_ca / max(noise_mask.sum(), 1)
                        cliff_fp_ca = fp_cliff_ca / max(cliff_mask.sum(), 1)

                        level_results['cliff_aware_estimator'] = {
                            'n_flagged': int(flagged_ca.sum()),
                            'true_positives': tp_ca,
                            'false_positives_cliff': fp_cliff_ca,
                            'precision': float(precision_ca),
                            'recall': float(recall_ca),
                            'cliff_fp_rate': float(cliff_fp_ca),
                        }
                        print(f" precision={precision_ca:.3f}, recall={recall_ca:.3f}, "
                              f"cliff_FP={cliff_fp_ca:.3f}")
                    except Exception as e:
                        print(f" FAILED: {e}")

            # Prediction error baseline
            print(f"      prediction_error...", end='', flush=True)
            X_concat_sub = X_concat[sub_idx] if len(X_concat) > len(sub_idx) else X_concat
            scaler = StandardScaler()
            X_cs = scaler.fit_transform(X_concat_sub)
            rf = RandomForestRegressor(n_estimators=100,
                                       random_state=42 + noise_seed, n_jobs=-1)
            if groups is not None:
                groups_sub = np.asarray(groups)[sub_idx]
                n_groups = len(np.unique(groups_sub))
                if n_groups >= 5:
                    cv_pe = GroupKFold(n_splits=5)
                    y_pred_cv = cross_val_predict(rf, X_cs, y_noisy_sub,
                                                  groups=groups_sub, cv=cv_pe)
                elif n_groups >= 2:
                    cv_pe = GroupKFold(n_splits=n_groups)
                    y_pred_cv = cross_val_predict(rf, X_cs, y_noisy_sub,
                                                  groups=groups_sub, cv=cv_pe)
                else:
                    # n_groups < 2 — GroupKFold can't split a single group.
                    # Fall back to plain KFold (no shuffle for reproducibility)
                    # and surface a warning.
                    print("      WARNING: only 1 distinct group in subsample; "
                          "GroupKFold cannot split — falling back to plain 5-fold KFold")
                    y_pred_cv = cross_val_predict(
                        rf, X_cs, y_noisy_sub,
                        cv=KFold(n_splits=5, shuffle=False))
            else:
                print("      WARNING: no groups available; using plain 5-fold KFold")
                y_pred_cv = cross_val_predict(rf, X_cs, y_noisy_sub,
                                              cv=KFold(n_splits=5, shuffle=False))
            pe_scores = np.abs(y_noisy_sub - y_pred_cv)
            threshold_pe = np.percentile(pe_scores, (1 - threshold_pct) * 100)
            flagged_pe = pe_scores >= threshold_pe

            flagged_records.setdefault((frac, 'prediction_error'), []).append(
                {'flagged': flagged_pe.copy(), 'noise_mask': noise_mask.copy()})

            tp_pe = int((flagged_pe & noise_mask).sum())
            fp_cliff_pe = int((flagged_pe & cliff_mask & ~noise_mask).sum())
            precision_pe = tp_pe / max(flagged_pe.sum(), 1)
            recall_pe = tp_pe / max(noise_mask.sum(), 1)
            cliff_fp_pe = fp_cliff_pe / max(cliff_mask.sum(), 1)

            level_results['prediction_error'] = {
                'n_flagged': int(flagged_pe.sum()),
                'true_positives': tp_pe,
                'false_positives_cliff': fp_cliff_pe,
                'precision': float(precision_pe),
                'recall': float(recall_pe),
                'cliff_fp_rate': float(cliff_fp_pe),
            }
            print(f" precision={precision_pe:.3f}, recall={recall_pe:.3f}, "
                  f"cliff_FP={cliff_fp_pe:.3f}")

            # Accumulate this seed's per-space results
            for key, entry in level_results.items():
                per_seed_acc.setdefault(key, []).append(entry)

        # Aggregate across seeds: report mean ± std of each numeric metric.
        # Sticky fields (n_flagged etc. that vary by integer count) are kept
        # as means too; if seed variance matters for that field, the std is
        # right there alongside.
        agg_level = {}
        numeric_fields = ('precision', 'recall', 'cliff_fp_rate',
                          'n_flagged', 'true_positives',
                          'false_positives_cliff', 'false_positives_other',
                          'false_negatives')
        for key, entries in per_seed_acc.items():
            agg = {'n_seeds': len(entries)}
            for fld in numeric_fields:
                vals = [e[fld] for e in entries
                        if isinstance(e, dict) and fld in e]
                if vals:
                    agg[f'{fld}_mean'] = float(np.mean(vals))
                    agg[f'{fld}_std'] = float(np.std(vals))
                    # Backwards-compat: also expose the mean under the plain
                    # field name (older consumers reading e.g. cliff_fp_rate
                    # directly continue to work, with the mean across seeds).
                    agg[fld] = float(np.mean(vals))
            # Forward gp_failures (sum across seeds, qualitative)
            if all('gp_failures' in e for e in entries):
                agg['gp_failures'] = sum(e['gp_failures'] for e in entries)
            agg_level[key] = agg

        results['per_noise_level'][str(frac)] = convert_numpy(agg_level)

    # =========================================================================
    # ADDITIVE: activity-cliff percentile knob SWEEP
    # =========================================================================
    # For EACH percentile knob, recompute the cliff false-positive rate per
    # (fraction, condition) by REUSING the retained `flagged` masks (the
    # noise-detection decision is independent of the cliff definition) against
    # that knob's cliff mask, then aggregate across seeds and compute the same
    # hybrid-win-rate the headline summary uses. No GP-LOO re-scoring happens.
    print("\n  === ac_percentile_sweep ===")
    ac_sweep = {}
    for pct in AC_PERCENTILES:
        c_mask = cliff_masks_by_pct[pct]
        n_cliffs_pct = int(c_mask.sum())
        cliff_count = max(int(c_mask.sum()), 1)

        # Per (fraction, condition): mean cliff_fp_rate across seeds.
        per_noise = {}
        for (frac, cond), records in flagged_records.items():
            rates = []
            for rec in records:
                flagged = rec['flagged']
                nmask = rec['noise_mask']
                fp_cliff = int((flagged & c_mask & ~nmask).sum())
                rates.append(fp_cliff / cliff_count)
            per_noise.setdefault(str(frac), {})[cond] = {
                'cliff_fp_rate': float(np.mean(rates)) if rates else float('nan'),
                'cliff_fp_rate_std': float(np.std(rates)) if rates else float('nan'),
                'n_seeds': len(rates),
            }

        # Win-rate: hybrid has the lowest cliff_fp_rate vs every other
        # condition, summed over all (fraction, condition) comparisons. Mirrors
        # the headline summary's logic. Degenerate (0-cliff) knobs are guarded.
        if n_cliffs_pct == 0:
            entry = {
                'similar_percentile': pct,
                'n_cliffs': 0,
                'pct_cliffs': 0.0,
                'cliffs_degenerate': True,
                'win_rate': None,
                'hybrid_lowest_cliff_fp_rate': None,
                'total_comparisons': 0,
                'per_noise_level': per_noise,
                'note': ('No cliff compounds at this knob: every condition has '
                         'cliff_fp_rate=0, win-rate is a meaningless 0<=0 tie.'),
            }
        elif not hybrid_available:
            entry = {
                'similar_percentile': pct,
                'n_cliffs': n_cliffs_pct,
                'pct_cliffs': float(100 * n_cliffs_pct / len(y_sub)),
                'cliffs_degenerate': False,
                'win_rate': None,
                'hybrid_lowest_cliff_fp_rate': None,
                'total_comparisons': 0,
                'per_noise_level': per_noise,
                'note': 'Hybrid not built; win-rate intentionally not reported.',
            }
        else:
            hybrid_wins = 0
            total_comparisons = 0
            for frac_key, level_res in per_noise.items():
                h_entry = level_res.get('hybrid')
                if h_entry is None or np.isnan(h_entry['cliff_fp_rate']):
                    continue
                h_cfp = h_entry['cliff_fp_rate']
                for cond, cond_res in level_res.items():
                    if cond == 'hybrid' or np.isnan(cond_res['cliff_fp_rate']):
                        continue
                    total_comparisons += 1
                    if h_cfp <= cond_res['cliff_fp_rate']:
                        hybrid_wins += 1
            entry = {
                'similar_percentile': pct,
                'n_cliffs': n_cliffs_pct,
                'pct_cliffs': float(100 * n_cliffs_pct / len(y_sub)),
                'cliffs_degenerate': False,
                'hybrid_lowest_cliff_fp_rate': hybrid_wins,
                'total_comparisons': total_comparisons,
                'win_rate': (float(hybrid_wins / total_comparisons)
                             if total_comparisons > 0 else None),
                'per_noise_level': per_noise,
            }
        ac_sweep[str(pct)] = convert_numpy(entry)
        _wr = entry['win_rate']
        print(f"    knob {pct:>5.1f}%: n_cliffs={n_cliffs_pct:>4} "
              f"({entry['pct_cliffs']:.1f}%)  "
              f"win_rate={'n/a' if _wr is None else f'{_wr:.3f}'}")

    results['ac_percentile_sweep'] = ac_sweep

    # Summary: does hybrid have the lowest cliff FP rate across noise levels?
    # Previously the missing-hybrid case fell through `.get('hybrid', {}).get(
    # 'cliff_fp_rate', 1.0)` which is a VALID value (cliff_fp_rate ∈ [0, 1]),
    # not a sentinel — making the win/loss count meaningless when the hybrid
    # build had silently failed. Now we explicitly skip the win-rate
    # computation in that case and flag the run.
    results['hybrid_available'] = bool(hybrid_available)
    # Self-diagnosing guard: the win-rate compares cliff false-positive RATES,
    # which are all identically 0 when no cliff compounds exist. Reporting a
    # 1.0 win-rate from `0 <= 0` comparisons is a false positive (this is
    # exactly the degeneracy the 2026-06-03 cliff-cutoff fix targets). Flag it.
    results['cliffs_degenerate'] = bool(n_cliffs == 0)
    if not hybrid_available:
        results['summary'] = {
            'hybrid_available': False,
            'note': ('Hybrid was not built (pool returned no candidate for the '
                     'requested downstream model); win-rate intentionally not '
                     'reported.'),
        }
        print(f"\n  Hybrid unavailable — win-rate not reported.")
    elif n_cliffs == 0:
        results['summary'] = {
            'hybrid_available': True,
            'cliffs_degenerate': True,
            'win_rate': None,
            'note': ('No cliff compounds identified (n_cliffs=0): every '
                     'condition has cliff_fp_rate=0, so the win-rate would be '
                     'a meaningless 0<=0 tie. Win-rate intentionally not '
                     'reported. If this recurs after the cliff-cutoff fix, the '
                     'dataset genuinely has no near-duplicate/different-activity '
                     'pairs at the top-2%% similarity cutoff.'),
        }
        print(f"\n  No cliffs identified — win-rate not reported (degenerate).")
    else:
        hybrid_wins = 0
        total_comparisons = 0
        for frac_key, level_res in results['per_noise_level'].items():
            h_entry = level_res.get('hybrid')
            if h_entry is None or 'cliff_fp_rate' not in h_entry:
                continue
            h_cfp = h_entry['cliff_fp_rate']
            for cond, cond_res in level_res.items():
                if cond == 'hybrid':
                    continue
                if not isinstance(cond_res, dict) or 'cliff_fp_rate' not in cond_res:
                    continue
                c_cfp = cond_res['cliff_fp_rate']
                total_comparisons += 1
                if h_cfp <= c_cfp:
                    hybrid_wins += 1

        results['summary'] = {
            'hybrid_available': True,
            'hybrid_lowest_cliff_fp_rate': hybrid_wins,
            'total_comparisons': total_comparisons,
            'win_rate': (float(hybrid_wins / total_comparisons)
                         if total_comparisons > 0 else None),
        }
        print(f"\n  Hybrid lowest cliff FP rate: {hybrid_wins}/{total_comparisons}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N04: GP-LOO Noise Detection')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n04_gp_loo_detection', args.dataset)
