#!/usr/bin/env python3
"""N06: BindingDB Ground-Truth Noise — replicate disagreement as real-world noise labels.

Paper: Noise (genuinely novel contribution)

BindingDB contains the SAME (drug, target) pair measured many times (different
labs, assays, days). When those repeat measurements disagree, that disagreement
is *real* assay noise — not synthetic noise we injected. N06 uses it as a
ground-truth label for "how noisy is this measurement" and asks: do our noise
estimators rank samples by estimated noise in a way that agrees with the true
replicate disagreement?

--- HOW THE GROUND TRUTH IS BUILT ---
A "replicate group" is the set of rows that share the same drug AND the same
target. For each replicate group with >= 2 measurements we compute the standard
deviation of its labels (pKd units). That per-group standard deviation is the
ground-truth noise magnitude, assigned to every row in the group. Rows that
appear only once have no ground truth (replicate_std = 0, has_replicate=False)
and are EXCLUDED from the correlation against the estimators.

The replicate IDs come from the loader's metadata: `metadata['drug_ids']` and
`metadata['protein_ids']`. The TDC BindingDB_Kd dump retains repeat measurements
of the same pair (verified locally: 3,487 pairs with disagreeing replicates over
~9.7k rows, mean std 0.99 pKd), so the replicate signal is genuinely present —
it is simply not pre-exposed as `data['replicate_groups']`.

--- HISTORY (2026-05-25 audit) ---
A previous version of this script reused `data['groups']` (the cold-target CV
grouping = one group per protein) as if it were a replicate ID. That measured
inter-DRUG variance within a protein (different drugs legitimately have
different Kd), which is biological signal, not assay noise. The experiment ran
without error but every number was meaningless. The fix (2026-06) is to build
replicate groups from (drug_id, target_id) pairs instead, which is the only
grouping under which repeated rows are genuine remeasurements of the same thing.

--- WHAT IT REPORTS ---
For each estimator, Spearman correlation between its per-sample noise score and
the per-sample ground-truth replicate std, computed OVER REPLICATE SAMPLES ONLY.
Higher correlation = the estimator's notion of "noisy" agrees with real assay
disagreement. Summary picks the best (estimator, feature-space) by that
correlation.

Usage:
    python n06_bindingdb_ground_truth.py --mode intermediary
    python n06_bindingdb_ground_truth.py --dataset bindingdb_kd --reps drug_ecfp4,drug_maccs,protein_aac
"""
import sys
import os
import inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_fitted_hybrid, get_fitted_pool_hybrid, extract_top3_base_reps,
)
from kirby.utils.noise_estimation import get_noise_estimator


# Estimators to benchmark (skip torch-dependent ones for screening)
ESTIMATORS_SCREENING = [
    'neighbor_residual',
    'local_variance_ratio',
    'distance_weighted_residual',
    'activity_cliff_aware',
    'cross_val_residual',
    'ensemble_disagreement',
    'leave_one_out',
    'gaussian_process',
    'conformal',
    'bayesian_rf',
]

# GP max samples
GP_MAX = 1000

# A replicate group needs at least this many measurements to define a std.
MIN_REPLICATES = 2

# Minimum number of replicate samples required to run the ground-truth
# correlation (below this, Spearman is too unstable to report).
MIN_GROUND_TRUTH_SAMPLES = 10


def build_replicate_groups(data):
    """Derive per-(drug, target) replicate group IDs from loader metadata.

    A replicate = rows sharing the same drug_id AND target_id. These are repeat
    measurements of the identical compound-target pair, so disagreement among
    their labels is genuine assay noise.

    Returns an integer array of group IDs (one per sample), or None if the
    metadata needed to identify replicates is absent.
    """
    # An explicitly-provided replicate grouping always wins.
    explicit = data.get('replicate_groups')
    if explicit is not None:
        return np.asarray(explicit)

    meta = data.get('metadata') or {}
    drug_ids = meta.get('drug_ids')
    protein_ids = meta.get('protein_ids')
    if drug_ids is None or protein_ids is None:
        return None

    n = len(data['labels'])
    if len(drug_ids) != n or len(protein_ids) != n:
        return None

    pair_keys = [f"{d}|{p}" for d, p in zip(drug_ids, protein_ids)]
    key_to_id = {}
    group_ids = np.empty(n, dtype=np.int64)
    for i, key in enumerate(pair_keys):
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        group_ids[i] = key_to_id[key]
    return group_ids


def compute_replicate_stds(labels, groups,
                           min_replicates=MIN_REPLICATES):
    """Ground-truth noise magnitude per sample = std of its replicate group.

    For every group with >= min_replicates measurements, the noise magnitude is
    the (population) standard deviation of that group's labels, broadcast to
    each member. Singletons get 0.0 and are flagged has_replicate=False.

    Returns:
        replicate_stds: (n,) float array, per-sample ground-truth noise.
        has_replicate:  (n,) bool array, True where a real replicate std exists.
    """
    labels = np.asarray(labels, dtype=np.float64)
    n = len(labels)
    replicate_stds = np.zeros(n, dtype=np.float64)
    has_replicate = np.zeros(n, dtype=bool)
    if groups is None:
        return replicate_stds, has_replicate

    groups = np.asarray(groups)
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() >= min_replicates:
            replicate_stds[mask] = float(np.std(labels[mask]))
            # Only flag as ground-truth where the replicates actually disagree;
            # a group whose repeats are byte-identical (std == 0) carries no
            # information about how well an estimator ranks noise.
            if np.std(labels[mask]) > 0:
                has_replicate[mask] = True
    return replicate_stds, has_replicate


def run(data, config, output_dir, dataset_name, **kwargs):
    """N06: BindingDB ground-truth noise."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N06: BINDINGDB GROUND-TRUTH NOISE")
    print("=" * 70)

    reps = data['reps']
    labels = np.asarray(data['labels'], dtype=np.float64)
    n_samples = len(labels)

    # Build replicate groups from (drug, target) IDs. data['groups'] is the
    # cold-target CV grouping (one group per protein) and must NOT be used here
    # — see the 2026-05-25 audit note in the module docstring.
    replicate_groups = build_replicate_groups(data)
    if replicate_groups is None:
        raise RuntimeError(
            "N06 could not identify replicate groups. It needs either "
            "data['replicate_groups'] or metadata with 'drug_ids' and "
            "'protein_ids' (one per sample) so that repeated measurements of "
            "the same (drug, target) pair can be grouped. The current data "
            "dict exposed neither. Do NOT fall back to data['groups'] — that "
            "is the cold-target CV grouping and measures inter-drug variance, "
            "not assay replicate noise.")

    replicate_stds, has_replicate = compute_replicate_stds(
        labels, replicate_groups)
    n_with_replicates = int(has_replicate.sum())
    n_replicate_groups = int(
        sum(1 for g in np.unique(replicate_groups)
            if (replicate_groups == g).sum() >= MIN_REPLICATES))
    print(f"  {n_samples} samples, {len(np.unique(replicate_groups))} unique "
          f"(drug,target) pairs")
    print(f"  {n_replicate_groups} pairs measured >= {MIN_REPLICATES}x; "
          f"{n_with_replicates} samples sit in a disagreeing replicate group")
    if n_with_replicates:
        gt = replicate_stds[has_replicate]
        print(f"  ground-truth replicate std (pKd): "
              f"mean={gt.mean():.3f} median={np.median(gt):.3f} "
              f"max={gt.max():.3f}")

    if n_with_replicates < MIN_GROUND_TRUTH_SAMPLES:
        print(f"  Only {n_with_replicates} samples with disagreeing replicates "
              f"(< {MIN_GROUND_TRUTH_SAMPLES}) — not enough for ground-truth "
              f"correlation. Reporting estimator score distributions only.")
        replicate_stds = None
        has_replicate = None

    augmentations = data.get('augmentations')
    task_type = data['task_type']

    # Build feature spaces
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    X_concat_s = StandardScaler().fit_transform(X_concat)

    spaces = {'concat': X_concat_s}
    _search = None
    try:
        # Cold-target CV grouping (data['groups']) is the correct grouping for
        # the hybrid search — it controls leakage during the search. Replicate
        # grouping is only used for the ground-truth comparison below.
        search_groups = data.get('groups')
        X_hybrid, fi, _search, _meta = get_fitted_pool_hybrid(
            reps, labels, search_groups,
            downstream_model='rf', budget=None,
            augmentations=augmentations,
            speed=config.get('search_speed', 'full'),
            data=data,
            dataset_name=dataset_name,
            task_type=task_type)
        spaces['hybrid'] = StandardScaler().fit_transform(X_hybrid)
        top3 = extract_top3_base_reps(_search)
        if top3:
            print(f"  Top-3 base reps (by RF): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    except Exception as e:
        print(f"  Hybrid space unavailable ({e}); continuing with concat only")

    # Subsample for GP. IMPORTANT: keep replicate samples in the subsample so
    # the ground-truth correlation is not silently emptied out.
    if n_samples > GP_MAX:
        rng = np.random.RandomState(42)
        if has_replicate is not None and has_replicate.sum() > 0:
            rep_idx = np.where(has_replicate)[0]
            non_rep_idx = np.where(~has_replicate)[0]
            n_extra = max(0, GP_MAX - len(rep_idx))
            if n_extra > 0 and len(non_rep_idx) > 0:
                extra = rng.choice(non_rep_idx,
                                   min(n_extra, len(non_rep_idx)),
                                   replace=False)
                sub_idx = np.concatenate([rep_idx, extra])
            else:
                # More replicate samples than GP_MAX — keep them all.
                sub_idx = rep_idx
            sub_idx.sort()
        else:
            sub_idx = rng.choice(n_samples, GP_MAX, replace=False)
            sub_idx.sort()
    else:
        sub_idx = np.arange(n_samples)

    top3 = extract_top3_base_reps(_search) if _search is not None else None
    results = {
        'n_samples': n_samples,
        'n_unique_pairs': int(len(np.unique(replicate_groups))),
        'n_replicate_groups': n_replicate_groups,
        'n_with_replicates': n_with_replicates,
        'n_subsampled': int(len(sub_idx)),
        'has_ground_truth': replicate_stds is not None,
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
        'estimators': {},
    }

    for space_name, X_space in spaces.items():
        print(f"\n  === Space: {space_name} (dim={X_space.shape[1]}) ===")
        space_results = {}

        X_sub = X_space[sub_idx]
        y_sub = labels[sub_idx]
        rep_stds_sub = replicate_stds[sub_idx] if replicate_stds is not None else None
        has_rep_sub = has_replicate[sub_idx] if has_replicate is not None else None

        # Cold-target scaffold groups for this subsample — used by estimators
        # that split internally (e.g. conformal) so their calibration set stays
        # scaffold-disjoint from training. Replicate grouping is NOT used here;
        # it is only the ground-truth for scoring below.
        _cv_groups = data.get('groups')
        sub_groups = None
        if _cv_groups is not None:
            _cv_groups = np.asarray(_cv_groups)
            if len(_cv_groups) == n_samples:
                sub_groups = _cv_groups[sub_idx]

        for est_name in ESTIMATORS_SCREENING:
            print(f"    {est_name}...", end='', flush=True)
            try:
                est_kwargs = {}
                try:
                    _params = inspect.signature(
                        type(get_noise_estimator(est_name)).__init__).parameters
                    if sub_groups is not None and 'groups' in _params:
                        est_kwargs['groups'] = sub_groups
                except Exception:
                    pass
                estimator = get_noise_estimator(est_name, **est_kwargs)
                estimator.fit(X_sub, y_sub)
                scores = np.asarray(estimator.noise_scores_)

                entry = {
                    'score_mean': float(np.mean(scores)),
                    'score_std': float(np.std(scores)),
                }

                # Correlation with ground-truth replicate disagreement —
                # computed ONLY over samples that actually have a replicate
                # std. Correlating against a vector that is mostly 0.0 (the
                # singletons) is meaningless and was a defect in the prior
                # version.
                if rep_stds_sub is not None and has_rep_sub is not None:
                    gt = rep_stds_sub[has_rep_sub]
                    sc = scores[has_rep_sub]
                    n_gt = len(gt)
                    if n_gt >= MIN_GROUND_TRUTH_SAMPLES and np.std(gt) > 0 and np.std(sc) > 0:
                        rho, p = spearmanr(sc, gt)
                        entry['ground_truth_correlation'] = float(rho)
                        entry['ground_truth_p'] = float(p)
                        entry['n_ground_truth'] = int(n_gt)
                        print(f" rho={rho:.3f} (p={p:.2e}, n={n_gt})")
                    else:
                        entry['ground_truth_correlation'] = None
                        print(f" (degenerate ground truth, n={n_gt})")
                else:
                    print(f" score_mean={entry['score_mean']:.4f}")

                # Weight range (downstream-utility sanity check)
                weights = estimator.get_weights(strength=1.0)
                entry['weight_range'] = [float(np.min(weights)), float(np.max(weights))]

                space_results[est_name] = entry

            except Exception as e:
                print(f" FAILED: {e}")
                space_results[est_name] = {'error': str(e)}

        results['estimators'][space_name] = convert_numpy(space_results)

    # Summary: best estimator by ground-truth correlation
    if replicate_stds is not None:
        best_rho = -1.0
        best_est = None
        best_space = None
        for space_name, space_res in results['estimators'].items():
            for est_name, est_res in space_res.items():
                rho = est_res.get('ground_truth_correlation')
                if rho is not None and rho > best_rho:
                    best_rho = rho
                    best_est = est_name
                    best_space = space_name

        if best_est is not None:
            results['summary'] = {
                'best_estimator': best_est,
                'best_space': best_space,
                'best_correlation': float(best_rho),
            }
            print(f"\n  Best: {best_est} in {best_space} space "
                  f"(rho={best_rho:.3f})")
        else:
            results['summary'] = {'best_estimator': None,
                                  'note': 'no estimator produced a valid '
                                          'ground-truth correlation'}
            print("\n  No estimator produced a valid ground-truth correlation")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('N06: BindingDB Ground-Truth Noise')
    parser.add_argument('--budget', type=int, default=100)
    args = parser.parse_args()
    config = get_config(args)

    # Default to bindingdb_kd dataset
    dataset = getattr(args, 'dataset', 'bindingdb_kd')
    if dataset == 'logd':  # parser default; N06 is a DTA experiment
        dataset = 'bindingdb_kd'
    data = load_data(dataset, args.reps)
    results = run(data, config, args.output, dataset, budget=args.budget)
    save_experiment_results(results, args.output, 'n06_bindingdb_ground_truth', dataset)
