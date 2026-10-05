#!/usr/bin/env python3
"""S01: Similarity-Activity Correlation Baseline -- how poorly standard distances work.

Paper: Similarity
Conclusion: Per-rep distances poorly correlate with activity.
Dependencies: None.

Per outer fold, for each representation, under each distance the kNN record
keeps for the dataset's group (knn_axes), measure:
- Similarity-Activity Correlation (SAC): Spearman(distance, |delta_y|) over
  sampled pairs of the fold's held-out molecules
- Violation rate: share of each held-out molecule's k nearest training
  molecules with a large |delta_y| (or a different class), at every decided
  neighbour count and counting scheme
- SALI: Structure-Activity Landscape Index, each held-out molecule against
  every training molecule

No model is fitted and nothing is selected: every representation is scored
(handoff 2.3, approved exception group A). Features are scaled with
kirby.scaling.FeatureScaler fitted on the fold's training rows (0/1 features
unscaled, the rest standardised; held-out values cut to the training range).
"""
import sys
import os
import time
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
)
from experiments.similarity.similarity_helpers import (
    run_antibody_cliff_analysis,
    _pairwise_delta, _sample_offdiag_pairs,
    sac_pair_sample, permuted_delta_ranks, sac_permutation_p,
    sim_outer_folds, fold_spread_note, knn_axes, scale_fold,
)


# Above this many molecules a pair sample is drawn without listing every pair
# (np.triu_indices over n molecules holds n^2/2 index pairs, ~7 GB at 30,000).
# Below it the pairs are drawn from the full list, as before. It no longer
# switches the scoring method: every distance is computed in row chunks of
# held-out molecules against training molecules (similarity_space's chunk
# size), so no n x n matrix is built at any n.
S01_MAX_DENSE_N = 15000

# Written into the checkpoint identity. Bumped 2026-10-05 (handoff Part 2,
# chat 4) when s01 moved from one whole-dataset pass under the process-wide
# distance to per-fold scores under each distance of the kNN record, so a
# sidecar from before cannot be read back into the new layout.
# Bumped again the same day when SALI's left-out pairs changed from
# "distance == 0" to "identical feature vectors, or distance within
# _SALI_ZERO_TOL of 0" (sali_n_zero_distance_not_identical added).
S01_RESULT_VERSION = 'per_fold_knn_record_sali_identity_2026-10-05'

# SALI's denominator offset, as compute_sali (similarity_helpers.py).
_SALI_EPS = 1e-6

# A pair at distance <= this is treated as distance 0 for SALI. scipy's
# cosine returns 2.2e-16 for some identical rows, so "D == 0" misses them
# and their SALI is |delta_y| x ~1e6. The tolerance is six orders below
# _SALI_EPS, so it only catches rounding, not near neighbours.
_SALI_ZERO_TOL = 1e-12


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Establish the problem: how poorly standard distances correlate with
    property similarity across representations, per outer fold."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S01: SIMILARITY-ACTIVITY CORRELATION BASELINE")
    print("=" * 70)

    n_sample = kwargs.get('n_sample', 5000)
    labels = np.asarray(data['labels'])
    n_samples = len(labels)

    task_type = data['task_type']
    is_classification = task_type == 'classification'
    if is_classification:
        print("  NOTE: classification labels — SAC uses |delta_y| on 0/1, "
              "violation rate still well-defined")

    results = {}
    # What every number in this file is (2026-10-05), as s02 does.
    # save_experiment_results stamps 'metric_name' and 'lower_is_better' from
    # the dataset registry when the results carry none, so an s01 file was
    # stamped auc, f1_weighted or rmse while every sac_rho in it is the
    # Spearman between the distance of a pair of molecules and the difference
    # in their activity (on a classification dataset, whether the two are in
    # different classes). On an rmse dataset the old stamp also said lower is
    # better. The dataset's own metric is kept under 'dataset_metric'.
    results['metric_name'] = 'spearman'
    results['lower_is_better'] = False
    results['metric_definition'] = (
        'Spearman correlation between pairwise distance and pairwise activity '
        'difference (1 if the two are in different classes on a '
        'classification dataset), over sampled pairs of the fold\'s held-out '
        'molecules')
    results['other_scores_direction'] = {
        'violation_rate': 'lower is better (share of neighbours with a very '
                          'different activity or another class)',
        'sali': 'descriptive; larger means steeper activity changes between '
                'close molecules',
    }
    try:
        from experiments._shared import get_domain_metric as _gdm
        _dm = _gdm(data)
        results['dataset_metric'] = _dm[0]
        results['dataset_metric_lower_is_better'] = bool(_dm[2])
    except Exception:
        results['dataset_metric'] = None
    results['result_version'] = S01_RESULT_VERSION
    # s01 fits no model and picks nothing (handoff 2.3, group A): every
    # representation is scored. The only thing fitted is the feature scaler,
    # and it is fitted on each fold's training rows.
    results['selection_scope'] = 'none: every representation scored'
    results['scaler_fit_scope'] = 'fold_train_rows'

    # Representations: DATASET_REPS[ds]['general'] only (handoff 2.4 step 9).
    # run_all already loads that list; the standalone path below loads
    # --reps, which can differ, so the loop is filtered here. data['reps'] is
    # not changed.
    reps, rep_audit = _scored_reps(data, dataset_name)
    results['representations'] = rep_audit

    # The kNN record (KNN_SEARCH_SPACE_RECORD.md axes 2, 4, 5, 6): which
    # distances, neighbour counts and counting schemes every number below is
    # computed under. Replaces the single process-wide distance
    # (KIRBY_DISTANCE_METRIC / --distance-metric, manhattan by default) and
    # the fixed 5 neighbours.
    axes = knn_axes(dataset_name, n_samples)
    results['knn_axes'] = axes
    print(f"\n  kNN record ({axes['group']}): distances {axes['metrics']}, "
          f"neighbour counts {axes['ks']}, counting schemes "
          f"{axes['schemes']}, scaffold group not excluded")

    # Outer folds from the one fold setting (config['n_splits']).
    folds = sim_outer_folds(data, config)
    n_folds = len(folds)
    results['n_outer_folds'] = n_folds
    print(f"  {n_folds} outer fold(s); {len(reps)} representations")

    # Each (fold, representation) is written to
    # s01_similarity_baseline/<dataset>.partial.jsonl as it finishes and read
    # back on the next attempt (_checkpoint.py). Its scores depend only on the
    # fold's rows, the matrix, the labels and the pairs/shuffles below (fixed
    # seeds), so reading one back gives the numbers a fresh computation gives.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's01_similarity_baseline', dataset_name),
        run_identity('s01_similarity_baseline', dataset_name, data, config,
                     kwargs=kwargs, version=S01_RESULT_VERSION,
                     n_outer_folds=n_folds, knn_axes=axes,
                     reps=sorted(reps)))

    fold_records = []
    for fold_i, (tr, va) in enumerate(folds):
        tr = np.asarray(tr)
        va = np.asarray(va)
        _t_fold = time.time()
        print(f"\n  -- fold {fold_i + 1}/{n_folds}: {len(tr)} training, "
              f"{len(va)} held-out molecules --")
        fold_rec = {
            'fold': fold_i,
            'n_train': int(len(tr)),
            'n_val': int(len(va)),
            'scaler_fit_scope': 'fold_train_rows',
            'reps': {},
        }
        if len(va) < 2 or len(tr) < 1:
            fold_rec['skipped'] = (f'{len(va)} held-out and {len(tr)} training '
                                   f'molecules: no pair to score')
            print(f"    skipped: {fold_rec['skipped']}")
            fold_records.append(fold_rec)
            continue

        y_tr = labels[tr]
        y_va = labels[va]
        # The held-out pairs and their 200 shuffled activity columns, once per
        # fold: neither depends on the representation or the distance.
        pairs = sac_pair_sample(len(va), n_sample,
                                scalable=len(va) > S01_MAX_DENSE_N)
        perm_ranks = permuted_delta_ranks(y_va, pairs[0], pairs[1],
                                          is_clf=is_classification)
        # The activity difference a neighbour counts as "very different",
        # from training-row pairs only (None on classification).
        viol_ref = _violation_reference(y_tr, is_classification,
                                        scalable=len(tr) > S01_MAX_DENSE_N)
        fold_rec['sac_sampling'] = {
            'n_pairs': int(len(pairs[0])),
            'n_pairs_available': int(len(va) * (len(va) - 1) // 2),
            'n_permutations': int(perm_ranks.shape[0]),
            'note': ('every sac_rho in this fold is one Spearman over these '
                     'pairs of held-out molecules, and every '
                     'sac_permutation_null_sd is the spread of that Spearman '
                     'over these shuffles of the held-out activity'),
        }
        fold_rec['violation_threshold_median_delta_y'] = viol_ref

        for rep_name, X in reps.items():
            if _ck.done(fold=fold_i, rep=rep_name):
                fold_rec['reps'][rep_name] = _ck.get(fold=fold_i, rep=rep_name)
                continue
            rec = _score_rep_fold(
                rep_name, np.asarray(X), tr, va, y_tr, y_va, axes,
                pairs=pairs, perm_ranks=perm_ranks, viol_ref=viol_ref,
                is_classification=is_classification)
            _ck.put(rec, fold=fold_i, rep=rep_name)
            fold_rec['reps'][rep_name] = rec
        print(f"    fold {fold_i + 1} done in {time.time() - _t_fold:.1f}s")
        fold_records.append(fold_rec)

    results['folds'] = fold_records
    results['by_rep'] = _summarise(fold_records, reps, axes)

    # Summary table: one row per representation and distance. SAC is the mean
    # over folds; the violation rate is at the smallest decided neighbour
    # count under each counting scheme.
    k0 = min(axes['ks'])
    print("\n  Summary (mean over folds):")
    hdr = f"  {'Rep':<15} {'Distance':<15} {'Dim':>5} {'SAC rho':>9}"
    for s in axes['schemes']:
        hdr += f" {('Viol k' + str(k0) + ' ' + s)[:22]:>22}"
    hdr += f" {'SALI':>9}"
    print(hdr)
    for name, by_metric in results['by_rep'].items():
        for metric, r in by_metric.items():
            line = (f"  {name:<15} {metric:<15} {r['dim']:>5} "
                    f"{_fmt(r['sac_rho_mean']):>9}")
            for s in axes['schemes']:
                line += f" {_fmt(r['violation_rate_mean'].get(f'k{k0}_{s}')):>22}"
            line += f" {_fmt(r['sali_mean_mean']):>9}"
            print(line)

    # Sequence-based cliff sweep (antibody datasets + nucleotide) — independent
    # of any representation choice. Tests whether close pairs in sequence space
    # carry label flips (classification) or large activity diffs (regression).
    # dataset_name is passed so CDR extraction is cached per-dataset. It
    # counts pairs by edit distance over the whole dataset and makes no
    # prediction, so it has no folds and no feature distance.
    cliff_results = run_antibody_cliff_analysis(
        data, is_clf=is_classification, dataset_name=dataset_name)
    if cliff_results is not None:
        results['_sequence_cliffs'] = cliff_results

    return results


def _fmt(v):
    return 'n/a' if v is None or not np.isfinite(v) else f'{v:.4f}'


def _scored_reps(data, dataset_name):
    """The representations s01 scores: DATASET_REPS[ds]['general'] that are
    loaded, in that order. A dataset with no DATASET_REPS row keeps every
    loaded representation, with a printed line."""
    from experiments.complexity.complexity_helpers import DATASET_REPS
    loaded = data['reps']
    allowed = (DATASET_REPS.get(dataset_name) or {}).get('general')
    if allowed is None:
        print(f"  {dataset_name} has no DATASET_REPS['general'] list: scoring "
              f"all {len(loaded)} loaded representations")
        return dict(loaded), {'source': 'all loaded (no DATASET_REPS row)',
                              'scored': list(loaded), 'not_scored': [],
                              'not_loaded': []}
    kept = {r: loaded[r] for r in allowed if r in loaded}
    not_scored = [r for r in loaded if r not in allowed]
    not_loaded = [r for r in allowed if r not in loaded]
    if not_scored:
        print(f"  not scored, outside DATASET_REPS['general']: {not_scored}")
    if not_loaded:
        print(f"  in DATASET_REPS['general'] but not loaded (or dropped by "
              f"the NaN/zero screen): {not_loaded}")
    return kept, {'source': "DATASET_REPS[dataset]['general']",
                  'scored': list(kept), 'not_scored': not_scored,
                  'not_loaded': not_loaded}


def _violation_reference(labels, is_classification, scalable=False):
    """The activity difference the violation rate counts neighbours against:
    the median |delta_y| over 50,000 pairs of the given molecules (all pairs
    when fewer), drawn under seed 42. Called with the fold's TRAINING labels.

    Classification has no threshold — the rate counts neighbours in a
    different class — so it returns None there.
    """
    if is_classification:
        return None
    y = np.asarray(labels, dtype=np.float64)
    n = len(y)
    if n < 2:
        return None
    if scalable:
        pi, pj = _sample_offdiag_pairs(n, 50000, 42)
        return float(np.median(np.abs(y[pi] - y[pj])))
    triu_i, triu_j = np.triu_indices(n, k=1)
    if len(triu_i) > 50000:
        sample_idx = np.random.RandomState(42).choice(len(triu_i), 50000,
                                                      replace=False)
        return float(np.median(np.abs(y[triu_i[sample_idx]] -
                                      y[triu_j[sample_idx]])))
    return float(np.median(np.abs(y[triu_i] - y[triu_j])))


def _pair_feature_distance(X, pi, pj, metric, block=8192):
    """Distance between row pi[t] and row pj[t] of X under one distance of the
    kNN record, one number per pair (no off-diagonal work). Matches
    similarity_space.feature_distance: cosine is undefined on a row with no
    length and such a pair is +inf, as that function sets it."""
    out = np.empty(len(pi), dtype=np.float64)
    for start in range(0, len(pi), block):
        end = min(start + block, len(pi))
        a = np.asarray(X[pi[start:end]], dtype=np.float64)
        b = np.asarray(X[pj[start:end]], dtype=np.float64)
        if metric == 'manhattan':
            out[start:end] = np.abs(a - b).sum(axis=1)
        elif metric == 'minkowski_p0.5':
            out[start:end] = np.sqrt(np.abs(a - b)).sum(axis=1) ** 2
        elif metric == 'cosine':
            num = (a * b).sum(axis=1)
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                d = np.clip(1.0 - num / den, 0.0, 2.0)
            out[start:end] = np.where(den > 0, d, np.inf)
        else:
            raise ValueError(f"s01: no row-wise form for distance {metric!r}")
    return out


def _neighbours_and_sali(X_val, X_train, y_val, y_train, metric, k_max,
                         is_classification):
    """Each held-out molecule's k_max nearest training molecules, and its
    SALI against every training molecule, from one row-chunked distance pass.

    Neighbours: ties broken by training-row index (similarity_space's
    _stable_topk). The query rows are held-out molecules and the candidates
    are training molecules, so no molecule can be its own neighbour and no
    twin is dropped in its place (dirty-run bug 7). No scaffold group is
    excluded (kNN record axis 6).

    SALI = |delta_y| / (1 - sim + 1e-6), sim = 1 / (1 + d), as compute_sali.
    Pairs at distance 0 are left out and counted (owner, 2026-10-05, eda240c):
    1 - sim is 0 for them and SALI is undefined. Two tests decide which pairs
    those are, because under cosine "D == 0" is not the identical-feature
    test: identical feature vectors are found by comparing the rows
    directly (cosine can put them at 2.2e-16), and any pair at distance
    <= _SALI_ZERO_TOL is also left out (under cosine this includes parallel
    vectors that are not identical, counted separately).
    """
    from experiments.noise import similarity_space as ss
    n_val, n_train = len(X_val), len(X_train)
    k = int(min(k_max, n_train))
    y_val = np.asarray(y_val)
    y_train = np.asarray(y_train)
    idx_parts, dist_parts = [], []
    per_max = np.empty(n_val)
    per_mean = np.empty(n_val)
    n_same = 0
    n_same_diff = 0
    n_zero_not_same = 0
    # Training rows keyed by their exact feature values, so a held-out row's
    # identical training rows are a lookup, independent of the distance.
    X_val_c = np.ascontiguousarray(X_val)
    X_train_c = np.ascontiguousarray(X_train)
    train_rows_by_key = {}
    for j in range(n_train):
        train_rows_by_key.setdefault(X_train_c[j].tobytes(), []).append(j)
    step = ss._chunk_rows(n_train)
    for start in range(0, n_val, step):
        stop = min(n_val, start + step)
        D = ss.feature_distance(X_val[start:stop], X_train, metric)
        i_, d_ = ss._stable_topk(D, k)
        idx_parts.append(i_)
        dist_parts.append(d_)
        if is_classification:
            dy = (y_val[start:stop, None] != y_train[None, :]).astype(float)
        else:
            dy = np.abs(y_val[start:stop, None].astype(np.float64)
                        - y_train[None, :].astype(np.float64))
        sim = 1.0 / (1.0 + D)
        sali = dy / (1.0 - sim + _SALI_EPS)
        same = np.zeros(D.shape, dtype=bool)
        for r in range(stop - start):
            js = train_rows_by_key.get(X_val_c[start + r].tobytes())
            if js:
                same[r, js] = True
        zero = D <= _SALI_ZERO_TOL
        n_same += int(same.sum())
        n_same_diff += int((same & (dy > 0)).sum())
        n_zero_not_same += int((zero & ~same).sum())
        sali = np.where(same | zero, np.nan, sali)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            per_max[start:stop] = np.nanmax(sali, axis=1)
            per_mean[start:stop] = np.nanmean(sali, axis=1)
    idx = np.vstack(idx_parts)
    dist = np.vstack(dist_parts)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sali_rec = {
            'mean': float(np.nanmean(per_max)),
            'max': float(np.nanmax(per_max)),
            'std': float(np.nanstd(per_max)),
            'median': float(np.nanmedian(per_max)),
            'per_sample_mean_avg': float(np.nanmean(per_mean)),
            # Held-out x training pairs, each counted once.
            'n_identical_pairs': n_same,
            'n_identical_pairs_with_activity_difference': n_same_diff,
            # Pairs left out at distance <= _SALI_ZERO_TOL whose feature
            # vectors differ (under cosine: parallel vectors).
            'n_zero_distance_not_identical': n_zero_not_same,
            'n_heldout_with_no_defined_pair': int(np.isnan(per_max).sum()),
        }
    return idx, dist, sali_rec


def _violation_rates(idx, dist, y_val, y_train, viol_ref, axes,
                     is_classification):
    """Violation rate at every decided neighbour count and counting scheme.

    For each held-out molecule: the weighted share of its k nearest training
    molecules whose activity differs from its own by more than viol_ref (on
    classification, that are in another class), weights from the counting
    scheme (similarity_space.neighbour_weights: rank = 1, 1/2, 1/3, ...;
    inverse_square = one over the distance squared). The rate is the mean of
    those shares over held-out molecules.
    """
    from experiments.noise import similarity_space as ss
    y_val = np.asarray(y_val)
    nb = np.asarray(y_train)[idx]
    if is_classification:
        bad = nb != y_val[:, None]
    else:
        bad = np.abs(nb.astype(np.float64)
                     - y_val[:, None].astype(np.float64)) > viol_ref
    rows = []
    for k in axes['ks']:
        if k > idx.shape[1]:
            rows.extend({'distance': None, 'k': int(k), 'scheme': s,
                         'violation_rate': None,
                         'note': f'only {idx.shape[1]} training molecules'}
                        for s in axes['schemes'])
            continue
        for scheme in axes['schemes']:
            w = ss.neighbour_weights(dist[:, :k], scheme)
            tot = w.sum(axis=1)
            hit = (w * bad[:, :k]).sum(axis=1)
            share = np.divide(hit, tot, out=np.full(len(tot), np.nan),
                              where=tot > 0)
            rows.append({'k': int(k), 'scheme': scheme,
                         'violation_rate': float(np.nanmean(share))
                         if np.isfinite(share).any() else None,
                         'n_heldout_no_weight': int((tot <= 0).sum())})
    return rows


def _score_rep_fold(rep_name, X, tr, va, y_tr, y_va, axes, *, pairs,
                    perm_ranks, viol_ref, is_classification):
    """SAC / violation rate / SALI for one representation in one fold, under
    every distance in axes['metrics']."""
    print(f"    {rep_name} (dim={X.shape[1]}):")
    # FeatureScaler fitted on the training rows (dirty-run bug 3): 0/1
    # features unscaled, the rest standardised; held-out values cut to the
    # training range. Computed in float64 for the distances only.
    X_tr, X_va = scale_fold(X[tr], X[va])
    pair_i, pair_j = pairs
    delta = _pairwise_delta(y_va[pair_i], y_va[pair_j],
                            is_clf=is_classification)
    rec = {'dim': int(X.shape[1]), 'by_distance': {}}
    for metric in axes['metrics']:
        pair_dists = _pair_feature_distance(X_va, pair_i, pair_j, metric)
        n_undef = int((~np.isfinite(pair_dists)).sum())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Only rho is kept; scipy's p-value is not stored (the owner bans
            # significance tests).
            sac_rho = float(spearmanr(pair_dists, delta)[0])
        # The spread of the Spearman over shuffles of the held-out activity on
        # these same pairs. Only the shuffled correlations are kept; the
        # permutation p-value is not stored (no significance tests). The
        # spread is what a difference between two representations' sac_rho
        # has to clear.
        _perm_p_unused, perm_rhos = sac_permutation_p(
            pair_dists, perm_ranks, sac_rho)

        idx, dist, sali = _neighbours_and_sali(
            X_va, X_tr, y_va, y_tr, metric, axes['k_max'], is_classification)
        viol = _violation_rates(idx, dist, y_va, y_tr, viol_ref, axes,
                                is_classification)
        for r in viol:
            r['distance'] = metric

        rec['by_distance'][metric] = {
            'distance': metric,
            'sac_rho': sac_rho,
            'sac_n_pairs_undefined_distance': n_undef,
            'sac_permutation_null_sd': float(np.std(perm_rhos))
                                       if len(perm_rhos) else float('nan'),
            'sac_permutation_null_max_abs': float(np.max(np.abs(perm_rhos)))
                                            if len(perm_rhos) else float('nan'),
            # One row per (neighbour count, counting scheme); each row names
            # its distance, k and scheme.
            'violation_rate': viol,
            'violation_threshold_median_delta_y': viol_ref,
            'sali_mean': sali['mean'],
            'sali_max': sali['max'],
            'sali_std': sali['std'],
            'sali_median': sali['median'],
            'sali_per_sample_mean_avg': sali['per_sample_mean_avg'],
            'sali_n_identical_pairs': sali['n_identical_pairs'],
            'sali_n_identical_pairs_with_activity_difference':
                sali['n_identical_pairs_with_activity_difference'],
            'sali_n_zero_distance_not_identical':
                sali['n_zero_distance_not_identical'],
            'sali_n_heldout_with_no_defined_pair':
                sali['n_heldout_with_no_defined_pair'],
        }
        v0 = viol[0]['violation_rate'] if viol else None
        print(f"      {metric:<15} SAC rho={_fmt(sac_rho)}, violation_rate "
              f"(k={viol[0]['k'] if viol else '-'}, "
              f"{viol[0]['scheme'] if viol else '-'})={_fmt(v0)}, "
              f"SALI_mean={_fmt(sali['mean'])}")
    return rec


def _summarise(fold_records, reps, axes):
    """Per representation and distance: each fold's scores, their mean, and
    their spread over folds (standard deviation; 'one fold, no spread' with
    a single fold). No test across folds."""
    out = {}
    for rep_name in reps:
        by_metric = {}
        for metric in axes['metrics']:
            sac, sali, viol, dim = [], [], {}, None
            for fr in fold_records:
                r = fr.get('reps', {}).get(rep_name)
                m = (r or {}).get('by_distance', {}).get(metric)
                if m is None:
                    sac.append(None)
                    sali.append(None)
                    continue
                dim = r.get('dim', dim)
                sac.append(m['sac_rho'])
                sali.append(m['sali_mean'])
                for v in m['violation_rate']:
                    viol.setdefault(f"k{v['k']}_{v['scheme']}", []).append(
                        v['violation_rate'])

            def _mean(vals):
                f = [v for v in vals if v is not None and np.isfinite(v)]
                return float(np.mean(f)) if f else None

            def _sd(vals):
                if fold_spread_note(vals):
                    return fold_spread_note(vals)
                f = [v for v in vals if v is not None and np.isfinite(v)]
                return float(np.std(f))

            by_metric[metric] = {
                'distance': metric,
                'dim': dim,
                'sac_rho_per_fold': sac,
                'sac_rho_mean': _mean(sac),
                'sac_rho_fold_sd': _sd(sac),
                'violation_rate_per_fold': viol,
                'violation_rate_mean': {key: _mean(v)
                                        for key, v in viol.items()},
                'violation_rate_fold_sd': {key: _sd(v)
                                           for key, v in viol.items()},
                'sali_mean_per_fold': sali,
                'sali_mean_mean': _mean(sali),
                'sali_mean_fold_sd': _sd(sali),
            }
        out[rep_name] = by_metric
    return out


if __name__ == '__main__':
    parser = create_experiment_parser('S01: Similarity-Activity Correlation Baseline',
                                     has_delta=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's01_similarity_baseline', args.dataset)
