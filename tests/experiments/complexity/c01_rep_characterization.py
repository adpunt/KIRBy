#!/usr/bin/env python3
"""C01: Representation Characterization -- profile each rep's complexity.

Paper: Complexity
Conclusion: Representations vary widely in intrinsic quality; no single best.
Dependencies: None (standalone).

Profiles each representation for:
- Intrinsic dimensionality (PCA thresholds at 90/95/99% + MLE)
- Mutual information with labels (KSG estimator)
- Effective rank (exp-entropy of normalised singular values)
- Per-model baseline R2 (single-rep CV)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split,
    compute_intrinsic_dimensionality, estimate_mutual_information_knn,
    compute_effective_rank, get_domain_metric, parse_rep_args,
)


def _single_rep_cv_folds(X, labels, groups, model_factory, n_folds, metric_fn,
                         task_type):
    """One representation, every feature, scored per fold.

    Added 2026-10-05 in place of evaluate_single_rep_cv(..., cap=False), which
    computes the same thing and returns only the mean and the spread, so the
    per-fold scores were never saved. Same steps per fold: StandardScaler
    fitted on the training rows, a fresh model, predictions through
    _predict_for_metric (class-1 probabilities for AUROC, the model's own
    predictions otherwise). Wrapping metric_fn to record the folds was not an
    option: _predict_for_metric picks probabilities by checking that metric_fn
    IS the built-in AUROC function, and a wrapper would fail that check.

    One difference: task_type is passed to cv_split, so on a classification
    dataset the folds are the StratifiedGroupKFold folds the hybrid search
    uses. evaluate_single_rep_cv passes none and cv_split guesses the task
    from whether the labels look like integers, which gives the same folds
    whenever the class labels are stored as integers.

    Returns (fold_scores, widths), one entry per fold.
    """
    from kirby.find_best_hybrid import _predict_for_metric
    scores, widths = [], []
    for train_idx, val_idx in cv_split(groups, n_folds, X, labels,
                                       task_type=task_type):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        widths.append(int(X_train.shape[1]))
        model = model_factory()
        model.fit(X_train, labels[train_idx])
        scores.append(float(metric_fn(
            labels[val_idx], _predict_for_metric(model, X_val, metric_fn))))
    return scores, widths


def run(data, config, output_dir, dataset_name, **kwargs):
    """Profile each representation's complexity metrics and per-model baselines."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C01: REPRESENTATION CHARACTERIZATION")
    print("=" * 70)

    models = get_model_factories(config['models'],
                                    task_type=data['task_type'],
                                    n_samples=len(data['labels']))
    n_folds = config['n_splits']
    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # Record the metric + its direction so the per-rep baseline scores in the
    # output JSON are interpretable without re-deriving the domain metric.
    # (MISS-24: c01 baselines are stored as bare numbers; a reader can't tell
    # whether higher or lower is better, nor which metric was used.)
    results = {
        '_meta': {
            'metric': metric_name,
            'lower_is_better': bool(lower_is_better),
            'task_type': data['task_type'],
        }
    }

    for rep_name, X in reps.items():
        print(f"\n  {rep_name} (dim={X.shape[1]}):")
        rep_result = {}

        # COST cap (§F): the MLE intrinsic-dim (O(n^2) per-rep loop) and the KSG
        # mutual-information both run on full n — the c01 large-n CPU wall. Both
        # converge well before ~2000 rows, so row-subsample to 2000 for BOTH
        # (mirrors c17/s03, which already cap MI). Deterministic. effective_rank
        # (SVD, already size-guarded) stays on full X below.
        if X.shape[0] > 2000:
            _idx = np.random.RandomState(42).choice(X.shape[0], 2000, replace=False)
            X_cost, labels_cost = X[_idx], np.asarray(labels)[_idx]
        else:
            X_cost, labels_cost = X, labels

        # Intrinsic dimensionality
        dim_info = compute_intrinsic_dimensionality(X_cost)
        rep_result['intrinsic_dim'] = dim_info
        if dim_info.get('mle') is not None:
            print(f"    PCA95={dim_info.get('pca_95', '?')}, MLE={dim_info.get('mle', '?'):.1f}")
        else:
            print(f"    PCA95={dim_info.get('pca_95', '?')}")

        # Mutual information (use PCA-reduced for stability)
        n_comp = min(50, X_cost.shape[1], X_cost.shape[0] - 1)
        X_reduced = PCA(n_components=n_comp).fit_transform(StandardScaler().fit_transform(X_cost))
        # How many components the MI value is a sum over (2026-10-05).
        # estimate_mutual_information_knn scores each component against the
        # label on its own and SUMS the per-component values, after a second
        # PCA down to min(50, rows // 5) components when its input is wider
        # than that. So the value grows with the number of components and can
        # exceed ln 2 = 0.693 nats on a binary label, which a joint MI cannot.
        # This mirrors the estimator's own rule; the estimator is unchanged.
        _n_rows_mi = int(X_reduced.shape[0])
        _mi_cap = max(1, min(50, _n_rows_mi // 5))
        rep_result['mutual_info_n_components_summed'] = (
            0 if _n_rows_mi < 10 else int(min(X_reduced.shape[1], _mi_cap)))
        rep_result['mutual_info_n_rows'] = _n_rows_mi
        try:
            mi = estimate_mutual_information_knn(X_reduced, labels_cost, k=5)
            rep_result['mutual_info'] = float(mi)
            print(f"    MI={mi:.4f} (sum over "
                  f"{rep_result['mutual_info_n_components_summed']} components)")
        except Exception as e:
            rep_result['mutual_info'] = None
            print(f"    MI failed: {e}")

        # Effective rank
        rep_result['effective_rank'] = compute_effective_rank(X)
        # Which singular values it was computed from (2026-10-05).
        # compute_effective_rank uses every singular value when the
        # representation has at most 2000 features and only the top
        # min(500, rows - 1) from a TruncatedSVD above that, so a wide
        # representation's value is computed from fewer singular values and
        # cannot exceed min(500, rows - 1). The count stored is the most
        # singular values used; values under 1e-10 are dropped inside
        # compute_effective_rank. Mirrors the rule in
        # compute_effective_rank (_shared.py); that function is unchanged.
        if X.shape[1] > 2000:
            _n_sv = int(min(500, X.shape[0] - 1))
            rep_result['effective_rank_method'] = f'truncated_svd_top_{_n_sv}'
            rep_result['effective_rank_n_singular_values_max'] = _n_sv
        else:
            rep_result['effective_rank_method'] = 'full_svd'
            rep_result['effective_rank_n_singular_values_max'] = int(
                min(X.shape[0], X.shape[1]))
        print(f"    Effective rank={rep_result['effective_rank']:.1f}")

        # Per-model baselines (domain-aware metric)
        rep_result['baselines'] = {}
        scores_str = []
        for model_name, model_factory in models.items():
            # cap=False: score every feature this representation has.
            #
            # evaluate_single_rep_cv used to PCA-rotate any representation wider
            # than the fold's training rows, and c01's question is how well each
            # representation predicts the label ON ITS OWN. Measured here on
            # flab_thermo_tresanco, random forest, spearman, the same five folds:
            # saprot 0.3775 rotated against 0.5896 at its own 1,280 features,
            # esmc 0.2405 against 0.5454, esm3 0.3714 against 0.5302. The
            # unrotated numbers are exactly what find_best_hybrid's own
            # full-dimension baselines print for the same three, so the rotated
            # ones put c01 and every search baseline 0.16 to 0.30 spearman apart
            # on the same representation. c11 took the same rotation off its
            # baselines (c11_ensemble_vs_hybrid.py _prep).
            #
            # The width fitted is still recorded per fold, which now equals
            # n_features_offered on every fold.
            #
            # Per-fold scores saved from 2026-10-05; see _single_rep_cv_folds.
            # score_mean / score_std are np.mean / np.std of them, as before.
            fold_scores, widths = _single_rep_cv_folds(
                X, labels, groups, model_factory, n_folds, metric_fn,
                data['task_type'])
            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            rep_result['baselines'][model_name] = {
                'score_mean': mean_score,
                'score_std': std_score,
                'fold_scores': fold_scores,
                'n_features_offered': int(X.shape[1]),
                'n_features_fitted_per_fold': widths,
            }
            scores_str.append(f"{model_name}={mean_score:.3f}")
        print(f"    Baselines ({metric_name}): {', '.join(scores_str)}")

        results[rep_name] = rep_result

    # Summary table
    print("\n" + "-" * 70)
    print(f"SUMMARY (metric: {metric_name})")
    print("-" * 70)
    header = f"  {'Rep':<20} {'Dim':>5} {'PCA95':>6} {'MLE':>6} {'MI':>8} {'ERank':>7}"
    for m in models:
        header += f" {m:>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rep_name, r in results.items():
        if rep_name == '_meta':
            continue
        dim_info = r['intrinsic_dim']
        mi_str = f"{r['mutual_info']:.4f}" if r['mutual_info'] is not None else "N/A"
        mle_str = f"{dim_info['mle']:.1f}" if dim_info.get('mle') is not None else "N/A"
        row = (f"  {rep_name:<20} {dim_info['nominal_dim']:>5} "
               f"{dim_info.get('pca_95', 'N/A'):>6} {mle_str:>6} {mi_str:>8} "
               f"{r['effective_rank']:>7.1f}")
        for m in models:
            bl = r['baselines'].get(m, {})
            row += f" {bl.get('score_mean', float('nan')):>8.3f}"
        print(row)

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C01: Representation Characterization',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c01_rep_characterization', args.dataset)
