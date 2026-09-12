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
    get_model_factories, evaluate_single_rep_cv,
    compute_intrinsic_dimensionality, estimate_mutual_information_knn,
    compute_effective_rank, get_domain_metric, parse_rep_args,
)


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
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'
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
        try:
            mi = estimate_mutual_information_knn(X_reduced, labels_cost, k=5)
            rep_result['mutual_info'] = float(mi)
            print(f"    MI={mi:.4f}")
        except Exception as e:
            rep_result['mutual_info'] = None
            print(f"    MI failed: {e}")

        # Effective rank
        rep_result['effective_rank'] = compute_effective_rank(X)
        print(f"    Effective rank={rep_result['effective_rank']:.1f}")

        # Per-model baselines (domain-aware metric)
        rep_result['baselines'] = {}
        scores_str = []
        for model_name, model_factory in models.items():
            # evaluate_single_rep_cv PCA-compresses any representation wider
            # than the fold's training rows, so the number below can come from
            # far fewer features than this representation has. Record the width
            # it actually fitted beside the score, one entry per fold.
            mean_score, std_score, widths = evaluate_single_rep_cv(
                X, labels, groups, model_factory, n_folds, mode=mode,
                metric_fn=metric_fn, return_width=True)
            rep_result['baselines'][model_name] = {
                'score_mean': mean_score,
                'score_std': std_score,
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
