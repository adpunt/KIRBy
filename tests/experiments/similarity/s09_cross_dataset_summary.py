#!/usr/bin/env python3
"""S09: Cross-Dataset Generalization Summary.

Paper: Similarity
Conclusion: Results hold across 4 diverse datasets.
Dependencies: s02, s04, s05, s06, s07, s08 (reads their JSON outputs).

Aggregates metrics across datasets: SAC deltas, cliff reductions, kNN deltas,
AD deltas, FP reductions, dim ratios. Runs Wilcoxon test if enough datasets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from scipy.stats import wilcoxon

from experiments._shared import (
    save_experiment_results, load_experiment_results, convert_numpy,
)


# Map from experiment IDs used internally to the result directory names
EXPERIMENT_IDS = [
    's02_hybrid_similarity',
    's04_activity_cliffs',
    's05_knn_prediction',
    's06_applicability_domain',
    's07_noise_bridge',
    's08_optimal_dimension',
]


def run(all_dataset_results):
    """Consolidate findings across datasets. Show improvement is consistent."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S09: CROSS-DATASET GENERALIZATION SUMMARY")
    print("=" * 70)

    results = {'per_dataset': {}, 'summary': {}}

    sac_deltas = []
    cliff_reductions = []
    knn_deltas = []
    ad_deltas = []
    fp_reductions = []
    dim_ratios = []

    for ds_name, ds_results in all_dataset_results.items():
        print(f"\n  {ds_name}:")
        ds_summary = {}

        # Exp 2: SAC improvement vs best individual rep
        exp2 = ds_results.get('s02_hybrid_similarity')
        if exp2:
            best_rep_info = exp2.get('best_individual_rep', {})
            best_rep_sac = best_rep_info.get('sac_rho', float('nan'))
            best_rep_name = best_rep_info.get('name', '?')
            best_hybrid_sac = exp2.get('best_sac', float('nan'))
            delta_sac = best_hybrid_sac - best_rep_sac
            ds_summary['sac_delta'] = delta_sac
            ds_summary['best_rep'] = best_rep_name
            sac_deltas.append(delta_sac)
            print(f"    SAC delta: {delta_sac:+.4f} "
                  f"(best_rep {best_rep_name}={best_rep_sac:.4f} -> "
                  f"hybrid={best_hybrid_sac:.4f})")

        # Exp 4: RMSE_cliff improvement (hybrid vs best per-rep kNN)
        exp4 = ds_results.get('s04_activity_cliffs')
        if exp4 and 'per_rep' in exp4 and 'hybrid' in exp4:
            per_rep_rmse = [v.get('rmse_cliff', float('nan'))
                           for v in exp4['per_rep'].values()]
            valid_rmse = [r for r in per_rep_rmse if not np.isnan(r)]
            best_rep_rmse = min(valid_rmse) if valid_rmse else float('nan')
            hybrid_w = exp4['hybrid'].get('weighted', {})
            hybrid_rmse = hybrid_w.get('rmse_cliff', float('nan'))
            if not np.isnan(best_rep_rmse) and not np.isnan(hybrid_rmse) and best_rep_rmse > 0:
                reduction = (best_rep_rmse - hybrid_rmse) / best_rep_rmse * 100
                ds_summary['cliff_rmse_reduction_pct'] = reduction
                cliff_reductions.append(reduction)
                print(f"    RMSE_cliff reduction: {reduction:.1f}% "
                      f"(best_rep={best_rep_rmse:.4f} -> hybrid={hybrid_rmse:.4f})")
            ds_summary['n_cliff_compounds'] = exp4.get('cliff_definition', {}).get('n_cliff_compounds', 0)

        # Exp 5: kNN improvement vs best individual rep
        exp5 = ds_results.get('s05_knn_prediction')
        if exp5 and 'conditions' in exp5:
            conds = exp5['conditions']
            hw_scores = [v['score_mean'] for k, v in conds.items()
                        if v.get('condition') == 'hybrid_weighted']
            # Compare against best individual rep kNN baseline
            per_rep = exp5.get('per_rep_baselines', {})
            if hw_scores and per_rep:
                best_rep_knn = max(v['score_mean'] for v in per_rep.values())
                delta_knn = max(hw_scores) - best_rep_knn
                ds_summary['knn_delta_r2'] = delta_knn
                knn_deltas.append(delta_knn)
                print(f"    kNN delta R2 (vs best rep): {delta_knn:+.4f}")

        # Exp 6: AD correlation (hybrid_weighted vs concat baseline)
        exp6 = ds_results.get('s06_applicability_domain')
        if exp6 and 'distance_error_correlation' in exp6:
            for k_key, corrs in exp6['distance_error_correlation'].items():
                hw = corrs.get('hybrid_weighted')
                # Compare against concat baseline (the main narrative claim)
                baseline = corrs.get('concat', corrs.get('hybrid'))
                baseline_label = 'concat' if 'concat' in corrs else 'hybrid_uw'
                if hw is not None and baseline is not None:
                    delta_ad = hw - baseline
                    ds_summary['ad_corr_delta'] = delta_ad
                    ds_summary['ad_corr_weighted'] = hw
                    ds_summary['ad_corr_baseline'] = baseline
                    ad_deltas.append(delta_ad)
                    print(f"    AD corr: hybrid_weighted={hw:.4f}, "
                          f"{baseline_label}={baseline:.4f} "
                          f"(delta={delta_ad:+.4f})")
                    break

        # Exp 7: noise FP reduction
        exp7 = ds_results.get('s07_noise_bridge')
        if exp7 and 'fp_rate_reduction_pct' in exp7:
            fp_red = exp7['fp_rate_reduction_pct']
            ds_summary['noise_fp_reduction_pct'] = fp_red
            fp_reductions.append(fp_red)
            print(f"    Noise FP reduction: {fp_red:.1f}%")

        # Exp 8: optimal dim ratio
        exp8 = ds_results.get('s08_optimal_dimension')
        if exp8 and 'optimal' in exp8:
            opt = exp8['optimal']
            knn_opt = opt.get('best_knn_budget', 0)
            rf_opt = opt.get('best_rf_budget', 0)
            if rf_opt > 0:
                ratio = knn_opt / rf_opt
                ds_summary['optimal_dim_ratio'] = ratio
                dim_ratios.append(ratio)
                print(f"    Optimal dim ratio (kNN/RF): {ratio:.2f}")

        results['per_dataset'][ds_name] = ds_summary

    # Summary statistics
    print("\n  Aggregate summary:")
    for name, values in [('SAC_delta', sac_deltas),
                          ('RMSE_cliff_reduction_%', cliff_reductions),
                          ('kNN_delta_R2', knn_deltas),
                          ('AD_corr_delta', ad_deltas),
                          ('noise_FP_reduction_%', fp_reductions),
                          ('optimal_dim_ratio', dim_ratios)]:
        if values:
            arr = np.array(values)
            results['summary'][name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'n_datasets': len(arr),
                'values': [float(v) for v in arr],
            }
            print(f"    {name}: mean={np.mean(arr):.4f} +/- {np.std(arr):.4f} "
                  f"(n={len(arr)})")

            # Wilcoxon test if enough samples
            if len(arr) >= 3:
                try:
                    stat, p = wilcoxon(arr)
                    results['summary'][name]['wilcoxon_p'] = float(p)
                    print(f"      Wilcoxon p={p:.4f}")
                except Exception:
                    pass

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S09: Cross-dataset summary')
    parser.add_argument('--datasets', nargs='+', default=['logd', 'caco2', 'herg', 'qm9'],
                        help='Datasets to aggregate (default: logd caco2 herg qm9)')
    parser.add_argument('--results-dir', type=str, default='tests/results',
                        help='Directory containing experiment result subdirs')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for summary JSON')
    args = parser.parse_args()

    print(f"Datasets: {args.datasets}")
    print(f"Results dir: {args.results_dir}")

    # Load results from each experiment for each dataset
    all_dataset_results = {}
    for ds in args.datasets:
        ds_results = {}
        for exp_id in EXPERIMENT_IDS:
            r = load_experiment_results(args.results_dir, exp_id, ds)
            if r:
                ds_results[exp_id] = r
                print(f"  Loaded {exp_id}/{ds}.json")
            else:
                print(f"  Missing {exp_id}/{ds}.json")
        if ds_results:
            all_dataset_results[ds] = ds_results
        else:
            print(f"  WARNING: no results found for {ds}, skipping")

    if not all_dataset_results:
        print("No results found for any dataset, exiting.")
        sys.exit(1)

    results = run(all_dataset_results)
    save_experiment_results(results, args.output, 's09_cross_dataset_summary')
