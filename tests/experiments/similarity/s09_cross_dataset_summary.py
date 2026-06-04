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
    save_experiment_results, load_experiment_results,
)

# Metrics where lower scores are better (RMSE, MAE).
_LOWER_IS_BETTER_METRICS = {'rmse', 'mae'}


def _metric_lower_is_better(metric_name):
    """Infer metric direction from name.  Default: higher is better."""
    return metric_name.lower() in _LOWER_IS_BETTER_METRICS if metric_name else False


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
    cliff_n_skipped_clf = 0
    knn_deltas = []
    ad_deltas = []
    fp_reductions = []
    fp_skipped_reasons = {}  # ds_name -> reason a dataset was dropped from FP reduction
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
        # Classification datasets are silently skipped (no rmse_cliff key).
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
            else:
                # rmse_cliff absent (e.g., classification dataset)
                cliff_n_skipped_clf += 1
            ds_summary['n_cliff_compounds'] = exp4.get('cliff_definition', {}).get('n_cliff_compounds', 0)

        # Exp 5: kNN improvement vs best individual rep
        exp5 = ds_results.get('s05_knn_prediction')
        if exp5 and 'per_budget' in exp5:
            # Infer metric direction from per-rep baselines. All reps must
            # report the same metric -- mixing directions silently would
            # invert the delta sign for some datasets.
            per_rep = exp5.get('per_rep_baselines', {})
            _metrics_seen = {_prv.get('metric') for _prv in per_rep.values()
                             if _prv.get('metric') is not None}
            if len(_metrics_seen) > 1:
                raise ValueError(
                    f"s05 per_rep_baselines for {ds_name} mix metrics "
                    f"{_metrics_seen}; cannot aggregate kNN delta."
                )
            _exp5_metric = next(iter(_metrics_seen)) if _metrics_seen else None
            _exp5_lib = _metric_lower_is_better(_exp5_metric)
            _best = min if _exp5_lib else max

            # Collect hybrid_weighted scores across all budgets
            hw_scores = []
            for b_res in exp5['per_budget'].values():
                for k, v in b_res.items():
                    if 'hybrid_weighted' in k:
                        hw_scores.append(v['score_mean'])
            # Compare against best individual rep kNN baseline
            if hw_scores and per_rep:
                best_rep_knn = _best(v['score_mean'] for v in per_rep.values())
                best_hybrid_knn = _best(hw_scores)
                # Delta should always be positive = improvement
                if _exp5_lib:  # lower is better
                    delta_knn = best_rep_knn - best_hybrid_knn
                else:
                    delta_knn = best_hybrid_knn - best_rep_knn
                ds_summary['knn_delta'] = delta_knn
                knn_deltas.append(delta_knn)
                _label = _exp5_metric or 'score'
                print(f"    kNN delta {_label} (vs best rep): {delta_knn:+.4f}")

        # Exp 6: AD correlation (hybrid_weighted vs concat baseline).
        # Average across all k_keys instead of picking whichever happens to
        # come first in dict-insertion order (silently mixed across datasets).
        exp6 = ds_results.get('s06_applicability_domain')
        if exp6 and 'distance_error_correlation' in exp6:
            per_k_deltas = []
            per_k_hw = []
            per_k_baseline = []
            baseline_label = 'concat'
            _exp6_missing_concat = False
            for k_key, corrs in exp6['distance_error_correlation'].items():
                hw = corrs.get('hybrid_weighted')
                baseline = corrs.get('concat')
                if baseline is None:
                    _exp6_missing_concat = True
                    continue
                if hw is not None:
                    per_k_deltas.append(hw - baseline)
                    per_k_hw.append(hw)
                    per_k_baseline.append(baseline)
            if _exp6_missing_concat and not per_k_deltas:
                print(f"    WARNING: s06 for {ds_name} has no 'concat' baseline; "
                      f"skipping AD delta.")
            if per_k_deltas:
                delta_ad = float(np.mean(per_k_deltas))
                hw_mean = float(np.mean(per_k_hw))
                base_mean = float(np.mean(per_k_baseline))
                ds_summary['ad_corr_delta'] = delta_ad
                ds_summary['ad_corr_weighted'] = hw_mean
                ds_summary['ad_corr_baseline'] = base_mean
                ds_summary['ad_corr_n_k'] = len(per_k_deltas)
                ad_deltas.append(delta_ad)
                print(f"    AD corr (mean over {len(per_k_deltas)} k): "
                      f"hybrid_weighted={hw_mean:.4f}, "
                      f"{baseline_label}={base_mean:.4f} "
                      f"(delta={delta_ad:+.4f})")

        # Exp 7: noise cliff-contamination reduction (computed from S07 summary
        # + hybrid results). s07 renamed fp_rate → cliff_contamination_rate
        # (the metric is precision of the noise flag w.r.t. cliffs, not classical FPR).
        exp7 = ds_results.get('s07_noise_bridge')
        if exp7:
            hw_fp = exp7.get('hybrid_weighted', {}).get('cliff_contamination_rate')
            best_rep_fp = exp7.get('summary', {}).get('best_rep_cliff_contamination_rate')
            if hw_fp is not None and best_rep_fp is not None and best_rep_fp > 0:
                fp_red = (best_rep_fp - hw_fp) / best_rep_fp * 100
                ds_summary['noise_cliff_contamination_reduction_pct'] = fp_red
                fp_reductions.append(fp_red)
                print(f"    Noise cliff-contamination reduction: {fp_red:.1f}% "
                      f"(best_rep={best_rep_fp:.3f} -> hybrid_w={hw_fp:.3f})")
            else:
                # Dataset dropped from the FP-reduction aggregate. Previously this
                # was silent; log + record WHY so a missing dataset is explainable
                # (SIMA-9). The divide-by-zero case (best_rep_fp == 0, no false
                # positives) is legitimate but must not vanish without trace.
                if best_rep_fp == 0:
                    _reason = 'best_rep_contamination_rate=0 (no false positives; ratio undefined)'
                elif hw_fp is None or best_rep_fp is None:
                    _reason = 'missing cliff_contamination_rate (hybrid or best-rep field absent)'
                else:
                    _reason = f'best_rep_contamination_rate<=0 ({best_rep_fp})'
                fp_skipped_reasons[ds_name] = _reason
                ds_summary['noise_fp_reduction_skipped'] = _reason
                print(f"    [skip] noise FP reduction not computed: {_reason}")

        # Exp 8: optimal dim ratio
        exp8 = ds_results.get('s08_optimal_dimension')
        if exp8 and 'optimal' in exp8:
            opt = exp8['optimal']
            # s08 may write None (via .get('knn')) when no budget was found;
            # treat None and missing the same.
            knn_opt = opt.get('best_knn_budget') or 0
            rf_opt = opt.get('best_rf_budget') or 0
            if rf_opt and knn_opt and rf_opt > 0:
                ratio = knn_opt / rf_opt
                ds_summary['optimal_dim_ratio'] = ratio
                dim_ratios.append(ratio)
                print(f"    Optimal dim ratio (kNN/RF): {ratio:.2f}")

        results['per_dataset'][ds_name] = ds_summary

    # Summary statistics
    print("\n  Aggregate summary:")
    for name, values in [('SAC_delta', sac_deltas),
                          ('RMSE_cliff_reduction_%', cliff_reductions),
                          ('kNN_delta', knn_deltas),
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

            # Wilcoxon test if enough samples. Ratios (centered at 1) must
            # be log-transformed so the test is against 0 in log-space
            # (equivalent to ratio=1); the default wilcoxon tests symmetry
            # around 0, which is meaningless for raw ratios.
            if len(arr) >= 3:
                try:
                    if name == 'optimal_dim_ratio':
                        with np.errstate(divide='ignore', invalid='ignore'):
                            log_arr = np.log(arr[arr > 0])
                        if len(log_arr) >= 3:
                            stat, p = wilcoxon(log_arr)
                            results['summary'][name]['wilcoxon_p_logratio'] = float(p)
                            print(f"      Wilcoxon p (log-ratio vs 1)={p:.4f}")
                    else:
                        stat, p = wilcoxon(arr)
                        results['summary'][name]['wilcoxon_p'] = float(p)
                        print(f"      Wilcoxon p={p:.4f}")
                except Exception:
                    pass

    if cliff_n_skipped_clf:
        print(f"\n  Note: {cliff_n_skipped_clf} dataset(s) skipped in "
              f"cliff_reductions (no rmse_cliff, likely classification).")
        results['summary']['cliff_n_skipped_clf'] = cliff_n_skipped_clf

    if fp_skipped_reasons:
        print(f"\n  Note: {len(fp_skipped_reasons)} dataset(s) dropped from "
              f"noise_FP_reduction (reasons logged per-dataset):")
        for _ds, _why in fp_skipped_reasons.items():
            print(f"    - {_ds}: {_why}")
        results['summary']['noise_fp_reduction_skipped'] = fp_skipped_reasons

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S09: Cross-dataset summary')
    parser.add_argument('--datasets', nargs='+', default=['logd', 'caco2', 'herg_ki', 'qm9'],
                        help='Datasets to aggregate (default: logd caco2 herg_ki qm9)')
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
