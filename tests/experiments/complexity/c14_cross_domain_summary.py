#!/usr/bin/env python3
"""C14: Cross-Domain Summary — hybrid improvement over best single rep.

Paper: Complexity
Conclusion: KIRBy hybrid consistently improves over best individual rep
    across domains (molecular, antibody, TCGA).
Dependencies: None (self-contained, runs find_best_hybrid per task).

For each task:
  1. Load data + reps via load_data (supports molecular, antibody, TCGA)
  2. Evaluate each individual rep with best model (domain-aware metric)
  3. Run find_best_hybrid to get best hybrid config
  4. Record: best_single_rep_name, best_single_score, hybrid_score, delta, pct_improvement
  5. Track per-fold scores for statistical testing

Aggregate across all tasks:
  - Wilcoxon signed-rank test (paired: hybrid vs best_single per task)
  - Win rate: fraction of tasks where hybrid > best single
  - Mean/median delta and pct improvement
  - Per-domain breakdown

Usage:
    python c14_cross_domain_summary.py --tasks logd,caco2,herg --reps all --mode full
    python c14_cross_domain_summary.py --tasks logd,flab_thermo_tresanco,tcga_luad_vs_lusc --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import datetime
import numpy as np
from scipy.stats import wilcoxon

from experiments._shared import (
    load_data, get_domain_metric, get_model_factories,
    evaluate_single_rep_cv, cv_split,
    save_experiment_results, convert_numpy,
    find_best_hybrid, get_config,
    extract_top3_base_reps,
    INTERNAL_EVALUATORS, SCREENING_BUDGETS, FULL_BUDGETS,
    get_best_search_for_downstream,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

EXPERIMENT_ID = 'c14_cross_domain_summary'


# =========================================================================
# Per-task evaluation
# =========================================================================

def _evaluate_task(task_name, rep_preset, config):
    """Run single-rep baselines + find_best_hybrid for one task.

    Returns dict with:
        task, domain, metric_name, lower_is_better,
        best_single_rep, best_single_score, best_single_fold_scores,
        hybrid_score, hybrid_fold_scores,
        delta, pct_improvement, n_reps, n_samples
    """
    print(f"\n  --- Task: {task_name} ---")

    # Load data
    data = load_data(task_name, rep_preset)
    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    domain = data.get('domain', 'molecular')
    task_type = data['task_type']

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    model_names = config['models']
    n_samples = len(labels)

    print(f"    Domain: {domain}, Metric: {metric_name} "
          f"({'lower better' if lower_is_better else 'higher better'})")
    print(f"    {n_samples} samples, {len(reps)} reps, {n_folds}-fold CV")

    # -----------------------------------------------------------------
    # Load augmentation reps from DATASET_REPS (if configured)
    # -----------------------------------------------------------------
    aug_rep_names = DATASET_REPS.get(task_name, {}).get('augmentation', [])
    aug_dict = None
    if aug_rep_names:
        missing = [r for r in aug_rep_names if r not in reps]
        if missing:
            print(f"    Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(task_name, rep_preset='all', rep_names=missing)
            aug_dict = {}
            for rn, arr in aug_data['reps'].items():
                if int(np.isnan(arr).sum()) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                aug_dict[rn] = arr
        else:
            aug_dict = {r: reps[r] for r in aug_rep_names}
        print(f"    Augmentations ({len(aug_dict)}): "
              f"{', '.join(f'{k}={v.shape[1]}' for k, v in aug_dict.items())}")

    # -----------------------------------------------------------------
    # 1. Evaluate each individual rep with each model, track fold scores
    # -----------------------------------------------------------------
    factories = get_model_factories(model_names, task_type=task_type,
                                    n_samples=n_samples)

    best_single_score = None
    best_single_rep = None
    best_single_model = None
    best_single_fold_scores = None
    all_rep_results = {}

    for rep_name, X in sorted(reps.items()):
        rep_best_score = None
        rep_best_model = None
        rep_best_folds = None

        for model_name, factory in factories.items():
            # Collect per-fold scores (not just mean/std)
            fold_scores = []
            for train_idx, val_idx in cv_split(groups, n_folds, X, labels,
                                               mode=mode):
                try:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X[train_idx])
                    X_val = scaler.transform(X[val_idx])

                    model = factory()
                    model.fit(X_train, labels[train_idx])
                    score = metric_fn(labels[val_idx], model.predict(X_val))
                    fold_scores.append(float(score))
                except Exception:
                    continue

            if not fold_scores:
                continue

            mean_score = float(np.mean(fold_scores))

            # Track best model for this rep
            if rep_best_score is None or _is_better(mean_score, rep_best_score,
                                                     lower_is_better):
                rep_best_score = mean_score
                rep_best_model = model_name
                rep_best_folds = fold_scores

        if rep_best_score is not None:
            all_rep_results[rep_name] = {
                'best_model': rep_best_model,
                'score': rep_best_score,
                'fold_scores': rep_best_folds,
            }

            # Track overall best single rep
            if best_single_score is None or _is_better(
                    rep_best_score, best_single_score, lower_is_better):
                best_single_score = rep_best_score
                best_single_rep = rep_name
                best_single_model = rep_best_model
                best_single_fold_scores = rep_best_folds

        print(f"      {rep_name:<25} best_model={rep_best_model:<10} "
              f"score={rep_best_score:.4f}" if rep_best_score is not None
              else f"      {rep_name:<25} FAILED")

    if best_single_score is None:
        print(f"    WARNING: no single rep succeeded for {task_name}")
        return None

    print(f"    Best single rep: {best_single_rep} ({best_single_model}) "
          f"= {best_single_score:.4f}")

    # -----------------------------------------------------------------
    # 2. Best hybrid via pool helper (best across all common models)
    # -----------------------------------------------------------------
    print(f"    Picking best hybrid via pool helper...")

    speed = config.get('search_speed', 'full')
    best_entry = None
    best_model_name = None
    search = None
    for candidate_model in ['rf', 'svm', 'lightgbm', 'mlp', 'knn']:
        try:
            s, m = get_best_search_for_downstream(
                reps, labels, groups,
                downstream_model=candidate_model, budget=None,
                augmentations=aug_dict if aug_dict else None,
                speed=speed,
                dataset_name=task_name,
                task_type=task_type, metric=metric_name,
            )
        except Exception:
            continue
        if not m:
            continue
        bc = m['best_config']
        score = bc.get('score_mean')
        if score is None:
            continue
        is_better = (best_entry is None) or (
            score < best_entry['score_mean'] if lower_is_better
            else score > best_entry['score_mean'])
        if is_better:
            best_entry = bc
            best_model_name = candidate_model
            search = s

    if search is None or best_entry is None:
        print(f"    WARNING: pool helper returned no result for {task_name}")
        return None

    top3 = extract_top3_base_reps(search)
    if top3:
        print(f"    Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
    hybrid_score = best_entry['score_mean']
    hybrid_fold_scores = best_entry.get('fold_scores', [])

    # Compute delta and pct improvement
    if lower_is_better:
        delta = best_single_score - hybrid_score  # positive = hybrid better
        pct = (delta / abs(best_single_score) * 100) if best_single_score != 0 else 0.0
    else:
        delta = hybrid_score - best_single_score  # positive = hybrid better
        pct = (delta / abs(best_single_score) * 100) if best_single_score != 0 else 0.0

    hybrid_wins = delta > 0

    print(f"    Hybrid: {hybrid_score:.4f}  "
          f"(delta={delta:+.4f}, {pct:+.1f}%, "
          f"{'WIN' if hybrid_wins else 'LOSS'})")

    return {
        'task': task_name,
        'domain': domain,
        'task_type': task_type,
        'metric_name': metric_name,
        'lower_is_better': lower_is_better,
        'n_reps': len(reps),
        'n_samples': n_samples,
        'n_folds': n_folds,
        'best_single_rep': best_single_rep,
        'best_single_model': best_single_model,
        'best_single_score': best_single_score,
        'best_single_fold_scores': best_single_fold_scores,
        'hybrid_score': hybrid_score,
        'hybrid_fold_scores': hybrid_fold_scores,
        'hybrid_config': {
            'model': best_entry.get('model', ''),
            'n_features': best_entry.get('n_features', 0),
            'allocation_method': best_entry.get('allocation_method', ''),
        },
        'delta': delta,
        'pct_improvement': pct,
        'hybrid_wins': hybrid_wins,
        'all_rep_scores': {
            k: v['score'] for k, v in all_rep_results.items()
        },
        'top3_base_reps': [(n, s) for n, s in top3] if top3 else [],
    }


def _is_better(a, b, lower_is_better):
    """Return True if a is better than b under the given metric direction."""
    if lower_is_better:
        return a < b
    return a > b


# =========================================================================
# Aggregation
# =========================================================================

def _aggregate(task_results):
    """Aggregate per-task results into summary statistics + Wilcoxon test.

    Returns dict with:
        per_task, summary, per_domain, wilcoxon_test
    """
    print("\n" + "=" * 70)
    print("  AGGREGATE SUMMARY")
    print("=" * 70)

    deltas = []
    pcts = []
    wins = 0
    total = len(task_results)

    # Per-domain tracking
    domain_results = {}

    for r in task_results:
        deltas.append(r['delta'])
        pcts.append(r['pct_improvement'])
        if r['hybrid_wins']:
            wins += 1

        domain = r['domain']
        if domain not in domain_results:
            domain_results[domain] = {'deltas': [], 'pcts': [], 'wins': 0,
                                       'tasks': []}
        domain_results[domain]['deltas'].append(r['delta'])
        domain_results[domain]['pcts'].append(r['pct_improvement'])
        domain_results[domain]['tasks'].append(r['task'])
        if r['hybrid_wins']:
            domain_results[domain]['wins'] += 1

    deltas_arr = np.array(deltas)
    pcts_arr = np.array(pcts)

    summary = {
        'n_tasks': total,
        'win_rate': wins / total if total > 0 else 0,
        'wins': wins,
        'losses': total - wins,
        'mean_delta': float(np.mean(deltas_arr)),
        'median_delta': float(np.median(deltas_arr)),
        'std_delta': float(np.std(deltas_arr)),
        'mean_pct_improvement': float(np.mean(pcts_arr)),
        'median_pct_improvement': float(np.median(pcts_arr)),
        'std_pct_improvement': float(np.std(pcts_arr)),
    }

    print(f"\n  Win rate: {wins}/{total} ({summary['win_rate']*100:.1f}%)")
    print(f"  Mean delta:   {summary['mean_delta']:+.4f} "
          f"(+/- {summary['std_delta']:.4f})")
    print(f"  Median delta: {summary['median_delta']:+.4f}")
    print(f"  Mean %improvement:   {summary['mean_pct_improvement']:+.1f}%")
    print(f"  Median %improvement: {summary['median_pct_improvement']:+.1f}%")

    # -----------------------------------------------------------------
    # Wilcoxon signed-rank test
    # -----------------------------------------------------------------
    wilcoxon_result = {}
    if total >= 3:
        # Test: are the deltas significantly > 0?
        # (i.e., does hybrid consistently improve over best single rep?)
        try:
            stat, p = wilcoxon(deltas_arr, alternative='greater')
            wilcoxon_result = {
                'statistic': float(stat),
                'p_value': float(p),
                'significant_005': p < 0.05,
                'significant_001': p < 0.01,
                'n': total,
            }
            print(f"\n  Wilcoxon signed-rank (one-sided, hybrid > single):")
            print(f"    statistic={stat:.4f}, p={p:.4f} "
                  f"({'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'})")
        except Exception as e:
            wilcoxon_result = {'error': str(e)}
            print(f"\n  Wilcoxon test failed: {e}")
    else:
        wilcoxon_result = {'error': f'Need >= 3 tasks, got {total}'}
        print(f"\n  Wilcoxon test skipped (need >= 3 tasks, got {total})")

    # -----------------------------------------------------------------
    # Per-domain breakdown
    # -----------------------------------------------------------------
    per_domain = {}
    print(f"\n  Per-domain breakdown:")
    for domain, dr in sorted(domain_results.items()):
        d_arr = np.array(dr['deltas'])
        p_arr = np.array(dr['pcts'])
        n_d = len(d_arr)
        per_domain[domain] = {
            'n_tasks': n_d,
            'win_rate': dr['wins'] / n_d if n_d > 0 else 0,
            'wins': dr['wins'],
            'mean_delta': float(np.mean(d_arr)),
            'median_delta': float(np.median(d_arr)),
            'mean_pct_improvement': float(np.mean(p_arr)),
            'tasks': dr['tasks'],
        }
        # Domain-level Wilcoxon if enough tasks
        if n_d >= 3:
            try:
                stat_d, p_d = wilcoxon(d_arr, alternative='greater')
                per_domain[domain]['wilcoxon_p'] = float(p_d)
            except Exception:
                pass

        # Complexity matching strength: how much does representation choice
        # matter in this domain? High std of deltas = some tasks benefit a lot,
        # some don't = representation matters differently by task.
        # High mean |delta| = representation consistently matters.
        per_domain[domain]['complexity_matching_strength'] = float(np.mean(np.abs(d_arr)))
        per_domain[domain]['delta_std'] = float(np.std(d_arr))

        print(f"    {domain}: {dr['wins']}/{n_d} wins, "
              f"mean_delta={np.mean(d_arr):+.4f}, "
              f"mean_%imp={np.mean(p_arr):+.1f}%, "
              f"matching_strength={np.mean(np.abs(d_arr)):.4f}")

    return {
        'summary': summary,
        'wilcoxon_test': wilcoxon_result,
        'per_domain': per_domain,
    }


# =========================================================================
# Main entry
# =========================================================================

def run(tasks, rep_preset, config, output_dir):
    """Run c14 across all tasks and aggregate.

    Args:
        tasks: list of dataset name strings.
        rep_preset: representation preset (e.g. 'all', 'fast').
        config: fast or full dict.
        output_dir: base output directory.

    Returns:
        Full results dict.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C14: CROSS-DOMAIN SUMMARY")
    print(f"  Tasks: {tasks}")
    print(f"  Reps: {rep_preset}, Mode: {'full' if config['n_splits'] > 1 else 'intermediary'}")
    print("=" * 70)

    task_results = []
    for task_name in tasks:
        try:
            result = _evaluate_task(task_name, rep_preset, config)
            if result is not None:
                task_results.append(result)
        except Exception as e:
            print(f"\n  FAILED: {task_name} — {e}")
            import traceback
            traceback.print_exc()

    if not task_results:
        print("\nNo tasks completed successfully.")
        return {'per_task': [], 'aggregate': {}}

    # Print per-task summary table
    print(f"\n{'='*70}")
    print(f"  {'Task':<30} {'Domain':<10} {'Metric':<8} "
          f"{'BestSingle':>10} {'Hybrid':>10} {'Delta':>10} {'%Imp':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in task_results:
        print(f"  {r['task']:<30} {r['domain']:<10} {r['metric_name']:<8} "
              f"{r['best_single_score']:>10.4f} {r['hybrid_score']:>10.4f} "
              f"{r['delta']:>+10.4f} {r['pct_improvement']:>+7.1f}%")

    # Aggregate
    aggregate = _aggregate(task_results)

    results = {
        'experiment': EXPERIMENT_ID,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'tasks': tasks,
            'rep_preset': rep_preset,
            'n_splits': config['n_splits'],
            'models': config['models'],
            'search_speed': config.get('search_speed', 'full'),
        },
        'per_task': convert_numpy(task_results),
        'aggregate': convert_numpy(aggregate),
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='C14: Cross-domain summary — hybrid vs best single rep')
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated list of dataset names '
                             '(e.g., "logd,caco2,herg,flab_thermo_tresanco,'
                             'tcga_luad_vs_lusc")')
    parser.add_argument('--reps', type=str, default='all',
                        help='Representation preset (fast, all)')
    parser.add_argument('--mode', choices=['intermediary', 'full'],
                        default='intermediary',
                        help='intermediary (1 split) or full (5-fold CV)')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for results JSON')
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    config = get_config(args)

    if not tasks:
        print("ERROR: --tasks must specify at least one dataset name")
        sys.exit(1)

    results = run(tasks, args.reps, config, args.output)
    save_experiment_results(results, args.output, EXPERIMENT_ID)
