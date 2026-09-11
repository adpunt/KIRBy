#!/usr/bin/env python3
"""F15 Pruning Analysis: CV-style validation of strategy pruning rules.

Strategies are the ONLY prunable dimension — evaluators and importance methods
always produce winners (established in F12 preliminary analysis).

For each dataset's F15 results:
- Groups configs by (importance, evaluator, model, budget) — strategies compete within groups
- For each fold k as the decision fold, applies pruning rules
- Checks if the overall winner strategy survives pruning
- Reports which rules are safe across ALL folds

Rules tested:
- Rule 1: Prune strategies never top-1 in any group on decision fold
- Rule 2: Prune strategies in top-3 in <X% of groups (X = 5, 10, 15, 20, 25, 30)
- Rule 2b: Prune strategies in top-2 in <X% of groups (X = 5, 10, 15, 20, 25, 30)
- Rule 3: Prune strategies top-1 in <X% of groups (X = 1, 2, 5, 10)

Usage:
    python tests/analysis/f15_pruning_analysis.py
    python tests/analysis/f15_pruning_analysis.py --results-dir tests/results/framework/f15_adaptive_pruning
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results(results_dir):
    """Load all F15 JSON result files, merging per-model splits.

    Handles both formats:
    - Single file: {dataset}.json (all models)
    - Split files: {dataset}_{model}.json (one model each, merged)
    """
    results_dir = Path(results_dir)
    datasets = {}
    for f in sorted(results_dir.glob('*.json')):
        with open(f) as fh:
            data = json.load(fh)
        if not data['results']:
            print(f"  SKIP {f.name}: 0 configs")
            continue

        ds_name = data['dataset']
        if ds_name not in datasets:
            datasets[ds_name] = data
        else:
            # Merge: append results, union budgets and models
            existing = datasets[ds_name]
            existing['results'].extend(data['results'])
            existing['config']['budgets'] = sorted(set(
                existing['config']['budgets'] + data['config']['budgets']))
            existing['config']['models'] = sorted(set(
                existing['config']['models'] + data['config']['models']))
            incomplete = not data.get('complete', True) or not existing.get('complete', True)
            if incomplete:
                existing['complete'] = False

    for ds_name, data in datasets.items():
        n_models = len(data['config']['models'])
        n_results = len(data['results'])
        complete = data.get('complete', True)
        status = '' if complete else ' (PARTIAL)'
        print(f"  {ds_name}: {n_results} configs, {n_models} models{status}")

    return datasets


def get_group_key(r):
    return (r['importance'], r['evaluator'], r['model'], r['budget'])


def analyze_dataset(data):
    """Run full pruning analysis on one dataset. Returns dict of results."""
    results = [r for r in data['results'] if r['n_valid_folds'] == data['n_folds']]
    strategies = data['config']['strategies']
    n_folds = data['n_folds']
    # A results file with no direction key cannot be ranked. Refuse rather
    # than default to higher-is-better, which inverts every rmse conclusion.
    # Fixed 2026-09-10 (metric-direction audit).
    if data.get('lower_is_better') is None:
        raise ValueError(
            'this results file records no lower_is_better, so its scores '
            'cannot be ordered; re-run it through save_experiment_results, '
            'which stamps the direction from the registry')
    lower_is_better = bool(data['lower_is_better'])

    # Direction-aware comparison helpers
    def _best(items, key):
        """Return best item (min if lower_is_better, max otherwise)."""
        return min(items, key=key) if lower_is_better else max(items, key=key)

    def _best_score(scores):
        """Return best score from a collection."""
        return min(scores) if lower_is_better else max(scores)

    def _rank_sort(items, key):
        """Sort items best-first."""
        return sorted(items, key=key, reverse=not lower_is_better)

    groups = defaultdict(list)
    for r in results:
        groups[get_group_key(r)].append(r)
    n_groups = len(groups)

    # Overall winner
    overall_best = _best(results, key=lambda r: r['score_mean'])

    # Per-strategy overall mean
    strat_means = {}
    for s in strategies:
        scores = [r['score_mean'] for r in results if r['strategy'] == s]
        strat_means[s] = np.mean(scores) if scores else float('nan')

    # Per-(model, budget) winner strategy and score — these are what pruning must preserve
    mb_winners = {}   # (model, budget) -> winning strategy
    mb_scores = {}    # (model, budget) -> {strategy: score_mean}
    mb_best_score = {}  # (model, budget) -> best score
    for (model, budget) in sorted(set((r['model'], r['budget']) for r in results)):
        mb_results = [r for r in results if r['model'] == model and r['budget'] == budget]
        best = _best(mb_results, key=lambda r: r['score_mean'])
        mb_winners[(model, budget)] = best['strategy']
        mb_best_score[(model, budget)] = best['score_mean']
        mb_scores[(model, budget)] = {r['strategy']: r['score_mean'] for r in mb_results}

    def _check_winners(kept_strategies):
        """Check winners and compute regret for lost ones.
        Returns (n_safe, n_total, lost_with_regret) where lost_with_regret is
        [(mb, winner_strat, winner_score, best_kept_score, regret), ...]."""
        kept_set = set(kept_strategies)
        lost = []
        for mb, winner_strat in mb_winners.items():
            if winner_strat not in kept_set:
                winner_score = mb_best_score[mb]
                # Best surviving strategy for this (model, budget)
                kept_scores = {s: sc for s, sc in mb_scores[mb].items() if s in kept_set}
                if kept_scores:
                    best_kept = _best_score(kept_scores.values())
                    regret = abs(winner_score - best_kept)
                else:
                    best_kept = float('nan')
                    regret = float('nan')
                lost.append((mb, winner_strat, winner_score, best_kept, regret))
        n_safe = len(mb_winners) - len(lost)
        return n_safe, len(mb_winners), lost

    # === Test pruning rules across all folds ===
    rule_results = {}

    def _test_rule(rule_name, top_n, thresholds):
        """Generic rule tester: prune strategies in top-N in <X% of groups."""
        for threshold_pct in thresholds:
            folds = []
            for k in range(n_folds):
                topn_counts = defaultdict(int)
                for group_configs in groups.values():
                    ranked = _rank_sort(group_configs, key=lambda r: r['fold_scores'][k])
                    for r in ranked[:top_n]:
                        topn_counts[r['strategy']] += 1
                threshold = n_groups * threshold_pct / 100
                kept = [s for s in strategies if topn_counts.get(s, 0) >= threshold]
                pruned = [s for s in strategies if topn_counts.get(s, 0) < threshold]
                n_safe, n_total, lost = _check_winners(kept)
                folds.append({
                    'fold': k, 'kept': len(kept), 'pruned': len(pruned),
                    'pruned_names': pruned, 'winners_safe': n_safe,
                    'winners_total': n_total, 'winners_lost': lost,
                })
            rule_results[f'{rule_name}_lt{threshold_pct}pct'] = folds

    # Rule 1: Never top-1 (equivalent to top-1 in <1 group, i.e. 0%)
    rule1_folds = []
    for k in range(n_folds):
        top1_strategies = set()
        for group_configs in groups.values():
            ranked = _rank_sort(group_configs, key=lambda r: r['fold_scores'][k])
            top1_strategies.add(ranked[0]['strategy'])
        pruned = [s for s in strategies if s not in top1_strategies]
        kept = [s for s in strategies if s in top1_strategies]
        n_safe, n_total, lost = _check_winners(kept)
        rule1_folds.append({
            'fold': k, 'kept': len(kept), 'pruned': len(pruned),
            'pruned_names': pruned, 'winners_safe': n_safe,
            'winners_total': n_total, 'winners_lost': lost,
        })
    rule_results['rule1_never_top1'] = rule1_folds

    # Rule 2: Top-3 in <X%
    _test_rule('rule2_top3', 3, [5, 10, 15, 20, 25, 30])

    # Rule 2b: Top-2 in <X%
    _test_rule('rule2b_top2', 2, [5, 10, 15, 20, 25, 30])

    # Rule 3: Top-1 in <X%
    _test_rule('rule3_top1', 1, [1, 2, 5, 10])

    return {
        'dataset': data['dataset'],
        'n_samples': data['n_samples'],
        'n_reps': data['n_reps'],
        'n_folds': n_folds,
        'n_configs': len(results),
        'n_groups': n_groups,
        'n_strategies': len(strategies),
        'strategies': strategies,
        'overall_best': overall_best,
        'mb_winners': mb_winners,
        'strat_means': strat_means,
        'lower_is_better': lower_is_better,
        'strat_ranking': sorted(strategies, key=lambda s: strat_means[s], reverse=not lower_is_better),
        'rule_results': rule_results,
    }


def print_dataset_report(analysis):
    """Print detailed report for one dataset."""
    a = analysis
    ob = a['overall_best']
    mb = a['mb_winners']
    print(f"\n{'='*70}")
    print(f"  {a['dataset']} ({a['n_samples']} samples, {a['n_reps']} reps)")
    print(f"  {a['n_configs']} configs, {a['n_groups']} groups, {a['n_strategies']} strategies, {a['n_folds']} folds")
    direction = 'lower=better' if a.get('lower_is_better') else 'higher=better'
    print(f"  Metric: {direction}")
    print(f"  Overall best: {ob['model']}/{ob['strategy']}/{ob['importance']}/b={ob['budget']}/ev={ob['evaluator']} = {ob['score_mean']:.4f}")
    print(f"  {len(mb)} (model, budget) winners to preserve")
    print(f"{'='*70}")

    # Per-(model, budget) winners
    print(f"\n  Per-(model, budget) winning strategies:")
    for (model, budget), strat in sorted(mb.items()):
        print(f"    {model:10s} b={budget:4d}: {strat}")

    # Strategy ranking
    print(f"\n  Strategy ranking (overall mean across all configs):")
    winner_strats = set(mb.values())
    for i, s in enumerate(a['strat_ranking']):
        marker = ' <-- mb winner' if s in winner_strats else ''
        print(f"    {i+1:2d}. {s:20s}: {a['strat_means'][s]:.4f}{marker}")

    # Per-rule summary
    print(f"\n  Pruning rules (safe = ALL {len(mb)} model×budget winners kept on ALL folds):")
    for rule_name, folds in a['rule_results'].items():
        all_safe = all(f['winners_safe'] == f['winners_total'] for f in folds)
        avg_pruned = np.mean([f['pruned'] for f in folds])

        # Compute max and mean regret across all folds
        all_regrets = []
        for f in folds:
            for (_, _, _, _, regret) in f['winners_lost']:
                if regret == regret:  # skip NaN
                    all_regrets.append(regret)

        if all_safe:
            print(f"    {rule_name:25s}: SAFE   — avg {avg_pruned:.1f} pruned")
        else:
            worst_fold = min(folds, key=lambda f: f['winners_safe'])
            lost_folds = [f['fold'] for f in folds if f['winners_safe'] < f['winners_total']]
            n_lost = worst_fold['winners_total'] - worst_fold['winners_safe']
            max_regret = max(all_regrets) if all_regrets else 0
            mean_regret = np.mean(all_regrets) if all_regrets else 0
            print(f"    {rule_name:25s}: UNSAFE — avg {avg_pruned:.1f} pruned, "
                  f"lost on folds {lost_folds} "
                  f"(worst: {n_lost}/{worst_fold['winners_total']} lost, "
                  f"regret: mean={mean_regret:.4f} max={max_regret:.4f})")
            # Show which (model, budget) winners were lost on worst fold with regret
            for (m_b, strat, winner_sc, kept_sc, regret) in worst_fold['winners_lost'][:5]:
                model, budget = m_b
                print(f"      {model}/b={budget}: winner={strat} ({winner_sc:.4f}) "
                      f"→ best kept ({kept_sc:.4f}), regret={regret:.4f}")
            if len(worst_fold['winners_lost']) > 5:
                print(f"      ... and {len(worst_fold['winners_lost'])-5} more")


def print_cross_dataset_summary(analyses):
    """Print pruning rule comparison across all datasets.

    For each rule and dataset: how many strategies pruned, how many
    model×budget winners lost, and what's the average regret when
    a winner IS lost. This is for tuning pruning thresholds.
    """
    if len(analyses) < 2:
        return

    print(f"\n{'='*90}")
    print(f"  PRUNING RULE COMPARISON ({len(analyses)} datasets)")
    print(f"{'='*90}")

    rule_names = list(analyses[0]['rule_results'].keys())
    ds_short = [a['dataset'][:15] for a in analyses]

    # Table: for each rule, show per-dataset (avg_pruned, winners_lost, avg_regret)
    print(f"\n  Per-dataset: strategies pruned / winners lost / avg regret (across folds)")
    print()
    header = f"  {'Rule':25s}" + "".join(f" | {n:>15s}" for n in ds_short)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rule_name in rule_names:
        cells = []
        for a in analyses:
            folds = a['rule_results'][rule_name]
            avg_pruned = np.mean([f['pruned'] for f in folds])

            # Collect all regrets across all folds
            all_regrets = []
            total_lost = 0
            for f in folds:
                for (_, _, _, _, regret) in f['winners_lost']:
                    if regret == regret:  # skip NaN
                        all_regrets.append(regret)
                total_lost += len(f['winners_lost'])

            avg_lost = total_lost / len(folds)
            if all_regrets:
                avg_regret = np.mean(all_regrets)
                cells.append(f"{avg_pruned:.0f}p {avg_lost:.1f}L r={avg_regret:.4f}")
            else:
                cells.append(f"{avg_pruned:.0f}p   SAFE")

        row = f"  {rule_name:25s}" + "".join(f" | {c:>15s}" for c in cells)
        print(row)

    # Overall best per dataset (sanity check)
    print(f"\n  Overall best config per dataset:")
    for a in analyses:
        ob = a['overall_best']
        direction = '↓' if a.get('lower_is_better') else '↑'
        print(f"    {a['dataset']:25s}: {ob['strategy']} "
              f"[{ob['model']}/b={ob['budget']}/ev={ob['evaluator']} = {ob['score_mean']:.4f} {direction}]")


def main():
    parser = argparse.ArgumentParser(description='F15 Pruning Analysis')
    parser.add_argument('--results-dir', type=str,
                        default='tests/results/framework/f15_adaptive_pruning',
                        help='Directory containing F15 JSON results')
    parser.add_argument('--min-budget', type=int, default=100,
                        help='Exclude results with budget below this (default: 100)')
    args = parser.parse_args()

    print("F15 Adaptive Pruning Analysis")
    print("Strategies are the ONLY prunable dimension.")
    if args.min_budget > 0:
        print(f"Excluding budgets < {args.min_budget}")
    print()

    datasets = load_results(args.results_dir)
    if not datasets:
        print("No results found.")
        return

    # Filter out low budgets (keep b=50 for abagym which only has small budgets)
    if args.min_budget > 0:
        for ds_name, data in datasets.items():
            if 'abagym' in ds_name:
                continue  # keep all budgets for abagym
            data['results'] = [r for r in data['results'] if r['budget'] >= args.min_budget]
            data['config']['budgets'] = [b for b in data['config']['budgets'] if b >= args.min_budget]

    print(f"Found {len(datasets)} dataset(s): {sorted(datasets.keys())}")

    analyses = []
    for name, data in sorted(datasets.items()):
        a = analyze_dataset(data)
        analyses.append(a)
        print_dataset_report(a)

    print_cross_dataset_summary(analyses)


if __name__ == '__main__':
    main()
