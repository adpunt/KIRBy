#!/usr/bin/env python3
"""Print hybrid vs best single-rep baseline from precompute cache.

Loads the .pkl SearchResult files from ~/.cache/kirby/hybrid_search/,
extracts baselines and best hybrids per model, prints a comparison table.

Usage (on server):
    python tests/analysis/print_hybrid_vs_baseline.py
    python tests/analysis/print_hybrid_vs_baseline.py --domain molecular
    python tests/analysis/print_hybrid_vs_baseline.py caco2 herg_ki
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))


def load_completed_searches(cache_dir):
    """Load all completed .pkl SearchResult files (skip .ckpt.pkl)."""
    results = {}
    for f in sorted(cache_dir.glob('*.pkl')):
        if '.ckpt.' in f.name:
            continue
        try:
            sr = pickle.loads(f.read_bytes())
            ds = getattr(sr, 'dataset_name', None)
            if ds:
                results.setdefault(ds, []).append(sr)
        except Exception:
            pass
    return results


def print_comparison(search_results, domain_filter=None):
    """Print hybrid vs baseline table from SearchResult objects."""

    # Group by dataset
    all_rows = []

    for ds_name in sorted(search_results.keys()):
        sr_list = search_results[ds_name]

        # Use first SearchResult for metric info
        sr0 = sr_list[0]
        metric = sr0.metric_name
        lower = sr0.lower_is_better

        # Merge baselines across all conditions (they should be identical
        # for same reps, but take the union)
        baselines = {}
        for sr in sr_list:
            if sr.baselines:
                for rep, model_scores in sr.baselines.items():
                    if rep.startswith('_'):
                        continue
                    baselines.setdefault(rep, {}).update(model_scores)

        # Merge best_by_model_budget across conditions (keep the best)
        best_hybrids = {}  # model -> budget -> config
        for sr in sr_list:
            if not sr.best_by_model_budget:
                continue
            for model_name, budget_dict in sr.best_by_model_budget.items():
                best_hybrids.setdefault(model_name, {})
                for budget, config in budget_dict.items():
                    existing = best_hybrids[model_name].get(budget)
                    if existing is None:
                        best_hybrids[model_name][budget] = config
                    else:
                        es = existing.get('score_mean', 0)
                        ns = config.get('score_mean', 0)
                        if (lower and ns < es) or (not lower and ns > es):
                            best_hybrids[model_name][budget] = config

        if not baselines or not best_hybrids:
            continue

        # For each model: best single rep vs best hybrid
        models = sorted(set(best_hybrids.keys()) &
                        set(m for rep_scores in baselines.values()
                            for m in rep_scores if m != 'dimension'))

        for model in models:
            # Best single rep for this model
            best_base_score = float('inf') if lower else float('-inf')
            best_base_rep = ''
            for rep, model_scores in baselines.items():
                s = model_scores.get(model)
                if s is None:
                    continue
                try:
                    s = float(s)
                except (TypeError, ValueError):
                    continue
                if (lower and s < best_base_score) or \
                   (not lower and s > best_base_score):
                    best_base_score = s
                    best_base_rep = rep

            # Best hybrid for this model (across all budgets)
            best_h_score = float('inf') if lower else float('-inf')
            best_h_budget = 0
            for budget, config in best_hybrids[model].items():
                s = config.get('score_mean', 0)
                if (lower and s < best_h_score) or \
                   (not lower and s > best_h_score):
                    best_h_score = s
                    best_h_budget = budget

            if best_base_rep == '':
                continue

            delta = (best_base_score - best_h_score) if lower \
                else (best_h_score - best_base_score)

            all_rows.append({
                'dataset': ds_name,
                'metric': metric,
                'lower': lower,
                'model': model,
                'base_score': best_base_score,
                'base_rep': best_base_rep,
                'hybrid_score': best_h_score,
                'hybrid_budget': best_h_budget,
                'delta': delta,
            })

    if not all_rows:
        print("No completed searches found.")
        return

    # Sort by delta descending
    all_rows.sort(key=lambda r: -r['delta'])

    # Print
    wins = sum(1 for r in all_rows if r['delta'] > 0.005)
    ties = sum(1 for r in all_rows if abs(r['delta']) <= 0.005)
    losses = sum(1 for r in all_rows if r['delta'] < -0.005)

    print(f"\nHYBRID vs BEST SINGLE REP ({len(all_rows)} comparisons, "
          f"{len(set(r['dataset'] for r in all_rows))} datasets)")
    print(f"Wins: {wins}  |  Ties: {ties}  |  Losses: {losses}")
    print()
    sep = '\t'
    header = sep.join(['Delta', 'Dataset', 'Metric', 'Direction', 'Model',
                       'Baseline', 'Best Rep', 'Hybrid', 'Budget', 'Win'])
    print(header)
    for r in all_rows:
        direction = 'lower=better' if r['lower'] else 'higher=better'
        flag = 'win' if r['delta'] > 0.005 else \
               ('loss' if r['delta'] < -0.005 else 'tie')
        print(sep.join([
            f"{r['delta']:+.4f}",
            r['dataset'],
            r['metric'],
            direction,
            r['model'],
            f"{r['base_score']:.4f}",
            r['base_rep'],
            f"{r['hybrid_score']:.4f}",
            str(r['hybrid_budget']),
            flag,
        ]))

    # Per-model summary
    print()
    model_stats = defaultdict(lambda: {'w': 0, 't': 0, 'l': 0, 'deltas': []})
    for r in all_rows:
        m = model_stats[r['model']]
        m['deltas'].append(r['delta'])
        if r['delta'] > 0.005:
            m['w'] += 1
        elif r['delta'] < -0.005:
            m['l'] += 1
        else:
            m['t'] += 1

    import numpy as np
    print(sep.join(['Model', 'Wins', 'Ties', 'Losses', 'Mean Delta',
                    'Min Delta', 'Max Delta']))
    for model in sorted(model_stats.keys()):
        s = model_stats[model]
        d = s['deltas']
        print(sep.join([
            model, str(s['w']), str(s['t']), str(s['l']),
            f"{np.mean(d):+.4f}", f"{min(d):+.4f}", f"{max(d):+.4f}",
        ]))


def main():
    parser = argparse.ArgumentParser(
        description='Print hybrid vs baseline from precompute cache.')
    parser.add_argument('datasets', nargs='*',
                        help='Specific datasets (default: all)')
    parser.add_argument('--domain', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else \
        Path(os.environ.get('KIRBY_CACHE_DIR',
             str(Path.home() / '.cache' / 'kirby'))) / 'hybrid_search'

    print(f"Loading from {cache_dir}...")
    all_searches = load_completed_searches(cache_dir)
    print(f"Found {sum(len(v) for v in all_searches.values())} completed "
          f"searches across {len(all_searches)} datasets")

    if args.datasets:
        all_searches = {k: v for k, v in all_searches.items()
                        if k in args.datasets}

    if args.domain:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from precompute_hybrids import _get_domain
        all_searches = {k: v for k, v in all_searches.items()
                        if _get_domain(k) == args.domain}

    print_comparison(all_searches, domain_filter=args.domain)


if __name__ == '__main__':
    main()
