#!/usr/bin/env python3
"""General Analysis: Hybrid vs Base Rep Performance.

For each dataset in DATASET_REPS, loads the precomputed hybrid search
(from cache), extracts baselines and best hybrids per model, and
produces a direction-aware comparison table.

Two modes:
  1. --export: Run on server after precompute. Loads data + cache,
     extracts baselines + hybrids, saves analysis JSON.
  2. --analyze: Read an exported JSON (no server/cache needed),
     produce tables and summary statistics.

Default (no flags): export + analyze in one pass.

Usage:
    # Full run on server after precompute:
    python tests/analysis/general_hybrid_vs_base.py

    # Specific datasets:
    python tests/analysis/general_hybrid_vs_base.py caco2 herg_ki

    # By domain:
    python tests/analysis/general_hybrid_vs_base.py --domain molecular

    # Export only (save JSON for local analysis):
    python tests/analysis/general_hybrid_vs_base.py --export -o results.json

    # Analyze from exported JSON (local, no cache needed):
    python tests/analysis/general_hybrid_vs_base.py --analyze results.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Metric direction ────────────────────────────────────────────────────────
# True = lower is better.  Source of truth: PIPELINE_READINESS_TODO.md
METRIC_LOWER_IS_BETTER = {
    'rmse': True,
    'mae': True,
    'r2': False,
    'spearman': False,
    'ci': False,
    'auc': False,
    'f1_weighted': False,
    'mcc': False,
    'accuracy': False,
}


def _is_better(a, b, lower_is_better):
    """True if score *a* is strictly better than score *b*."""
    return a < b if lower_is_better else a > b


def _delta(hybrid, base, lower_is_better):
    """Positive = hybrid is better, regardless of metric direction."""
    return (base - hybrid) if lower_is_better else (hybrid - base)


def _best(scores, lower_is_better):
    """Return the best score from a list, respecting direction."""
    return min(scores) if lower_is_better else max(scores)


# ── Domain detection (matches precompute_hybrids.py) ────────────────────────

def _get_domain(dataset_name):
    if dataset_name.startswith(('tcga_', 'luad', 'brca')):
        return 'tcga'
    if dataset_name.startswith(('davis', 'kiba', 'bindingdb')):
        return 'dta'
    if dataset_name.startswith(('buchwald', 'doyle', 'suzuki', 'bh_')):
        return 'reaction'
    if any(dataset_name.startswith(p) for p in (
            'flab_', 'abagym_', 'abcov_', 'ddg_', 'tz_', 'humanness',
            'ginkgo_', 'graphinity_')):
        return 'antibody'
    if dataset_name.startswith(('nt_', 'gb_')):
        return 'nucleotide'
    return 'molecular'


# ── Export: load from cache and extract comparison data ──────────────────────

def export_dataset(dataset_name, speed='fast', aug_condition='uncapped'):
    """Load precomputed search for one dataset, return comparison dict.

    Returns dict with keys:
        dataset, domain, metric, lower_is_better,
        n_samples, n_reps,
        baselines: {rep: {model: score, dimension: int}},
        hybrids: {model: {budget: {score_mean, score_std, n_features, reps, allocation}}},
        comparisons: [{model, best_base_rep, best_base_score,
                       top3: [(rep, score)],
                       hybrid_score, hybrid_budget, hybrid_n_features,
                       delta, hybrid_wins}]
    """
    from experiments._shared import (
        load_data, get_precomputed_search, SCREENING_BUDGETS,
    )
    from experiments.complexity.complexity_helpers import DATASET_REPS

    ds_config = DATASET_REPS.get(dataset_name)
    if ds_config is None:
        print(f"  {dataset_name}: not in DATASET_REPS, skipping")
        return None

    domain = _get_domain(dataset_name)

    # Only load the reps we actually need (general + augmentation)
    general_names = ds_config.get('general', [])
    aug_names = ds_config.get('augmentation', [])
    needed_reps = sorted(set(general_names + aug_names))

    try:
        data = load_data(dataset_name, rep_names=needed_reps)
    except Exception as e:
        print(f"  {dataset_name}: load failed — {e}")
        return None

    all_reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')

    general_reps = {k: all_reps[k] for k in general_names if k in all_reps}
    aug_reps = {k: all_reps[k] for k in aug_names if k in all_reps}

    if len(general_reps) < 2:
        print(f"  {dataset_name}: only {len(general_reps)} general reps, skipping")
        return None

    # Get precomputed search (hits disk cache)
    try:
        search = get_precomputed_search(
            reps=general_reps,
            labels=labels,
            groups=groups,
            aug_condition=aug_condition,
            augmentations=aug_reps if aug_reps else None,
            speed=speed,
            domain=domain,
        )
    except Exception as e:
        print(f"  {dataset_name}: search failed — {e}")
        return None

    metric = search.metric_name
    lower = search.lower_is_better

    # Sanity-check direction against our lookup table
    expected_lower = METRIC_LOWER_IS_BETTER.get(metric)
    if expected_lower is not None and expected_lower != lower:
        print(f"  WARNING: {dataset_name} metric={metric} "
              f"search says lower_is_better={lower}, "
              f"lookup says {expected_lower}. Using search value.")

    # ── Extract baselines ──
    baselines_raw = search.baselines or {}
    baselines = {}
    for rep_name, model_scores in baselines_raw.items():
        if rep_name.startswith('_'):
            continue
        entry = {}
        for k, v in model_scores.items():
            if k == '_secondary':
                continue
            if k == 'dimension':
                entry[k] = int(v) if v is not None else 0
                continue
            # Model score — convert to float, skip NaN
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isnan(fv):
                entry[k] = round(fv, 6)
        baselines[rep_name] = entry

    # ── Extract best hybrids per model × budget ──
    hybrids = {}
    bymb = search.best_by_model_budget or {}
    for model_name, budget_dict in bymb.items():
        hybrids[model_name] = {}
        for budget, r in budget_dict.items():
            bp = r.get('blueprint')
            hybrids[model_name][int(budget)] = {
                'score_mean': round(float(r.get('score_mean', 0)), 6),
                'score_std': round(float(r.get('score_std', 0)), 6),
                'n_features': int(r.get('n_features', 0)),
                'allocation': r.get('allocation_method', ''),
                'reps': bp.reps if bp else [],
            }

    # ── Build per-model comparisons ──
    comparisons = []
    models_in_search = set()
    for rep_scores in baselines.values():
        models_in_search.update(
            k for k in rep_scores if k != 'dimension')
    models_in_search = sorted(models_in_search)

    for model_name in models_in_search:
        # Collect base rep scores for this model
        rep_scores = []
        for rep_name, scores in baselines.items():
            s = scores.get(model_name)
            if s is not None:
                rep_scores.append((rep_name, s))

        if not rep_scores:
            continue

        # Sort best-first (direction-aware)
        rep_scores.sort(key=lambda x: x[1], reverse=not lower)
        top3 = rep_scores[:3]
        best_base_rep, best_base_score = top3[0]

        # Best hybrid for this model (across all budgets)
        hybrid_score = None
        hybrid_budget = None
        hybrid_n_features = 0
        hybrid_reps = []

        if model_name in hybrids:
            budget_results = hybrids[model_name]
            if budget_results:
                best_budget = (min if lower else max)(
                    budget_results.items(),
                    key=lambda kv: kv[1]['score_mean'],
                )
                hybrid_budget = best_budget[0]
                hybrid_score = best_budget[1]['score_mean']
                hybrid_n_features = best_budget[1]['n_features']
                hybrid_reps = best_budget[1].get('reps', [])

        if hybrid_score is None:
            continue

        d = _delta(hybrid_score, best_base_score, lower)
        wins = _is_better(hybrid_score, best_base_score, lower)

        comparisons.append({
            'model': model_name,
            'best_base_rep': best_base_rep,
            'best_base_score': best_base_score,
            'top3': [(r, s) for r, s in top3],
            'hybrid_score': hybrid_score,
            'hybrid_budget': hybrid_budget,
            'hybrid_n_features': hybrid_n_features,
            'hybrid_reps': hybrid_reps,
            'delta': round(d, 6),
            'hybrid_wins': bool(wins),
        })

    result = {
        'dataset': dataset_name,
        'domain': domain,
        'metric': metric,
        'lower_is_better': lower,
        'n_samples': int(len(labels)),
        'n_reps': len(baselines),
        'baselines': baselines,
        'hybrids': hybrids,
        'comparisons': comparisons,
    }

    n_wins = sum(1 for c in comparisons if c['hybrid_wins'])
    n_total = len(comparisons)
    print(f"  {dataset_name}: {metric} ({'lower' if lower else 'higher'}=better), "
          f"{n_reps_str(baselines)} reps, "
          f"{n_wins}/{n_total} hybrid wins across {n_total} models")

    return result


def n_reps_str(baselines):
    return str(len(baselines))


# ── Analyze: produce tables from exported data ──────────────────────────────

def analyze(results, verbose=True):
    """Produce summary tables from a list of per-dataset result dicts."""

    if not results:
        print("No results to analyze.")
        return {}

    # ── Per-dataset table ──
    rows = []
    for r in results:
        for c in r['comparisons']:
            rows.append({
                'dataset': r['dataset'],
                'domain': r['domain'],
                'metric': r['metric'],
                'lower_is_better': r['lower_is_better'],
                'n_samples': r['n_samples'],
                'model': c['model'],
                'best_base_rep': c['best_base_rep'],
                'best_base_score': c['best_base_score'],
                'hybrid_score': c['hybrid_score'],
                'hybrid_budget': c['hybrid_budget'],
                'hybrid_n_features': c['hybrid_n_features'],
                'delta': c['delta'],
                'hybrid_wins': c['hybrid_wins'],
            })

    if not rows:
        print("No comparison rows produced.")
        return {}

    # ── Aggregate summary ──
    total = len(rows)
    wins = sum(1 for r in rows if r['hybrid_wins'])
    losses = total - wins
    win_rate = wins / total * 100 if total > 0 else 0

    # By domain
    domain_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'deltas': []})
    for r in rows:
        d = domain_stats[r['domain']]
        d['total'] += 1
        if r['hybrid_wins']:
            d['wins'] += 1
        d['deltas'].append(r['delta'])

    # By model
    model_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'deltas': []})
    for r in rows:
        m = model_stats[r['model']]
        m['total'] += 1
        if r['hybrid_wins']:
            m['wins'] += 1
        m['deltas'].append(r['delta'])

    # By dataset
    dataset_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'deltas': [],
                                         'domain': '', 'metric': ''})
    for r in rows:
        ds = dataset_stats[r['dataset']]
        ds['total'] += 1
        ds['domain'] = r['domain']
        ds['metric'] = r['metric']
        if r['hybrid_wins']:
            ds['wins'] += 1
        ds['deltas'].append(r['delta'])

    # ── Print results ──
    if verbose:
        print("\n" + "=" * 80)
        print("GENERAL ANALYSIS: HYBRID vs BASE REP PERFORMANCE")
        print("=" * 80)

        # Aggregate
        print(f"\n{'AGGREGATE':=<80}")
        print(f"  Total comparisons: {total}")
        print(f"  Hybrid wins:       {wins} ({win_rate:.1f}%)")
        print(f"  Hybrid losses:     {losses} ({100 - win_rate:.1f}%)")

        # By domain
        print(f"\n{'BY DOMAIN':=<80}")
        print(f"  {'Domain':<20s} {'Wins':>6s} {'Total':>6s} {'Win%':>7s} "
              f"{'MeanΔ':>10s} {'MedianΔ':>10s}")
        print(f"  {'-'*18:<20s} {'-'*6:>6s} {'-'*6:>6s} {'-'*7:>7s} "
              f"{'-'*10:>10s} {'-'*10:>10s}")
        for domain in sorted(domain_stats.keys()):
            s = domain_stats[domain]
            wr = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0
            mean_d = np.mean(s['deltas'])
            median_d = np.median(s['deltas'])
            print(f"  {domain:<20s} {s['wins']:>6d} {s['total']:>6d} "
                  f"{wr:>6.1f}% {mean_d:>+10.4f} {median_d:>+10.4f}")

        # By model
        print(f"\n{'BY MODEL':=<80}")
        print(f"  {'Model':<20s} {'Wins':>6s} {'Total':>6s} {'Win%':>7s} "
              f"{'MeanΔ':>10s} {'MedianΔ':>10s}")
        print(f"  {'-'*18:<20s} {'-'*6:>6s} {'-'*6:>6s} {'-'*7:>7s} "
              f"{'-'*10:>10s} {'-'*10:>10s}")
        for model in sorted(model_stats.keys()):
            s = model_stats[model]
            wr = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0
            mean_d = np.mean(s['deltas'])
            median_d = np.median(s['deltas'])
            print(f"  {model:<20s} {s['wins']:>6d} {s['total']:>6d} "
                  f"{wr:>6.1f}% {mean_d:>+10.4f} {median_d:>+10.4f}")

        # Per-dataset summary
        print(f"\n{'PER DATASET':=<80}")
        print(f"  {'Dataset':<35s} {'Domain':<12s} {'Metric':<10s} "
              f"{'Wins':>5s} {'Tot':>4s} {'Win%':>6s} {'MeanΔ':>9s}")
        print(f"  {'-'*33:<35s} {'-'*10:<12s} {'-'*8:<10s} "
              f"{'-'*5:>5s} {'-'*4:>4s} {'-'*6:>6s} {'-'*9:>9s}")
        for ds_name in sorted(dataset_stats.keys()):
            s = dataset_stats[ds_name]
            wr = s['wins'] / s['total'] * 100 if s['total'] > 0 else 0
            mean_d = np.mean(s['deltas'])
            print(f"  {ds_name:<35s} {s['domain']:<12s} {s['metric']:<10s} "
                  f"{s['wins']:>5d} {s['total']:>4d} {wr:>5.1f}% "
                  f"{mean_d:>+9.4f}")

        # Detailed per-dataset per-model table (top 20 best/worst deltas)
        print(f"\n{'TOP HYBRID WINS (by delta)':=<80}")
        rows_sorted = sorted(rows, key=lambda r: r['delta'], reverse=True)
        print(f"  {'Dataset':<30s} {'Model':<10s} {'Metric':<8s} "
              f"{'BestBase':>10s} {'Hybrid':>10s} {'Budget':>7s} {'Δ':>9s}")
        print(f"  {'-'*28:<30s} {'-'*8:<10s} {'-'*6:<8s} "
              f"{'-'*10:>10s} {'-'*10:>10s} {'-'*7:>7s} {'-'*9:>9s}")
        for r in rows_sorted[:20]:
            print(f"  {r['dataset']:<30s} {r['model']:<10s} "
                  f"{r['metric']:<8s} "
                  f"{r['best_base_score']:>10.4f} "
                  f"{r['hybrid_score']:>10.4f} "
                  f"{r['hybrid_budget']:>7d} "
                  f"{r['delta']:>+9.4f}")

        print(f"\n{'WORST HYBRID LOSSES (by delta)':=<80}")
        print(f"  {'Dataset':<30s} {'Model':<10s} {'Metric':<8s} "
              f"{'BestBase':>10s} {'Hybrid':>10s} {'Budget':>7s} {'Δ':>9s}")
        print(f"  {'-'*28:<30s} {'-'*8:<10s} {'-'*6:<8s} "
              f"{'-'*10:>10s} {'-'*10:>10s} {'-'*7:>7s} {'-'*9:>9s}")
        for r in rows_sorted[-20:]:
            print(f"  {r['dataset']:<30s} {r['model']:<10s} "
                  f"{r['metric']:<8s} "
                  f"{r['best_base_score']:>10.4f} "
                  f"{r['hybrid_score']:>10.4f} "
                  f"{r['hybrid_budget']:>7d} "
                  f"{r['delta']:>+9.4f}")

    return {
        'total': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'domain_stats': {k: dict(v) for k, v in domain_stats.items()},
        'model_stats': {k: dict(v) for k, v in model_stats.items()},
        'dataset_stats': {k: dict(v) for k, v in dataset_stats.items()},
        'rows': rows,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='General analysis: hybrid vs base rep performance.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('datasets', nargs='*',
                        help='Dataset names (default: all in DATASET_REPS)')
    parser.add_argument('--domain', type=str, default=None,
                        help='Analyze all datasets in a domain')
    parser.add_argument('--all', action='store_true',
                        help='Analyze ALL datasets')
    parser.add_argument('--speed', type=str, default='fast',
                        choices=['fast', 'full'],
                        help='Search speed tier to load (must match precompute)')
    parser.add_argument('--aug', type=str, default='uncapped',
                        choices=['none', 'uncapped', 'capped'],
                        help='Augmentation condition to load')
    parser.add_argument('--export', action='store_true',
                        help='Export mode: load from cache, save JSON')
    parser.add_argument('--analyze', type=str, default=None, metavar='JSON',
                        help='Analyze mode: read exported JSON, no cache needed')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON path (default: tests/analysis/general_analysis.json)')

    args = parser.parse_args()

    output_path = args.output or str(
        Path(__file__).resolve().parent / 'general_analysis.json')

    # ── Analyze-only mode ──
    if args.analyze:
        print(f"Loading exported results from {args.analyze}")
        with open(args.analyze) as f:
            data = json.load(f)
        results = data.get('datasets', data) if isinstance(data, dict) else data
        if isinstance(results, dict):
            results = list(results.values())
        summary = analyze(results)
        return

    # ── Export mode (or default: export + analyze) ──
    from experiments.complexity.complexity_helpers import DATASET_REPS

    # Resolve dataset list
    if args.all:
        datasets = sorted(DATASET_REPS.keys())
    elif args.domain:
        datasets = sorted(k for k in DATASET_REPS
                          if _get_domain(k) == args.domain)
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(DATASET_REPS.keys())

    print(f"Analyzing {len(datasets)} datasets (speed={args.speed}, "
          f"aug={args.aug})")
    print(f"Datasets: {datasets}\n")

    # Process each dataset
    results = []
    t_start = time.time()
    for i, ds in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {ds}...")
        r = export_dataset(ds, speed=args.speed, aug_condition=args.aug)
        if r is not None:
            results.append(r)

    elapsed = time.time() - t_start
    print(f"\nExport complete: {len(results)}/{len(datasets)} datasets "
          f"in {elapsed:.0f}s")

    # Save JSON
    export_data = {
        'metadata': {
            'n_datasets': len(results),
            'speed': args.speed,
            'aug_condition': args.aug,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'datasets': results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    print(f"Saved to {output_path}")

    # Analyze
    if not args.export:
        summary = analyze(results)

        # Save summary alongside
        summary_path = output_path.replace('.json', '_summary.json')
        # Convert numpy types for JSON
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        summary_safe = json.loads(
            json.dumps(summary, default=_convert))
        with open(summary_path, 'w') as f:
            json.dump(summary_safe, f, indent=2)
        print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
