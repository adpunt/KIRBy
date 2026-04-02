#!/usr/bin/env python3
"""F09: Cross-domain synthesis analysis.

Paper: Framework
Question: Which hybrid search settings are universal vs domain-specific?

Consumes JSON outputs from f00–f08 across all framework tasks.
Produces cross-domain comparison tables and recommendations for:
  - Allocation method ranking per domain
  - Importance method ranking (from f05 correlation + f01 downstream)
  - Budget/step sensitivity summary (plateau points from f03)
  - Model ranking comparison (from f01+f03)
  - Rep screening recommendation (from f00)
  - Pruning threshold recommendation (from f02)
  - Augmentation strategy comparison (from f06+f08)
  - Internal evaluator comparison (from f07)
  - Proposed new SPEED_TIERS based on all data

Usage:
    python f09_cross_domain_analysis.py --input tests/results/framework
    python f09_cross_domain_analysis.py --input tests/results/framework --output tests/results/framework/synthesis
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    FRAMEWORK_TASKS, convert_numpy,
    INTERNAL_EVALUATORS, DOWNSTREAM_MODELS, extract_top3_base_reps,
)


# =========================================================================
# RESULT LOADING
# =========================================================================

def load_all_results(input_dir: str) -> dict:
    """Load all framework experiment JSONs from input_dir.

    Returns nested dict: {experiment_id: {dataset: data}}.
    """
    input_path = Path(input_dir)
    results = {}

    for exp_dir in sorted(input_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_id = exp_dir.name
        results[exp_id] = {}
        for json_file in sorted(exp_dir.glob('*.json')):
            dataset = json_file.stem
            with open(json_file) as f:
                results[exp_id][dataset] = json.load(f)

    # Also check for flat JSONs (e.g. f05 saves per-dataset directly)
    for json_file in sorted(input_path.glob('*.json')):
        if json_file.stem not in results:
            with open(json_file) as f:
                results[json_file.stem] = json.load(f)

    return results


def get_domain(dataset: str) -> str:
    """Map dataset name to domain."""
    if dataset.startswith('tcga_'):
        return 'tcga'
    elif dataset.startswith('flab_') or dataset.startswith('abcov_') or \
         dataset.startswith('ddg_') or dataset.startswith('abagym_'):
        return 'antibody'
    elif dataset in ('buchwald_hartwig', 'suzuki', 'bh_doyle_plates'):
        return 'reaction'
    else:
        return 'molecular'


# =========================================================================
# ANALYSIS MODULES
# =========================================================================

def analyze_allocation_rankings(results: dict) -> dict:
    """From f01 results: which allocation method wins per domain?

    Returns {domain: {allocation: avg_rank, allocation: avg_score}}.
    """
    if 'f01' not in results:
        return {'status': 'f01 results not found'}

    domain_rankings = defaultdict(lambda: defaultdict(list))

    for dataset, data in results['f01'].items():
        domain = get_domain(dataset)
        if 'results' not in data:
            continue

        # Extract per-allocation best scores
        alloc_scores = defaultdict(list)
        for entry in data['results']:
            alloc = entry.get('allocation', entry.get('allocation_method'))
            score = entry.get('score_mean', entry.get('comparable_score'))
            if alloc and score is not None and not np.isnan(score):
                alloc_scores[alloc].append(score)

        # Rank allocations by mean score
        alloc_means = {a: np.mean(s) for a, s in alloc_scores.items()}
        ranked = sorted(alloc_means.items(), key=lambda x: x[1], reverse=True)

        for rank, (alloc, score) in enumerate(ranked, 1):
            domain_rankings[domain][alloc].append({
                'dataset': dataset,
                'rank': rank,
                'score': score,
            })

    # Summarize
    summary = {}
    for domain, allocs in domain_rankings.items():
        summary[domain] = {}
        for alloc, entries in allocs.items():
            summary[domain][alloc] = {
                'avg_rank': np.mean([e['rank'] for e in entries]),
                'avg_score': np.mean([e['score'] for e in entries]),
                'n_datasets': len(entries),
                'per_dataset': entries,
            }
    return summary


def analyze_step_size_sensitivity(results: dict) -> dict:
    """From f03 results: where do step sizes plateau?

    Returns {domain: {step_size: avg_score, plateau_point: N}}.
    """
    if 'f03' not in results:
        return {'status': 'f03 results not found'}

    domain_step_scores = defaultdict(lambda: defaultdict(list))

    for dataset, data in results['f03'].items():
        domain = get_domain(dataset)
        if 'results' not in data:
            continue

        for entry in data['results']:
            step = entry.get('step_size')
            score = entry.get('score_mean', entry.get('comparable_score'))
            if step is not None and score is not None and not np.isnan(score):
                domain_step_scores[domain][step].append(score)

    summary = {}
    for domain, steps in domain_step_scores.items():
        step_means = {s: np.mean(scores) for s, scores in steps.items()}
        sorted_steps = sorted(step_means.items())

        # Find plateau: first step where score is within 1% of best
        best_score = max(step_means.values())
        plateau = None
        for step, score in sorted_steps:
            if score >= 0.99 * best_score:
                plateau = step
                break

        summary[domain] = {
            'step_scores': {str(s): float(v) for s, v in sorted_steps},
            'best_step': max(step_means, key=step_means.get),
            'plateau_point': plateau,
        }
    return summary


def analyze_importance_methods(results: dict) -> dict:
    """From f05 results: which importance methods are redundant?

    Returns correlation matrix summary and timing comparison.
    """
    if 'f05' not in results:
        return {'status': 'f05 results not found'}

    summary = {}
    for dataset, data in results['f05'].items():
        domain = get_domain(dataset)
        entry = {
            'domain': domain,
        }

        if 'correlation_matrix' in data:
            entry['correlation_matrix'] = data['correlation_matrix']
        if 'timing' in data:
            entry['timing'] = data['timing']
        if 'redundant_pairs' in data:
            entry['redundant_pairs'] = data['redundant_pairs']
        if 'recommendations' in data:
            entry['recommendations'] = data['recommendations']

        summary[dataset] = entry

    return summary


def analyze_pruning_thresholds(results: dict) -> dict:
    """From f02 results: what pruning threshold balances speed vs regret?

    Returns {domain: {threshold: {speedup, regret, true_best_survives}}}.
    """
    if 'f02' not in results:
        return {'status': 'f02 results not found'}

    domain_thresholds = defaultdict(lambda: defaultdict(list))

    for dataset, data in results['f02'].items():
        domain = get_domain(dataset)
        if 'results' not in data:
            continue

        for entry in data['results']:
            thresh = entry.get('threshold')
            if thresh is None:
                continue
            domain_thresholds[domain][thresh].append({
                'dataset': dataset,
                'speedup_pct': entry.get('speedup_pct', 0),
                'regret': entry.get('regret', 0),
                'true_best_survives': entry.get('true_best_survives', 'UNKNOWN'),
            })

    summary = {}
    for domain, thresholds in domain_thresholds.items():
        summary[domain] = {}
        for thresh, entries in thresholds.items():
            summary[domain][str(thresh)] = {
                'avg_speedup': np.mean([e['speedup_pct'] for e in entries]),
                'avg_regret': np.mean([e['regret'] for e in entries]),
                'best_survives_rate': sum(
                    1 for e in entries if e['true_best_survives'] == 'YES'
                ) / len(entries),
                'n_datasets': len(entries),
            }

        # Recommend: highest speedup with regret < 1% and survival rate 100%
        best_thresh = None
        best_speedup = 0
        for thresh, stats in summary[domain].items():
            if stats['avg_regret'] < 0.01 and stats['best_survives_rate'] >= 1.0:
                if stats['avg_speedup'] > best_speedup:
                    best_thresh = thresh
                    best_speedup = stats['avg_speedup']
        summary[domain]['recommended_threshold'] = best_thresh

    return summary


def analyze_rep_screening(results: dict) -> dict:
    """From f00 results: which screening option is best?"""
    if 'f00' not in results:
        return {'status': 'f00 results not found'}

    domain_options = defaultdict(lambda: defaultdict(list))

    for dataset, data in results['f00'].items():
        domain = get_domain(dataset)
        if 'results' not in data:
            continue

        for entry in data['results']:
            option = entry.get('option', entry.get('screening_option'))
            score = entry.get('score', entry.get('final_score'))
            regret = entry.get('regret', 0)
            speedup = entry.get('speedup', entry.get('speedup_pct', 0))
            if option and score is not None:
                domain_options[domain][option].append({
                    'dataset': dataset,
                    'score': score,
                    'regret': regret,
                    'speedup': speedup,
                })

    summary = {}
    for domain, options in domain_options.items():
        summary[domain] = {}
        for opt, entries in options.items():
            summary[domain][opt] = {
                'avg_score': np.mean([e['score'] for e in entries]),
                'avg_regret': np.mean([e['regret'] for e in entries]),
                'avg_speedup': np.mean([e['speedup'] for e in entries]),
            }
    return summary


def analyze_internal_evaluator(results: dict) -> dict:
    """From f07 results: does the internal evaluator matter?"""
    if 'f07' not in results:
        return {'status': 'f07 results not found'}

    summary = {}
    for dataset, data in results['f07'].items():
        domain = get_domain(dataset)
        summary[dataset] = {
            'domain': domain,
            'results': data.get('results', data),
        }
    return summary


def analyze_two_phase(results: dict) -> dict:
    """From f04 results: does two-phase match full grid?"""
    if 'f04' not in results:
        return {'status': 'f04 results not found'}

    summary = {}
    for dataset, data in results['f04'].items():
        domain = get_domain(dataset)
        summary[dataset] = {
            'domain': domain,
            'verdict': data.get('verdict', 'UNKNOWN'),
            'regret': data.get('regret', data.get('results', {}).get('regret')),
            'eval_reduction_pct': data.get('eval_reduction_pct',
                                           data.get('results', {}).get('eval_reduction_pct')),
        }
    return summary


def propose_speed_tiers(analyses: dict) -> dict:
    """Based on all analyses, propose new SPEED_TIERS.

    This is the key deliverable — empirically-grounded search settings.
    """
    proposal = {
        'fast': {
            'description': 'Minimal search — use when time is critical',
            'notes': [],
        },
        'medium': {
            'description': 'Balanced search — default for most use cases',
            'notes': [],
        },
        'long': {
            'description': 'Exhaustive search — use for final results',
            'notes': [],
        },
    }

    # Incorporate allocation rankings
    alloc_analysis = analyses.get('allocation_rankings', {})
    if isinstance(alloc_analysis, dict) and 'status' not in alloc_analysis:
        # Find universally good allocations (top 3 across all domains)
        alloc_ranks = defaultdict(list)
        for domain, allocs in alloc_analysis.items():
            for alloc, stats in allocs.items():
                alloc_ranks[alloc].append(stats['avg_rank'])

        universal_top = sorted(alloc_ranks.items(),
                               key=lambda x: np.mean(x[1]))[:3]
        proposal['fast']['allocations'] = [universal_top[0][0]]
        proposal['medium']['allocations'] = [a for a, _ in universal_top[:2]]
        proposal['long']['allocations'] = [a for a, _ in universal_top]
        proposal['fast']['notes'].append(
            f"Top allocation: {universal_top[0][0]} "
            f"(avg rank {np.mean(universal_top[0][1]):.1f})"
        )

    # Incorporate step size plateau
    step_analysis = analyses.get('step_size_sensitivity', {})
    if isinstance(step_analysis, dict) and 'status' not in step_analysis:
        plateaus = [v.get('plateau_point') for v in step_analysis.values()
                    if v.get('plateau_point') is not None]
        if plateaus:
            proposal['fast']['step_sizes'] = [max(plateaus)]
            proposal['medium']['step_sizes'] = [max(plateaus), max(plateaus) // 2]
            proposal['long']['step_sizes'] = [1, 2, 5, 10]
            proposal['fast']['notes'].append(
                f"Step size plateau at {max(plateaus)} across domains"
            )

    # Incorporate pruning
    pruning_analysis = analyses.get('pruning_thresholds', {})
    if isinstance(pruning_analysis, dict) and 'status' not in pruning_analysis:
        thresholds = [v.get('recommended_threshold') for v in pruning_analysis.values()
                      if v.get('recommended_threshold') is not None]
        if thresholds:
            # Use most aggressive safe threshold for fast
            proposal['fast']['pruning_threshold'] = max(float(t) for t in thresholds)
            proposal['medium']['pruning_threshold'] = min(float(t) for t in thresholds)
            proposal['long']['pruning_threshold'] = None  # no pruning

    return proposal


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='F09: Cross-domain synthesis analysis'
    )
    parser.add_argument('--input', type=str, default='tests/results/framework',
                        help='Directory containing f00-f08 result JSONs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: {input}/synthesis)')
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.input, 'synthesis')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  F09: Cross-Domain Synthesis Analysis")
    print("=" * 70)

    # Load all results
    print("\nLoading results from:", args.input)
    results = load_all_results(args.input)
    print(f"  Found {len(results)} experiment directories:")
    for exp_id, datasets in results.items():
        if isinstance(datasets, dict) and not any(k in datasets for k in ('results', 'metadata')):
            print(f"    {exp_id}: {list(datasets.keys())}")
        else:
            print(f"    {exp_id}: (single result)")

    # Run all analyses
    print("\n" + "=" * 70)
    print("  Running analyses...")
    print("=" * 70)

    analyses = {}

    print("\n--- Allocation rankings (f01) ---")
    analyses['allocation_rankings'] = analyze_allocation_rankings(results)
    _print_status(analyses['allocation_rankings'])

    print("\n--- Step size sensitivity (f03) ---")
    analyses['step_size_sensitivity'] = analyze_step_size_sensitivity(results)
    _print_status(analyses['step_size_sensitivity'])

    print("\n--- Importance methods (f05) ---")
    analyses['importance_methods'] = analyze_importance_methods(results)
    _print_status(analyses['importance_methods'])

    print("\n--- Pruning thresholds (f02) ---")
    analyses['pruning_thresholds'] = analyze_pruning_thresholds(results)
    _print_status(analyses['pruning_thresholds'])

    print("\n--- Rep screening (f00) ---")
    analyses['rep_screening'] = analyze_rep_screening(results)
    _print_status(analyses['rep_screening'])

    print("\n--- Internal evaluator (f07) ---")
    analyses['internal_evaluator'] = analyze_internal_evaluator(results)
    _print_status(analyses['internal_evaluator'])

    print("\n--- Two-phase validation (f04) ---")
    analyses['two_phase'] = analyze_two_phase(results)
    _print_status(analyses['two_phase'])

    print("\n" + "=" * 70)
    print("  Proposing SPEED_TIERS...")
    print("=" * 70)
    analyses['proposed_speed_tiers'] = propose_speed_tiers(analyses)

    for tier, config in analyses['proposed_speed_tiers'].items():
        print(f"\n  {tier}:")
        for key, val in config.items():
            if key != 'notes':
                print(f"    {key}: {val}")
        for note in config.get('notes', []):
            print(f"    NOTE: {note}")

    # Save synthesis
    output_path = os.path.join(output_dir, 'synthesis.json')
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(analyses), f, indent=2)
    print(f"\n  [saved {output_path}]")

    # Also save a human-readable summary
    summary_path = os.path.join(output_dir, 'SYNTHESIS_SUMMARY.md')
    _write_summary_md(analyses, summary_path)
    print(f"  [saved {summary_path}]")


def _print_status(analysis):
    """Print a brief status for an analysis result."""
    if isinstance(analysis, dict):
        if 'status' in analysis:
            print(f"  {analysis['status']}")
        else:
            print(f"  {len(analysis)} entries")


def _write_summary_md(analyses: dict, path: str):
    """Write a human-readable markdown summary."""
    lines = ['# Framework Experiment Synthesis', '',
             'Auto-generated by f09_cross_domain_analysis.py', '']

    # Allocation rankings
    lines.append('## Allocation Method Rankings')
    alloc = analyses.get('allocation_rankings', {})
    if isinstance(alloc, dict) and 'status' not in alloc:
        for domain, allocs in sorted(alloc.items()):
            lines.append(f'\n### {domain}')
            lines.append('| Allocation | Avg Rank | Avg Score | N Datasets |')
            lines.append('|-----------|----------|-----------|------------|')
            for a, stats in sorted(allocs.items(), key=lambda x: x[1]['avg_rank']):
                lines.append(
                    f"| {a} | {stats['avg_rank']:.1f} | "
                    f"{stats['avg_score']:.4f} | {stats['n_datasets']} |"
                )
    else:
        lines.append('Not available (f01 results needed)')

    # Step sizes
    lines.append('\n## Step Size Sensitivity')
    steps = analyses.get('step_size_sensitivity', {})
    if isinstance(steps, dict) and 'status' not in steps:
        for domain, data in sorted(steps.items()):
            lines.append(f"\n### {domain}")
            lines.append(f"Best step: {data.get('best_step')}, "
                         f"Plateau: {data.get('plateau_point')}")
    else:
        lines.append('Not available (f03 results needed)')

    # Pruning
    lines.append('\n## Pruning Thresholds')
    pruning = analyses.get('pruning_thresholds', {})
    if isinstance(pruning, dict) and 'status' not in pruning:
        for domain, data in sorted(pruning.items()):
            rec = data.get('recommended_threshold', 'none')
            lines.append(f"- **{domain}**: recommended threshold = {rec}")
    else:
        lines.append('Not available (f02 results needed)')

    # Speed tiers
    lines.append('\n## Proposed SPEED_TIERS')
    tiers = analyses.get('proposed_speed_tiers', {})
    for tier, config in tiers.items():
        lines.append(f"\n### {tier}")
        lines.append(f"_{config.get('description', '')}_")
        for key, val in config.items():
            if key not in ('description', 'notes'):
                lines.append(f"- {key}: `{val}`")
        for note in config.get('notes', []):
            lines.append(f"- {note}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
