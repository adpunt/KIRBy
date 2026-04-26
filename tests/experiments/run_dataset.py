#!/usr/bin/env python3
"""Unified runner for per-dataset experiments across all categories.

Replaces the per-dataset .py files (one per dataset per category) with a
single dispatch point.  All parallelism is preserved — SLURM still submits
one job per (category, dataset) pair.

Usage:
    python tests/experiments/run_dataset.py CATEGORY DATASET [OPTIONS]

Examples:
    python tests/experiments/run_dataset.py complexity herg
    python tests/experiments/run_dataset.py noise herg --experiments n01 n02
    python tests/experiments/run_dataset.py similarity logd --output /custom/path
    python tests/experiments/run_dataset.py federated herg
"""
import sys
import os
import argparse
import importlib

# Path setup — same as the per-dataset files this replaces
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))        # tests/
sys.path.insert(0, os.path.dirname(__file__))                            # tests/experiments/

# ── Category registry ────────────────────────────────────────────────────
# Adding a new category: add one entry here + create {category}_helpers.py
# with run_all() and SINGLE_DATASET_EXPERIMENTS.

CATEGORIES = {
    'complexity': {
        'module': 'experiments.complexity.complexity_helpers',
        'default_output': 'tests/results/complexity',
    },
    'similarity': {
        'module': 'experiments.similarity.similarity_helpers',
        'default_output': 'tests/results/similarity',
    },
    'noise': {
        'module': 'experiments.noise.noise_helpers',
        'default_output': 'tests/results/noise',
    },
    'federated': {
        'module': 'experiments.federated.federated_helpers',
        'default_output': 'tests/results/federated',
    },
}


def main():
    parser = argparse.ArgumentParser(
        description='Run per-dataset experiments for any category',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            'Categories:',
            *[f'  {k:12s}  ({v["module"].split(".")[-1]})'
              for k, v in sorted(CATEGORIES.items())],
        ]),
    )
    parser.add_argument(
        'category', choices=sorted(CATEGORIES.keys()),
        help='Experiment category')
    parser.add_argument(
        'dataset', type=str,
        help='Dataset name (e.g. herg, logd, flab_thermo_tresanco)')
    parser.add_argument(
        '--experiments', nargs='+', default=None,
        help='Experiment IDs to run (default: all for category)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory (default: per-category default)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default: 'default_pruning' = current "
             "single-stage 0.15 strategy prune). 'aggressive_pruning' adds "
             "td=0.5 at b=50 mid-flight prune to gp/svm/mlp on datasets with "
             "n_samples >= 1000. See PRUNING_MODES in tests/experiments/_shared.py.")
    args = parser.parse_args()

    cat_info = CATEGORIES[args.category]
    output_dir = args.output or cat_info['default_output']

    # Import the category helpers
    helpers = importlib.import_module(cat_info['module'])

    # Validate dataset against DATASET_REPS (early fail with clear error)
    from experiments.complexity.complexity_helpers import DATASET_REPS
    if args.dataset not in DATASET_REPS:
        print(f"ERROR: Unknown dataset '{args.dataset}'.")
        print(f"Available datasets ({len(DATASET_REPS)}): "
              f"{', '.join(sorted(DATASET_REPS.keys()))}")
        sys.exit(1)

    # Validate experiment IDs if specified
    if args.experiments:
        valid = helpers.SINGLE_DATASET_EXPERIMENTS
        bad = [e for e in args.experiments if e not in valid]
        if bad:
            print(f"ERROR: Unknown experiments for {args.category}: {bad}")
            print(f"Valid: {valid}")
            sys.exit(1)

    # Threading pruning_mode is opt-in per category. complexity has the param;
    # the others currently inherit the default. Unknown kwarg raises a clear
    # TypeError if a category helper hasn't adopted it yet.
    import inspect
    run_all_params = inspect.signature(helpers.run_all).parameters
    extra_kwargs = {}
    if 'pruning_mode' in run_all_params:
        extra_kwargs['pruning_mode'] = args.pruning_mode
    elif args.pruning_mode != 'default_pruning':
        print(f"WARNING: --pruning-mode={args.pruning_mode} ignored: "
              f"{args.category} run_all does not yet accept pruning_mode.")

    helpers.run_all(
        dataset_name=args.dataset,
        output_dir=output_dir,
        experiments=args.experiments,
        **extra_kwargs,
    )


if __name__ == '__main__':
    main()
