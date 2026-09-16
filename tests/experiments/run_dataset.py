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


from kirby.similarity.distance_config import (
    DISTANCE_METRICS, DEFAULT_DISTANCE_METRIC, set_distance_metric,
)
from kirby.noise_spec import REGRESSION_STRATEGIES, CLASSIFICATION_STRATEGIES

NOISE_CONDITIONS = list(REGRESSION_STRATEGIES) + list(CLASSIFICATION_STRATEGIES)


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
        '--force-rerun', action='store_true',
        help='Recompute experiments even if their output JSON already exists. '
             'Default is RESUME: skip experiments already done on disk, so a '
             'job that hit the wall mid-sequence advances on resubmit instead '
             'of redoing c01.. from scratch. A resume only reuses a file whose '
             'stored input_fingerprint — row count, label hash, group hash, '
             'split count, representation names and widths — matches the data '
             'just loaded, so a fold-grouping, label or representation change '
             'recomputes on its own. A file written before that stamp existed '
             'has no fingerprint and is always recomputed. Use this flag for a '
             'clean re-run after a CODE change, which the fingerprint cannot '
             'see.')
    parser.add_argument(
        '--c11-recipe-search', dest='c11_recipe_search',
        choices=['per_fold', 'whole_data'], default='per_fold',
        help="Where c11 chooses the hybrid's recipe. 'per_fold' (default) "
             "searches inside each outer training fold, which is what the "
             "ensemble arm already does, so the ensemble-versus-hybrid delta "
             "is a comparison; it costs one pool of searches per outer fold "
             "on this dataset. 'whole_data' takes one recipe from a search "
             "over every row: free, but c11 then renames the head-to-head "
             "fields to say the hybrid saw rows the ensemble did not, and "
             "does not write auto_minus_hybrid. Use 'per_fold' on any dataset "
             "whose ensemble-versus-hybrid number gets quoted.")
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default: 'default_pruning' = current "
             "single-stage 0.15 strategy prune). 'aggressive_pruning' = "
             "F16-validated multi-stage pruning (checkpoints=[25,50,100], "
             "td=0.5) applied globally to all models. See PRUNING_MODES in "
             "tests/experiments/_shared.py.")
    parser.add_argument(
        '--distance-metric', type=str, default=DEFAULT_DISTANCE_METRIC,
        choices=list(DISTANCE_METRICS),
        help="Pairwise distance for every neighbour, cliff, "
             "applicability-domain, noise-score and retrieval quantity these "
             "experiments compute. Default manhattan. Set process-wide before "
             "any experiment runs, so a category helper needs no argument of "
             "its own. k-means, the RBF kernels and the ECFP4-Tanimoto cliff "
             "reference are fixed by what they are and ignore this. "
             "KIRBY_DISTANCE_METRIC in the environment does the same thing, "
             "which is how a SLURM script sets it without a new argument.")
    parser.add_argument(
        '--noise-condition', type=str, default=None,
        choices=NOISE_CONDITIONS,
        help="Which noise condition the noise experiments inject, e.g. "
             "censoring or student_t_nu3. Default None leaves every "
             "experiment on its own default, which is gaussian for n09/n11/"
             "n12 and an outlier condition for n07. The condition is written "
             "into the output filename, so two conditions on one dataset land "
             "on two files. noise only; any other category warns and ignores "
             "it.")
    args = parser.parse_args()

    # Before anything imports data or builds a neighbour index.
    set_distance_metric(args.distance_metric)

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
    # Resume is the default; --force-rerun disables it. Only pass to helpers
    # whose run_all has adopted the param (complexity today; others inherit
    # their own behaviour until they add it).
    if 'force_rerun' in run_all_params:
        extra_kwargs['force_rerun'] = args.force_rerun
    elif args.force_rerun:
        print(f"WARNING: --force-rerun ignored: "
              f"{args.category} run_all does not yet accept force_rerun.")
    if 'c11_recipe_search' in run_all_params:
        extra_kwargs['c11_recipe_search'] = args.c11_recipe_search
    elif args.c11_recipe_search != 'per_fold':
        print(f"WARNING: --c11-recipe-search ignored: "
              f"{args.category} run_all does not yet accept it.")
    if 'noise_condition' in run_all_params:
        extra_kwargs['noise_condition'] = args.noise_condition
    elif args.noise_condition is not None:
        print(f"WARNING: --noise-condition={args.noise_condition} ignored: "
              f"{args.category} run_all does not accept noise_condition.")

    helpers.run_all(
        dataset_name=args.dataset,
        output_dir=output_dir,
        experiments=args.experiments,
        **extra_kwargs,
    )


if __name__ == '__main__':
    main()
