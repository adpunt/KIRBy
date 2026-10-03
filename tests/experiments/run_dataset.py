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
        # EXTRA_EXPERIMENTS: run only on some datasets, added to the default
        # list by run_all itself (complexity's c12 on tcga_*). They must be
        # nameable here, or a one-experiment-per-process job cannot run them.
        valid = (list(helpers.SINGLE_DATASET_EXPERIMENTS)
                 + list(getattr(helpers, 'EXTRA_EXPERIMENTS', ())))
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
    # whose run_all has adopted the param. All four do as of 2026-09-25
    # (federated was the last); in noise it also reaches the per-condition
    # files a sweep writes (noise_helpers.sweep_conditions).
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

    # ⛔ FAILURES MUST BE VISIBLE (2026-09-25). Every category's run_all
    # catches a failed experiment, records {'error': ...} for it and carries on,
    # so one broken experiment does not end the others. But this runner then
    # exited 0 whatever happened, so a SLURM job with every experiment failed
    # read as COMPLETED and the only record was a traceback in the log. Now:
    # all experiments still run; afterwards each failed one is written to
    # <output>/_errors/<dataset>[__<condition>][__<experiment>].json with its
    # traceback text,
    # and the process exits 1. A run_all that raises (for example federated's
    # sys.exit(1) when FLuID is missing) is recorded the same way.
    results, run_all_error = None, None
    try:
        results = helpers.run_all(
            dataset_name=args.dataset,
            output_dir=output_dir,
            experiments=args.experiments,
            **extra_kwargs,
        )
    except KeyboardInterrupt:
        raise
    except SystemExit as exc:
        if exc.code not in (0, None):
            import traceback
            run_all_error = {'error': f'run_all called sys.exit({exc.code})',
                             'error_kind': 'SystemExit',
                             'traceback': traceback.format_exc()}
    except Exception as exc:
        import traceback
        run_all_error = {'error': str(exc) or type(exc).__name__,
                         'error_kind': type(exc).__name__,
                         'traceback': traceback.format_exc()}

    cond = args.noise_condition if 'noise_condition' in extra_kwargs else None
    # One experiment per process (submit_dataset.sh, stage1_prelim_noise.py):
    # the report is named after that experiment too. Named only after the
    # dataset and condition, as it was until 2026-10-03, the NEXT experiment's
    # process found no failure of its own and deleted the report, so a job's
    # _errors/ kept only failures from its last experiment.
    only = args.experiments[0] if args.experiments and len(args.experiments) == 1 else None
    n_failed = write_error_report(output_dir, args.category, args.dataset,
                                  cond, results, run_all_error,
                                  experiment=only)
    if n_failed:
        sys.exit(1)


def failed_experiments(results):
    """{exp_id: record} for every experiment run_all recorded an 'error' for.

    Keys starting with '_' are run_all's own records (_precompute_errors,
    _rep_pool_audit), not experiments, and are not read."""
    out = {}
    for exp_id, rec in (results or {}).items():
        if str(exp_id).startswith('_'):
            continue
        if isinstance(rec, dict) and rec.get('error') is not None:
            out[str(exp_id)] = rec
    return out


def write_error_report(output_dir, category, dataset, condition, results,
                       run_all_error=None, experiment=None):
    """Write <output_dir>/_errors/<dataset>[__<condition>][__<experiment>].json
    listing each failed experiment and its traceback; return how many failures
    there are. `experiment` is set when the process ran exactly one experiment.

    With no failure, a report left by an earlier run of the same dataset,
    condition and experiment is removed, so the directory only ever lists what
    is broken now.
    A run_all that raised counts as one failure under the key '_run_all'.

    A record with no 'traceback' key is one of two things, and the report
    says so rather than guessing: an experiment that RETURNED an error record
    instead of raising (n02/n09/n12/n13 return {'error': 'no_blueprint'} when
    the precompute holds no recipe, s11 {'error': 'dataset_too_small'}), or an
    exception caught by complexity_helpers.run_all, which stores no traceback
    (that file is not edited here). Both count as failures: neither wrote a
    measurement.
    """
    import json
    import time
    failed = failed_experiments(results)
    if run_all_error is not None:
        failed['_run_all'] = run_all_error
    name = f"{dataset}__{condition}" if condition else f"{dataset}"
    name = f"{name}__{experiment}.json" if experiment else f"{name}.json"
    path = os.path.join(output_dir, '_errors', name)
    if not failed:
        if os.path.exists(path):
            os.remove(path)
            print(f"  no failures; removed the earlier error report {path}")
        return 0
    report = {
        'category': category, 'dataset': dataset, 'noise_condition': condition,
        'written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_failed': len(failed),
        'failed': {
            exp_id: {
                'error': str(rec.get('error')),
                'error_kind': rec.get('error_kind'),
                'traceback': rec.get('traceback') or (
                    'none recorded: either the experiment returned this '
                    'error record itself without raising, or '
                    'complexity_helpers.run_all caught an exception and kept '
                    'no traceback; see the job log'),
            } for exp_id, rec in failed.items()},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(report, fh, indent=1, default=str)
    os.replace(tmp, path)
    print(f"\n{'=' * 70}")
    print(f"FAILED: {len(failed)} experiment(s) on {dataset}"
          f"{' under ' + condition if condition else ''}: "
          f"{', '.join(sorted(failed))}")
    print(f"  tracebacks written to {path}")
    print(f"{'=' * 70}")
    return len(failed)


if __name__ == '__main__':
    main()
