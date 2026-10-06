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
import time

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
# Two of the six decided kinds (noise_helpers.DECIDED_REGRESSION_CONDITIONS,
# Adelaide 2026-10-05) sit outside the NoiseInject registry and are injected by
# noise_helpers.inject_condition. Named here so a job can be given them.
NOISE_CONDITIONS += [c for c in ('shift_only_p10', 'sparse_dprime_up_p10')
                     if c not in NOISE_CONDITIONS]


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
        choices=['per_fold'], default='per_fold',
        help="Where c11 chooses the hybrid's recipe. Only 'per_fold' since "
             "2026-10-05: the recipe is searched inside each outer training "
             "fold. 'whole_data' searched every row and was removed with every "
             "other whole-dataset search (handoff 2.4 step 4).")
    parser.add_argument(
        '--pruning-mode', type=str, default=None,
        help="No effect since 2026-10-05: pruning is off in every per-fold "
             "search, and no suite runs a whole-dataset search at start-up. "
             "Accepted so existing launch lines still parse; a value given is "
             "printed as ignored.")
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
    # A dataset that borrows another's representation list (noise_helpers
    # REP_LIST_FROM: herg_ic50 takes herg_ki's) is a valid name for that
    # category even though it is in no DATASET_REPS loop.
    _borrowed = getattr(helpers, 'REP_LIST_FROM', {})
    if args.dataset not in DATASET_REPS and args.dataset not in _borrowed:
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

    # ── The ledger (2026-10-06) ───────────────────────────────────────────
    # When a run is configured (KIRBY_RUN, or tests/results/CURRENT_RUN
    # written by orchestrate/ledger.py init) and this process runs ONE
    # experiment, it reports to that run's ledger: output goes under the run
    # root when --output was not given; an experiment the ledger already has
    # as ok, or running in a live job, is not run again; start, end and
    # seconds are recorded. That is how a held job submitted before the
    # ledger existed, or an srun typed by hand, lands in the same document
    # as the worker jobs. KIRBY_LEDGER_MANAGED=1 (set by worker.py around
    # its subprocess) means the worker records, so this hook stays quiet.
    _ledger = _ledger_hook(args, output_dir)
    if _ledger is not None:
        output_dir = _ledger['output_dir']

    # Threading pruning_mode is opt-in per category. complexity has the param;
    # the others currently inherit the default. Unknown kwarg raises a clear
    # TypeError if a category helper hasn't adopted it yet.
    import inspect
    run_all_params = inspect.signature(helpers.run_all).parameters
    extra_kwargs = {}
    if args.pruning_mode and 'pruning_mode' in run_all_params:
        extra_kwargs['pruning_mode'] = args.pruning_mode
    elif args.pruning_mode:
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
    _t0 = time.time()
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
    if _ledger is not None:
        _ledger_finish(_ledger, n_failed, time.time() - _t0, results, run_all_error)
    if n_failed:
        sys.exit(1)


def _ledger_hook(args, output_dir):
    """Register this one-experiment process with the run's ledger, or None.

    Returns {'L', 'row', 'stage', 'unit', 'job', 'output_dir'} when the
    process should run; exits 0 when the ledger says the unit is already ok
    or running elsewhere. Any ledger failure is printed and ignored: the
    experiment runs as it would have without a ledger."""
    if os.environ.get('KIRBY_LEDGER_MANAGED') == '1':
        return None
    if not args.experiments or len(args.experiments) != 1:
        return None
    try:
        from experiments.orchestrate import ledger as L_mod, plan
        root = L_mod.resolve_run()
        if root is None or not (root / 'ledger.json').exists():
            return None
        exp = args.experiments[0]
        stage = args.category
        if stage == 'noise' and os.environ.get('KIRBY_NOISE_PROMOTE'):
            _prom = {x.strip() for x in os.environ['KIRBY_NOISE_PROMOTE'].split(',')}
            if 'all' in _prom or exp in _prom:
                stage = 'noise_full'
        unit = f"{exp}__{args.noise_condition}" if (stage in ('noise', 'noise_full')
                                                   and args.noise_condition) else exp
        out = output_dir if args.output else str(root / plan.CATEGORY_DIR[stage])
        L = L_mod.Ledger(root)
        job = L_mod.this_job()
        import socket
        host = socket.gethostname()
        log = os.environ.get('KIRBY_LEDGER_LOG') or None
        with L.locked():
            if args.dataset not in L.state['rows']:
                print(f"ledger: {args.dataset} is not a row of run {root.name}; running unrecorded")
                return None
            row = L.row(args.dataset)
            if L.stage_of(row, stage) is None:
                print(f"ledger: {args.dataset} has no stage {stage} in run {root.name}; running unrecorded")
                return None
            started = L.start_unit(args.dataset, stage, unit, job, host, log=log)
        if not started:
            print(f"ledger: {args.dataset} {stage} {unit} is already ok or running in a live job; nothing to do")
            sys.exit(0)
        print(f"ledger: recording {args.dataset} {stage} {unit} as {job}; output {out}")
        return {'L': L, 'row': args.dataset, 'stage': stage, 'unit': unit,
                'job': job, 'output_dir': out}
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ledger: hook failed ({type(exc).__name__}: {exc}); running unrecorded")
        return None


def _ledger_finish(h, n_failed, seconds, results, run_all_error):
    try:
        status = 'ok' if not n_failed else 'failed'
        err = None
        if n_failed:
            if run_all_error:
                err = run_all_error.get('error')
            else:
                bad = failed_experiments(results)
                err = '; '.join(f"{k}: {v.get('error')}" for k, v in bad.items())[:600]
        with h['L'].locked():
            h['L'].finish_unit(h['row'], h['stage'], h['unit'], status,
                               seconds=seconds, rc=1 if n_failed else 0, error=err)
    except Exception as exc:
        print(f"ledger: finish failed ({type(exc).__name__}: {exc})")


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
