#!/usr/bin/env python3
"""STAGE 1 (and, through stage4_noise.sh, stage 4): every noise experiment on a
list of datasets, one SLURM array task per (dataset, experiment, condition).

Written 2026-09-25 for the owner's four-stage plan (runbook/README.md).

WHAT ONE TASK IS. One line of a task file: <dataset> <experiment> <condition>.
runbook/noise_task.sbatch reads line $SLURM_ARRAY_TASK_ID and runs
    submit_dataset.sh noise <dataset> <experiment>
with NOISE_CONDITION set, so the task is one python process for one
experiment under one condition. A crash in it touches nothing else.

WHICH TASKS. Every experiment in noise_helpers.SINGLE_DATASET_EXPERIMENTS
(or the subset --experiments names):
  - one that takes a condition (HONOURS_NOISE_CONDITION | SWEEPS_ALL_CONDITIONS)
    gets one task per condition of noise_helpers.conditions_for_task for the
    dataset's task type: the flip models for classification, the fourteen
    shapes in SWEEP_ORDER for regression. The one exception is n21, which gets
    only the conditions in DETECTION_IS_DEFINED: it raises under any other
    (n21_neighbour_distance.py:671-677).
  - one that names its own condition gets one task with no condition.
  - PHASE B, submitted to start after every phase-A array has ended
    (afterany), holds two kinds of no-condition task:
      n06, because it reads the rankings n11 and n13 write in the same output
      directory (noise_helpers.py, n06's entry: "n06 runs LAST").
      n21, n23, n32, n40 and n41, because their own condition lists hold names
      --noise-condition cannot pass (run_dataset.py:125 takes the registry
      only): shift_only_* and sparse_dprime_* (DETECTION_IS_DEFINED,
      noise_helpers.py:573-578, swept by n21:651, n23:553, n32:509, n41:458),
      and dense_oofmse (n40:158, swept at n40:496). With no condition each
      sweeps its whole list through sweep_conditions, which skips a condition
      whose file is already on disk (noise_helpers.sweep_conditions docstring),
      so phase B computes only those extra conditions. Nothing is cut.
    n13 and n34 get no phase-B task: with no condition they write one file for
    every condition at once rather than one per condition (n34:351-357), so it
    would recompute what phase A wrote.

ARRAYS. One array per (phase, memory class, chunk). Memory: 128G for nt_*,
gb_*, tcga_*, kiba, bindingdb_kd and qm9, 64G otherwise -- the same split as
queue_noise_conditions.py:80. Every task of one array gets the same
resources, so the two classes cannot share one. Chunks of --array-chunk tasks
(default 1000): SLURM's default MaxArraySize is 1001 and ARC's own value is
not known here -- `scontrol show config | grep -i MaxArraySize` gives it.

WHERE. Results under --output (default tests/results/noise_prelim_sweep,
never the real tests/results/noise), task files and the environment file under
<output>/_tasks/, one cache report per task under <output>/_cache_reports/,
one summary row per experiment under <output>/_job_summaries/, one log per
experiment under --log-dir, and SLURM's own slurm-<array>_<task>.out in the
directory you submit from.

CLUSTER AND ACCOUNT. Default cluster htc; the account is the best fair share
on it, from where_to_submit.sh --best-cluster-account htc. --clusters best
picks the cluster too. An explicit --account wins; --account none passes none.

--dry-run prints the counts and the sbatch lines and calls no sbatch.

Needs the repo environment to list experiments and task types: run it after
`. setup.sh` (KIRBY_SKIP_INSTALLS=1), on the login node or in an srun.

Usage:
    python tests/slurm_scripts/runbook/stage1_prelim_noise.py --datasets "herg_ki herg" --dry-run
    python tests/slurm_scripts/runbook/stage1_prelim_noise.py --datasets "herg_ki herg" --max-concurrent 40
"""
import argparse
import datetime as _dt
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SLURM_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SLURM_DIR))
sys.path[:0] = [os.path.join(REPO, 'src'), os.path.join(REPO, 'tests'),
                os.path.join(REPO, 'tests', 'experiments')]

EMAIL = 'adelaide.punt@stcatz.ox.ac.uk'
# Same split as queue_noise_conditions.py:80.
BIG_MEM = ('nt_', 'gb_', 'tcga_', 'kiba', 'bindingdb_kd', 'qm9')
# Same exclusion as submit_everything.sh and precompute_status.py:
# chembl_qsar has no load_data path; davis is parked.
EXCLUDED = {'chembl_qsar', 'davis'}
# No-condition sweeps whose own list holds conditions --noise-condition cannot
# pass. See the module docstring for the file:line of each list.
REMAINDER_SWEEP = ('n21', 'n23', 'n32', 'n40', 'n41')
RUNS_LAST = ('n06',)


def best_cluster_account(cluster):
    cmd = ['bash', os.path.join(SLURM_DIR, 'where_to_submit.sh'),
           '--best-cluster-account'] + ([cluster] if cluster else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        return None, str(e)
    parts = r.stdout.split()
    if r.returncode != 0 or len(parts) != 2:
        return None, (r.stderr.strip() or 'no output')
    return parts, ''


def build_tasks(datasets, experiments):
    """[(phase, dataset, experiment, condition or '-')] in submission order:
    condition-major inside phase A, so every dataset's first condition comes
    before any dataset's second, as queue_noise_conditions.py orders them."""
    import experiments.noise.noise_helpers as NH
    from experiments._shared import registry_lookup

    takes = NH.HONOURS_NOISE_CONDITION | NH.SWEEPS_ALL_CONDITIONS
    exps = [e for e in NH.SINGLE_DATASET_EXPERIMENTS
            if not experiments or e in experiments]
    unknown = sorted(set(experiments or ()) - set(NH.SINGLE_DATASET_EXPERIMENTS))
    if unknown:
        sys.exit(f'Not in noise_helpers.SINGLE_DATASET_EXPERIMENTS: {unknown}')

    per_ds_conds = {}
    for ds in datasets:
        tt = getattr(registry_lookup(ds), 'task_type', None)
        per_ds_conds[ds] = NH.conditions_for_task({'task_type': tt})

    phase_a_cond, phase_a_plain, phase_b = [], [], []
    for ds in datasets:
        for e in exps:
            if e in RUNS_LAST:
                phase_b.append(('B', ds, e, '-'))
                continue
            if e not in takes:
                phase_a_plain.append(('A', ds, e, '-'))
                continue
            conds = per_ds_conds[ds]
            # n21, n24 and n25 run only under the conditions in
            # CONDITION_LIMITED_TO; any other condition is a skip marker, so it
            # gets no task (was n21 alone until 2026-09-27).
            _lim = getattr(NH, 'CONDITION_LIMITED_TO', {}).get(e)
            if _lim:
                conds = [c for c in conds if c in _lim]
            for i, c in enumerate(conds):
                phase_a_cond.append((i, ('A', ds, e, c)))
            if e in REMAINDER_SWEEP:
                phase_b.append(('B', ds, e, '-'))
    phase_a_cond.sort(key=lambda t: t[0])   # stable: condition-major
    return [t for _, t in phase_a_cond] + phase_a_plain + phase_b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--datasets', help='space-separated dataset names')
    g.add_argument('--all-datasets', action='store_true',
                   help='every dataset in DATASET_REPS except chembl_qsar and davis')
    ap.add_argument('--experiments', default='',
                    help='space-separated subset; default every registered one')
    ap.add_argument('--clusters', default='htc',
                    help="arc, htc, or 'best' for the best fair share of the two")
    ap.add_argument('--account', default='',
                    help="explicit account wins; 'none' passes no --account")
    ap.add_argument('--partition', default='medium')
    ap.add_argument('--wall', default='47:59:00')
    ap.add_argument('--cpus', type=int, default=8)
    ap.add_argument('--mem-small', default='64G')
    ap.add_argument('--mem-big', default='128G')
    ap.add_argument('--max-concurrent', type=int, default=0,
                    help='%%N on each array: at most N of its tasks run at once')
    ap.add_argument('--array-chunk', type=int, default=1000)
    ap.add_argument('--output', default='tests/results/noise_prelim_sweep')
    ap.add_argument('--log-dir', default='logs/noise_prelim')
    ap.add_argument('--pruning-mode', default='calibrated')
    ap.add_argument('--controls', action='store_true',
                    help='KIRBY_NOISE_CONTROLS=1: run the random and shuffled twins')
    ap.add_argument('--similarity-settings', default='',
                    help='KIRBY_SIMILARITY_SETTINGS for every task; unset = '
                         "the experiments' default, 'all'")
    ap.add_argument('--exp-timeout', default='',
                    help='EXP_TIMEOUT for every task (coreutils timeout syntax)')
    ap.add_argument('--job-prefix', default='ns1')
    ap.add_argument('--begin', default='')
    ap.add_argument('--hold', action='store_true')
    ap.add_argument('--after', default='',
                    help='job ids (colon- or comma-separated) every array waits '
                         'for with afterany; used by submit_all_stages.sh')
    ap.add_argument('--wait-for-precompute', action='store_true',
                    help="afterany on every queued or running pc_<dataset>_* job "
                         "of these datasets on the target cluster")
    ap.add_argument('--print-datasets', action='store_true',
                    help='print the dataset list and exit (with --all-datasets)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.all_datasets:
        from experiments.complexity.complexity_helpers import DATASET_REPS
        datasets = sorted(set(DATASET_REPS) - EXCLUDED)
    else:
        datasets = args.datasets.split()
    experiments = args.experiments.split()
    if args.print_datasets:
        print(' '.join(datasets))
        return
    out = os.path.abspath(os.path.join(REPO, args.output))
    if os.path.abspath(os.path.join(REPO, 'tests', 'results', 'noise')) == out and not args.controls:
        print('NOTE: writing to the real tests/results/noise tree without '
              '--controls. Stage 4 is stage4_noise.sh, which sets both.')
    log_dir = os.path.abspath(os.path.join(REPO, args.log_dir))

    tasks = build_tasks(datasets, experiments)

    # ── counts ──
    by_ds_exp = defaultdict(Counter)
    for ph, ds, e, c in tasks:
        by_ds_exp[ds][(e, ph)] += 1
    print(f'{len(tasks)} tasks over {len(datasets)} dataset(s)')
    for ds in datasets:
        row = by_ds_exp[ds]
        n = sum(row.values())
        mem = args.mem_big if ds.startswith(BIG_MEM) else args.mem_small
        print(f'  {ds:28s} {n:4d} tasks  {mem}')
        line = '    ' + '  '.join(
            f'{e}{"(B)" if ph == "B" else ""}={k}' for (e, ph), k in row.items())
        print(line)

    # ── cluster, account ──
    cluster, account = args.clusters, args.account
    unresolved = ''
    if cluster == 'best' or not account:
        pair, why = best_cluster_account(None if cluster == 'best' else cluster)
        if pair is None:
            unresolved = why
        else:
            if cluster == 'best':
                cluster = pair[0]
            account = account or pair[1]
    if cluster == 'best' or (not account and not args.dry_run):
        sys.exit(f'Could not resolve the cluster/account ({unresolved}). '
                 f'Pass --clusters and --account (or --account none).')
    acct_flag = [] if account in ('none', '') else [f'--account={account}']
    print(f'\ncluster {cluster}   account '
          f'{account if account not in ("none", "") else "<SLURM default>"}'
          f'{"   (lookup failed: " + unresolved + ")" if unresolved and not account else ""}'
          f'   partition {args.partition}   wall {args.wall}   cpus {args.cpus}')
    print(f'output {out}\nlogs   {log_dir}  (plus slurm-<array>_<task>.out in '
          f'the directory you submit from)')

    # ── extra dependencies (2026-09-25, owner: submit every stage with
    # dependencies). --after names jobs of an earlier stage; with
    # --wait-for-precompute every pc_<dataset>_* job still queued or running on
    # the target cluster is added, so a task never starts while that dataset's
    # searches are being written. A dependency cannot name a job on the other
    # cluster, so only the target cluster's queue is read.
    extra_dep = [j for j in args.after.replace(',', ':').split(':') if j.strip()]
    if args.wait_for_precompute:
        try:
            q = subprocess.run(['squeue', '-M', cluster, '-u', os.environ.get('USER', ''),
                                '-h', '-o', '%i %j'], capture_output=True, text=True).stdout
        except OSError:
            q = ''
        for line in q.splitlines():
            parts = line.split()
            # Exact dataset match: pc_<dataset>_<link> where link is a number
            # or x<number>. A prefix test matched herg_ki's jobs for herg
            # (pc_herg_ki_x1 starts with pc_herg_), 2026-09-28.
            if len(parts) == 2 and parts[0].isdigit() and any(
                    re.fullmatch(rf'pc_{re.escape(ds)}_x?\d+', parts[1])
                    for ds in datasets):
                extra_dep.append(parts[0])
    if extra_dep:
        print(f'waits for {len(extra_dep)} job(s): {":".join(extra_dep)}')

    # ── group into arrays ──
    groups = defaultdict(list)
    for t in tasks:
        ph, ds = t[0], t[1]
        mem = args.mem_big if ds.startswith(BIG_MEM) else args.mem_small
        groups[(ph, mem)].append(t)

    stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    task_dir = os.path.join(out, '_tasks', f'{args.job_prefix}_{stamp}')
    env_file = os.path.join(task_dir, 'env.sh')
    env = {
        'KIRBY_REPO': REPO,
        'KIRBY_OUTPUT_DIR': out,
        'KIRBY_EXP_LOG_DIR': log_dir,
        'PRUNING_MODE': args.pruning_mode,
        'KIRBY_NO_COLD_SEARCH': '1',
        'KIRBY_NO_PULL': '1',
        'KIRBY_SKIP_INSTALLS': '1',
        'KIRBY_NOISE_CONTROLS': '1' if args.controls else '0',
    }
    if args.similarity_settings:
        env['KIRBY_SIMILARITY_SETTINGS'] = args.similarity_settings
    if args.exp_timeout:
        env['EXP_TIMEOUT'] = args.exp_timeout

    if not args.dry_run:
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        with open(env_file, 'w') as fh:
            fh.write('# Written by stage1_prelim_noise.py; sourced by noise_task.sbatch.\n')
            for k, v in env.items():
                fh.write(f'export {k}={shlex.quote(v)}\n')
    print(f'env    {env_file}')
    for k, v in env.items():
        print(f'         {k}={v}')

    phase_a_ids = []
    all_ids = []
    n_ok = n_arrays = 0
    for phase in ('A', 'B'):
        for (ph, mem), items in sorted(groups.items()):
            if ph != phase:
                continue
            for ci in range(0, len(items), args.array_chunk):
                chunk = items[ci:ci + args.array_chunk]
                k = ci // args.array_chunk + 1
                name = f'{args.job_prefix}_{phase}_{mem}_{k}'
                tfile = os.path.join(task_dir, f'{name}.tsv')
                if not args.dry_run:
                    with open(tfile, 'w') as fh:
                        fh.write('# dataset\texperiment\tcondition (- = none)\n')
                        for _, ds, e, c in chunk:
                            fh.write(f'{ds}\t{e}\t{c}\n')
                arr = f'1-{len(chunk)}' + (f'%{args.max_concurrent}' if args.max_concurrent else '')
                cmd = ['sbatch', '--parsable', f'--clusters={cluster}', *acct_flag,
                       f'--partition={args.partition}', f'--time={args.wall}',
                       '--nodes=1', '--ntasks=1',
                       f'--cpus-per-task={args.cpus}', f'--mem={mem}',
                       f'--array={arr}', f'--job-name={name}',
                       f'--mail-user={EMAIL}', '--mail-type=FAIL',
                       f'--export=ALL,TASKS={tfile},RUNBOOK_ENV={env_file}']
                deps = list(extra_dep)
                if phase == 'B':
                    deps += phase_a_ids if phase_a_ids else (
                        ['<phase A array ids>'] if args.dry_run else [])
                if deps:
                    cmd.append(f'--dependency=afterany:{":".join(deps)}')
                if args.begin:
                    cmd.append(f'--begin={args.begin}')
                if args.hold:
                    cmd.append('--hold')
                cmd.append(os.path.join(HERE, 'noise_task.sbatch'))
                n_arrays += 1
                print(f'\n{name}: {len(chunk)} tasks, {mem}'
                      + ('   (starts after every phase-A array ends)' if phase == 'B' else ''))
                print('  ' + ' '.join(shlex.quote(x) for x in cmd))
                if args.dry_run:
                    for _, ds, e, c in chunk[:3]:
                        print(f'    task: {ds} {e} {c}')
                    if len(chunk) > 3:
                        print(f'    ... {len(chunk) - 3} more')
                    continue
                r = subprocess.run(cmd, capture_output=True, text=True)
                jid = r.stdout.strip().split(';')[0]
                if r.returncode != 0 or not jid:
                    print(f'  FAILED: {r.stderr.strip()}')
                    continue
                print(f'  submitted array {jid}')
                all_ids.append(jid)
                n_ok += 1
                if phase == 'A':
                    phase_a_ids.append(jid)

    if args.dry_run:
        print(f'\n--dry-run: {n_arrays} array(s) would be submitted; nothing sent, '
              f'no file written.')
    else:
        print(f'\nsubmitted {n_ok} of {n_arrays} array(s). Task files: {task_dir}')
        print(f'Progress: bash {os.path.join(HERE, "progress.sh")} {out} {log_dir}')
        # One machine-readable line for submit_all_stages.sh.
        print(f'JOBIDS {":".join(all_ids)}')


if __name__ == '__main__':
    main()
