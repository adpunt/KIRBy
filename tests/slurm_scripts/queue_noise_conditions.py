#!/usr/bin/env python3
"""Queue one noise job per (dataset, noise condition), 12 hours each.

Written 2026-09-23. The x_noise_<dataset> job runs every noise experiment in one
job, and seven of them sweep every condition one after another: 92 (experiment,
condition) runs per regression dataset, measured between 1,991 and 7,095 s each
for one condition on rf alone (noise_helpers.sweep_conditions docstring). That
does not fit a 2-day wall. This splits the sweep so each condition is its own
short job.

WHAT ONE JOB RUNS. submit_dataset.sh noise <dataset> with NOISE_CONDITION set.
run_all then runs only the experiments that take a condition -- n07, n09, n11,
n12, n13, n16, n20, n21, n23 -- and each writes
<experiment>/<dataset>__<condition>.json. n21 raises on the conditions outside
DETECTION_IS_DEFINED; the job logs that and moves on. The experiments that take
no condition (c10, n01, n02, n06, n14, n15, n17, n18, n19, n22) stay in the
x_noise_<dataset> job.

ORDER. Condition-major: every dataset's first condition is submitted before any
dataset's second, in SWEEP_ORDER (noise_helpers.py) for regression and
CLASSIFICATION_STRATEGIES order for classification. Earlier submission, earlier
start. Nothing is cut; --conditions narrows it only if you ask.

WAITS FOR. Every pending pc_<dataset>_* job, and on the datasets complexity is to
finish (rewire_held_jobs.FILL_BY_COMPLEXITY) the first x_complexity_<dataset>.

OVERLAP WITH x_noise_<dataset>. That job also sweeps every condition, skipping a
condition whose file is already on disk when it reaches it. If both are on the
same condition at once, the work is done twice and the later write wins; the two
files hold the same computation.

Needs the repo environment (it reads the condition lists and each dataset's task
type): run it after `. setup.sh`, in an srun or on the login node.

Usage:
    python tests/slurm_scripts/queue_noise_conditions.py                   # print only
    python tests/slurm_scripts/queue_noise_conditions.py --hold --apply
    python tests/slurm_scripts/queue_noise_conditions.py --datasets "herg_ki caco2" --links 2
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, HERE)

from rewire_held_jobs import FILL_BY_COMPLEXITY  # noqa: E402

EMAIL = 'adelaide.punt@stcatz.ox.ac.uk'
BIG_MEM = ('nt_', 'gb_', 'tcga_', 'kiba', 'bindingdb_kd', 'qm9')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--hold', action='store_true',
                    help='submit held; release with scontrol release')
    ap.add_argument('--datasets', default='')
    ap.add_argument('--conditions', default='',
                    help='space-separated subset; default every condition')
    ap.add_argument('--links', type=int, default=1,
                    help='jobs chained per (dataset, condition); a later link '
                         'resumes, skipping experiments already written')
    ap.add_argument('--wall', default='12:00:00')
    ap.add_argument('--partition', default='short,medium')
    ap.add_argument('--account', default='')
    ap.add_argument('--begin', default='')
    args = ap.parse_args()

    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments._shared import registry_lookup
    from experiments.noise.noise_helpers import SWEEP_ORDER
    from kirby.noise_spec import REGRESSION_STRATEGIES, CLASSIFICATION_STRATEGIES

    datasets = (args.datasets.split() if args.datasets
                else sorted(set(DATASET_REPS) - {'chembl_qsar', 'davis'}))
    reg = [c for c in SWEEP_ORDER if c in REGRESSION_STRATEGIES]
    reg += [c for c in REGRESSION_STRATEGIES if c not in reg]
    clf = list(CLASSIFICATION_STRATEGIES)
    only = set(args.conditions.split())

    per_ds = {}
    for ds in datasets:
        tt = getattr(registry_lookup(ds), 'task_type', None)
        conds = clf if tt == 'classification' else reg
        per_ds[ds] = [c for c in conds if not only or c in only]

    account = args.account
    if not account:
        r = subprocess.run(['bash', os.path.join(HERE, 'where_to_submit.sh'), '--emit'],
                           capture_output=True, text=True)
        account = (r.stdout.split() or [''])[0]
    if not account:
        sys.exit('Could not resolve an account. Pass --account.')

    user = os.environ.get('USER')
    rows = subprocess.run(['squeue', '-u', user, '-h', '-t', 'PD,R', '-o', '%i|%j'],
                          capture_output=True, text=True).stdout.splitlines()
    pc, first_cx, existing = defaultdict(list), {}, set()
    for r in rows:
        jid, name = r.split('|', 1)
        if not jid.isdigit():
            continue
        existing.add(name)
        m = re.fullmatch(r'pc_(.+)_(\d+|x\d+)', name)
        if m:
            pc[m.group(1)].append(int(jid))
        m = re.fullmatch(r'x_complexity_(.+)', name)
        if m and (m.group(1) not in first_cx or int(jid) < first_cx[m.group(1)]):
            first_cx[m.group(1)] = int(jid)

    order = []
    for i in range(max(len(v) for v in per_ds.values())):
        for ds in datasets:
            if i < len(per_ds[ds]):
                order.append((ds, per_ds[ds][i]))

    n_jobs = len(order) * args.links
    print(f"account {account}   partition {args.partition}   wall {args.wall}")
    print(f"{len(order)} (dataset, condition) pairs x {args.links} link(s) = {n_jobs} jobs"
          f"{'   HELD' if args.hold else ''}")
    for ds in datasets:
        dep = pc.get(ds, []) + ([first_cx[ds]] if ds in FILL_BY_COMPLEXITY and ds in first_cx else [])
        print(f"  {ds:28s} {len(per_ds[ds]):2d} conditions   waits for "
              f"{', '.join(map(str, dep)) if dep else 'nothing'}")
    if not args.apply:
        print("\nPrint only. Re-run with --apply to submit.")
        return

    ok = 0
    for ds, cond in order:
        mem = '128G' if ds.startswith(BIG_MEM) else '64G'
        dep = pc.get(ds, []) + ([first_cx[ds]] if ds in FILL_BY_COMPLEXITY and ds in first_cx else [])
        prev = None
        for link in range(1, args.links + 1):
            name = f"xn_{ds}__{cond}_{link}"
            if name in existing:
                print(f"  {name}: already queued, skipped")
                continue
            ids = [prev] if prev else dep
            cmd = ['sbatch', '--parsable', f'--account={account}',
                   f'--partition={args.partition}', f'--time={args.wall}',
                   '--cpus-per-task=8', f'--mem={mem}', f'--job-name={name}',
                   f'--mail-user={EMAIL}', '--mail-type=FAIL',
                   f'--export=ALL,NOISE_CONDITION={cond},PRUNING_MODE=calibrated,'
                   f'KIRBY_NO_COLD_SEARCH=1']
            if ids:
                cmd.append(f"--dependency=afterany:{':'.join(map(str, ids))}")
            if args.hold:
                cmd.append('--hold')
            if args.begin:
                cmd.append(f'--begin={args.begin}')
            cmd += [os.path.join(HERE, 'submit_dataset.sh'), 'noise', ds]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not r.stdout.strip():
                print(f"  FAILED {name}: {r.stderr.strip()}")
                break
            prev = int(r.stdout.strip().split(';')[0])
            ok += 1
    print(f"\nsubmitted {ok} of {n_jobs}")


if __name__ == '__main__':
    main()
