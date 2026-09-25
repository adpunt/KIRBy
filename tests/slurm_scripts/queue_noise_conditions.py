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
n12, n13, n16, n20, n21, n23, n32, n33, n34, n40, n41 -- and each writes
<experiment>/<dataset>__<condition>.json. n21 raises on the conditions outside
DETECTION_IS_DEFINED; the job logs that and moves on. The experiments that take
no condition (c10, n01, n02, n06, n14, n15, n17, n18, n19, n22, n26, n27,
n29, n30, n31, n35, n36, n37, n38, n39) stay in the x_noise_<dataset> job. n30 does inject,
but only censoring, which it names in its own file. n31 names its own five
conditions, two of which (shift_only_*) are outside the registry; n32 runs
those two only in the x_noise_<dataset> job, for the same reason. n38 sweeps
the conditions sized by the model's own error (RELATIVE_CONDITIONS), all
outside the registry; n41 runs the sparse_dprime_* ones and n40 dense_oofmse
only in the x_noise_<dataset> job, likewise.

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

CLUSTER AND ACCOUNT. CHANGED 2026-09-25. Fair share is kept per cluster AND per
account. Measured 2026-09-24, FairShare, higher starts sooner: htc stat-cadd
0.0491, htc stat-ecr 0.0360, arc stat-cadd 0.0326, arc stat-ecr 0.0195
(extend_precompute_chain.sh:48-56). The account used to come from
`where_to_submit.sh --emit`, which reads the login node's cluster only, and no
--clusters was passed, so every job went to the login node's cluster.
Now:
  --clusters   default: the cluster of `where_to_submit.sh --best-cluster-account`.
               Passed to sbatch as --clusters and to squeue as -M.
  --account    an explicit value always wins. 'none' passes no --account and
               SLURM bills the default association. Default: the best account
               ON THE CHOSEN CLUSTER, from --best-cluster-account <cluster>.
A dependency cannot reach a job on another cluster. A dataset with pc_ jobs
queued on a cluster other than the one chosen is therefore SKIPPED with a
message, never queued with no wait: its noise jobs would start before its
precompute ends.

Usage:
    python tests/slurm_scripts/queue_noise_conditions.py                   # print only
    python tests/slurm_scripts/queue_noise_conditions.py --hold --apply
    python tests/slurm_scripts/queue_noise_conditions.py --datasets "herg_ki caco2" --links 2
    python tests/slurm_scripts/queue_noise_conditions.py --clusters htc --account stat-cadd
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
    ap.add_argument('--account', default='',
                    help="explicit account; wins over the fair-share lookup. "
                         "'none' passes no --account")
    ap.add_argument('--clusters', default='',
                    help='arc or htc; default: the cluster with the best fair '
                         'share (where_to_submit.sh --best-cluster-account)')
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

    cluster, account = args.clusters, args.account
    if not cluster or not account:
        # An explicit --clusters restricts the lookup to that cluster, so the
        # account is the best one THERE, not the best overall.
        cmd = ['bash', os.path.join(HERE, 'where_to_submit.sh'),
               '--best-cluster-account'] + ([cluster] if cluster else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        best = r.stdout.split()
        if len(best) != 2:
            sys.exit(f'Could not resolve a cluster and account '
                     f'({r.stderr.strip() or "no output"}). '
                     f'Pass --clusters and --account.')
        cluster = cluster or best[0]
        account = account or best[1]
    account_flag = [] if account == 'none' else [f'--account={account}']

    user = os.environ.get('USER')
    rows = subprocess.run(['squeue', '-M', cluster, '-u', user, '-h', '-t', 'PD,R',
                           '-o', '%i|%j'],
                          capture_output=True, text=True).stdout.splitlines()
    # pc_ jobs on the OTHER cluster, which no dependency can reach.
    other_pc = defaultdict(list)
    for other in [c for c in ('arc', 'htc') if c != cluster]:
        for r in subprocess.run(['squeue', '-M', other, '-u', user, '-h',
                                 '-t', 'PD,R', '-o', '%i|%j'],
                                capture_output=True, text=True).stdout.splitlines():
            jid, _, name = r.partition('|')
            m = re.fullmatch(r'pc_(.+)_(general_.*|\d+|x\d+)', name)
            if jid.strip().isdigit() and m:
                other_pc[m.group(1)].append(f'{other}:{jid.strip()}')
    pc, first_cx, existing = defaultdict(list), {}, set()
    for r in rows:
        # squeue -M can print a "CLUSTER: <name>" line; it has no '|'.
        if '|' not in r:
            continue
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

    blocked = {ds: other_pc[ds] for ds in datasets if other_pc.get(ds)}
    for ds, ids in blocked.items():
        print(f"  {ds}: SKIPPED -- precompute queued on another cluster "
              f"({', '.join(ids)}); a dependency cannot reach it. Queue this "
              f"dataset on that cluster, or after its precompute ends.")
    datasets = [ds for ds in datasets if ds not in blocked]
    if not datasets:
        print("Nothing left to queue.")
        return

    order = []
    for i in range(max(len(per_ds[ds]) for ds in datasets)):
        for ds in datasets:
            if i < len(per_ds[ds]):
                order.append((ds, per_ds[ds][i]))

    n_jobs = len(order) * args.links
    print(f"cluster {cluster}   account {account if account != 'none' else '<SLURM default>'}"
          f"   partition {args.partition}   wall {args.wall}")
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
            cmd = ['sbatch', '--parsable', f'--clusters={cluster}', *account_flag,
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
