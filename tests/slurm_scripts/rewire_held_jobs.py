#!/usr/bin/env python3
"""Fix the queued experiment jobs IN PLACE, so none of them loses its queue position.

Written 2026-09-23. Standard library only, so it runs on the login node with no
environment sourced. Prints every scontrol command; nothing changes without --apply.

WHAT IT CHANGES, AND WHY

1. THE WALL. Every x_* job was submitted with no --time and inherited
   7-00:00:00 from the header of submit_dataset.sh, on partitions short,medium.
   short allows 12 hours and medium 2 days (sinfo, 2026-09-23), so none of them
   could ever start. Each is lowered to 2-00:00:00. Slurm lets an owner LOWER a
   pending job's limit; the job keeps its place.

2. THE SECOND COPY OF EACH JOB. Every x_ and opig_ job was queued twice (two
   waves, same names, same --begin). Run together, the two copies write one
   dataset's result_db and the same result files at once. The later copy is made
   to wait for the earlier one (afterany). It then starts when the first ends and
   resumes where it stopped: finished experiments are skipped, and each noise
   condition already on disk is skipped. A second run of the same job, not a
   collision.

3. DATASETS WHOSE PRECOMPUTE WAS QUEUED TODAY. The 12 in NEW_PC got fresh
   precompute chains on 2026-09-23. The experiment jobs already queued for them
   carry no dependency on those chains, so they would start before the searches
   exist. The first copy of each is made to wait for every pending pc_<dataset>_*
   job.

4. DATASETS COMPLEXITY IS TO FINISH. The 13 in FILL_BY_COMPLEXITY have under 2
   core-hours of precompute left, and the owner's call (2026-09-23) is that the
   complexity job fills it. So on those, the similarity, noise and federated
   jobs, and any opig job reading them, wait for the first x_complexity job. The
   cost: they start only when complexity has ENDED, up to its 2-day wall.

Jobs stay HELD. Release them yourself when the code is ready.

Usage:
    python3 tests/slurm_scripts/rewire_held_jobs.py            # print only
    python3 tests/slurm_scripts/rewire_held_jobs.py --apply
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

NEW_PC = {
    'buchwald_hartwig', 'herg', 'ginkgo_titer', 'bh_doyle_plates', 'caco2',
    'adme_hlm', 'ginkgo_acsins', 'ginkgo_colloidal',
    'lhasa_herg', 'flab_binding_adams', 'flab_binding_koenig',
    'flab_expression_adams',
}
FILL_BY_COMPLEXITY = {
    'abagym_Ang2', 'abagym_EGFR', 'abagym_VEGF', 'abcov_kd', 'adme_mdr1',
    'adme_rlm', 'cdr_flexibility', 'ddg_combined', 'flab_thermo_tresanco',
    'ginkgo_hic', 'ginkgo_pr_cho', 'herg_ki', 'suzuki',
}
NEW_WALL = '2-00:00:00'
X_NAME = re.compile(r'^x_(complexity|similarity|noise|federated)_(.+)$')


def wall_seconds(s):
    """'7-00:00:00' / '1-23:59:00' / '12:00:00' -> seconds."""
    days = 0
    if '-' in s:
        d, s = s.split('-', 1)
        days = int(d)
    parts = [int(p) for p in s.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def opig_reads(repo):
    """opig job name -> datasets it reads, from OPIG_JOBS in submit_everything.sh.
    Read from that file rather than copied, so the two cannot disagree."""
    path = os.path.join(repo, 'tests/slurm_scripts/submit_everything.sh')
    out, inside = {}, False
    with open(path) as f:
        for line in f:
            if line.startswith("OPIG_JOBS='"):
                inside = True
                continue
            if inside and line.startswith("'"):
                break
            if inside and '|' in line:
                name, _mem, _cmd, ds = line.rstrip('\n').split('|')
                out[f'opig_{name}'] = ds.split()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    user = os.environ.get('USER')
    rows = subprocess.run(
        ['squeue', '-u', user, '-h', '-t', 'PD', '-o', '%i|%j|%l|%E'],
        capture_output=True, text=True, check=True).stdout.splitlines()
    jobs = []
    for r in rows:
        jid, name, limit, dep = r.split('|', 3)
        if '_' in jid and '[' in jid:      # array records: not ours to touch
            continue
        jobs.append({'id': int(jid), 'name': name, 'limit': limit,
                     'dep': [int(x) for x in re.findall(r'(\d+)', dep)]
                            if dep not in ('(null)', '') else []})

    by_name = defaultdict(list)
    for j in jobs:
        if j['name'].startswith(('x_', 'opig_')):
            by_name[j['name']].append(j)
    for v in by_name.values():
        v.sort(key=lambda j: j['id'])

    pc_ids = defaultdict(list)            # dataset -> pending precompute job ids
    for j in jobs:
        for ds in NEW_PC:
            if re.fullmatch(rf'pc_{re.escape(ds)}_(\d+|x\d+)', j['name']):
                pc_ids[ds].append(j['id'])

    first_complexity = {}
    for name, v in by_name.items():
        m = X_NAME.match(name)
        if m and m.group(1) == 'complexity':
            first_complexity[m.group(2)] = v[0]['id']

    reads = opig_reads(repo)
    cmds, notes = [], []

    for j in jobs:
        if X_NAME.match(j['name']) and wall_seconds(j['limit']) > wall_seconds(NEW_WALL):
            cmds.append(f"scontrol update JobId={j['id']} TimeLimit={NEW_WALL}")

    for name, copies in sorted(by_name.items()):
        first = copies[0]
        m = X_NAME.match(name)
        if m:
            cat, datasets = m.group(1), [m.group(2)]
        else:
            cat, datasets = None, reads.get(name)
            if datasets is None:
                notes.append(f"{name}: not in OPIG_JOBS (cut from the suite?) -- left alone")
                continue
        extra = []
        for ds in datasets:
            if ds in NEW_PC:
                if pc_ids.get(ds):
                    extra += pc_ids[ds]
                else:
                    notes.append(f"{name}: no pending pc_{ds}_* job found -- "
                                 f"was the new precompute submitted?")
            if ds in FILL_BY_COMPLEXITY and cat != 'complexity':
                if ds in first_complexity:
                    extra.append(first_complexity[ds])
                else:
                    notes.append(f"{name}: no x_complexity_{ds} job queued to wait for")
        if extra:
            ids = sorted(set(first['dep']) | set(extra))
            cmds.append(f"scontrol update JobId={first['id']} "
                        f"Dependency=afterany:{':'.join(map(str, ids))}")
        for prev, later in zip(copies, copies[1:]):
            cmds.append(f"scontrol update JobId={later['id']} "
                        f"Dependency=afterany:{prev['id']}")

    print(f"{len(cmds)} scontrol commands\n")
    for c in cmds:
        print(c)
    if notes:
        print("\nNOTES")
        for n in notes:
            print(f"  {n}")
    if not args.apply:
        print("\nPrint only. Re-run with --apply to change the queue.")
        return
    failed = 0
    for c in cmds:
        r = subprocess.run(c.split(), capture_output=True, text=True)
        if r.returncode != 0:
            failed += 1
            print(f"FAILED: {c}\n  {r.stderr.strip()}")
    print(f"\napplied {len(cmds) - failed} of {len(cmds)}; {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
