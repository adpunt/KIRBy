#!/usr/bin/env python3
"""What each noise experiment has: a smoke, a result, and optimisation loops.

WHY A SCRIPT AND NOT A TABLE. A table typed into a document is a dated claim.
Three of the four columns here can be read off disk, so they are, every time
this runs. Only the loop column cannot be derived, and that one lives in a
ledger a loop appends to when it closes.

WHAT ONE ROW IS. One noise experiment: the file `tests/experiments/noise/nNN_*.py`
that exposes `run(data, config, output_dir, dataset_name, **kwargs)`.

THE COLUMNS.

  smoke     which standalone check files import this experiment, and whether
            the last verification pass had them pass. A smoke that has never
            been run in a pass shows as `-`, not as passing.
  result    how many result files exist for it under the run directories, and
            the newest one's date. A result file means it ran to completion
            once; it says nothing about whether the numbers are good.
  loops     how many optimisation loops the ledger records: measured, changed
            one thing, re-measured, kept or reverted. A loop is only a loop
            when a before and an after number are both recorded.
  kind      method or investigation, and the one-line reason. A METHOD is an
            experiment whose output is something to ship -- a way of finding or
            repairing noisy labels -- so it is the kind that optimisation loops
            are for. An INVESTIGATION measures a property of the data or the
            pipeline; it needs to run and be believed, not to be tuned.

Run:  python tests/experiments/noise/coverage.py            # print the table
      python tests/experiments/noise/coverage.py --write    # and update the .md
"""
import argparse
import glob
import json
import os
import re
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, 'coverage_loops.json')
DOC = os.path.join(HERE, 'NOISE_COVERAGE.md')

# Where result files land. Add a directory here rather than teaching the script
# to guess.
RESULT_DIRS = [
    '/Volumes/seagate/local_noise_run/results',
    '/Volumes/seagate/local_noise_run/fast',
    '/Volumes/seagate/local_noise_run/fast/A_base',
    '/Volumes/seagate/local_noise_run/fast/B_weak',
    '/Volumes/seagate/local_noise_run/fast/C_saprot',
    '/Volumes/seagate/local_noise_run/fast/D_strong',
    '/Volumes/seagate/local_noise_run/loop_n07',
    '/Volumes/seagate/local_noise_run/loop_n11',
    # n11's first result file is written one level deeper than the other
    # loop directories, because its driver passes loop_n11/results as the
    # output root. Without this line the file exists and this table still
    # reports n11 as having none.
    '/Volumes/seagate/local_noise_run/loop_n11/results',
    '/Volumes/seagate/local_noise_run/loop_n18',
    '/Volumes/seagate/local_noise_run/loop_n20',
    '/Volumes/seagate/local_noise_run/loop_n23',
]

# Where a verification pass writes its SUMMARY.txt. Newest wins.
VERIFY_GLOB = '/Volumes/seagate/local_noise_run/verify_*/SUMMARY.txt'

# method / investigation, with the reason stated so it can be argued with.
# ⛔ This is the ONE judgement in the file. Everything else is read off disk.
KIND = {
    'n01': ('investigation', 'measures prediction error on cliff vs non-cliff molecules'),
    'n02': ('method', 'weights training rows by how reliable their label looks'),
    'n06': ('investigation', 'checks a detector against a real disagreement between sources'),
    'n07': ('method', 'seven corrections applied after noise is detected'),
    'n09': ('method', 'a survey of learning-with-noisy-labels methods'),
    'n11': ('method', 'every noise estimator crossed against every correction'),
    'n12': ('method', 'weights the distance by feature importance to find noisy labels'),
    'n13': ('investigation', 'how the result moves as the injected noise changes shape'),
    'n14': ('investigation', 'the cliff landscape and the stage-0 diagnostics'),
    'n15': ('investigation', 'whether a noisy search picks a different recipe'),
    'n16': ('investigation', 'whether feature importance moves as labels get noisier'),
    'n17': ('method', 'cleaning where the noise is real, measured against replicates'),
    'n18': ('method', 'cleaning the labels as they come, nothing injected'),
    'n19': ('investigation', 'what a measurement count and a measurement spread are worth'),
    'n20': ('method', 'cleaning before the recipe search rather than after'),
    'n21': ('method', 'which distance the nearest neighbours are found in'),
    'n22': ('investigation', 'whether detectors find the records a human marked wrong'),
    'n23': ('method', "cleaning from a model's own per-molecule uncertainty"),
    'n26': ('investigation', 'how large the label noise is, measured from repeat measurements, beside the model\'s own error'),
    'n27': ('investigation', 'what training on fewer molecules costs, and how much training-label error reaches the held-out error'),
    'n29': ('method', 'fits one offset per ChEMBL document and removes it from the training labels'),
    'n30': ('method', 'clipped labels: lift, drop, winsorise, or a censored likelihood'),
    'n31': ('investigation', 'the detector quality the model error predicts, beside what it reaches, and the flag precision where cleaning breaks even'),
    'n32': ('method', 'trimmed refits, two learners picking for each other, shrinking flagged labels only, and four more detectors'),
    'n33': ('method', 'robust losses, a transformed target, uncertainty weights, mixup and target-noise bagging'),
    'n34': ('method', 'a shrunk random offset per scaffold group, against subtracting each group mean'),
    'n35': ('investigation', 'how far apart the labels of near-identical molecules are, and downweighting pairs that disagree'),
    'n36': ('method', 'which molecules to give a second measurement, by five rules'),
}


def experiments():
    """Every nNN_*.py in this directory that exposes a run()."""
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'n[0-9][0-9]_*.py'))):
        stem = os.path.basename(p)[:-3]
        if stem.endswith('_analyse'):
            continue
        with open(p) as fh:
            if not re.search(r'^def run\(data, config', fh.read(), re.M):
                continue
        out[stem[:3]] = stem
    return out


def smokes_by_experiment(exp_ids):
    """Which smoke files import which experiment."""
    out = {e: [] for e in exp_ids}
    for p in sorted(glob.glob(os.path.join(HERE, '_smoke_*.py'))):
        name = os.path.basename(p)[:-3]
        with open(p) as fh:
            src = fh.read()
        for e in exp_ids:
            # Both import forms are in use in this directory:
            #   import experiments.noise.n21_neighbour_distance as n21
            #   from experiments.noise import n01_activity_cliff_framework as n01
            imported = (
                re.search(rf'import experiments\.noise\.{e}_\w+', src)
                or re.search(rf'from experiments\.noise import[^\n]*{e}_\w+', src))
            # A smoke that drives an experiment through _smoke_harness, or as a
            # subprocess, imports nothing. Its FILENAME is the author saying
            # what it covers: _smoke_n09_n13.py covers n09 and n13. Searching
            # the body for a bare 'n13' instead would match prose and comments.
            named = re.search(rf'_{e}(_|$)', name)
            if imported or named:
                out[e].append(name)
    return out


def latest_verify():
    """{smoke name: return code} from the newest verification pass, and its date."""
    files = sorted(glob.glob(VERIFY_GLOB), key=os.path.getmtime, reverse=True)
    if not files:
        return {}, None
    rc, path = {}, files[0]
    for line in open(path):
        m = re.match(r'(\S+)\s+rc=(\S+)', line)
        if m:
            rc[m.group(1)] = int(m.group(2))
    when = datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
    return rc, when


def results_by_experiment(exp_ids, full_names):
    out = {}
    for e in exp_ids:
        files = []
        for d in RESULT_DIRS:
            files += glob.glob(os.path.join(d, full_names[e], '*.json'))
        newest = (datetime.date.fromtimestamp(max(os.path.getmtime(f) for f in files))
                  .isoformat() if files else None)
        out[e] = (len(files), newest)
    return out


def ledger():
    if not os.path.exists(LEDGER):
        return {'loops': []}
    with open(LEDGER) as fh:
        return json.load(fh)


def build():
    full = experiments()
    ids = sorted(full)
    sm = smokes_by_experiment(ids)
    rc, verified_on = latest_verify()
    res = results_by_experiment(ids, full)
    led = ledger()
    loops = {}
    for L in led['loops']:
        loops.setdefault(L['experiment'], []).append(L)

    rows = []
    for e in ids:
        kind, why = KIND.get(e, ('?', ''))
        names = sm[e]
        if not names:
            smoke = 'NONE'
        else:
            seen = [rc[n] for n in names if n in rc]
            if not seen:
                smoke = f'{len(names)} file(s), not yet run'
            elif any(x != 0 for x in seen):
                bad = [n for n in names if rc.get(n, 0) != 0]
                smoke = f'FAIL ({", ".join(bad)})'
            elif len(seen) < len(names):
                smoke = f'{len(seen)} of {len(names)} pass, rest not yet run'
            else:
                smoke = 'pass'
        n_res, newest = res[e]
        result = f'{n_res} file(s), newest {newest}' if n_res else 'none'
        mine = loops.get(e, [])
        kept = sum(1 for L in mine if L.get('kept'))
        loop = f'{len(mine)} ({kept} kept)' if mine else '0'
        rows.append((e, full[e], kind, why, smoke, result, loop))
    return rows, verified_on, led


def render(rows, verified_on, led):
    out = []
    out.append('# Noise suite coverage\n')
    out.append('Generated by `tests/experiments/noise/coverage.py`. Do not edit '
               'by hand — the smoke, result and loop columns are read off disk '
               'and a ledger, so a hand edit is overwritten on the next run.\n')
    out.append(f'Generated {datetime.date.today().isoformat()}. '
               f'Smoke column from the verification pass of '
               f'{verified_on or "(none yet)"}.\n')
    out.append('One row is one noise experiment. **Smoke** is whether a '
               'standalone check exists and passed in the last pass. **Result** '
               'counts result files on disk; a result means it ran to '
               'completion once and says nothing about the numbers. **Loops** '
               'counts optimisation loops in the ledger — measured, changed one '
               'thing, re-measured — with how many of those changes were '
               'kept.\n')

    out.append('## Methods — these are what optimisation loops are for\n')
    out.append('| | What it is | Smoke | Result | Loops |')
    out.append('|---|---|---|---|---|')
    for e, _f, kind, why, smoke, result, loop in rows:
        if kind == 'method':
            out.append(f'| {e} | {why} | {smoke} | {result} | {loop} |')

    out.append('\n## Investigations — these need to run and be believed\n')
    out.append('| | What it measures | Smoke | Result | Loops |')
    out.append('|---|---|---|---|---|')
    for e, _f, kind, why, smoke, result, loop in rows:
        if kind == 'investigation':
            out.append(f'| {e} | {why} | {smoke} | {result} | {loop} |')

    n_method = sum(1 for r in rows if r[2] == 'method')
    n_no_smoke = sum(1 for r in rows if r[4] == 'NONE')
    n_no_result = sum(1 for r in rows if r[5] == 'none')
    n_no_loop = sum(1 for r in rows if r[2] == 'method' and r[6] == '0')
    out.append(f'\n## Where it stands\n')
    out.append(f'- {len(rows)} experiments, {n_method} of them methods.')
    out.append(f'- {n_no_smoke} have no smoke file at all.')
    out.append(f'- {n_no_result} have never written a result file.')
    out.append(f'- {n_no_loop} of the {n_method} methods have had no '
               f'optimisation loop.')

    if led['loops']:
        out.append('\n## The loops, newest first\n')
        out.append('| | Date | What changed | Before | After | Kept |')
        out.append('|---|---|---|---|---|---|')
        for L in sorted(led['loops'], key=lambda x: x.get('date', ''),
                        reverse=True):
            out.append(
                f"| {L['experiment']} | {L.get('date', '')} | "
                f"{L.get('changed', '')} | {L.get('before', '')} | "
                f"{L.get('after', '')} | {'yes' if L.get('kept') else 'no'} |")
        out.append('\nEvery before and after names its own units. A loop with '
                   'no before and after number is not a loop and is not listed '
                   'here.')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true',
                    help=f'write {os.path.basename(DOC)} as well as printing')
    a = ap.parse_args()
    rows, verified_on, led = build()
    text = render(rows, verified_on, led)
    print(text)
    if a.write:
        with open(DOC, 'w') as fh:
            fh.write(text)
        print(f'wrote {DOC}')


if __name__ == '__main__':
    main()
