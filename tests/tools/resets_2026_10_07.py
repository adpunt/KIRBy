"""Resets after the 2026-10-07 code changes. Run on the server, from the repo.

    python tests/tools/resets_2026_10_07.py              # apply
    python tests/tools/resets_2026_10_07.py --dry-run    # print only
    python tests/tools/resets_2026_10_07.py --with-c22   # also c22 (after the
                                                         # c22 SHAP change)

Safe to run more than once. A unit is reset only when (a) the ledger says ok
or failed, never pending or running, and (b) its last attempt STARTED before
this script first ran for that change (the time is kept in
<run>/.resets_2026_10_07.json). So a unit that was running the first time is
reset on a later run once it finishes, and a unit that started after the
first run (new code) is never reset again.

Every ledger call is retried: the ledger file sits on a shared filesystem and
reading it while a worker writes it raises "Stale file handle".
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER = os.path.join(REPO, 'tests', 'experiments', 'orchestrate', 'ledger.py')
RUN = os.path.join(REPO, 'tests', 'results', 'run_2026_10_06')
STAMPS = os.path.join(RUN, '.resets_2026_10_07.json')

FAST_13 = ['abagym_Ang2', 'abagym_EGFR', 'abagym_VEGF', 'abcov_kd',
           'cdr_flexibility', 'ddg_combined', 'ginkgo_acsins',
           'ginkgo_colloidal', 'ginkgo_hic', 'ginkgo_pr_cho', 'ginkgo_titer',
           'herg_ki', 'tz_her2_binary']
FOLD_EXPS = {'c05', 'c06', 'c07', 'c13', 'c15', 'c23', 'c33', 'n15', 'n26'}


def ledger(*args, tries=6):
    """stdout of one ledger call, or None when the row does not exist."""
    err = ''
    for i in range(tries):
        p = subprocess.run([sys.executable, LEDGER, *args],
                           capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        err = p.stderr
        if 'no row' in err:
            return None
        time.sleep(2 + 2 * i)
    raise RuntimeError(f'ledger {" ".join(args)} failed {tries} times:\n{err}')


def units(row):
    """[(unit, status, log path)] for one ledger row."""
    out = ledger('show', row)
    if out is None:
        return []
    got = []
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 3 and not f[0].startswith('[') and re.match(r'^(ok|failed|pending|running|stale|'
                                    r'skipped|blocked|held)$', f[1]):
            log = next((x[4:] for x in f if x.startswith('log=')), '')
            got.append((f[0], f[1], log))
    return got


def last_start(log):
    """Start time of the last attempt in a unit's log, or None."""
    if not log or not os.path.exists(log):
        return None
    t = None
    with open(log, errors='replace') as fh:
        for line in fh:
            m = re.match(r'^=== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
            if m:
                t = m.group(1)
    return dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') if t else None


def last_attempt_text(log):
    if not log or not os.path.exists(log):
        return ''
    text = open(log, errors='replace').read()
    i = text.rfind('\n=== ')
    return text[i:] if i >= 0 else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--with-c22', action='store_true')
    a = ap.parse_args()

    stamps = json.load(open(STAMPS)) if os.path.exists(STAMPS) else {}
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def cutoff(key):
        if key not in stamps:
            stamps[key] = now
        return dt.datetime.strptime(stamps[key], '%Y-%m-%d %H:%M:%S')

    grid = json.load(open(os.path.join(REPO, 'tests', 'experiments',
                                       'decided_grid.json')))['datasets']
    rows = sorted(grid) + ['opig:trastuzumab']

    # Which change each unit belongs to.
    def change_of(row, unit):
        if unit.startswith('c10'):
            return 'c10_part3'
        if unit == 'c18':
            return 'c18_every_rep'
        if unit == 's02':
            return 's02_weighted'
        if unit.startswith('n09'):
            return 'n09_restored'
        if unit == 'c22':
            return 'c22_shap' if a.with_c22 else None
        if unit.startswith('n12'):
            return 'n12_no_smiles'
        if row == 'flab_binding_adams' and unit.startswith('search_f'):
            return 'flab_binding_adams_all'
        if row == 'suzuki' and unit.startswith('search_f'):
            return 'suzuki_budgets'
        if row == 'opig:trastuzumab':
            return 'trastuzumab_search'
        if row in FAST_13 and unit in FOLD_EXPS:
            return 'all_five_folds'
        return None

    # n12 files that only record the 2026-10-06 "no SMILES" skip.
    n12_skip_rows = set()
    for f in glob.glob(os.path.join(RUN, '**', '*.json'), recursive=True):
        if 'n12' not in f:
            continue
        try:
            txt = open(f, errors='replace').read(4000)
        except OSError:
            continue
        if 'no_smiles' in txt:
            n12_skip_rows.add(os.path.basename(f).split('__')[0].replace('.json', ''))
            print(f'delete {f}')
            if not a.dry_run:
                os.remove(f)

    n_reset, waiting = 0, []
    for row in rows:
        for unit, status, log in units(row):
            key = change_of(row, unit)
            if key is None:
                continue
            if key == 'n12_no_smiles' and row not in n12_skip_rows:
                continue
            if key == 'all_five_folds' and \
                    'running outer folds 0..2' not in last_attempt_text(log):
                continue
            cut = cutoff(key)
            started = last_start(log)
            if status in ('pending', 'running', 'stale'):
                if status == 'running' and started and started < cut:
                    waiting.append(f'{row} {unit}')
                continue
            if status not in ('ok', 'failed'):
                continue
            if started is not None and started >= cut:
                continue                     # ran on the new code already
            print(f'reset {row} {unit}   ({key}; last start {started})')
            n_reset += 1
            if not a.dry_run:
                ledger('reset', row, unit)

    if not a.dry_run:
        with open(STAMPS, 'w') as fh:
            json.dump(stamps, fh, indent=1)
    print(f'\n{n_reset} unit(s) reset{" (dry run)" if a.dry_run else ""}.')
    if waiting:
        print('Running on the old code; run this script again after they '
              'finish: ' + ', '.join(waiting))


if __name__ == '__main__':
    main()
