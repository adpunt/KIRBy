#!/usr/bin/env python3
"""f31 analysis — one table per dataset, never pooled across datasets.

Every metric here is higher-is-better (r2 for buchwald_hartwig and caco2,
spearman for flab_expression_koenig), so a bigger number is a better number in
every column. The datasets are NOT comparable to each other and are never
averaged together.

Three outer folds rank the settings. They do not support quoting a delta as a
result.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

from f31_internal_evaluator_tuning import DEFAULT_ARM_OF, EVALUATOR_SETTINGS


def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def analyse(path):
    d = json.loads(Path(path).read_text())
    ds, primary = d['dataset'], d['primary_metric']
    rows = [r for r in d['rows'] if r.get('downstream')]
    settings = [s for s in EVALUATOR_SETTINGS if any(r['setting'] == s for r in rows)]
    downstream = d['downstream']
    n_folds = len(set(r['fold'] for r in rows))

    print(f'\n{"="*78}\n{ds} — primary {primary} (higher is better), '
          f'{n_folds} folds, budget {d["budget"]}\n{"="*78}')

    # ---- setting x downstream, averaged over folds and allocations ----
    hdr = f'{"setting":26}' + ''.join(f'{m:>12}' for m in downstream) + f'{"all":>12}{"vs default":>12}'
    print(hdr)
    print('-' * len(hdr))
    fam_default = {}
    per_setting = {}
    for s in settings:
        cells = {m: _mean([r['primary'] for r in rows
                           if r['setting'] == s and r['downstream'] == m])
                 for m in downstream}
        overall = _mean([r['primary'] for r in rows if r['setting'] == s])
        per_setting[s] = (cells, overall)
        fam = EVALUATOR_SETTINGS[s][0]
        if s == DEFAULT_ARM_OF.get(fam):
            fam_default[fam] = overall

    last_fam = None
    for s in settings:
        fam = EVALUATOR_SETTINGS[s][0]
        if last_fam is not None and fam != last_fam:
            print()
        last_fam = fam
        cells, overall = per_setting[s]
        base = fam_default.get(fam)
        delta = ('' if base is None or not np.isfinite(base)
                 else f'{overall - base:+.4f}')
        mark = '  (default)' if s == DEFAULT_ARM_OF.get(fam) else ''
        print(f'{s:26}' + ''.join(f'{cells[m]:>12.4f}' for m in downstream)
              + f'{overall:>12.4f}{delta:>12}' + mark)

    # ---- is anything in that table bigger than the fold noise? ----
    # The means above are taken over 3 folds. If one setting's score moves more
    # between folds than the settings move against each other within a fold,
    # the table is noise and only the paired win counts below say anything.
    within, between = [], []
    for s in settings:
        for a in d['allocations']:
            for m in downstream:
                v = [r['primary'] for r in rows
                     if r['setting'] == s and r['allocation'] == a
                     and r['downstream'] == m and r['primary'] is not None
                     and np.isfinite(r['primary'])]
                if len(v) > 1:
                    within.append(np.std(v, ddof=1))
    for f in sorted(set(r['fold'] for r in rows)):
        for a in d['allocations']:
            for m in downstream:
                v = [r['primary'] for r in rows
                     if r['fold'] == f and r['allocation'] == a
                     and r['downstream'] == m and r['primary'] is not None
                     and np.isfinite(r['primary'])]
                if len(v) > 1:
                    between.append(np.std(v, ddof=1))
    means = [per_setting[s][1] for s in settings if np.isfinite(per_setting[s][1])]
    print(f'\nfold-to-fold sd, one setting fixed    : {_mean(within):.4f}')
    print(f'setting-to-setting sd, one fold fixed : {_mean(between):.4f}')
    print(f'full range of the table above         : '
          f'{(max(means)-min(means)) if means else float("nan"):.4f}')
    if _mean(within) > _mean(between):
        print('-> one setting moves MORE between folds than the settings move '
              'against each other.\n   Read the paired win counts below, not '
              'the means above.')

    # ---- does any tuned setting beat its family default, cell by cell? ----
    print(f'\nwin count against the family default, over '
          f'{n_folds} folds x {len(d["allocations"])} allocations x '
          f'{len(downstream)} downstream = '
          f'{n_folds*len(d["allocations"])*len(downstream)} cells')
    for fam in dict.fromkeys(EVALUATOR_SETTINGS[s][0] for s in settings):
        dflt = DEFAULT_ARM_OF[fam]
        base = {(r['fold'], r['allocation'], r['downstream']): r['primary']
                for r in rows if r['setting'] == dflt}
        for s in settings:
            if EVALUATOR_SETTINGS[s][0] != fam or s == dflt:
                continue
            wins = tot = 0
            for r in rows:
                if r['setting'] != s:
                    continue
                b = base.get((r['fold'], r['allocation'], r['downstream']))
                if b is None or r['primary'] is None:
                    continue
                if not (np.isfinite(b) and np.isfinite(r['primary'])):
                    continue
                tot += 1
                wins += r['primary'] > b
            print(f'  {s:26} {wins}/{tot}')

    # ---- did the setting move the budget split at all? ----
    print('\ndistinct budget splits across settings, per (fold, allocation):')
    for fold in sorted(set(r['fold'] for r in rows)):
        for alloc in d['allocations']:
            here = [r for r in rows
                    if r['fold'] == fold and r['allocation'] == alloc]
            seen = {json.dumps(r['allocation_dict'], sort_keys=True) for r in here}
            n_set = len({r['setting'] for r in here})
            print(f'  fold {fold} {alloc:24} {len(seen)} distinct '
                  f'of {n_set} settings')

    # ---- build cost ----
    print('\nseconds per hybrid build, mean over folds and allocations:')
    for s in settings:
        secs = _mean([r['build_seconds'] for r in rows if r['setting'] == s])
        print(f'  {s:26} {secs:7.1f}s')

    errs = [r for r in d['rows'] if r.get('error')]
    if errs:
        print(f'\n{len(errs)} errors:')
        for e in errs[:10]:
            print(f"  fold {e.get('fold')} {e.get('setting')}/"
                  f"{e.get('allocation')}: {e['error']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(_ROOT / 'tests' / 'results' / 'framework'))
    ap.add_argument('--datasets', nargs='+', default=None)
    a = ap.parse_args()
    files = sorted(Path(a.results).glob('f31_*.json'))
    if a.datasets:
        files = [f for f in files if f.stem.replace('f31_', '') in a.datasets]
    if not files:
        print('no f31 result files yet')
        return
    for f in files:
        analyse(f)


if __name__ == '__main__':
    main()
