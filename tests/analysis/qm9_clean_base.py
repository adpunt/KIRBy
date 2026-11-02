#!/usr/bin/env python3
"""Each noise method's no-noise result on QM9 beside its no-noise result on the
measured datasets.

WHY THIS READING, in Adelaide's words (2026-11-01, HANDOFF_NOISE_RESTRUCTURE.md
section 10):

  QM9's three targets have computed labels. They are the only clean data in the
  suite. Every measured dataset carries real noise as delivered, even with no
  noise added.

  A cleanup that raises the score on a measured dataset with no noise added is a
  SUCCESS. It found noise that was already there. Never count it as a failure.

  The ideal method also does nothing, or does slightly worse, on QM9 with no
  noise added. A method that raises QM9's score with no noise added is improving
  training, not removing noise.

So each method gets two numbers, both at level 0 (no noise added): what it does
to the measured datasets, and what it does to QM9. The pairing of the two is the
result; neither number alone says anything.

NO NEW RUNS. Level 0 is the first level of every noise grid
(noise_helpers.DECIDED_LEVELS, DECIDED_CENSORING_LEVELS,
SCREEN_CLASSIFICATION_LEVELS), so the rows this reads are already written by any
run of n42 to n47.

WHAT IT READS. The D-versus-D' rows every experiment appends,
<run>/<experiment>/<dataset>[__condition].rows.jsonl (n07.rows_path). One row is
one fit of one method on one (dataset, model, model input, fold, level, draw):

  score_D        the plain model on the same labels
  score_Dprime   the method
  all_D, all_Dprime   the same two in every metric (d_vs_dprime.ALL_METRICS)

The gain is score_D - score_Dprime where lower is better, and the other way
round where higher is better, so a positive gain always means the method helped.
Gains are reported per fold and as the spread across folds; no significance test
is run and no count of wins is reported (Adelaide's standing rules).

Usage:
    python tests/analysis/qm9_clean_base.py <run dir> [<run dir> ...]
        [--metric rmse] [--out table.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

# The clean base, from the one place that names it.
try:
    from experiments.noise.noise_helpers import CLEAN_BASE_DATASETS
except Exception:                                   # a checkout without the suite
    CLEAN_BASE_DATASETS = ('qm9', 'qm9_mu', 'qm9_alpha')

# Datasets left out of every count and table, whatever a run holds
# (CLAUDE.md, 2026-10-09). Patient or participant data.
PATIENT_PREFIXES = ('tcga_', 'annitia', 'nhanes')

LEVEL_0 = 0.0


def _is_patient(ds: str) -> bool:
    return str(ds).lower().startswith(PATIENT_PREFIXES)


def read_rows(run_dirs):
    """Every level-0 row under these run directories, patient data left out."""
    rows = []
    for run in run_dirs:
        for dirpath, _dirnames, filenames in os.walk(run):
            for fn in filenames:
                if not fn.endswith('.rows.jsonl'):
                    continue
                with open(os.path.join(dirpath, fn)) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if _is_patient(r.get('dataset', '')):
                            continue
                        try:
                            if float(r.get('level')) != LEVEL_0:
                                continue
                        except (TypeError, ValueError):
                            continue
                        rows.append(r)
    return rows


def gain(row, metric=None):
    """score_D - score_Dprime, signed so that positive always means the method
    helped. `metric` reads all_D / all_Dprime instead of the dataset's own."""
    if metric:
        d = (row.get('all_D') or {}).get(metric)
        p = (row.get('all_Dprime') or {}).get(metric)
        lower_is_better = metric in ('rmse', 'mae')
    else:
        d, p = row.get('score_D'), row.get('score_Dprime')
        lower_is_better = bool(row.get('lower_is_better'))
    if d is None or p is None:
        return None
    try:
        d, p = float(d), float(p)
    except (TypeError, ValueError):
        return None
    if d != d or p != p:                            # nan
        return None
    return (d - p) if lower_is_better else (p - d)


def _spread(values):
    """(lowest, highest) of a list, or (None, None). No mean and no test."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return min(vals), max(vals)


def collect(rows, metric=None):
    """{method: {'clean': [...], 'measured': {dataset: [per-fold gains]}}}"""
    out = defaultdict(lambda: {'clean': defaultdict(list),
                               'measured': defaultdict(list)})
    for r in rows:
        g = gain(r, metric)
        if g is None:
            continue
        ds = r.get('dataset')
        side = 'clean' if ds in CLEAN_BASE_DATASETS else 'measured'
        out[r.get('method')][side][ds].append(g)
    return out


def table(collected, metric_name):
    """One row per method. Says what one row is, above the table."""
    lines = [
        f'One row is one noise method at level 0, no noise added, in '
        f'{metric_name}.',
        '',
        'The measured columns count the datasets whose labels were measured; '
        'the QM9 columns count the three computed-label targets '
        f'({", ".join(CLEAN_BASE_DATASETS)}). A gain is the plain model\'s '
        'score minus the method\'s, signed so a positive number means the '
        'method raised the score. Lowest and highest are over every fold of '
        'every dataset on that side; no mean, no test.',
        '',
        '| method | measured: fits | measured: lowest gain | measured: highest '
        'gain | QM9: fits | QM9: lowest gain | QM9: highest gain | reading |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for method in sorted(collected):
        side = collected[method]
        m_all = [g for v in side['measured'].values() for g in v]
        c_all = [g for v in side['clean'].values() for g in v]
        m_lo, m_hi = _spread(m_all)
        c_lo, c_hi = _spread(c_all)
        reading = _reading(m_hi, c_hi)
        f = lambda x: '—' if x is None else f'{x:+.4f}'   # noqa: E731
        lines.append(
            f'| {method} | {len(m_all)} | {f(m_lo)} | {f(m_hi)} | '
            f'{len(c_all)} | {f(c_lo)} | {f(c_hi)} | {reading} |')
    return '\n'.join(lines)


def _reading(measured_best, clean_best):
    """Adelaide's reading of the pair, in her terms. Never a verdict on a
    method's worth: it says which of her two cases the pair falls in."""
    if measured_best is None or clean_best is None:
        return 'one side has no fit'
    helped_measured = measured_best > 0
    helped_clean = clean_best > 0
    if helped_measured and not helped_clean:
        return 'found real noise'
    if helped_measured and helped_clean:
        return 'raised QM9 too, so better training'
    if not helped_measured and not helped_clean:
        return 'raised neither'
    return 'raised QM9 only'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dirs', nargs='+', help='run directories to read')
    ap.add_argument('--metric', default=None,
                    help="read all_D / all_Dprime in this metric instead of "
                         "the dataset's own (rmse, mae, r2, spearman, pearson, "
                         "auc, accuracy, mcc, f1_weighted)")
    ap.add_argument('--out', default=None, help='write the table here')
    a = ap.parse_args(argv)

    rows = read_rows(a.run_dirs)
    if not rows:
        print(f'no level-0 rows under {", ".join(a.run_dirs)}. Nothing to '
              f'read: the experiments write <run>/<experiment>/<dataset>'
              f'.rows.jsonl, and level 0 is the first level of every noise '
              f'grid, so a run that got that far has them.')
        return 1
    collected = collect(rows, a.metric)
    datasets = sorted({r['dataset'] for r in rows})
    clean = [d for d in datasets if d in CLEAN_BASE_DATASETS]
    print(f'{len(rows)} level-0 rows, {len(datasets)} datasets, '
          f'{len(collected)} methods; clean base present: '
          f'{", ".join(clean) if clean else "none"}')
    if not clean:
        print('WITHOUT QM9 THERE IS NO COMPARISON POINT. Every gain below is '
              'on measured data, where a gain means the method found noise '
              'that was already there; nothing here can tell that from better '
              'training.')
    out = table(collected, a.metric or "each dataset's own metric")
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write(out + '\n')
        print(f'wrote {a.out}')
    else:
        print()
        print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
