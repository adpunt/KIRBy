#!/usr/bin/env python3
"""Read an n21 loop result, and count the disagreement at the right unit.

WHY THIS EXISTS. n21 calls a distance better than another when the mean of the
per-cell gains is larger than the spread of those same gains
(`_compare.gain_exceeds_its_own_spread`). A cell is one (fold, draw, detector,
neighbour count). That flag is a ratio of a mean to a spread, and the spread of
the per-cell gains does not get smaller when more folds and more draws are run
-- it is a property of how much the gain varies from cell to cell, not of how
many cells were measured. So no number of folds can turn that flag from false
to true. What more folds and more draws buy is a mean that is known more
precisely and a count of wins out of more tries.

AND THE CELLS ARE NOT INDEPENDENT. Six of them -- two detectors by three
neighbour counts -- are read off the SAME fold, the SAME corruption draw and
the same molecules, and the neighbour sets are nested: the five nearest
molecules are the first five of the twenty nearest. Ninety cells over fifteen
(fold, draw) passes is fifteen independent corruptions, not ninety. So this
averages the gain within each pass first and reports the spread ACROSS passes,
which is the unit that is repeated.

Three numbers per comparison, and what each counts:
  passes won of passes paired    one pass is one (fold, draw) -- one corruption
                                 of one fold's training molecules
  mean gain over passes          average precision, the first distance minus
                                 the second, averaged within a pass and then
                                 over passes
  spread over passes             the standard deviation of those per-pass gains

Run:  python tests/experiments/noise/_loop_n21_read.py <result.json> [...]
"""
import json
import sys
from collections import defaultdict

import numpy as np


def _by_pass(cells):
    """{pass key: mean average precision over that pass's cells}."""
    out = defaultdict(list)
    for cell, v in cells.items():
        if v is None or not np.isfinite(float(v)):
            continue
        out[cell.split('|')[-1]].append(float(v))
    return {p: float(np.mean(vs)) for p, vs in out.items()}


def compare_by_pass(a_cells, b_cells):
    a, b = _by_pass(a_cells), _by_pass(b_cells)
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    g = np.array([a[p] - b[p] for p in shared], dtype=float)
    return {
        'passes_paired': int(g.size),
        'passes_won': int((g > 0).sum()),
        'passes_lost': int((g < 0).sum()),
        'mean_gain_over_passes': float(g.mean()),
        'spread_over_passes': float(g.std(ddof=1)) if g.size > 1
        else float('nan'),
    }


def read(path):
    with open(path) as fh:
        d = json.load(fh)
    cells = (d.get('average_precision_per_cell')
             or d.get('average_precision_per_arm') or {})
    ranked = sorted(((k, v) for k, v in
                     (d.get('average_precision_by_space')
                      or d.get('average_precision_by_arm') or {}).items()
                     if v is not None and np.isfinite(v)),
                    key=lambda t: -t[1])
    print(f"\n{path}")
    print(f"  {d.get('dataset')}, {d.get('molecules')} molecules, "
          f"{d.get('folds')} folds, draws {d.get('draws')}, "
          f"level {d.get('level')}, condition {d.get('noise_condition')}")
    print(f"  plain representation {d.get('plain_representation')}, "
          f"{d.get('plain_representation_features')} features")
    corrupted = d.get('molecules_corrupted_mean')
    in_fold = d.get('molecules_in_training_fold_mean')
    if corrupted and in_fold:
        print(f"  the injector moved the label of {corrupted:.0f} molecules "
              f"of the {in_fold:.0f} in a training fold, on average")
    if not ranked:
        print("  no finished reading in this file yet")
        return d
    n_passes = len(_by_pass(cells[ranked[0][0]]))
    print(f"  one row is one distance; detection average precision, averaged "
          f"over {n_passes} passes of (fold, draw)")
    for k, v in ranked:
        print(f"    {k:42s} {v:.4f}")

    _against(cells, ranked, ranked[0][0],
             'the distance that leads the table')
    shipped = (d.get('shipped_setting') or {}).get('arm')
    if shipped and shipped in cells and shipped != ranked[0][0]:
        _against(cells, ranked, shipped,
                 'the setting n21 ships, which is what this file exists to '
                 'test')
    return d


def _against(cells, ranked, top, what):
    print(f"\n  {top} -- {what} -- against each of the others. One pass is "
          f"one (fold, draw): the six readings inside a pass -- two "
          f"detectors by three neighbour counts -- are averaged first, "
          f"because they are the same molecules under the same corruption.")
    for k, _v in ranked:
        if k == top:
            continue
        c = compare_by_pass(cells.get(top, {}), cells.get(k, {}))
        if c is None:
            print(f"    {k}: no pass was scored on both sides")
            continue
        sep = ('   ahead on every pass' if c['passes_won'] == c['passes_paired']
               else '')
        print(f"    vs {k:40s} won {c['passes_won']} of {c['passes_paired']} "
              f"passes, mean {c['mean_gain_over_passes']:+.4f} average "
              f"precision (spread over passes "
              f"{c['spread_over_passes']:.4f}){sep}")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        read(p)
