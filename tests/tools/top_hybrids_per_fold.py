"""Which representation counts and importance methods the best hybrids use.

Read-only. For each dataset, each per-fold search on disk and each of the
dataset's two decided models, takes the 3 best configurations of the general
search and prints, per representation count and per importance method, how
many of those were the best hybrid (top 1), among the best 2, and among the
best 3. With 5 folds and 2 models the denominators are 10, 20 and 30.

Only configurations inside the dataset's current decided grid are ranked
(models, budgets, importance methods, allocation methods, internal
evaluators), except representation counts: every stored count is ranked, so
a count already cut still shows whether it would have won. Rows scored on
fewer folds than the search asked for, non-finite scores and the
fold-0-pruned rows are left out. Ranking is score_mean in the dataset's
metric, the order the experiments read hybrids in.

    python tests/tools/top_hybrids_per_fold.py                 # every dataset
    python tests/tools/top_hybrids_per_fold.py lhasa_herg suzuki

Files: $KIRBY_CACHE_DIR/result_db/<dataset>__ocv5f<k>s42.db, k = 0..4.
Added 2026-10-07.
"""
import glob
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRID = json.loads((REPO / 'tests' / 'experiments' / 'decided_grid.json')
                  .read_text())['datasets']
PRUNED = '__prunedf0__'          # find_best_hybrid.PRUNED_FOLD0_SUBSET
LOWER_IS_BETTER = {'rmse', 'mae', 'mse'}


def _root():
    base = os.environ.get('KIRBY_CACHE_DIR') or os.path.expanduser(
        '~/.cache/kirby')
    return Path(base) / 'result_db'


def _rc(v):
    return 'all' if v in (None, '', 'None', 'all') else str(int(float(v)))


def fold_files(ds, root):
    return sorted(glob.glob(str(root / f'{ds}__ocv5f[0-4]s42.db')))


def top3(path, ds):
    """{model: [(rep_count, importance_method), best first]} for one fold."""
    g = GRID[ds]['general']
    ev = set(g['internal_evaluators'])
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    rows = con.execute(
        "SELECT model, internal_evaluator, budget, importance_method, "
        "allocation_method, rep_count, rep_subset, score_mean, metric, "
        "fold_scores_json, n_folds FROM results "
        "WHERE tier='general' AND aug_strategy='none' "
        "AND score_mean IS NOT NULL").fetchall()
    con.close()
    out = {}
    for m in g['models']:
        if m == 'knn':
            continue
        keep = []
        for (mo, e, b, im, al, rc, sub, s, metric, fs, nf) in rows:
            if mo != m or str(sub or '').startswith(PRUNED):
                continue
            if int(float(b)) not in g['budgets'] or im not in \
                    g['importance_methods'] or al not in g['allocation_methods']:
                continue
            if not (e in ev or ('match' in ev and e == m)):
                continue
            try:
                n_scored = len(json.loads(fs)) if fs else None
            except (TypeError, ValueError):
                n_scored = None
            if n_scored is not None and nf and n_scored < int(float(nf)):
                continue
            s = float(s)
            if s != s or s in (float('inf'), float('-inf')):
                continue
            keep.append((s, _rc(rc), im, metric))
        if not keep:
            out[m] = []
            continue
        lower = str(keep[0][3]).lower() in LOWER_IS_BETTER
        keep.sort(key=lambda r: r[0], reverse=not lower)
        out[m] = [(r[1], r[2]) for r in keep[:3]]
    return out


def report(ds, root):
    files = fold_files(ds, root)
    if not files:
        print(f'{ds}: no per-fold search files under {root}')
        return
    picks = [top3(f, ds) for f in files]
    n_slots = sum(1 for p in picks for m in p if p[m])
    g = GRID[ds]['general']
    in_grid = {'all' if x is None else str(x) for x in g['rep_counts']}
    print(f'\n{ds}: {len(files)} fold searches, '
          f'models {[m for m in g["models"] if m != "knn"]}, '
          f'{n_slots} (fold, model) pairs with results')
    for label, idx in (('representation count', 0),
                       ('importance method', 1)):
        tops = [Counter(), Counter(), Counter()]
        for p in picks:
            for m, lst in p.items():
                for rank, pick in enumerate(lst):
                    for upto in range(rank, 3):
                        tops[upto][pick[idx]] += 1
        keys = sorted(set().union(*tops), key=lambda k: (k == 'all', k))
        if idx == 0:
            keys = sorted(set(keys) | in_grid, key=lambda k: (k == 'all', k))
        print(f'  {label:22s} top 1 / {n_slots}   top 2 / {2 * n_slots}   '
              f'top 3 / {3 * n_slots}')
        for k in keys:
            note = '' if idx or k in in_grid else '   (not in the grid now)'
            print(f'    {k:20s} {tops[0][k]:>9d} {tops[1][k]:>13d} '
                  f'{tops[2][k]:>13d}{note}')


if __name__ == '__main__':
    root = _root()
    names = sys.argv[1:] or sorted(GRID)
    for ds in names:
        if ds not in GRID:
            print(f'{ds}: not in decided_grid.json')
            continue
        report(ds, root)
