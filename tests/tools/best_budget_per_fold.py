"""Best hybrid at each budget, per outer fold and downstream model.

Read-only. For each per-fold search on disk (one per outer fold), and each of
the dataset's decided models, prints the best configuration of the general
search at every budget: its mean score over the inner folds, the standard
deviation over those inner folds, and how many features the hybrid used. The
best budget for that fold and model is marked with *. All scores are on that
outer fold's training rows only; the held-out fold is never used here.

Rows are filtered as in top_hybrids_per_fold.py: inside the decided grid, no
augmentation, scored on the most inner folds any row in the file has, finite
scores, no fold-0-pruned rows.

    python tests/tools/best_budget_per_fold.py flab_thermo_tresanco

Files: $KIRBY_CACHE_DIR/result_db/<dataset>__ocv5f<k>s42.db, k = 0..4.
Added 2026-10-09.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from top_hybrids_per_fold import (GRID, LOWER_IS_BETTER, PRUNED, _root,  # noqa: E402
                                  fold_files)


def _n(fs):
    try:
        return len(json.loads(fs)) if fs else None
    except (TypeError, ValueError):
        return None


def best_per_budget(path, ds):
    """{model: {budget: (score_mean, score_std, n_features)}}, and the metric."""
    g = GRID[ds]['general']
    ev = set(g['internal_evaluators'])
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    rows = con.execute(
        "SELECT model, internal_evaluator, budget, importance_method, "
        "allocation_method, rep_subset, score_mean, score_std, metric, "
        "fold_scores_json, n_features FROM results "
        "WHERE tier='general' AND aug_strategy='none' "
        "AND score_mean IS NOT NULL").fetchall()
    con.close()
    full = max((_n(r[9]) or 0 for r in rows), default=0)
    metric = rows[0][8] if rows else None
    lower = str(metric).lower() in LOWER_IS_BETTER
    out = {}
    for (mo, e, b, im, al, sub, s, sd, _m, fs, nf) in rows:
        if mo == 'knn' or mo not in g['models']:
            continue
        if str(sub or '').startswith(PRUNED):
            continue
        b = int(float(b))
        if b not in g['budgets'] or im not in g['importance_methods'] \
                or al not in g['allocation_methods']:
            continue
        if not (e in ev or ('match' in ev and e == mo)):
            continue
        n_scored = _n(fs)
        if n_scored is not None and n_scored < full:
            continue
        s = float(s)
        if s != s or s in (float('inf'), float('-inf')):
            continue
        cur = out.setdefault(mo, {}).get(b)
        if cur is None or (s < cur[0] if lower else s > cur[0]):
            out[mo][b] = (s, None if sd is None else float(sd), nf)
    return out, metric, lower


def report(ds, root):
    files = fold_files(ds, root)
    if not files:
        print(f'{ds}: no per-fold search files under {root}')
        return
    models = [m for m in GRID[ds]['general']['models'] if m != 'knn']
    print(f'{ds}: {len(files)} outer folds, models {models}')
    for path in files:
        fold = re.search(r'ocv5f(\d)s42', path).group(1)
        res, metric, lower = best_per_budget(path, ds)
        print(f'\nouter fold {fold}  (metric {metric}, '
              f'{"lower" if lower else "higher"} is better; '
              f'mean and SD over inner folds)')
        for m in models:
            per_b = res.get(m, {})
            if not per_b:
                print(f'  {m}: no rows')
                continue
            pick = (min if lower else max)(per_b, key=lambda b: per_b[b][0])
            print(f'  {m}')
            print(f'    {"budget":>7s} {"mean":>8s} {"SD":>8s} {"features":>9s}')
            for b in sorted(per_b):
                s, sd, nf = per_b[b]
                sd_txt = f'{sd:8.4f}' if sd is not None else f'{"-":>8s}'
                star = '  *' if b == pick else ''
                print(f'    {b:>7d} {s:8.4f} {sd_txt} {str(nf):>9s}{star}')


if __name__ == '__main__':
    root = _root()
    for ds in sys.argv[1:]:
        if ds not in GRID:
            print(f'{ds}: not in decided_grid.json')
            continue
        report(ds, root)
