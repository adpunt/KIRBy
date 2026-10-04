#!/usr/bin/env python3
"""Read f36's rows and print, per task, what the owner asked for (F36.md).

A task is (dataset, model input, model, noise condition, level). The
settings f36 decides are named in each row's method_setting. Level is kept
inside the task: averaging level 0.2 with level 1.5 would average two different
amounts of noise. Nothing is averaged across datasets. No significance test.

One combination's score in a task: per fold, the mean of score_Dprime over the
noise draws; reported as the per-fold values, their mean and their standard
deviation over folds. Ranking is by that mean, best first, in the direction
lower_is_better says. 'no_smoothing' (the model fitted on D) is ranked beside
the combinations, so a task where nothing beats doing nothing shows it.

A combination missing a score on any fold of a task (it declined, or a fit
failed) is not ranked in that task, and the count of those is printed.

Printed:
  (1) per task: the best 10 by mean, with per-fold scores and rank; the full
      ranking of every combination goes to <rows dir>/f36_rankings.csv;
  (2) per combination: tasks won outright, and tasks tied for first;
  (3) for each tolerance (0%, 1%, 2% of the task's best score): the smallest
      set of combinations that has one within tolerance of the best in every
      task, solved exactly as a set cover (scipy.optimize.milp), and for each
      member the tasks only it covers.

Usage:
    python tests/experiments/framework/f36_analyse.py tests/results/f36/f36_label_smoothing_settings
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

NO_SMOOTHING = 'no_smoothing'


def read_rows(root):
    for dirpath, _d, files in os.walk(root):
        for f in files:
            if not f.endswith('.rows.jsonl'):
                continue
            with open(os.path.join(dirpath, f)) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue


def collect(root):
    """{task: {'lib': bool, 'metric': str,
               'combos': {name: {fold: [scores over draws]}}}}"""
    tasks = {}
    for r in read_rows(root):
        task = (r['dataset'], r['model_input'], r['model'],
                r['noise_condition'], float(r['level']))
        t = tasks.setdefault(task, {'lib': bool(r['lower_is_better']),
                                    'metric': r['metric'],
                                    'combos': defaultdict(lambda: defaultdict(dict))})
        f, d = int(r['fold']), int(r['draw'])
        t['combos'][r['method_setting']][f][d] = float(r['score_Dprime'])
        t['combos'][NO_SMOOTHING][f][d] = float(r['score_D'])
    return tasks


def task_table(t):
    """[(name, per-fold means, mean, sd)] for combinations scored on every
    fold of the task, best first; and how many were left out."""
    folds = sorted({f for c in t['combos'].values() for f in c})
    out, left_out = [], 0
    for name, by_fold in t['combos'].items():
        per = []
        for f in folds:
            vals = [v for v in by_fold.get(f, {}).values()]
            if not vals or not all(np.isfinite(vals)):
                per = None
                break
            per.append(float(np.mean(vals)))
        if per is None:
            left_out += 1
            continue
        out.append((name, per, float(np.mean(per)), float(np.std(per))))
    out.sort(key=lambda x: x[2] if t['lib'] else -x[2])
    return folds, out, left_out


def within(score, best, tol, lib):
    margin = tol * abs(best)
    return score <= best + margin if lib else score >= best - margin


def smallest_cover(cover):
    """cover: {task: set(names)}. The fewest names hitting every task, exact,
    by integer programming. Returns (names, tasks no name covers)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    tasks = [t for t, s in cover.items() if s]
    empty = [t for t, s in cover.items() if not s]
    names = sorted({n for s in cover.values() for n in s})
    if not tasks:
        return [], empty
    col = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(tasks), len(names)))
    for i, t in enumerate(tasks):
        for n in cover[t]:
            A[i, col[n]] = 1.0
    res = milp(c=np.ones(len(names)),
               constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(len(names)), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f'set cover did not solve: {res.message}')
    chosen = [names[i] for i in np.flatnonzero(res.x > 0.5)]
    return chosen, empty


def tname(task):
    ds, mi, m, cond, lvl = task
    return f'{ds} | {mi} | {m} | {cond} | level {lvl:g}'


def analyse(root, tolerances=(0.0, 0.01, 0.02), out=sys.stdout, top=10):
    tasks = collect(root)
    P = lambda *a: print(*a, file=out)  # noqa: E731
    P(f'{len(tasks)} tasks read from {root}')
    tables = {}
    csv_path = os.path.join(root, 'f36_rankings.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'model_input', 'model', 'condition', 'level',
                    'metric', 'lower_is_better', 'rank', 'combination',
                    'mean_over_folds', 'sd_over_folds', 'per_fold'])
        for task in sorted(tasks):
            t = tasks[task]
            folds, tab, left_out = task_table(t)
            tables[task] = (t, tab)
            for rank, (name, per, mu, sd) in enumerate(tab, 1):
                w.writerow([*task, t['metric'], t['lib'], rank, name,
                            f'{mu:.6g}', f'{sd:.6g}',
                            ' '.join(f'{x:.6g}' for x in per)])
            P(f'\n(1) {tname(task)}  [{t["metric"]}, '
              f'{"lower" if t["lib"] else "higher"} is better, '
              f'{len(folds)} folds, {len(tab)} ranked, {left_out} not scored '
              f'on every fold]')
            for rank, (name, per, mu, sd) in enumerate(tab[:top], 1):
                P(f'    {rank:>4}  {mu:.4f} (sd {sd:.4f})  folds '
                  f'{" ".join(f"{x:.4f}" for x in per)}  {name}')
            nos = [i for i, x in enumerate(tab) if x[0] == NO_SMOOTHING]
            if nos:
                i = nos[0]
                P(f'    no_smoothing is rank {i + 1} of {len(tab)}: '
                  f'{tab[i][2]:.4f}')
    P(f'\nfull rankings: {csv_path}')

    # (2) wins and ties
    wins, ties = defaultdict(int), defaultdict(int)
    for task, (t, tab) in tables.items():
        if not tab:
            continue
        best = tab[0][2]
        top_names = [n for n, _p, mu, _s in tab if mu == best]
        for n in top_names:
            (wins if len(top_names) == 1 else ties)[n] += 1
    P(f'\n(2) tasks won outright and tasks tied for first, '
      f'of {len(tables)} tasks')
    for n in sorted(set(wins) | set(ties),
                    key=lambda n: (-wins[n], -ties[n], n)):
        P(f'    won {wins[n]:>4}  tied {ties[n]:>4}  {n}')

    # (3) smallest covering set
    report = {'n_tasks': len(tables), 'covers': {}}
    for tol in tolerances:
        cover = {}
        for task, (t, tab) in tables.items():
            if not tab:
                cover[task] = set()
                continue
            best = tab[0][2]
            cover[task] = {n for n, _p, mu, _s in tab
                           if within(mu, best, tol, t['lib'])}
        chosen, empty = smallest_cover(cover)
        P(f'\n(3) tolerance {tol:.0%} of the task best: {len(chosen)} '
          f'combinations cover all {len(cover) - len(empty)} tasks'
          + (f' ({len(empty)} tasks have no ranked combination)'
             if empty else ''))
        for n in chosen:
            only = [task for task, s in cover.items()
                    if n in s and not (s & (set(chosen) - {n}))]
            hit = [task for task, s in cover.items() if n in s]
            P(f'    {n}')
            P(f'        within tolerance in {len(hit)} tasks; the only chosen '
              f'one in {len(only)}:')
            for task in sorted(only):
                P(f'          {tname(task)}')
        report['covers'][tol] = chosen
    return report


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    analyse(sys.argv[1])
