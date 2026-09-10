#!/usr/bin/env python3
"""Read every f27 result file and answer the two questions it was built for.

    python tests/analysis/read_f27_objective_and_budget.py
    python tests/analysis/read_f27_objective_and_budget.py --results-dir DIR

QUESTION 1 — which objective builds the best hybrid?
    For every (dataset, pool, model), each objective's hybrid is reported on all
    three numbers, not only the one it selected by. An objective that wins only
    on its own number has proved nothing, and the table shows that directly.

QUESTION 2 — is there one best budget?
    The best budget per (dataset, pool, objective, model), with the margin over
    the runner-up and the fold-to-fold spread beside it. A margin inside the
    spread is a tie. If the best budget is the same everywhere, budget can leave
    the search space; if it moves, it stays.

Nothing here averages across datasets. Every count is a count of cells.

MODEL_MAX_BUDGET is EMPTY since 2026-09-10, so every model runs at every
budget asked for and a missing (model, budget) cell is a cell that FAILED, not
one that was never run. Files written before that date have their caps in
'budget_caps' and are read the old way.
"""
import os
import sys
import json
import glob
import argparse
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(HERE, '..', 'experiments', 'results')

SIDE_LABEL = {
    'side_predictive': 'predict',
    'side_readacross': 'neighbour',
    'struct_fidelity': 'MCS',
}
SIDE_ORDER = ['side_predictive', 'side_readacross', 'struct_fidelity']


def load_files(results_dir):
    pattern = os.path.join(results_dir, 'f27_objective_and_budget_*.json')
    out = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith('.partial'):
            continue
        try:
            with open(path) as f:
                out.append((path, json.load(f)))
        except Exception as exc:
            print(f'  could not read {os.path.basename(path)}: {exc!r}')
    return out


def _fmt(v, width=9):
    return f'{v:+{width}.4f}' if isinstance(v, (int, float)) else f'{"n/a":>{width}s}'


def objective_table(files):
    """Question 1, per dataset x pool x model."""
    print('\n' + '=' * 100)
    print('QUESTION 1 — which objective builds the best hybrid?')
    print('  Every number is delta_mean_of_folds: the hybrid minus the best')
    print('  single representation, paired per outer fold. + means the hybrid')
    print('  is better. The three columns are the SAME hybrid read three ways.')
    print('=' * 100)

    # How often each objective wins each of the three numbers.
    wins = {k: Counter() for k in SIDE_ORDER}
    cells = Counter()

    for path, out in files:
        ds = out.get('dataset')
        for pool, per_obj in (out.get('results') or {}).items():
            models = sorted({m for c in per_obj.values()
                             for m in (c.get('by_model') or {})})
            for model in models:
                print(f'\n  {ds} | pool={pool} | model={model}')
                header = f'    {"objective":14s}'
                for k in SIDE_ORDER:
                    header += f' {SIDE_LABEL[k]:>10s}'
                print(header + '    picked-by')
                rows = {}
                for name in out.get('objectives', []):
                    c = per_obj.get(name)
                    e = (c or {}).get('by_model', {}).get(model)
                    if not e:
                        continue
                    line = f'    {name:14s}'
                    vals = {}
                    for k in SIDE_ORDER:
                        v = (e.get(k) or {}).get('delta_mean_of_folds')
                        vals[k] = v
                        line += f' {_fmt(v, 10)}'
                    own = e.get('delta_mean_of_folds')
                    line += f'    {_fmt(own)}'
                    print(line)
                    rows[name] = vals
                if not rows:
                    continue
                cells[(ds, pool, model)] += 1
                for k in SIDE_ORDER:
                    scored = {n: v[k] for n, v in rows.items()
                              if v[k] is not None}
                    if scored:
                        wins[k][max(scored, key=scored.get)] += 1

    print('\n' + '-' * 100)
    print(f'WIN COUNTS over {sum(cells.values())} (dataset, pool, model) cells.')
    print('  A win is the highest delta on that number. Counts, never means.')
    for k in SIDE_ORDER:
        total = sum(wins[k].values())
        parts = ', '.join(f'{n} {c}' for n, c in wins[k].most_common())
        print(f'  {SIDE_LABEL[k]:>10s} ({total} cells): {parts or "none"}')
    print('  An objective that tops only its own column bought nothing on the'
          ' other two.')


def budget_table(files):
    """Question 2, per dataset x pool x objective x model."""
    print('\n' + '=' * 100)
    print('QUESTION 2 — is there one best budget?')
    print('  ~ marks a best budget whose margin over the runner-up is inside')
    print('  the fold-to-fold spread. That is a tie, not a winner.')
    print('=' * 100)

    chosen = defaultdict(Counter)     # objective -> Counter(best budget)
    ties = Counter()
    decided = Counter()

    for path, out in files:
        ds = out.get('dataset')
        for pool, per_obj in (out.get('results') or {}).items():
            for name, c in per_obj.items():
                bb = c.get('best_budget_per_model') or {}
                if not bb:
                    continue
                print(f'\n  {ds} | pool={pool} | objective={name}')
                print(f'    {"model":10s} {"best":>6s} {"delta":>9s} '
                      f'{"margin":>9s} {"fold sd":>9s}  verdict')
                for model, e in sorted(bb.items()):
                    exceeds = e.get('margin_exceeds_fold_std')
                    mark = ('tie' if exceeds is False else
                            ('clear' if exceeds is True else 'unknown'))
                    if exceeds is True:
                        decided[name] += 1
                        chosen[name][e['best_budget']] += 1
                    elif exceeds is False:
                        ties[name] += 1
                    print(f'    {model:10s} {e["best_budget"]:>6d} '
                          f'{_fmt(e.get("best_delta"))} '
                          f'{_fmt(e.get("margin_over_runner_up"))} '
                          f'{_fmt(e.get("fold_std_at_best"))}  {mark}')

    print('\n' + '-' * 100)
    print('BEST BUDGET, counted only where the margin cleared the fold spread:')
    if not decided:
        print('  No cell cleared it. On this evidence budget is not the lever,')
        print('  and the cheapest budget that ties is the one to take.')
    for name in sorted(decided):
        parts = ', '.join(f'{b}: {c}' for b, c in
                          sorted(chosen[name].items()))
        print(f'  {name:14s} {decided[name]:>3d} clear, {ties[name]:>3d} tied'
              f'   -> {parts}')
        if len(chosen[name]) > 1:
            print(f'  {"":14s} more than one budget wins, so budget stays a '
                  f'searched axis for this objective.')
        elif len(chosen[name]) == 1:
            print(f'  {"":14s} one budget wins every clear cell for this '
                  f'objective.')


def coverage(files):
    print('\n' + '=' * 100)
    print('COVERAGE — what ran, what did not')
    print('=' * 100)
    for path, out in files:
        prog = out.get('progress') or {}
        n_skip = len(out.get('skipped') or [])
        print(f'\n  {os.path.basename(path)}')
        print(f'    dataset {out.get("dataset")}  n={out.get("n_samples")}  '
              f'pools {out.get("pools")}  '
              f'cells {prog.get("cells_attempted")}/'
              f'{prog.get("cells_planned")}  '
              f'complete {prog.get("complete")}  '
              f'elapsed {out.get("elapsed_seconds")}s')
        for pool, m in (out.get('mcs') or {}).items():
            pf = m.get('per_fold') or {}
            print(f'    MCS {pool}: {m.get("n_pairs")} pairs, '
                  f'min per outer fold {pf.get("min")} '
                  f'(needs {pf.get("threshold")}), usable {pf.get("usable")}')
        caps = out.get('budget_caps') or {}
        for model, c in sorted(caps.items()):
            print(f'    {model} never ran budgets {c["budgets_never_run"]} '
                  f'(MODEL_MAX_BUDGET {c["max_budget"]})')
        if n_skip:
            print(f'    SKIPPED {n_skip}:')
            for s in out['skipped']:
                print(f'      {s.get("pool")}/{s.get("objective")}: '
                      f'{str(s.get("error"))[:150]}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', dest='results_dir',
                    default=DEFAULT_RESULTS)
    args = ap.parse_args()

    files = load_files(args.results_dir)
    if not files:
        print(f'No f27 result files in {args.results_dir}')
        return 1
    print(f'{len(files)} f27 result file(s) in {args.results_dir}')
    objective_table(files)
    budget_table(files)
    coverage(files)
    return 0


if __name__ == '__main__':
    sys.exit(main())
