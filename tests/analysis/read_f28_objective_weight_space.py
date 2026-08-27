#!/usr/bin/env python3
"""Read every f28 result file and say what the weight space did.

    python tests/analysis/read_f28_objective_weight_space.py
    python tests/analysis/read_f28_objective_weight_space.py --results-dir DIR

THE QUESTION f28 ASKS
    The search maximises one number. Three are available — a fitted model's
    score, each molecule's property predicted from its five nearest neighbours,
    and MCS structural fidelity — and f28 runs all three on their own plus every
    mixture of them in steps of 0.25.

WHAT THIS PRINTS
    1. Per (dataset, model): every cell read on all four recorded numbers.
       'own' is the number that cell was maximising and is NOT comparable
       across rows; the four columns after it are comparable down a column.
    2. Win counts per column, over (dataset, model) cells. Counts, never means,
       and never averaged across datasets.
    3. Does mixing ever beat both of its ends? For each pair of objectives, how
       often a mixture beat BOTH pure objectives on the same column. That is the
       only result that would justify blending at all.
    4. Coverage: what ran, what was skipped, and where the blend identity check
       failed.

MODEL_MAX_BUDGET caps svm at 500 and gp/mlp/bnn/dnn/cnn at 1000, so those
(model, budget) cells never ran. Reported as absent, never as a loss.
"""
import os
import sys
import json
import glob
import argparse
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(HERE, '..', 'experiments', 'results')

# (key in the result file, short column header, what it is in words)
COLUMNS = [
    ('side_predictive', 'predict', "the dataset's own metric"),
    ('side_predictive_corr', 'corr', 'prediction as a correlation'),
    ('side_readacross', 'neighbour', 'five nearest neighbours'),
    ('struct_fidelity', 'MCS', 'MCS fidelity'),
]
PLAIN = {'predictive': 'predict', 'similarity': 'neighbour',
         'struct_fid': 'MCS'}
PURE = {'predict', 'neighbour', 'MCS'}


def load_files(results_dir):
    out = []
    for path in sorted(glob.glob(
            os.path.join(results_dir, 'f28_objective_weight_space_*.json'))):
        if path.endswith('.partial'):
            continue
        try:
            with open(path) as f:
                out.append((path, json.load(f)))
        except Exception as exc:
            print(f'  could not read {os.path.basename(path)}: {exc!r}')
    return out


def _d(entry, key):
    return (entry.get(key) or {}).get('delta_mean_of_folds')


def _fmt(v, w=10):
    return f'{v:+{w}.4f}' if isinstance(v, (int, float)) else f'{"n/a":>{w}s}'


def per_dataset_tables(files):
    print('\n' + '=' * 108)
    print('1. EVERY CELL, READ FOUR WAYS')
    print("   Each number is the hybrid minus the best single representation,")
    print('   paired per outer fold. + means the hybrid won.')
    print('=' * 108)
    for _path, out in files:
        ds = out.get('dataset')
        per_cell = out.get('results') or {}
        models = sorted({m for c in per_cell.values()
                         for m in (c.get('by_model') or {})})
        for model in models:
            print(f'\n  {ds} | model={model} | at most '
                  f'{out.get("max_reps")} representations')
            head = f'    {"cell":32s}' + ''.join(
                f'{short:>11s}' for _k, short, _ in COLUMNS)
            print(head + '   reps')
            for cell in out.get('grid', []):
                e = (per_cell.get(cell) or {}).get('by_model', {}).get(model)
                if not e:
                    continue
                row = f'    {cell:32s}'
                for key, _short, _label in COLUMNS:
                    row += _fmt(_d(e, key), 11)
                reps = [r for r in (e.get('reps_per_fold') or []) if r]
                row += f'   {min(reps)}-{max(reps)}' if reps else '   n/a'
                print(row)


def win_counts(files):
    print('\n' + '=' * 108)
    print('2. WIN COUNTS PER COLUMN, over (dataset, model) cells')
    print('   A win is the highest delta in that column. Counts, never means.')
    print('=' * 108)
    wins = {k: Counter() for k, _short, _label in COLUMNS}
    n_cells = 0
    for _path, out in files:
        per_cell = out.get('results') or {}
        models = sorted({m for c in per_cell.values()
                         for m in (c.get('by_model') or {})})
        for model in models:
            n_cells += 1
            for key, _short, _label in COLUMNS:
                scored = {}
                for cell in out.get('grid', []):
                    e = (per_cell.get(cell) or {}).get('by_model', {}).get(model)
                    v = None if not e else _d(e, key)
                    if v is not None:
                        scored[cell] = v
                if scored:
                    wins[key][max(scored, key=scored.get)] += 1
    print(f'\n  {n_cells} (dataset, model) cells\n')
    for key, _short, label in COLUMNS:
        total = sum(wins[key].values())
        print(f'  {label} ({total} cells):')
        for cell, c in wins[key].most_common(6):
            mark = '  [a pure objective]' if cell in PURE else ''
            print(f'    {c:>3d}  {cell}{mark}')
        if not total:
            print('    nothing scored')
    print('\n  A pure objective topping every column means mixing bought '
          'nothing.')


def mixing_beats_both_ends(files):
    print('\n' + '=' * 108)
    print('3. DOES MIXING EVER BEAT BOTH OF ITS ENDS?')
    print('   For each pair, how often a mixture of the two beat BOTH pure')
    print('   objectives on the same column. Anything else is not a reason to')
    print('   blend.')
    print('=' * 108)
    pairs = [('predict', 'neighbour'), ('predict', 'MCS'),
             ('neighbour', 'MCS')]
    tally = defaultdict(lambda: {'beat_both': 0, 'compared': 0,
                                 'where': Counter()})
    for _path, out in files:
        ds = out.get('dataset')
        per_cell = out.get('results') or {}
        weights = out.get('weights') or {}
        models = sorted({m for c in per_cell.values()
                         for m in (c.get('by_model') or {})})
        for model in models:
            for a, b in pairs:
                # mixtures of exactly these two
                mix = [cell for cell, w in weights.items()
                       if set(PLAIN[k] for k in w) == {a, b}]
                for key, _short, _label in COLUMNS:
                    ea = (per_cell.get(a) or {}).get('by_model', {}).get(model)
                    eb = (per_cell.get(b) or {}).get('by_model', {}).get(model)
                    va = None if not ea else _d(ea, key)
                    vb = None if not eb else _d(eb, key)
                    if va is None or vb is None:
                        continue
                    best_end = max(va, vb)
                    for cell in mix:
                        e = (per_cell.get(cell) or {}).get(
                            'by_model', {}).get(model)
                        v = None if not e else _d(e, key)
                        if v is None:
                            continue
                        t = tally[(a, b, key)]
                        t['compared'] += 1
                        if v > best_end:
                            t['beat_both'] += 1
                            t['where'][f'{ds}/{model}/{cell}'] += 1
    for a, b in pairs:
        print(f'\n  {a} vs {b}')
        for key, _short, label in COLUMNS:
            t = tally.get((a, b, key))
            if not t or not t['compared']:
                print(f'    {label:34s} nothing comparable')
                continue
            print(f'    {label:34s} {t["beat_both"]:>3d} of '
                  f'{t["compared"]:>3d} mixtures beat both ends')
            for where, _c in t['where'].most_common(4):
                print(f'      {where}')


def coverage(files):
    print('\n' + '=' * 108)
    print('4. COVERAGE')
    print('=' * 108)
    for path, out in files:
        prog = out.get('progress') or {}
        mcs = out.get('mcs') or {}
        pf = mcs.get('per_fold') or {}
        print(f'\n  {os.path.basename(path)}')
        print(f'    {out.get("dataset")}  n={out.get("n_samples")}  '
              f'cells {prog.get("cells_attempted")}/'
              f'{prog.get("cells_planned")}  '
              f'complete {prog.get("complete")}  '
              f'elapsed {out.get("elapsed_seconds")}s')
        print(f'    at most {out.get("max_reps")} representations via '
              f'{out.get("allocation_methods")}, '
              f'{out.get("n_outer_folds_requested")} outer folds')
        print(f'    MCS {mcs.get("n_pairs")} pairs, min per outer fold '
              f'{pf.get("min")} (needs {pf.get("threshold")}), '
              f'usable {pf.get("usable")}')
        for model, c in sorted((out.get('budget_caps') or {}).items()):
            print(f'    {model} never ran budgets {c["budgets_never_run"]} '
                  f'(MODEL_MAX_BUDGET {c["max_budget"]})')
        bad = []
        for cell, c in (out.get('results') or {}).items():
            for model, e in (c.get('by_model') or {}).items():
                gap = (e.get('blend_identity') or {}).get('gap')
                if gap is not None and gap > 1e-9:
                    bad.append((cell, model, gap))
        if bad:
            print("    a blend's own delta did NOT equal the weighted mean "
                  'of its parts:')
            for cell, model, gap in bad:
                print(f'      {cell} / {model}: gap {gap:.2e}')
        skipped = out.get('skipped') or []
        if skipped:
            print(f'    SKIPPED {len(skipped)}:')
            for s in skipped:
                print(f'      {s.get("cell")}: {str(s.get("error"))[:150]}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', dest='results_dir',
                    default=DEFAULT_RESULTS)
    args = ap.parse_args()

    files = load_files(args.results_dir)
    if not files:
        print(f'No f28 result files in {args.results_dir}')
        return 1
    print(f'{len(files)} f28 result file(s) in {args.results_dir}')
    per_dataset_tables(files)
    win_counts(files)
    mixing_beats_both_ends(files)
    coverage(files)
    return 0


if __name__ == '__main__':
    sys.exit(main())
