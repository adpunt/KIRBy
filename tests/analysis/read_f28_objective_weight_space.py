#!/usr/bin/env python3
"""Read every f28 result file and say what the weight space did.

    python tests/analysis/read_f28_objective_weight_space.py
    python tests/analysis/read_f28_objective_weight_space.py --results-dir DIR

THE QUESTION f28 ASKS
    The search maximises one number. Three are available — a fitted model's
    score, each molecule's property predicted from its five nearest neighbours,
    and MCS structural fidelity — and f28 runs all three on their own plus every
    mixture of them in steps of 0.25.

WHICH NUMBER ANSWERS THE QUESTION
    Two numbers are recorded for every row and they answer different things.

    LEVEL -- the hybrid's own score on one of the four numbers. Every row is
    scored on the SAME outer folds, because fold membership comes from labels,
    groups, seed and fold count and never from the objective. So a level reads
    DOWN the table: hybrid against hybrid, same rows, same quantity. This is
    what says whether a mixture is worth having.

    DELTA -- that row minus its own best single representation. It reads ALONG
    a row only. Each row picks its baseline by its own objective, so the number
    being subtracted is a different representation from row to row. Ranking
    rows by delta compares two subtractions with different baselines, which is
    what this reader used to do.

WHAT THIS PRINTS
    1. Per (dataset, model): every row's LEVEL on all four numbers, with the
       spread over outer folds, then the same rows' DELTAS separately.
    2. Win counts per number, over (dataset, model) cells, on LEVELS. Counts,
       never means, and never averaged across datasets.
    3. Does mixing ever beat both of its ends? For each pair of objectives, how
       often a mixture beat BOTH pure objectives on the same number. That is the
       only result that would justify blending at all.
    4. Coverage: what ran, what was skipped, whether per-fold scores are in the
       file, and where the blend identity check failed.

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
         'struct_fid': 'MCS', 'predictive_corr': 'predict_corr'}
PURE = {'predict', 'predict_corr', 'neighbour', 'MCS'}


def _end_row(per_cell, name):
    """Which row is the pure END of an edge, for section 3.

    For prediction that is 'predict_corr' whenever it ran. Every mixture scores
    prediction as a correlation, so the corner measured the same way is the end
    of the edge; 'predict' maximises the dataset's own metric and is a
    different quantity. 'predict' is the fallback, for a file written before
    the 16th row existed.
    """
    if name == 'predict' and (per_cell.get('predict_corr') or {}).get('by_model'):
        return 'predict_corr'
    return name


def _canonical_grid():
    """The 15 cell names in the order the experiment defines them.

    Imported rather than copied, so the report cannot drift from the grid.
    Falls back to whatever order the files arrive in if the import fails.
    """
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'experiments', 'framework'))
        from f28_objective_weight_space import build_grid
        return [n for n, _ in build_grid()]
    except Exception:
        return []


def load_files(results_dir):
    """One entry per DATASET, merging the files that dataset was split across.

    A run that gives every cell its own array task writes one file per
    (dataset, cell) with a --tag in the name. A run that does all 15 cells in
    one task writes one file. Both land here as one entry per dataset.
    """
    order = _canonical_grid()
    merged, first_path = {}, {}
    for path in sorted(glob.glob(
            os.path.join(results_dir, 'f28_objective_weight_space_*.json'))):
        if path.endswith('.partial'):
            continue
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception as exc:
            print(f'  could not read {os.path.basename(path)}: {exc!r}')
            continue
        ds = blob.get('dataset')
        # A smoke file is 3 cheap representations and must not be merged into
        # the real run for the same dataset.
        if blob.get('smoke'):
            ds = f'{ds} (smoke)'
            blob['dataset'] = ds
        if ds not in merged:
            merged[ds] = blob
            first_path[ds] = path
            continue
        into = merged[ds]
        into.setdefault('results', {}).update(blob.get('results') or {})
        into.setdefault('skipped', []).extend(blob.get('skipped') or [])
        into['grid'] = list(dict.fromkeys(
            (into.get('grid') or []) + (blob.get('grid') or [])))
        into.setdefault('weights', {}).update(blob.get('weights') or {})

    for ds, blob in merged.items():
        have = set(blob.get('grid') or [])
        if order:
            blob['grid'] = [n for n in order if n in have]
        blob['cells_present'] = len(blob.get('results') or {})
        blob['cells_expected'] = len(order) if order else None
    return [(first_path[ds], blob) for ds, blob in sorted(merged.items())]


def _d(entry, key):
    """That row minus ITS OWN best single representation. Along a row only."""
    return (entry.get(key) or {}).get('delta_mean_of_folds')


def _level(entry, key):
    """The hybrid's own mean on one number. This is the one that reads down."""
    return (entry.get(key) or {}).get('mean')


def _spread(entry, key):
    """Standard deviation over the outer folds, or None if not in the file."""
    fs = [v for v in ((entry.get(key) or {}).get('fold_scores') or [])
          if v is not None]
    if len(fs) < 2:
        return None
    m = sum(fs) / len(fs)
    return (sum((v - m) ** 2 for v in fs) / len(fs)) ** 0.5


def _direction(out):
    """(lower_is_better for 'predict', its name, whether the rows disagree).

    'predict' carries the dataset's own metric, so on rmse a SMALLER number is
    the better one. The other three are correlations and are always
    higher-is-better. Every row records the metric and its direction; if two
    rows disagree that column cannot be read down, and it is said so.
    """
    per_cell = out.get('results') or {}
    dirs = {bool(c.get('side_metric_lower_is_better'))
            for c in per_cell.values()}
    names = {c.get('side_metric_name') for c in per_cell.values()
             if c.get('side_metric_name')}
    lower = bool(next(iter(dirs))) if len(dirs) == 1 else False
    name = (next(iter(names)) if len(names) == 1
            else '/'.join(sorted(n for n in names)))
    return lower, (name or '?'), (len(dirs) > 1 or len(names) > 1)


def _up(v, key, side_lower):
    """v turned so that BIGGER is always better, whatever the metric."""
    if v is None:
        return None
    return -v if (key == 'side_predictive' and side_lower) else v


def _fmt(v, w=10):
    return f'{v:+{w}.4f}' if isinstance(v, (int, float)) else f'{"n/a":>{w}s}'


def _fmt_level(entry, key, w=17):
    v = _level(entry, key)
    if not isinstance(v, (int, float)):
        return f'{"n/a":>{w}s}'
    sd = _spread(entry, key)
    return (f'{v:.4f}+/-{sd:.3f}' if sd is not None else f'{v:.4f}').rjust(w)


def per_dataset_tables(files):
    print('\n' + '=' * 112)
    print('1. EVERY ROW, READ FOUR WAYS')
    print('=' * 112)
    for _path, out in files:
        ds = out.get('dataset')
        per_cell = out.get('results') or {}
        side_lower, side_name, mixed = _direction(out)
        models = sorted({m for c in per_cell.values()
                         for m in (c.get('by_model') or {})})
        for model in models:
            rows = [(cell, ((per_cell.get(cell) or {})
                            .get('by_model', {}).get(model)))
                    for cell in out.get('grid', [])]
            rows = [(cell, e) for cell, e in rows if e]
            if not rows:
                continue
            print(f'\n  {ds} | model={model} | at most '
                  f'{out.get("max_reps")} representations')
            if mixed:
                print('    WARNING: rows disagree on what "predict" is. '
                      'That column cannot be read down.')

            print('    LEVELS -- the hybrid\'s own score. Same outer folds '
                  'every row, so a column reads DOWN.')
            print(f'    {"row":31s}' + ''.join(f'{short:>17s}'
                                               for _k, short, _ in COLUMNS)
                  + '   reps')
            for cell, e in rows:
                line = f'    {cell:31s}'
                for key, _short, _label in COLUMNS:
                    line += _fmt_level(e, key)
                reps = [r for r in (e.get('reps_per_fold') or []) if r]
                line += f'   {min(reps)}-{max(reps)}' if reps else '   n/a'
                print(line)
            print(f'    predict is {side_name}, '
                  f'{"lower" if side_lower else "higher"} is better; '
                  'the other three are correlations, higher is better.')
            if all(_spread(e, 'side_predictive') is None for _c, e in rows):
                print('    no +/- : this file predates per-fold scores being '
                      'stored, so there is no spread behind any number.')

            print('\n    DELTAS -- each row minus ITS OWN best single '
                  'representation. Read ALONG a row, never down.')
            print(f'    {"row":31s}' + ''.join(f'{short:>11s}'
                                               for _k, short, _ in COLUMNS)
                  + '      folds')
            for cell, e in rows:
                line = f'    {cell:31s}'
                for key, _short, _label in COLUMNS:
                    line += _fmt(_d(e, key), 11)
                line += f'{int(e.get("n_paired_folds") or 0):>11d}'
                print(line)


def win_counts(files):
    print('\n' + '=' * 112)
    print('2. WIN COUNTS PER NUMBER, over (dataset, model) cells')
    print('   A win is the best LEVEL in that column -- hybrid against hybrid,')
    print('   on the same outer folds. Counts, never means.')
    print('=' * 112)
    wins = {k: Counter() for k, _short, _label in COLUMNS}
    n_cells = 0
    for _path, out in files:
        per_cell = out.get('results') or {}
        side_lower, _name, _mixed = _direction(out)
        models = sorted({m for c in per_cell.values()
                         for m in (c.get('by_model') or {})})
        for model in models:
            n_cells += 1
            for key, _short, _label in COLUMNS:
                scored = {}
                for cell in out.get('grid', []):
                    e = (per_cell.get(cell) or {}).get('by_model', {}).get(model)
                    v = None if not e else _up(_level(e, key), key, side_lower)
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
    print('\n' + '=' * 112)
    print('3. DOES MIXING EVER BEAT BOTH OF ITS ENDS?')
    print('   For each pair, how often a mixture of the two beat BOTH pure')
    print('   objectives on the same number, by LEVEL. Anything else is not a')
    print('   reason to blend.')
    print('=' * 112)
    pairs = [('predict', 'neighbour'), ('predict', 'MCS'),
             ('neighbour', 'MCS')]
    tally = defaultdict(lambda: {'beat_both': 0, 'compared': 0,
                                 'where': Counter()})
    ends = defaultdict(set)
    for _path, out in files:
        ds = out.get('dataset')
        per_cell = out.get('results') or {}
        weights = out.get('weights') or {}
        side_lower, _name, _mixed = _direction(out)
        models = sorted({m for c in per_cell.values()
                         for m in (c.get('by_model') or {})})
        for model in models:
            for a, b in pairs:
                # mixtures of exactly these two
                mix = [cell for cell, w in weights.items()
                       if set(PLAIN.get(k, k) for k in w) == {a, b}]
                row_a, row_b = _end_row(per_cell, a), _end_row(per_cell, b)
                ends[(a, b)].add(f'{row_a} / {row_b}')
                for key, _short, _label in COLUMNS:
                    ea = (per_cell.get(row_a) or {}).get('by_model', {}).get(model)
                    eb = (per_cell.get(row_b) or {}).get('by_model', {}).get(model)
                    va = None if not ea else _up(_level(ea, key), key, side_lower)
                    vb = None if not eb else _up(_level(eb, key), key, side_lower)
                    if va is None or vb is None:
                        continue
                    best_end = max(va, vb)
                    for cell in mix:
                        e = (per_cell.get(cell) or {}).get(
                            'by_model', {}).get(model)
                        v = None if not e else _up(_level(e, key), key,
                                                   side_lower)
                        if v is None:
                            continue
                        t = tally[(a, b, key)]
                        t['compared'] += 1
                        if v > best_end:
                            t['beat_both'] += 1
                            t['where'][f'{ds}/{model}/{cell}'] += 1
    for a, b in pairs:
        used = ', '.join(sorted(ends.get((a, b)) or [])) or f'{a} / {b}'
        print(f'\n  {a} vs {b}   (ends read as: {used})')
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
    print('\n' + '=' * 112)
    print('4. COVERAGE')
    print('=' * 112)
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
        _rows = [e for c in (out.get('results') or {}).values()
                 for e in (c.get('by_model') or {}).values()]
        _withfolds = sum(1 for e in _rows
                         if (e.get('side_predictive') or {}).get('fold_scores'))
        print(f'    per-fold scores present in {_withfolds} of {len(_rows)} '
              f'(row, model) entries')
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
