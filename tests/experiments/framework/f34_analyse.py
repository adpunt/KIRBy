#!/usr/bin/env python3
"""f34 replay — read the stored grids and answer the three questions.

Nothing is fitted here. f34_neighbour_weighting_cross.py stored every cell's
per-fold scores; this file reads them and prints the tables.

THE THREE QUESTIONS, per dataset, never averaged across datasets:

  1. Which cell wins, and by how much over second place, in that dataset's own
     metric. The raw pair is printed, not a ratio.
  2. Is the winner the same under every representation?
  3. Do the two settings interact — is the best coefficient the same under
     every weighting scheme? If it is, the two could have been swept one at a
     time and crossing them cost nothing. If it is not, crossing them was the
     only way to get the answer.

And one more, which decides how much of this has to become a swept axis rather
than a default: does the same cell win on all three datasets?

Usage:
    python tests/experiments/framework/f34_analyse.py
    python tests/experiments/framework/f34_analyse.py --datasets caco2
    python tests/experiments/framework/f34_analyse.py --top 10
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = (_ROOT / 'tests' / 'results' / 'framework'
               / 'f34_neighbour_weighting_cross')


def load(results_dir, datasets=None):
    out = {}
    for path in sorted(Path(results_dir).glob('*.json')):
        payload = json.loads(path.read_text())
        # save_experiment_results may nest the payload under 'results'.
        body = payload.get('results', payload)
        ds = body.get('dataset') or path.stem
        if datasets and ds not in datasets:
            continue
        body.setdefault('metric_name', payload.get('metric_name'))
        body.setdefault('lower_is_better', payload.get('lower_is_better'))
        out[ds] = body
    return out


def _sign(lower_is_better):
    return 1 if lower_is_better else -1


def all_cells(body):
    """Every cell of every representation, as (representation, key, entry)."""
    for space, cells in (body.get('cells_by_representation') or {}).items():
        for key, entry in cells.items():
            yield space, key, entry


def print_winners(ds, body, top):
    metric = body.get('metric_name') or 'score'
    lower = bool(body.get('lower_is_better'))
    sign = _sign(lower)
    n_folds = body.get('n_folds')
    direction = 'lower is better' if lower else 'higher is better'

    print(f'\n{"=" * 78}\n{ds}: {body.get("n_molecules")} molecules, '
          f'{n_folds} folds, metric {metric} ({direction})\n{"=" * 78}')

    print(f'\n  Winner per representation. One row is one representation. '
          f'The score is the mean over {n_folds} folds, and the spread is the '
          f'largest fold minus the smallest.')
    print(f'    {"representation":26s} {"weighting":15s} {"coefficient":16s} '
          f'{"k":>3s} {metric:>10s} {"spread":>9s} {"2nd":>10s} {"gap":>8s}')
    winners = (body.get('summary') or {}).get('winner_per_representation') or {}
    for space, w in winners.items():
        second = w.get(f'runner_up_{metric}')
        gap = w.get('gap_to_runner_up')
        print(f'    {space:26s} {w["weighting_scheme"]:15s} '
              f'{w["coefficient"]:16s} {w["n_neighbours"]:>3d} '
              f'{w[f"winner_{metric}"]:>10.4f} '
              f'{w["winner_spread_over_folds"]:>9.4f} '
              + (f'{second:>10.4f}' if second is not None else f'{"n/a":>10s}')
              + (f' {gap:>8.4f}' if gap is not None else f' {"n/a":>8s}'))

    same = (body.get('summary') or {}).get(
        'same_winner_across_representations') or {}
    print(f'    {same.get("note", "")}')

    print(f'\n  The {top} best cells over every representation. One row is one '
          f'(weighting scheme, coefficient, neighbour count) on one '
          f'representation.')
    rows = sorted(all_cells(body), key=lambda r: r[2]['mean_of_folds'] * sign)
    print(f'    {"representation":26s} {"weighting":15s} {"coefficient":16s} '
          f'{"k":>3s} {metric:>10s} {"spread":>9s}')
    for space, _key, e in rows[:top]:
        print(f'    {space:26s} {e["weighting_scheme"]:15s} '
              f'{e["coefficient"]:16s} {e["n_neighbours"]:>3d} '
              f'{e["mean_of_folds"]:>10.4f} {e["spread_of_folds"]:>9.4f}')

    inter = (body.get('summary') or {}).get('axes_interact') or {}
    print(f'\n  Do the two settings interact? {inter.get("note", "")}')
    print(f'    One row is one weighting scheme. Each number is that scheme\'s '
          f'mean {metric} under one coefficient, averaged over the '
          f'{len(body.get("neighbour_counts") or [])} neighbour counts and '
          f'every representation.')
    coeffs = body.get('coefficients') or []
    print(f'    {"weighting":15s} ' + ' '.join(f'{c:>16s}' for c in coeffs)
          + '   best')
    for scheme, block in (inter.get(
            'best_coefficient_per_weighting_scheme') or {}).items():
        means = block[f'mean_{metric}_per_coefficient']
        print(f'    {scheme:15s} '
              + ' '.join(f'{means.get(c, float("nan")):>16.4f}' for c in coeffs)
              + f'   {block["best_coefficient"]}')

    print(f'\n  The other way round. One row is one coefficient; the number is '
          f'that coefficient\'s mean {metric} under one weighting scheme.')
    schemes = body.get('weighting_schemes') or []
    print(f'    {"coefficient":16s} ' + ' '.join(f'{s:>15s}' for s in schemes)
          + '   best')
    for coefficient, block in ((body.get('summary') or {}).get(
            'best_weighting_scheme_per_coefficient') or {}).items():
        means = block[f'mean_{metric}_per_scheme']
        print(f'    {coefficient:16s} '
              + ' '.join(f'{means.get(s, float("nan")):>15.4f}' for s in schemes)
              + f'   {block["best_weighting_scheme"]}')

    per_k = (body.get('summary') or {}).get(
        'best_cell_per_neighbour_count') or {}
    if per_k:
        print(f'\n  Per neighbour count, not collapsed. One row is one '
              f'neighbour count k; the cell named is the best over every '
              f'representation at that k.')
        print(f'    {"k":>3s} {"best cell":40s} {metric:>10s}')
        for k, block in sorted(per_k.items(), key=lambda kv: int(kv[0])):
            print(f'    {int(k):>3d} {block["best_cell"]:40s} '
                  f'{block[f"best_{metric}"]:>10.4f}')

    n_skipped = body.get('n_skipped_cells') or 0
    total = (len(body.get('representations_swept') or [])
             * (body.get('n_cells_per_representation') or 0)
             * (n_folds or 0))
    print(f'\n  {n_skipped} of {total} (cell, fold) fits were skipped.')
    if n_skipped:
        reasons = {}
        for s in body.get('skipped_cells') or []:
            reasons[s['reason']] = reasons.get(s['reason'], 0) + s['n_cells']
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f'    {n}: {reason}')


def print_across_datasets(bodies):
    """Does the same cell win on all three datasets?

    Each dataset is scored in its own metric, so the scores are never averaged
    or ranked against each other. Only the NAME of the winning cell is
    compared.
    """
    print(f'\n{"=" * 78}\nAcross datasets\n{"=" * 78}')
    print('\n  One row is one dataset. The score is in that dataset\'s own '
          'metric, so the column is not comparable down the page — only the '
          'cell names are.')
    print(f'    {"dataset":14s} {"metric":10s} {"weighting":15s} '
          f'{"coefficient":16s} {"k":>3s} {"score":>10s} {"gap to 2nd":>11s}')
    best_cells = {}
    for ds, body in bodies.items():
        metric = body.get('metric_name') or 'score'
        lower = bool(body.get('lower_is_better'))
        rows = sorted(all_cells(body),
                      key=lambda r: r[2]['mean_of_folds'] * _sign(lower))
        if not rows:
            print(f'    {ds:14s} no cells')
            continue
        _space, key, e = rows[0]
        gap = (abs(e['mean_of_folds'] - rows[1][2]['mean_of_folds'])
               if len(rows) > 1 else float('nan'))
        best_cells[ds] = key
        print(f'    {ds:14s} {metric:10s} {e["weighting_scheme"]:15s} '
              f'{e["coefficient"]:16s} {e["n_neighbours"]:>3d} '
              f'{e["mean_of_folds"]:>10.4f} {gap:>11.4f}')

    distinct = set(best_cells.values())
    print()
    if len(distinct) == 1 and len(best_cells) > 1:
        print(f'  One cell wins on all {len(best_cells)} datasets: '
              f'{next(iter(distinct))}. That is a default.')
    else:
        print(f'  {len(distinct)} different cells win across '
              f'{len(best_cells)} datasets. No single default; the setting '
              f'stays a swept axis.')

    # The same question for each setting on its own, which is what decides
    # whether ONE of the two can be fixed even when the pair cannot.
    for axis, field in (('weighting scheme', 'weighting_scheme'),
                        ('coefficient', 'coefficient'),
                        ('neighbour count', 'n_neighbours')):
        picked = {}
        for ds, body in bodies.items():
            lower = bool(body.get('lower_is_better'))
            rows = sorted(all_cells(body),
                          key=lambda r: r[2]['mean_of_folds'] * _sign(lower))
            if rows:
                picked[ds] = rows[0][2][field]
        vals = set(picked.values())
        verdict = ('the same on every dataset' if len(vals) == 1
                   else f'{len(vals)} different values')
        print(f'  best {axis}: {verdict} — '
              + ', '.join(f'{d}={v}' for d, v in picked.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-dir', default=str(RESULTS_DIR))
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--top', type=int, default=8,
                    help='how many of the best cells to list per dataset')
    args = ap.parse_args()

    bodies = load(args.results_dir, args.datasets)
    if not bodies:
        print(f'no f34 result files under {args.results_dir}')
        return 2
    for ds, body in bodies.items():
        print_winners(ds, body, args.top)
    if len(bodies) > 1:
        print_across_datasets(bodies)
    return 0


if __name__ == '__main__':
    sys.exit(main())
