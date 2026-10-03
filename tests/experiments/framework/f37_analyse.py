#!/usr/bin/env python3
"""f37 analysis: score drop per kind of damage, and prediction correlation between kinds.

Reads <output>/f37_label_damage_map/<dataset>/{rows,folds}.jsonl|json and writes,
next to them:
  drop_table.csv       one row per kind, one column per (model on rep, level position);
                       the value is the mean over folds and draws of (damaged score - clean
                       score) in the dataset's own metric; positive = worse for rmse
  pred_corr_L<i>.csv   level position i (0 low, 1 middle, 2 high): Pearson correlation of
                       held-out predictions between two kinds, mean over the pairs; the
                       diagonal is the same kind under draw 0 vs draw 1 (the yardstick)
  pred_corr_L<i>.png   the same as a heatmap, one row = one kind of damage
  report.md            the above as tables

A held-out prediction vector is the concatenation of the three folds' held-out
predictions of one (pair, kind, level, draw), so it covers every molecule once.
Levels differ in unit between kinds, so "level position" means the low, middle or high
of that kind's own three levels. No significance tests, no intervals.
"""
import json
import os
import sys

import numpy as np


def load(ds_dir):
    rows = [json.loads(l) for l in open(os.path.join(ds_dir, 'rows.jsonl'))]
    folds = json.load(open(os.path.join(ds_dir, 'folds.json')))
    return [r for r in rows if r['preds'] is not None], folds


def main(ds_dir):
    rows, folds = load(ds_dir)
    n = sum(len(v) for v in folds['val_idx'])
    pos = np.concatenate([np.asarray(v) for v in folds['val_idx']])
    metric = rows[0]['metric']
    lower = rows[0]['lower_is_better']
    pairs = sorted({(r['model'], r['rep']) for r in rows})
    kinds = []
    levels_of = {}
    for r in rows:
        if r['kind'] == 'clean':
            continue
        if r['kind'] not in levels_of:
            kinds.append(r['kind'])
            levels_of[r['kind']] = set()
        levels_of[r['kind']].add(r['level'])
    levels_of = {k: sorted(v) for k, v in levels_of.items()}

    clean = {}
    for r in rows:
        if r['kind'] == 'clean':
            clean[(r['model'], r['rep'], r['fold'])] = r['score']

    # score drop: mean over folds and draws
    cell = {}
    for r in rows:
        if r['kind'] == 'clean':
            continue
        li = levels_of[r['kind']].index(r['level'])
        c = clean.get((r['model'], r['rep'], r['fold']))
        if c is None:
            continue
        cell.setdefault((r['kind'], r['model'], r['rep'], li), []).append(
            (r['score'] - c) if lower else (c - r['score']))
    cols = [(m, rp, li) for m, rp in pairs for li in range(3)]
    lines = ['kind,' + ','.join(f'{m}_{rp}_L{li}' for m, rp, li in cols)]
    for k in kinds:
        vals = []
        for m, rp, li in cols:
            v = cell.get((k, m, rp, li))
            vals.append('' if not v else f'{np.mean(v):.4f}')
        lines.append(k + ',' + ','.join(vals))
    open(os.path.join(ds_dir, 'drop_table.csv'), 'w').write('\n'.join(lines) + '\n')

    # predictions: concatenated over folds, one vector per (pair, kind, level pos, draw)
    vec = {}
    for r in rows:
        if r['kind'] == 'clean':
            continue
        li = levels_of[r['kind']].index(r['level'])
        vec.setdefault((r['model'], r['rep'], r['kind'], li, r['draw']), {})[
            r['fold']] = r['preds']
    full = {k: np.concatenate([np.asarray(v[f]) for f in sorted(v)])
            for k, v in vec.items() if len(v) == len(folds['val_idx'])}

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    report = [f'# f37 {os.path.basename(ds_dir)}', '',
              f'Metric {metric} ({"lower" if lower else "higher"} is better). '
              'Drop table: mean over folds and draws of damaged score minus clean score, '
              'sign flipped when higher is better, so positive = worse.', '',
              'Correlations: Pearson between held-out predictions of two kinds on the '
              f'same {n} molecules, mean over the pairs. Diagonal = same kind, draw 0 vs 1.', '']
    for li in range(3):
        M = np.full((len(kinds), len(kinds)), np.nan)
        for i, a in enumerate(kinds):
            for j, b in enumerate(kinds):
                cs = []
                for m, rp in pairs:
                    if i == j:
                        x, y = full.get((m, rp, a, li, 0)), full.get((m, rp, a, li, 1))
                    else:
                        x, y = full.get((m, rp, a, li, 0)), full.get((m, rp, b, li, 0))
                    if x is not None and y is not None:
                        cs.append(corr(x, y))
                if cs:
                    M[i, j] = np.mean(cs)
        with open(os.path.join(ds_dir, f'pred_corr_L{li}.csv'), 'w') as fh:
            fh.write('kind,' + ','.join(kinds) + '\n')
            for i, a in enumerate(kinds):
                fh.write(a + ',' + ','.join('' if np.isnan(v) else f'{v:.4f}'
                                            for v in M[i]) + '\n')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(M, vmin=np.nanmin(M), vmax=1.0, cmap='viridis')
            ax.set_xticks(range(len(kinds)), kinds, rotation=90, fontsize=7)
            ax.set_yticks(range(len(kinds)), kinds, fontsize=7)
            fig.colorbar(im, label='prediction correlation')
            ax.set_title(f'{os.path.basename(ds_dir)}, level position {li}')
            fig.tight_layout()
            fig.savefig(os.path.join(ds_dir, f'pred_corr_L{li}.png'), dpi=120)
            plt.close(fig)
        except Exception as exc:
            print('heatmap skipped:', exc)
        report += [f'## Level position {li}', '',
                   'Same kind, draw 0 vs 1 (yardstick): ' + ', '.join(
                       f'{k} {M[i, i]:.3f}' for i, k in enumerate(kinds)
                       if not np.isnan(M[i, i])), '']
    open(os.path.join(ds_dir, 'report.md'), 'w').write(
        '\n'.join(report) + '\n\n## Drop table\n\n```\n'
        + open(os.path.join(ds_dir, 'drop_table.csv')).read() + '```\n')
    print('wrote', ds_dir)


if __name__ == '__main__':
    for d in sys.argv[1:]:
        main(d)
