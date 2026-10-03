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
    import glob
    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(ds_dir, 'rows*.jsonl'))):
        for l in open(path):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            k = (r['model'], r['rep'], r['kind'], r['level'], r['draw'], r['fold'])
            if k not in seen:
                seen.add(k)
                rows.append(r)
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
    summ = []
    clean_vec = {}
    for r in rows:
        if r['kind'] == 'clean':
            clean_vec.setdefault((r['model'], r['rep']), {})[r['fold']] = r['preds']
    clean_full = {k: np.concatenate([np.asarray(v[f]) for f in sorted(v)])
                  for k, v in clean_vec.items()
                  if len(v) == len(folds['val_idx'])}
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
        summ.append(f'== level position {li} ==')
        summ.append('kind | corr with clean-label predictions (mean over pairs) | '
                    'same-kind draw0-vs-1 | kinds correlating at least as high as the '
                    'lower of the two kinds\' same-kind values (censoring has no draw, '
                    'so its own value is skipped)')
        yard = {a: (np.nan if a == 'censoring' else M[i, i])
                for i, a in enumerate(kinds)}
        for i, a in enumerate(kinds):
            cc = []
            for m, rp in pairs:
                x = full.get((m, rp, a, li, 0))
                c = clean_full.get((m, rp))
                if x is not None and c is not None:
                    cc.append(corr(x, c))
            same = []
            for j, b in enumerate(kinds):
                if j == i or np.isnan(M[i, j]):
                    continue
                thr = np.nanmin([yard[a], yard[b]])
                if M[i, j] >= thr:
                    same.append(b)
            summ.append(f'{a:<22} clean {np.mean(cc) if cc else float("nan"):.3f} | '
                        f'own {M[i, i]:.3f} | '
                        f'{len(same)}: ' + ', '.join(same))
        summ.append('')
        report += [f'## Level position {li}', '',
                   'Same kind, draw 0 vs 1 (yardstick): ' + ', '.join(
                       f'{k} {M[i, i]:.3f}' for i, k in enumerate(kinds)
                       if not np.isnan(M[i, i])), '']
    open(os.path.join(ds_dir, 'report.md'), 'w').write(
        '\n'.join(report) + '\n\n## Drop table\n\n```\n'
        + open(os.path.join(ds_dir, 'drop_table.csv')).read() + '```\n')

    # ---- change in predictions: damaged minus clean, per molecule --------------
    # The clean predictions are in every kind's predictions, which is why the plain
    # correlations above are all 0.9+. Subtracting them leaves only what the damage
    # did, and two kinds that do the same thing to a model move the same molecules
    # the same way.
    dsumm = []
    for li in range(3):
        D = np.full((len(kinds), len(kinds)), np.nan)
        for i, a in enumerate(kinds):
            for j, b in enumerate(kinds):
                cs = []
                for m, rp in pairs:
                    c = clean_full.get((m, rp))
                    xa = full.get((m, rp, a, li, 0))
                    xb = (full.get((m, rp, a, li, 1)) if i == j
                          else full.get((m, rp, b, li, 0)))
                    if c is None or xa is None or xb is None:
                        continue
                    cs.append(corr(xa - c, xb - c))
                if cs:
                    D[i, j] = np.mean(cs)
        with open(os.path.join(ds_dir, f'delta_corr_L{li}.csv'), 'w') as fh:
            fh.write('kind,' + ','.join(kinds) + '\n')
            for i, a in enumerate(kinds):
                fh.write(a + ',' + ','.join('' if np.isnan(v) else f'{v:.4f}'
                                            for v in D[i]) + '\n')
        dsumm.append(f'== change in predictions, level position {li}: kind | '
                     'same kind, draw 0 vs 1 | five most similar other kinds ==')
        for i, a in enumerate(kinds):
            others = sorted((j for j in range(len(kinds))
                             if j != i and not np.isnan(D[i, j])),
                            key=lambda j: -D[i, j])[:5]
            dsumm.append(f'{a:<22} own {D[i, i]:.3f} | ' + ', '.join(
                f'{kinds[j]} {D[i, j]:.3f}' for j in others))
        dsumm.append('')
    open(os.path.join(ds_dir, 'summary_delta.txt'), 'w').write('\n'.join(dsumm))

    # ---- auc_norm per kind and pair (the noise suite's own robustness number) ---
    # Curve = clean score at level 0 plus the three levels, mean over folds and
    # draws. Four points, the minimum robustness_metrics accepts. Levels are in
    # each kind's own unit, so compare across pairs for one kind, not across kinds.
    try:
        from kirby.noise_robustness import robustness_metrics
        al = ['kind,' + ','.join(f'{m}_{rp}' for m, rp in pairs)]
        sc = {}
        for r in rows:
            sc.setdefault((r['model'], r['rep'], r['kind'], r['level']), []).append(
                r['score'])
        for k in kinds:
            vals = []
            for m, rp in pairs:
                base = sc.get((m, rp, 'clean', 0.0))
                if not base:
                    vals.append('')
                    continue
                x = [0.0] + list(levels_of[k])
                y = [np.mean(base)] + [np.mean(sc[(m, rp, k, l)])
                                       for l in levels_of[k]]
                v = robustness_metrics(x, y, baseline_threshold=None,
                                       lower_is_better=lower)['auc_norm']
                vals.append(f'{v:.4f}')
            al.append(k + ',' + ','.join(vals))
        open(os.path.join(ds_dir, 'auc_norm.csv'), 'w').write('\n'.join(al) + '\n')
    except Exception as exc:
        print('auc_norm skipped:', exc)

    # ---- draw-to-draw wobble: |rmse of draw 0 - rmse of draw 1|, mean over folds ----
    sd = {}
    for r in rows:
        if r['kind'] == 'clean':
            continue
        sd.setdefault((r['kind'], r['model'], r['rep'], r['level'], r['fold']),
                      {})[r['draw']] = r['score']
    wl = ['kind,' + ','.join(f'{m}_{rp}_L{li}' for m, rp in pairs for li in range(3))]
    for k in kinds:
        vals = []
        for m, rp in pairs:
            for li, lv in enumerate(levels_of[k]):
                d = [abs(v[0] - v[1]) for (kk, mm, rr, ll, ff), v in sd.items()
                     if (kk, mm, rr, ll) == (k, m, rp, lv) and 0 in v and 1 in v]
                vals.append(f'{np.mean(d):.4f}' if d else '')
        wl.append(k + ',' + ','.join(vals))
    open(os.path.join(ds_dir, 'draw_spread.csv'), 'w').write('\n'.join(wl) + '\n')
    open(os.path.join(ds_dir, 'summary.txt'), 'w').write(
        '\n'.join(summ) + '\n== drop table (mean over folds and draws) ==\n'
        + open(os.path.join(ds_dir, 'drop_table.csv')).read())
    print('wrote', ds_dir)


if __name__ == '__main__':
    for d in sys.argv[1:]:
        main(d)
