"""Tables for PRCHO_REAL_CENSORING.md, read from the n30b result files on seagate.
Mean over the 5 outer folds; the difference from no treatment is per fold, then its
mean and its standard deviation over folds."""
import json, numpy as np
D = '/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n30b_real_censoring/'
ARMS = [('as_delivered', 'No treatment'), ('clean', 'Unclipped labels (QM9 only)'),
        ('constant_lift', 'Constant lift'), ('per_molecule_lift', 'Per-molecule lift'),
        ('per_molecule_lift_shuffled/seed0', 'Per-molecule lift, shuffled (seed 0)'),
        ('truncated_normal_mean', 'Truncated-normal mean'), ('censored_em', 'Censored EM'),
        ('drop_clipped', 'Delete labels at the limit'),
        ('not_at_floor_control/seed0', 'Same move, labels NOT at the limit (seed 0)'),
        ('not_at_floor_control/seed1', 'Same move, labels NOT at the limit (seed 1)'),
        ('not_at_floor_control/seed2', 'Same move, labels NOT at the limit (seed 2)'),
        ('winsor_c1', 'Winsorise, c = 1'), ('winsor_c1.25', 'Winsorise, c = 1.25'), ('winsor_c1.5', 'Winsorise, c = 1.5')]

def table(f, score, model, spaces=('base_1', 'hybrid')):
    d = json.load(open(D + f))
    folds = sorted(d['folds'], key=int)
    head = '| Treatment | ' + ' | '.join(f'{s} mean | {s} difference (fold sd)' for s in spaces) + ' |'
    out = [head, '|' + '---|' * (1 + 2 * len(spaces))]
    for arm, name in ARMS:
        cells = []
        for sp in spaces:
            key = f'{sp}/{model}'
            v = [d['folds'][k]['scores'][key]['per_arm'].get(arm, {}).get(score) for k in folds]
            if any(x is None for x in v):
                cells = None; break
            b = [d['folds'][k]['scores'][key]['per_arm']['as_delivered'][score] for k in folds]
            diff = np.array(v) - np.array(b)
            cells += [f'{np.mean(v):.4f}', '' if arm == 'as_delivered' else f'{np.mean(diff):+.4f} ({np.std(diff, ddof=1):.4f})']
        if cells:
            out.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)

if __name__ == '__main__':
    for f, score, models in [('ginkgo_pr_cho.json', 'spearman_all', ['gp', 'dnn']),
                             ('ginkgo_pr_cho.json', 'spearman_above_floor', ['gp', 'dnn']),
                             ('ginkgo_pr_cho.json', 'harrell_c', ['gp', 'dnn']),
                             ('qm9__injected0.25.json', 'rmse', ['dnn', 'lightgbm'])]:
        for m in models:
            print(f'\n#### {f.split(".json")[0]}, {m}, {score}\n')
            print(table(f, score, m))
