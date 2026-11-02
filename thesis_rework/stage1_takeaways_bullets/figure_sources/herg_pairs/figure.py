"""Two-panel hERG Ki figure: |delta pKi| against ECFP4 Tanimoto similarity (left) and
against Mordred Manhattan distance on standardised features (right), all 996,166 pairs
of the 1,412 molecules, binned; median and interquartile band per bin."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = '/Users/apunt/repos/KIRBy/thesis_rework/stage1_takeaways_bullets/figures_code_input'
os.makedirs(OUT, exist_ok=True)
dy = np.load(f'{HERE}/herg_ki_pair_dy.npy')
t = 1.0 - np.load(f'{HERE}/herg_ki_pairdist_ecfp4.npy')
m = np.load(f'{HERE}/herg_ki_pairdist_mordred.npy')
sim_mask = t >= 0.55

res = {'n_pairs': int(len(dy))}


def binned(x, edges):
    idx = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        v = dy[idx == b]
        if v.size < 20:
            continue
        rows.append(((edges[b] + edges[b + 1]) / 2, *np.percentile(v, [25, 50, 75]), v.size))
    return np.array(rows)


e1 = np.linspace(0, 1, 21)
B1 = binned(t, e1)
# right: Mordred distance as a percentile of all pairs (0 = closest), 20 bins of 5%
rank_pct = 100.0 * (np.argsort(np.argsort(m, kind='stable')) + 1) / len(m)
e2 = np.linspace(0, 100, 21); e2[-1] += 1e-6
B2 = binned(rank_pct, e2)
res['left_bins'] = B1.tolist(); res['right_bins'] = B2.tolist()
res['right_bins_note'] = 'x = percentile of Mordred distance among all pairs; columns x, p25, median, p75, n_pairs'
res['n_pairs_tanimoto_ge_055'] = int(sim_mask.sum())
res['mordred_rank_pct_of_tanimoto_ge_055_pairs'] = {
    'median': float(np.median(rank_pct[sim_mask])),
    'p90': float(np.percentile(rank_pct[sim_mask], 90)),
    'share_outside_closest_10pct': float(np.mean(rank_pct[sim_mask] > 10))}
top = t >= 0.8
res['n_pairs_tanimoto_ge_08'] = int(top.sum())
res['dy_tanimoto_ge_08'] = {'median': float(np.median(dy[top])), 'p75': float(np.percentile(dy[top], 75)),
                            'share_gt_1pKi': float(np.mean(dy[top] > 1.0))}
res['dy_all_pairs_median'] = float(np.median(dy))
res['share_pairs_tanimoto_below_020'] = float(np.mean(t < 0.20))
res['share_pairs_tanimoto_ge_055'] = float(np.mean(sim_mask))

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for a, B, xl, c in [(ax[0], B1, 'ECFP4 Tanimoto similarity', '#1f5f99'),
                    (ax[1], B2, 'Mordred distance, percentile of all pairs\n(0 = closest pairs)', '#b5541c')]:
    a.fill_between(B[:, 0], B[:, 1], B[:, 3], color=c, alpha=0.25, lw=0, label='25th to 75th percentile')
    a.plot(B[:, 0], B[:, 2], color=c, lw=2, marker='o', ms=3, label='median')
    a.set_xlabel(xl)
    a.grid(alpha=0.3, lw=0.5)
    a.spines[['top', 'right']].set_visible(False)
ax[0].set_ylabel(r'$|\Delta$pK$_i|$ between the two molecules')
ax[0].axvline(0.55, color='0.4', ls='--', lw=1)
ax[0].text(0.56, ax[0].get_ylim()[1] * 0.92, '0.55', color='0.3', fontsize=8)
ax[0].legend(frameon=False, fontsize=8, loc='upper left')
ax[0].set_title('(a) ECFP4', fontsize=10, loc='left')
ax[1].set_title('(b) Mordred', fontsize=10, loc='left')
fig.tight_layout()
fig.savefig(f'{OUT}/herg_ki_pair_distance_vs_dpki.pdf')
fig.savefig(f'{OUT}/herg_ki_pair_distance_vs_dpki.png', dpi=200)
json.dump(res, open(f'{HERE}/figure_herg_ki.json', 'w'), indent=1)
print({k: v for k, v in res.items() if 'bins' not in k})
