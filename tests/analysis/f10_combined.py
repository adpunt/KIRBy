"""F10 as one figure: one row per dataset (hERG Ki, Caco-2, LogD), one column per damage kind,
redrawn from the per-dataset CSVs that thesis_figures.py --figure f10 writes. 2026-10-10: replaces
the three per-dataset 2x2 images, whose empty fourth slot left the panels in a triangle."""
import csv, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
sys.path.insert(0, os.path.dirname(__file__))
import thesis_figure_style as S

D = 'figures_ch5_7'
ROWS = [('herg_ki', '(a) hERG $K_i$'), ('caco2', '(b) Caco-2'), ('logd', '(c) LogD')]
COL_OF = {'outlier_p05': 0, 'shift_only_p10': 1, 'censoring': 2, 'sparse_dprime_up_p10': 2}
WORDS = {'outlier_p05': 'Outlier', 'shift_only_p10': 'Shift in 10%', 'censoring': 'Clipped',
         'sparse_dprime_up_p10': 'Shift in 10%, model-sized'}
COLORS = {0.05: '#1b9e77', 0.1: '#d95f02', 0.2: '#7570b3'}
MARKERS = {0.05: 'o', 0.1: 's', 0.2: '^'}

fig, axes = plt.subplots(3, 3, figsize=(S.TEXTWIDTH_IN, 6.6), squeeze=False)
used = set()
xmax = 0.1
data = {}
for ds, _ in ROWS:
    rows = list(csv.DictReader(open(f'{D}/f10_detection_vs_cleaning_{ds}.csv')))
    data[ds] = rows
    for r in rows:
        for k in ('precision', 'break_even_precision'):
            if r[k]:
                xmax = max(xmax, float(r[k]))
for i, (ds, label) in enumerate(ROWS):
    kinds = {}
    for r in data[ds]:
        kinds.setdefault(r['damage_kind'], []).append(r)
    for kind, ms in kinds.items():
        j = COL_OF[kind]
        ax = axes[i][j]
        used.add((i, j))
        for m in sorted(ms, key=lambda m: float(m['flag_share'])):
            s = float(m['flag_share'])
            ax.scatter(float(m['precision']), float(m['after_value']), color=COLORS[s], marker=MARKERS[s],
                       s=26, lw=0, zorder=3)
            if m['break_even_precision']:
                ax.axvline(float(m['break_even_precision']), color=COLORS[s], lw=1.0, alpha=0.8, zorder=1)
        ax.axhline(float(ms[0]['no_cleaning_value']), color='#555555', ls='--', lw=0.8, zorder=2)
        ax.axhline(float(ms[0]['clean_value']), color=S.CLEAN_COLOR, ls=':', lw=0.9, zorder=2)
        ax.set_xlim(0, min(1.0, xmax * 1.1 + 0.02))
        ax.set_title(WORDS[kind], fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if i == 2 or (i + 1, j) not in {(a, b) for a in range(3) for b in range(3)}:
            pass
for i in range(3):
    for j in range(3):
        if (i, j) not in used:
            axes[i][j].set_visible(False)
    first = min(j for j in range(3) if (i, j) in used)
    axes[i][first].set_ylabel(f'{ROWS[i][1]}\nRMSE after cleaning', fontsize=8)
for j in range(3):
    axes[2][j].set_xlabel('Precision of the\ndeleted molecules', fontsize=8)
h = [Line2D([], [], ls='', marker=MARKERS[s], color=COLORS[s], ms=5, label=f'{s * 100:g}% flagged') for s in (0.05, 0.1, 0.2)]
h += [Line2D([], [], color='#777777', label='Break-even precision'),
      Line2D([], [], ls='--', color='#555555', label='Damaged, no cleaning'),
      Line2D([], [], ls=':', color=S.CLEAN_COLOR, label='Clean labels')]
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=7, frameon=False)
fig.savefig(f'{D}/f10_detection_vs_cleaning_combined.png', dpi=300)
print('written')
