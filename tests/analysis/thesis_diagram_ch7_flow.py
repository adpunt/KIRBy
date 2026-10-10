"""Chapter 7 Methods diagram: how one noise method is run and scored.

Boxes and arrows only, no data. Words follow thesis.tex Chapter 7 Methods
(sec:noise:methods to sec:noise:tests) and THESIS_OUTLINE.md "NAMES FOR THE
NOISE METHODS". Writes figures_ch5_7/d7_noise_flow.png at the thesis text
width (5.91 in). Run from the repo root:

    python tests/analysis/thesis_diagram_ch7_flow.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thesis_figure_style as S  # noqa: E402  (sets rcParams)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(REPO, 'figures_ch5_7', 'd7_noise_flow.png')

FS = 7.5
GREY = '#F2F2F2'
DATA = '#DCEBF7'      # labels, features, score
DETECT = '#FCE5CD'    # detector
FLAG = '#E2F0D9'      # known flag
EVERY = '#EDE3F3'     # every molecule
ROBUST = '#FFF2CC'    # noise-robust training
EDGE = '#404040'


def box(ax, x, y, w, h, text, color, nhead=1, dashed=False, fs=FS):
    """Rounded box centred at (x, y). The first nhead lines are bold."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.4,rounding_size=1.2',
                       fc=color, ec=EDGE, lw=0.8,
                       ls='--' if dashed else '-')
    ax.add_patch(p)
    lines = text.split('\n')
    step = 2.7
    top = y + (len(lines) - 1) * step / 2
    for i, line in enumerate(lines):
        ax.text(x, top - i * step, line, ha='center', va='center',
                fontsize=fs, fontweight='bold' if i < nhead else 'normal')
    return (x, y, w, h)


def arrow(ax, p0, p1, dashed=False, rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=9,
                        lw=0.9, color=EDGE, ls='--' if dashed else '-',
                        connectionstyle=f'arc3,rad={rad}',
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(S.TEXTWIDTH_IN, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Row 1: damaged labels
    box(ax, 50, 95, 54, 6,
        'Damaged training labels\none outer fold; held-out labels untouched',
        DATA)
    # Row 2: features
    box(ax, 50, 80, 80, 9,
        'Features, chosen per fold and per downstream model\n'
        'two best single representations and two best hybrids;\n'
        "the hybrid's features re-ranked on the damaged labels",
        DATA)
    arrow(ax, (50, 91.6), (50, 85.3))

    # Frame around noise mitigation: selection row + action row
    frame = FancyBboxPatch((0.5, 31.5), 73.5, 38.5,
                           boxstyle='round,pad=0.3,rounding_size=1.5',
                           fc='none', ec='#9A9A9A', lw=0.7, ls=':')
    ax.add_patch(frame)
    ax.text(2.0, 33.5, 'Noise mitigation', ha='left', va='center',
            fontsize=FS, style='italic', color='#555555')

    # Row 3: what selects the molecules (three) + noise-robust training
    xs = [12.5, 37.0, 61.5, 87.5]
    y3 = 59
    w3 = 20
    box(ax, xs[0], y3, w3, 10,
        'Detector\nsuspicion score\nper molecule', DETECT)
    box(ax, xs[1], y3, w3, 10,
        'Known flag\nclipped label,\nscaffold family', FLAG)
    box(ax, xs[2], y3, w3, 10,
        'Every molecule\nno selection', EVERY)
    box(ax, xs[3], y3, w3, 10,
        'Noise-robust\ntraining\nselects nothing', ROBUST, nhead=2)
    for x in xs:
        arrow(ax, (x, 74.7), (x, 64.6))

    # Row 4: action (mitigation) vs change the model (robust)
    y4 = 42
    box(ax, 37.0, y4, 69, 7,
        'Action on the selected molecules\n'
        'delete, count less, or move the label',
        GREY)
    box(ax, xs[3], y4, w3, 12,
        'Change\nthe model\nloss, kernel or\ntraining loop', ROBUST,
        nhead=2)
    for x in xs[:3]:
        arrow(ax, (x, 53.4), (x, 46.2))
    arrow(ax, (xs[3], 53.4), (xs[3], 48.6))

    # Row 5: held-out score
    y5 = 22
    box(ax, 50, y5, 80, 7,
        'Downstream model fitted, then held-out score\n'
        'against the same model and features with no change',
        DATA)
    arrow(ax, (37.0, 37.8), (37.0, 26.2))
    arrow(ax, (xs[3], 35.4), (xs[3], 26.2))

    # Controls
    box(ax, 50, 6, 80, 9,
        'Controls\nsame action on molecules chosen at random;\n'
        'shuffled suspicion scores; QM9 with no noise added',
        'white', dashed=True)
    arrow(ax, (50, 11.2), (50, 17.8), dashed=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    print(OUT)


if __name__ == '__main__':
    main()
