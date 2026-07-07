"""Generate two preliminary plots for the supervisor meeting.

Saves to tests/analysis/figures/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'tests' / 'analysis' / 'figures'
OUT.mkdir(exist_ok=True)

DATASETS = [
    'abagym_Ang2', 'abagym_EGFR', 'abagym_HER2', 'abagym_VEGF', 'abcov_kd',
    'flab_aggregation_jain', 'flab_immunogenicity_marks', 'flab_pk_jain_heprt',
    'flab_polyreact_jain_bvp', 'flab_polyreact_jain_cic',
    'ginkgo_heparin', 'humanness_therapeutic',
]
REGRESSION = {
    'abagym_Ang2', 'abagym_EGFR', 'abagym_HER2', 'abagym_VEGF',
    'abcov_kd', 'flab_aggregation_jain', 'flab_pk_jain_heprt', 'ginkgo_heparin',
}


def load_c02():
    rows = []
    for ds in DATASETS:
        p = REPO / 'tests' / 'results' / 'complexity' / \
            'c02_complexity_matching_grid' / f'{ds}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        # best_single_score is None (JSON null) for a model whose reps all
        # failed — c02 no longer writes ±inf for that case. Keep only finite
        # numeric baselines; mirrors the finite() guard the other c02 readers
        # (quick_hybrid_summary / firstpass_notready) already use.
        finite_baselines = [
            m['best_single_score']
            for m in d.get('baselines', {}).values()
            if isinstance(m, dict)
            and isinstance(m.get('best_single_score'), (int, float))
            and np.isfinite(m['best_single_score'])
        ]
        baseline = max(finite_baselines) if finite_baselines else None
        cond_best = {}
        for cond, models in (d.get('conditions') or {}).items():
            scores = [m.get('score_mean') for m in models.values()
                      if isinstance(m, dict) and m.get('score_mean') is not None]
            if scores:
                cond_best[cond] = max(scores)
        hybrid = max(cond_best.values()) if cond_best else None
        pca = None
        for dim, models in (d.get('pca_baselines') or {}).items():
            mx = max(models.values()) if isinstance(models, dict) else None
            if mx is not None and (pca is None or mx > pca):
                pca = mx
        if baseline is None or hybrid is None:
            # No finite single-rep baseline (every model's reps failed) or no
            # hybrid condition — nothing to compare on this dataset, so it can't
            # go in the baseline-vs-hybrid plot. Skip it (keeps delta a real
            # float for every kept row, so the downstream sort/format/median
            # can't hit a None).
            continue
        rows.append({'ds': ds, 'baseline': baseline, 'hybrid': hybrid,
                     'pca': pca, 'delta': hybrid - baseline})
    return rows


def load_s02():
    rows = []
    for ds in DATASETS:
        p = REPO / 'tests' / 'results' / 'similarity' / \
            's02_hybrid_similarity' / f'{ds}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        bi = (d.get('best_individual_rep') or {}).get('sac_rho', 0) or 0
        bw = (d.get('best_individual_rep_weighted') or {}).get('sac_rho', 0) or 0
        concat = (d.get('concat') or {}).get('sac_rho', 0) or 0
        rows.append({'ds': ds, 'best_individual': bi, 'weighted': bw,
                     'concat': concat, 'delta': bw - bi})
    return rows


def plot_c02(rows):
    rows = sorted(rows, key=lambda r: r['delta'], reverse=True)
    n = len(rows)
    x = np.arange(n)
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, [r['pca'] for r in rows], w,
           label='PCA→concat (best dim)', color='#cccccc')
    ax.bar(x, [r['baseline'] for r in rows], w,
           label='Best single rep', color='#7faecf')
    ax.bar(x + w, [r['hybrid'] for r in rows], w,
           label='KIRBy hybrid (best condition)', color='#d97a4a')
    for i, r in enumerate(rows):
        sign = '+' if r['delta'] >= 0 else ''
        ax.text(i + w, r['hybrid'] + 0.01, f"{sign}{r['delta']:.2f}",
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['ds'].replace('_', '\n') for r in rows],
                       rotation=0, fontsize=8)
    ax.set_ylabel('Primary metric (R² / ρ / AUROC)')
    ax.set_title('c02 — KIRBy hybrid vs best single rep vs PCA-concat '
                 '(12 antibody datasets)')
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)
    fig.tight_layout()
    fp = OUT / 'c02_hybrid_vs_baselines.png'
    fig.savefig(fp, dpi=160)
    print(f'Saved {fp}')
    plt.close(fig)


def plot_s02(rows):
    rows = sorted(rows, key=lambda r: r['delta'], reverse=True)
    n = len(rows)
    x = np.arange(n)
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, [r['concat'] for r in rows], w,
           label='Concat similarity', color='#cccccc')
    ax.bar(x, [r['best_individual'] for r in rows], w,
           label='Best individual rep', color='#7faecf')
    ax.bar(x + w, [r['weighted'] for r in rows], w,
           label='KIRBy importance-weighted hybrid', color='#d97a4a')
    for i, r in enumerate(rows):
        ax.text(i + w, r['weighted'] + 0.01, f"+{r['delta']:.2f}",
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['ds'].replace('_', '\n') for r in rows],
                       rotation=0, fontsize=8)
    ax.set_ylabel('Sequence-Activity-Cliff Spearman ρ')
    ax.set_title('s02 — KIRBy importance-weighted similarity vs '
                 'best single rep vs concat (12 antibody datasets)')
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)
    fig.tight_layout()
    fp = OUT / 's02_weighted_vs_baselines.png'
    fig.savefig(fp, dpi=160)
    print(f'Saved {fp}')
    plt.close(fig)


if __name__ == '__main__':
    c02 = load_c02()
    s02 = load_s02()
    plot_c02(c02)
    plot_s02(s02)
    # Also print the median deltas for the brief
    print('c02 wins:', sum(1 for r in c02 if r['delta'] > 0), '/', len(c02))
    print('c02 median Δ:', round(float(np.median([r['delta'] for r in c02])), 3))
    print('s02 wins:', sum(1 for r in s02 if r['delta'] > 0), '/', len(s02))
    print('s02 median Δ:', round(float(np.median([r['delta'] for r in s02])), 3))
