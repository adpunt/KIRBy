"""Figures for Chapters 5, 6 and 7 of the thesis, one per slot of THESIS_OUTLINE.md.

    python tests/analysis/thesis_figures.py --list
    python tests/analysis/thesis_figures.py --dry-run
    python tests/analysis/thesis_figures.py --figure all
    python tests/analysis/thesis_figures.py --figure f8 --results-root <run folder>

Each figure writes <name>.png and <name>.csv into --out. The CSV holds exactly
the numbers plotted, one row per mark. Marks whose numbers are missing are left
out and named on stderr. Read-only on the results; fits no model; one thread.

Slots and where their numbers come from (experiment codes appear only here and
in --help, never on a figure):
  f1   hybrid against best single representation      c02 standard_selection_summary
  f2a  best number of features                         c09 bias_variance curves (or c16)
  f2b  best number of base representations             c18 recipe_per_fold rep_count
  f3   blueprints, features per representation         c22 allocation_built
  f5   distance against activity difference            s02 comparisons (knn distance hybrid)
  f6   prediction from nearest neighbours               s05 summary.fixed_ref
  f8   did the cleaning work                            n07 n18 n20 n23 n24 n25 n27 n30 n34 n38
  f9   the censoring ladder                             n30
  f10  detection against repair                         n32
  f11  score against damage level                       n07

Written by Claude, 2026-10-09. The loaders are thesis_loader_*.py beside this file.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import thesis_figure_style as S  # noqa: E402  (sets rcParams and seaborn style)
import thesis_loader_f1 as L1  # noqa: E402
import thesis_loader_f2 as L2  # noqa: E402
import thesis_loader_f3 as L3  # noqa: E402
import thesis_loader_f56 as L56  # noqa: E402
import thesis_loader_f8 as L8  # noqa: E402
import thesis_loader_f9_11 as L911  # noqa: E402

LAPTOP_ROOT = Path('/Volumes/seagate/kirby_results_2026-10-08')


def default_results_root():
    cur = REPO / 'tests' / 'results' / 'CURRENT_RUN'
    if cur.exists() and cur.read_text().strip():
        return Path(cur.read_text().strip())
    return LAPTOP_ROOT


# --------------------------------------------------------------------------- shared

class Missing:
    """Collects one message per absent mark and prints it on stderr."""

    def __init__(self, fig, quiet=False):
        self.fig, self.quiet, self.msgs = fig, quiet, []

    def __call__(self, msg):
        self.msgs.append(msg)
        if not self.quiet:
            print(f'[{self.fig}] missing: {msg}', file=sys.stderr)


def write_csv(path, rows, columns):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: _cell(r.get(c)) for c in columns})


def _cell(v):
    if isinstance(v, float):
        return '' if math.isnan(v) else repr(v)
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v)
    return '' if v is None else v


def save(fig, out, name, rows, columns):
    png = out / f'{name}.png'
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    write_csv(out / f'{name}.csv', rows, columns)
    print(f'wrote {png} and {name}.csv ({len(rows)} marks)')


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def dataset_order(datasets):
    """Domain first (small molecules, antibodies, reactions, patient data), then label."""
    def key(d):
        dom = S.DATASET_DOMAIN.get(d)
        return (S.DOMAIN_ORDER.index(dom) if dom in S.DOMAIN_ORDER else 99, S.dataset_label(d))
    return sorted(set(datasets), key=key)


def metric_order(metrics):
    pref = ['rmse', 'mae', 'spearman', 'pearson', 'auroc', 'auc', 'mcc', 'f1_weighted']
    return sorted(set(metrics), key=lambda m: (pref.index(m) if m in pref else 99, m))


def one_fold(rows, fold, fig):
    """Keep the rows of one outer fold. The Chapter 7 loaders return one row per fold; until it is
    decided how a figure shows several folds, a figure draws one and says what else exists."""
    other = sorted({r.get('fold') for r in rows if r.get('fold') != fold}, key=str)
    if other:
        print(f'[{fig}] drawing outer fold {fold}; the results also hold folds {other} '
              f'(--noise-fold picks another)', file=sys.stderr)
    return [r for r in rows if r.get('fold') == fold]


def norm_metric(m):
    return 'auroc' if m in ('auc', 'roc_auc') else m


def diagonal(ax, xs, ys, log=False):
    lo, hi = min(xs + ys), max(xs + ys)
    pad = (hi - lo) * 0.05 or 0.05
    lo, hi = (lo / 1.1, hi * 1.1) if log else (lo - pad, hi + pad)
    ax.plot([lo, hi], [lo, hi], color='#999999', lw=0.8, ls='--', zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')


def model_legend(fig, models, ncol=None, y=1.0):
    models = S.sort_models_by_family(models)
    h = [Line2D([], [], ls='', marker=S.model_marker(m), color=S.model_color(m), ms=6,
                label=S.model_label(m)) for m in models]
    ncol = ncol or len(h)
    fig.legend(S._to_colmajor(h, ncol), S._to_colmajor([x.get_label() for x in h], ncol),
               ncol=ncol, loc='lower center', bbox_to_anchor=(0.5, y))


def domain_legend(fig, domains, y=1.0):
    doms = [d for d in S.DOMAIN_ORDER if d in domains]
    h = [Line2D([], [], ls='', marker='o', color=S.DOMAIN_COLORS[d], ms=6, label=d) for d in doms]
    fig.legend(h, [d for d in doms], ncol=len(h), loc='lower center', bbox_to_anchor=(0.5, y))


def panel_grid(n, ncol=2, h_per_row=3.0):
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.TEXTWIDTH_IN, h_per_row * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


# --------------------------------------------------------------------------- F1

def fig_f1(root, out, args, missing):
    """F1: one dot per dataset and model; x best single, y hybrid; one panel per metric."""
    rows = L1.load(root, missing)
    marks = []
    for r in rows:
        if not (_finite(r.get('best_single_score')) and _finite(r.get('hybrid_score'))):
            missing(f"{r['dataset']} / {r['model']}: best single or hybrid score not finite")
            continue
        marks.append(dict(r, metric=norm_metric(r['metric']), dataset_label=S.dataset_label(r['dataset']),
                          model_label=S.model_label(r['model']),
                          best_single_rep_label=S.rep_label(r['best_single_rep']) if r.get('best_single_rep') else ''))
    metrics = metric_order(m['metric'] for m in marks)
    fig, axes = panel_grid(len(metrics))
    for ax, met in zip(axes, metrics):
        ms = [m for m in marks if m['metric'] == met]
        log = met == 'rmse' and args.f1_rmse_log
        for m in ms:
            ax.scatter(m['best_single_score'], m['hybrid_score'], color=S.model_color(m['model']),
                       marker=S.model_marker(m['model']), s=22, lw=0, zorder=2)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        diagonal(ax, [m['best_single_score'] for m in ms], [m['hybrid_score'] for m in ms], log=log)
        ax.set_title(S.metric_label(met) + (', log scale' if log else ''))
        ax.set_xlabel('best single representation')
        ax.set_ylabel('hybrid')
        sns_despine(ax)
    model_legend(fig, {m['model'] for m in marks})
    fig.tight_layout()
    save(fig, out, 'f01_hybrid_vs_best_single', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'metric', 'best_single_rep',
          'best_single_rep_label', 'best_single_rep_by_fold', 'best_single_score', 'hybrid_label',
          'hybrid_label_by_fold', 'hybrid_score', 'n_folds_best_single', 'n_folds_hybrid', 'source'])


def sns_despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --------------------------------------------------------------------------- F2a / F2b

def _f2_datasets(root):
    """The same dataset order for F2a and F2b, so the two sit side by side."""
    q = Missing('f2', quiet=True)
    ds = [r['dataset'] for r in L2.load_f2a(root, q, source='c09')] + \
         [r['dataset'] for r in L2.load_f2b(root, q)]
    return dataset_order(ds)


def _model_offsets(marks):
    """{(dataset, model): vertical offset}, spreading only the models a dataset has."""
    out = {}
    for ds in {m['dataset'] for m in marks}:
        models = S.sort_models_by_family({m['model'] for m in marks if m['dataset'] == ds})
        n = len(models)
        for i, mo in enumerate(models):
            out[(ds, mo)] = (i - (n - 1) / 2) * 0.3
    return out


# F2a carries the dataset names; F2b, drawn in the same order, does not, so the
# two PNGs sit side by side at their saved widths (together 6.5 inches).
F2A_WIDTH_IN, F2B_WIDTH_IN = 4.0, 2.5


def fig_f2a(root, out, args, missing):
    """F2a: best number of features per dataset and model. y = dataset, x = features."""
    rows = L2.load_f2a(root, missing, source=args.f2a_source)
    marks = [dict(r, dataset_label=S.dataset_label(r['dataset']), model_label=S.model_label(r['model']))
             for r in rows if r.get('best_value') is not None]
    for r in rows:
        if r.get('best_value') is None:
            missing(f"{r['dataset']} / {r['model']}: no best number of features")
    order = _f2_datasets(root)
    off = _model_offsets(marks)
    fig, ax = plt.subplots(figsize=(F2A_WIDTH_IN, 0.24 * len(order) + 1.4))
    for m in marks:
        y = order.index(m['dataset']) + off[(m['dataset'], m['model'])]
        ax.scatter(m['best_value'], y, color=S.model_color(m['model']), marker=S.model_marker(m['model']),
                   s=22, lw=0)
    ax.set_xscale('log', base=2)
    ticks = sorted({v for m in marks for v in m['grid'] if v})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=90, fontsize=8)
    ax.minorticks_off()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([S.dataset_label(d) for d in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel('best number of features')
    sns_despine(ax)
    model_legend(fig, {m['model'] for m in marks}, ncol=3)
    fig.tight_layout()
    save(fig, out, 'f02a_best_number_of_features', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'metric', 'grid', 'best_value', 'best_score',
          'score_by_budget', 'n_folds', 'ties', 'best_is', 'source'])


def fig_f2b(root, out, args, missing):
    """F2b: number of base representations the search's top hybrid used, one mark per fold."""
    rows = L2.load_f2b(root, missing)
    marks = []
    for r in rows:
        for fold, lab in zip(r.get('per_fold_folds') or [], r.get('per_fold_labels') or []):
            marks.append(dict(dataset=r['dataset'], dataset_label=S.dataset_label(r['dataset']),
                              model=r['model'], model_label=S.model_label(r['model']), fold=fold,
                              n_representations=lab, grid=r.get('grid'), at_budget=r.get('at_budget'),
                              metric=r.get('metric'), best_score_at_budget=r.get('best_score'),
                              source=r.get('source')))
    cats = sorted({m['n_representations'] for m in marks}, key=lambda s: (s == 'no limit', int(s) if s.isdigit() else 0))
    order = _f2_datasets(root)
    off = _model_offsets(marks)
    fig, ax = plt.subplots(figsize=(F2B_WIDTH_IN, 0.24 * len(order) + 1.4))
    for m in marks:
        y = order.index(m['dataset']) + off[(m['dataset'], m['model'])]
        x = cats.index(m['n_representations']) + (int(m['fold']) - 2) * 0.08
        ax.scatter(x, y, color=S.model_color(m['model']), marker=S.model_marker(m['model']), s=12, lw=0)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace(' ', '\n') for c in cats], fontsize=9)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel('base representations\n(one mark per fold)')
    sns_despine(ax)
    model_legend(fig, {m['model'] for m in marks}, ncol=3)
    fig.tight_layout()
    save(fig, out, 'f02b_best_number_of_representations', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'fold', 'n_representations', 'grid',
          'at_budget', 'metric', 'best_score_at_budget', 'source'])


# --------------------------------------------------------------------------- F3

def fig_f3(root, out, args, missing):
    """F3: one horizontal bar per dataset, segments = features taken from each representation."""
    rows = L3.load_f3(root, missing)
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r['dataset'], []).append(r)
    marks = []
    for ds, rs in by_ds.items():
        models = [r['model'] for r in rs]
        model = args.f3_model if args.f3_model else models[0]   # first decided model for the dataset
        pick = [r for r in rs if r['model'] == model and int(r['fold']) == args.f3_fold]
        if not pick:
            missing(f'{ds}: no blueprint for model {model} on fold {args.f3_fold}')
            continue
        for r in sorted(pick, key=lambda r: -r['n_features']):
            marks.append(dict(r, dataset_label=S.dataset_label(ds), model_label=S.model_label(model),
                              representation_label=S.rep_label(r['representation'])))
    order = dataset_order(m['dataset'] for m in marks)
    domains = [d for d in S.DOMAIN_ORDER if any(S.DATASET_DOMAIN.get(ds) == d for ds in order)]
    blocks = []
    for dom in domains:
        ds_dom = [ds for ds in order if S.DATASET_DOMAIN.get(ds) == dom]
        reps = sorted({m['representation'] for m in marks if m['dataset'] in ds_dom},
                      key=lambda rp: -sum(m['n_features'] for m in marks
                                          if m['representation'] == rp and m['dataset'] in ds_dom))
        if len(reps) > len(S.REP_SEGMENT_COLORS):
            print(f'[f3] {dom}: {len(reps)} representations and {len(S.REP_SEGMENT_COLORS)} colours; '
                  f'colours repeat; the CSV names every segment', file=sys.stderr)
        colors = {rp: S.REP_SEGMENT_COLORS[i % len(S.REP_SEGMENT_COLORS)] for i, rp in enumerate(reps)}
        blocks.append((dom, ds_dom, reps, colors))
    ncol = 4
    heights = []
    for bi, (_, ds_dom, reps, _) in enumerate(blocks):
        last = bi == len(blocks) - 1
        heights += [0.3 * len(ds_dom) + 0.5, 0.2 * -(-len(reps) // ncol) + (0.9 if last else 0.55)]
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, sum(heights)))
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.05)
    xmax = max(sum(m['n_features'] for m in marks if m['dataset'] == ds) for ds in order)
    for bi, (dom, ds_dom, reps, colors) in enumerate(blocks):
        ax = fig.add_subplot(gs[2 * bi])
        for i, ds in enumerate(ds_dom):
            left = 0
            for m in [m for m in marks if m['dataset'] == ds]:
                ax.barh(i, m['n_features'], left=left, color=colors[m['representation']], height=0.7,
                        edgecolor='white', lw=0.3)
                m['segment_start'] = left
                m['segment_color'] = colors[m['representation']]
                left += m['n_features']
        ax.set_yticks(range(len(ds_dom)))
        ax.set_yticklabels([f"{S.dataset_label(d)} ({S.model_label(next(m['model'] for m in marks if m['dataset'] == d))})"
                            for d in ds_dom], fontsize=9)
        ax.set_ylim(len(ds_dom) - 0.5, -0.5)
        ax.set_xlim(0, xmax * 1.02)
        ax.set_title(dom, fontsize=10, loc='left')
        sns_despine(ax)
        if bi < len(blocks) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f'features in the hybrid, by base representation (outer fold {args.f3_fold})')
        lax = fig.add_subplot(gs[2 * bi + 1])
        lax.axis('off')
        h = [Patch(color=colors[rp], label=S.rep_label(rp)) for rp in reps]
        lax.legend(S._to_colmajor(h, ncol), S._to_colmajor([x.get_label() for x in h], ncol), ncol=ncol,
                   loc='lower center' if bi == len(blocks) - 1 else 'center', fontsize=7,
                   handlelength=1.0, columnspacing=1.0)
    save(fig, out, 'f03_blueprints', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'fold', 'slot', 'hybrid_label',
          'representation', 'representation_label', 'n_features', 'segment_start', 'segment_color',
          'total_features', 'source'])
    # The deeper small-molecule panel: the loader says why it cannot be drawn.
    L3.load_f3_detail(root, args.f3_detail_dataset, missing)


# --------------------------------------------------------------------------- F5 / F6

def _single_label(v):
    if isinstance(v, list):
        return '; '.join(S.rep_label(x) for x in v)
    return S.rep_label(v) if v else ''


def _scatter_by_dataset(root, out, missing, rows, name, xlab, ylab, extra_cols):
    marks = []
    for r in rows:
        if not r.get('primary', True):
            continue
        if not (_finite(r.get('single_value')) and _finite(r.get('hybrid_value'))):
            missing(f"{r['dataset']}: single or hybrid value not finite")
            continue
        marks.append(dict(r, metric=norm_metric(r['metric']), dataset_label=S.dataset_label(r['dataset']),
                          domain=S.DATASET_DOMAIN.get(r['dataset'], ''),
                          single_rep_label=_single_label(r.get('single_rep'))))
    metrics = metric_order(m['metric'] for m in marks)
    fig, axes = panel_grid(len(metrics), ncol=min(2, len(metrics)), h_per_row=3.0)
    if len(metrics) == 1:
        fig.set_size_inches(S.HALFWIDTH_IN + 0.4, 3.4)
    for ax, met in zip(axes, metrics):
        ms = [m for m in marks if m['metric'] == met]
        for m in ms:
            ax.scatter(m['single_value'], m['hybrid_value'], color=S.DOMAIN_COLORS.get(m['domain'], '#333333'),
                       s=24, lw=0, zorder=2)
        diagonal(ax, [m['single_value'] for m in ms], [m['hybrid_value'] for m in ms])
        ax.set_title(S.metric_label(met))
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        sns_despine(ax)
    domain_legend(fig, {m['domain'] for m in marks})
    fig.tight_layout()
    save(fig, out, name, marks,
         ['dataset', 'dataset_label', 'domain', 'metric', 'single_rep', 'single_rep_label', 'single_value',
          'hybrid_label', 'hybrid_value', 'setting', 'n_folds', 'per_fold_single', 'per_fold_hybrid']
         + extra_cols + ['source'])


def fig_f5(root, out, args, missing):
    """F5: Spearman of distance against activity difference, best single vs distance hybrid."""
    _scatter_by_dataset(root, out, missing, L56.load_f5(root, missing), 'f05_distance_tracks_activity',
                        'best single representation', 'hybrid built for distance', ['hybrid_kind'])


def fig_f6(root, out, args, missing):
    """F6: prediction from the 5 nearest neighbours, best single vs hybrid, setting fixed before the run."""
    _scatter_by_dataset(root, out, missing, L56.load_f6(root, missing), 'f06_nearest_neighbour_prediction',
                        'best single representation', 'hybrid', ['model'])


# --------------------------------------------------------------------------- F8

DETECTOR_WORDS = {
    'cross_val': 'out-of-fold residual',
    'neighbor_residual': 'neighbour-label gap',
    'neighbour-label gap': 'neighbour-label gap',
    'activity_cliff_aware': 'neighbour-label gap, cliff-aware',
    'labels at the assay limit': 'labels at the assay limit',
    'none (every label)': 'every label',
    'scaffold groups': 'scaffold groups',
    'gaussian_process uncertainty': 'GP predicted spread',
    'hetero_gp uncertainty': 'heteroscedastic GP predicted spread',
    'heteroscedastic_nn uncertainty': 'variance-head network predicted spread',
    'ngboost uncertainty': 'NGBoost predicted spread',
    'residual beyond c SD': 'residual beyond c SD',
}
# What each experiment does, in words, so two rows with the same method from
# different experiments can be told apart without the code.
EXPERIMENT_WORDS = {
    'n07': 'remediation grid', 'n18': 'labels as delivered', 'n20': 'cleaning before the search',
    'n23': 'uncertainty cleaning', 'n24': 'delete or keep', 'n25': 'cliff molecules put back',
    'n27': 'removal cost', 'n30': 'censored labels', 'n34': 'scaffold-group offsets',
    'n38': 'noise relative to model error',
}
DAMAGE_WORDS = {
    'censoring': 'clipped', 'gaussian': 'Gaussian', 'outlier_p05': 'outliers', 'shift_only_p10': 'shift',
    'sparse_dprime_up_p10': 'sparse spread', 'grouped_shifted': 'group offset', 'grouped_wider': 'group spread',
    'uniform': 'uniform flips', 'confusion_directed': 'directed flips', 'class_imbalance': 'imbalance flips',
    'instance_noise': 'instance flips', 'binary_asymmetric': 'asymmetric flips', 'class_dependent': 'class flips',
}


def f8_row_key(r):
    return f"{r['experiment']}|{r['detector']}|{r['handling']}|{r['variant']}"


def f8_row_label(r):
    det = DETECTOR_WORDS.get(r['detector'])
    if det is None:
        print(f"[f8] no words for detector {r['detector']!r}", file=sys.stderr)
        det = r['detector']
    lab = r['handling'] + ('' if det in ('every label',) else f', by {det}')
    if r.get('variant'):
        lab += f"; {r['variant']}"
    return f"{lab} [{EXPERIMENT_WORDS.get(r['experiment'], 'other')}]"


def fig_f8(root, out, args, missing):
    """F8: rows = methods, columns = datasets; shade = pairs improved, of 8; hatched = no run."""
    rows = L8.load_f8(root, missing)
    keys = sorted({f8_row_key(r) for r in rows})
    if args.f8_rows:
        want = [ln.strip() for ln in open(args.f8_rows) if ln.strip() and not ln.startswith('#')]
        for k in want:
            if k not in keys:
                missing(f'row {k!r} from --f8-rows has no result')
        keys = [k for k in want if k in keys]
    label = {f8_row_key(r): f8_row_label(r) for r in rows}
    keys = sorted(keys, key=lambda k: label[k])
    cell = {}
    for r in rows:
        cell[(f8_row_key(r), r['dataset'], r['grid'])] = r
    left_cols = dataset_order(r['dataset'] for r in rows if r['grid'] == 'delivered')
    dmg = {(r['dataset'], r['damage_kind']) for r in rows if r['grid'] != 'delivered'}
    dorder = dataset_order(d for d, _ in dmg)
    right_cols = sorted(dmg, key=lambda c: (dorder.index(c[0]), DAMAGE_WORDS.get(c[1], c[1])))
    cmap = plt.get_cmap('Purples', 9)
    marks = []
    h = 0.17 * len(keys) + 2.4
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(S.TEXTWIDTH_IN * 1.0, h),
                                   gridspec_kw={'width_ratios': [max(len(left_cols), 1), max(len(right_cols), 1)]})
    for ax, cols, side in ((axl, [(d, None) for d in left_cols], 'delivered'), (axr, right_cols, 'damage')):
        for i, k in enumerate(keys):
            for j, (ds, kind) in enumerate(cols):
                grid = 'delivered' if side == 'delivered' else f'damage:{kind}'
                r = cell.get((k, ds, grid))
                val = r.get('n_pairs_improved') if r else None
                if val is None:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='#BBBBBB',
                                               hatch='////', lw=0))
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(val / 8), edgecolor='white', lw=0.4))
                    if r.get('n_pairs_run', 8) < 8:
                        ax.text(j + 0.5, i + 0.5, '•', ha='center', va='center', fontsize=6,
                                color='white' if val >= 5 else 'black')
                if r is not None:
                    marks.append(dict(r, row_key=k, row_label=label[k], dataset_label=S.dataset_label(ds),
                                      column=S.dataset_label(ds) + ('' if side == 'delivered'
                                                                     else ', ' + DAMAGE_WORDS.get(kind, kind)),
                                      side=side, shade_value=val))
        ax.set_xlim(0, max(len(cols), 1))
        ax.set_ylim(len(keys), 0)
        ax.set_xticks([j + 0.5 for j in range(len(cols))])
        ax.set_xticklabels([S.dataset_label(d) + ('' if kind is None else ', ' + DAMAGE_WORDS.get(kind, kind))
                            for d, kind in cols], rotation=90, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axl.set_yticks([i + 0.5 for i in range(len(keys))])
    axl.set_yticklabels([label[k] for k in keys], fontsize=7)
    axr.set_yticks([])
    axl.set_title('labels as delivered')
    axr.set_title('injected damage, main level')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.5, 8.5))
    cb = fig.colorbar(sm, ax=[axl, axr], orientation='horizontal', fraction=0.03, pad=0.12, ticks=range(9))
    cb.set_label('model-and-configuration pairs improved, of 8')
    save(fig, out, 'f08_did_the_cleaning_work', marks,
         ['row_key', 'row_label', 'experiment', 'detector', 'handling', 'variant', 'dataset', 'dataset_label',
          'side', 'grid', 'damage_kind', 'column', 'level', 'metric', 'n_pairs_expected', 'n_pairs_run',
          'n_pairs_improved', 'shade_value', 'pairs', 'pairs_improved', 'pair_settings',
          'n_pairs_improved_by_level0_copy', 'n_draws', 'source'])


# --------------------------------------------------------------------------- F9

CONFIG_MARKERS = {'base_1': 'o', 'base_2': 's', 'hybrid': '^', 'hybrid__general2': 'D'}
STEPS = [('delivered', 'labels as\ndelivered'), ('clipped', 'after\nclipping'),
         ('corrected', 'clipped,\ncorrected'), ('deleted', 'clipped\nmolecules\ndeleted')]
CORRECTION_WORDS = {
    'constant_lift': 'clipped labels\nraised by\none constant', 'per_molecule_lift': 'clipped labels\nraised per\nmolecule',
    'truncated_normal_mean': 'truncated-normal\nmean', 'censored_em': 'censored EM\nestimate',
    'winsor_c1': 'pulled toward\nthe limit, c=1', 'winsor_c1.25': 'pulled toward\nthe limit, c=1.25',
    'winsor_c1.5': 'pulled toward\nthe limit, c=1.5',
}


def config_legend_handles(configs):
    return [Line2D([], [], ls='-', marker=CONFIG_MARKERS[c], color='#555555', ms=5, label=S.CONFIG_LABELS[c])
            for c in CONFIG_MARKERS if c in configs]


def fig_f9(root, out, args, missing):
    """F9: four steps on one axis (delivered, clipped, corrected, deleted), one panel per dataset."""
    rows = one_fold(L911.load_f9(root, missing), args.noise_fold, 'f9')
    datasets = args.f9_datasets.split(',')
    marks = []
    for r in rows:
        if r['dataset'] not in datasets or abs(r['level'] - args.f9_level) > 1e-9:
            continue
        if r['step'] == 'corrected' and r['correction_method'] != args.f9_correction:
            continue
        marks.append(dict(r, dataset_label=S.dataset_label(r['dataset']), model_label=S.model_label(r['model']),
                          configuration_label=S.CONFIG_LABELS.get(r['configuration'], r['configuration']),
                          step_index=[s for s, _ in STEPS].index(r['step'])))
    fig, axes = plt.subplots(1, len(datasets), figsize=(S.TEXTWIDTH_IN, 3.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ms = [m for m in marks if m['dataset'] == ds]
        if not ms:
            missing(f'{ds}: no ladder at clipped share {args.f9_level}')
        for (model, cfg) in sorted({(m['model'], m['configuration']) for m in ms}):
            line = sorted([m for m in ms if m['model'] == model and m['configuration'] == cfg],
                          key=lambda m: m['step_index'])
            for s, _ in STEPS:
                if not any(m['step'] == s for m in line):
                    missing(f'{ds} / {model} / {cfg}: no {s} score')
            ax.plot([m['step_index'] for m in line], [m['value'] for m in line], color=S.model_color(model),
                    marker=CONFIG_MARKERS.get(cfg, 'o'), ms=4, lw=1.0)
        ax.set_xticks(range(len(STEPS)))
        ax.set_xticklabels([CORRECTION_WORDS.get(args.f9_correction, lab) if st == 'corrected' else lab
                            for st, lab in STEPS], fontsize=8)
        ax.set_title(S.dataset_label(ds))
        ax.set_ylabel(f'held-out {S.metric_label("rmse")}')
        sns_despine(ax)
    h = [Line2D([], [], ls='-', color=S.model_color(m), label=S.model_label(m))
         for m in S.sort_models_by_family({m['model'] for m in marks})] + config_legend_handles(
        {m['configuration'] for m in marks})
    fig.legend(h, [x.get_label() for x in h], ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    save(fig, out, 'f09_censoring_ladder', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'configuration', 'configuration_label', 'step',
          'step_index', 'correction_method', 'arm', 'metric', 'value', 'level', 'n_draws', 'fold', 'source'])


# --------------------------------------------------------------------------- F10

MAIN_LEVEL = {'censoring': 0.25}


def _main_level(ds, kind):
    if kind in MAIN_LEVEL:
        return MAIN_LEVEL[kind]
    return 0.1 if ds == 'herg' else 1.0


def fig_f10(root, out, args, missing):
    """F10: x = detection AUROC, y = held-out score after the repair it drives; one dot per detector
    and repair; one panel per dataset and damage kind; one model-and-configuration pair."""
    rows = one_fold(L911.load_f10(root, missing), args.noise_fold, 'f10')
    model, cfg = args.f10_pair.split('@')
    rows = [r for r in rows if r.get('model') == model and r.get('configuration') == cfg]
    panels = {}
    for r in rows:
        panels.setdefault((r['dataset'], r['damage_kind']), set()).add(r['level'])
    chosen = {}
    for (ds, kind), levels in panels.items():
        target = _main_level(ds, kind)
        lv = min(levels, key=lambda x: abs(x - target))
        if abs(lv - target) > 1e-9:
            print(f'[f10] {ds} {kind}: no level {target}; drawn at {lv}, the nearest stored', file=sys.stderr)
        chosen[(ds, kind)] = lv
    marks = [dict(r, dataset_label=S.dataset_label(r['dataset']), model_label=S.model_label(model),
                  configuration_label=S.CONFIG_LABELS.get(cfg, cfg))
             for r in rows if chosen.get((r['dataset'], r['damage_kind'])) == r['level']
             and _finite(r.get('detection_auroc')) and _finite(r.get('repair_value'))]
    for r in rows:
        if chosen.get((r['dataset'], r['damage_kind'])) == r['level'] and not (
                _finite(r.get('detection_auroc')) and _finite(r.get('repair_value'))):
            missing(f"{r['dataset']} {r['damage_kind']} {r['detector']} {r.get('repair_method')}: "
                    f"detection AUROC or repair score absent")
    graded = L911.n32_detection_auroc(root)
    drawn = {m['detector'] for m in marks}
    for d in sorted({g.get('detector', str(g)) if isinstance(g, dict) else str(g) for g in (graded or [])} - drawn):
        missing(f'detector {d}: has a detection AUROC but drives no repair, so it has no y value')
    keys = dataset_order(ds for ds, _ in chosen)
    pkeys = sorted(chosen, key=lambda p: (keys.index(p[0]), p[1]))
    dets = sorted({m['detector'] for m in marks})
    dcol = {d: S.REP_SEGMENT_COLORS[i % len(S.REP_SEGMENT_COLORS)] for i, d in enumerate(dets)}
    fig, axes = panel_grid(len(pkeys), ncol=min(3, max(len(pkeys), 1)), h_per_row=2.4)
    for ax, (ds, kind) in zip(axes, pkeys):
        ms = [m for m in marks if (m['dataset'], m['damage_kind']) == (ds, kind)]
        for m in ms:
            ax.scatter(m['detection_auroc'], m['repair_value'], color=dcol[m['detector']], s=16, lw=0, zorder=2)
        nc = {m['no_cleaning_value'] for m in ms if _finite(m.get('no_cleaning_value'))}
        cl = {m['clean_value'] for m in ms if _finite(m.get('clean_value'))}
        for v in nc:
            ax.axhline(v, color='#555555', ls='--', lw=0.8, zorder=1)
        for v in cl:
            ax.axhline(v, color=S.CLEAN_COLOR, ls=':', lw=0.8, zorder=1)
        met = ms[0]['metric'] if ms else 'rmse'
        ax.set_title(f'{S.dataset_label(ds)}, {DAMAGE_WORDS.get(kind, kind)} {chosen[(ds, kind)]:g}', fontsize=9)
        ax.set_xlabel('detection AUROC ↑', fontsize=9)
        ax.set_ylabel(f'after repair, {S.metric_label(met)}', fontsize=9)
        sns_despine(ax)
    h = [Line2D([], [], ls='', marker='o', color=dcol[d],
                label=next(m['detector_label'] for m in marks if m['detector'] == d)) for d in dets]
    h += [Line2D([], [], ls='--', color='#555555', label='damaged labels, no cleaning'),
          Line2D([], [], ls=':', color=S.CLEAN_COLOR, label='clean labels')]
    fig.legend(h, [x.get_label() for x in h], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.suptitle(f'{S.model_label(model)}, {S.CONFIG_LABELS.get(cfg, cfg)}', y=0.995, fontsize=9)
    fig.tight_layout()
    save(fig, out, 'f10_detection_vs_repair', marks,
         ['dataset', 'dataset_label', 'damage_kind', 'level', 'detector', 'detector_label', 'detection_auroc',
          'repair_method', 'repair_label', 'repair_quantity_name', 'repair_value', 'no_cleaning_value',
          'clean_value', 'metric', 'model', 'model_label', 'configuration', 'configuration_label',
          'n_draws', 'fold', 'source'])


# --------------------------------------------------------------------------- F11

def fig_f11(root, out, args, missing):
    """F11: held-out score against damage level, two panels, one line per method."""
    rows = one_fold(L911.load_f11(root, missing), args.noise_fold, 'f11')
    model, cfg = args.f11_pair.split('@')
    methods = args.f11_methods.split(',')
    panels = [tuple(p.split(':')) for p in args.f11_panels.split(',')]
    marks = []
    for r in rows:
        if (r['dataset'], r['damage_kind']) in panels and r['model'] == model and r['configuration'] == cfg \
                and r['method'] in methods and r.get('features_from') == args.f11_features_from \
                and _finite(r.get('value')):
            marks.append(dict(r, dataset_label=S.dataset_label(r['dataset']), model_label=S.model_label(model),
                              configuration_label=S.CONFIG_LABELS.get(cfg, cfg)))
    mcol = {m: ('#000000' if m == 'noisy_baseline' else S.REP_SEGMENT_COLORS[i % len(S.REP_SEGMENT_COLORS)])
            for i, m in enumerate(methods)}
    fig, axes = plt.subplots(1, len(panels), figsize=(S.TEXTWIDTH_IN, 3.0), squeeze=False)
    for ax, (ds, kind) in zip(axes[0], panels):
        ms = [m for m in marks if (m['dataset'], m['damage_kind']) == (ds, kind)]
        levels = sorted({r['level'] for r in rows if (r['dataset'], r['damage_kind']) == (ds, kind)})
        for meth in methods:
            line = sorted([m for m in ms if m['method'] == meth], key=lambda m: m['level'])
            have = {m['level'] for m in line}
            for lv in levels:
                if lv not in have:
                    missing(f'{ds} {kind} {meth} {model}@{cfg}: no score at level {lv}')
            if line:
                ax.plot([m['level'] for m in line], [m['value'] for m in line], color=mcol[meth], marker='o',
                        ms=3.5, lw=1.2, label=line[0]['method_label'])
        met = ms[0]['metric'] if ms else ''
        ax.set_title(f'{S.dataset_label(ds)}, {DAMAGE_WORDS.get(kind, kind)}')
        ax.set_xlabel('clipped share' if kind == 'censoring' else 'damage level (share of label spread)')
        ax.set_ylabel(f'held-out {S.metric_label(met)}')
        sns_despine(ax)
    h, seen = [], set()
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in seen:
                seen.add(ll)
                h.append(hh)
    fig.legend(h, [x.get_label() for x in h], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.suptitle(f'{S.model_label(model)}, {S.CONFIG_LABELS.get(cfg, cfg)}', y=0.995, fontsize=9)
    fig.tight_layout()
    save(fig, out, 'f11_score_vs_damage_level', marks,
         ['dataset', 'dataset_label', 'damage_kind', 'method', 'method_label', 'model', 'model_label',
          'configuration', 'configuration_label', 'features_from', 'level', 'metric', 'value', 'n_draws',
          'fold', 'source'])


# --------------------------------------------------------------------------- registry

FIGURES = {
    'f1': ('F1', 'Hybrid against the best single representation', fig_f1, L1.INPUTS),
    'f2a': ('F2a', 'Best number of features, per dataset and model', fig_f2a, L2.INPUTS_F2A),
    'f2b': ('F2b', 'Best number of base representations, per dataset and model', fig_f2b, L2.INPUTS_F2B),
    'f3': ('F3', 'Blueprints: features taken from each representation', fig_f3, L3.INPUTS),
    'f5': ('F5', 'Distance against activity difference, single against hybrid', fig_f5, L56.INPUTS_F5),
    'f6': ('F6', 'Prediction from nearest neighbours, single against hybrid', fig_f6, L56.INPUTS_F6),
    'f8': ('F8', 'Did the cleaning work: pairs improved, of 8', fig_f8, L8.INPUTS),
    'f9': ('F9', 'The censoring ladder', fig_f9, L911.INPUTS_F9),
    'f10': ('F10', 'Detection against repair', fig_f10, L911.INPUTS_F10),
    'f11': ('F11', 'Score against damage level', fig_f11, L911.INPUTS_F11),
}
LOADER_CALLS = {
    'f1': lambda r, m, a: L1.load(r, m),
    'f2a': lambda r, m, a: L2.load_f2a(r, m, source=a.f2a_source),
    'f2b': lambda r, m, a: L2.load_f2b(r, m),
    'f3': lambda r, m, a: (L3.load_f3(r, m), L3.load_f3_detail(r, a.f3_detail_dataset, m)),
    'f5': lambda r, m, a: L56.load_f5(r, m),
    'f6': lambda r, m, a: L56.load_f6(r, m),
    'f8': lambda r, m, a: L8.load_f8(r, m),
    'f9': lambda r, m, a: L911.load_f9(r, m),
    'f10': lambda r, m, a: L911.load_f10(r, m),
    'f11': lambda r, m, a: L911.load_f11(r, m),
}


def _input_status(root, globs):
    """Files found per result folder. A loader reads final .json files, checkpoints or partial files,
    whichever exist, so a folder counts as present when any of its patterns matches."""
    by_dir = {}
    for g in globs:
        d = '/'.join(g.split('/')[:2])
        by_dir.setdefault(d, set()).update(glob.glob(str(root / g)))
    return [(d, len(hits)) for d, hits in by_dir.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--figure', default=None, help='one of ' + ', '.join(FIGURES) + ', or all')
    p.add_argument('--results-root', type=Path, default=None,
                   help=f'run folder with complexity/ similarity/ noise/ federated/ opig/ ledger.json. Default: '
                        f'the path in tests/results/CURRENT_RUN, else {LAPTOP_ROOT}')
    p.add_argument('--analysis-root', type=Path, default=Path('/Volumes/seagate/claude_analysis_2026-10-08'),
                   help='the 2026-10-08 analysis folder. Not read by any figure: every loader reads the raw '
                        'results, so the script runs where that folder does not exist. Kept for --list only.')
    p.add_argument('--out', type=Path, default=REPO / 'tests' / 'results' / 'thesis_figures')
    p.add_argument('--list', action='store_true', help='each figure, its inputs, and whether they exist')
    p.add_argument('--dry-run', action='store_true', help='run every loader, print what is missing, plot nothing')
    # Choices the outline does not fix. Defaults are stated in the reply that delivered this script.
    p.add_argument('--f1-rmse-log', action=argparse.BooleanOptionalAction, default=True,
                   help='log axes on the RMSE panel of f1 (Buchwald-Hartwig yield RMSE is about 15, the rest under 1)')
    p.add_argument('--f2a-source', choices=['c09', 'c16'], default='c09')
    p.add_argument('--f3-fold', type=int, default=0)
    p.add_argument('--f3-model', default=None, help='default: the first decided model of each dataset')
    p.add_argument('--f3-detail-dataset', default='logd')
    p.add_argument('--f8-rows', default=None,
                   help='file with one row key per line (experiment|detector|handling|variant, as in the CSV '
                        'row_key column). Default: every method.')
    p.add_argument('--noise-fold', type=int, default=0,
                   help='outer fold drawn by f9, f10 and f11 (f8 reads fold 0 inside its loader)')
    p.add_argument('--f9-datasets', default='caco2,herg_ki')
    p.add_argument('--f9-level', type=float, default=0.25)
    p.add_argument('--f9-correction', default='constant_lift')
    p.add_argument('--f10-pair', default='dnn@hybrid', help='model@configuration')
    p.add_argument('--f11-panels', default='herg_ki:gaussian,herg_ki:censoring', help='dataset:damage_kind,...')
    p.add_argument('--f11-pair', default='dnn@hybrid', help='model@configuration')
    p.add_argument('--f11-methods', default='noisy_baseline,label_smooth,oof_smooth,remove@flag_share0.1,js_shrink')
    p.add_argument('--f11-features-from', default='damaged labels', choices=['damaged labels', 'clean labels'])
    args = p.parse_args()
    root = args.results_root or default_results_root()

    if args.list:
        print(f'results root: {root} ({"exists" if root.exists() else "MISSING"})')
        for key, (slot, title, _, globs) in FIGURES.items():
            print(f'\n{slot:4s} --figure {key:4s} {title}')
            for g, n in _input_status(root, globs):
                print(f'       {"ok " if n else "NO "} {n:4d} files  {g}')
        return 0

    keys = list(FIGURES) if args.figure in (None, 'all') else [args.figure]
    if args.figure is None and not args.dry_run:
        p.error('give --figure, --list or --dry-run')
    for k in keys:
        if k not in FIGURES:
            p.error(f'unknown figure {k}')

    if args.dry_run:
        print(f'results root: {root}')
        for k in keys:
            slot, title, _, globs = FIGURES[k]
            absent = [g for g, n in _input_status(root, globs) if n == 0]
            miss = Missing(k, quiet=True)
            try:
                LOADER_CALLS[k](root, miss, args)
                err = None
            except Exception as e:  # report and go on to the next figure
                err = f'{type(e).__name__}: {e}'
            print(f'\n{slot} {title}')
            for g in absent:
                print(f'  no files: {g}')
            if err:
                print(f'  LOADER FAILED: {err}')
            print(f'  {len(miss.msgs)} missing marks' + (':' if miss.msgs else ''))
            for m in miss.msgs:
                print(f'    {m}')
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for k in keys:
        miss = Missing(k)
        try:
            FIGURES[k][2](root, args.out, args, miss)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append(k)
            print(f'[{k}] FAILED: {type(e).__name__}: {e}', file=sys.stderr)
        if miss.msgs:
            print(f'[{k}] {len(miss.msgs)} missing marks (listed above on stderr)')
    if failed:
        print('figures that failed: ' + ', '.join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
