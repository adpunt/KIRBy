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
  f1   table: hybrid against best single representation c02 standard_selection_summary
  f2a  best number of features                         c09 bias_variance curves (or c16)
  f3   blueprints, features per representation         c22 allocation_built
  f5   table: distance against activity difference     s02 comparisons (knn distance hybrid)
  f6   table: prediction from nearest neighbours        s05 summary.fixed_ref
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


def thesis_datasets(root, args):
    """The datasets a cross-dataset figure shows, or None for every dataset.

    PLACEHOLDER until the thesis list is fixed (Adelaide, 2026-10-09): by default the
    dataset rows of <results root>/ledger.json whose noise_full stage holds at least one
    unit, i.e. the datasets promoted to further outer folds. --datasets a,b,c names
    them instead; --datasets all shows every dataset."""
    if args.datasets == 'all':
        return None
    if args.datasets != 'promoted':
        return set(args.datasets.split(','))
    path = root / 'ledger.json'
    try:
        rows = json.load(open(path))['rows']
    except (OSError, ValueError, KeyError) as e:
        print(f'[datasets] cannot read promoted datasets from {path} ({e}); showing every dataset',
              file=sys.stderr)
        return None
    keep = {name for name, row in rows.items() if row.get('kind') == 'dataset' and any(
        st.get('name') == 'noise_full' and st.get('units') for st in row.get('stages', []))}
    if not keep:
        print(f'[datasets] {path} has no promoted datasets; showing every dataset', file=sys.stderr)
        return None
    print(f'[datasets] promoted in the ledger, so drawn: {", ".join(sorted(keep))}', file=sys.stderr)
    return keep


def keep_datasets(rows, root, args):
    keep = thesis_datasets(root, args)
    return rows if keep is None else [r for r in rows if r['dataset'] in keep]


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


def place_legend(fig, handles, ncol=None, side=False, fontsize=None):
    """One legend for the whole figure, underneath it or to its right, never on top
    (the paper script's shared_legend). Call after fig.tight_layout()."""
    labels = [h.get_label() for h in handles]
    kw = dict(frameon=False, fontsize=fontsize) if fontsize else dict(frameon=False)
    if side:
        fig.legend(handles, labels, ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), **kw)
        return
    ncol = ncol or len(handles)
    fig.legend(S._to_colmajor(handles, ncol), S._to_colmajor(labels, ncol), ncol=ncol,
               loc='upper center', bbox_to_anchor=(0.5, 0.0), **kw)


def model_handles(models):
    return [Line2D([], [], ls='', marker=S.model_marker(m), color=S.model_color(m), ms=6,
                   label=S.model_label(m)) for m in S.sort_models_by_family(models)]


def model_legend(fig, models, ncol=None, side=False):
    place_legend(fig, model_handles(models), ncol=ncol, side=side)


def row_bands(ax, n):
    """Grey band behind every other dataset row, so a dot reads against its own row."""
    for i in range(0, n, 2):
        ax.axhspan(i - 0.5, i + 0.5, color='#EFEFEF', lw=0, zorder=0)


def panel_grid(n, ncol=2, h_per_row=3.0):
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.TEXTWIDTH_IN, h_per_row * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


# --------------------------------------------------------------------------- F1

TEX_METRIC = {
    'rmse': r'RMSE $\downarrow$', 'mae': r'MAE $\downarrow$', 'spearman': r'Spearman $\uparrow$',
    'pearson': r'Pearson $\uparrow$', 'auroc': r'AUROC $\uparrow$', 'mcc': r'MCC $\uparrow$',
    'f1_weighted': r'Weighted F1 $\uparrow$', 'r2': r'$R^2$ $\uparrow$',
}


def _tex(text):
    return str(text).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def _pair_cells(a, b, metric):
    """Two stored scores, the better one in bold; equal at four decimals, neither bold."""
    fa, fb = f'{a:.4f}', f'{b:.4f}'
    if fa == fb:
        return fa, fb
    better_a = (a < b) if metric in S.LOWER_IS_BETTER else (a > b)
    return (rf'\textbf{{{fa}}}', fb) if better_a else (fa, rf'\textbf{{{fb}}}')


def write_pair_table(out, name, rows, head, caption, label, csv_cols, with_model):
    """Rows grouped by metric, a header line per metric; one stored number per cell; the better of
    each pair in bold (Adelaide, 2026-10-09: tables replace the diagonal plots of F1, F5 and F6)."""
    metrics = metric_order(r['metric'] for r in rows)
    ncol = 4 if with_model else 3
    spec = '@{}ll' + 'r' * (ncol - 2) + '@{}' if with_model else '@{}l' + 'r' * (ncol - 1) + '@{}'
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\begin{adjustbox}{max width=\textwidth,max totalheight=0.78\textheight}',
             rf'\begin{{tabular}}{{{spec}}}', r'\toprule',
             ' & '.join(rf'\textbf{{{h}}}' for h in head) + r' \\']
    for met in metrics:
        lines += [r'\midrule', rf'\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{TEX_METRIC.get(met, _tex(met))}}}}} \\']
        ms = [r for r in rows if r['metric'] == met]
        prev = None
        for ds in dataset_order(r['dataset'] for r in ms):
            for r in sorted([r for r in ms if r['dataset'] == ds],
                            key=lambda r: S.MODEL_ORDER.index(r['model']) if r.get('model') in S.MODEL_ORDER else 99):
                ca, cb = _pair_cells(r['single'], r['hybrid'], met)
                name_cell = _tex(S.dataset_label(ds)) if ds != prev else ''
                cells = [name_cell] + ([_tex(S.model_label(r['model']))] if with_model else []) + [ca, cb]
                lines.append(' & '.join(cells) + r' \\')
                prev = ds
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', rf'\caption{{{caption}}}',
              rf'\label{{{label}}}', r'\end{table}', '']
    (out / f'{name}.tex').write_text('\n'.join(lines))
    write_csv(out / f'{name}.csv', rows, csv_cols)
    print(f'wrote {out / name}.tex and {name}.csv ({len(rows)} rows)')


def fig_f1(root, out, args, missing):
    """F1, now a table: one row per dataset and model; best single base representation and hybrid,
    each in the dataset's own metric, the better in bold."""
    rows = keep_datasets(L1.load(root, missing), root, args)
    marks = []
    for r in rows:
        if not (_finite(r.get('best_single_score')) and _finite(r.get('hybrid_score'))):
            missing(f"{r['dataset']} / {r['model']}: best single or hybrid score not finite")
            continue
        marks.append(dict(r, metric=norm_metric(r['metric']), dataset_label=S.dataset_label(r['dataset']),
                          model_label=S.model_label(r['model']), single=r['best_single_score'],
                          hybrid=r['hybrid_score'],
                          best_single_rep_label=S.rep_label(r['best_single_rep']) if r.get('best_single_rep') else ''))
    write_pair_table(out, 'tab_hybrid_vs_best_single', marks,
                     ['Dataset', 'Model', 'Best single base representation', 'Hybrid'],
                     'Held-out score of the best single base representation and the hybrid, per dataset and '
                     'model, mean over outer folds. Bold: the better of the two.',
                     'tab:hybrid_vs_best_single',
                     ['dataset', 'dataset_label', 'model', 'model_label', 'metric', 'best_single_rep',
                      'best_single_rep_label', 'best_single_rep_by_fold', 'best_single_score', 'hybrid_label',
                      'hybrid_label_by_fold', 'hybrid_score', 'n_folds_best_single', 'n_folds_hybrid', 'source'],
                     with_model=True)


def sns_despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --------------------------------------------------------------------------- F2a

def _f2_datasets(root, args):
    """F2a's datasets, domain first."""
    q = Missing('f2', quiet=True)
    ds = [r['dataset'] for r in L2.load_f2a(root, q, source=args.f2a_source)]
    keep = thesis_datasets(root, args)
    return dataset_order(d for d in ds if keep is None or d in keep)


def _model_offsets(marks):
    """{(dataset, model): vertical offset}, spreading only the models a dataset has."""
    out = {}
    for ds in {m['dataset'] for m in marks}:
        models = S.sort_models_by_family({m['model'] for m in marks if m['dataset'] == ds})
        n = len(models)
        step = min(0.2, 0.5 / max(n - 1, 1))   # every dot stays inside its own row band
        for i, mo in enumerate(models):
            out[(ds, mo)] = (i - (n - 1) / 2) * step
    return out


F2A_WIDTH_IN = 4.0


def fig_f2a(root, out, args, missing):
    """F2a: best number of features per dataset and model. y = dataset, x = features."""
    rows = keep_datasets(L2.load_f2a(root, missing, source=args.f2a_source), root, args)
    marks = [dict(r, dataset_label=S.dataset_label(r['dataset']), model_label=S.model_label(r['model']))
             for r in rows if r.get('best_value') is not None]
    for r in rows:
        if r.get('best_value') is None:
            missing(f"{r['dataset']} / {r['model']}: no best number of features")
    order = _f2_datasets(root, args)
    off = _model_offsets(marks)
    fig, ax = plt.subplots(figsize=(F2A_WIDTH_IN, 0.24 * len(order) + 1.4))
    row_bands(ax, len(order))
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
    ax.set_xlabel('Best number\nof features')
    ax.tick_params(axis='y', length=0)
    sns_despine(ax)
    fig.tight_layout()
    model_legend(fig, {m['model'] for m in marks}, ncol=3)
    save(fig, out, 'f02a_best_number_of_features', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'metric', 'grid', 'best_value', 'best_score',
          'score_by_budget', 'n_folds', 'ties', 'best_is', 'source'])


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
            ax.set_xlabel(f'Features in the hybrid, by base representation (outer fold {args.f3_fold})')
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
    """One name per fold; a fold whose result names no representation reads 'none stored'."""
    if isinstance(v, list):
        return '; '.join(S.rep_label(x) if x else 'none stored' for x in v)
    return S.rep_label(v) if v else ''


def _pair_table_by_dataset(root, out, args, missing, rows, name, head, caption, label, extra_cols):
    marks = []
    for r in keep_datasets(rows, root, args):
        if not r.get('primary', True):
            continue
        if not (_finite(r.get('single_value')) and _finite(r.get('hybrid_value'))):
            missing(f"{r['dataset']}: single or hybrid value not finite")
            continue
        marks.append(dict(r, metric=norm_metric(r['metric']), dataset_label=S.dataset_label(r['dataset']),
                          single=r['single_value'], hybrid=r['hybrid_value'],
                          single_rep_label=_single_label(r.get('single_rep'))))
    write_pair_table(out, name, marks, head, caption, label,
                     ['dataset', 'dataset_label', 'metric', 'single_rep', 'single_rep_label', 'single_value',
                      'hybrid_label', 'hybrid_value', 'setting', 'n_folds', 'per_fold_single', 'per_fold_hybrid']
                     + extra_cols + ['source'], with_model=False)


def fig_f5(root, out, args, missing):
    """F5, now a table: Spearman of distance against activity difference, best single base
    representation against the hybrid built for distance."""
    _pair_table_by_dataset(root, out, args, missing, L56.load_f5(root, missing), 'tab_distance_tracks_activity',
                           ['Dataset', 'Best single base representation', 'Hybrid built for distance'],
                           'Spearman correlation between the distance between two samples and the difference in '
                           'their labels, per dataset, mean over outer folds. Bold: the better of the two.',
                           'tab:distance_tracks_activity', ['hybrid_kind'])


def fig_f6(root, out, args, missing):
    """F6, now a table: prediction from the five nearest neighbours, best single base representation
    against the hybrid, at the setting fixed before the run."""
    _pair_table_by_dataset(root, out, args, missing, L56.load_f6(root, missing), 'tab_nearest_neighbour_prediction',
                           ['Dataset', 'Best single base representation', 'Hybrid'],
                           'Held-out score of predicting each molecule from its five nearest training molecules, '
                           'per dataset, mean over outer folds. Bold: the better of the two.',
                           'tab:nearest_neighbour_prediction', ['model'])


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


VARIANT_WORDS = {
    '': '', 'cliff molecules put back': 'Cliff molecules put back', 'plain cleanup': 'Plain cleanup',
    'cleaned for the feature search only, fit on labels as delivered': 'Cleaned for the search only',
    'fit on cleaned labels': 'Cleaned for search and fit',
}
F8_CELL_IN = 0.145      # one square, inches; 7 pt text fits a row this tall
F8_LABEL_IN = 2.1       # room for the row names
F8_PAGE_W, F8_PAGE_H = S.TEXTWIDTH_IN, 9.0


def _det_words(d):
    w = DETECTOR_WORDS.get(d)
    if w is None:
        print(f"[f8] no words for detector {d!r}", file=sys.stderr)
        w = d
    return w


def fig_f8(root, out, args, missing):
    """F8, one file per experiment so every file fits the page at 7 pt (Adelaide, 2026-10-09: the
    single 102-row figure was 12.7 x 15.8 inches). Rows = what is done to the flagged labels, grouped
    under the detector that flags them; column blocks = labels as delivered, then injected damage at the
    main level, once per variant of the experiment; shade = pairs improved, of 8; hatched = no run."""
    rows = L8.load_f8(root, missing)
    if args.f8_rows:
        want = [ln.strip() for ln in open(args.f8_rows) if ln.strip() and not ln.startswith('#')]
        have = {f8_row_key(r) for r in rows}
        for k in want:
            if k not in have:
                missing(f'row {k!r} from --f8-rows has no result')
        rows = [r for r in rows if f8_row_key(r) in set(want)]
    cmap = plt.get_cmap('Purples', 9)
    for exp in sorted({r['experiment'] for r in rows}, key=lambda e: EXPERIMENT_WORDS.get(e, e)):
        er = [r for r in rows if r['experiment'] == exp]
        cell = {(r['detector'], r['handling'], r['variant'], r['dataset'], r['grid']): r for r in er}
        # rows: detector header, then one row per handling under it
        dets = sorted({r['detector'] for r in er}, key=_det_words)
        ylines = []
        for d in dets:
            ylines.append(('header', d, None))
            for h in sorted({r['handling'] for r in er if r['detector'] == d}):
                ylines.append(('row', d, h))
        # column blocks: per variant, delivered then damage; columns with no run anywhere are left out
        blocks = []
        for v in sorted({r['variant'] for r in er}, key=lambda v: list(VARIANT_WORDS).index(v) if v in VARIANT_WORDS else 99):
            vr = [r for r in er if r['variant'] == v and r.get('n_pairs_improved') is not None]
            deliv = [(ds, None) for ds in dataset_order(r['dataset'] for r in vr if r['grid'] == 'delivered')]
            dmg = {(r['dataset'], r['damage_kind']) for r in vr if r['grid'] != 'delivered'}
            dorder = dataset_order(ds for ds, _ in dmg)
            dmg = sorted(dmg, key=lambda c: (dorder.index(c[0]), DAMAGE_WORDS.get(c[1], c[1])))
            vw = VARIANT_WORDS.get(v, v)
            for side, cols in (('delivered', deliv), ('damage', dmg)):
                if cols:
                    blocks.append((v, side, cols, 'Delivered' if side == 'delivered' else 'Damaged'))
        if not blocks:
            continue
        ncols = sum(len(b[2]) for b in blocks)
        names = [('By ' + _det_words(d)) if k == 'header' else S.cap(h) for k, d, h in ylines]
        label_in = max(len(n) for n in names) * 0.057 + 0.1     # 7 pt, bold headers included
        gaps = [0.15 if blocks[i][0] == blocks[i - 1][0] else 0.4 for i in range(1, len(blocks))]
        multi = len({b[0] for b in blocks}) > 1
        width = max(label_in + ncols * F8_CELL_IN + sum(gaps) + 0.2, 3.6)
        longest = max(len(S.dataset_label(ds) + ('' if dk is None else ', ' + DAMAGE_WORDS.get(dk, dk)))
                      for b in blocks for ds, dk in b[2])
        bottom_in = 0.8 + longest * 0.045         # colour bar, then the rotated column names
        top_in = 0.75 if multi else 0.55
        height = len(ylines) * F8_CELL_IN + bottom_in + top_in
        if width > F8_PAGE_W + 0.01 or height > F8_PAGE_H + 0.01:
            print(f'[f8] {EXPERIMENT_WORDS.get(exp, exp)}: {width:.1f} x {height:.1f} in, over the '
                  f'{F8_PAGE_W} x {F8_PAGE_H} in page; pick rows with --f8-rows', file=sys.stderr)
        fig = plt.figure(figsize=(width, height))
        left = label_in / width
        x0 = left
        bottom = bottom_in / height
        top = 1 - top_in / height
        marks = []
        axes = []
        for bi, (v, side, cols, title) in enumerate(blocks):
            w = len(cols) * F8_CELL_IN / width
            ax = fig.add_axes([x0, bottom, w, top - bottom])
            axes.append(ax)
            x0 += w + (gaps[bi] / width if bi < len(gaps) else 0)
            for i, (kind, d, h) in enumerate(ylines):
                if kind == 'header':
                    continue
                for j, (ds, dk) in enumerate(cols):
                    grid = 'delivered' if side == 'delivered' else f'damage:{dk}'
                    r = cell.get((d, h, v, ds, grid))
                    val = r.get('n_pairs_improved') if r else None
                    if val is None:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='#BBBBBB',
                                                   hatch='////', lw=0))
                    else:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(val / 8), edgecolor='white', lw=0.4))
                        if r.get('n_pairs_run', 8) < 8:
                            ax.text(j + 0.5, i + 0.5, '•', ha='center', va='center', fontsize=5,
                                    color='white' if val >= 5 else 'black')
                    if r is not None:
                        marks.append(dict(r, row_key=f8_row_key(r), row_label=f'{_det_words(d)} / {h}',
                                          dataset_label=S.dataset_label(ds), side=side, shade_value=val,
                                          column=S.dataset_label(ds) + ('' if dk is None
                                                                         else ', ' + DAMAGE_WORDS.get(dk, dk))))
            ax.set_xlim(0, len(cols))
            ax.set_ylim(len(ylines), 0)
            ax.set_xticks([j + 0.5 for j in range(len(cols))])
            ax.set_xticklabels([S.dataset_label(ds) + ('' if dk is None else ', ' + DAMAGE_WORDS.get(dk, dk))
                                for ds, dk in cols], rotation=90, fontsize=6.5)
            ax.set_title(title, fontsize=6.5, pad=2)
            ax.tick_params(length=0, pad=1)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_yticks([])
        if multi:   # one variant name over its pair of blocks
            for v in dict.fromkeys(b[0] for b in blocks):
                axs = [a for a, b in zip(axes, blocks) if b[0] == v]
                x_l, x_r = axs[0].get_position().x0, axs[-1].get_position().x1
                fig.text((x_l + x_r) / 2, top + 0.3 / height, VARIANT_WORDS.get(v, v), ha='center',
                         va='bottom', fontsize=7)
        ax0 = axes[0]
        ax0.set_yticks([i + 0.5 for i in range(len(ylines))])
        ax0.set_yticklabels([('By ' + _det_words(d)) if kind == 'header' else S.cap(h).replace(' gp ', ' GP ')
                             for kind, d, h in ylines], fontsize=7)
        for lab, (kind, _, _) in zip(ax0.get_yticklabels(), ylines):
            if kind == 'header':
                lab.set_fontweight('bold')
        cb_w = 2.4
        cax = fig.add_axes([(width - cb_w) / 2 / width, 0.4 / height, cb_w / width, 0.12 / height])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.5, 8.5)), cax=cax,
                          orientation='horizontal', ticks=range(9))
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label('Model-and-configuration pairs improved, of 8', fontsize=7)
        fig.suptitle(S.cap(EXPERIMENT_WORDS.get(exp, exp)), x=0.02, ha='left', fontsize=9, y=1 - 0.12 / height)
        slug = EXPERIMENT_WORDS.get(exp, exp).replace(' ', '_').replace("'", '')
        png = out / f'f08_did_the_cleaning_work_{slug}.png'
        fig.savefig(png, dpi=300)     # fixed size, no tight crop: the file is the size it prints at
        plt.close(fig)
        write_csv(out / f'f08_did_the_cleaning_work_{slug}.csv', marks,
                  ['row_key', 'row_label', 'experiment', 'detector', 'handling', 'variant', 'dataset',
                   'dataset_label', 'side', 'grid', 'damage_kind', 'column', 'level', 'metric', 'n_pairs_expected',
                   'n_pairs_run', 'n_pairs_improved', 'shade_value', 'pairs', 'pairs_improved', 'pair_settings',
                   'n_pairs_improved_by_level0_copy', 'n_draws', 'source'])
        print(f'wrote {png} ({width:.1f} x {height:.1f} in, {len(marks)} marks)')


# --------------------------------------------------------------------------- F9

# Colour = what the model was fitted on: base representations orange and yellow, hybrids
# blue and green (Adelaide, 2026-10-09). Shape = model. CLEAN_COLOR is not used here.
CONFIG_COLORS = {'base_1': '#E69F00', 'base_2': '#DDCC77', 'hybrid': '#4477AA', 'hybrid__general2': '#228833'}
F9_MODEL_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
STEPS = [('delivered', 'Labels as\ndelivered'), ('clipped', 'After\nclipping'),
         ('corrected', 'Clipped,\ncorrected'), ('deleted', 'Clipped\nmolecules\ndeleted')]
CORRECTION_WORDS = {
    'constant_lift': 'Clipped labels\nraised by\none constant', 'per_molecule_lift': 'Clipped labels\nraised per\nmolecule',
    'truncated_normal_mean': 'Truncated-normal\nmean', 'censored_em': 'Censored EM\nestimate',
    'winsor_c1': 'Pulled toward\nthe limit, c=1', 'winsor_c1.25': 'Pulled toward\nthe limit, c=1.25',
    'winsor_c1.5': 'Pulled toward\nthe limit, c=1.5',
}


def fig_f9(root, out, args, missing):
    """F9: four steps on one axis (delivered, clipped, corrected, deleted), one panel per dataset.
    One line per model and configuration: colour = configuration, shape = model."""
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
    models = S.sort_models_by_family({m['model'] for m in marks})
    mmark = {mo: F9_MODEL_MARKERS[i % len(F9_MODEL_MARKERS)] for i, mo in enumerate(models)}
    fig, axes = plt.subplots(1, len(datasets), figsize=(S.TEXTWIDTH_IN, 3.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ms = [m for m in marks if m['dataset'] == ds]
        if not ms:
            missing(f'{ds}: no ladder at clipped share {args.f9_level}')
        for (model, cfg) in sorted({(m['model'], m['configuration']) for m in ms}):
            line = sorted([m for m in ms if m['model'] == model and m['configuration'] == cfg],
                          key=lambda m: m['step_index'])
            for st, _ in STEPS:
                if not any(m['step'] == st for m in line):
                    missing(f'{ds} / {model} / {cfg}: no {st} score')
            ax.plot([m['step_index'] for m in line], [m['value'] for m in line],
                    color=CONFIG_COLORS.get(cfg, '#555555'), marker=mmark[model], ms=4.5, lw=1.0,
                    mec='white', mew=0.4)
        ax.set_xticks(range(len(STEPS)))
        ax.set_xticklabels([CORRECTION_WORDS.get(args.f9_correction, lab) if st == 'corrected' else lab
                            for st, lab in STEPS], fontsize=8)
        ax.set_title(S.dataset_label(ds))
        ax.set_ylabel(f'Held-out {S.metric_label("rmse")}')
        sns_despine(ax)
    fig.tight_layout()
    cfgs = [c for c in CONFIG_COLORS if any(m['configuration'] == c for m in marks)]
    h = [Line2D([], [], ls='-', lw=2, color=CONFIG_COLORS[c], label=S.CONFIG_LABELS[c]) for c in cfgs]
    h += [Line2D([], [], ls='', marker=mmark[mo], color='#555555', ms=5, label=S.model_label(mo)) for mo in models]
    place_legend(fig, h, ncol=3, fontsize=8)
    for m in marks:
        m['line_color'], m['marker'] = CONFIG_COLORS.get(m['configuration']), mmark[m['model']]
    save(fig, out, 'f09_censoring_ladder', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'configuration', 'configuration_label', 'step',
          'step_index', 'correction_method', 'arm', 'metric', 'value', 'level', 'n_draws', 'fold',
          'line_color', 'marker', 'source'])


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
    dets = sorted({m['detector'] for m in marks})
    dcol = {d: S.REP_SEGMENT_COLORS[i % len(S.REP_SEGMENT_COLORS)] for i, d in enumerate(dets)}
    for m in marks:
        m['dot_color'] = dcol[m['detector']]
    # One file per dataset, one panel per damage kind, two panels to a row (Adelaide,
    # 2026-10-09: eleven panels in one figure were too small to read).
    from matplotlib.ticker import MaxNLocator
    for ds in dataset_order(ds for ds, _ in chosen):
        kinds = sorted(k for d, k in chosen if d == ds)
        ncol = min(2, len(kinds))
        nrow = -(-len(kinds) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(S.TEXTWIDTH_IN if ncol == 2 else S.HALFWIDTH_IN + 0.6,
                                                      2.7 * nrow + 0.3), squeeze=False)
        for ax in axes.flat[len(kinds):]:
            ax.set_visible(False)
        ds_marks = [m for m in marks if m['dataset'] == ds]
        for ax, kind in zip(axes.flat, kinds):
            ms = [m for m in ds_marks if m['damage_kind'] == kind]
            for m in ms:
                ax.scatter(m['detection_auroc'], m['repair_value'], color=dcol[m['detector']], s=20, lw=0, zorder=2)
            for v in {m['no_cleaning_value'] for m in ms if _finite(m.get('no_cleaning_value'))}:
                ax.axhline(v, color='#555555', ls='--', lw=0.8, zorder=1)
            for v in {m['clean_value'] for m in ms if _finite(m.get('clean_value'))}:
                ax.axhline(v, color=S.CLEAN_COLOR, ls=':', lw=0.8, zorder=1)
            met = ms[0]['metric'] if ms else 'rmse'
            word = DAMAGE_WORDS.get(kind, kind)
            ax.set_title(f'{S.cap(word)}, level {chosen[(ds, kind)]:g}', fontsize=9)
            ax.set_xlabel('Detection AUROC ↑', fontsize=9)
            ax.set_ylabel(f'After repair, {S.metric_label(met)}', fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(labelsize=8)
            sns_despine(ax)
        fig.suptitle(f'{S.dataset_label(ds)}: {S.model_label(model)} on the {S.CONFIG_LABELS.get(cfg, cfg).lower()}', fontsize=10)
        fig.tight_layout()
        here = sorted({m['detector'] for m in ds_marks})
        h = [Line2D([], [], ls='', marker='o', color=dcol[d],
                    label=S.cap(next(m['detector_label'] for m in ds_marks if m['detector'] == d))) for d in here]
        h += [Line2D([], [], ls='--', color='#555555', label='Damaged labels, no cleaning'),
              Line2D([], [], ls=':', color=S.CLEAN_COLOR, label='Clean labels')]
        place_legend(fig, h, ncol=1 if ncol == 1 else 2, fontsize=7)
        save(fig, out, f'f10_detection_vs_repair_{ds}', ds_marks,
             ['dataset', 'dataset_label', 'damage_kind', 'level', 'detector', 'detector_label', 'detection_auroc',
              'repair_method', 'repair_label', 'repair_quantity_name', 'repair_value', 'no_cleaning_value',
              'clean_value', 'metric', 'model', 'model_label', 'configuration', 'configuration_label',
              'dot_color', 'n_draws', 'fold', 'source'])


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
                        ms=3.5, lw=1.2, label=S.cap(line[0]['method_label']))
        met = ms[0]['metric'] if ms else ''
        ax.set_title(f'{S.dataset_label(ds)}, {S.cap(DAMAGE_WORDS.get(kind, kind))}')
        ax.set_xlabel('Clipped share' if kind == 'censoring' else 'Damage level (share of label spread)')
        ax.set_ylabel(f'Held-out {S.metric_label(met)}')
        sns_despine(ax)
    h, seen = [], set()
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in seen:
                seen.add(ll)
                h.append(hh)
    fig.suptitle(f'{S.model_label(model)} on the {S.CONFIG_LABELS.get(cfg, cfg).lower()}', fontsize=9)
    fig.tight_layout()
    place_legend(fig, h, ncol=2, fontsize=8)
    save(fig, out, 'f11_score_vs_damage_level', marks,
         ['dataset', 'dataset_label', 'damage_kind', 'method', 'method_label', 'model', 'model_label',
          'configuration', 'configuration_label', 'features_from', 'level', 'metric', 'value', 'n_draws',
          'fold', 'source'])


# --------------------------------------------------------------------------- registry

FIGURES = {
    'f1': ('F1', 'Table: hybrid against the best single base representation', fig_f1, L1.INPUTS),
    'f2a': ('F2a', 'Best number of features, per dataset and model', fig_f2a, L2.INPUTS_F2A),
    'f3': ('F3', 'Blueprints: features taken from each representation', fig_f3, L3.INPUTS),
    'f5': ('F5', 'Table: distance against activity difference, single against hybrid', fig_f5, L56.INPUTS_F5),
    'f6': ('F6', 'Table: prediction from nearest neighbours, single against hybrid', fig_f6, L56.INPUTS_F6),
    'f8': ('F8', 'Did the cleaning work: pairs improved, of 8', fig_f8, L8.INPUTS),
    'f9': ('F9', 'The censoring ladder', fig_f9, L911.INPUTS_F9),
    'f10': ('F10', 'Detection against repair, one file per dataset', fig_f10, L911.INPUTS_F10),
    'f11': ('F11', 'Score against damage level', fig_f11, L911.INPUTS_F11),
}
LOADER_CALLS = {
    'f1': lambda r, m, a: L1.load(r, m),
    'f2a': lambda r, m, a: L2.load_f2a(r, m, source=a.f2a_source),
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
    p.add_argument('--datasets', default='promoted',
                   help='datasets drawn by f1, f2a, f5 and f6: "promoted" (default; the datasets with noise_full '
                        'units in ledger.json, a placeholder until the thesis list is fixed), "all", or a,b,c')
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
