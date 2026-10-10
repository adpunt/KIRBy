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
  f2a  best number of features                         c09 curves, c02 per-fold picks, c17 rfe/random
  f3   blueprints, features per representation         c22 allocation_built
  f5   table: distance against activity difference     s02 comparisons (knn distance hybrid)
  f6   table: prediction from nearest neighbours        s05 summary.fixed_ref
  f8   did the cleaning work, three files               n07 n13 n20 n21 n23 n24 n30 n40 (fold 0)
  f9   the censoring ladder                             n30
  f10  detection against cleaning                       n32 (two-learner deletion), n31 break-even
  f11  score against damage level                       n24 (hERG flips), n30 (Caco-2 clipping)

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

# ⛔ NO PATIENT DATA IN THE THESIS (Adelaide, 2026-10-09). TCGA, ANNITIA and NHANES result
# files are dropped from every file search here, before any loader sees them. Every loader
# finds its files through glob.glob, so this one filter covers all of them.
PATIENT_DATA_PREFIXES = ('tcga', 'annitia', 'nhanes')
_glob_unfiltered = glob.glob


def _glob_without_patient_data(pathname, *a, **k):
    return [p for p in _glob_unfiltered(pathname, *a, **k)
            if not os.path.basename(p).lower().startswith(PATIENT_DATA_PREFIXES)]


glob.glob = _glob_without_patient_data

import thesis_figure_style as S  # noqa: E402  (sets rcParams and seaborn style)
import thesis_loader_f1 as L1  # noqa: E402
import thesis_loader_f2 as L2  # noqa: E402
import thesis_loader_f3 as L3  # noqa: E402
import thesis_loader_f56 as L56  # noqa: E402
import thesis_loader_f8 as L8  # noqa: E402
import thesis_loader_f9_11 as L911  # noqa: E402

LAPTOP_ROOT = Path('/Volumes/seagate/kirby_results_2026-10-10')


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


# The signed-off Chapter 5 and 6 datasets (Adelaide, 2026-10-09; THESIS_OUTLINE.md, block
# "CHAPTERS 5 AND 6 DATASETS, SIGNED OFF"). Change it there first, then here.
# Chapter 6 figures draw the same list; a dataset with no similarity result simply has no row.
# Out on purpose: adme_hlm, abcov_kd, cdr_flexibility, abagym_VEGF, and every patient dataset.
THESIS_DATASETS = frozenset({
    # small molecules
    'qm9', 'caco2', 'herg_ki', 'logd', 'herg', 'adme_mdr1', 'adme_rlm',
    # reaction (said in the text; never a single point in a by-domain panel)
    'buchwald_hartwig',
    # antibodies
    'flab_expression_koenig', 'flab_binding_koenig', 'flab_thermo_tresanco', 'abagym_EGFR',
    'ddg_combined', 'ginkgo_titer', 'ginkgo_pr_cho', 'ginkgo_colloidal', 'ginkgo_acsins',
    'ginkgo_hic',
    # one result each
    'abagym_Ang2', 'humanness_therapeutic',
})

# Left out of the Chapter 6 tables F5 and F6 (Adelaide, 2026-10-09): humanness_therapeutic is used
# for one result only and never carries a hybrid-helps row.
CH6_TABLE_EXCLUDE = frozenset({'humanness_therapeutic'})


def thesis_datasets(root, args):
    """The datasets a cross-dataset figure shows, or None for every dataset.

    PLACEHOLDER until the thesis list is fixed (Adelaide, 2026-10-09): by default the
    dataset rows of <results root>/ledger.json whose noise_full stage holds at least one
    unit, i.e. the datasets promoted to further outer folds. --datasets a,b,c names
    them instead; --datasets all shows every dataset."""
    if args.datasets == 'all':
        return None
    if args.datasets == 'thesis':
        return set(THESIS_DATASETS)
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


MAX_BOLD = 4   # at most 4 bold cells per table, headers plain (Adelaide, 2026-10-10)


def _gain(r, metric):
    """Hybrid minus single, signed so that positive means the hybrid is better in the dataset's own
    metric direction, divided by the single score's magnitude. Only used to pick which cells are bold:
    raw differences mix units (Buchwald-Hartwig RMSE is in yield percent, Spearman is unitless), and
    ranked raw, a 0.55-point yield RMSE gain outranks every Spearman gain. The table prints raw scores."""
    d = r['hybrid'] - r['single']
    d = -d if metric in S.LOWER_IS_BETTER else d
    return d / abs(r['single']) if r['single'] else d


def _bold_rows(rows, bold):
    """ids of the rows whose one cell is bold, at most MAX_BOLD.
    bold='better' (rule a, F1): the MAX_BOLD rows with the largest gain of the hybrid over the best
    single representation; the hybrid cell is bold. Only rows where the hybrid is better qualify.
    bold='hybrid_worse' (rule b, F5/F6): rows where the hybrid is worse, the MAX_BOLD largest losses;
    the hybrid cell is bold. Pairs equal at four decimals never count."""
    cand = []
    for r in rows:
        if f"{r['single']:.4f}" == f"{r['hybrid']:.4f}":
            continue
        g = _gain(r, r['metric'])
        if bold == 'better' and g > 0:
            cand.append((-g, id(r)))
        elif bold == 'hybrid_worse' and g < 0:
            cand.append((g, id(r)))
    return {i for _, i in sorted(cand)[:MAX_BOLD]}


def _pair_cells(r, bolded):
    """Two stored scores (single, hybrid); the hybrid cell bold only if the row was picked."""
    fa, fb = f"{r['single']:.4f}", f"{r['hybrid']:.4f}"
    return (fa, rf'\textbf{{{fb}}}') if id(r) in bolded else (fa, fb)


def write_pair_table(out, name, rows, head, caption, label, csv_cols, with_model, bold='better'):
    """Rows grouped by metric, a header line per metric; one stored number per cell; the better of
    at most MAX_BOLD hybrid cells bold, picked by _bold_rows (Adelaide, 2026-10-09: tables replace the
    diagonal plots of F1, F5 and F6; 2026-10-10: at most 4 bold cells, headers plain)."""
    metrics = metric_order(r['metric'] for r in rows)
    ncol = 4 if with_model else 3
    spec = '@{}ll' + 'r' * (ncol - 2) + '@{}' if with_model else '@{}l' + 'r' * (ncol - 1) + '@{}'
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\begin{adjustbox}{max width=\textwidth,max totalheight=0.78\textheight}',
             rf'\begin{{tabular}}{{{spec}}}', r'\toprule',
             ' & '.join(_tex(h) for h in head) + r' \\']
    bolded = _bold_rows(rows, bold)
    for met in metrics:
        lines += [r'\midrule', rf'\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{TEX_METRIC.get(met, _tex(met))}}}}} \\']
        ms = [r for r in rows if r['metric'] == met]
        prev = None
        for ds in dataset_order(r['dataset'] for r in ms):
            for r in sorted([r for r in ms if r['dataset'] == ds],
                            key=lambda r: S.MODEL_ORDER.index(r['model']) if r.get('model') in S.MODEL_ORDER else 99):
                ca, cb = _pair_cells(r, bolded)
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
                     'model, mean over outer folds, in each dataset\'s own metric. Bold: the four rows where the '
                     'hybrid gains most over the best single base representation, as a fraction of the single score.',
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


F2A_NCOL = 3
F2A_LINES = {'hybrid': ('-', 'o', 'Hybrid'), 'rfe': ('--', None, 'Recursive elimination'),
             'random': (':', None, 'Random features')}


def fig_f2a(root, out, args, missing):
    """F2a: one panel per dataset. x = number of features (budget). Lines: held-out score of the
    rank-1 hybrid at each budget (c09), and, where c17 ran, recursive feature elimination and random
    features at the same budgets, one colour per downstream model. Strip under each panel: the budget
    each outer fold's search picked on its training molecules (c02), one tick per fold."""
    if args.f2a_source != 'c09':
        print(f'[f2a] the score curves come from c09; --f2a-source {args.f2a_source} is ignored',
              file=sys.stderr)
    curves = keep_datasets(L2.load_f2a(root, missing, source='c09'), root, args)
    picks = keep_datasets(L2.load_f2a_picks(root, missing), root, args) if args.f2a_marks == 'c02' else []
    other = keep_datasets(L2.load_f2a_c17(root, missing), root, args) if args.f2a_baselines else []
    order = dataset_order(r['dataset'] for r in curves)
    for ds in sorted({r['dataset'] for r in picks} - set(order)):
        missing(f'{ds}: c02 picks exist but no c09 score curve, so no panel')
    # Two files, so each fits one portrait page (Adelaide, 2026-10-10): small molecules with the
    # reaction dataset, and antibodies.
    for slug, doms in F2A_FILES:
        sub = [ds for ds in order if S.DATASET_DOMAIN.get(ds) in doms]
        if not sub:
            missing(f'f2a {slug}: no dataset with a score curve')
            continue
        _draw_f2a(out, f'f02a_best_number_of_features_{slug}', sub, curves, picks, other)


F2A_FILES = [('smallmol', ('Small molecules', 'Reactions')), ('antibodies', ('Antibodies',))]
F2A_ROW_IN = 1.45     # one row of panels: score panel, pick strip, tick labels, title
F2A_TOP_IN, F2A_BOTTOM_IN = 0.12, 0.42


def _draw_f2a(out, name, order, curves, picks, other):
    nrow = -(-len(order) // F2A_NCOL)
    height = F2A_ROW_IN * nrow + F2A_TOP_IN + F2A_BOTTOM_IN
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, height))
    outer = fig.add_gridspec(nrow, F2A_NCOL, hspace=0.85, wspace=0.55, bottom=F2A_BOTTOM_IN / height,
                             top=1 - F2A_TOP_IN / height)
    marks = []
    for i, ds in enumerate(order):
        sub = outer[i // F2A_NCOL, i % F2A_NCOL].subgridspec(2, 1, height_ratios=[3.2, 1.3], hspace=0.08)
        ax, axs = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        cr = [r for r in curves if r['dataset'] == ds]
        models = S.sort_models_by_family({r['model'] for r in cr})
        grid = sorted({v for r in cr for v in r['grid']})
        for r in cr:
            col = S.model_color(r['model'])
            xs = [g for g, v in zip(r['grid'], r['score_by_budget']) if _finite(v)]
            ys = [v for v in r['score_by_budget'] if _finite(v)]
            ax.plot(xs, ys, ls='-', marker='o', ms=2.5, lw=1.1, color=col)
            for g, v, nf in zip(r['grid'], r['score_by_budget'], r['n_folds_by_budget']):
                marks.append(dict(dataset=ds, dataset_label=S.dataset_label(ds), model=r['model'],
                                  model_label=S.model_label(r['model']), metric=r['metric'], line='hybrid',
                                  budget=g, score=v, n_folds=nf, source=r['source']))
            for meth in ('rfe', 'random'):
                o = sorted([x for x in other if x['dataset'] == ds and x['model'] == r['model']
                            and x['method'] == meth], key=lambda x: x['budget'])
                if not o:
                    continue
                ax.plot([x['budget'] for x in o], [x['score'] for x in o], ls=F2A_LINES[meth][0],
                        lw=0.9, color=col)
                for x in o:
                    marks.append(dict(dataset=ds, dataset_label=S.dataset_label(ds), model=r['model'],
                                      model_label=S.model_label(r['model']), metric=r['metric'], line=meth,
                                      budget=x['budget'], score=x['score'], n_folds=x['n_folds'],
                                      fold_scores=x['fold_scores'], source=x['source']))
        # strip: one row per model, one short tick per fold, folds stacked inside the row
        for mi, mo in enumerate(models):
            pk = sorted([p for p in picks if p['dataset'] == ds and p['model'] == mo], key=lambda p: p['fold'])
            seen = {}
            for p in pk:
                k = seen.get(p['picked_budget'], 0)
                seen[p['picked_budget']] = k + 1
                y = mi + (k - 2) * 0.2
                axs.plot([p['picked_budget']], [y], ls='', marker='o', ms=1.6, mew=0,
                         color=S.model_color(mo), clip_on=False)
                marks.append(dict(dataset=ds, dataset_label=S.dataset_label(ds), model=mo,
                                  model_label=S.model_label(mo), metric=cr[0]['metric'], line='fold pick',
                                  budget=p['picked_budget'], fold=p['fold'], n_features=p['n_features'],
                                  source=p['source']))
        met = cr[0]['metric']
        ax.set_title(S.dataset_short(ds), fontsize=8, pad=3)
        ax.set_ylabel(S.metric_label(met), fontsize=6.5, labelpad=2)
        for a_ in (ax, axs):
            a_.set_xscale('log', base=2)
            a_.set_xticks(grid)
            a_.minorticks_off()
            pad = 1.25
            a_.set_xlim(grid[0] / pad, grid[-1] * pad)
            sns_despine(a_)
        ax.set_xticklabels([])
        ax.tick_params(axis='both', labelsize=6, length=2, pad=1)
        axs.set_xticklabels([str(g) for g in grid], rotation=90, fontsize=6)
        axs.tick_params(axis='x', length=2, pad=1)
        axs.set_yticks([])
        axs.set_ylim(len(models) - 0.45, -0.55)
        axs.spines['left'].set_visible(False)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    fig.supxlabel('Number of features (budget)', fontsize=8, y=0.04 / height, va='bottom')
    handles = model_handles({r['model'] for r in curves if r['dataset'] in order})
    for h in handles:
        h.set_ls('-'); h.set_ms(3); h.set_lw(1.1)
    handles += [Line2D([], [], color='0.25', ls=F2A_LINES[k][0], marker=F2A_LINES[k][1], ms=2.5, lw=1.1,
                       label=F2A_LINES[k][2] + (' (held-out mean)' if k == 'hybrid' else ''))
                for k in ('hybrid', 'rfe', 'random') if k == 'hybrid' or any(x['method'] == k and x['dataset'] in order
                                                                              for x in other)]
    if any(p['dataset'] in order for p in picks):
        handles.append(Line2D([], [], color='0.25', ls='', marker='o', ms=2.5, mew=0,
                              label='Budget picked by one fold on its training molecules'))
    leg = fig.legend(handles, [h.get_label() for h in handles], ncol=3, loc='upper center',
                     bbox_to_anchor=(0.5, 0.0), fontsize=6.5, frameon=False,
                     handlelength=2.6, columnspacing=1.2)
    save(fig, out, name, marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'metric', 'line', 'budget', 'score', 'n_folds',
          'fold', 'n_features', 'fold_scores', 'source'])


# --------------------------------------------------------------------------- F3

def _fig_f3(root, out, args, missing):
    """F3: one horizontal bar per dataset, segments = features taken from each representation."""
    rows = L3.load_f3(root, missing)
    # Chapter 5 datasets only; antibody humanness has its one named result elsewhere (Adelaide, 2026-10-10).
    rows = [r for r in rows if r['dataset'] in THESIS_DATASETS - F3_EXCLUDE]
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
        n_leg = len(reps) + any(rp in S.BUILT_FOR_THESIS_REPS for rp in reps)
        heights += [0.3 * len(ds_dom) + 0.5, 0.2 * -(-n_leg // ncol) + (0.9 if last else 0.55)]
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, sum(heights)))
    # Left margin sized to the longest 'short name · model' label, so the file prints at the text width.
    lab_in = max(len(S.dataset_short(d).replace('$', '').replace('_', '').replace('\\Delta', 'D')) +
                 len(S.model_label(next(m['model'] for m in marks if m['dataset'] == d))) for d in order) * 0.085 + 0.45
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.05, left=lab_in / S.TEXTWIDTH_IN,
                          right=0.97)
    xmax = max(sum(m['n_features'] for m in marks if m['dataset'] == ds) for ds in order)
    for bi, (dom, ds_dom, reps, colors) in enumerate(blocks):
        ax = fig.add_subplot(gs[2 * bi])
        for i, ds in enumerate(ds_dom):
            left = 0
            for m in [m for m in marks if m['dataset'] == ds]:
                built = m['representation'] in S.BUILT_FOR_THESIS_REPS
                ax.barh(i, m['n_features'], left=left, color=colors[m['representation']], height=0.7,
                        edgecolor='white', lw=0.3, hatch=F3_HATCH if built else None)
                m['built_for_thesis'] = built
                m['segment_start'] = left
                m['segment_color'] = colors[m['representation']]
                left += m['n_features']
        ax.set_yticks(range(len(ds_dom)))
        ax.set_yticklabels([f"{S.dataset_short(d)} · {S.model_label(next(m['model'] for m in marks if m['dataset'] == d))}"
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
        h = [Patch(facecolor=colors[rp], edgecolor='white', lw=0.3, label=S.rep_label(rp),
                   hatch=F3_HATCH if rp in S.BUILT_FOR_THESIS_REPS else None) for rp in reps]
        if any(rp in S.BUILT_FOR_THESIS_REPS for rp in reps):
            h.append(Patch(facecolor='white', edgecolor='#444444', lw=0.5, hatch=F3_HATCH,
                           label='built for this thesis'))
        lax.legend(S._to_colmajor(h, ncol), S._to_colmajor([x.get_label() for x in h], ncol), ncol=ncol,
                   loc='lower center' if bi == len(blocks) - 1 else 'center', fontsize=7,
                   handlelength=1.0, columnspacing=1.0)
    save(fig, out, 'f03_blueprints', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'fold', 'slot', 'hybrid_label',
          'representation', 'representation_label', 'n_features', 'segment_start', 'segment_color',
          'total_features', 'built_for_thesis', 'source'])
    small = next((b for b in blocks if b[0] == 'Small molecules'), None)
    fig_f3_detail(root, out, args, missing, small[3] if small else {})


F3_EXCLUDE = frozenset({'humanness_therapeutic'})
# Hatch for the representations built for this thesis. White lines on the segment colour.
F3_HATCH = '////'


def fig_f3(root, out, args, missing):
    """F3 and F3b, with white hatch lines set for these two figures only."""
    with plt.rc_context({'hatch.color': 'white', 'hatch.linewidth': 0.6}):
        _fig_f3(root, out, args, missing)


def fig_f3_detail(root, out, args, missing, colors):
    """F3b: the features one model's LogD hybrid picked on every outer fold, Mordred descriptors
    and ECFP4 bits in named groups, every other representation as one bar."""
    ds, model = args.f3_detail_dataset, args.f3_detail_model
    rows = L3.load_f3_detail(root, ds, missing, model=model)
    if not rows:
        missing(f'{ds} {model}: no features to draw for the detail figure')
        return
    n_folds = rows[0]['folds_total']
    counts = {}
    for r in rows:
        key = (r['representation'], r['group'] or S.rep_label(r['representation']))
        counts.setdefault(key, []).append(r)
    sections = []
    for rep in ('mordred', 'ecfp4'):
        items = sorted(((g, rs) for (rp, g), rs in counts.items() if rp == rep), key=lambda x: -len(x[1]))
        other = [x for x in items if x[0].startswith('Other')]
        items = [x for x in items if not x[0].startswith('Other')] + other
        if items:
            n = sum(len(rs) for _, rs in items)
            sections.append((f'{S.rep_label(rep)}: {n} {"descriptors" if rep == "mordred" else "bits"}', rep, items))
    rest = sorted(((S.rep_label(rp), rs) for (rp, g), rs in counts.items() if rp not in ('mordred', 'ecfp4')),
                  key=lambda x: -len(x[1]))
    if rest:
        sections.append((f'Other representations, unnamed: {sum(len(rs) for _, rs in rest)} features', None, rest))
    nrows = sum(len(it) for _, _, it in sections)
    fig, ax = plt.subplots(figsize=(S.TEXTWIDTH_IN, 0.2 * (nrows + 2 * len(sections)) + 0.6))
    y, yt, yl, marks = 0, [], [], []
    xmax = max(len(rs) for _, _, it in sections for _, rs in it)
    for title, rep, items in sections:
        ax.text(-0.01, y - 0.15, title, transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=9, fontweight='bold')
        y += 1
        for g, rs in items:
            rp = rs[0]['representation']
            c = colors.get(rp, '#888888')
            ax.barh(y, len(rs), color=c, height=0.7, edgecolor='white', lw=0.3)
            ax.text(len(rs) + xmax * 0.01, y, str(len(rs)), va='center', fontsize=7)
            yt.append(y)
            yl.append(g)
            marks.append({'dataset': ds, 'dataset_label': S.dataset_label(ds), 'model': model,
                          'model_label': S.model_label(model), 'representation': rp,
                          'representation_label': S.rep_label(rp), 'group': g, 'n_features': len(rs),
                          'folds_required': n_folds, 'members': [r['name'] or r['position'] for r in rs],
                          'positions': [r['position'] for r in rs], 'segment_color': c,
                          'source': rs[0]['source']})
            y += 1
        y += 0.6
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8)
    ax.set_ylim(y - 0.6, -0.8)
    ax.set_xlim(0, xmax * 1.1)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(f'Features picked on all {n_folds} outer folds', loc='right')
    sns_despine(ax)
    fig.tight_layout()
    save(fig, out, f'f03b_blueprint_{ds}', marks,
         ['dataset', 'dataset_label', 'model', 'model_label', 'representation', 'representation_label',
          'group', 'n_features', 'folds_required', 'members', 'positions', 'segment_color', 'source'])


# --------------------------------------------------------------------------- F5 / F6

def _single_label(v):
    """One name per fold; a fold whose result names no representation reads 'none stored'."""
    if isinstance(v, list):
        return '; '.join(S.rep_label(x) if x else 'none stored' for x in v)
    return S.rep_label(v) if v else ''


def _pair_table_by_dataset(root, out, args, missing, rows, name, head, caption, label, extra_cols, bold='better'):
    marks = []
    for r in keep_datasets(rows, root, args):
        if r['dataset'] in CH6_TABLE_EXCLUDE:
            continue
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
                     + extra_cols + ['source'], with_model=False, bold=bold)


def fig_f5(root, out, args, missing):
    """F5, now a table: Spearman of distance against activity difference, best single base
    representation against the hybrid built for distance."""
    _pair_table_by_dataset(root, out, args, missing, L56.load_f5(root, missing), 'tab_distance_tracks_activity',
                           ['Dataset', 'Best single base representation', 'Hybrid built for distance'],
                           'Spearman correlation between the distance between two samples and the difference in '
                           'their labels, per dataset, mean over outer folds. Bold: a dataset where the hybrid is '
                           'lower, at most four.',
                           'tab:distance_tracks_activity', ['hybrid_kind'], bold='hybrid_worse')


def fig_f6(root, out, args, missing):
    """F6, now a table: prediction from the five nearest neighbours, best single base representation
    against the hybrid, at the setting fixed before the run."""
    _pair_table_by_dataset(root, out, args, missing, L56.load_f6(root, missing), 'tab_nearest_neighbour_prediction',
                           ['Dataset', 'Best single base representation', 'Hybrid'],
                           'Held-out score of predicting each molecule from its five nearest training molecules, '
                           'per dataset, mean over outer folds, in each dataset\'s own metric. Bold: a dataset where '
                           'the hybrid is worse, at most four.',
                           'tab:nearest_neighbour_prediction', ['model'], bold='hybrid_worse')


# --------------------------------------------------------------------------- F8

# Damage kinds in the words of the Chapter 7 damage table (tab:noise:kinds).
DAMAGE_WORDS = {
    'gaussian': 'Gaussian', 'outlier_p05': 'outlier', 'grouped_shifted': 'grouped-shifted', 'censoring': 'clipped',
    'shift_only_p10': 'shift in 10%', 'sparse_dprime_up_p10': 'shift in 10%, model-sized',
    'dense_oofmse': 'Gaussian, model-sized',
    'uniform': 'uniform flips', 'class_imbalance': 'rarer-class flips', 'binary_asymmetric': 'one-way flips',
    'instance_noise': 'per-molecule flips', 'class_dependent': 'class-dependent flips',
    'confusion_directed': 'confusion-matrix flips',
}
F8_KIND_ORDER = list(L8.KINDS)
F8_SLUG = {'detector': 'detector', 'flag': 'known_flag', 'training': 'noise_robust_training'}
F8_TITLE = {'detector': 'Mitigation, molecules selected by a detector',
            'flag': 'Mitigation, molecules selected by a known flag:\nthe label sits at the assay limit',
            'training': 'Noise-robust training: the model changes, the labels do not'}


def f8_row_key(r):
    return r['row_key']


F8_ROW_IN = 0.1         # one dataset-and-condition row, inches
F8_COL_IN = 0.2         # one method column, inches (room for a two-line rotated name)
F8_ACT_GAP = 0.6        # gap between action groups, in columns
F8_BLOCK_GAP_IN = 0.08  # between row blocks, above each block title
F8_BLOCK_TITLE_IN = 0.17
F8_KEY_IN = 1.25        # key to the right of the grid
F8_FS = 6               # row labels and method names


def _text_w_in(fig, text, fontsize, **kw):
    """Printed width of one text, inches (measured, so mathtext counts right)."""
    t = fig.text(0, 0, text, fontsize=fontsize, **kw)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def fig_f8(root, out, args, missing):
    """F8, one file per group of methods (Chapter 7): mitigation selected by a detector, mitigation selected
    by a known flag, noise-robust training. Columns = signed-off DEEP methods, under their action; rows =
    labels as delivered, injected damage at the main level (one row per dataset and kind), and QM9 with no
    noise added, set apart. Shade = share of model-and-configuration pairs improved (n in the CSV). Dot =
    beat its control in at least half its pairs; cross = did not; no mark = no control. Hatched = no run.
    Portrait since 2026-10-10 (the detector file was 10 in wide with datasets as columns)."""
    import textwrap
    rows = L8.load_f8(root, missing)
    if args.f8_rows:
        want = [ln.strip() for ln in open(args.f8_rows) if ln.strip() and not ln.startswith('#')]
        rows = [r for r in rows if r['row_key'] in set(want)]
    cmap = plt.get_cmap('Purples')
    for block in ('detector', 'flag', 'training'):
        br = [r for r in rows if r['block'] == block]
        if not br:
            missing(f'f8 {block}: no rows')
            continue
        cell = {(r['row_key'], r['dataset'], r['grid']): r for r in br}
        # columns: methods in the order of L8.ROWS, grouped by action, a gap between groups
        order = [s for s in L8.ROWS if s['block'] == block and L8.row_key(s) in {r['row_key'] for r in br}]
        cols, xpos, act_spans, x = [], [], [], 0.0
        for act in L8.ACTION_ORDER:
            specs = [s for s in order if s['action'] == act]
            if not specs:
                continue
            if cols:
                x += F8_ACT_GAP
            act_spans.append((act, x, x + len(specs)))
            for s_ in specs:
                cols.append(s_)
                xpos.append(x)
                x += 1
        xmax = x
        # row blocks
        deliv = [(ds, 'delivered') for ds in dataset_order(r['dataset'] for r in br if r['grid'] == 'delivered')]
        dmg = {(r['dataset'], r['grid']) for r in br if r['grid'].startswith('damage:')}
        dorder = dataset_order(ds for ds, _ in dmg)
        dmg = sorted(dmg, key=lambda c: (dorder.index(c[0]), F8_KIND_ORDER.index(c[1][7:])
                                         if c[1][7:] in F8_KIND_ORDER else 99))
        qm9 = [('qm9', 'qm9')] if any(r['grid'] == 'qm9' for r in br) else []
        groups = [(g, t) for g, t in ((deliv, 'As delivered'), (dmg, 'Injected damage, main level'),
                                      (qm9, 'QM9, no noise added')) if g]

        def row_name(ds, grid):
            if grid == 'qm9':
                return 'QM9'
            if grid == 'delivered':
                return S.dataset_short(ds)
            return S.dataset_short(ds) + ', ' + DAMAGE_WORDS.get(grid[7:], grid[7:])

        def col_full(ds, grid):
            if grid == 'qm9':
                return 'QM9 HOMO-LUMO gap'
            if grid == 'delivered':
                return S.dataset_label(ds)
            return S.dataset_label(ds) + ', ' + DAMAGE_WORDS.get(grid[7:], grid[7:])

        heads = [textwrap.fill(s_['words'], 30) for s_ in cols]
        # measure on a scratch figure
        tmp = plt.figure()
        label_in = max(_text_w_in(tmp, row_name(*c), F8_FS)[0] for g, _ in groups for c in g) + 0.08
        label_in = max(label_in, max(_text_w_in(tmp, t, F8_FS + 0.5, fontweight='bold')[0] for _, t in groups) - 0.2)
        head_in = max(_text_w_in(tmp, h, F8_FS, rotation=90)[1] for h in heads) + 0.06
        title_lines = F8_TITLE[block].count('\n') + 1
        plt.close(tmp)
        act_in = 0.2 if block != 'training' else 0.0
        top_in = 0.12 + 0.16 * title_lines + 0.08 + act_in + head_in
        grid_w = xmax * F8_COL_IN
        width = min(S.TEXTWIDTH_IN, max(label_in + grid_w + F8_KEY_IN + 0.1, 3.4))
        nrows_total = sum(len(g) for g, _ in groups)
        grid_h = nrows_total * F8_ROW_IN + len(groups) * (F8_BLOCK_GAP_IN + F8_BLOCK_TITLE_IN)
        key = []
        if any(r['has_control'] for r in br):
            key += ['●  beat its control in at\n    least half its pairs', '✕  did not', 'no mark: no control']
        key += ['hatched: not run at\nthis level']
        key_h = sum(0.11 * (k.count('\n') + 1) + 0.05 for k in key)
        height = top_in + max(grid_h, F8_BLOCK_GAP_IN + F8_BLOCK_TITLE_IN + 1.3 + 0.25 + key_h) + 0.08
        fig = plt.figure(figsize=(width, height))
        x0 = label_in / width
        y_in = height - top_in        # top of the next block, inches from the bottom
        marks, axes = [], []
        for gi, (rws, title) in enumerate(groups):
            y_in -= F8_BLOCK_GAP_IN + F8_BLOCK_TITLE_IN
            h_in = len(rws) * F8_ROW_IN
            ax = fig.add_axes([x0, (y_in - h_in) / height, grid_w / width, h_in / height])
            axes.append(ax)
            grey = rws is qm9
            fig.text(0.04 / width, (y_in + 0.03) / height, title, fontsize=F8_FS + 0.5, va='bottom', ha='left',
                     fontweight='normal' if grey else 'bold', color='#555555' if grey else 'black',
                     bbox=dict(boxstyle='square,pad=0.2', fc='#E3E3E3', ec='none') if grey else None)
            y_in -= h_in
            for i, (ds, grid) in enumerate(rws):
                for j, spec in enumerate(cols):
                    xj = xpos[j]
                    r = cell.get((L8.row_key(spec), ds, grid))
                    if r is None:
                        ax.add_patch(plt.Rectangle((xj, i), 1, 1, facecolor='white', edgecolor='#BBBBBB',
                                                   hatch='////', lw=0))
                        continue
                    v = r['share_improved']
                    ax.add_patch(plt.Rectangle((xj, i), 1, 1, facecolor=cmap(0.04 + 0.96 * v), edgecolor='white',
                                               lw=0.4))
                    ink = 'white' if v >= 0.6 else 'black'
                    if r['control_mark'] == 'dot':
                        ax.plot(xj + 0.5, i + 0.5, 'o', ms=2.0, color=ink, mec=ink, mew=0)
                    elif r['control_mark'] == 'cross':
                        ax.plot(xj + 0.5, i + 0.5, 'x', ms=2.4, color=ink, mew=0.6)
                    marks.append(dict(r, column=col_full(ds, grid), row_name=row_name(ds, grid), shade_value=v))
            ax.set_xlim(0, xmax)
            ax.set_ylim(len(rws), 0)
            ax.set_yticks([i + 0.5 for i in range(len(rws))])
            ax.set_yticklabels([row_name(*c) for c in rws], fontsize=F8_FS)
            ax.set_xticks([])
            ax.tick_params(length=0, pad=1.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
        # method names on top of the first block, action labels above them
        ax0 = axes[0]
        ax0.xaxis.set_ticks_position('top')
        ax0.set_xticks([xp + 0.5 for xp in xpos])
        ax0.set_xticklabels(heads, rotation=90, fontsize=F8_FS, va='bottom', ha='center', linespacing=1.0)
        ax0.tick_params(axis='x', length=0, pad=1.5)
        fig.suptitle(F8_TITLE[block], x=0.04 / width, ha='left', va='top', fontsize=8, y=1 - 0.08 / height)
        # Method names hang from the top of the first block's title line, so push them up past it.
        ax0.tick_params(axis='x', pad=1.5 + (F8_BLOCK_TITLE_IN + F8_BLOCK_GAP_IN) * 72)
        head_base_in = height - top_in + 0.02
        if block != 'training':
            for act, a, b in act_spans:
                xa = (label_in + a * F8_COL_IN) / width
                xb = (label_in + b * F8_COL_IN) / width
                yy = (head_base_in + head_in + 0.04) / height
                fig.add_artist(Line2D([xa + 0.01 / width, xb - 0.01 / width], [yy, yy], color='#444444', lw=0.6))
                lab = act if (b - a) * F8_COL_IN >= _text_w_in(fig, act, F8_FS + 0.5)[0] else act.replace(' ', '\n', 1) \
                    if (b - a) * F8_COL_IN >= 0.3 else act
                fig.text((xa + xb) / 2, yy + 0.02 / height, lab, ha='center', va='bottom', fontsize=F8_FS + 0.5,
                         fontweight='bold', linespacing=0.95)
        # key to the right of the grid
        kx_in = label_in + grid_w + 0.2
        ky_top = height - top_in - F8_BLOCK_GAP_IN - F8_BLOCK_TITLE_IN
        cax = fig.add_axes([kx_in / width, (ky_top - 1.3) / height, 0.09 / width, 1.3 / height])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.04 / 0.96, 1.0)), cax=cax,
                          orientation='vertical', ticks=[0, 0.25, 0.5, 0.75, 1])
        cb.ax.set_ylim(0, 1)
        cb.ax.tick_params(labelsize=F8_FS, length=2, pad=1)
        cb.set_label('Share of model-and-\nconfiguration pairs\nimproved', fontsize=F8_FS, labelpad=3, linespacing=1.0)
        kt = ky_top - 1.3 - 0.25
        for line in key:
            fig.text((kx_in - 0.05) / width, kt / height, line, fontsize=F8_FS, va='top', ha='left', linespacing=1.0)
            kt -= 0.11 * (line.count('\n') + 1) + 0.05
        if kt < 0:
            print(f'[f8] {block}: key runs below the grid by {-kt:.2f} in', file=sys.stderr)
        stem = f'f08_did_the_cleaning_work_{F8_SLUG[block]}'
        png = out / f'{stem}.png'
        fig.savefig(png, dpi=300)      # fixed size: the file is the size it prints at
        plt.close(fig)
        write_csv(out / f'{stem}.csv', marks,
                  ['row_key', 'block', 'action', 'row_label', 'experiment', 'method', 'dataset', 'grid',
                   'damage_kind', 'column', 'row_name', 'level', 'metric', 'lower_is_better', 'n_pairs',
                   'n_pairs_improved', 'share_improved', 'has_control', 'control', 'n_control_pairs',
                   'n_control_beaten', 'control_mark', 'control_pairs', 'pairs', 'pairs_improved', 'pair_settings',
                   'settings_left_out_fewer_draws', 'n_draw_copies', 'source'])
        print(f'wrote {png} ({width:.2f} x {height:.2f} in, {len(marks)} cells)')


# --------------------------------------------------------------------------- F9

# Colour = what the model was fitted on: base representations orange and yellow, hybrids
# blue and green (Adelaide, 2026-10-09). Shape = model. CLEAN_COLOR is not used here.
CONFIG_COLORS = {'base_1': '#E69F00', 'base_2': '#DDCC77', 'hybrid': '#4477AA', 'hybrid__general2': '#228833'}
F9_MODEL_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
STEPS = [('delivered', 'Labels as\ndelivered'), ('clipped', 'After\nclipping'),
         ('corrected', 'Clipped,\ncorrected'), ('deleted', 'Clipped\nmolecules\ndeleted')]
CORRECTION_WORDS = {
    'constant_lift': 'Clipped labels\nraised by\none constant', 'per_molecule_lift': 'Clipped labels\nraised per\nmolecule',
    'truncated_normal_mean': 'Truncated-normal\nmean', 'censored_em': 'Censored\nEM',
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
    # Two panels per row: four step names do not fit under a quarter of the page width.
    ncol = min(2, len(datasets))
    nrow = -(-len(datasets) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.TEXTWIDTH_IN, 3.2 * nrow), squeeze=False)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    for ax, ds in zip(axes.flat, datasets):
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
        ax.set_title(S.dataset_short(ds))
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


F10_SHARE_COLORS = {0.05: '#1b9e77', 0.1: '#d95f02', 0.2: '#7570b3'}
F10_SHARE_MARKERS = {0.05: 'o', 0.1: 's', 0.2: '^'}


def fig_f10(root, out, args, missing):
    """F10: x = precision of the molecules the two-learner deletion removed, y = held-out score
    after removing them; one dot per flagged share; one panel per damage kind; one file per
    dataset; one model-and-configuration pair. Vertical line per share: n31's break-even
    precision for deleting a flagged set of that share, same dataset, damage kind, level,
    model and configuration, where n31 has one."""
    rows = one_fold(L911.load_f10(root, missing), args.noise_fold, 'f10')
    be = L911.load_f10_breakeven(root, missing)
    model, cfg = args.f10_pair.split('@')
    keep = set(args.f10_datasets.split(',')) if args.f10_datasets else None
    rows = [r for r in rows if r.get('model') == model and r.get('configuration') == cfg
            and (keep is None or r['dataset'] in keep)]
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
    marks = []
    for r in rows:
        if chosen.get((r['dataset'], r['damage_kind'])) != r['level']:
            continue
        if not (_finite(r.get('precision')) and _finite(r.get('after_value'))):
            missing(f"{r['dataset']} {r['damage_kind']} {r['method']}: precision or score absent")
            continue
        b = be.get((r['dataset'], r['damage_kind'], r['level'], cfg, model, r['flag_share']))
        if r['damage_kind'] in L911.F10_DAMAGE_WITH_BREAKEVEN and b is None:
            missing(f"{r['dataset']} {r['damage_kind']} level {r['level']:g} share {r['flag_share']:g}: "
                    f"no n31 break-even for this dataset, kind and level")
        elif r['damage_kind'] not in L911.F10_DAMAGE_WITH_BREAKEVEN:
            missing(f"{r['dataset']} {r['damage_kind']}: n31 does not run this damage kind, no break-even line")
        marks.append(dict(r, dataset_label=S.dataset_label(r['dataset']), model_label=S.model_label(model),
                          configuration_label=S.CONFIG_LABELS.get(cfg, cfg),
                          break_even_precision=(b or {}).get('value'),
                          break_even_refused_because=(b or {}).get('refused'),
                          break_even_source=(b or {}).get('source'),
                          dot_color=F10_SHARE_COLORS[r['flag_share']]))
    from matplotlib.ticker import MaxNLocator
    for ds in dataset_order(ds for ds, _ in chosen):
        ds_marks = [m for m in marks if m['dataset'] == ds]
        kinds = [k for k in F8_KIND_ORDER if any(m['damage_kind'] == k for m in ds_marks)]
        kinds += sorted({m['damage_kind'] for m in ds_marks} - set(kinds))
        if not kinds:
            continue
        ncol = min(2, len(kinds))
        nrow = -(-len(kinds) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(S.TEXTWIDTH_IN if ncol == 2 else S.HALFWIDTH_IN + 0.6,
                                                      2.6 * nrow + 0.3), squeeze=False)
        for ax in axes.flat[len(kinds):]:
            ax.set_visible(False)
        any_line = False
        xmax = max([m['precision'] for m in ds_marks] +
                   [m['break_even_precision'] for m in ds_marks if _finite(m.get('break_even_precision'))] + [0.1])
        for ax, kind in zip(axes.flat, kinds):
            ms = sorted((m for m in ds_marks if m['damage_kind'] == kind), key=lambda m: m['flag_share'])
            for m in ms:
                ax.scatter(m['precision'], m['after_value'], color=m['dot_color'],
                           marker=F10_SHARE_MARKERS[m['flag_share']], s=30, lw=0, zorder=3)
                if _finite(m.get('break_even_precision')):
                    ax.axvline(m['break_even_precision'], color=m['dot_color'], ls='-', lw=1.0, alpha=0.8, zorder=1)
                    any_line = True
            for v in {m['no_cleaning_value'] for m in ms if _finite(m.get('no_cleaning_value'))}:
                ax.axhline(v, color='#555555', ls='--', lw=0.8, zorder=2)
            for v in {m['clean_value'] for m in ms if _finite(m.get('clean_value'))}:
                ax.axhline(v, color=S.CLEAN_COLOR, ls=':', lw=0.9, zorder=2)
            refused = [f"{m['flag_share'] * 100:g}%" for m in ms
                       if kind in L911.F10_DAMAGE_WITH_BREAKEVEN and m.get('break_even_source')
                       and not _finite(m.get('break_even_precision'))]
            ax.set_xlim(0, min(1.0, xmax * 1.1 + 0.02))
            met = ms[0]['metric'] if ms else 'rmse'
            word = DAMAGE_WORDS.get(kind, kind)
            note = '' if all(m['unit_finished'] for m in ms) else \
                f"\nunfinished run, {min(m['n_draws'] for m in ms)} draw{'s' if min(m['n_draws'] for m in ms) > 1 else ''}"
            if refused:   # in the title, so it can never sit on a dot
                note += ('\n' if not note else '; ') + 'no break-even at ' + ', '.join(refused)
            ax.set_title(f'{S.cap(word)}, level {chosen[(ds, kind)]:g}{note}', fontsize=8.5)
            ax.set_xlabel('Precision of the deleted molecules', fontsize=8.5)
            ax.set_ylabel(f'After cleaning, {S.metric_label(met)}', fontsize=8.5)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(labelsize=7.5)
            sns_despine(ax)
        fig.suptitle(f'{S.dataset_short(ds)} · {S.model_label(model)} on the {S.CONFIG_LABELS.get(cfg, cfg).lower()}',
                     fontsize=10)
        fig.tight_layout()
        shares = sorted({m['flag_share'] for m in ds_marks})
        left = [Line2D([], [], ls='', marker=F10_SHARE_MARKERS[s_], color=F10_SHARE_COLORS[s_], ms=5,
                       label=f'Two learners, delete, {s_ * 100:g}% flagged') for s_ in shares]
        right = [Line2D([], [], ls='-', color='#777777', label='Break-even precision (color = share)')] \
            if any_line else []
        right += [Line2D([], [], ls='--', color='#555555', label='Damaged labels, no cleaning'),
                  Line2D([], [], ls=':', color=S.CLEAN_COLOR, label='Clean labels')]
        if ncol == 1:
            h = left + right
        else:   # row-major for place_legend: left column the shares, right column the lines
            h = [x for pair in zip(left + [None] * (len(right) - len(left)), right + [None] * (len(left) - len(right)))
                 for x in pair if x is not None]
        place_legend(fig, h, ncol=1 if ncol == 1 else 2, fontsize=7)
        save(fig, out, f'f10_detection_vs_cleaning_{ds}', ds_marks,
             ['dataset', 'dataset_label', 'damage_kind', 'level', 'detector', 'action', 'method', 'flag_share',
              'precision', 'removed_mean', 'removed_hits_mean', 'training_molecules_mean', 'molecules_moved_mean',
              'average_precision', 'detection_auroc', 'after_value', 'no_cleaning_value', 'clean_value',
              'break_even_precision', 'break_even_refused_because', 'break_even_source', 'metric', 'model',
              'model_label', 'configuration', 'configuration_label', 'dot_color', 'n_draws', 'fold',
              'unit_finished', 'source'])


# --------------------------------------------------------------------------- F11

F11_COLORS = {
    'no_cleaning': '#000000', 'delete': '#D55E00', 'label_smooth': '#009E73', 'downweight': '#CC79A7',
    'best_random_deletion': '#999999',
    'censored_em': '#E69F00', 'constant_lift': '#56B4E9', 'drop_clipped': '#D55E00',
}
F11_ORDER = ['no_cleaning', 'delete', 'label_smooth', 'downweight', 'best_random_deletion',
             'censored_em', 'constant_lift', 'drop_clipped']
F11_WORDS = {
    'no_cleaning': 'No cleaning', 'delete': 'Delete flagged molecules',
    'label_smooth': "Move flagged labels toward neighbors' mean",
    'downweight': 'Down-weight by suspicion score (reads no share)',
    'best_random_deletion': 'Best of five random deletions, same number',
    'censored_em': 'Censored EM on clipped labels', 'constant_lift': 'Raise clipped labels by one constant',
    'drop_clipped': 'Delete clipped molecules',
}
F11_LEGEND_MERGE = {'delete': 'Delete flagged (HB) or clipped (Caco-2) molecules', 'drop_clipped': None}
F11_SPACE_LS = {'hybrid': '-', 'base_1': '--'}
F11_SPACE_WORDS = {'hybrid': 'Top hybrid', 'base_1': 'Best single representation'}
F11_FLIP_WORDS = {'confusion_directed': 'along a confusion matrix', 'uniform': 'uniformly',
                  'class_dependent': 'by class', 'class_imbalance': 'by class frequency',
                  'instance_noise': 'at a rate per molecule', 'binary_asymmetric': 'in one direction'}


def fig_f11(root, out, args, missing):
    """F11: one panel per damage kind. hERG label flips (n24): held-out AUROC against the share
    of training molecules flagged, at the one damaged level the file holds, with the undamaged
    labels as a reference line. Caco-2 clipping (n30): held-out RMSE against the clipped share.
    Top hybrid solid, best single representation dashed."""
    panels = args.f11_panels.split(',')
    rows = L911.load_f11(root, missing, panels=panels, flip_model=args.f11_flip_model,
                         flip_detector=args.f11_flip_detector, clip_model=args.f11_clip_model,
                         fold=args.noise_fold)
    for r in rows:
        r.update(dataset_label=S.dataset_label(r['dataset']), model_label=S.model_label(r['model']),
                 configuration_label=F11_SPACE_WORDS.get(r['configuration'], r['configuration']))
    marks = []
    fig, axes = plt.subplots(1, len(panels), figsize=(S.TEXTWIDTH_IN, 2.9), squeeze=False)
    used = set()
    for ax, panel in zip(axes[0], panels):
        ds, kind = panel.split(':')
        pr = [r for r in rows if r['panel'] == panel]
        if kind != 'censoring':
            damaged = sorted({r['damage_level'] for r in pr if r['damage_level'] > 0})
            lvl = damaged[0] if damaged else None
            ref = {r['configuration']: r for r in pr if r['damage_level'] == 0 and r['method'] == 'no_cleaning'}
            draw = [r for r in pr if r['damage_level'] == lvl]
            xs_all = sorted({r['x'] for r in draw})
            for sp, r in ref.items():
                ax.axhline(r['value'], color=S.CLEAN_COLOR, ls=F11_SPACE_LS[sp], lw=0.9, alpha=0.8, zorder=1)
                marks.append(dict(r, role='undamaged labels, reference line'))
                used.add('undamaged')
        else:
            draw = pr
            xs_all = sorted({r['x'] for r in draw})
            lvl = None
        for sp in ('base_1', 'hybrid'):
            for meth in F11_ORDER:
                pts = {}
                for r in draw:
                    if r['configuration'] == sp and r['method'] == meth:
                        pts[r['x']] = r
                if not pts:
                    continue
                if meth == 'no_cleaning' and kind != 'censoring':
                    r = pts[0.0]
                    ax.axhline(r['value'], color=F11_COLORS[meth], ls=F11_SPACE_LS[sp], lw=1.0, zorder=2)
                    ax.plot([0.0], [r['value']], color=F11_COLORS[meth], marker='o', ms=3.5, zorder=3)
                    marks.append(dict(r, role='no cleaning, line across the panel'))
                    used.add(meth)
                    continue
                want = [x for x in xs_all if x > 0] if kind != 'censoring' else xs_all
                for x in want:
                    if x not in pts:
                        missing(f'F11 {panel} {sp} {meth}: no score at x={x}; point left out')
                xs = sorted(pts)
                ax.plot(xs, [pts[x]['value'] for x in xs], color=F11_COLORS[meth], ls=F11_SPACE_LS[sp],
                        marker='o', ms=3.2, lw=1.2, zorder=3)
                marks += [dict(pts[x], role='line point') for x in xs]
                used.add(meth)
        met = draw[0]['metric'] if draw else ''
        model = draw[0]['model'] if draw else ''
        if kind == 'censoring':
            ax.set_title(f'{S.dataset_short(ds)} · {S.model_label(model)}\nlabels clipped\nat the top value', fontsize=8)
            ax.set_xlabel('Labels clipped (%)', fontsize=8)
        else:
            lv = f'{lvl * 100:g}% of labels ' if lvl is not None else ''
            ax.set_title(f'{S.dataset_short(ds)} · {S.model_label(model)}\n{lv}flipped\n'
                         f'{F11_FLIP_WORDS.get(kind, kind)}', fontsize=8)
            ax.set_xlabel('Molecules flagged (%)', fontsize=8)
        ax.set_xticks(xs_all)
        ax.set_xticklabels([f'{x * 100:g}' for x in xs_all])
        ax.set_ylabel(f'Held-out {S.metric_label(norm_metric(met))}', fontsize=8)
        ax.tick_params(labelsize=7.5)
        sns_despine(ax)
    fig.tight_layout(w_pad=1.2)
    merge = 'delete' in used and 'drop_clipped' in used

    def mh(m):
        return Line2D([], [], color=F11_COLORS[m], marker='o', ms=3.2, lw=1.2,
                      label=(F11_LEGEND_MERGE.get(m) or F11_WORDS[m]) if merge else F11_WORDS[m])
    left = [mh(m) for m in ('no_cleaning', 'delete', 'label_smooth', 'downweight', 'best_random_deletion')
            if m in used]
    right = [mh(m) for m in ('censored_em', 'constant_lift') if m in used]
    if 'drop_clipped' in used and not merge:
        right.append(mh('drop_clipped'))
    right += [Line2D([], [], color='#444444', ls=F11_SPACE_LS[sp], lw=1.2, label=F11_SPACE_WORDS[sp])
              for sp in ('hybrid', 'base_1')]
    if 'undamaged' in used:
        right.append(Line2D([], [], color=S.CLEAN_COLOR, lw=0.9, label='Undamaged labels, no cleaning'))
    n = max(len(left), len(right))
    left += [Line2D([], [], ls='', label=' ')] * (n - len(left))
    right += [Line2D([], [], ls='', label=' ')] * (n - len(right))
    h = [x for pair in zip(left, right) for x in pair]
    place_legend(fig, h, ncol=2, fontsize=7)
    save(fig, out, 'f11_score_vs_damage_level', marks,
         ['panel', 'dataset', 'dataset_label', 'damage_kind', 'damage_level', 'x', 'x_meaning', 'method',
          'method_label', 'role', 'rate_blind', 'model', 'model_label', 'configuration', 'configuration_label',
          'detector', 'metric', 'lower_is_better', 'value', 'n_draws', 'n_seeds', 'best_seed', 'fold', 'source'])


# --------------------------------------------------------------------------- registry

FIGURES = {
    'f1': ('F1', 'Table: hybrid against the best single base representation', fig_f1, L1.INPUTS),
    'f2a': ('F2a', 'Best number of features, per dataset and model', fig_f2a, L2.INPUTS_F2A),
    'f3': ('F3', 'Blueprints: features taken from each representation', fig_f3, L3.INPUTS),
    'f5': ('F5', 'Table: distance against activity difference, single against hybrid', fig_f5, L56.INPUTS_F5),
    'f6': ('F6', 'Table: prediction from nearest neighbours, single against hybrid', fig_f6, L56.INPUTS_F6),
    'f8': ('F8', 'Did the cleaning work: share of pairs improved, one file per group of methods', fig_f8, L8.INPUTS),
    'f9': ('F9', 'The censoring ladder', fig_f9, L911.INPUTS_F9),
    'f10': ('F10', 'Detection against cleaning, one file per dataset', fig_f10,
            L911.INPUTS_F10 + L911.INPUTS_F10_BREAKEVEN),
    'f11': ('F11', 'Score against damage level', fig_f11, L911.INPUTS_F11),
}
LOADER_CALLS = {
    'f1': lambda r, m, a: L1.load(r, m),
    'f2a': lambda r, m, a: L2.load_f2a(r, m, source=a.f2a_source),
    'f3': lambda r, m, a: (L3.load_f3(r, m), L3.load_f3_detail(r, a.f3_detail_dataset, m, model=a.f3_detail_model)),
    'f5': lambda r, m, a: L56.load_f5(r, m),
    'f6': lambda r, m, a: L56.load_f6(r, m),
    'f8': lambda r, m, a: L8.load_f8(r, m),
    'f9': lambda r, m, a: L911.load_f9(r, m),
    'f10': lambda r, m, a: L911.load_f10(r, m),
    'f11': lambda r, m, a: L911.load_f11(r, m, panels=a.f11_panels.split(','), flip_model=a.f11_flip_model,
                                         flip_detector=a.f11_flip_detector, clip_model=a.f11_clip_model,
                                         fold=a.noise_fold),
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
    p.add_argument('--datasets', default='thesis',
                   help='datasets drawn by f1, f2a, f5 and f6: "thesis" (default; THESIS_DATASETS, the signed-off '
                        'Chapter 5 and 6 list), "promoted" (datasets with noise_full units in ledger.json), "all", '
                        'or a,b,c')
    p.add_argument('--f2a-source', choices=['c09', 'c16'], default='c09',
                   help='F2a score curves come from c09 only; kept so older commands still run')
    p.add_argument('--f2a-marks', choices=['c02', 'none'], default='c02',
                   help="F2a strip under each panel: the budget each outer fold's search picked on its "
                        'training molecules (c02), or none')
    p.add_argument('--no-f2a-baselines', dest='f2a_baselines', action='store_false',
                   help='leave out the recursive-elimination and random-feature lines (c17)')
    p.add_argument('--f3-fold', type=int, default=0)
    p.add_argument('--f3-model', default=None, help='default: the first decided model of each dataset')
    p.add_argument('--f3-detail-dataset', default='logd')
    p.add_argument('--f3-detail-model', default='svm', help='model whose all-fold features F3b draws')
    p.add_argument('--f8-rows', default=None,
                   help='file with one row key per line (block|action|words, as in the CSV row_key column). '
                        'Default: every signed-off DEEP method.')
    p.add_argument('--noise-fold', type=int, default=0,
                   help='outer fold drawn by f9, f10 and f11 (f8 reads fold 0 inside its loader)')
    p.add_argument('--f9-datasets', default='caco2,herg_ki,logd,qm9')
    p.add_argument('--f9-level', type=float, default=0.25)
    p.add_argument('--f9-correction', default='censored_em')
    p.add_argument('--f10-pair', default='dnn@hybrid', help='model@configuration')
    p.add_argument('--f10-datasets', default='', help='comma list; default every dataset n32 holds')
    p.add_argument('--f11-panels', default='herg:confusion_directed,herg:uniform,caco2:censoring',
                   help='dataset:damage_kind,... ; herg:<flip kind> reads n24, <dataset>:censoring reads n30')
    p.add_argument('--f11-flip-model', default='xgboost', help='model drawn on the hERG flip panels')
    p.add_argument('--f11-flip-detector', default='neighbor_residual@screen1',
                   help='n24 detector on the hERG flip panels (default: neighbor-label gap, top screened setting)')
    p.add_argument('--f11-clip-model', default='dnn', help='model drawn on the clipping panel')
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
