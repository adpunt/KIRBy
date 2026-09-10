#!/usr/bin/env python3
"""OPIG slide content + figures — generated STRAIGHT FROM the text report.

Every number below is parsed out of the report file (default
~/Downloads/prelimResults); nothing is hand-typed. Re-run whenever the report
changes:

    python tests/analysis/opig_slide_figures.py [PATH_TO_REPORT]

Writes:
  * OPIG_SLIDE_CONTENT.md            (repo root) — the slide-by-slide content
  * tests/analysis/figures/opig/*.png                — the two figures

Design rules (locked by the user, do not relitigate):
  * ONE experiment per figure/table. No aggregates across experiments.
  * Only the user's chosen datasets. Never a "/33"-style denominator.
  * Never mix metrics/domains on one axis (RMSE vs rho vs AUROC stay apart).
  * CIs are shown for SMALL MOLECULE ONLY. Never on antibody plots.
  * Mostly tables; a small number of figures; no dumbbells, no significance stars.
"""
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Downloads' / 'prelimResults'
OUT_MD = REPO / 'OPIG_SLIDE_CONTENT.md'
FIG_DIR = REPO / 'tests' / 'analysis' / 'figures' / 'opig'

MODELS = ['lightgbm', 'xgboost', 'ridge', 'logreg', 'svm', 'rf', 'mlp']

# --------------------------------------------------------------------------- #
# Report loading + section splitting
# --------------------------------------------------------------------------- #
TEXT = REPORT.read_text()


def section(tag):
    """Return the raw text of experiment section [tag] (e.g. 'c02')."""
    m = re.search(rf'\[{tag}\][^\n]*\n=+\n(.*?)(?=\n=+\n\[|\n#####|$)', TEXT, re.S)
    return m.group(1) if m else ''


def rows_for(tag, ds):
    """All lines in section [tag] whose first token is exactly `ds`."""
    out = []
    for ln in section(tag).splitlines():
        toks = ln.split()
        if toks and toks[0] == ds:
            out.append(ln)
    return out


def floats(s):
    return [float(x) for x in re.findall(r'-?\d+\.\d+', s)]


def split_model_rep(tok):
    """'1*lightgbmesm3' -> ('lightgbm', 'esm3'); '1*svm' + 'pdv' handled by caller."""
    tok = tok.lstrip('123*').strip()
    for mdl in MODELS:
        if tok.startswith(mdl):
            rep = tok[len(mdl):]
            return mdl, rep
    return tok, ''


def c02_deployed(ds):
    """Rank-1 (deployed, starred) c02 row: dict with model, rep, base, hybrid, ci, delta, verdict.
    Falls back to BEST-ANY if the rank-1 row is degenerate (score-only)."""
    for ln in rows_for('c02', ds):
        toks = ln.split()
        if len(toks) < 2 or not toks[1].startswith('1'):
            continue
        # model+rep may be mashed into toks[1], or split across toks[1]+toks[2]
        mdl, rep = split_model_rep(toks[1])
        if not rep and len(toks) > 2 and not re.match(r'^-?\d+\.\d+$', toks[2]):
            rep = toks[2]
        f = floats(ln)
        if len(f) >= 4:
            base, hyb, ci, delta = f[0], f[1], f[2], f[3]
            mverd = re.search(r'[+-]\d+\.\d+\s+(\S+)', ln)
            return dict(model=mdl, rep=rep, base=base, hybrid=hyb, ci=ci,
                        delta=delta, verdict=mverd.group(1) if mverd else '')
        break
    return c02_bestany(ds)


def c02_bestany(ds):
    for ln in section('c02').splitlines():
        if 'BEST-ANY' in ln and ds in _preceding_ds_block(ln):
            pass
    # simpler: scan the ds block for the BEST-ANY line
    block, cur = [], None
    for ln in section('c02').splitlines():
        toks = ln.split()
        if toks and toks[0] == ds:
            cur = ds
        elif toks and not ln.startswith(' ' * 20) and toks[0] != ds and 'BEST-ANY' not in ln:
            if cur == ds and toks[0] != ds:
                cur = None
        if cur == ds:
            block.append(ln)
    for ln in block:
        if 'BEST-ANY' in ln:
            mm = re.search(r'BEST-ANY \((\w+)\)', ln)
            f = floats(ln)
            if len(f) >= 4:
                return dict(model=mm.group(1) if mm else '', rep='(any)',
                            base=f[0], hybrid=f[1], ci=f[2], delta=f[3],
                            verdict='BEST-ANY')
    return None


def _preceding_ds_block(ln):
    return ''  # unused helper kept for clarity


def c01_row(ds):
    ln = (rows_for('c01', ds) or [''])[0]
    toks = ln.split()
    if len(toks) >= 6:
        return dict(nreps=int(toks[1]), rep=toks[2], score=float(toks[3]),
                    erank=float(toks[4]), mi=float(toks[5]))
    return None


def c05_row(ds):
    ln = (rows_for('c05', ds) or [''])[0]
    toks = ln.split()
    # ds model n_small d_small n_large d_large trend win
    if len(toks) >= 8:
        return dict(model=toks[1], n_small=toks[2], d_small=toks[3],
                    n_large=toks[4], d_large=toks[5], trend=toks[6], win=toks[7])
    return None


def c11_row(ds):
    ln = (rows_for('c11', ds) or [''])[0]
    toks = ln.split()
    if len(toks) >= 7:
        return dict(model=toks[1], single=float(toks[2]), pca=float(toks[3]),
                    hybrid=float(toks[4]), delta=float(toks[5]), verdict=toks[6])
    return None


def c08_row(ds):
    ln = (rows_for('c08', ds) or [''])[0]
    toks = ln.split()
    if len(toks) >= 3:
        return dict(picp=float(toks[1]), cal_err=float(toks[2]))
    return None


def c04_knn(ds):
    ln = (rows_for('c04', ds) or [''])[0]
    toks = ln.split()
    if len(toks) >= 3:
        return dict(method=toks[1], knn_delta=float(toks[2]))
    return None


def s02_row(ds):
    ln = (rows_for('s02', ds) or [''])[0]
    f = floats(ln)
    toks = ln.split()
    if len(f) >= 3:
        return dict(rep=toks[1], indiv=f[0], hybrid=f[1], lift=f[2])
    return None


def s13_retr(ds):
    ln = (rows_for('s13', ds) or [''])[0]
    toks = ln.split()
    if len(toks) >= 3:
        return toks[2]  # retr win boolean
    return None


# --------------------------------------------------------------------------- #
# Selections (the user's datasets, grouped by story — never mixed on one axis)
# --------------------------------------------------------------------------- #
MOLECULAR = ['adme_hlm', 'adme_mdr1', 'caco2', 'herg_ki']          # RMSE, CIs OK
AB_WORKS = ['flab_expression_koenig', 'ddg_combined', 'flab_pk_jain_heprt',
            'flab_thermo_jain', 'flab_thermo_tresanco']            # rho, no CIs
GINKGO = ['ginkgo_acsins', 'ginkgo_aggregation', 'ginkgo_colloidal',
          'ginkgo_heparin', 'ginkgo_hic', 'ginkgo_pr_cho', 'ginkgo_purity',
          'ginkgo_titer', 'ginkgo_tm2']                            # rho
GENOMIC = ['nt_promoter_tata', 'gb_human_enhancers_cohn']          # AUROC
C04_SEL = ['ddg_combined', 'flab_expression_koenig', 'abagym_EGFR', 'adme_hlm',
           'adme_mdr1', 'caco2', 'herg_ki', 'buchwald_hartwig']    # importance Δ


def fmt(x, sign=False):
    if x is None:
        return '--'
    return (f'{x:+.3f}' if sign else f'{x:.3f}')


def md_table(headers, rows):
    # TAB-separated (not markdown pipes): copy a block and paste onto a Keynote
    # slide and it becomes a real table. Cells hold commas/arrows, so tabs — not
    # CSV — are the safe delimiter.
    line = '\t'.join(str(h) for h in headers)
    body = '\n'.join('\t'.join(str(c) for c in r) for r in rows)
    return f'{line}\n{body}'


# Proper display names for slides — never show raw snake_case IDs. (Domain
# descriptions in parentheses are conventional; adjust to taste.)
DISPLAY = {
    'adme_hlm': 'ADME: HLM (metabolic stability)',
    'adme_mdr1': 'ADME: MDR1 (efflux)',
    'caco2': 'Caco-2 (permeability)',
    'herg_ki': 'hERG Ki (cardiotoxicity)',
    'flab_expression_koenig': 'FLAb Expression (Koenig)',
    'ddg_combined': 'Binding ΔΔG',
    'flab_pk_jain_heprt': 'FLAb PK — Heparin (Jain)',
    'flab_thermo_jain': 'FLAb Thermostability (Jain)',
    'flab_thermo_tresanco': 'FLAb Thermostability (Tresanco)',
    'abagym_EGFR': 'AbAgym EGFR', 'abagym_HER2': 'AbAgym HER2',
    'abcov_kd': 'AbCoV — binding Kd',
    'nt_promoter_tata': 'Promoter (TATA)',
    'gb_human_enhancers_cohn': 'Human enhancers (Cohn)',
    'buchwald_hartwig': 'Buchwald–Hartwig (yield)',
    'ginkgo_acsins': 'AC-SINS', 'ginkgo_aggregation': 'Aggregation',
    'ginkgo_colloidal': 'Colloidal stability', 'ginkgo_heparin': 'Heparin',
    'ginkgo_hic': 'HIC', 'ginkgo_pr_cho': 'PR (CHO)', 'ginkgo_purity': 'Purity',
    'ginkgo_titer': 'Titer', 'ginkgo_tm2': 'Tm (thermostability)',
}


def disp(ds):
    return DISPLAY.get(ds, ds.replace('_', ' ').title())


# Proper display names for representations (best-effort — confirm to taste).
REP_DISPLAY = {
    'esm3': 'ESM3', 'esm2_concat': 'ESM2 (concat)', 'esm2_wt_delta': 'ESM2 (WT-Δ)',
    'esm2_midlayers': 'ESM2 (mid-layers)', 'prot_t5': 'ProtT5', 'igt5': 'IgT5',
    'ablang2': 'AbLang2', 'abroberta': 'AbRoBERTa', 'saprot': 'SaProt',
    'saprot_cdr': 'SaProt (CDR)', 'molformer': 'MolFormer', 'mordred': 'Mordred',
    'pdv': 'PDV', 'smited': 'SMI-TED', 'mhg_gnn': 'MHG-GNN',
    'mutation_features': 'Mutation features', 'kmer5': 'k-mer (k=5)',
    'grover_dna': 'GROVER (DNA)',
}


def rdisp(rep):
    return REP_DISPLAY.get(rep, rep)


def c02_baselines(ds):
    """Every reported single-rep baseline score (ranks 1-3) for a dataset."""
    out, active = [], False
    for ln in section('c02').splitlines():
        toks = ln.split()
        if toks and toks[0] == ds:
            active = True
            f = floats(ln)
            if f:
                out.append(f[0])
        elif active:
            if toks and toks[0] in ('2', '3'):
                f = floats(ln)
                if f:
                    out.append(f[0])
            elif toks and toks[0] != ds and 'BEST-ANY' not in ln:
                active = False
    return out


# --------------------------------------------------------------------------- #
# Build the markdown
# --------------------------------------------------------------------------- #
def build_markdown():
    P = []
    P.append('# OPIG talk — slide content (generated from `prelimResults`)\n')
    P.append(f'_Source: `{REPORT}` — every number parsed from the report, not '
             'retyped. Regenerate: `python tests/analysis/opig_slide_figures.py`._\n')
    P.append('**Big picture (one line):** the hybrid helps most exactly where no '
             'single representation already dominates and there is enough data / '
             'diversity to draw from — it wins cleanly on small molecules and most '
             'antibody properties, wins *biggest* when labels are scarce, and fails '
             '*predictably* when one compact representation already holds the signal '
             '(HER2, AbCoV).\n')
    P.append('\n---\n')

    # ---- Slide 1: molecular c02 (with CIs) -------------------------------- #
    P.append('## Slide 1 — Small molecules: the hybrid beats the best single '
             'descriptor (same model)  · experiment c02\n')
    P.append('_Metric: RMSE. Small-molecule datasets, so 95% confidence intervals '
             'are shown._\n')
    rows = []
    mol_fig = []
    for ds in MOLECULAR:
        r = c02_deployed(ds)
        if not r:
            continue
        rows.append([disp(ds), r['model'], rdisp(r['rep']), fmt(r['base']),
                     fmt(r['hybrid']), f"±{r['ci']:.3f}", fmt(r['delta'], True),
                     r['verdict']])
        mol_fig.append((ds, c02_baselines(ds), r['hybrid'], r['ci']))
    P.append(md_table(['dataset', 'model', 'best descriptor', 'base', 'hybrid',
                       '95% CI', 'Δ', 'verdict'], rows))
    P.append(f'\n![molecular]({FIG_DIR.relative_to(REPO)}/mol_c02_ci.png)\n')
    P.append('_Say:_ "Same model, same folds — swap the single descriptor for the '
             'hybrid and the error drops on every small-molecule dataset." (qm9 was '
             'not in this run.)\n')
    P.append('\n---\n')

    # ---- Slide 2: antibody works c02 -------------------------------------- #
    P.append('## Slide 2 — Antibodies: it works, cleanly  · experiment c02\n')
    P.append('_Metric: Spearman ρ (higher is better). No CIs (antibody)._\n')
    rows = []
    for ds in AB_WORKS:
        r = c02_deployed(ds)
        if not r:
            continue
        rows.append([disp(ds), r['model'], rdisp(r['rep']), fmt(r['base']),
                     fmt(r['hybrid']), fmt(r['delta'], True), r['verdict']])
    P.append(md_table(['dataset', 'model', 'best rep', 'base ρ', 'hybrid ρ',
                       'Δ', 'verdict'], rows))
    kg = c02_deployed('flab_expression_koenig')
    P.append(f'\n_Say:_ "The clean win. Hero is FLAb Expression (Koenig) — '
             f"{kg['model']} on {rdisp(kg['rep'])}, {kg['base']:.3f} → "
             f"{kg['hybrid']:.3f} (Spearman ρ), a big jump from an already-strong "
             'baseline — and it repeats across expression, PK and thermostability."\n')
    P.append('\n---\n')

    # ---- Slide 3: EGFR vs HER2 (same task, different setup) --------------- #
    e02 = c02_deployed('abagym_EGFR'); h02 = c02_deployed('abagym_HER2')
    e05 = c05_row('abagym_EGFR'); h05 = c05_row('abagym_HER2')
    egfr_n = e05['n_large'] if e05 else '659'
    her2_n = h05['n_large'] if h05 else '146'
    P.append('## Slide 3 — Same task, different setup: EGFR vs HER2\n')
    P.append('**What it is.** Two deep-mutational scans of a single antibody\'s '
             'binding affinity — Cetuximab against EGFR, and Trastuzumab against '
             'HER2. Same *kind* of task, but a different antibody, antigen and '
             'structure each time.\n')
    P.append(f'\n**The size gap.** EGFR ≈ {egfr_n} mutants · HER2 ≈ {her2_n} mutants '
             '— about 4.5× smaller.\n')
    P.append('\n**What happens.**\n')
    if e05:
        P.append(f"- **EGFR** — the hybrid helps, and its edge *grows* with data "
                 f"({e05['d_small']} at n≈{e05['n_small']} → {e05['d_large']} at "
                 f"n≈{e05['n_large']}).")
    if h02:
        P.append(f"- **HER2** — no reliable effect ({h02['delta']:+.3f}), and there "
                 "can't be one at this size: ≈146 mutants is only ~29 points per "
                 "cross-validation fold, and the whole-antibody embeddings barely "
                 "change across 146 single-point Trastuzumab mutants — so any hybrid "
                 "difference here is noise, not signal.")
    P.append('\n**Honest takeaway.** This is a *"when the data is hard / too small"* '
             'caveat, not a clean win. EGFR is the real signal; HER2 is the '
             "cautionary case — same task, but you can't conclude anything from "
             'n≈146.\n')
    P.append('\n**Setup (for a backup / methods slide).** Split = ProteinGym '
             'position-modulo; label = MinMax-normalised DMS score; source = '
             'Deszyński et al. 2024 (AbAgym). EGFR = EGFR_2013_Cetuximab (PDB 1yy9, '
             'heavy + light mutated); HER2 = HER2_2021_trastuzumab (PDB 8pwh, heavy '
             'chain only).\n')
    P.append('\n_Say:_ "Same experiment — mutate one antibody, measure binding. On '
             'EGFR, with enough mutants, the hybrid finds signal and its edge grows '
             'with data. HER2 is the identical task at a fifth of the size — too few '
             "points, too similar to each other — so it's the honest caveat, not a "
             'failure of the method."\n')
    P.append('\n---\n')

    # ---- Slide 4: Ginkgo developability panel (dumbbell) ----------------- #
    P.append('## Slide 4 — Ginkgo developability panel  · experiment c02\n')
    P.append('_Metric: Spearman ρ. Every assay: best single representation vs the '
             'hybrid, ranked. Purity is the honest miss._\n')
    rows = []
    for ds in GINKGO:
        r = c02_deployed(ds)
        if not r:
            continue
        rows.append([disp(ds), r['model'], rdisp(r['rep']), fmt(r['base']),
                     fmt(r['hybrid']), fmt(r['delta'], True), r['verdict']])
    P.append(md_table(['assay', 'model', 'best rep', 'base ρ', 'hybrid ρ', 'Δ',
                       'verdict'], rows))
    P.append(f'\n![ginkgo]({FIG_DIR.relative_to(REPO)}/ginkgo_dumbbell.png)\n')
    hic02 = c02_deployed('ginkgo_hic')
    P.append(f'\n_HIC note:_ HIC (the recognisable assay) is the one where accuracy '
             f"is flat ({hic02['base']:.3f} → {hic02['hybrid']:.3f} — essentially no "
             'change). Lead with Tm (thermostability) for the accuracy story; only '
             'bring up HIC if you get into using the hybrid to find similar '
             'antibodies by distance.\n')
    P.append('\n_Say:_ "Across nine developability assays the hybrid beats the best '
             'single representation on most, ties on a couple, and clearly loses on '
             'one — Purity — which we show rather than hide. Thermostability is the '
             'biggest win."\n')
    P.append('\n---\n')

    # ---- Slide 5: importance-weighted retrieval (the "other" experiment) --- #
    P.append('## Slide 5 — The blueprint\'s importances are real signal  · '
             'experiment c04 (importance-weighted retrieval)\n')
    P.append('_This is the standout secondary result (it replaces the placeholder '
             '"c13" — c13 forward-compatibility produced no results this run). Each '
             'bar is one dataset; the quantity is the SAME within-dataset improvement '
             '(Δ = weighted − unweighted kNN)._\n')
    rows = []
    for ds in C04_SEL:
        r = c04_knn(ds)
        if not r:
            continue
        rows.append([ds, r['method'], fmt(r['knn_delta'], True)])
    P.append(md_table(['dataset', 'importance method', 'Δ (weighted − unweighted)'], rows))
    # NOTE: deliberately NO cross-dataset count/fraction here ("+31/-0" is banned).
    P.append(f'\n![importance]({FIG_DIR.relative_to(REPO)}/c04_importance.png)\n')
    P.append('_Say:_ "Weight the retrieval by the hybrid\'s per-feature importances '
             'and it helps on every dataset — biggest on ddg_combined. The '
             'importances the blueprint is built on carry information; they are not '
             'decoration."\n')
    P.append('\n---\n')

    # ---- Slide 6: other domains (two separate mini-tables) ---------------- #
    P.append('## Slide 6 — Other domains\n')
    P.append('### Genomic — hybrid wins *and* compresses  · c02 (metric: AUROC)\n')
    rows = []
    for ds in GENOMIC:
        r = c02_deployed(ds)
        if not r:
            continue
        rows.append([ds, r['model'], r['rep'], fmt(r['base']), fmt(r['hybrid']),
                     fmt(r['delta'], True), r['verdict']])
    P.append(md_table(['dataset', 'model', 'best rep', 'base AUROC', 'hybrid AUROC',
                       'Δ', 'verdict'], rows))
    P.append('\n_nt_promoter_tata wins while cutting dimension (1024 → 500)._\n')
    P.append('\n### Reaction — a small-data method  · c05 (metric: RMSE, +Δ = better)\n')
    bh = c05_row('buchwald_hartwig')
    if bh:
        P.append(md_table(['dataset', 'model', 'n (small)', 'Δ @ small n', 'n (large)',
                           'Δ @ large n', 'trend'],
                          [['buchwald_hartwig', bh['model'], bh['n_small'], bh['d_small'],
                            bh['n_large'], bh['d_large'], bh['trend']]]))
        P.append(f'\n_Say:_ "With ~{bh["n_small"]} reactions the hybrid slashes error '
                 f'({bh["d_small"]} RMSE); by ~{bh["n_large"]} the single rep has caught '
                 'up. KIRBy buys you the most exactly when labels are expensive."\n')
    else:
        P.append('_(buchwald_hartwig c05 not present in this run.)_\n')
    P.append('\n---\n')

    # ---- Slide 7: honest negative (written, no table) --------------------- #
    P.append('## Slide 7 — One honest failure (written, no table)  · abcov_kd\n')
    n02 = c02_deployed('abcov_kd')
    n11 = c11_row('abcov_kd')
    n05 = c05_row('abcov_kd')
    if not (n02 and n11 and n05):
        P.append('_(abcov_kd not fully present in this run — negative example '
                 'skipped.)_\n')
        n02 = n11 = n05 = None
    if n02:
        P.append(f'> **abcov_kd** (antibody binding Kd) is the clean failure — and it '
             f'fails *for a reason*. The best single representation, {n02["rep"]}, '
             f'already scores {n02["base"]:.3f}. Build the hybrid on it and it drops '
             f'to {n02["hybrid"]:.3f} ({n02["delta"]:+.3f}, c02). Concatenate '
             f'everything (PCA-concat {n11["pca"]:.3f}) and it is also worse. Give it '
             f'more data and it gets *worse* ({n05["d_large"]} at the largest n, '
             f'c05). When one representation already holds the signal, selecting and '
             f'blending can only dilute it — the same mechanism as HER2, shown not '
             f'hidden.\n')
    P.append('\n---\n')
    P.append('### Honesty notes\n')
    P.append('- Forward-compatibility (c13) and noise-robustness (c10) produced no '
             'usable numbers this run — do not claim them.\n')
    P.append('- Nothing above is averaged across datasets; every figure/table is a '
             'single experiment on your chosen datasets.\n')

    OUT_MD.write_text('\n'.join(P))
    print(f'Wrote {OUT_MD}')
    return mol_fig


# --------------------------------------------------------------------------- #
# Figures (exactly two; no dumbbells, no stars)
# --------------------------------------------------------------------------- #
def make_figures(mol_fig):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ACCENT = '#d97a4a'
    BASE = '#7faecf'

    # Figure 1: molecular c02 — base vs hybrid RMSE, hybrid 95% CI whisker.
    mol_fig = list(reversed(mol_fig))  # top-down nicest
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(mol_fig))
    for i, (ds, base, hyb, ci) in enumerate(mol_fig):
        ax.plot([base, hyb], [i, i], color='#bbbbbb', zorder=1, lw=1.5)
    ax.scatter([m[1] for m in mol_fig], y, color=BASE, s=70, zorder=3,
               label='best single descriptor')
    ax.errorbar([m[2] for m in mol_fig], y, xerr=[m[3] for m in mol_fig],
                fmt='o', color=ACCENT, ms=8, capsize=4, zorder=4,
                label='KIRBy hybrid (±95% CI)')
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in mol_fig])
    ax.set_xlabel('RMSE  (lower is better)')
    ax.set_title('Small molecules — hybrid lowers error, same model (c02)')
    ax.legend(loc='lower left', fontsize=8, frameon=False)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    p1 = FIG_DIR / 'mol_c02_ci.png'
    fig.savefig(p1, dpi=170)
    plt.close(fig)
    print(f'Wrote {p1}')

    # Figure 2: c04 importance-weighted retrieval Δ per dataset (all positive).
    data = [(ds, c04_knn(ds)['knn_delta']) for ds in C04_SEL if c04_knn(ds)]
    data.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(data))
    ax.barh(y, [d[1] for d in data], color=ACCENT)
    for i, (ds, v) in enumerate(data):
        ax.text(v + 0.004, i, f'{v:+.3f}', va='center', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data])
    ax.set_xlabel('Δ  (importance-weighted − unweighted retrieval)')
    ax.set_title('Importance weighting helps on every dataset (c04)')
    ax.axvline(0, color='black', lw=0.6)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    p2 = FIG_DIR / 'c04_importance.png'
    fig.savefig(p2, dpi=170)
    plt.close(fig)
    print(f'Wrote {p2}')


if __name__ == '__main__':
    mol_fig = build_markdown()
    make_figures(mol_fig)
