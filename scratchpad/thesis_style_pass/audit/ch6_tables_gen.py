import json, os, statistics as st
R='/Volumes/seagate/kirby_results_2026-10-10/similarity'
DS=['qm9','caco2','adme_mdr1','herg_ki','logd','herg','abagym_EGFR','flab_thermo_tresanco','ginkgo_acsins','ginkgo_colloidal','ginkgo_pr_cho','ginkgo_titer','ddg_combined','buchwald_hartwig']
def load(exp,ds):
    p=f'{R}/{exp}/{ds}.json'
    if not os.path.exists(p): return None
    return json.load(open(p))
"""Build the Chapter 6 tables from the 10 Oct results (read only). Writes one .tex snippet per table."""
import sys, json, os, statistics as st
from collections import defaultdict

sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests/analysis')
OUT = "/private/tmp/ch6_tables_out"
os.makedirs(OUT, exist_ok=True)

# representation labels: copied from tests/analysis/thesis_figure_style.py REP_LABELS (read there)
import re
src = open('/Users/apunt/repos/KIRBy/tests/analysis/thesis_figure_style.py').read()
REP = dict(re.findall(r"'([a-z0-9_]+)': '([^']+)'", src[src.find('REP_LABELS = {'):src.find('# Configurations the noise')]))

NAME = {'qm9': 'QM9', 'caco2': 'Caco-2', 'adme_mdr1': 'MDR1-MDCK efflux', 'herg_ki': 'hERG $K_i$',
        'logd': 'LogD', 'herg': 'hERG blocking', 'abagym_EGFR': 'EGFR binding',
        'flab_thermo_tresanco': 'Nanobody thermostability', 'ginkgo_acsins': 'AC-SINS self-association',
        'ginkgo_colloidal': 'SMAC colloidal stability', 'ginkgo_pr_cho': 'CHO polyreactivity',
        'ginkgo_titer': 'Expression titer', 'ddg_combined': 'Binding free energy ($\\Delta\\Delta G$)',
        'buchwald_hartwig': 'Buchwald--Hartwig yield'}
GROUP = {'qm9': 'Small molecules', 'caco2': 'Small molecules', 'adme_mdr1': 'Small molecules',
         'herg_ki': 'Small molecules', 'logd': 'Small molecules', 'herg': 'Small molecules',
         'buchwald_hartwig': 'Reactions'}
METRIC = {'rmse': 'RMSE $\\downarrow$', 'auc': 'AUROC $\\uparrow$', 'spearman': 'Spearman $\\uparrow$'}
MODEL = {'dnn': 'NN', 'svm': 'SVM', 'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'gp': 'GP', 'rf': 'RF'}
DIST = {'cosine': 'cosine', 'manhattan': 'Manhattan', 'minkowski_p0.5': 'Minkowski'}
BOLD = set()  # (table, ds, column key, row key) filled per table


def group(ds):
    return GROUP.get(ds, 'Antibodies')


def dp_of(ds):
    return 1 if ds == 'buchwald_hartwig' else 3


def fmt(v, dp):
    if v is None:
        return ''
    if v != 0 and float(f'{v:.{dp}f}') == 0:
        dp += 1  # a nonzero value that would print as zero keeps one more decimal
    s = f'{v:.{dp}f}'
    if s.startswith('-'):
        s = '$-$' + s[1:]
    return s


def fmt_group(vals, dp):
    """Tie rule (DECIMALS_RULE.md 4): a value that prints the same as another value it is compared
    with keeps one more decimal."""
    out = []
    for i, v in enumerate(vals):
        if v is None:
            out.append('')
            continue
        tie = any(j != i and w is not None and w != v and f'{w:.{dp}f}' == f'{v:.{dp}f}' for j, w in enumerate(vals))
        out.append(fmt(v, dp + 1 if tie else dp))
    return out


def bold(s):
    return '\\textbf{' + s + '}' if s else s


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def grouped_rows(rows_by_ds, ncols):
    """rows_by_ds: list of (ds, [row strings]); inserts italic group headings."""
    lines, cur = [], None
    for ds, rows in rows_by_ds:
        g = group(ds)
        if g != cur:
            if cur is not None:
                pass
            lines.append(f'\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{{g}}}}} \\\\')
            cur = g
        lines.extend(rows)
    return '\n'.join(lines)


def write(name, text):
    open(os.path.join(OUT, name + '.tex'), 'w').write(text)
    nb = text.count('\\textbf{')
    print(f'== {name}: {nb} bold')


def metric_of(exp, ds):
    d = load(exp, ds)
    m = d.get('metric') or d.get('metric_name') or d.get('dataset_metric')
    return {'auroc': 'auc'}.get(m, m)


DSMETRIC = {ds: ('auc' if ds == 'herg' else 'spearman' if group(ds) == 'Antibodies' else 'rmse') for ds in DS}

# ---------------------------------------------------------------- T1 single spread (s01)
rows = []
data_t1 = {}
for ds in DS:
    d = load('s01_similarity_baseline', ds)
    combos = []
    for rep, bd in d['by_rep'].items():
        for dist, v in bd.items():
            if isinstance(v, dict) and 'sac_rho_mean' in v:
                combos.append((v['sac_rho_mean'], rep, dist, v['sac_rho_per_fold']))
    combos.sort(key=lambda c: -c[0])
    b = combos[0]
    vals = [c[0] for c in combos]
    tied = [DIST[c[2]] for c in combos if c[1] == b[1] and c[0] == b[0]]
    data_t1[ds] = dict(best=b[0], rep=b[1], dist=' or '.join(tied), lo=min(b[3]), hi=max(b[3]), med=st.median(vals),
                       low=combos[-1][0], n=len(combos))
    x = data_t1[ds]
    cells = [fmt(x['best'], 3), fmt(x['lo'], 3), fmt(x['hi'], 3), fmt(x['med'], 3), fmt(x['low'], 3)]
    if ds == 'qm9':
        cells[0] = bold(cells[0])
    if ds == 'ginkgo_colloidal':
        cells[0] = bold(cells[0]); cells[3] = bold(cells[3])
    if ds == 'herg_ki':
        pass
    rows.append((ds, [f"{NAME[ds]} & {REP.get(x['rep'], x['rep'])}, {x['dist']} & " + ' & '.join(cells) + ' \\\\']))
t1 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s01_similarity_baseline/<dataset>.json, by_rep[<representation>][<distance>].sac_rho_mean
% and .sac_rho_per_fold (5 outer folds), every representation under every distance, no weighting; read 2026-10-10 (files identical to kirby_results_2026-10-09).
% Best = highest sac_rho_mean over all pairs; median and lowest over the same pairs. Pairs at or below zero (counts, not printed): MDR1 5 of 30, EGFR 2 of 30,
% AC-SINS 6 of 30, SMAC 15 of 30, CHO 5 of 30, all others 0. Script: scratchpad/thesis_style_pass/audit/ch6_tables_gen.py (copy of the session script).
\begin{table}[htbp]
\centering
\begin{tabular}{@{}llrrrrr@{}}
\toprule
 & \multicolumn{4}{c}{Best pair of representation and distance} & \multicolumn{2}{c}{All pairs} \\
\cmidrule(lr){2-5}\cmidrule(l){6-7}
Dataset & Pair & Mean & \makecell[r]{Lowest\\fold} & \makecell[r]{Highest\\fold} & Median & Lowest \\
\midrule
""" + grouped_rows(rows, 7) + r"""
\bottomrule
\end{tabular}
\caption[Distance against label difference in single representations]{Distance against label difference in single representations. One row is one dataset. Values are the Spearman correlation between the distance of two held-out molecules and their absolute label difference (higher is better), over pairs of one representation and one distance, unweighted. Best pair: picked on the held-out molecules; mean over five outer folds, and its lowest and highest fold. All pairs: median and lowest of the fold means.}
\label{tab:similarity:single_spread}
\end{table}
"""
write('t1_single_spread', t1)

# ---------------------------------------------------------------- T2 feature spaces (s02)
def s02_means(ds):
    d = load('s02_hybrid_similarity', ds)
    if d is None:
        return None
    acc = defaultdict(list)
    for f in d['folds']:
        for k, c in f['comparisons'].items():
            for r in c['rows']:
                acc[(k, r['distance'], r['hybrid_weighting'])].append((r['single_sac_rho'], r['hybrid_sac_rho']))
    out = {}
    for key, v in acc.items():
        out[key] = (mean([a for a, _ in v]), mean([b for _, b in v]), len(v))
    return out, d['models']

rows = []
for ds in DS:
    r = s02_means(ds)
    if r is None:
        continue
    m, models = r
    K = 'knn distance hybrid - knn base rank 1'
    S = 'similarity hybrid - knn base rank 1'
    su, hu = m[(K, 'manhattan', 'unweighted')][:2]
    sw, hw = (m.get((K, 'manhattan', 'lightgbm_gain')) or (None, None))[:2]
    simu = (m.get((S, 'manhattan', 'unweighted')) or (None, None))[:2][1]
    simw = (m.get((S, 'manhattan', 'lightgbm_gain')) or (None, None))[:2][1]
    vals = [su, sw, hu, hw, simu, simw]
    cells = fmt_group(vals, 3)
    if ds == 'logd':
        cells[1] = bold(cells[1])
    if ds == 'caco2':
        cells[4] = bold(cells[4])
    rows.append((ds, [f'{NAME[ds]} & ' + ' & '.join(cells) + ' \\\\']))
t2 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s02_hybrid_similarity/<dataset>.json, folds[*].comparisons["knn distance hybrid - knn base rank 1"]
% and ["similarity hybrid - knn base rank 1"].rows[distance=manhattan, hybrid_weighting=unweighted|lightgbm_gain].{single_sac_rho,hybrid_sac_rho}, mean of 5 folds;
% read 2026-10-10. Unweighted columns match tab:distance_tracks_activity. Weighted cells are empty on Caco-2, hERG blocking, hERG K_i, expression titer and
% SMAC colloidal stability (their files hold unweighted distances only); similarity hybrid empty where it is undefined. No s02 file for adme_mdr1 (only a
% partial .jsonl), so MDR1-MDCK efflux has no row.
\begin{table}[htbp]
\centering
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Single representation} & \multicolumn{2}{c}{Distance hybrid} & \multicolumn{2}{c}{Similarity hybrid} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Dataset & \makecell[r]{Un-\\weighted} & Weighted & \makecell[r]{Un-\\weighted} & Weighted & \makecell[r]{Un-\\weighted} & Weighted \\
\midrule
""" + grouped_rows(rows, 7) + r"""
\bottomrule
\end{tabular}
\caption[Distance against label difference in each feature space]{Distance against label difference in each feature space, with and without importance weighting. One row is one dataset. Values are the Spearman correlation between the distance of two held-out molecules and their absolute label difference (higher is better), mean over five outer folds, Manhattan distance. The single representation is the one the nearest-neighbor model ranked first in each fold.}
\label{tab:similarity:feature_spaces}
\end{table}
"""
write('t2_feature_spaces', t2)

# ---------------------------------------------------------------- T3 predictive hybrids (s02)
rows = []
for ds in DS:
    r = s02_means(ds)
    if r is None:
        continue
    m, models = r
    rr = []
    for i, mod in enumerate(models):
        a = m.get((f'{mod} hybrid 1 - {mod} base rank 1', 'manhattan', 'unweighted'))
        b = m.get((f'{mod} hybrid 2 - {mod} base rank 1', 'manhattan', 'unweighted'))
        if a is None:
            continue
        cells = fmt_group([a[0], a[1], b[1] if b else None], 3)
        if ds == 'ginkgo_pr_cho':
            cells[1] = bold(cells[1]); cells[2] = bold(cells[2])
        rr.append(f"{NAME[ds] if not rr else ''} & {MODEL[mod]} & " + ' & '.join(cells) + ' \\\\')
    rows.append((ds, rr))
t3 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s02_hybrid_similarity/<dataset>.json, folds[*].comparisons["<model> hybrid 1|2 - <model> base rank 1"]
% .rows[distance=manhattan, hybrid_weighting=unweighted].{single_sac_rho,hybrid_sac_rho}, mean of 5 folds; read 2026-10-10. No s02 file for adme_mdr1.
\begin{table}[htbp]
\centering
\begin{tabular}{@{}llrrr@{}}
\toprule
Dataset & Model & \makecell[r]{Best single\\representation} & \makecell[r]{First\\hybrid} & \makecell[r]{Second\\hybrid} \\
\midrule
""" + grouped_rows(rows, 5) + r"""
\bottomrule
\end{tabular}
\caption[Distance against label difference in the predictive hybrids]{Distance against label difference in each downstream model's best single representation and its two predictive hybrids. One row is one dataset and model. Values are the Spearman correlation between the distance of two held-out molecules and their absolute label difference (higher is better), mean over five outer folds, Manhattan distance, unweighted.}
\label{tab:similarity:predictive_hybrids}
\end{table}
"""
write('t3_predictive_hybrids', t3)

# ---------------------------------------------------------------- T4 number of features (s08), network only
rows = []
for ds in DS:
    d = load('s08_optimal_dimension', ds)
    if d is None or 'dnn' not in d['models']:
        continue
    rr = [f"\\multicolumn{{3}}{{@{{}}l}}{{{NAME[ds]}, {METRIC[DSMETRIC[ds]]}}} \\\\"]
    for b in d['general_budgets']:
        p = d['prediction_by_budget']['dnn'].get(str(b), {}).get('rank1')
        sp = []
        for f in d['folds']:
            try:
                sp.append(f['prediction']['dnn'][str(b)]['hybrids'][0]['neighbour']['by_distance']['manhattan']['sac_unweighted'])
            except (KeyError, IndexError, TypeError):
                pass
        sc = fmt(p['mean'], dp_of(ds)) if p else ''
        sa = fmt(mean(sp), 3)
        if ds == 'logd' and b == 2000:
            sc = bold(sc)
        if ds == 'logd' and b == 500:
            sa = bold(sa)
        rr.append(f'\\quad {b:,} & {sc} & {sa} \\\\'.replace(',', '{,}'))
    rows.append((ds, rr))
t4 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s08_optimal_dimension/<dataset>.json; score = prediction_by_budget.dnn[<budget>].rank1.mean;
% Spearman = mean over folds of folds[*].prediction.dnn[<budget>].hybrids[0].neighbour.by_distance.manhattan.sac_unweighted (the same rank-1 hybrid); read 2026-10-10.
% Network (dnn) only, as the text describes; hERG blocking, AC-SINS self-association and Buchwald--Hartwig fit no network. No s08 file for adme_mdr1.
% QM9 is the 10 Oct file (9 Oct 14:51), which the TODO(missing result) above does not yet know about.
\begin{table}[htbp]
\centering
\begin{tabular}{@{}lrr@{}}
\toprule
Number of features & \makecell[r]{Prediction\\score} & \makecell[r]{Spearman\\$\uparrow$} \\
\midrule
""" + '\n'.join(sum([rr for _, rr in rows], [])) + r"""
\bottomrule
\end{tabular}
\caption[Prediction score and distance against label difference by number of features]{Prediction score and distance against label difference of the network's top hybrid at each number of features of its search. One row is one dataset and number of features. Prediction score: held-out score in the dataset's metric (direction in the heading). Spearman: correlation between the distance of two held-out molecules and their absolute label difference in the same hybrid, Manhattan distance, unweighted. Means over five outer folds.}
\label{tab:similarity:feature_count}
\end{table}
"""
write('t4_feature_count', t4)

# ---------------------------------------------------------------- T5 kNN picked again (s05)
rows = []
for ds in DS:
    d = load('s05_knn_prediction', ds)
    r = d['fold_reselection_result']['knn']
    cells = fmt_group([r['best_single_reselected'], r['hybrid_reselected']], dp_of(ds))
    if ds in ('logd', 'ginkgo_colloidal'):
        cells[1] = bold(cells[1])
    rows.append((ds, [f"{NAME[ds]} & {METRIC[DSMETRIC[ds]]} & " + ' & '.join(cells) + ' \\\\']))
t5 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s05_knn_prediction/<dataset>.json, fold_reselection_result.knn.{best_single_reselected,hybrid_reselected}
% (mean of 5 folds; each fold's setting picked on the other four folds' held-out scores); read 2026-10-10. MDR1-MDCK efflux is the 9 Oct 23:46 file.
\begin{table}[htbp]
\centering
\begin{tabular}{@{}llrr@{}}
\toprule
Dataset & Metric & \makecell[r]{Single\\representation} & Hybrid \\
\midrule
""" + grouped_rows(rows, 4) + r"""
\bottomrule
\end{tabular}
\caption[Prediction from nearest neighbors with the setting picked again for each fold]{Prediction of each held-out molecule from its nearest training molecules, with the feature space and setting picked again for each outer fold. One row is one dataset; mean over five outer folds, in the dataset's own metric (arrow gives the direction). Single representation: the nearest-neighbor model's two best, unweighted. Hybrid: up to six hybrids, unweighted or weighted. Buchwald--Hartwig RMSE is in percentage points of yield.}
\label{tab:similarity:knn_picked_again}
\end{table}
"""
write('t5_knn_picked_again', t5)

# ---------------------------------------------------------------- T6 joined representations (c04)
C04 = '/Volumes/seagate/kirby_results_2026-10-10/complexity/c04_importance_as_metric'
rows = []
for ds in DS:
    d = json.load(open(f'{C04}/{ds}.json'))
    if 'methods' not in d:
        continue
    vals = {}
    u = None
    for meth, v in d['methods'].items():
        k = v.get('knn', {}).get('10')
        if k:
            u = k['unweighted_dataset_metric']
            vals[meth] = k['weighted_dataset_metric']
    seq = [u, vals.get('lightgbm_gain'), vals.get('deeplift'), vals.get('treeshap')]
    cells = fmt_group(seq, dp_of(ds))
    if ds == 'qm9':
        cells[1] = bold(cells[1])
    if ds == 'caco2':
        cells[2] = bold(cells[2])
    if ds == 'ginkgo_colloidal':
        cells[1] = bold(cells[1])
    rows.append((ds, [f"{NAME[ds]} & {METRIC[DSMETRIC[ds]]} & " + ' & '.join(cells) + ' \\\\']))
t6 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/complexity/c04_importance_as_metric/<dataset>.json, methods[<method>].knn["10"].{unweighted_dataset_metric,
% weighted_dataset_metric}, mean of 5 folds; read 2026-10-10 (files identical to kirby_results_2026-10-09). hERG blocking skipped (classification_not_supported).
% Empty cell: that importance method was not run on the dataset.
\begin{table}[htbp]
\centering
\begin{tabular}{@{}llrrrr@{}}
\toprule
 & & & \multicolumn{3}{c}{Weighted by} \\
\cmidrule(l){4-6}
Dataset & Metric & Unweighted & \makecell[r]{LightGBM\\gain} & DeepLIFT & TreeSHAP \\
\midrule
""" + grouped_rows(rows, 6) + r"""
\bottomrule
\end{tabular}
\caption[Importance weighting on all representations joined]{Prediction from the ten nearest neighbors in all representations joined, unweighted and with features weighted by each importance method. One row is one regression dataset; mean over five outer folds, in the dataset's own metric (arrow gives the direction). Buchwald--Hartwig RMSE is in percentage points of yield.}
\label{tab:similarity:joined_weighting}
\end{table}
"""
write('t6_joined_weighting', t6)

# ---------------------------------------------------------------- T7 kNN by setting + folds (s05), landscape
rows = []
DISTS = ['cosine', 'manhattan', 'minkowski_p0.5']
for ds in DS:
    d = load('s05_knn_prediction', ds)
    c = d['cells']
    dpp = dp_of(ds)
    def cm(arm, cond, dist=None, k=None):
        xs = []
        for key, v in c.items():
            a, cd, di, kk, sch = key.split('|')
            if a != arm or cd != cond:
                continue
            if dist and di != dist:
                continue
            if k and kk != k:
                continue
            # only settings both arms scored
            other = ('distance_hybrid' if arm == 'single_knn1' else 'single_knn1') + '|' + '|'.join([cd, di, kk, sch])
            if other in c and v.get('score_mean') is not None and c[other].get('score_mean') is not None:
                xs.append(v['score_mean'])
        return mean(xs)
    S, H = 'single_knn1', 'distance_hybrid'
    colsS = [cm(S, 'unweighted', dist=x) for x in DISTS] + [cm(S, 'unweighted', k=f'k{x}') for x in (5, 10, 20)] + [cm(S, 'unweighted'), cm(S, 'importance_weighted')]
    colsH = [cm(H, 'unweighted', dist=x) for x in DISTS] + [cm(H, 'unweighted', k=f'k{x}') for x in (5, 10, 20)] + [cm(H, 'unweighted'), cm(H, 'importance_weighted')]
    fs = c['single_knn1|unweighted|manhattan|k5|rank']['fold_scores']
    fh = c['distance_hybrid|unweighted|manhattan|k5|rank']['fold_scores']
    allS, allH = colsS + list(fs), colsH + list(fh)
    cellS, cellH = [], []
    for a, b in zip(allS, allH):
        x, y = fmt_group([a, b], dpp)
        cellS.append(x); cellH.append(y)
    rows.append((ds, [f"{NAME[ds]}, {METRIC[DSMETRIC[ds]]} & & & & & & & & & & & & & \\\\",
                      '\\quad Single & ' + ' & '.join(cellS) + ' \\\\',
                      '\\quad Hybrid & ' + ' & '.join(cellH) + ' \\\\']))
t7 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s05_knn_prediction/<dataset>.json, cells["<arm>|<condition>|<distance>|k<k>|<scheme>"].score_mean
% (mean of 5 folds) for arm single_knn1 and distance_hybrid, averaged over the settings both arms scored; fold columns = cells[...|unweighted|manhattan|k5|rank]
% .fold_scores; read 2026-10-10. The pooled gains in the text (LogD 0.078 to 0.023) also average the similarity hybrid
% (scratchpad/thesis_ch5_8/ch6_conditional_knn.py); the table shows the distance hybrid only. Minkowski is not scored on hERG blocking, LogD and QM9.
\begin{landscape}
\begin{table}[htbp]
\centering
\begin{tabular}{@{}lrrrrrrrrrrrrr@{}}
\toprule
 & \multicolumn{6}{c}{Unweighted, by setting} & \multicolumn{2}{c}{All settings} & \multicolumn{5}{c}{Setting fixed before the run, by fold} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-9}\cmidrule(l){10-14}
 & cosine & Manhattan & Minkowski & \makecell[r]{5\\neighbors} & \makecell[r]{10\\neighbors} & \makecell[r]{20\\neighbors} & \makecell[r]{Un-\\weighted} & Weighted & 1 & 2 & 3 & 4 & 5 \\
\midrule
""" + grouped_rows(rows, 14) + r"""
\bottomrule
\end{tabular}
\caption[Prediction from nearest neighbors by setting and by fold]{Prediction of each held-out molecule from its nearest training molecules, in the nearest-neighbor model's best single representation and in the distance hybrid. One row is one dataset and feature space, in the dataset's own metric (arrow gives the direction). By setting: mean over five outer folds and over the other settings. Fold columns: Manhattan distance, 5 neighbors weighted by the inverse of their rank, unweighted. Buchwald--Hartwig RMSE is in percentage points of yield.}
\label{tab:similarity:knn_by_setting}
\end{table}
\end{landscape}
"""
write('t7_knn_by_setting', t7)

# ---------------------------------------------------------------- T8 retrieval by setting + folds (s13), landscape
rows = []
for ds in DS:
    d = load('s13_clustering_retrieval', ds)
    r = d['retrieval']
    S = r['rep_knn1']['features_unweighted']
    H = r['hybrid_knn_distance1']['features_unweighted']
    dpp = dp_of(ds)
    def rm(space, dist=None, k=None):
        xs = []
        for key, v in space.items():
            di, kk, sch = key.split('|')
            if dist and di != dist:
                continue
            if k and kk != k:
                continue
            if key in S and key in H:
                xs.append(v['mean_cost'])
        return mean(xs)
    colsS = [rm(S, dist=x) for x in DISTS] + [rm(S, k=f'k{x}') for x in (5, 10, 20)]
    colsH = [rm(H, dist=x) for x in DISTS] + [rm(H, k=f'k{x}') for x in (5, 10, 20)]
    fx = 'manhattan|k5|rank'
    allS = colsS + list(S[fx]['fold_scores'])
    allH = colsH + list(H[fx]['fold_scores'])
    cellS, cellH = [], []
    for i, (a, b) in enumerate(zip(allS, allH)):
        x, y = fmt_group([a, b], dpp)
        if ds == 'ginkgo_colloidal' and i in (0, 1):
            x, y = bold(x), bold(y)
        cellS.append(x); cellH.append(y)
    rows.append((ds, [f"{NAME[ds]} & & & & & & & & & & & \\\\",
                      '\\quad Single & ' + ' & '.join(cellS) + ' \\\\',
                      '\\quad Hybrid & ' + ' & '.join(cellH) + ' \\\\']))
t8 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s13_clustering_retrieval/<dataset>.json, retrieval.{rep_knn1,hybrid_knn_distance1}.features_unweighted
% ["<distance>|k<k>|<scheme>"].mean_cost (mean of 5 folds), averaged over the settings both spaces scored; fold columns = ["manhattan|k5|rank"].fold_scores;
% read 2026-10-10. Same reading as scratchpad/thesis_ch5_8/ch6_conditional_retrieval.py. Single representations are unweighted only in s13.
\begin{landscape}
\begin{table}[htbp]
\centering
\begin{tabular}{@{}lrrrrrrrrrrr@{}}
\toprule
 & \multicolumn{6}{c}{By setting} & \multicolumn{5}{c}{Setting fixed before the run, by fold} \\
\cmidrule(lr){2-7}\cmidrule(l){8-12}
 & cosine & Manhattan & Minkowski & \makecell[r]{5\\neighbors} & \makecell[r]{10\\neighbors} & \makecell[r]{20\\neighbors} & 1 & 2 & 3 & 4 & 5 \\
\midrule
""" + grouped_rows(rows, 12) + r"""
\bottomrule
\end{tabular}
\caption[Label gap to the nearest neighbors by setting and by fold]{Label gap between each held-out molecule and its nearest training molecules (lower is better), in the nearest-neighbor model's best single representation and in the distance hybrid, unweighted. One row is one dataset and feature space. The gap is in the dataset's label units; on hERG blocking it is one minus the weighted share of neighbors in the molecule's class. By setting: mean over five outer folds and over the other settings. Fold columns: Manhattan distance, 5 neighbors weighted by the inverse of their rank.}
\label{tab:similarity:retrieval_by_setting}
\end{table}
\end{landscape}
"""
write('t8_retrieval_by_setting', t8)

# ---------------------------------------------------------------- T9 cliffs (s12, ECFP4 partition)
rows = []
for ds in ['qm9', 'caco2', 'adme_mdr1', 'herg_ki', 'logd', 'herg']:
    d = load('s12_cross_domain_cliffs', ds)
    rr = []
    for mod, v in d['summary']['models'].items():
        p = v['ecfp4_tanimoto']
        cs, ch = p['score_cliff']['base1']['mean'], p['score_cliff']['hybrid1']['mean']
        ns, nh = p['score_noncliff']['base1']['mean'], p['score_noncliff']['hybrid1']['mean']
        a = fmt_group([cs, ch], 3); b = fmt_group([ns, nh], 3)
        if ds in ('caco2', 'qm9') and mod == 'dnn':
            a[1] = bold(a[1])
        rr.append(f"{NAME[ds] if not rr else ''} & {MODEL[mod]} & {a[0]} & {a[1]} & {b[0]} & {b[1]} \\\\")
    rows.append((ds, rr))
t9 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s12_cross_domain_cliffs/<dataset>.json, summary.models[<model>].ecfp4_tanimoto.{score_cliff,score_noncliff}
% .{base1,hybrid1}.mean (folds with at least 10 cliff and 10 other held-out molecules, min_rows_per_fold_score); read 2026-10-10. These are the numbers the
% text quotes (Caco-2 0.340 to 0.288, QM9 0.242 to 0.280). The s04 file of the same name flags the same molecules; its model_summary means use every fold.
% QM9 cliff columns: folds 2 and 5 only (per_fold null elsewhere). MDR1-MDCK efflux is the 9 Oct 23:02 file; the text does not yet discuss it.
\begin{table}[htbp]
\centering
\begin{tabular}{@{}llrrrr@{}}
\toprule
 & & \multicolumn{2}{c}{Cliff molecules} & \multicolumn{2}{c}{Other molecules} \\
\cmidrule(lr){3-4}\cmidrule(l){5-6}
Dataset & Model & Single & Hybrid & Single & Hybrid \\
\midrule
\multicolumn{6}{@{}l}{\textit{RMSE $\downarrow$}} \\
""" + '\n'.join(sum([rr for ds, rr in rows if ds != 'herg'], [])) + r"""
\multicolumn{6}{@{}l}{\textit{AUROC $\uparrow$}} \\
""" + '\n'.join(sum([rr for ds, rr in rows if ds == 'herg'], [])) + r"""
\bottomrule
\end{tabular}
\caption[Prediction on activity-cliff molecules]{Prediction on held-out cliff molecules and on the other held-out molecules, by each downstream model's best single representation and its first hybrid. One row is one dataset and model; mean over the outer folds with at least 10 molecules of each kind (two folds on QM9, five elsewhere).}
\label{tab:similarity:cliffs}
\end{table}
"""
write('t9_cliffs', t9)

# ---------------------------------------------------------------- T10 distance against error (s06)
rows = []
for ds in DS:
    d = load('s06_applicability_domain', ds)
    dec = d['distance_error_correlation']['manhattan']['k5']
    rr = []
    for mod in d['models']:
        e = dec[f'{mod}_all_representations']['per_fold']
        cells = fmt_group([mean(e['knn_base1']), mean(e['hybrid'])], 3)
        if ds == 'herg_ki' and mod == 'svm':
            cells[1] = bold(cells[1])
        if ds == 'ginkgo_colloidal' and mod == 'dnn':
            cells[1] = bold(cells[1])
        rr.append(f"{NAME[ds] if not rr else ''} & {MODEL[mod]} & " + ' & '.join(cells) + ' \\\\')
    rows.append((ds, rr))
t10 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s06_applicability_domain/<dataset>.json, distance_error_correlation.manhattan.k5
% ["<model>_all_representations"].per_fold.{knn_base1,hybrid}, mean of 5 folds; read 2026-10-10 (errors of the model fitted on all representations joined).
\begin{table}[htbp]
\centering
\begin{tabular}{@{}llrr@{}}
\toprule
Dataset & Model & \makecell[r]{Single\\representation} & \makecell[r]{Distance\\hybrid} \\
\midrule
""" + grouped_rows(rows, 4) + r"""
\bottomrule
\end{tabular}
\caption[Distance to the nearest training molecules against prediction error]{Distance to the nearest training molecules against prediction error. One row is one dataset and downstream model. Values are the Spearman correlation between a held-out molecule's mean distance to its 5 nearest training molecules and the model's error on it (higher is better), mean over five outer folds, Manhattan distance, unweighted. The model is fitted on all representations joined; distance is measured in the nearest-neighbor model's best single representation and in the distance hybrid.}
\label{tab:similarity:error_distance}
\end{table}
"""
write('t10_error_distance', t10)

# ---------------------------------------------------------------- T11 retrieval picked again + T12 clustering (s13)
rows11, rows12 = [], []
for ds in DS:
    d = load('s13_clustering_retrieval', ds)
    r = d['fold_reselection_retrieval_unweighted_result']['knn']
    p = d['fold_reselection_retrieval_unweighted_vs_pca_result']['knn']
    dpp = dp_of(ds)
    cells = fmt_group([r['best_single_reselected'], r['hybrid_reselected'], p['best_single_reselected']], dpp)
    if ds == 'ddg_combined':
        cells[1] = bold(cells[1])
    if ds == 'abagym_EGFR':
        cells[2] = bold(cells[2])
    if ds == 'qm9':
        cells[1] = bold(cells[1])
    rows11.append((ds, [f"{NAME[ds]} & " + ' & '.join(cells) + ' \\\\']))
    c = d['clustering_delta_by_cluster_count']['clustering']
    cc = []
    for n in ('3', '5', '10'):
        x, y = fmt_group([c[n]['best_single_reselected'], c[n]['hybrid_reselected']], 3)
        if ds == 'logd' and n == '10':
            y = bold(y)
        if ds == 'ginkgo_colloidal' and n == '10':
            y = bold(y)
        cc += [x, y]
    rows12.append((ds, [f"{NAME[ds]} & " + ' & '.join(cc) + ' \\\\']))
t11 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s13_clustering_retrieval/<dataset>.json, fold_reselection_retrieval_unweighted_result.knn
% .{best_single_reselected,hybrid_reselected} and fold_reselection_retrieval_unweighted_vs_pca_result.knn.best_single_reselected, mean of 5 folds; read 2026-10-10.
% The hybrid side picks among every hybrid of the dataset (hybrid_knn_distance1, hybrid_similarity and the model hybrids), not the distance hybrid alone.
\begin{table}[htbp]
\centering
\begin{tabular}{@{}lrrr@{}}
\toprule
Dataset & \makecell[r]{Single\\representation} & Hybrid & \makecell[r]{Principal\\components} \\
\midrule
""" + grouped_rows(rows11, 4) + r"""
\bottomrule
\end{tabular}
\caption[Label gap to the nearest neighbors with the setting picked again]{Label gap between each held-out molecule and its nearest training molecules (lower is better), with the feature space and setting picked again for each outer fold, unweighted. One row is one dataset; mean over five outer folds, in the dataset's label units (on hERG blocking, one minus the weighted share of neighbors in the molecule's class). Single representation: the nearest-neighbor model's two best. Hybrid: every hybrid of the dataset. Principal components: of all representations, as many as the hybrid has features.}
\label{tab:similarity:retrieval}
\end{table}
"""
write('t11_retrieval', t11)
t12 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s13_clustering_retrieval/<dataset>.json, clustering_delta_by_cluster_count.clustering["3"|"5"|"10"]
% .{best_single_reselected,hybrid_reselected}, mean of 5 folds; read 2026-10-10. k-means under squared Euclidean distance (config.clustering_distance).
\begin{table}[htbp]
\centering
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{3 clusters} & \multicolumn{2}{c}{5 clusters} & \multicolumn{2}{c}{10 clusters} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Dataset & Single & Hybrid & Single & Hybrid & Single & Hybrid \\
\midrule
""" + grouped_rows(rows12, 7) + r"""
\bottomrule
\end{tabular}
\caption[Label coherence of k-means clusters]{Label coherence of k-means clusters of the held-out molecules (higher is better), in the single representation and the hybrid picked again for each outer fold. One row is one dataset; mean over five outer folds. Coherence is one minus the mean within-cluster standard deviation of the labels over their overall standard deviation; on hERG blocking, the mean share of the largest class in each cluster.}
\label{tab:similarity:clustering}
\end{table}
"""
write('t12_clustering', t12)

# ---------------------------------------------------------------- T13 label information (s03)
rows = []
for ds in DS:
    d = load('s03_feature_concentration', ds)
    S, P, Rr = [], [], []
    for f in d['folds']:
        for m, mm in f['models'].items():
            for h in mm.get('hybrids', []):
                if 'top1' not in (h.get('slots') or []):
                    continue
                for rep, r in h['per_rep'].items():
                    try:
                        a = r['summed_component_mi_selected']['held_out']; b = r['summed_component_mi_random']['held_out']; c = r['summed_component_mi_pca']['held_out']
                    except (KeyError, TypeError):
                        continue
                    if None in (a, b, c):
                        continue
                    S.append(a); Rr.append(b); P.append(c)
    cells = fmt_group([st.median(S), st.median(P), st.median(Rr)], 2)
    if ds == 'logd':
        cells = [bold(x) for x in cells[:2]] + cells[2:]
    rows.append((ds, [f"{NAME[ds]} & " + ' & '.join(cells) + ' \\\\']))
t13 = r"""% TODO: manual review
% SOURCE: /Volumes/seagate/kirby_results_2026-10-10/similarity/s03_feature_concentration/<dataset>.json, folds[*].models[*].hybrids[slot top1].per_rep[<rep>]
% .summed_component_mi_{selected,pca,random}.held_out; median over fold x model x representation units (QM9 58, LogD 100, hERG blocking 51, Buchwald--Hartwig 20);
% read 2026-10-10. Two decimals, as the text gives LogD (1.63, 1.68, 1.53); these are summed estimates in nats, not scores.
\begin{table}[htbp]
\centering
\begin{tabular}{@{}lrrr@{}}
\toprule
Dataset & \makecell[r]{Kept\\features} & \makecell[r]{Principal\\components} & \makecell[r]{Random\\subset} \\
\midrule
""" + grouped_rows(rows, 4) + r"""
\bottomrule
\end{tabular}
\caption[Label information in the features a hybrid keeps]{Label information in the features a hybrid keeps from each representation, against principal components and a random subset of the same number. One row is one dataset. Values are the median, over folds, models and representations, of a mutual-information estimate summed over at most 50 components of the held-out molecules, in nats, minus its mean under three label permutations (higher is more information).}
\label{tab:similarity:label_information}
\end{table}
"""
write('t13_label_information', t13)
