"""Write the three Chapter 7 count tables (LaTeX + CSV) from the rows.json that build.py reads.

    python tests/analysis/ch7_sources/group_table/tables_tex.py <rows.json> <out_dir>

rows.json is written by ch7_numbers.py's group-table step (rebuild_group_table) into its temp dir:
meeting_methods.py run read-only on <results>/noise (outer fold 0), then parse.py. The counting
rules are build.py's: one row = one dataset x noise condition x model x method; "both" = the
page's "yes" verdict (improved the best single representation and the best hybrid, and beat its
control where one exists). Each method's best setting is picked on the held-out score.

Outputs: tab_noise_bygroup, tab_noise_mitigation_broad, tab_noise_robust_broad (.tex and .csv).
Method rows: signed-off BROAD methods only (THESIS_OUTLINE.md "CHAPTER 7 METHODS, SIGNED OFF").
The n09 sweep methods are left out of the broad tables; they have their own table
(tab:noise:res:sweep). Written by Claude, 2026-10-10.
"""
import collections, csv, json, os, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

IN = ['qm9', 'caco2', 'herg_ki', 'logd', 'herg', 'herg_ic50', 'adme_mdr1', 'buchwald_hartwig', 'flab_thermo_tresanco',
      'ddg_combined', 'ginkgo_titer', 'ginkgo_pr_cho', 'ginkgo_colloidal', 'ginkgo_acsins']
T, D, F, E = ('noise-robust training', 'mitigation, selected by a detector', 'mitigation, selected by a known flag',
              'mitigation, every molecule')

# same in-scope map as build.py (kept identical; a change there must be copied here)
M = {}
for e, ms, g in [
    ('n40', ['stack', 'seed_avg', 'retuned', 'tuned_clean_reused'], T),
    ('n09', ['heteroscedastic_gp', 'evidential_kernel', 'early_learning', 'uncertainty_curriculum', 'meta_weight_net',
             'mentornet', 'contrast_to_divide', 'DivideMix'], T),
    ('n33', ['huber_dnn'], T), ('n38', ['robust_loss'], T), ('n32', ['two_learner'], T), ('n29', ['per_record'], T),
    ('n19', ['ngboost_count_feature', 'count_feature', 'xgboost_interval_aft'], T),
    ('n20', ['search and fit on cleaned labels'], D), ('n21', ['best distance'], D), ('n13', ['detector weights'], D),
    ('n23', ['activity_cliff_aware', 'gaussian_process', 'hetero_gp', 'heteroscedastic_nn', 'ngboost'], D),
    ('n07', ['remove', 'label_smooth', 'downweight', 'oof_smooth', 'js_shrink'], D),
    ('n24', ['delete', 'downweight', 'huber_weight', 'label_smooth'], D),
    ('n12', ['importance-scaled features', 'unscaled features'], D), ('n35', ['downweight_pairs'], D),
    ('n31', ['downweight'], D), ('n18', ['remove', 'downweight', 'label_smooth', 'oof_smooth'], D),
    ('n02', ['neighbour-agreement weights'], D), ('n25', ['cliff_put_back'], D),
    ('n09', ['small_loss', 'confident_learning', 'distance_selection', 'multistage_cleaning'], D),
    ('n30', ['Censored EM', 'truncated_normal_mean', 'Constant lift', 'drop_clipped', 'per_molecule_lift'], F),
    ('n07', ['Constant lift'], F), ('n18', ['Constant lift'], F),
    ('n34', ['group_center', 'group_center_3plus', 'random_intercept'], F),
    ('n39', ['offset_removed', 'offset_removed_within_var_weights', 'weights_n19_fitted', 'weights_within_var'], F),
    ('n19', ['count_weighted', 'spread_pooled_weighted', 'shrunk'], F),
    ('n33', ['huber_teacher_blend'], E)]:
    for m in ms:
        M[(e, m)] = g

# signed-off BROAD rows, in table order: (block heading, [(exp, method, thesis name)])
MIT_BROAD = [
    ('Selected by a detector', [
        ('n12', 'importance-scaled features', 'Neighbor weights, importance-scaled features'),
        ('n12', 'unscaled features', 'Neighbor weights, unscaled features'),
        ('n07', 'oof_smooth', 'Move toward prediction; Forest residual'),
        ('n07', 'js_shrink', 'Score-scaled shrink; Forest residual'),
        ('n24', 'downweight', "Score down-weight; Forest residual or Neighbor-label gap"),
        ('n24', 'huber_weight', "Forest residual or Neighbor-label gap; Huber-style weight"),
        ('n24', 'label_smooth', "Neighbor-mean smoothing; Forest residual or Neighbor-label gap"),
        ('n35', 'downweight_pairs', 'Close-pair weights'),
        ('n31', 'downweight', 'Top-share down-weight; LightGBM residual'),
        ('n18', 'remove', 'Delete flagged; Forest residual, labels as delivered only'),
        ('n02', 'neighbour-agreement weights', 'Neighbor-agreement weights'),
        ('n25', 'cliff_put_back', 'Cliff put-back'),
    ]),
    ('Selected by a known flag', [
        ('n34', 'group_center', 'Scaffold-family offset'),
        ('n34', 'group_center_3plus', 'Scaffold-family offset, families of 3 or more'),
        ('n34', 'random_intercept', 'Scaffold-family offset, random intercept'),
        ('n39', 'offset_removed', "Source offset"),
        ('n39', 'offset_removed_within_var_weights', 'Source offset, then Source weights'),
        ('n39', 'weights_n19_fitted', 'Source weights, from error against number of sources'),
        ('n39', 'weights_within_var', 'Source weights, by spread within and between sources'),
        ('n19', 'count_weighted', 'Count and spread weights, by count'),
        ('n19', 'spread_pooled_weighted', 'Count and spread weights, inverse error variance, pooled spread'),
        ('n19', 'shrunk', 'Shrink by spread'),
    ]),
    ('Every molecule', [
        ('n33', 'huber_teacher_blend', 'Huber blend'),
    ]),
]
ROB_BROAD = [
    ('Several datasets', [
        ('n09', 'heteroscedastic_gp', 'Heteroscedastic GP'),
        ('n09', 'evidential_kernel', 'Evidential network'),
        ('n33', 'huber_dnn', 'Huber loss, DNN'),
        ('n38', 'robust_loss', "Each model's robust loss"),
        ('n32', 'two_learner', 'Two-learner selection'),
    ]),
    ('Repeat measurements, hERG IC$_{50}$ only', [
        ('n29', 'per_record', 'One row per measurement'),
        ('n19', 'count_feature', 'Log-count feature'),
        ('n19', 'ngboost_count_feature', 'NGBoost with the log-count feature'),
        ('n19', 'xgboost_interval_aft', "Interval model"),
    ]),
]


def load(path):
    rows = []
    for r in json.load(open(path)):
        if r['ds'] not in IN:
            continue
        g = M.get((r['exp'], r['method']))
        if g is None:
            continue
        r['group'] = g
        r['L0'] = r['noise'].startswith('none added')
        r['scored'] = r['base'] is not None and r['hyb'] is not None
        rows.append(r)
    return rows


def cls(r):
    if not r['scored']:
        return 'noscore'
    return {'yes': 'yes', 'no': 'neither', 'single representation only': 'single', 'hybrid only': 'hybrid'}[r['works']]


def stats(rs):
    """build.py's stats(): rows, yes, yes hybrid better after, yes single better after, yes tie,
    single only, hybrid only, neither, no score."""
    c = collections.Counter(cls(r) for r in rs)
    y = [r for r in rs if cls(r) == 'yes']
    return [len(rs), c['yes'], sum(r['hb'] == 'yes' for r in y), sum(r['hb'] == 'no' for r in y),
            sum(r['hb'] == 'tie' for r in y), c['single'], c['hybrid'], c['neither'], c['noscore']]


def wrap(h, width):
    """Split a header into lines of at most `width` characters at spaces, joined by \\\\ for \\makecell."""
    lines = []
    for w in h.split(' '):
        if lines and len(lines[-1]) + 1 + len(w) <= width:
            lines[-1] += ' ' + w
        else:
            lines.append(w)
    return '\\\\'.join(lines)


def tex_esc(s):
    return s.replace('&', r'\&').replace('%', r'\%')


def bygroup(rows, out):
    main = [r for r in rows if not (r['ds'] == 'qm9' and r['L0'])]
    q0 = [r for r in rows if r['ds'] == 'qm9' and r['L0']]
    MIT = [D, F, E]
    NAMES = [(T, [T], 'Noise-robust training'), (D, [D], 'Mitigation, selected by a detector'),
             (F, [F], 'Mitigation, selected by a known flag'), (E, [E], 'Mitigation, every molecule'),
             ('mit', MIT, 'Mitigation, all three'), ('all', [T] + MIT, 'All four groups')]
    LAB = [('as delivered', lambda r: r['L0']), ('added damage', lambda r: not r['L0']), ('both', lambda r: True)]
    LABTEX = {'as delivered': 'As delivered', 'added damage': 'Added damage', 'both': 'All'}
    csvrows, body = [], []
    body.append(r'\multicolumn{11}{@{}l}{\textit{All 14 datasets; QM9 with no noise added left out}} \\')
    for key, sel, name in NAMES:
        if key == 'mit':
            body.append(r'\midrule')
        for i, (lab, f) in enumerate(LAB):
            s = stats([r for r in main if r['group'] in sel and f(r)])
            csvrows.append(['all 14 datasets', name, lab] + s)
            body.append(f'{name if i == 0 else ""} & {LABTEX[lab]} & ' + ' & '.join(str(x) for x in s) + r' \\')
        if key not in ('mit', 'all') and key != E:
            body.append(r'\addlinespace')
    body.append(r'\midrule')
    body.append(r'\multicolumn{11}{@{}l}{\textit{QM9, no noise added (computed labels)}} \\')
    for key, sel, name in NAMES[:5]:
        s = stats([r for r in q0 if r['group'] in sel])
        csvrows.append(['qm9 no noise added', name, 'as delivered'] + s)
        body.append(f'{name} & As delivered & ' + ' & '.join(str(x) for x in s) + r' \\')
    head = ['Group', 'Labels', 'Conditions', 'Both improved', 'Both, hybrid ahead after',
            'Both, single ahead after', 'Both, tie after', 'Single only', 'Hybrid only', 'Neither', 'No score']
    # [2026-10-10] Longtable at the document's table size (no \footnotesize, no adjustbox shrinking). Eleven
    # columns do not fit the portrait text width, so additional_files_body.tex opens a landscape page before the
    # heading; a longtable breaks over two pages if needed (a rotated float ran off the page). Group column wraps;
    # count headers wrap at 8 characters; header repeated.
    hrow = ' & '.join(h if i < 2 else r'\makecell[r]{' + wrap(h, 8) + '}' for i, h in enumerate(head)) + r' \\'
    tex = [r'\begin{longtable}{@{}P{4.5cm}lrrrrrrrrr@{}}',
           r'\caption[Does noise handling work, by group.]{Conditions where a method improved the best single '
           r'representation, the hybrid, both (and beat its control, where one exists) or neither. One condition is one dataset, noise condition, model and method. '
           r"``As delivered'' means no noise added; ``added damage'' means the main damaged level of each noise kind. "
           r"``Hybrid ahead after'' means the hybrid scores better than the single representation after the method. "
           r'Outer fold 0.}',
           r'\label{tab:noise:res:bygroup}\\',
           r'\toprule', hrow, r'\midrule', r'\endfirsthead',
           r'\toprule', hrow, r'\midrule', r'\endhead',
           r'\bottomrule', r'\endfoot'] + body + [r'\end{longtable}']
    write(out, 'tab_noise_bygroup', tex, ['block', 'group', 'labels'] + [h.lower() for h in head[2:]], csvrows)


def broad(rows, blocks, name, label, short, caption, out):
    main = [r for r in rows if not (r['ds'] == 'qm9' and r['L0'])]
    head = ['Method', 'Datasets', 'Conditions', 'Both improved', 'Single only', 'Hybrid only', 'Neither', 'No score']
    body, csvrows = [], []
    for bi, (bname, items) in enumerate(blocks):
        if bi:
            body.append(r'\midrule')
        body.append(r'\multicolumn{8}{@{}l}{\textit{' + bname + r'}} \\')
        for e, m, nice in items:
            rs = [r for r in main if (r['exp'], r['method']) == (e, m)]
            if not rs:
                raise SystemExit(f'no rows for {e} {m}')
            s = stats(rs)
            nds = len({r['ds'] for r in rs if r['scored']})
            cells = [nds, s[0], s[1], s[5], s[6], s[7], s[8]]
            dsl = ' '.join(d for d in IN if any(r['ds'] == d and r['scored'] for r in rs))
            csvrows.append([bname, nice, e, m] + cells + [dsl])
            body.append(tex_esc(nice) + ' & ' + ' & '.join(str(x) for x in cells) + r' \\')
    # [2026-10-10] Document's table size (no \footnotesize, no adjustbox shrinking). A longtable, so a long
    # table breaks across pages instead of running off the foot of one; the method column wraps (P, defined in
    # preamble.tex), count headers wrap at 6 characters, column gaps 3pt (font size unchanged), header repeated.
    hrow = ' & '.join(h if i == 0 else r'\makecell[r]{' + wrap(h, 6) + '}' for i, h in enumerate(head)) + r' \\'
    tex = [r'{\setlength{\tabcolsep}{3pt}',
           r'\begin{longtable}{@{}P{4.95cm}rrrrrrr@{}}',
           r'\caption[' + short + ']{' + caption + '}', r'\label{' + label + r'}\\',
           r'\toprule', hrow, r'\midrule', r'\endfirsthead',
           r'\toprule', hrow, r'\midrule', r'\endhead',
           r'\bottomrule', r'\endfoot'] + body + [
           r'\end{longtable}}']
    write(out, name, tex, ['block', 'method', 'experiment', 'method_code', 'datasets', 'conditions', 'both improved',
                           'single only', 'hybrid only', 'neither', 'no score', 'datasets_with_scores'], csvrows)


def write(out, name, tex, header, csvrows):
    with open(os.path.join(out, name + '.tex'), 'w') as fh:
        fh.write('% Written by tests/analysis/ch7_sources/group_table/tables_tex.py (Claude, 2026-10-10).\n')
        fh.write('% Counts from rows.json of ch7_numbers.py (group-table step) on /Volumes/seagate/kirby_results_2026-10-10b/noise, outer fold 0.\n')
        fh.write('\n'.join(tex) + '\n')
    with open(os.path.join(out, name + '.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(csvrows)
    print('wrote', os.path.join(out, name + '.tex'))


CAP = (r"One row is one method. ``Datasets'' counts datasets with a score; a condition is one dataset, noise condition "
       r'and model. The other columns count conditions where the method improved both the best single representation and '
       r'the hybrid (and beat its control, where one exists), only one of them, or neither. QM9 with no noise added is left out. Outer fold 0.')

if __name__ == '__main__':
    rows = load(sys.argv[1])
    out = sys.argv[2]
    bygroup(rows, out)
    broad(rows, MIT_BROAD, 'tab_noise_mitigation_broad', 'tab:noise:res:mitigation_broad',
          'Broad noise mitigation methods.', CAP, out)
    broad(rows, ROB_BROAD, 'tab_noise_robust_broad', 'tab:noise:res:robust_broad',
          'Broad noise-robust training methods.', CAP, out)
