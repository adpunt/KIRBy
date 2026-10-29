"""Group table for Chapter 7 from rows_raw.json (rebuilt from raw files). Claude, 2026-10-09."""
import json, collections, sys
IN = ['qm9','caco2','herg_ki','logd','herg','herg_ic50','adme_mdr1','buchwald_hartwig','flab_thermo_tresanco','ddg_combined',
      'ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins']
DOM = {d: 'small molecules' for d in ['qm9','caco2','herg_ki','logd','herg','herg_ic50','adme_mdr1']}
DOM['buchwald_hartwig'] = 'reaction'
DOM.update({d: 'antibodies' for d in ['flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins']})
T, D, F, E = 'noise-robust training', 'mitigation, selected by a detector', 'mitigation, selected by a known flag', 'mitigation, every molecule'
M = {}
for e, ms, g in [
    ('n40', ['stack','seed_avg','retuned','tuned_clean_reused'], T),
    ('n09', ['heteroscedastic_gp','evidential_kernel','early_learning','self_paced_learning','meta_weight_net','mentornet','contrast_to_divide','DivideMix'], T),
    ('n33', ['huber_dnn'], T), ('n38', ['robust_loss'], T), ('n32', ['two_learner'], T), ('n29', ['per_record'], T),
    ('n19', ['ngboost_count_feature','count_feature','xgboost_interval_aft'], T),
    ('n20', ['search and fit on cleaned labels'], D), ('n21', ['best distance'], D), ('n13', ['detector weights'], D),
    ('n23', ['activity_cliff_aware','gaussian_process','hetero_gp','heteroscedastic_nn','ngboost'], D),
    ('n07', ['remove','label_smooth','downweight','oof_smooth','js_shrink'], D),
    ('n24', ['delete','downweight','huber_weight','label_smooth'], D),
    ('n12', ['importance-scaled features','unscaled features'], D), ('n35', ['downweight_pairs'], D), ('n31', ['downweight'], D),
    ('n18', ['remove','downweight','label_smooth','oof_smooth'], D), ('n02', ['neighbour-agreement weights'], D),
    ('n25', ['cliff_put_back'], D), ('n09', ['small_loss','confident_learning','distance_selection','multistage_cleaning'], D),
    ('n30', ['Censored EM','truncated_normal_mean','Constant lift','drop_clipped','per_molecule_lift'], F),
    ('n07', ['Constant lift'], F), ('n18', ['Constant lift'], F),
    ('n34', ['group_center','group_center_3plus','random_intercept'], F),
    ('n39', ['offset_removed','offset_removed_within_var_weights','weights_n19_fitted','weights_within_var'], F),
    ('n19', ['count_weighted','spread_pooled_weighted','shrunk'], F),
    ('n33', ['huber_teacher_blend'], E)]:
    for m in ms: M[(e, m)] = g
allrows = json.load(open(sys.argv[1]))
rows, outds, outm = [], collections.Counter(), collections.Counter()
for r in allrows:
    if r['ds'] not in IN: outds[r['ds']] += 1; continue
    g = M.get((r['exp'], r['method']))
    if g is None: outm[(r['exp'], r['method'])] += 1; continue
    r['group'] = g; r['dom'] = DOM[r['ds']]; r['L0'] = r['noise'].startswith('none added')
    r['scored'] = r['base'] is not None and r['hyb'] is not None
    rows.append(r)
seen = {(r['exp'], r['method']) for r in rows}
missing = [k for k in M if k not in seen]
def cls(r):
    if not r['scored']: return 'noscore'
    return {'yes': 'yes', 'no': 'neither', 'single representation only': 'single', 'hybrid only': 'hybrid'}[r['works']]
def stats(rs):
    c = collections.Counter(cls(r) for r in rs)
    y = [r for r in rs if cls(r) == 'yes']
    return [len(rs), c['yes'], sum(r['hb'] == 'yes' for r in y), sum(r['hb'] == 'no' for r in y), sum(r['hb'] == 'tie' for r in y),
            c['single'], c['hybrid'], c['neither'], c['noscore']]
H = ('| group | labels | rows | yes | yes, hybrid better after | yes, single better after | yes, tie after | single only | hybrid only | neither | no score |\n'
     '|---|---|---|---|---|---|---|---|---|---|---|')
def line(name, lab, rs):
    return f'| {name} | {lab} | ' + ' | '.join(str(x) for x in stats(rs)) + ' |'
GROUPS = [T, D, F, E]
MIT = [D, F, E]
def block(rs, qm9line=True):
    main = [r for r in rs if not (r['ds'] == 'qm9' and r['L0'])]
    out = [H]
    for name, sel in [(g, [g]) for g in GROUPS] + [('mitigation total (detector + flag + every molecule)', MIT), ('all four groups', GROUPS)]:
        s = [r for r in main if r['group'] in sel]
        out += [line(name, 'as delivered', [r for r in s if r['L0']]), line(name, 'added damage', [r for r in s if not r['L0']]),
                line(name, 'both', s)]
    return out, main
md = ['# Overall: does it work, by group (Chapter 7 signed-off datasets and methods)', '',
      'Written by Claude 2026-10-09. Fold 0 only; each method\'s best setting picked on the held-out score.', '',
      'One row = one dataset x noise condition x model x method, comparing the best single representation and the best hybrid, each before and after the method. '
      '"yes" = the method improved both the single representation and the hybrid (and beat its control, where the table has one). '
      '"hybrid better after" = the hybrid\'s score after the method beats the single representation\'s score after the method (4 decimals). '
      '"no score" = one or both of the two cells is n/a. "as delivered" = no noise added (level 0); "added damage" = the main damaged level of each noise kind.', '',
      '## A. All 14 datasets (QM9 labels-as-delivered rows left out; they are on their own below)', '']
o, main = block(rows); md += o
q0 = [r for r in rows if r['ds'] == 'qm9' and r['L0']]
md += ['', 'QM9, labels as delivered (no noise added). Any "yes" here is a gain on clean labels, so it is not noise removal.', '', H]
md += [line(g, 'as delivered', [r for r in q0 if r['group'] == g]) for g in GROUPS]
md += [line('mitigation total', 'as delivered', [r for r in q0 if r['group'] in MIT])]
md += ['', '## B. By domain', '',
       'Small molecules = qm9 (added-damage rows only), caco2, herg_ki, logd, herg, herg_ic50, adme_mdr1. Reaction = buchwald_hartwig. '
       'Antibodies = flab_thermo_tresanco, ddg_combined, ginkgo_titer, ginkgo_pr_cho, ginkgo_colloidal, ginkgo_acsins.', '']
for dom in ['small molecules', 'reaction', 'antibodies']:
    o, _ = block([r for r in rows if r['dom'] == dom]); md += [f'### {dom}', ''] + o + ['']
# C
def before(r):
    a, b = round(r['base'][0], 3), round(r['hyb'][0], 3)
    if a == b: return 'tie'
    return 'yes' if ((b < a) if r['low'] else (b > a)) else 'no'
HC = ('| domain | group | labels | rows with scores | hybrid better before and after | hybrid worse before, better after | hybrid better before, worse after | hybrid worse before and after | tie before or after |\n'
      '|---|---|---|---|---|---|---|---|---|')
md += ['## C. Hybrid against single representation, before and after the method', '',
       'One row of this table counts table rows with both scores. "Before" compares the two scores with no method, at the 3 decimals the page prints; '
       '"after" is the page\'s own column, at 4 decimals. QM9 labels-as-delivered rows left out.', '', HC]
for dom in ['small molecules', 'reaction', 'antibodies', 'all domains']:
    for gname, sel in [('noise-robust training', [T]), ('mitigation total', MIT)]:
        for lab, f in [('as delivered', lambda r: r['L0']), ('added damage', lambda r: not r['L0']), ('both', lambda r: True)]:
            rs = [r for r in main if r['scored'] and r['group'] in sel and f(r) and (dom == 'all domains' or r['dom'] == dom)]
            c = collections.Counter()
            for r in rs:
                b, a = before(r), r['hb']
                c['tie' if 'tie' in (a, b) else b + a] += 1
            md.append(f'| {dom} | {gname} | {lab} | {len(rs)} | {c["yesyes"]} | {c["noyes"]} | {c["yesno"]} | {c["nono"]} | {c["tie"]} |')
# D
SHORT = {'buchwald_hartwig': 'BH', 'flab_thermo_tresanco': 'tresanco', 'ddg_combined': 'ddg', 'ginkgo_titer': 'titer', 'ginkgo_pr_cho': 'prCHO',
         'ginkgo_colloidal': 'colloidal', 'ginkgo_acsins': 'acsins', 'herg_ki': 'herg_ki', 'herg_ic50': 'herg_ic50'}
md += ['', '## D. Per method (signed-off DEEP and BROAD only)', '',
       'rows = rows with both scores, QM9 labels-as-delivered left out; yes as above. "QM9 clean" = yes of rows on QM9 labels as delivered. '
       'Datasets listed are those with at least one yes row (qm9 = a yes under added damage).', '',
       '| group | exp | method | rows | yes | yes as delivered | yes added damage | QM9 clean: yes of rows | datasets with a yes row |', '|---|---|---|---|---|---|---|---|---|']
for (e, m), g in M.items():
    rs = [r for r in main if (r['exp'], r['method']) == (e, m) and r['scored']]
    q = [r for r in q0 if (r['exp'], r['method']) == (e, m) and r['scored']]
    y = [r for r in rs if r['works'] == 'yes']
    ds = [SHORT.get(d, d) for d in IN if any(r['ds'] == d for r in y)]
    md.append(f'| {g} | {e} | {m} | {len(rs)} | {len(y)} | {sum(r["L0"] for r in y)} | {sum(not r["L0"] for r in y)} | '
              f'{sum(r["works"]=="yes" for r in q)} of {len(q)} | {", ".join(ds) or "-"} |')
md += ['', '## Rows dropped', '',
       f'- Rows on datasets outside the 14: {sum(outds.values())} ({", ".join(f"{k} {v}" for k, v in sorted(outds.items()))}).',
       f'- Rows on the 14 datasets for methods not in scope: {sum(outm.values())}: ' + '; '.join(f'{e} {m} {v}' for (e, m), v in sorted(outm.items())) + '.',
       f'- In-scope methods with no rows on the page: {missing or "none"}.',
       f'- In-scope rows with no score (counted in the "no score" column, not in yes/neither): {sum(not r["scored"] for r in rows)}.']
open('GROUP_TABLE.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
