"""Collect every raw value the Chapter 7 tables need into vals.json (read-only on the result files)."""
import sys, json, statistics, os; sys.path.insert(0, sys.argv[1]); from load import c7
mean = c7.mean
V = {}
raw = json.load(open(sys.argv[1] + '/raw.json'))
# --- T1 n07 + n23 hERG flips
rs = c7._a_dedupe(c7.rows('n07_remediation', 'herg__confusion_directed__*.rows.jsonl', where=('noise', 'full'), model='xgboost', level=0.1))
f0 = [x for x in rs if x['fold'] == 0]
sm = {'hybrid': ('k=10|scheme=inverse_square|flag_share=0.1', 'forest_proximity|none|forest'),
      'base_1': ('k=20|scheme=inverse_square|flag_share=0.1', 'hybrid_similarity|none|manhattan')}
for sp in ('base_1', 'hybrid'):
    D, rem, _ = c7._a_cell(f0, model_input=sp, method='remove', method_setting='flag_share=0.1')
    V[f'T1 none {sp}'] = D; V[f'T1 delete {sp}'] = rem
    V[f'T1 move {sp}'] = c7._a_cell(f0, model_input=sp, method='label_smooth', method_setting=sm[sp][0], similarity_setting=sm[sp][1])[1]
    V[f'T1 down {sp}'] = c7._a_cell(f0, model_input=sp, method='downweight')[1]
d = c7.J('noise', 'n23_uncertainty_cleaning', 'herg__confusion_directed.json')
q = d['per_regime']['injected']['0.1']
for sp in ('base_1', 'hybrid'):
    for det in ('activity_cliff_aware', 'gaussian_process', 'heteroscedastic_nn', 'ngboost', 'hetero_gp'):
        V[f'T1 {det} {sp}'] = mean(q[sp][det]['drop@0.1']['xgboost'].values())
    V[f'T1 n23none {sp}'] = mean(q[sp]['ngboost']['none']['xgboost'].values())
fr = c7.rows('n23_uncertainty_cleaning', 'herg__confusion_directed__*.rows.jsonl', where=('noise', 'full'), model='xgboost', level=0.1)
for sp in ('base_1', 'hybrid'):
    s = [x for x in fr if x['fold'] == 1 and x['draw'] == 0 and c7.cfg(x['model_input']) == sp and x['method_setting'].endswith('drop@0.1')]
    V[f'T1 f1 none {sp}'] = s[0]['score_D']
    best = {}
    for x in s: best[x['method']] = max(best.get(x['method'], -1), x['score_Dprime'])
    for k, v in best.items(): V[f'T1 f1 {k} {sp}'] = v
# --- T2 n24
d = c7.J('noise', 'n24_delete_or_keep', 'herg__confusion_directed.json')
x = d['per_level']['0.1']['hybrid']['neighbor_residual@screen1']['0.1']
V['T2 herg nbr none'] = mean(x['no_cleaning']['xgboost'].values())
V['T2 herg nbr delete'] = mean(x['delete/remove']['xgboost'].values())
V['T2 herg nbr random'] = max(mean(v['xgboost'].values()) for a, v in x.items() if a.startswith('delete_at_random'))
n24 = c7._a_dedupe(c7.rows('n24_delete_or_keep', 'herg__confusion_directed__*.rows.jsonl', where=('noise', 'full'), model='xgboost', level=0.1))
for fo in (0, 1, 2):
    c = c7._a_n24_fold(n24, fo, 'hybrid', 'xgboost', 'cross_val|flag_rate=0.1', 0.1)
    V[f'T2 herg cv f{fo}'] = c[:3]; V[f'T2 herg cv f{fo} n'] = c[4]
n24 = c7._a_dedupe(c7.rows('n24_delete_or_keep', 'logd__outlier_p05__*.rows.jsonl', where=('noise', 'full'), model='dnn', level=1.0))
for fo, sp, rate in ((0, 'hybrid__general2', '0.05'), (0, 'hybrid', '0.1'), (1, 'hybrid__general2', '0.05')):
    c = c7._a_n24_fold(n24, fo, sp, 'dnn', f'cross_val|flag_rate={rate}', 1.0)
    V[f'T2 logd {sp} {rate} f{fo}'] = c[:3]
d = c7.J('noise', 'n24_delete_or_keep', 'buchwald_hartwig__censoring.json')
for sp in ('hybrid', 'base_2'):
    x = d['per_level']['0'][sp]['neighbor_residual@screen1']['0.2']
    V[f'T2 bh {sp}'] = (mean(x['no_cleaning']['lightgbm'].values()), mean(x['delete/remove']['lightgbm'].values()),
                        min(mean(v['lightgbm'].values()) for a, v in x.items() if a.startswith('delete_at_random')))
# --- T3 LogD n07 + n23
rs = c7._a_dedupe(c7.rows('n07_remediation', 'logd__outlier_p05__*.rows.jsonl', where=('noise', 'full'), model='dnn', level=1.0))
f0 = [x for x in rs if x['fold'] == 0]
LS = dict(method_setting='k=20|scheme=inverse_square', similarity_setting='hybrid_similarity|none|manhattan')
for sp in ('base_1', 'hybrid'):
    D, rem, _ = c7._a_cell(f0, model_input=sp, method='remove', method_setting='flag_share=0.05')
    V[f'T3 none {sp}'] = D; V[f'T3 delete {sp}'] = rem
    V[f'T3 down {sp}'] = c7._a_cell(f0, model_input=sp, method='downweight')[1]
    V[f'T3 smooth {sp}'] = c7._a_cell(f0, model_input=sp, method='label_smooth', **LS)[1]
    V[f'T3 shrink {sp}'] = c7._a_cell(f0, model_input=sp, method='js_shrink')[1]
d2 = c7.J('noise', 'n23_uncertainty_cleaning', 'logd__outlier_p05.json')
inj = d2['per_regime']['injected']['1']
for sp in ('base_1', 'hybrid'):
    V[f'T3 n23none {sp}'] = mean(inj[sp]['gaussian_process']['none']['dnn'].values())
    for det in ('activity_cliff_aware', 'gaussian_process', 'heteroscedastic_nn', 'ngboost', 'hetero_gp'):
        V[f'T3 {det} {sp}'] = mean(inj[sp][det]['drop@0.1']['dnn'].values())
# --- T5 weighting
V['T5'] = raw['N7-07']
# --- T6 n18
cells = {'hk1': ('herg_ki', 'base_1', 'neighbor_residual', '0.2', 'dnn', 'label_smooth'), 'hkh': ('herg_ki', 'hybrid', 'cross_val', '0.05', 'dnn', 'label_smooth'),
         'hb1': ('herg', 'base_1', 'neighbor_residual', '0.01', 'xgboost', 'label_smooth'), 'hbh2': ('herg', 'hybrid__general2', 'neighbor_residual', '0.01', 'xgboost', 'label_smooth'),
         'cc2': ('caco2', 'base_2', 'cross_val', '0.2', 'dnn', 'downweight'), 'ld2': ('logd', 'base_2', 'neighbor_residual', '0.05', 'dnn', 'downweight'),
         'sm1': ('ginkgo_colloidal', 'base_1', 'cross_val', '0.2', 'svm', 'remove'), 'smh2': ('ginkgo_colloidal', 'hybrid__general2', 'cross_val', '0.2', 'svm', 'remove'),
         'sm1n': ('ginkgo_colloidal', 'base_1', 'neighbor_residual', '0.05', 'dnn', 'oof_smooth'), 'smhn': ('ginkgo_colloidal', 'hybrid', 'cross_val', '0.01', 'dnn', 'oof_smooth'),
         'dd1': ('ddg_combined', 'base_1', 'cross_val', '0.2', 'dnn', 'label_smooth'), 'ddh': ('ddg_combined', 'hybrid', 'cross_val', '0.2', 'dnn', 'label_smooth'),
         'ti2': ('ginkgo_titer', 'base_2', 'cross_val', '0.01', 'rf', 'remove'), 'tih2': ('ginkgo_titer', 'hybrid__general2', 'cross_val', '0.01', 'rf', 'remove')}
for k, c in cells.items():
    r = c7._B_n18(*c); V[f'T6 {k}'] = [r[0], r[1], r[2]['0'], r[2]['1'], r[2]['2']]
# --- T7
V['T7'] = raw['N7-10']
# --- T8 n07 detection
T8 = []
for ds in c7.CH7_DATASETS:
    for kind in c7._n07_kinds('noise', ds):
        x = (c7.n07_detection('noise', ds, kind) or {}).get((0, c7.main_level_of(kind)))
        if not x: continue
        aps = sorted(x['cliff'].values())
        T8.append([ds, kind, x['chance'], x['resid'], aps[-1] if aps else None, statistics.median(aps) if aps else None])
V['T8'] = T8
F = c7._n21_files('view')
V['T9'] = [[ds, kind, res['chance'], res['best'].get('plain'), res['best'].get('importance-weighted'), res['best'].get('forest proximity'), res['best'].get('hybrid')]
           for (ds, kind), (lv, res, folds) in sorted(F.items()) if ds in c7.CH7_DATASETS]
d0 = c7.CTX.path('noise', 'n23_uncertainty_cleaning'); T10 = {}
for f in c7.files(d0, '*__*.json'):
    if 'fingerprint' in f: continue
    ds, kind = f[:-5].split('__', 1)
    if ds not in c7.CH7_DATASETS: continue
    x = c7.n23_sources(c7.jload(os.path.join(d0, f)), c7._c_lvkey(c7.main_level_of(kind)))
    if x: T10.setdefault(f'{ds}|{kind}', {}).update(gp=x[0].get('gaussian_process'), cliff=x[0].get('activity_cliff_aware'))
names = set()
for sub in ('', '_checkpoints'):
    for f in c7.files(os.path.join(c7.CTX.path('noise', 'n32_iterative_and_dual_selection'), sub), '*__*.json'): names.add(f[:-5])
for nm in sorted(names):
    ds, kind = nm.split('__', 1)
    if ds not in c7.CH7_DATASETS: continue
    det_all, src, folds = c7.n32_detection(ds, kind)
    lv = c7._c_lvkey(c7.main_level_of(kind))
    if det_all and lv in det_all:
        fm = c7.n32_family_means(det_all[lv])
        T10.setdefault(f'{ds}|{kind}', {}).update(resid=fm.get('oof_residual', (None,))[0], shap=fm.get('knn_shapley', (None,))[0],
                                                  cleanlab=fm.get('cleanlab', (None,))[0], src=src)
V['T10'] = T10
T11 = []
for ds in ('caco2', 'herg_ki'):
    for cond in ('outlier_only_p05', 'shift_only_p10', 'sparse_dprime_up_p10'):
        for sh in ('share0.05', 'share0.1', 'share0.2'):
            cells_ = c7.n31_cells(ds, cond, '1', sh)
            if not cells_: continue
            pr = [p for _, _, p, _ in cells_ if p is not None]
            ops = {}
            for op in ('remove', 'downweight', 'rewrite', 'bayes_shrink'):
                v = [b[op] for *_, b in cells_ if b.get(op) is not None]
                if v: ops[op] = statistics.median(v)
            T11.append([ds, cond, sh, mean(pr), ops])
V['T11'] = T11
# --- n30
for ds in ('logd', 'qm9', 'herg_ki', 'caco2', 'buchwald_hartwig', 'flab_thermo_tresanco'):
    cells_ = c7.n30_cells(ds)
    V[f'n30 {ds}'] = {f'{k[1]}|{k[2]}|{k[3]}': v for k, v in cells_.items() if k[0] == '0.25' and k[4] == 0}
for ds, src in (('logd', ('full_rows',)), ('caco2', ('full_rows',))):
    cells_ = c7.n30_cells(ds, sources=src)
    V[f'n30full {ds}'] = {f'{k[4]}|{k[3]}': v for k, v in cells_.items() if k[0] == '0.25' and k[1] == 'hybrid' and k[2] == 'dnn'}
# --- n34
V['T15'] = [[x['dataset'], x['draw'], x['method'], x['score_D'], x['score_Dprime'], x['score_clean']]
            for ds in ('caco2', 'logd', 'herg_ki')
            for x in c7.rows('n34_group_random_effects', f'{ds}__grouped_shifted*.rows.jsonl', model='dnn', level=1.0, fold=0, model_input='hybrid')]
# --- n19 / n29
rs = [x for x in c7.rows('n19_measurement_count', 'herg_ic50*.rows.jsonl', where=('noise', 'full'), dataset='herg_ic50')
      if x.get('method_setting') == 'label_build=averaged' and c7._cfg_any(x.get('model_input')) == 'hybrid' and x['fold'] == 0]
V['T17'] = [[x['model'], x['method'], x['draw'], x['score_D'], x['score_Dprime']] for x in rs]
d = c7.J('noise', 'n29_document_bias', 'herg_ic50.json')['scores']['all']['hybrid']
V['T17 n29'] = {m: (d['mean_label'][m]['f0'], d['per_record'][m]['f0']) for m in ('svm', 'dnn')}
# --- n20
E = {'titer': ('ginkgo_titer__gaussian', 'as_they_come', 'none', 'base_1', 'remove', '0.2', 'dnn__hybrid', 'lnoned0f0'),
     'smac_cens': ('ginkgo_colloidal__censoring', 'injected', '0.25', 'base_1', 'remove', '0.05', 'dnn__hybrid', 'l0.25d0f0'),
     'smac_js': ('ginkgo_colloidal__censoring', 'injected', '0', 'base_2', 'js_shrink', '0.05', 'dnn__hybrid__general2', 'l0d0f0'),
     'smac_rm': ('ginkgo_colloidal__censoring', 'injected', '0', 'base_1', 'remove', '0.05', 'dnn__hybrid', 'l0d0f0'),
     'hk_g_d0': ('herg_ki__gaussian', 'injected', '1', 'base_1', 'remove', '0.05', 'dnn__hybrid__general2', 'l1d0f0'),
     'hk_g_d1': ('herg_ki__gaussian', 'injected', '1', 'base_1', 'remove', '0.05', 'dnn__hybrid__general2', 'l1d1f0'),
     'hk_g_d2': ('herg_ki__gaussian', 'injected', '1', 'base_1', 'remove', '0.05', 'dnn__hybrid__general2', 'l1d2f0'),
     'hk_0_f0': ('herg_ki__grouped_shifted', 'injected', '0', 'base_2', 'label_smooth', '0.05', 'dnn__hybrid', 'l0d0f0'),
     'hk_0_f1': ('herg_ki__grouped_shifted', 'injected', '0', 'base_2', 'label_smooth', '0.05', 'dnn__hybrid', 'l0d0f1'),
     'hk_0_f2': ('herg_ki__grouped_shifted', 'injected', '0', 'base_2', 'label_smooth', '0.05', 'dnn__hybrid', 'l0d0f2')}
for k, a in E.items():
    src, v = c7._n20_get(*a); V[f'T18 {k}'] = v; V[f'T18 {k} src'] = src
# --- n15
T19 = {}
for ds in ('caco2', 'logd', 'herg_ki', 'ginkgo_colloidal', 'buchwald_hartwig'):
    d = c7.J('view', 'n15_clean_vs_noisy_hybrid', f'{ds}.json')
    ms = set()
    for f, Fd in d['per_fold'].items():
        for k, v in Fd['draws'].items(): ms |= set(v.get('models', {}))
    for m in sorted(ms):
        for f, Fd in sorted(d['per_fold'].items()):
            c = []; n = []
            for k, v in Fd['draws'].items():
                if not k.startswith('gaussian_L1_'): continue
                x = (v.get('models', {}).get(m) or {}).get('by_slot', {}).get('hybrid__general2')
                if x and x.get('clean_arm_score') is not None and x.get('noisy_arm_score') is not None:
                    c.append(x['clean_arm_score']); n.append(x['noisy_arm_score'])
            if c: T19[f'{ds}|{m}|{f}'] = (mean(c), mean(n), len(c))
V['T19'] = T19
# --- n40
T20 = {}
for fn, lv in (('herg_ki__gaussian.json', '1'), ('herg_ki__censoring.json', '0.25'), ('logd__gaussian.json', '0'), ('herg_ki__gaussian.json', '0')):
    d = c7.J('noise', c7.N40, fn)
    for sp in ('base_1', 'base_2', 'hybrid', 'hybrid__general2'):
        for m in d['models']:
            s = c7._n40_scores(d, lv, sp, m)
            if s: T20[f'{fn}|{lv}|{sp}|{m}'] = {k: v for k, v in s.items() if isinstance(v, float)}
V['T20'] = T20
json.dump(V, open(sys.argv[1] + '/vals.json', 'w'), indent=0, default=str)
print('ok', len(V))
