import json, sys, os
D = sys.argv[1]; V = json.load(open(D + '/vals.json')); OUT = D + '/tables'
WARN = []

def f(x, d=3, bold=False):
    if x is None: return ''
    s = f'{x:.{d}f}'
    if s.startswith('-'): s = '$-$' + s[1:]
    return r'\textbf{' + s + '}' if bold else s


def tie(a, b, d=3, label=''):
    if a is None or b is None: return False
    if f'{a:.{d}f}' != f'{b:.{d}f}': return False
    if f'{a:.{d+1}f}' == f'{b:.{d+1}f}': return False
    WARN.append(f'tie {label}: {a} {b}'); return True

def rowcells(vals, ref=None, pairs=(), d=3, bold=()):
    """Format a row; any value tying (at d decimals) with vals[ref], or any listed pair that ties, prints with d+1."""
    dec = [d] * len(vals)
    pp = list(pairs) + ([(ref, i) for i in range(len(vals)) if i != ref] if ref is not None else [])
    for i, j in pp:
        if tie(vals[i], vals[j], d):
            dec[i] = dec[j] = d + 1
    return [f(x, dec[k], k in bold) for k, x in enumerate(vals)]

def pair(a, b, d=3, label='', ba=False, bb=False):
    dd = d + 1 if tie(a, b, d, label) else d
    return f(a, dd, ba), f(b, dd, bb)

SN = {'qm9': 'QM9', 'caco2': 'Caco-2', 'herg_ki': 'HK$_i$', 'logd': 'LogD', 'herg': 'HB', 'herg_ic50': 'HIC$_{50}$',
      'buchwald_hartwig': 'BH', 'flab_thermo_tresanco': 'Nb-Tm', 'ddg_combined': '$\\Delta\\Delta G$', 'ginkgo_titer': 'Titer',
      'ginkgo_pr_cho': 'PR-CHO', 'ginkgo_colloidal': 'SMAC', 'ginkgo_acsins': 'AC-SINS'}
KIND = {'censoring': 'Clipped 25\\%', 'outlier_p05': 'Outliers', 'shift_only_p10': 'Shift in 10\\%',
        'sparse_dprime_up_p10': 'Shift in 10\\%, model error', 'uniform': 'Uniform flips', 'binary_asymmetric': 'One-way flips',
        'class_dependent': 'Class-dependent flips', 'class_imbalance': 'Flips by class frequency', 'instance_noise': 'Instance flips',
        'confusion_directed': 'Confusion-matrix flips'}
CFG = {'base_1': 'First single', 'base_2': 'Second single', 'hybrid': 'First hybrid', 'hybrid__general2': 'Second hybrid'}
MOD = {'dnn': 'Network', 'svm': 'SVR', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'rf': 'Random forest', 'ngboost': 'NGBoost', 'xgboost_aft': 'XGBoost AFT'}
ORDER = list(SN)

def write(name, text):
    open(f'{OUT}/{name}.tex', 'w').write(text.strip() + '\n')

def table(label, short, caption, colspec, head, body, source):
    return (f"% SOURCE: {source}\n% TODO: manual review\n\\begin{{table}}[htbp]\n\\centering\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
            + body + f"\\bottomrule\n\\end{{tabular}}\n\\caption[{short}]{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}")

def blk(text, n):
    return f"\\multicolumn{{{n}}}{{@{{}}l}}{{\\textit{{{text}}}}} \\\\\n"

# ---------------- T1
v = lambda k: V[k]
b = ''
b += ' & '.join(['No cleaning', f(v('T1 none base_1')), f(v('T1 none hybrid')), f(v('T1 f1 none base_1'), 4), f(v('T1 f1 none hybrid'))]) + ' \\\\\n'
b += blk("Act on the 10\\% the forest residual flags", 5)
s, s2 = pair(v('T1 delete base_1'), v('T1 move base_1'), label='T1 delete/move single')
b += f"\\quad Delete & {s} & {f(v('T1 delete hybrid'), bold=True)} & & \\\\\n"
b += f"\\quad Move toward neighbors' mean & {s2} & {f(v('T1 move hybrid'))} & & \\\\\n"
b += f"\\quad Down-weight & {f(v('T1 down base_1'))} & {f(v('T1 down hybrid'))} & & \\\\\n"
b += blk("Delete the top 10\\% by each detector", 5)
for det, nm in (('activity_cliff_aware', 'Cliff-aware gap'), ('gaussian_process', 'GP spread and residual'), ('heteroscedastic_nn', 'Variance-head network'),
                ('ngboost', 'NGBoost'), ('hetero_gp', 'Heteroscedastic GP')):
    bo = det == 'activity_cliff_aware'
    a1, a0 = pair(v(f'T1 f1 {det} base_1'), v('T1 f1 none base_1'), label=f'T1 f1 {det} single vs none')
    b += f"\\quad {nm} & {f(v(f'T1 {det} base_1'))} & {f(v(f'T1 {det} hybrid'), bold=bo)} & {a1} & {f(v(f'T1 f1 {det} hybrid'), bold=bo)} \\\\\n"
assert abs(v('T1 n23none base_1') - v('T1 none base_1')) < 1e-9 and abs(v('T1 n23none hybrid') - v('T1 none hybrid')) < 1e-9
head = (" & \\multicolumn{2}{c}{Fold 0, mean of 3 draws} & \\multicolumn{2}{c}{Fold 1, draw 0} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
        "Action & First single & First hybrid & First single & First hybrid")
write('t01_herg_flip_actions', table('tab:noise:res:herg_flip_actions', 'Actions and detectors under hERG label flips.',
      "Actions and detectors under hERG label flips. hERG blocking with 10\\% of training labels flipped along a confusion matrix, XGBoost. One row is one action on the flagged molecules. Held-out AUROC, higher is better. Fold 1 was run for the detectors only.",
      '@{}lrrrr@{}', head, b,
      "n07_remediation herg__confusion_directed rows (no cleaning and the three actions, fold 0, level 0.1, xgboost; move toward neighbors' mean at its setting run on all three draws); n23_uncertainty_cleaning herg__confusion_directed.json per_regime.injected.0.1.<config>.<detector>.drop@0.1.xgboost (fold 0) and noise_full n23 rows (fold 1, draw 0, best setting per detector); /Volumes/seagate/kirby_results_2026-10-10, read 2026-10-10 [N7-03, N7-05]"))

# ---------------- T2
b = blk('hERG blocking, 10\\% confusion-matrix flips, XGBoost on the first hybrid, AUROC (higher is better)', 6)
b += f"Neighbor-label gap & 10\\% & 0 & {f(v('T2 herg nbr none'))} & {f(v('T2 herg nbr delete'), bold=True)} & {f(v('T2 herg nbr random'))} \\\\\n"
for fo in (0, 1, 2):
    a = v(f'T2 herg cv f{fo}')
    b += f"Forest residual & 10\\% & {fo} & {f(a[0])} & {f(a[1])} & {f(a[2])} \\\\\n"
b += '\\midrule\n' + blk('LogD, 5\\% outliers, network, RMSE (lower is better)', 6)
for sp, rate, fo, nm in (('hybrid__general2', '0.05', 0, 'Second hybrid'), ('hybrid', '0.1', 0, 'First hybrid'), ('hybrid__general2', '0.05', 1, 'Second hybrid')):
    a = v(f'T2 logd {sp} {rate} f{fo}')
    b += f"{nm}, forest residual & {int(float(rate)*100)}\\% & {fo} & {f(a[0])} & {f(a[1], bold=(fo == 0 and sp == 'hybrid__general2'))} & {f(a[2])} \\\\\n"
b += '\\midrule\n' + blk('Buchwald--Hartwig, labels as delivered, LightGBM, RMSE in percentage points of yield (lower is better)', 6)
for sp, nm in (('hybrid', 'First hybrid'), ('base_2', 'Second single')):
    a = v(f'T2 bh {sp}')
    b += f"{nm}, neighbor-label gap & 20\\% & 0 & {f(a[0], 1)} & {f(a[1], 1, bold=(sp == 'hybrid'))} & {f(a[2], 1)} \\\\\n"
head = "\\makecell[l]{Configuration and\\\\detector} & \\makecell[r]{Share\\\\deleted} & Fold & \\makecell[r]{No\\\\cleaning} & \\makecell[r]{Delete\\\\flagged} & \\makecell[r]{Best random\\\\deletion}"
write('t02_delete_vs_random', table('tab:noise:res:delete_vs_random', 'Deleting flagged molecules against random deletion.',
      "Deleting flagged molecules against random deletion. One row is one configuration, detector and outer fold. The random control deletes the same number of molecules at five seeds; the best of the five is shown. Scores are means over the noise draws.",
      '@{}lrrrrr@{}', head, b,
      "n24_delete_or_keep: herg__confusion_directed.json per_level.0.1.hybrid.neighbor_residual@screen1.0.1 (fold 0) and herg rows cross_val|flag_rate=0.1 folds 0-2 (noise + noise_full; fold 2 differs from the pre-cut text, raw now 0.8193/0.8346/0.8279); logd__outlier_p05 rows, dnn, level 1.0; buchwald_hartwig__censoring.json per_level.0.{hybrid,base_2}.neighbor_residual@screen1.0.2.lightgbm; /Volumes/seagate/kirby_results_2026-10-10 [N7-03, N7-04, N7-11]"))

# ---------------- T3
b = ' & '.join(['No cleaning', f(v('T3 none base_1')), f(v('T3 none hybrid'))]) + ' \\\\\n'
b += blk('Act on the 5\\% the forest residual flags', 3)
for k, nm in (('delete', 'Delete'), ('down', 'Down-weight'), ('smooth', 'Label smoothing'), ('shrink', 'Score-scaled shrink')):
    c1 = rowcells([v('T3 none base_1'), v(f'T3 {k} base_1')], ref=0, bold=(1,) if k == 'shrink' else ())[1]
    c2 = rowcells([v('T3 none hybrid'), v(f'T3 {k} hybrid')], ref=0, bold=(1,) if k == 'down' else ())[1]
    b += f"\\quad {nm} & {c1} & {c2} \\\\\n"
b += '\\midrule\n' + blk('Separate run: delete the top 10\\% by each detector', 3)
b += f"\\quad No cleaning & {f(v('T3 n23none base_1'))} & {f(v('T3 n23none hybrid'))} \\\\\n"
for det, nm in (('activity_cliff_aware', 'Cliff-aware gap'), ('gaussian_process', 'GP spread and residual'), ('heteroscedastic_nn', 'Variance-head network'),
                ('ngboost', 'NGBoost'), ('hetero_gp', 'Heteroscedastic GP')):
    b += f"\\quad {nm} & {f(v(f'T3 {det} base_1'))} & {f(v(f'T3 {det} hybrid'), bold=(det == 'gaussian_process'))} \\\\\n"
write('t03_logd_outlier_actions', table('tab:noise:res:logd_outlier_actions', 'Actions and detectors on LogD with outliers.',
      "Actions and detectors on LogD with 5\\% outliers. Network, outer fold 0, mean of 3 draws. One row is one action. Held-out RMSE, lower is better.",
      '@{}lrr@{}', 'Action & First single & First hybrid', b,
      "n07_remediation logd__outlier_p05 rows, dnn, level 1.0, fold 0 (remove flag_share=0.05, downweight, label_smooth k=20 inverse_square hybrid_similarity manhattan, js_shrink); n23_uncertainty_cleaning logd__outlier_p05.json per_regime.injected.1.<config>.<detector>.{none,drop@0.1}.dnn; /Volumes/seagate/kirby_results_2026-10-10 [N7-04, N7-05]"))

# ---------------- T5 weighting
W = V['T5']
b = blk('hERG blocking, 10\\% confusion-matrix flips, XGBoost, AUROC (higher is better)', 4)
x0, x1 = pair(W['herg n13 xgboost base_1 untreated'], W['herg n13 xgboost base_1 weighted'], label='T5 herg single')
b += f"First single & Gap over neighbor spread & {x0} & {x1} \\\\\n"
b += f"First hybrid & Gap over neighbor spread & {f(W['herg n13 xgboost hybrid untreated'])} & {f(W['herg n13 xgboost hybrid weighted'], bold=True)} \\\\\n"
b += '\\midrule\n' + blk('LogD, 5\\% outliers, network, RMSE (lower is better)', 4)
b += f"First single & Neighbor gap, best of four distances & {f(W['logd n21 outlier dnn base_1 untreated'])} & {f(W['logd n21 outlier dnn base_1 weighted'])} \\\\\n"
b += f"First hybrid & Neighbor gap, best of four distances & {f(W['logd n21 outlier dnn hybrid untreated'])} & {f(W['logd n21 outlier dnn hybrid weighted'], bold=True)} \\\\\n"
b += '\\midrule\n' + blk('Buchwald--Hartwig, Gaussian noise at 1.0, LightGBM, RMSE in percentage points of yield (lower is better)', 4)
b += f"First hybrid & Near-identical pairs that disagree & {f(W['buchwald_hartwig n35 gaussian lightgbm hybrid untreated'], 1)} & {f(W['buchwald_hartwig n35 gaussian lightgbm hybrid weighted'], 1, bold=True)} \\\\\n"
b += '\\midrule\n' + blk('QM9, no noise added, RMSE (lower is better)', 4)
b += f"Network, second hybrid & Neighbor weights & {f(W['qm9 L0 n21 dnn hybrid__general2 untreated'])} & {f(W['qm9 L0 n21 dnn hybrid__general2 neighbour weights'], bold=True)} \\\\\n"
b += f"Network, second hybrid & Detector weights & & {f(W['qm9 L0 n13 dnn hybrid__general2 detector weights'])} \\\\\n"
x0, x1 = pair(W['qm9 L0 n21 lightgbm hybrid untreated'], W['qm9 L0 n21 lightgbm hybrid neighbour weights'], label='T5 qm9 lgbm nbr')
b += f"LightGBM, first hybrid & Neighbor weights & {x0} & {x1} \\\\\n"
b += f"LightGBM, first hybrid & Detector weights & & {f(W['qm9 L0 n13 lightgbm hybrid detector weights'])} \\\\\n"
write('t05_weighting', table('tab:noise:res:weighting', 'Weighting flagged molecules.',
      "Weighting flagged molecules. One row is one configuration and weighting, at its best setting. Outer fold 0. The detector-weight run on QM9 stores no unweighted score.",
      '@{}llrr@{}', 'Configuration & Weighting & Unweighted & Weighted', b,
      "n13_noise_distributions herg rows (confusion_directed 0.1, xgboost, local_variance_ratio); n21_neighbour_distance logd outlier_p05 rows (dnn, level 1) and qm9 rows (level 0); n35 buchwald_hartwig gaussian rows (lightgbm, level 1); n13 qm9__censoring.json per_strength 0.0 track_b (minimum over detector fits and estimators); values as re-read by tests/analysis/ch7_numbers.py N7-07 on /Volumes/seagate/kirby_results_2026-10-10 [N7-07]"))

# ---------------- T6 shuffled
def row6(lab, k, bold_clean=False, d=3):
    a = V[f'T6 {k}']
    c = rowcells(a, ref=0, pairs=[(1, 2), (1, 3), (1, 4)], d=d, bold=(1,) if bold_clean else ())
    return f"{lab} & " + ' & '.join(c) + " \\\\\n"
b = blk('hERG $K_i$, network, RMSE (lower is better)', 6)
b += row6('First single, label smoothing, neighbor-label gap', 'hk1', True)
b += row6('First hybrid, label smoothing, forest residual', 'hkh')
b += blk('hERG blocking, XGBoost, AUROC (higher is better)', 6)
b += row6('First single, label smoothing, neighbor-label gap', 'hb1')
b += row6('Second hybrid, label smoothing, neighbor-label gap', 'hbh2')
b += blk('Caco-2, network, RMSE (lower is better)', 6)
b += row6('Second single, down-weight, forest residual', 'cc2')
b += blk('LogD, network, RMSE (lower is better)', 6)
b += row6('Second single, down-weight, neighbor-label gap', 'ld2')
b += '\\midrule\n' + blk('SMAC colloidal stability, Spearman (higher is better)', 6)
b += row6('SVR, first single, delete 20\\%, forest residual', 'sm1', True)
b += row6('SVR, second hybrid, delete 20\\%, forest residual', 'smh2')
b += row6('Network, first single, move toward out-of-fold, neighbor-label gap', 'sm1n')
b += row6('Network, first hybrid, move toward out-of-fold, forest residual', 'smhn')
b += blk('$\\Delta\\Delta G$, network, Spearman (higher is better)', 6)
b += row6('First single, label smoothing, forest residual', 'dd1', True)
b += row6('First hybrid, label smoothing, forest residual', 'ddh')
b += blk('Expression titer, random forest, Spearman (higher is better)', 6)
b += row6('Second single, delete, forest residual', 'ti2')
b += row6('Second hybrid, delete, forest residual', 'tih2')
head = "Configuration, cleanup and detector & \\makecell[r]{No\\\\cleaning} & Cleaned & \\makecell[r]{Shuffled\\\\seed 0} & \\makecell[r]{Shuffled\\\\seed 1} & \\makecell[r]{Shuffled\\\\seed 2}"
write('t06_shuffled', table('tab:noise:res:shuffled', 'Cleaning the labels as delivered against shuffled detector scores.',
      "Cleaning the labels as delivered against the same cleanup driven by shuffled detector scores. One row is one model, configuration, cleanup and detector. Outer fold 0.",
      '@{}P{0.42\\linewidth}rrrrr@{}', head, b,
      "n18_clean_the_real_labels/<ds>.json per_space.<config>.per_detector.<det>.<slot>.per_model.<model>.{no_cleaning_mean, <cleanup>.cleaned_mean} and shuffled_control.<det>.<slot>.<model>.<cleanup>.<seed>.shuffled_mean, fold 0; slots as in tests/analysis/ch7_numbers.py N7-09, N7-30, N7-31; /Volumes/seagate/kirby_results_2026-10-10 [N7-09, N7-30, N7-31]"))

# ---------------- T7
R = V['T7']; b = ''
for fo in (0, 1, 2):
    g = lambda s: R[f'herg_ki L0 dnn {s} fold {fo} {"" if False else ""}'] if False else None
    hn = R[f'herg_ki L0 dnn hybrid fold {fo} no cleaning']; hl = R[f'herg_ki L0 dnn hybrid fold {fo} label_smooth']
    sn = R[f'herg_ki L0 dnn base_2 fold {fo} no cleaning']; sl = R[f'herg_ki L0 dnn base_2 fold {fo} label_smooth']
    a, c = pair(hn, hl, label=f'T7 hyb f{fo}')
    a = f(hn, 4 if tie(hn, hl) else 3); c = f(hl, 4 if tie(hn, hl) else 3, bold=True)
    b += f"{fo} & {a} & {c} & {f(sn)} & {f(sl)} \\\\\n"
head = (" & \\multicolumn{2}{c}{First hybrid} & \\multicolumn{2}{c}{Second single} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
        "Fold & \\makecell[r]{No\\\\cleaning} & \\makecell[r]{Label\\\\smoothing} & \\makecell[r]{No\\\\cleaning} & \\makecell[r]{Label\\\\smoothing}")
write('t07_hergki_smoothing_folds', table('tab:noise:res:hergki_smoothing_folds', 'Label smoothing on hERG $K_i$ by fold.',
      "Label smoothing on hERG $K_i$'s labels as delivered, by outer fold. Network. One row is one outer fold. Held-out RMSE, lower is better. First hybrid: forest residual, flag share 5\\%. Second single: the best detector setting on each fold.",
      '@{}lrrrr@{}', head, b,
      "n24_delete_or_keep/herg_ki__outlier_p05.json per_level['0'][config][det][rate]['keep_rewrite/label_smooth'].dnn (noise: fold 0; noise_full: folds 1, 2); hybrid cross_val@screen3 0.05; second single best label_smooth over detectors per fold (as tests/analysis/ch7_numbers.py N7-10); /Volumes/seagate/kirby_results_2026-10-10 [N7-10]"))

# ---------------- T8 detection n07
rows = sorted(V['T8'], key=lambda r: (ORDER.index(r[0]), list(KIND).index(r[1])))
b = ''; prev = None
for ds, kind, ch, rs_, cb, cm in rows:
    bo_r = (ds, kind) in (('caco2', 'outlier_p05'), ('buchwald_hartwig', 'censoring'))
    bo_m = (ds, kind) == ('caco2', 'shift_only_p10')
    b += f"{SN[ds] if ds != prev else ''} & {KIND[kind]} & {f(ch)} & {f(rs_, bold=bo_r)} & {f(cb)} & {f(cm, bold=bo_m)} \\\\\n"; prev = ds
head = "Dataset & Damage & Chance & \\makecell[r]{Forest\\\\residual} & \\makecell[r]{Cliff-aware,\\\\best setting} & \\makecell[r]{Cliff-aware,\\\\median setting}"
write('t08_detect_residual', table('tab:noise:res:detect_residual', 'Detection by the forest residual and the cliff-aware gap.',
      "Detection by the out-of-fold forest residual and the cliff-aware gap. One row is one dataset and damage kind at its main level (25\\% clipped, 10\\% flipped, otherwise 1.0). Average precision of the damaged molecules, higher is better; chance is the share damaged. Outer fold 0. Dataset short names as in Table~\\ref{tab:dataset_short_names}.",
      '@{}llrrrr@{}', head, b,
      "n07_remediation/<ds>__<kind>.units.jsonl, detection_by_detector.cross_val.detection_average_precision (mean over passes) and sim_detection activity_cliff_aware (mean AP per neighbor setting; best and median over settings), fold 0, main level; read with tests/analysis/ch7_numbers.py n07_detection on /Volumes/seagate/kirby_results_2026-10-10 [N7-13, N7-14]"))

# ---------------- T9 families
rows = sorted(V['T9'], key=lambda r: (ORDER.index(r[0]), list(KIND).index(r[1])))
b = ''; prev = None
for ds, kind, ch, pl, iw, fp, hy in rows:
    bi = (ds, kind) == ('caco2', 'outlier_p05')
    bh = (ds, kind) in (('ddg_combined', 'outlier_p05'), ('ginkgo_colloidal', 'censoring'), ('caco2', 'sparse_dprime_up_p10'))
    b += f"{SN[ds] if ds != prev else ''} & {KIND[kind]} & {f(ch)} & {f(pl)} & {f(iw, bold=bi)} & {f(fp)} & {f(hy, bold=bh)} \\\\\n"; prev = ds
head = "Dataset & Damage & Chance & Plain & \\makecell[r]{Importance-\\\\weighted} & \\makecell[r]{Forest\\\\proximity} & \\makecell[r]{Hybrid's\\\\features}"
write('t09_detect_families', table('tab:noise:res:detect_families', 'Detection by distance family.',
      "Detection by the neighbor detectors in four families of distance. One row is one dataset and damage kind at its main level. Each cell is the best average precision over the detector settings and spaces in that family; higher is better. Chance is the share damaged. Outer fold 0. Dataset short names as in Table~\\ref{tab:dataset_short_names}.",
      '@{}llrrrrr@{}', head, b,
      "noise_view n21_neighbour_distance/<ds>__<kind>.json per_level[main].per_space[space].detection_per_detector[det].average_precision.mean, best per family (spaces[space].family); chance = molecules_corrupted_mean / molecules_in_training_fold_mean; read with tests/analysis/ch7_numbers.py n21_families on /Volumes/seagate/kirby_results_2026-10-09_noise_view [N7-15]"))

# ---------------- T10
keys = sorted(V['T10'], key=lambda k: (ORDER.index(k.split('|')[0]), list(KIND).index(k.split('|')[1])))
b = ''; prev = None
for k in keys:
    ds, kind = k.split('|'); x = V['T10'][k]
    mark = '$^{a}$' if x.get('src') == 'checkpoint' else ''
    bs = k in ('buchwald_hartwig|censoring', 'ginkgo_colloidal|censoring'); bc = k == 'buchwald_hartwig|shift_only_p10'
    b += (f"{SN[ds] if ds != prev else ''} & {KIND[kind]}{mark} & {f(x.get('gp'))} & {f(x.get('cliff'))} & {f(x.get('resid'))} & "
          f"{f(x.get('shap'), bold=bs)} & {f(x.get('cleanlab'), bold=bc)} \\\\\n"); prev = ds
head = (" & & \\multicolumn{2}{c}{First run} & \\multicolumn{3}{c}{Second run} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(l){5-7}\n"
        "Dataset & Damage & \\makecell[r]{GP spread\\\\and residual} & \\makecell[r]{Cliff-aware\\\\gap} & \\makecell[r]{Forest\\\\residual} & \\makecell[r]{Nearest-neighbor\\\\Shapley} & cleanlab")
write('t10_detect_other', table('tab:noise:res:detect_other', 'Model-uncertainty detectors and detectors from outside cheminformatics.',
      "Model-uncertainty detectors and detectors from outside cheminformatics. The first two columns and the last three come from two separate runs. One row is one dataset and damage kind at its main level. Average precision of the damaged molecules, mean over configurations and settings; higher is better. Chance is in Table~\\ref{tab:noise:res:detect_residual}. Outer fold 0. $^{a}$Second run unfinished. Dataset short names as in Table~\\ref{tab:dataset_short_names}.",
      '@{}llrrrrr@{}', head, b,
      "first run: n23_uncertainty_cleaning/<ds>__<kind>.json detection[main level][space][model][source].detection_average_precision, mean over configurations; second run: n32_iterative_and_dual_selection/<ds>__<kind>.json detection (final file) or _checkpoints state.det_acc, mean AP per detector family over configurations and settings; read with tests/analysis/ch7_numbers.py n23_sources, n32_detection on /Volumes/seagate/kirby_results_2026-10-10 [N7-17]"))

# ---------------- T11 break-even
CN = {'outlier_only_p05': 'Wide errors in 5\\%', 'shift_only_p10': 'Shift in 10\\%', 'sparse_dprime_up_p10': 'Shift in 10\\%, model error'}
b = ''; prev = None
for ds, cond, sh, pr, ops in V['T11']:
    bp = (ds, cond, sh) in (('caco2', 'sparse_dprime_up_p10', 'share0.05'), ('herg_ki', 'shift_only_p10', 'share0.1'))
    b += (f"{SN[ds] if ds != prev else ''} & {CN[cond]} & {int(round(float(sh[5:])*100))}\\% & {f(pr, bold=bp)} & {f(ops.get('remove'), bold=bp)} & "
          f"{f(ops.get('downweight'))} & {f(ops.get('rewrite'))} & {f(ops.get('bayes_shrink'))} \\\\\n"); prev = ds
head = (" & & & & \\multicolumn{4}{c}{Break-even precision} \\\\\n\\cmidrule(l){5-8}\n"
        "Dataset & Damage & \\makecell[r]{Share\\\\flagged} & Precision & Delete & \\makecell[r]{Down-\\\\weight} & Relabel & Shrink")
write('t11_breakeven', table('tab:noise:res:breakeven', 'Detector precision against break-even precision.',
      "Precision of the forest residual's flags against the break-even precision of each action. One row is one dataset, damage kind at level 1.0 and flagged share. Precision is the share of flagged molecules that were damaged, mean over configurations and models; break-even is the median over configurations and models. An action helps when the precision is above its break-even. Outer fold 0. Dataset short names as in Table~\\ref{tab:dataset_short_names}.",
      '@{}llrrrrrr@{}', head, b,
      "n31_regime_and_breakeven/<ds>.json summary.per_condition[cond]['1'][config].per_model[m].synthetic[share<s>][op].{break_even_precision, detector_precision_at_this_share_mean}, fold 0; read with tests/analysis/ch7_numbers.py n31_cells on /Volumes/seagate/kirby_results_2026-10-10 [N7-19]"))

# ---------------- T12 censoring folds
b = ''
for ds in ('logd', 'caco2'):
    Fd = V[f'n30full {ds}']
    folds = sorted({int(k.split('|')[0]) for k in Fd})
    for i, fo in enumerate(folds):
        b += f"{SN[ds] if i == 0 else ''} & {fo} & {f(Fd[f'{fo}|clean'])} & {f(Fd[f'{fo}|clipped'])} & {f(Fd[f'{fo}|censored_em'])} \\\\\n"
head = "Dataset & Fold & \\makecell[r]{Before\\\\clipping} & Clipped & \\makecell[r]{Censored\\\\EM}"
write('t12_censoring_folds', table('tab:noise:res:censoring_folds', 'Censored EM on further outer folds.',
      "Censored EM on further outer folds, with 25\\% of training labels clipped. Network on the first hybrid. One row is one outer fold. Held-out RMSE, lower is better. Dataset short names as in Table~\\ref{tab:dataset_short_names}.",
      '@{}lrrrr@{}', head, b,
      "noise_full/n30_censored_likelihood/{logd,caco2}__censoring__fold_aligned_censored_em_2026-10-06.rows.jsonl, level 0.25, dnn, hybrid: score_clean / score_D / score_Dprime (censored_em); LogD fold 4 is newer than the pre-cut text; Caco-2 fold 0 is a separate run (Figure fig:noise:censoring_ladder); /Volumes/seagate/kirby_results_2026-10-10 [N7-21]"))

# ---------------- T13 censoring moves
def t13(dss, name, label, short, cap, unitnote):
    b = ''
    for j, ds in enumerate(dss):
        C = V[f'n30 {ds}']
        models = [m for m in ('dnn', 'svm', 'lightgbm', 'rf') if any(k.split('|')[1] == m for k in C)]
        if j: b += '\\midrule\n'
        b += blk(unitnote[ds], 7)
        for m in models:
            for cfg in ('base_1', 'base_2', 'hybrid', 'hybrid__general2'):
                g = lambda a: C.get(f'{cfg}|{m}|{a}')
                d = 1 if ds == 'buchwald_hartwig' else 3
                c = rowcells([g('clean'), g('clipped'), g('constant_lift'), g('truncated_normal_mean'), g('censored_em')], ref=1, pairs=[(2, 3), (2, 4), (3, 4)], d=d)
                b += f"{MOD[m]} & {CFG[cfg]} & " + ' & '.join(c) + " \\\\\n"
    head = "Model & Configuration & \\makecell[r]{Before\\\\clipping} & Clipped & \\makecell[r]{Constant\\\\lift} & \\makecell[r]{Truncated-\\\\normal mean} & \\makecell[r]{Censored\\\\EM}"
    write(name, table(label, short, cap, '@{}llrrrrr@{}', head, b,
          "n30_censored_likelihood/<ds>.json per_level['0.25'][config][arm][model]['l0.25d0f0'] (noise/, rows files where the json lacks a cell), fold 0; read with tests/analysis/ch7_numbers.py n30_cells on /Volumes/seagate/kirby_results_2026-10-10 [N7-22]"))
t13(('logd', 'qm9', 'herg_ki', 'caco2'), 't13_censoring_moves', 'tab:noise:res:censoring_moves', 'Three ways to raise clipped labels: small molecules.',
    "Three ways to raise clipped training labels, small molecules, with 25\\% of training labels clipped. One row is one model and configuration. Held-out RMSE in the label's units, lower is better. Outer fold 0.",
    {'logd': 'LogD', 'qm9': 'QM9', 'herg_ki': 'hERG $K_i$', 'caco2': 'Caco-2'})
t13(('buchwald_hartwig', 'flab_thermo_tresanco'), 't13b_censoring_moves_other', 'tab:noise:res:censoring_moves_other', 'Three ways to raise clipped labels: reactions and nanobodies.',
    "Three ways to raise clipped training labels, Buchwald--Hartwig and nanobody thermostability, with 25\\% of training labels clipped. One row is one model and configuration. Outer fold 0.",
    {'buchwald_hartwig': 'Buchwald--Hartwig, RMSE in percentage points of yield (lower is better)', 'flab_thermo_tresanco': 'Nanobody thermostability, Spearman (higher is better)'})

# ---------------- T14 lift
b = ''
for j, ds in enumerate(('logd', 'herg_ki', 'caco2', 'qm9')):
    C = V[f'n30 {ds}']
    for i, cfg in enumerate(('base_1', 'base_2', 'hybrid', 'hybrid__general2')):
        g = lambda a: C.get(f'{cfg}|dnn|{a}')
        bo = (ds, cfg) == ('logd', 'hybrid')
        c = rowcells([g('per_molecule_lift')] + [g(f'per_molecule_lift_shuffled/seed{z}') for z in range(3)], ref=0, bold=(0, 3) if bo else ())
        b += f"{SN[ds] if i == 0 else ''} & {CFG[cfg]} & " + ' & '.join(c) + " \\\\\n"
head = "Dataset & Configuration & \\makecell[r]{Own\\\\lift} & \\makecell[r]{Shuffled\\\\seed 0} & \\makecell[r]{Shuffled\\\\seed 1} & \\makecell[r]{Shuffled\\\\seed 2}"
write('t14_censoring_lift', table('tab:noise:res:censoring_lift', 'Per-molecule lift against shuffled lifts.',
      "Each clipped label raised by its own estimated lift, against the same lifts dealt to the clipped molecules at random. Network, 25\\% of training labels clipped. One row is one dataset and configuration. Held-out RMSE, lower is better. Outer fold 0. Dataset short names as in Table~\\ref{tab:dataset_short_names}.",
      '@{}llrrrr@{}', head, b,
      "n30_censored_likelihood/<ds>.json per_level['0.25'][config][per_molecule_lift | per_molecule_lift_shuffled/seed0-2].dnn, fold 0; /Volumes/seagate/kirby_results_2026-10-10 [N7-24]"))

# ---------------- T15 offsets
by = {}
for ds, dr, meth, D_, P_, C_ in V['T15']:
    by.setdefault((ds, dr), {})[meth] = P_; by[(ds, dr)]['_D'] = D_; by[(ds, dr)]['_C'] = C_
b = ''
for ds in ('caco2', 'logd', 'herg_ki'):
    for i, dr in enumerate(sorted(k[1] for k in by if k[0] == ds)):
        x = by[(ds, dr)]
        bo = (ds, dr) == ('caco2', 0)
        c = rowcells([x['_D'], x.get('group_center'), x.get('random_intercept'), x['_C']], ref=0, pairs=[(1, 2)], bold=(2,) if bo else ())
        b += f"{SN[ds] if i == 0 else ''} & {dr} & " + ' & '.join(c) + " \\\\\n"
head = "Dataset & Draw & Damaged & \\makecell[r]{Family\\\\mean} & \\makecell[r]{Random\\\\intercept} & \\makecell[r]{Clean\\\\labels}"
write('t15_family_offsets', table('tab:noise:res:family_offsets', 'Removing a shared offset per scaffold family.',
      "Removing a shared offset per scaffold family under grouped-shifted noise at level 1.0. Network on the first hybrid. One row is one dataset and noise draw. Held-out RMSE, lower is better. Outer fold 0. Dataset short names as in Table~\\ref{tab:dataset_short_names}.",
      '@{}lrrrrr@{}', head, b,
      "n34_group_random_effects/<ds>__grouped_shifted rows, fold 0, level 1.0, dnn, hybrid: score_D (damaged), score_Dprime (group_center, random_intercept), score_clean; LogD draw 2 is newer than the pre-cut text; /Volumes/seagate/kirby_results_2026-10-10 [N7-26]"))

# ---------------- T17 hERG IC50
R = {}
for m, meth, dr, D_, P_ in V['T17']:
    R.setdefault((m, meth), {})[dr] = P_; R.setdefault((m, 'none'), {})[dr] = D_
b = ''
spec = [('svm', [('none', 'No weighting'), ('count_weighted', 'Weight by measurement count'), ('spread_pooled_weighted', 'Weight by inverse error variance')]),
        ('dnn', [('none', 'No weighting'), ('count_weighted', 'Weight by measurement count'), ('count_feature', 'Log count as a feature'), ('shrunk', 'Shrink toward a prediction by spread')]),
        ('ngboost', [('none', 'No extra feature'), ('ngboost_count_feature', 'Log count as a feature')]),
        ('xgboost_aft', [('none', 'Point model'), ('xgboost_interval_aft', 'Interval model')])]
for j, (m, items) in enumerate(spec):
    if j: b += '\\midrule\n'
    b += blk({'svm': 'SVR', 'dnn': 'Network', 'ngboost': 'NGBoost', 'xgboost_aft': 'XGBoost, accelerated failure time'}[m], 4)
    dec = {(meth, dr): 3 for meth, _ in items for dr in range(3)}
    for meth, _ in items:
        for dr in range(3):
            if meth != 'none' and tie(R[(m, 'none')][dr], R[(m, meth)][dr]):
                dec[(meth, dr)] = dec[('none', dr)] = 4
    for meth, nm in items:
        x = R[(m, meth)]; bo = (m, meth) == ('svm', 'count_weighted')
        b += f"\\quad {nm} & " + ' & '.join(f(x[dr], dec[(meth, dr)], bo) for dr in range(3)) + " \\\\\n"
b += '\\midrule\n' + blk('Separate run, one fit: mean label against one row per measurement', 4)
for m in ('svm', 'dnn'):
    a, c = V['T17 n29'][m]
    b += f"\\quad {MOD[m]}, mean label & \\multicolumn{{3}}{{c}}{{{f(a)}}} \\\\\n\\quad {MOD[m]}, one row per measurement & \\multicolumn{{3}}{{c}}{{{f(c)}}} \\\\\n"
head = "Model and method & Draw 0 & Draw 1 & Draw 2"
write('t17_herg_ic50', table('tab:noise:res:herg_ic50', 'Repeat measurements on hERG IC$_{50}$.',
      "Using the repeat measurements on hERG IC$_{50}$. First hybrid. One row is one model and method. Held-out RMSE on molecules with three or more measurements, lower is better. Outer fold 0.",
      '@{}lrrr@{}', head, b,
      "n19_measurement_count/herg_ic50 rows, method_setting label_build=averaged, configuration hybrid, fold 0, draws 0-2 (score_D = no weighting, score_Dprime = method; spread_pooled_weighted = inverse error variance); n29_document_bias/herg_ic50.json scores.all.hybrid.{mean_label, per_record}.{svm,dnn}.f0; /Volumes/seagate/kirby_results_2026-10-10 [N7-34]"))

# ---------------- T18 n20
def r18(lab, k, bold=None, d=3):
    a = V[f'T18 {k}']
    return f"{lab} & " + ' & '.join(rowcells(a, ref=0, pairs=[(1, 2), (1, 3), (2, 3)], d=d, bold=(bold,) if bold is not None else ())) + ' \\\\\n'
b = blk('Spearman (higher is better)', 5)
b += r18('Titer, as delivered, first hybrid, delete 20\\%', 'titer', bold=2)
b += r18('SMAC, 25\\% clipped, first hybrid, delete 5\\%', 'smac_cens', bold=2)
b += r18('SMAC, as delivered, second hybrid, shrink toward out-of-fold', 'smac_js', bold=3)
b += r18('SMAC, as delivered, first hybrid, delete 5\\%', 'smac_rm')
b += '\\midrule\n' + blk('hERG $K_i$, RMSE (lower is better)', 5)
for dr in (0, 1, 2): b += r18(f'Gaussian 1.0, second hybrid, delete 5\\%, draw {dr}', f'hk_g_d{dr}')
for fo in (0, 1, 2): b += r18(f'As delivered, first hybrid, label smoothing, fold {fo}', f'hk_0_f{fo}')
head = "\\makecell[l]{Dataset, labels, configuration\\\\and cleanup} & \\makecell[r]{No\\\\cleaning} & \\makecell[r]{Fit\\\\only} & \\makecell[r]{Search\\\\and fit} & \\makecell[r]{Search\\\\only}"
write('t18_clean_before_search', table('tab:noise:res:clean_before_search', 'Cleaning before the feature search.',
      "Cleaning the fit, the hybrid search, or both. Network; the flagged molecules are those the forest residual flags in a single representation. One row is one dataset, labels, configuration and cleanup. Outer fold 0 unless a fold is named.",
      '@{}P{0.46\\linewidth}rrrr@{}', head, b,
      "n20_clean_before_search/<ds>__<kind>.json (noise final, else _partial; hERG Ki folds 1-2 from noise_full _partial) per_regime[regime][L][cleaned_in][cleanup][flag_rate][dnn__slot][arm]['l{L}d{d}f{f}'], arms search_asis__fit_asis / search_asis__fit_cleaned / search_cleaned__fit_cleaned / search_cleaned__fit_asis; hERG Ki draw 2 and folds 0, 2 are newer than the pre-cut text; /Volumes/seagate/kirby_results_2026-10-10 [N7-35, N7-36]"))

# ---------------- T19 n15
T = V['T19']; b = ''
spec = [('caco2', 'Caco-2, RMSE (lower is better)', ('dnn', 'svm')), ('logd', 'LogD, RMSE (lower is better)', ('dnn', 'svm')),
        ('herg_ki', 'hERG $K_i$, RMSE (lower is better)', ('dnn', 'svm')), ('ginkgo_colloidal', 'SMAC colloidal stability, Spearman (higher is better)', ('dnn', 'svm')),
        ('buchwald_hartwig', 'Buchwald--Hartwig, RMSE in percentage points of yield (lower is better)', ('lightgbm', 'rf'))]
for j, (ds, nm, ms) in enumerate(spec):
    if j: b += '\\midrule\n'
    b += blk(nm, 7)
    d = 1 if ds == 'buchwald_hartwig' else 3
    for m in ms:
        for arm, an in ((0, 'clean labels'), (1, 'damaged labels')):
            cells = []
            for fo in range(5):
                x = T.get(f'{ds}|{m}|{fo}')
                bo = ds == 'herg_ki' and m == 'dnn' and fo == 1
                dd = d + 1 if x and tie(x[0], x[1], d) else d
                cells.append(f(x[arm], dd, bo) if x else '')
            b += f"{MOD[m]} & {an} & " + ' & '.join(cells) + ' \\\\\n'
head = "Model & Searched on & Fold 0 & Fold 1 & Fold 2 & Fold 3 & Fold 4"
write('t19_search_clean_vs_damaged', table('tab:noise:res:search_clean_vs_damaged', 'Searching on clean against damaged labels.',
      "The second hybrid searched on clean labels or on labels with Gaussian noise at level 1.0, both fitted on the damaged labels and scored on the held-out labels as delivered. One row is one model and search labels. Mean over three noise draws; empty cells were not run.",
      '@{}llrrrrr@{}', head, b,
      "noise_view n15_clean_vs_noisy_hybrid/<ds>.json per_fold[f].draws['gaussian_L1_d<d>'].models[m].by_slot.hybrid__general2.{clean_arm_score, noisy_arm_score}, mean over draws (SMAC SVR fold 3: 2 draws); /Volumes/seagate/kirby_results_2026-10-09_noise_view [N7-37]"))

# ---------------- T20 n40
T = V['T20']; b = ''
spec = [('herg_ki__gaussian.json|1', 'hERG $K_i$, Gaussian noise at 1.0', True), ('herg_ki__censoring.json|0.25', 'hERG $K_i$, 25\\% clipped', True),
        ('herg_ki__gaussian.json|0', 'hERG $K_i$, labels as delivered', False), ('logd__gaussian.json|0', 'LogD, labels as delivered', False)]
for j, (key, nm, dmg) in enumerate(spec):
    if j: b += '\\midrule\n'
    b += blk(nm, 8)
    for m in ('dnn', 'svm'):
        for cfg in ('base_1', 'base_2', 'hybrid', 'hybrid__general2'):
            x = T.get(f'{key}|{cfg}|{m}')
            if not x: continue
            bsel = key == 'herg_ki__gaussian.json|1' and m == 'dnn' and cfg == 'hybrid__general2'
            vals = [x.get('fixed'), x.get('seed_avg'), x.get('stack'), x.get('retuned')] + ([x.get('tuned_clean_reused'), x.get('_clean_mean')] if dmg else [None, None])
            cells = rowcells(vals, ref=0, pairs=[(3, 4)], bold=(1, 3, 4) if bsel else ())
            b += f"{MOD[m]} & {CFG[cfg]} & " + ' & '.join(cells) + ' \\\\\n'
head = ("Model & Configuration & Plain & \\makecell[r]{Seed\\\\averaging} & Stacking & \\makecell[r]{Retuned on\\\\these labels} & "
        "\\makecell[r]{Settings tuned\\\\on clean labels} & \\makecell[r]{Plain, clean\\\\labels}")
write('t20_change_model', table('tab:noise:res:change_model', 'Changing the model instead of the labels.',
      "Changing the model instead of the labels. One row is one model and configuration. Held-out RMSE, lower is better. Stacking combines both models and has one score per configuration. Outer fold 0; damaged labels are means over three noise draws. On labels as delivered the last two columns would repeat retuning and the plain model.",
      '@{}llrrrrrr@{}', head, b,
      "n40_model_side_regularisation/{herg_ki,logd}__{gaussian,censoring}.json summary.per_level[L][config].per_model[m]: plain = fixed_mean, method = fixed_mean - per_arm[arm].vs_fixed.mean_gain, stack from [config].stack, clean = per_arm.retuned.clean_mean (as tests/analysis/ch7_numbers.py _n40_scores); /Volumes/seagate/kirby_results_2026-10-10 [N7-38, N7-39, N7-40]"))
print('\n'.join(WARN))
