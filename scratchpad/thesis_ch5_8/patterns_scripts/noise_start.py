import json,collections,statistics as st
R=json.load(open('/private/tmp/claude-501/noise_rows.json'))
N={'qm9':9978,'herg_ki':1412,'caco2':2158,'logd':5028,'herg':7772,'herg_ic50':None,'abcov_kd':141,'cdr_flexibility':328,'ddg_combined':429,'flab_thermo_tresanco':672,'ginkgo_acsins':242,'ginkgo_colloidal':242,'ginkgo_pr_cho':197,'ginkgo_titer':239,'humanness_therapeutic':564,'buchwald_hartwig':3955,'tcga_brca_pam50':821}
g=collections.defaultdict(list)
for r in R:
    if r['noise'].startswith('none added') and r['s0'] is not None and r['dom']!='QM9 CONTROL': g[(r['dom'],r['ds'])].append(r)
print('domain|dataset|molecules (c01)|metric|rows as delivered|median untreated score, single|median untreated, hybrid|rows where the method helped both|helped hybrid')
for k in sorted(g):
    rs=g[k]
    print(f"{k[0]}|{k[1]}|{N.get(k[1])}|{rs[0]['metric'].split()[0]}|{len(rs)}|{st.median(r['s0'] for r in rs):.3f}|{st.median(r['h0'] for r in rs if r['h0'] is not None):.3f}|{sum(1 for r in rs if r['s_ok'] and r['h_ok'])}|{sum(1 for r in rs if r['h_ok'])}")
