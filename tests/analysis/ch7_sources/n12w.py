import json,glob,os,re,statistics as st
BAD=re.compile('tcga|annitia|nhanes|pam50',re.I)
DS=['qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','ginkgo_colloidal','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_acsins']
R='/Volumes/seagate/kirby_results_2026-10-09/noise/n12_importance_weighted_distance/'
rows=[]
for f in sorted(glob.glob(R+'*.json')):
    b=os.path.basename(f)
    if BAD.search(b) or 'fingerprint' in b: continue
    ds=b.split('__')[0]
    if ds not in DS: continue
    d=json.load(open(f)); lib=d.get('lower_is_better')
    for sg,p in d.get('per_sigma',{}).items():
        for imp,q in p.get('per_importance_method',{}).items():
            for uw,w in [('base_1_unweighted','base_1_imp_weighted'),('base_2_unweighted','base_2_imp_weighted'),('hybrid_unweighted','hybrid_imp_base'),('hybrid__general2_unweighted','hybrid__general2_imp_base')]:
                for key,ea in q.get(uw,{}).items():
                    eb=q.get(w,{}).get(key)
                    if not eb: continue
                    pa={tuple(x):(y,z) for x,y,z in zip(ea['passes'],ea['denoised_per_pass'],ea['noisy_per_pass'])}
                    pb={tuple(x):y for x,y in zip(eb['passes'],eb['denoised_per_pass'])}
                    com=sorted(set(pa)&set(pb))
                    if not com: continue
                    rows.append(dict(ds=ds,file=b,sigma=sg,imp=imp,space=uw,key=key,passes=com,noisy=[pa[c][1] for c in com],unw=[pa[c][0] for c in com],wt=[pb[c] for c in com],lib=lib))
import pandas as pd
u=pd.DataFrame(rows)
for c in ['noisy','unw','wt']: u[c+'_m']=u[c].apply(st.mean)
u['wt_better']=[(w<x) if l else (w>x) for w,x,l in zip(u.wt_m,u.unw_m,u.lib)]
u.to_pickle('n12w.pkl')
u['model']=u.key.str.split('@').str[0]
print(u.groupby(['ds','model']).agg(n=('key','size'),wt_better=('wt_better','sum')).to_string())
