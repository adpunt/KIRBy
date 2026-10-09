import json,glob,os,re,statistics as st
BAD=re.compile('tcga|annitia|nhanes|pam50',re.I)
DS=['caco2','herg_ki','logd','herg','buchwald_hartwig','ginkgo_colloidal','flab_thermo_tresanco','ddg_combined','ginkgo_titer']
R='/Volumes/seagate/kirby_results_2026-10-09/noise/n12_importance_weighted_distance/'
units=[]
for f in sorted(glob.glob(R+'*.json')):
    b=os.path.basename(f)
    if BAD.search(b) or 'fingerprint' in b: continue
    ds=b.split('__')[0]
    if ds not in DS: continue
    d=json.load(open(f)); lib=d.get('lower_is_better')
    for sg,p in d.get('per_sigma',{}).items():
        if float(sg)==0: continue
        for imp,q in p.get('per_importance_method',{}).items():
            for bx in ['base_1','base_2']:
                A=q.get(bx+'_unweighted',{}); B=q.get('hybrid_unweighted',{}); W=q.get(bx+'_imp_weighted',{})
                for key,ea in A.items():
                    if '@'+bx+'|' not in key: continue
                    eb=B.get(key)
                    if not eb: continue
                    pa={tuple(x):(y,z) for x,y,z in zip(ea['passes'],ea['denoised_per_pass'],ea['noisy_per_pass'])}
                    pb={tuple(x):(y,z) for x,y,z in zip(eb['passes'],eb['denoised_per_pass'],eb['noisy_per_pass'])}
                    com=sorted(set(pa)&set(pb))
                    if not com: continue
                    noisy=st.mean(pa[c][1] for c in com); da=st.mean(pa[c][0] for c in com); db=st.mean(pb[c][0] for c in com)
                    better=(lambda x: x<noisy) if lib else (lambda x: x>noisy)
                    units.append(dict(ds=ds,file=b,cond=d.get('noise_condition'),level=sg,imp=imp,key=key,passes=com,noisy=noisy,own=da,hyb=db,own_imp=better(da),hyb_imp=better(db),lib=lib,
                        own_pp=[pa[c][0] for c in com],hyb_pp=[pb[c][0] for c in com],noisy_pp=[pa[c][1] for c in com]))
import pandas as pd
u=pd.DataFrame(units); u.to_pickle('n12cmp.pkl')
print(len(u))
u['mc']=u.key.str.split('|').str[0]
print(u.groupby('ds').agg(n=('key','size'),own_improved=('own_imp','sum'),hyb_improved=('hyb_imp','sum')).to_string())
print('ALL',len(u),u.own_imp.sum(),u.hyb_imp.sum())
# per (file, level, model@config) pair: mean over matched keys
g=u.groupby(['ds','file','level','mc']).agg(noisy=('noisy','mean'),own=('own','mean'),hyb=('hyb','mean'),lib=('lib','first')).reset_index()
g['own_imp']=[(o<n) if l else (o>n) for o,n,l in zip(g.own,g.noisy,g.lib)]; g['hyb_imp']=[(o<n) if l else (o>n) for o,n,l in zip(g.hyb,g.noisy,g.lib)]
print(g.groupby('ds').agg(pairs=('mc','size'),own_improved=('own_imp','sum'),hyb_improved=('hyb_imp','sum')).to_string())
print('ALL pairs',len(g),g.own_imp.sum(),g.hyb_imp.sum())
