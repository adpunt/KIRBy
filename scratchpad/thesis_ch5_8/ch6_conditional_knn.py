import json,statistics as st,collections
R='/Volumes/seagate/kirby_results_2026-10-09/similarity/'
DOM={'qm9':'small','caco2':'small','herg_ki':'small','logd':'small','herg':'small','buchwald_hartwig':'reaction',
 'flab_thermo_tresanco':'ab','abagym_EGFR':'ab','ddg_combined':'ab','ginkgo_titer':'ab','ginkgo_pr_cho':'ab','ginkgo_colloidal':'ab','ginkgo_acsins':'ab'}
rows=[]  # per dataset x fold, fixed setting
for ds,dom in DOM.items():
    d=json.load(open(R+f's05_knn_prediction/{ds}.json'))
    fr=d['summary']['fixed_ref']; pf=d['per_fold']
    for i,(fk,f) in enumerate(sorted(pf.items(),key=lambda x:int(x[0]))):
        h=f.get('distance_hybrid') or {}
        rows.append(dict(ds=ds,dom=dom,fold=i,n_train=f['n_train'],gain=fr['per_fold_delta'][i],budget=h.get('budget_features'),alloc=h.get('allocation'),imp=h.get('importance'),ev=h.get('evaluator'),reps=h.get('rep_count'),knn_base=(f.get('knn_base') or [None])[0]))
print('=== per dataset x fold, fixed setting; gain>0 = hybrid better, in dataset metric')
for r in rows: print(r)
json.dump(rows,open('/private/tmp/claude-501/-Users-apunt-repos-KIRBy/76de6a63-5aa3-4191-8dbb-519a12b31b4e/scratchpad/cond/s05_rows.json','w'))
# per-setting: hybrid arm minus single_knn1 at the same condition/distance/k/scheme
print('\n=== per setting, mean over folds, hybrid minus single (sign: + = hybrid better)')
for ds,dom in DOM.items():
    d=json.load(open(R+f's05_knn_prediction/{ds}.json')); lib=d['lower_is_better']; c=d['cells']
    out=collections.defaultdict(list)
    for key,v in c.items():
        arm,cond,dist,k,sch=key.split('|')
        if arm not in('distance_hybrid','similarity_hybrid'): continue
        sk='|'.join(['single_knn1',cond,dist,k,sch])
        if sk not in c or v.get('score_mean') is None or c[sk].get('score_mean') is None: continue
        g=(c[sk]['score_mean']-v['score_mean']) if lib else (v['score_mean']-c[sk]['score_mean'])
        out[('arm',arm)].append(g); out[('cond',cond)].append(g); out[('dist',dist)].append(g); out[('k',k)].append(g); out[('scheme',sch)].append(g)
    print(ds,dom,d['metric'],{f'{a}={b}':round(st.mean(v),4) for (a,b),v in sorted(out.items())})
