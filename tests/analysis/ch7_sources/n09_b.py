import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-apunt-repos-KIRBy/f55db1bf-8fce-4a29-ab8b-783ed5b81253/scratchpad')
from n09_load import *
import pickle
rows=[]
for ds,kind,f in files():
    d=json.load(open(f))
    if d.get('result_status')!='ok': continue
    vp=d['vs_plain_dnn_same_labels']; lib=d['lower_is_better']
    for lv,c in vp.items():
        ps=d['per_sigma'][lv]
        for sp in ['base_1','base_2','hybrid','hybrid__general2','weighted_hybrid']:
            for m,r in c.get(sp+'_lnl',{}).items():
                rows.append(dict(ds=ds,kind=kind,lv=float(lv),sp=sp,m=m,plain=r['plain_dnn_mean'],meth=r['method_mean'],gain=r['gain_mean'],n=r['n_passes'],won=r['passes_won'],passes=r['passes'],metric=d['metric'],lib=lib,folds=d.get('folds_run'),plainall=ps.get(sp+'_no_lnl',{}).get('mean')))
pickle.dump(rows,open('/private/tmp/claude-501/-Users-apunt-repos-KIRBy/f55db1bf-8fce-4a29-ab8b-783ed5b81253/scratchpad/n09_rows.pkl','wb'))
print(len(rows))
import collections
# level0 identical across kinds?
z=collections.defaultdict(set)
for r in rows:
    if r['lv']==0 and r['m']=='small_loss': z[(r['ds'],r['sp'])].add(round(r['plain'],6) if r['plain']==r['plain'] else None)
for k,v in sorted(z.items()):
    if k[1]=='base_1': print(k,v)
print(sorted({(r['ds'],r['kind'],r['lv'],str(r['folds'])) for r in rows}))
