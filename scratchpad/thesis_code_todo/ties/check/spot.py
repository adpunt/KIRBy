"""Independent spot-check of ties_results.json numbers (checker). Reads cached .npy and meta .npz only."""
import numpy as np, json
from collections import Counter
C='/Volumes/seagate/kirby_cache/rep_cache'; T='/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo'
out={}
def chk(ds, rep):
    y=np.load(f'{T}/meta_{ds}.npz',allow_pickle=True)['labels'].astype(float)
    X=np.load(f'{C}/{ds}/{rep}.npy',mmap_mode='r')
    keys=[np.asarray(X[i]).tobytes() for i in range(X.shape[0])]
    cnt=Counter(keys); groups={}
    for i,k in enumerate(keys):
        if cnt[k]>1: groups.setdefault(k,[]).append(i)
    sse=sum(((y[g]-y[g].mean())**2).sum() for g in groups.values())
    rmse=float(np.sqrt(sse/len(y)))
    sp=[float(y[g].max()-y[g].min()) for g in groups.values()]
    r={'n':len(y),'tied':sum(len(g) for g in groups.values()),'n_groups':len(groups),'rmse_min':rmse,
       'best_r2':1-rmse**2/float(np.var(y)),'max_spread':max(sp),'mean_spread':float(np.mean(sp)),
       'gt054':sum(s>0.54 for s in sp),'label_sd':float(np.std(y))}
    if ds=='herg_ki' and rep=='ecfp4':
        r['pair_1149_1150_same']=bool(np.array_equal(X[1149],X[1150])); r['pair_labels']=[y[1149],y[1150]]
        r['bits_1149']=int((np.asarray(X[1149])!=0).sum())
        r['group_of_1149']=[g for g in groups.values() if 1149 in g][0]
    out[f'{ds}/{rep}']=r; print(ds,rep,r,flush=True)
for ds,rep in [('herg_ki','ecfp4'),('herg_ki','maccs'),('logd','maccs'),('logd','top_torsion'),('caco2','graph_kernel')]:
    chk(ds,rep)
json.dump(out,open(f'{T}/ties/check/spot.json','w'),indent=1,default=int)
