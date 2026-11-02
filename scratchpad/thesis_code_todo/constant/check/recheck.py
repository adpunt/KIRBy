import numpy as np, json
R='/Volumes/seagate/kirby_cache/rep_cache'
res={}
def cnt(X):
    X=np.asarray(X,dtype=np.float64); n=len(X)
    c=nc=0; vals=[]
    for j in range(X.shape[1]):
        v,k=np.unique(X[:,j],return_counts=True)
        if len(v)==1: c+=1; vals.append(float(v[0]))
        if k.max()>=0.95*n: nc+=1
    return dict(n=n,constant=c,near=nc,const_values=sorted(set(vals)),const_nonzero=sum(1 for x in vals if x!=0))
for ds,rep in [('herg_ki','ecfp4'),('herg_ki','mordred'),('qm9','mordred'),('logd','mordred'),('caco2','ecfp4'),('herg_ki','maccs'),('adme_rlm','atom_pair')]:
    X=np.load(f'{R}/{ds}/{rep}.npy',mmap_mode='r')
    res[f'{ds}.{rep}']=cnt(X); print(ds,rep,res[f'{ds}.{rep}'],flush=True)
json.dump(res,open('constant/check/recheck.json','w'),indent=1)
