import numpy as np, json
R='/Volumes/seagate/kirby_cache/rep_cache'; res={}
for ds in ['adme_mdr1','adme_hlm','adme_rlm','herg']:
    X=np.load(f'{R}/{ds}/mordred.npy',mmap_mode='r'); X=np.asarray(X,dtype=np.float64)
    const=(X==X[0]).all(axis=0); res[ds]=dict(constant=int(const.sum()),const_nonzero=int((X[0,const]!=0).sum()))
print(res)
J=json.load(open('constant/check/recheck.json')); J['mordred_constant_nonzero_other']=res; json.dump(J,open('constant/check/recheck.json','w'),indent=1)
