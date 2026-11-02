"""Independent spot-check of distance numbers (checker). Writes recheck.json."""
import json, numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
C='/Volumes/seagate/kirby_cache/rep_cache'; M='/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo'
out={}
# 1. hERG Ki ECFP4: all-pairs spearman, 5-NN outside own scaffold
m=np.load(f'{M}/meta_herg_ki.npz',allow_pickle=True); y=m['labels'].astype(float); g=np.asarray(m['groups'])
X=np.load(f'{C}/herg_ki/ecfp4.npy').astype(bool)
d=pdist(X,'jaccard')  # = 1 - Tanimoto on bits
dy=pdist(y[:,None],'cityblock')
out['herg_ecfp4_allpairs']=float(spearmanr(d,dy).correlation)
D=squareform(d); D[g[:,None]==g[None,:]]=np.inf
nb=np.argsort(D,axis=1,kind='stable')[:,:5]; p=y[nb].mean(1)
out['herg_ecfp4_knn5_spearman']=float(spearmanr(p,y).correlation)
out['herg_ecfp4_knn5_rmse']=float(np.sqrt(np.mean((p-y)**2)))
out['herg_label_sd']=float(y.std())
T=1-squareform(d); iu=np.triu_indices(len(y),1); sim=T[iu]>=0.55
I,J=iu[0][sim],iu[1][sim]
out['herg_n_similar_pairs']=int(sim.sum())
gap=1.76; mk=np.abs(y[I]-y[J])>gap
out['herg_cliff_mols_gap1.76']=int(np.unique(np.r_[I[mk],J[mk]]).size); out['herg_cliff_pairs_gap1.76']=int(mk.sum())
out['herg_share_T_below_0.2']=float(np.mean(T[iu]<0.2)); out['herg_n_T_ge_0.8']=int((T[iu]>=0.8).sum())
del D,T,d
# 2. QM9 ECFP4 similar pairs and cliff molecules at gap 0.0544
m=np.load(f'{M}/meta_qm9.npz',allow_pickle=True); y=m['labels'].astype(float)
X=np.load(f'{C}/qm9/ecfp4.npy').astype(np.float32); cnt=X.sum(1); n=len(y)
I=[];J=[]
for s in range(0,n,1000):
    inter=X[s:s+1000]@X.T; T=inter/(cnt[s:s+1000,None]+cnt[None,:]-inter)
    ii,jj=np.nonzero(T>=0.55); ii=ii+s; k=jj>ii; I.append(ii[k]); J.append(jj[k])
I=np.concatenate(I); J=np.concatenate(J)
out['qm9_n_similar_pairs']=int(len(I))
mad=float(np.median(np.abs(y-np.median(y)))); out['qm9_mad']=mad
for gap in [0.05442208189278119, 2*mad]:
    mk=np.abs(y[I]-y[J])>gap; out[f'qm9_cliff_mols_gap{gap:.4f}']=int(np.unique(np.r_[I[mk],J[mk]]).size)
# 3. Caco-2 PDV all-pairs on 2000 sample seed 0, Manhattan on standardised
m=np.load(f'{M}/meta_caco2.npz',allow_pickle=True); y=m['labels'].astype(float); n=len(y)
S=np.sort(np.random.RandomState(0).choice(n,2000,replace=False))
A=np.nan_to_num(np.load(f'{C}/caco2/pdv.npy').astype(float)); sd=A.std(0); k=sd>0
Z=(A[:,k]-A[:,k].mean(0))/sd[k]
out['caco2_pdv_allpairs']=float(spearmanr(pdist(Z[S],'cityblock'),pdist(y[S][:,None],'cityblock')).correlation)
out['caco2_pdv_n_features_used']=int(k.sum())
json.dump(out,open('/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo/distance/check/recheck.json','w'),indent=1)
print(out)
