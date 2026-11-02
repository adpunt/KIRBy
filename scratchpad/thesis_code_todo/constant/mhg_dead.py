import numpy as np, json
R='/Volumes/seagate/kirby_cache/rep_cache'
sets={}
for d in ['herg_ki','caco2','logd','adme_mdr1','adme_hlm','adme_rlm','herg']:
    X=np.load(f'{R}/{d}/mhg_gnn.npy',mmap_mode='r')
    z=np.where(~np.asarray(X).any(axis=0))[0]; sets[d]=z.tolist()
base=set(sets['herg_ki'])
res={'mhg_gnn_zero_feature_count':{d:len(v) for d,v in sets.items()},
     'mhg_gnn_same_zero_indices_all_datasets':all(set(v)==base for v in sets.values())}
print(res)
J=json.load(open('constant/constant_counts.json')); J['mhg_gnn_dead']=res
json.dump(J,open('constant/constant_counts.json','w'),indent=1)
