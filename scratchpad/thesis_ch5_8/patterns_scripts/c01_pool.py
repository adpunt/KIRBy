import json,glob,os,sys,statistics as st
sys.path.insert(0,os.path.dirname(__file__)); from common import *
D='/Volumes/seagate/kirby_results_2026-10-09/complexity/c01_rep_characterization'
print('dom|ds|n molecules|reps|median pca_90 (features to 90% variance)|median effective rank|per model: best single, 3rd best, worst (held-out mean, 5 folds)|ESM-2 variants in pool|hand-built reps in pool')
HAND={'onehot','blosum','aac','mutation_features','structural','pdv','mordred','maccs','ecfp4','avalon','atom_pair','top_torsion','graph_kernel'}
rows=[]
for f in sorted(glob.glob(D+'/*.json')):
    ds=os.path.basename(f)[:-5]
    if ds in CUT: continue
    d=json.load(open(f)); m=d['_meta']; low=d['lower_is_better']
    reps=m['representations']
    p90=[d[r]['intrinsic_dim'].get('pca_90') for r in reps if r in d and 'intrinsic_dim' in d[r]]
    er=[d[r].get('effective_rank') for r in reps if r in d and isinstance(d[r].get('effective_rank'),(int,float))]
    per=[]
    for mod in m['models_fitted']:
        sc=sorted([d[r]['baselines'][mod]['score_mean'] for r in reps if r in d and mod in d[r].get('baselines',{}) and d[r]['baselines'][mod].get('score_mean') is not None], reverse=not low)
        if sc: per.append(f"{mod} {sc[0]:.4f}/{sc[min(2,len(sc)-1)]:.4f}/{sc[-1]:.4f}")
    n=sum(m['outer_split']['held_out_rows_per_fold'])
    esm2=sum(1 for r in reps if r.startswith('esm2'))
    hand=sum(1 for r in reps if r in HAND)
    print('|'.join(map(str,[domain(ds),ds,n,len(reps),st.median(p90) if p90 else '-',f"{st.median(er):.0f}" if er else '-','; '.join(per),esm2,hand])))
