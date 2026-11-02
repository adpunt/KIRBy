import json,collections,sys,os
sys.path.insert(0,os.path.dirname(__file__)); from common import *
R=json.load(open('/private/tmp/claude-501/c22_rows.json'))
def fam(rep):
    if rep in ('onehot','blosum','aac','mutation_features','structural'): return 'hand-built sequence'
    if rep in ('pdv','mordred'): return 'computed descriptors'
    if rep in ('ecfp4','avalon','atom_pair','top_torsion','maccs','graph_kernel'): return 'fingerprints/keys/graph kernel'
    if rep in ('grover','mhg_gnn','graphmvp'): return 'pretrained graph'
    if rep in ('molformer','smited','chembert','chemberta','selformer','mol2vec'): return 'pretrained SMILES/SELFIES'
    if rep.startswith(('esm2','esm3','esmc','prot_t5','saprot')): return 'protein LM'
    if rep.startswith(('antiberty','ablang','abroberta','igt5')): return 'antibody LM'
    return 'other:'+rep
agg=collections.defaultdict(lambda:collections.Counter())
for r in R:
    tot=sum(r['alloc'].values())
    for k,v in r['alloc'].items(): agg[(r['dom'],r['ds'])][fam(k)]+=v/tot
    pt=sum(max(0,x or 0) for x in (r['perm'] or {}).values())
    for k,v in (r['perm'] or {}).items():
        if pt>0: agg[(r['dom'],r['ds'],'perm')][fam(k)]+=max(0,v or 0)/pt
n=collections.Counter((r['dom'],r['ds']) for r in R)
for k in sorted(agg):
    if len(k)==3: continue
    c=agg[k]; p=agg[k+('perm',)]; N=n[k]
    print(f"{k[0]}|{k[1]}|features: "+'; '.join(f'{f} {v/N:.2f}' for f,v in c.most_common())+" || permutation drop: "+'; '.join(f'{f} {v/N:.2f}' for f,v in p.most_common()))
print()
for ds in ['ddg_combined','flab_binding_koenig','ginkgo_acsins','abagym_VEGF']:
    for r in R:
        if r['ds']==ds and r['fold']==0: print(ds,r['model'],'fold0 features',r['alloc'],'perm',{k:round(v,3) for k,v in (r['perm'] or {}).items()})
