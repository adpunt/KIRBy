import json,glob,os,sys,math,collections
sys.path.insert(0,os.path.dirname(__file__)); from common import *
D='/Volumes/seagate/kirby_results_2026-10-09/complexity/c22_explainability'
def eff(dct):
    v=[max(x,0) for x in dct.values() if x is not None and not (isinstance(x,float) and math.isnan(x))]
    s=sum(v)
    if s<=0: return None,None,None
    p=[x/s for x in v]
    return 1/sum(q*q for q in p), max(p), sum(1 for q in p if q>0)
out=[]
for f in sorted(glob.glob(D+'/*.json')):
    ds=os.path.basename(f)[:-5]
    if ds in CUT: continue
    d=json.load(open(f)); low=d['lower_is_better']
    if 'per_fold' not in d: continue
    npool=len(d.get('representations',[]))
    for pf in d['per_fold']:
        for m,md in pf.get('models',{}).items():
            b=[x for x in md.get('base',[]) if x['slot']=='base1']
            h=[x for x in md.get('hybrids',[]) if x['slot']=='hybrid1']
            if not b or not h: continue
            b,h=b[0],h[0]
            alloc=h.get('allocation_built') or h.get('allocation') or {}
            alloc={k:v for k,v in alloc.items() if v}
            ea,ta,na=eff(alloc)
            ep,tp,_=eff(h.get('perm_share_by_rep') or {})
            es,ts,_=eff(h.get('shap_share_by_rep') or {})
            topalloc=max(alloc,key=alloc.get) if alloc else None
            a,hh=b['heldout_score'],h['heldout_score']
            win=None if a is None or hh is None else (hh<a if low else hh>a)
            out.append(dict(ds=ds,dom=domain(ds),model=m,fold=pf['fold'],npool=npool,nrep_alloc=na,eff_alloc=ea,top_alloc_share=ta,
                eff_perm=ep,top_perm_share=tp,eff_shap=es,top_shap_share=ts,base1=b['rep'],top_alloc_rep=topalloc,
                base1_in_hybrid=(b['rep'] in alloc), base1_alloc_share=(alloc.get(b['rep'],0)/sum(alloc.values()) if alloc else None),
                budget=h['budget'],nfeat=h['n_features'],single=a,hybrid=hh,win=win,metric=d['metric'],
                perm=h.get('perm_share_by_rep'),alloc=alloc))
json.dump(out,open('/private/tmp/claude-501/c22_rows.json','w'))
print(len(out))
g=collections.defaultdict(list)
for r in out: g[(r['dom'],r['ds'],r['model'])].append(r)
def m(xs):
    xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else float('nan')
print('dom|ds|model|pool|folds|reps given features (mean)|effective reps by features|top rep share of features|effective reps by permutation drop|top rep share of permutation drop|base1 share of features|single mean|hybrid mean|hybrid better folds|top rep by features (per fold)')
for k in sorted(g):
    rs=g[k]
    print('|'.join(map(str,[*k,rs[0]['npool'],len(rs),f"{m([r['nrep_alloc'] for r in rs]):.1f}",f"{m([r['eff_alloc'] for r in rs]):.2f}",f"{m([r['top_alloc_share'] for r in rs]):.2f}",
      f"{m([r['eff_perm'] for r in rs]):.2f}",f"{m([r['top_perm_share'] for r in rs]):.2f}",f"{m([r['base1_alloc_share'] for r in rs]):.2f}",
      f"{m([r['single'] for r in rs]):.4f}",f"{m([r['hybrid'] for r in rs]):.4f}",f"{sum(1 for r in rs if r['win'])}/{len(rs)}",
      ','.join(f'{a}x{b}' for a,b in collections.Counter(r['top_alloc_rep'] for r in rs).most_common())])))
