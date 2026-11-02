import json, re, collections
A='/Volumes/seagate/claude_analysis_2026-10-08/answers/'
rows=json.load(open('rows.json'))
G={(g['section'],g['method']):g for g in json.load(open(A+'_scripts/method_groups.json'))}
EXTRA={('Label smoothing toward the nearest neighbours',None):('n24','label_smooth','detector','move the label','every label nudged toward its neighbours\' mean, by its suspicion score'),
 ('kNN label replacement',None):('n07','knn_smooth','detector','move the label','top 5% suspect labels replaced by the neighbours\' mean'),
 ('Detect and remove (checked against random removal)',None):('n18','remove (vs random removal)','detector','delete','most suspect molecules deleted; control deletes the same number at random'),
 ("DivideMix (training two networks on each other's clean molecules)",None):('n09','DivideMix','none','none','two networks each train on the molecules the other finds clean'),
}
SEC_EXP={'Fixing labels clipped at an assay limit':'n30'}
def exp_of(sec):
    m=re.search(r'\((n\d+)',sec); return m.group(1) if m else SEC_EXP.get(sec,'?')
inst=collections.OrderedDict()
for r in rows:
    k=(r['section'],r['method'])
    if k in G:
        g=G[k]; e=exp_of(k[0]); name=k[1]; sel=g['selects']; act=g['action']; note=g['note']+(' (CHECK)' if g['check'] else '')
    else:
        e,name,sel,act,note=EXTRA[k]
    if k[0].startswith('Using how often') : e='n19'
    if k[0].startswith('Training on every'): e='n29'
    inst.setdefault((e,name,sel,act,note),[]).append(r)
def better(metric,a,b): return b<a if metric=='rmse' else b>a
out=[]
for (e,name,sel,act,note),rs in inst.items():
    v=[r for r in rs if r['base'] and r['hyb']]
    ds=sorted({r['ds'] for r in rs})
    bh=sum(r['base'][2] for r in v); hh=sum(r['hyb'][2] for r in v); both=sum(r['works']=='yes' for r in v)
    hb=sum(r['hb']=='yes' for r in v)
    real=[r for r in v if r['noise'].startswith('none added')]
    # biggest hybrid gain relative
    def gain(r):
        a,b,_=r['hyb']; d=(a-b) if r['metric']=='rmse' else (b-a)
        return d/abs(a) if r['metric']=='rmse' else d
    ex=max(v,key=gain) if v else None
    out.append(dict(exp=e,method=name,sel=sel,act=act,note=note,ds=ds,N=len(v),na=len(rs)-len(v),bh=bh,hh=hh,both=both,hb=hb,
        realN=len(real),real_b=sum(r['base'][2] for r in real),real_h=sum(r['hyb'][2] for r in real),
        ex=(f"{ex['ds']}, {ex['noise']}, {ex['model']}: base {ex['base'][0]}→{ex['base'][1]}, hybrid {ex['hyb'][0]}→{ex['hyb'][1]} {ex['metric']}" if ex else '')))
json.dump(out,open('agg.json','w'),indent=0)
for o in out: print(o['exp'],o['method'],o['sel'],o['act'],o['N'],o['bh'],o['hh'],len(o['ds']))
