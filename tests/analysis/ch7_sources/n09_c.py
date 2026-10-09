import pickle,collections,math,sys
rows=pickle.load(open('n09_rows.pkl','rb'))
G=collections.defaultdict(dict)  # (ds,kind,lv,m) -> sp -> row
for r in rows: G[(r['ds'],r['kind'],r['lv'],r['m'])][r['sp']]=r
DOM={'qm9':'small','caco2':'small','herg_ki':'small','logd':'small','herg':'small','buchwald_hartwig':'reaction'}
def dom(ds): return DOM.get(ds,'antibody')
def ok(x): return x is not None and x==x
methods=sorted({r['m'] for r in rows})
mode=sys.argv[1] if len(sys.argv)>1 else 'rank'
def pick(spd,cands):
    if mode=='rank': return spd.get(cands[0])
    # held-out best plain among cands
    key='plain' if mode=='heldout' else 'meth'
    c=[spd[s] for s in cands if s in spd and ok(spd[s][key])]
    if not c: return None
    lib=c[0]['lib']
    return (min if lib else max)(c,key=lambda r:r[key])
# Build row-level verdicts. damaged rows: each (ds,kind,lv>0). level0: once per ds, mean gain over kind files.
verd=collections.defaultdict(dict) # (ds,label) -> m -> (gs,gh,detail)
for (ds,kind,lv,m),spd in G.items():
    s=pick(spd,['base_1','base_2']); h=pick(spd,['hybrid','hybrid__general2'])
    if not s or not h or not ok(s['gain']) or not ok(h['gain']): continue
    if lv>0: verd[(ds,f'{kind}@{lv}')][m]=(s['gain'],h['gain'],s,h)
    else: verd[(ds,'L0')].setdefault(m,[]).append((s['gain'],h['gain'],s,h))
for k,v in verd.items():
    if k[1]=='L0':
        for m,lst in v.items():
            v[m]=(sum(a for a,_,_,_ in lst)/len(lst),sum(b for _,b,_,_ in lst)/len(lst),lst[0][2],lst[0][3],len(lst))
tab=collections.defaultdict(lambda:[0,0]); tab0=collections.defaultdict(lambda:[0,0])
for (ds,lab),v in verd.items():
    for m,t in v.items():
        both=t[0]>0 and t[1]>0
        tab[(m,dom(ds))][0]+=both; tab[(m,dom(ds))][1]+=1
        if lab=='L0': tab0[(m,dom(ds))][0]+=both; tab0[(m,dom(ds))][1]+=1
print('mode',mode)
print('rows per domain:',collections.Counter(dom(ds) for (ds,lab) in verd))
print('method | small | reaction | antibody || L0 small | L0 reaction | L0 antibody')
for m in methods:
    print(m,' | '.join(f"{tab[(m,d)][0]}/{tab[(m,d)][1]}" for d in ['small','reaction','antibody']),'||',' | '.join(f"{tab0[(m,d)][0]}/{tab0[(m,d)][1]}" for d in ['small','reaction','antibody']))
pickle.dump({k:{m:t[:2] for m,t in v.items()} for k,v in verd.items()},open(f'n09_verd_{mode}.pkl','wb'))
