import json,collections,re
R=json.load(open('/private/tmp/claude-501/noise_rows.json'))
def ntype(n):
    if n.startswith('none added'): return 'none (labels as delivered)'
    if 'clipped' in n: return 'clipped at limit'
    if 'outliers' in n: return 'outliers 5%'
    if '10% of labels shifted' in n: return '10% shifted'
    if 'shifted upward' in n: return '10% shifted upward'
    if 'random noise on every' in n: return 'gaussian, every label'
    if 'series shifted' in n: return 'per-scaffold-group offset'
    if 'flip' in n: return 'label flips (herg)'
    if 'simulated sources' in n: return 'simulated sources'
    if "model's error" in n: return 'heteroscedastic gaussian'
    return n
for r in R:
    r['nt']=ntype(r['noise']); r['grp']=f"{r['group']} | {r['sel']} | {r['action']}"
    r['mfam']={'dnn':'dnn','svm':'svm','lightgbm':'trees','rf':'trees','xgboost':'trees'}.get(r['model'],'other')
D=[r for r in R if r['dom']!='QM9 CONTROL']
def rate(rows,k):
    v=[r[k] for r in rows if r[k] is not None]; return f"{sum(v)}/{len(v)}"
def tab(rows,key,title):
    print(f'\n### {title}')
    print('| value | rows | helped single | helped hybrid | both (works, ignoring control) | datasets |')
    g=collections.defaultdict(list)
    for r in rows: g[key(r)].append(r)
    for k in sorted(g,key=str):
        rs=g[k]; both=sum(1 for r in rs if r['s_ok'] and r['h_ok'])
        print(f"| {k} | {len(rs)} | {rate(rs,'s_ok')} | {rate(rs,'h_ok')} | {both}/{len(rs)} | {len(set(r['ds'] for r in rs))} |")
for dom in ['small molecules','antibodies','reactions']:
    rs=[r for r in D if r['dom']==dom]
    print(f'\n## {dom}')
    tab(rs,lambda r:r['nt'],'by noise type')
    tab(rs,lambda r:r['grp'],'by group / what selects / action')
    tab(rs,lambda r:r['mfam'],'by model family (unpaired; dnn over-represented because the twelve-method sweep is dnn only)')

print('\n\n######## PAIRED model comparison: same method, section, dataset, noise condition')
key=lambda r:(r['method'],r['section'],r['ds'],r['noise'],r['desc'][:60])
for dom in ['small molecules','antibodies','reactions']:
    g=collections.defaultdict(dict)
    for r in D:
        if r['dom']==dom: g[key(r)][r['mfam']]=r
    for a,b in [('dnn','svm'),('dnn','trees'),('svm','trees')]:
        pairs=[(v[a],v[b]) for v in g.values() if a in v and b in v]
        if not pairs: continue
        wa=sum(1 for x,y in pairs if x['s_ok'] and x['h_ok']); wb=sum(1 for x,y in pairs if y['s_ok'] and y['h_ok'])
        ha=sum(1 for x,y in pairs if x['h_ok']); hb=sum(1 for x,y in pairs if y['h_ok'])
        dss=sorted(set(x['ds'] for x,y in pairs))
        print(f'{dom}: {a} vs {b}: {len(pairs)} paired conditions on {dss}; works for both reps: {a} {wa}, {b} {wb}; helped hybrid: {a} {ha}, {b} {hb}')
        # per dataset
        for ds in dss:
            pp=[(x,y) for x,y in pairs if x['ds']==ds]
            print(f'    {ds}: {len(pp)} conditions; works {a} {sum(1 for x,y in pp if x["s_ok"] and x["h_ok"])}, {b} {sum(1 for x,y in pp if y["s_ok"] and y["h_ok"])}')

print('\n\n######## noise type x what selects x action, small molecules only, "works (both)"')
rs=[r for r in D if r['dom']=='small molecules']
g=collections.defaultdict(lambda:[0,0])
for r in rs:
    k=(r['nt'],r['sel'],r['action']); g[k][0]+= (r['s_ok'] and r['h_ok']) or 0; g[k][1]+=1
for k in sorted(g): print(' | '.join(k), f'{g[k][0]}/{g[k][1]}')
