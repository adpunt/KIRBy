import json,collections,statistics as st
R=json.load(open('/private/tmp/claude-501/dbm_rows.json'))
n12=[r for r in R if r['exp']=='n12']
for r in n12:
    r['space']=r['det'].split('neighbours in ')[-1]; r['detf']=r['det'].split(',')[0]
    r['c']='as delivered' if r['cond'].startswith('as delivered') else 'damaged'
for dom in ['small_molecule','antibody','reaction']:
    print('\n##',dom)
    for key,name in [(lambda r:r['space'],'neighbour space'),(lambda r:r['mit'],'distance scaling'),(lambda r:(r['c'],r['space']),'condition x space')]:
        g=collections.defaultdict(list)
        for r in n12:
            if r['dom']==dom: g[key(r)].append(r)
        print(f'  [{name}]')
        for k in sorted(g,key=str):
            rs=g[k]; print(f"    {k}: cells {len(rs)}; pairs improved {sum(r['k'] for r in rs)}/{sum(r['N'] for r in rs)}; median of cell medians {st.median(r['med'] for r in rs):+.4f}; datasets {len(set(r['ds'] for r in rs))}")
# per dataset small molecules: hybrid space vs base_1 space, improved share
print('\n## per dataset: share of pairs improved, neighbours in base_1 vs hybrid (n12, all conditions, all scalings)')
g=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0]))
for r in n12:
    g[(r['dom'],r['ds'])][r['space']][0]+=r['k']; g[(r['dom'],r['ds'])][r['space']][1]+=r['N']
for k in sorted(g): print(k, {s:f'{a}/{b}' for s,(a,b) in sorted(g[k].items())})
