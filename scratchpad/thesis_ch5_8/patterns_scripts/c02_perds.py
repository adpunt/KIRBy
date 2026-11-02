import json,collections
R=json.load(open('/private/tmp/claude-501/c02_rows.json'))
g=collections.defaultdict(list)
for r in R: g[(r['dom'],r['ds'],r['model'])].append(r)
print('dom|ds|model|metric|ntrain(f0)|single mean|hybrid mean|folds hybrid better|budgets chosen|rc chosen|imp chosen|alloc chosen|ev chosen|base1 names')
for k in sorted(g):
    rs=sorted(g[k],key=lambda r:r['fold'])
    s=sum(r['single'] for r in rs)/len(rs); h=sum(r['hybrid'] for r in rs)/len(rs)
    C=lambda f: ','.join(f'{a}x{b}' if b>1 else str(a) for a,b in collections.Counter(str(r[f]) for r in rs).most_common())
    print('|'.join(map(str,[k[0],k[1],k[2],rs[0]['metric'],rs[0]['ntrain'],f'{s:.4f}',f'{h:.4f}',f"{sum(r['win'] for r in rs)}/{len(rs)}",C('budget'),C('rc'),C('imp'),C('alloc'),C('ev'),C('base1')])))
