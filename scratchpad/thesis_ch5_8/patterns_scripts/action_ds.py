import json,collections
R=json.load(open('/private/tmp/claude-501/noise_rows.json'))
g=collections.defaultdict(lambda:[0,0])
for r in R:
    if r['group']!='Noise mitigation' and r['group']!='Noise-robust training': continue
    a=('robust training' if r['group']=='Noise-robust training' else f"{r['sel']}, {r['action']}")
    k=(r['dom'],r['ds'],a); g[k][0]+= bool(r['s_ok'] and r['h_ok']); g[k][1]+=1
cur=None
for k in sorted(g):
    if k[:2]!=cur: cur=k[:2]; print('\n'+' / '.join(cur), end=': ')
    print(f"{k[2]} {g[k][0]}/{g[k][1]}", end='; ')
print()
