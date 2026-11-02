import json,collections,statistics as st
R=json.load(open('/private/tmp/claude-501/dbm_rows.json'))
n11=[r for r in R if r['exp']=='n11']
print('n11 datasets',sorted(set(r['ds'] for r in n11)),'mitigations',sorted(set(r['mit'] for r in n11)),'conds',sorted(set(r['cond'] for r in n11)))
def summ(rows,key,title):
    print(f'\n### {title}')
    print('| value | cells | pairs improved (sum k / sum N, at each cell\'s best setting) | cells with at least half the pairs improved | median of cell medians (metric units, mixed metrics!) | datasets |')
    g=collections.defaultdict(list)
    for r in rows: g[key(r)].append(r)
    for k in sorted(g,key=str):
        rs=g[k]
        print(f"| {k} | {len(rs)} | {sum(r['k'] for r in rs)}/{sum(r['N'] for r in rs)} | {sum(1 for r in rs if 2*r['k']>=r['N'])}/{len(rs)} | {st.median(r['med'] for r in rs):+.4f} | {','.join(sorted(set(r['ds'] for r in rs)))} |")
for dom in ['small_molecule','antibody','reaction']:
    rs=[r for r in n11 if r['dom']==dom]
    if rs: summ(rs,lambda r:r['det'],f'n11 detectors, {dom}')
    if rs: summ(rs,lambda r:r['mit'],f'n11 mitigations, {dom}')
    if rs: summ(rs,lambda r:r['cond'].split(' (')[0],f'n11 noise condition, {dom}')
