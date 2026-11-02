import json,collections,statistics as st
R=json.load(open('/private/tmp/claude-501/dbm_rows.json'))
for r in R:
    r['detf']=r['det'].split(',')[0].split('@')[0]
    r['c']=r['cond'].split(' (')[0]
def line(rs):
    ctl=[r for r in rs if r['beat'] is not None]
    s=f"cells {len(rs)}; pairs improved {sum(r['k'] for r in rs)}/{sum(r['N'] for r in rs)}; cells >=half improved {sum(1 for r in rs if 2*r['k']>=r['N'])}/{len(rs)}"
    if ctl: s+=f"; with control: pairs improved AND beat control {sum(r['beat'] for r in ctl)}/{sum(r['N'] for r in ctl)} ({len(ctl)} cells)"
    return s
print('## per experiment x detector x domain (n11 excluded)')
g=collections.defaultdict(list)
for r in R:
    if r['exp']=='n11': continue
    g[(r['dom'],r['exp'],r['detf'])].append(r)
for k in sorted(g): print(' | '.join(k),'|',line(g[k]),'|',','.join(sorted(set(r['ds'] for r in g[k]))))
print('\n## control beat by dataset, as delivered vs damaged, experiments with a control')
g=collections.defaultdict(list)
for r in R:
    if r['beat'] is None: continue
    g[(r['dom'],r['ds'],'as delivered' if r['c']=='as delivered' else 'damaged')].append(r)
for k in sorted(g): print(' | '.join(k),'|',line(g[k]),'|',','.join(sorted(set(r['exp']+':'+r['ctl_name'] for r in g[k]))))
