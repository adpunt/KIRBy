"""Parse answers/NOISE_BY_GROUP.md (built 2026-10-09 from raw fold-0 results) into rows. Read-only."""
import re,json,sys,os
sys.path.insert(0,os.path.dirname(__file__)); from common import *
P='/Volumes/seagate/claude_analysis_2026-10-08/answers/NOISE_BY_GROUP.md'
rows=[]; group=sel=action=method=section=None; header=None
num=r'(-?\d+\.\d+)'
cell=re.compile(num+r' → '+num+r' ([✓✗])')
for line in open(P):
    line=line.rstrip('\n')
    if line.startswith('## '):
        group=line[3:].strip(); sel=action=method=None; continue
    if line.startswith('### '):
        m=re.match(r'### Selected by (.*?)(?:; action: (.*))?$',line)
        if m: sel,action=m.group(1),(m.group(2) or 'none')
        continue
    if line.startswith('#### '):
        method=line[5:].strip(); section=None; continue
    if line.startswith('From section'):
        m=re.search(r'\((n\d+[^)]*|c\d+)\)',line); section=m.group(1) if m else line[14:80]
        desc=line; continue
    if line.startswith('| domain | dataset | noise | model'):
        header=[h.strip() for h in line.strip('|').split('|')]; continue
    if line.startswith('|') and header and method and not line.startswith('|---'):
        c=[x.strip() for x in line.strip('|').split('|')]
        if len(c)!=len(header): continue
        r=dict(zip(header,c))
        ds=r['dataset'].lstrip('*'); strong=r['dataset'].startswith('*')
        run=None
        m=re.match(r'(\S+) \((n\d+) run\)',ds)
        if m: ds,run=m.group(1),m.group(2)
        s=cell.search(r['single representation']); h=cell.search(r['hybrid'])
        ctl=[v for k,v in r.items() if k.startswith(('random','shuffled'))]
        rows.append(dict(group=group,sel=sel,action=action,method=method,section=section,ds=ds,dom=r['domain'],noise=r['noise'],model=r['model'],
            metric=r['metric'],low='lower' in r['metric'],
            s0=float(s.group(1)) if s else None,s1=float(s.group(2)) if s else None,s_ok=(s.group(3)=='✓') if s else None,
            h0=float(h.group(1)) if h else None,h1=float(h.group(2)) if h else None,h_ok=(h.group(3)=='✓') if h else None,
            works=r['does it work?'],hyb_better=r['hybrid better than base?'],control=(ctl[0] if ctl else None),strong=strong,desc=desc))
json.dump(rows,open('/private/tmp/claude-501/noise_rows.json','w'))
print(len(rows)); import collections
print(collections.Counter((r['group'],r['sel'],r['action']) for r in rows))
print(collections.Counter(r['noise'] for r in rows).most_common())
print(collections.Counter(r['model'] for r in rows))
print(collections.Counter(r['dom'] for r in rows))
