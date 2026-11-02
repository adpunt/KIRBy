import re,json,collections,sys,os
sys.path.insert(0,os.path.dirname(__file__)); from common import *
P='/Volumes/seagate/claude_analysis_2026-10-08/answers/DETECTOR_BY_MITIGATION.md'
rows=[]; ds=cond=exp=None; cols=None
for line in open(P):
    line=line.rstrip('\n')
    if line.startswith('## ') and not line.startswith('## How'): ds=line[3:].strip(); continue
    if line.startswith('### '): cond=line[4:].strip(); continue
    if line.startswith('#### '): exp=line[5:].split(',')[0].split(' ')[0]; expfull=line[5:]; continue
    if line.startswith('| detector |'): cols=[c.strip() for c in line.strip('|').split('|')][1:]; continue
    if line.startswith('|---'): continue
    if line.startswith('|') and cols and 'the three best' not in (cond or ''):
        c=[x.strip() for x in line.strip('|').split('|')]
        det=c[0]
        for mit,cellv in zip(cols,c[1:]):
            m=re.search(r'(\d+)/(\d+) improved, median ([+-]?\d+\.\d+), best ([+-]?\d+\.\d+)',cellv)
            if not m: continue
            b=re.search(r'beat (.*?) (\d+)/(\d+)',cellv)
            sh=re.search(r'single ([+-]?\d+\.\d+|—) / hybrid ([+-]?\d+\.\d+|—)',cellv)
            rows.append(dict(ds=ds,dom=domain(ds),cond=cond,exp=exp,expfull=expfull,det=det,mit=mit,k=int(m.group(1)),N=int(m.group(2)),
                med=float(m.group(3)),best=float(m.group(4)),ctl_name=b.group(1) if b else None,beat=int(b.group(2)) if b else None,
                sbest=sh.group(1) if sh else None,hbest=sh.group(2) if sh else None))
    elif not line.startswith('|'): cols=None if line.startswith('#') else cols
json.dump(rows,open('/private/tmp/claude-501/dbm_rows.json','w'))
print(len(rows)); print(collections.Counter(r['exp'] for r in rows)); print(collections.Counter(r['det'].split(',')[0] for r in rows).most_common(60))
print(collections.Counter(r['cond'] for r in rows).most_common())
