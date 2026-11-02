import sqlite3,re,json,sys
sys.path.insert(0,'.')
from axes import decided,SMALL,top10
pair,rng,ic,ac,ec=decided()
L=open('/Users/apunt/repos/KIRBy/scratchpad/precompute_cuts/DECISIONS.md').read().split('\n')
s6=[i for i,l in enumerate(L) if l.startswith('## Step 6')][0]; s7=[i for i,l in enumerate(L) if l.startswith('## Step 7')][0]
rc={}
for i in range(s6,s7):
    m=re.match(r'\| (\S+) \|.*\*\*cut ([^*]+)\*\*',L[i])
    if m: rc[m.group(1)]=set(m.group(2).replace(' ','').split(','))|set(m.group(2).split('/'))
D={r['ds']:r for r in json.load(open('dist_all.json')) if r['n']}
out={}
for ds in pair:
    c=sqlite3.connect(f'slim/{ds}.slim.db'); pr=pair[ds]
    metric=c.execute('select metric from results limit 1').fetchone()[0]; lower=metric=='rmse'; band=0.03 if ds in SMALL else 0.01
    for tier in ('general','general_distance'):
        if tier=='general_distance':
            if ds not in D: continue
            r=D[ds]; lo,hi=r['budget']['range']; icu=set(r['importance']['cut']);acu=set(r['allocation']['cut']);ecu=set(r['evaluator']['cut']);rcu=set(r['repcount']['cut'])
            if ds in('buchwald_hartwig','ginkgo_colloidal','ginkgo_hic'): ecu.discard('knn')
        else:
            lo,hi=rng[ds];icu=ic.get(ds,set());acu=ac.get(ds,set());ecu=ec.get(ds,set());rcu=rc.get(ds,set())
        rows=[(aug,s) for b,im,al,ev,mo,rep,aug,s in c.execute("select budget,importance_method,allocation_method,internal_evaluator,model,rep_count,aug_strategy,score_mean from results where tier=? and score_mean is not null and model in (?,?)",(tier,pr[0],pr[1]))
              if lo<=int(b)<=hi and im not in icu and al not in acu and not(ev in ecu and ev!=mo) and (rep or 'all') not in rcu]
        res={}
        for a in ('none','capped','uncapped'):
            v=[s for aa,s in rows if aa==a]
            res[a]=(top10(v,lower),len(v)) if len(v)>=10 else None
        out[(ds,tier)]=(metric,lower,band,res)
f=lambda x:'-' if x is None else ('%.3f'%x if abs(x)<100 else '%.1f'%x)
def verdict(res,a,lower,band):
    n=res['none']; x=res[a]
    if not n or not x: return None
    d=(n[0]-x[0]) if lower else (x[0]-n[0])
    if d>band*abs(n[0]): return 'helps'
    if d<-band*abs(n[0]): return 'hurts'
    return 'same'
rows=[]
for (ds,tier),(m,lower,band,res) in out.items():
    cells=[]
    for a in ('none','capped','uncapped'):
        r=res[a]; v=verdict(res,a,lower,band) if a!='none' else ''
        cells.append(f"{f(r[0])} ({r[1]}){' '+v if v else ''}" if r else 'not run')
    rows.append((ds,tier,m,cells,[verdict(res,a,lower,band) for a in('capped','uncapped')]))
json.dump([(r[0],r[1],r[2],r[3],r[4]) for r in rows],open('aug_res.json','w'))
import collections
cnt=collections.Counter()
for r in rows:
    for a,v in zip(('capped','uncapped'),r[4]): cnt[(r[1],a,v)]+=1
print(sorted(cnt.items(),key=str))
for r in rows:
    if any(v=='helps' for v in r[4]): print(r[0],r[1],r[2],r[3])
