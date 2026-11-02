import sqlite3,re,json
LOW={'rmse'}
dec={}
for l in open('/Users/apunt/repos/KIRBy/scratchpad/precompute_cuts/DECISIONS.md'):
    m=re.match(r'\| (\S+) \|.*\*\*(\S+)\*\* \(2026',l)
    if m: dec[m.group(1)]=m.group(2).split('+')
BUD=[10,25,50,100,250,500,1000,2000]
res={}
for ds,pair in dec.items():
    c=sqlite3.connect(f'slim/{ds}.slim.db')
    m=c.execute('select metric from results limit 1').fetchone()[0]
    lower=m in LOW
    q=f"select budget,score_mean,score_std from results where aug_strategy='none' and score_mean is not null and model in ({','.join('?'*2)}) and internal_evaluator in ({','.join('?'*2)})"
    rows=c.execute(q,pair+pair).fetchall()
    by={}
    for b,s,sd in rows:
        b=int(b)
        if b not in by or (s<by[b][0] if lower else s>by[b][0]): by[b]=(s,sd or 0)
    best=(min if lower else max)(by.values(),key=lambda x:x[0])
    spread=best[1]
    keep=[b for b in by if abs(by[b][0]-best[0])<=spread]
    bb=[b for b in by if by[b][0]==best[0]][0]
    res[ds]=dict(pair=pair,metric=m,by=by,best=bb,spread=spread,keep=sorted(keep),n=len(rows))
json.dump({k:{**v,'by':{str(a):b for a,b in v['by'].items()}} for k,v in res.items()},open('budgets.json','w'))
f=lambda x:('%.4f'%x if abs(x)<100 else '%.2f'%x)
print(f"{'dataset':24}{'metric':11}{'pair':20}"+''.join(f'{b:>9}' for b in BUD)+'  best  spread  keep range')
for ds,v in res.items():
    cells=''
    for b in BUD:
        if b in v['by']:
            s=v['by'][b][0]; cells+=f"{f(s)+('*' if b==v['best'] else ' '):>9}"
        else: cells+=f"{'-':>9}"
    k=v['keep']
    print(f"{ds:24}{v['metric']:11}{'+'.join(v['pair']):20}{cells}  {v['best']:>4}  {f(v['spread'])}  {k[0]}-{k[-1]}")
