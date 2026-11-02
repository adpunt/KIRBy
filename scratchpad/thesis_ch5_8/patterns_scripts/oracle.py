import json,glob,os,sys,collections
sys.path.insert(0,os.path.dirname(__file__)); from common import *
D='/Volumes/seagate/kirby_results_2026-10-09/complexity/c01_rep_characterization'
R=json.load(open('/private/tmp/claude-501/c02_rows.json'))
g=collections.defaultdict(list)
for r in R: g[(r['ds'],r['model'])].append(r)
cnt=collections.defaultdict(lambda:[0,0,0,0])
for (ds,mod),rs in sorted(g.items()):
    d=json.load(open(f'{D}/{ds}.json')); low=d['lower_is_better']; reps=d['_meta']['representations']
    best=sorted([(d[r]['baselines'][mod]['score_mean'],r) for r in reps if mod in d[r].get('baselines',{}) and d[r]['baselines'][mod].get('score_mean') is not None], reverse=not low)[0]
    s=sum(r['single'] for r in rs)/len(rs); h=sum(r['hybrid'] for r in rs)/len(rs)
    sel_ok = abs(s-best[0])<1e-9
    beat = h<best[0] if low else h>best[0]
    c=cnt[domain(ds)]; c[0]+=1; c[1]+=beat; c[2]+=sel_ok; c[3]+= (h<s if low else h>s)
    print(f'{domain(ds)}|{ds}|{mod}|selected single {s:.4f}|best single by held-out ({best[1]}) {best[0]:.4f}|hybrid {h:.4f}|selection found the best single every fold: {sel_ok}|hybrid beats held-out-best single: {beat}')
for k,v in cnt.items(): print(k, f'dataset-model pairs {v[0]}; hybrid beats selected single {v[3]}; hybrid beats held-out-best single {v[1]}; selected single == held-out-best {v[2]}')
