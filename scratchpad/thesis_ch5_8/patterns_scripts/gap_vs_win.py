import json,glob,os,sys,collections
sys.path.insert(0,os.path.dirname(__file__)); from common import *
D='/Volumes/seagate/kirby_results_2026-10-09/complexity/c01_rep_characterization'
R=json.load(open('/private/tmp/claude-501/c02_rows.json'))
g=collections.defaultdict(list)
for r in R: g[(r['ds'],r['model'])].append(r)
out=[]
for (ds,mod),rs in g.items():
    f=f'{D}/{ds}.json'
    if not os.path.exists(f): continue
    d=json.load(open(f)); low=d['lower_is_better']; reps=d['_meta']['representations']
    sc=sorted([d[r]['baselines'][mod]['score_mean'] for r in reps if r in d and mod in d[r].get('baselines',{}) and d[r]['baselines'][mod].get('score_mean') is not None], reverse=not low)
    if len(sc)<3: continue
    gap=abs(sc[0]-sc[2]); s=sum(r['single'] for r in rs)/len(rs); h=sum(r['hybrid'] for r in rs)/len(rs)
    out.append((domain(ds),ds,mod,sc[0],sc[2],gap,s,h,sum(r['win'] for r in rs),len(rs)))
for o in sorted(out,key=lambda o:(o[0],o[5])):
    print(f'{o[0]}|{o[1]}|{o[2]}|best single (c01) {o[3]:.4f}|3rd {o[4]:.4f}|gap {o[5]:.4f}|c02 single {o[6]:.4f} -> hybrid {o[7]:.4f}|hybrid better {o[8]}/{o[9]} folds')
