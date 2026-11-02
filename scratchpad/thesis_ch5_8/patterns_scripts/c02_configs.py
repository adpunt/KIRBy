import json,glob,os,sys,math,collections
sys.path.insert(0,os.path.dirname(__file__)); from common import *
D='/Volumes/seagate/kirby_results_2026-10-09/complexity/c02_complexity_matching_grid'
C01='/Volumes/seagate/kirby_results_2026-10-09/complexity/c01_rep_characterization'
rows=[]
for f in sorted(glob.glob(D+'/*.json')):
    ds=os.path.basename(f)[:-5]
    if ds in CUT: continue
    d=json.load(open(f))
    if 'standard_selection_summary' not in d: continue
    low=d['lower_is_better']
    ntr={pf['fold']:pf.get('n_train') for pf in d['per_fold']}
    for m in d['models']:
        s=d['standard_selection_summary'][m]
        b={p['fold']:p for p in s['base1']['per_fold']}
        for p in s['hybrid1']['per_fold']:
            fo=p['fold']
            if fo not in b: continue
            a,h=b[fo]['heldout_score'],p['heldout_score']
            if a is None or h is None or (isinstance(a,float) and math.isnan(a)) or (isinstance(h,float) and math.isnan(h)): continue
            parts=p['name'].split('/')
            win = h<a if low else h>a
            rows.append(dict(ds=ds,dom=domain(ds),metric=d['metric_name'],low=low,model=m,fold=fo,
              alloc=parts[0],imp=parts[1],budget=int(parts[2]),ev=parts[3],rc=parts[5],
              base1=b[fo]['name'],single=a,hybrid=h,win=win,ntrain=ntr.get(fo),
              delta=(a-h) if low else (h-a)))
json.dump(rows,open('/private/tmp/claude-501/c02_rows.json','w'))
print(len(rows),'fold rows')
