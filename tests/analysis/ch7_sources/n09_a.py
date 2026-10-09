import sys; sys.path.insert(0,'/private/tmp/claude-501/-Users-apunt-repos-KIRBy/f55db1bf-8fce-4a29-ab8b-783ed5b81253/scratchpad')
from n09_load import *
for ds,kind,f in files():
    d=json.load(open(f))
    vp=d['vs_plain_dnn_same_labels']
    print(ds,kind,d['metric'],'folds',d.get('folds_run'),'levels',list(vp.keys()),'draws',d.get('noise_draws_per_nonzero_level'),'status',d.get('result_status'))
    lv=list(vp.keys())[0]; c=vp[lv]
    print('  conds',list(c.keys()), 'methods',len(c[list(c)[0]]))
    ps=d['per_sigma'][lv]; print('  per_sigma keys',list(ps.keys())[:8]); 
    k=[k for k in ps if k.endswith('_no_lnl')][0]; print('  ',k,{a:b for a,b in ps[k].items() if not isinstance(b,list)})
    break
