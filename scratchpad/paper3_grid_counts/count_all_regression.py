"""First-run results per Paper 3 regression dataset, from n44.run_unit with fitting stubbed out.
Level 0 once. No put-back, no twins. Before the grid review. Sizes: dataset_policy, else listed below."""
import os,sys,types,collections
os.environ['KIRBY_NOISE_PROMOTE']='n44'
sys.path[:0]=['src','tests','tests/experiments']
import numpy as np
from experiments.noise import n44_detect_and_clean as dc, similarity_space as ss, noise_helpers as nh, noise_cache as nc
from experiments.orchestrate import plan as OP
from experiments.papers import plan as P
from experiments.decided_grid import decided_models
from experiments import dataset_policy as dp
N=40
dc.nc.cached_detector_scores=lambda *a,**k: np.linspace(0,1,N)
dc.detector_scores=lambda sp,name: np.linspace(0,1,N)
dc.detection_scores=lambda *a,**k: None
dc._transfer=lambda c,X: c
CONF=[f'conformal_bh@{i}' for i in range(len(dc.CONFORMAL_FLAG_SETTINGS))]
TWO=[f'{dc.TWO_LEARNER_DETECTOR}@{s}' for s in dc.n32.ITLM_ALPHAS]
dc.flag_detectors=lambda sp: [(n,np.zeros(N,bool)) for n in CONF+TWO]
dc.oracle_flags=lambda u: [(f'oracle@precision={p:g}|flag_share={s:g}',np.zeros(N,bool)) for s in dc.FLAG_SHARES for p in dc.PRECISIONS] if u.oracle else []
dc.apply_cleanup=lambda *a,**k:{'ok':1}
memo={}
def unit(hy,tw,oracle):
    k=(hy,tw,oracle)
    if k in memo: return memo[k]
    n=[0]
    u=types.SimpleNamespace(is_clf=False,takes_weight=tw,mi='m:hybrid' if hy else 'm:base_1',counts={},unavailable={},errors=types.SimpleNamespace(add=lambda *a:None),cmask=None,cliff_train=None,level=1.0,ds='x',y0=np.zeros(N),g_tr=None,condition='outlier_p05',oracle=oracle,detection={})
    u.emit=lambda fam,meth,ms,simn,cl,exempt=False:n.__setitem__(0,n[0]+1)
    sp=types.SimpleNamespace(u=u,X_tr=None,X_va=None)
    S=[types.SimpleNamespace(name=f's{i}',space='other') for i in range(3)]
    sim=types.SimpleNamespace(kept=[(s,5,'rank') for s in S],nb={s.name:{} for s in S},with_scheme=lambda s,sch:{})
    dc.run_unit(u,None,None,{'own':sp} if hy else {'own':sp,'hybrid':sp},sim,{})
    memo[k]=n[0]; return n[0]
SIZES={'qm9':5000,'qm9_mu':5000,'qm9_alpha':5000}
tot=0
print('dataset,molecules,size_source,grid,versions,pieces,results')
for ds in P.PAPER3_DATASETS:
    if OP.dataset_info(ds)[1+1-1] is None: pass
    info=OP.dataset_info(ds)
    if 'regression' not in info: continue
    pol=dp.DATASETS.get(ds) if hasattr(dp,'DATASETS') else None
    n=info[-1] or SIZES.get(ds) or getattr(pol,'n_samples',None)
    src='known' if n else 'NOT KNOWN'
    grp=ss.knn_group(ds,n) if n else ss.knn_group(ds)
    data={'task_type':'regression','labels':[0]*(n or (6000 if grp=='fast_knn' else 1000)),'reps':{'r':None},'smiles':[str(i) for i in range(n or (6000 if grp=='fast_knn' else 1000))]}
    ms=[m for m in decided_models(ds,'general') if m!='knn']
    fac=dc.get_model_factories(ms,task_type='regression',n_samples=len(data['labels']))
    tw={m:bool(nc.fit_takes_sample_weight(fac[m])) for m in fac}
    res=vers=pieces=0
    for k in dc.default_conditions(data):
        lv=[float(x) for x in dc._condition_levels(data,ds,k)]; dr=len(nh.draws_for(k,(0,1,2)))
        for l in lv:
            if l==0 and not nh.runs_level0(data,k): continue
            np_=1 if l==0 else dr; vers+=np_
            for m in fac:
                for hy in (False,True):
                    res+=unit(hy,tw[m],k in nh.DETECTION_IS_DEFINED and l>0)*np_*5; pieces+=np_*5
    tot+=res
    print(f'{ds},{n},{src},{grp},{vers},{pieces},{res}')
print('total results',tot)
