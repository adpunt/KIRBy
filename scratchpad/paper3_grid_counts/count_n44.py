"""Rows one n44 unit writes, from n44.run_unit itself with the fitting stubbed out."""
import sys, types, collections
sys.path[:0]=['src','tests','tests/experiments']
import numpy as np
from experiments.noise import n44_detect_and_clean as dc
from experiments.noise import similarity_space as ss
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
def unit(hybrid_input,tw=True,oracle=True,forest=0):
    rows=collections.Counter(); keys=collections.Counter()
    u=types.SimpleNamespace(is_clf=False,takes_weight=tw,mi='m:hybrid' if hybrid_input else 'm:base_1',
        counts={},unavailable={},errors=types.SimpleNamespace(add=lambda *a:None),cmask=None,cliff_train=None,level=1.0,
        ds='x',y0=np.zeros(N),g_tr=None,condition='outlier_p05',oracle=oracle,detection={})
    u.emit=lambda fam,meth,ms,simn,cl,exempt=False:(rows.update([meth]),keys.update([(meth.split('/')[1],ms.split('|',1)[0],simn)]))
    sp=types.SimpleNamespace(u=u,X_tr=None,X_va=None)
    spaces={'own':sp} if hybrid_input else {'own':sp,'hybrid':sp}
    S=[types.SimpleNamespace(name=f's{i}',space=ss.FOREST_SPACE if i<forest else 'other') for i in range(3)]
    sim=types.SimpleNamespace(kept=[(s,5,'rank') for s in S],nb={s.name:{} for s in S},with_scheme=lambda s,sch:{})
    dc.run_unit(u,None,None,spaces,sim,{})
    return rows
def summ(r):
    bycl=collections.Counter(); bydet=collections.Counter()
    for k,v in r.items():
        d,c=k.split('/'); bycl[c]+=v; bydet[d.split('@')[0]]+=v
    return sum(r.values()),bycl,bydet
for lab,kw in [('base input, weights ok, oracle',dict(hybrid_input=False)),
               ('hybrid input, weights ok, oracle',dict(hybrid_input=True)),
               ('base, no oracle (level 0 / no subset)',dict(hybrid_input=False,oracle=False)),
               ('hybrid, no oracle',dict(hybrid_input=True,oracle=False)),
               ('base, model takes no weight',dict(hybrid_input=False,tw=False)),
               ('base, 3 kept settings all forest',dict(hybrid_input=False,forest=3))]:
    t,c,d=summ(unit(**kw)); print(f'{lab}: {t} D-prime rows')
t,c,d=summ(unit(False))
print('by cleanup',c.most_common()); print('by detector',d.most_common(12))
print('reliability_weight rows',c['reliability_weight'],'of which distinct D-prime =',len(ss.__dict__) and 3*4)

# check against the smoke: no neighbour lists, no second learner
sv=dc.flag_detectors
dc.flag_detectors=lambda sp:[(n,np.zeros(N,bool)) for n in CONF]
def smoke_like():
    rows=collections.Counter()
    u=types.SimpleNamespace(is_clf=False,takes_weight=True,mi='rf:base_1',counts={},unavailable={},errors=types.SimpleNamespace(add=lambda *a:None),cmask=None,cliff_train=None,level=1.0,ds='x',y0=np.zeros(N),g_tr=None,condition='outlier_p05',oracle=True,detection={})
    u.emit=lambda fam,meth,ms,simn,cl,exempt=False:rows.update([meth])
    sp=types.SimpleNamespace(u=u,X_tr=None,X_va=None)
    dc.run_unit(u,None,None,{'own':sp,'hybrid':sp},None,{})
    return rows
r=smoke_like(); t,c,d=summ(r); print('smoke-shaped unit:',t,'rows;',len(d),'detectors;',len(c),'cleanups')
dc.flag_detectors=sv

# totals. oracle flags exist only where the corrupted subset is defined and level>0
os_env=__import__('os'); os_env.environ['KIRBY_NOISE_PROMOTE']='1'
from experiments.noise import noise_helpers as nh, noise_cache as nc
from experiments.decided_grid import decided_models
from experiments.noise import n07_remediation
fac=dc.get_model_factories(['dnn','svm'],task_type='regression',n_samples=2000)
tw={m:bool(nc.fit_takes_sample_weight(fac[m])) for m in fac}; print('takes weight',tw)
for ds,n in [('caco2',2158),('logd',5028)]:
    data={'task_type':'regression','labels':list(range(n)),'reps':{'r':None},'smiles':[str(i) for i in range(n)]}
    tot=0; units=0; bykind={}
    for k in dc.default_conditions(data):
        lv=[float(x) for x in dc._condition_levels(data,ds,k)]; dr=len(nh.draws_for(k,(0,1,2)))
        defined=k in nh.DETECTION_IS_DEFINED
        sub=0
        for l in lv:
            np_=1 if l==0 else dr
            for m in ('dnn','svm'):
                for hy in (False,True):
                    x=sum(unit(hybrid_input=hy,tw=tw[m],oracle=defined and l>0).values())
                    sub+=x*np_*5; units+=np_*5
        bykind[k]=sub; tot+=sub
    print(ds,'units',units,'D-prime rows',tot,'with cliff and random put-back x3',3*tot,'with twins as well x9',9*tot); print(' ',bykind)
