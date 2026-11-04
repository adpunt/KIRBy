import sys,types,collections,os
os.environ['KIRBY_NOISE_PROMOTE']='1'
sys.path[:0]=['src','tests','tests/experiments']
import numpy as np
from experiments.noise import n43_noise_aware_training as na, n45_detector_grading as dg, n44_detect_and_clean as dc, n07_remediation as n07, similarity_space as ss
from experiments.decided_grid import decided_models
from experiments.noise.noise_helpers import draws_for
from kirby.utils import noise_estimation
print('n40 knobs: SEED_AVG_N',na.n40.SEED_AVG_N,'tune settings',getattr(na.n40,'TUNE_N',None),'inner',na.n40.TUNE_INNER_FOLDS,'gp cap',na.n40.gp_max_samples(),'mult',len(na.n40.GP_NUGGET_MULT))
for ds,n,smi in [('caco2',2158,True),('logd',5028,True)]:
    settings=n07.sweep_settings(ds,n,smi); ks=n07.sweep_ks(); sch=n07.sweep_schemes()
    print(ds,'group',ss.knn_group(ds,n),'settings',len(settings),[s.name for s in settings][:30],'ks',ks,'schemes',sch)
    for name,mdl in [('n43',decided_models(ds,'general'))]:
        factories=na.get_model_factories(mdl,task_type='regression',n_samples=n)
        swaps={k:v for k,v in na.n33.build_swaps(na.HUBER_C).items() if v in mdl}
        print(' models',mdl,'swaps',swaps)
        rows=collections.Counter()
        u=types.SimpleNamespace(is_clf=False,unavailable={},errors=types.SimpleNamespace(add=lambda *a:None),ds=ds,
           y0=np.random.rand(900),y_clean=np.random.rand(n),y_val=np.random.rand(40),g_tr=np.arange(n),fold=0,level=1.0,draw=0,condition='gaussian',cmask=None,metric=('rmse',lambda a,b:0,True),classes_=None,
           record=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(dict))),_memo={})
        u.groups_or_rows=lambda:np.arange(n)
        u.ref=lambda key,fn:(1.0,{'intercept':1,'k_max':5,'n_used':5}) if key[0]=='gamma' else None
        u.emit=lambda method,setting,model,space,make,D,C,sim=None,not_run=None:rows.update([(method.split('/')[0],method)])
        host=na.n40.HOST_MODEL if na.n40.HOST_MODEL in mdl else mdl[0]
        ngb=na.n33.ngboost_host(mdl) is not None
        spaces={sp:(np.zeros((n,2)),np.zeros((40,2))) for sp in na.FEATURE_SPACES}
        neigh=lambda space,X,Xv:[{'name':s.name,'blocks':1,'idx_val':np.zeros((40,20)),'dist_val':np.zeros((40,20))} for s in settings]
        na.lnl_space(u,'x',None,None,na.LNL_METHODS)
        na.n33_space(u,'x',None,None,mdl,factories,swaps,ngb,{'resid_sd':1},1.0)
        try:
            na.n40_space(u,'x',np.zeros((min(n,900),2)),np.zeros((40,2)),mdl,factories,host,neigh,ks,sch,None)
        except Exception as e: print(' n40 stop',repr(e)[:200])
        byf=collections.Counter(); 
        for (f,m),v in rows.items(): byf[f]+=v
        print(' n43 rows, one feature space, one (fold,level,draw):',dict(byf),'total',sum(rows.values()))
        print(' n40 methods',{m:v for (f,m),v in rows.items() if f=='n40'})
    # n45
    nd=len(dc.NEIGHBOUR_DETECTORS); det=dg.SCORE_DETECTORS
    rs={d:dg.reads_scheme(d) for d in dc.NEIGHBOUR_DETECTORS}
    keys=sum(len(settings)*len(ks)*(len(sch) if rs[d] else 1) for d in dc.NEIGHBOUR_DETECTORS)
    oracle=len(dg.FLAG_SHARES)*len(dg.PRECISIONS)*len(dg.FLAG_SEEDS)*len(dg.OPERATIONS)
    print(' n45 per unit: score detectors',len(det),'neighbour-setting grades',keys,rs,'oracle fits',oracle,'(shares,prec,seeds,ops)',len(dg.FLAG_SHARES),len(dg.PRECISIONS),len(dg.FLAG_SEEDS),len(dg.OPERATIONS))
