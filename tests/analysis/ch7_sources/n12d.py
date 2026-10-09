import json,glob,os
R='/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n12_importance_weighted_distance'
DS=['qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins']
GR={'outlier_p05':'1.0','shift_only_p10':'1.0','censoring':'0.25','sparse_dprime_up_p10':'1.0','binary_asymmetric':'0.1','class_dependent':'0.1','class_imbalance':'0.1','confusion_directed':'0.1','instance_noise':'0.1','uniform':'0.1'}
from statistics import mean
res={}
for f in sorted(glob.glob(R+'/*__*.json')):
    b=os.path.basename(f)[:-5]; ds,cond=b.split('__',1)
    if ds not in ('ginkgo_titer','herg_ki','ddg_combined','ginkgo_acsins','flab_thermo_tresanco','caco2') or cond not in GR: continue
    d=json.load(open(f)); ps=d['per_sigma']
    lv=GR[cond]
    if lv not in ps:
        print(b,'levels',list(ps)); continue
    p=ps[lv]; det=p['detection']
    won=n=0; U=[];W=[]; wonS=nS=0; bestU=bestW=0; ch=None
    for m,conds in det.items():
        for c,bym in conds.items():
            if not c.endswith('_unweighted'): continue
            stem=c[:-len('_unweighted')]
            for wc in [x for x in conds if x.startswith(stem+'_imp') and x not in d.get('duplicate_conditions',{}).get(m,{})]:
                for model,keys in bym.items():
                    for k,x in keys.items():
                        u=x.get('detection_average_precision'); y=conds[wc].get(model,{}).get(k,{}).get('detection_average_precision')
                        if u is None or y is None: continue
                        ch=x['detection_n_corrupted']/x['detection_n_training']
                        n+=1; won+=y>u; U.append(u); W.append(y); bestU=max(bestU,u); bestW=max(bestW,y)
                        if stem.startswith('base'): nS+=1; wonS+=y>u
    print(f"{ds:18s} {cond:18s} L{lv} folds{d['folds_run']} draws{p['n_draws']} models{d['models']} chance {ch:.4f} | scaled>unscaled {won}/{n} (single reps {wonS}/{nS}) | mean AP unscaled {mean(U):.4f} scaled {mean(W):.4f} | best unscaled {bestU:.4f} scaled {bestW:.4f}")
