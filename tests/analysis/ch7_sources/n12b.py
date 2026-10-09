import json,glob,os
R='/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n12_importance_weighted_distance'
DS=['caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_colloidal','ginkgo_acsins']
GR={'outlier_p05':'1.0','shift_only_p10':'1.0','censoring':'0.25','sparse_dprime_up_p10':'1.0','binary_asymmetric':'0.1','class_dependent':'0.1','class_imbalance':'0.1','confusion_directed':'0.1','instance_noise':'0.1','uniform':'0.1'}
HYB=['hybrid','hybrid__general2','hybrid__distance1','hybrid_similarity']
for ds in DS:
  out=[];passes=0;kinds=0
  for f in sorted(glob.glob(f'{R}/{ds}__*.json')):
    cond=os.path.basename(f)[:-5].split('__',1)[1]
    if cond not in GR: continue
    d=json.load(open(f)); p=d['per_sigma'].get(GR[cond])
    if not p: continue
    det=p['detection']; imps=list(det); w=n=0
    for sp in ['base_1','base_2']+HYB:
      for m in d['models']:
        for key,ru in det[imps[0]].get(sp+'_unweighted',{}).get(m,{}).items():
          for imp in imps:
            c=sp+'_imp_weighted' if sp.startswith('base') else sp+'_imp_own'
            rs=det[imp].get(c,{}).get(m,{}).get(key)
            if rs is None or ru.get('detection_average_precision') is None: continue
            n+=1; w+=rs['detection_average_precision']>ru['detection_average_precision']
    kinds+=1; passes+= (w>=n/2); out.append(f"{cond} {w}/{n}")
  print(ds, 'PASS' if passes>=kinds/2 else 'fail', f"{passes}/{kinds} kinds |", '; '.join(out))
