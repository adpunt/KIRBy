import json,glob,os,re
R='/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n21_neighbour_distance'
DS=['qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins']
MAIN={'outlier_p05':'1','shift_only_p10':'1','censoring':'0.25','sparse_dprime_up_p10':'1'}
fam_map={'plain':'plain','importance_weighted':'impw','forest_proximity':'forest','hybrid':'hybrid','hybrid_similarity':'hybrid'}
for f in sorted(glob.glob(R+'/*__*.json')):
    b=os.path.basename(f)[:-5]; ds,cond=b.split('__',1)
    if ds not in DS: continue
    d=json.load(open(f))
    lv=MAIN.get(cond,'0.1')
    pl=d['per_level']
    if lv not in pl:
        cands=[k for k in pl if k!='0']; print(b,'levels',list(pl)); lv=cands[0] if cands else None
        if lv is None: continue
    p=pl[lv]; ch=p['molecules_corrupted_mean']/p['molecules_in_training_fold_mean']
    best={}; bestmean={}; hyb_nosim={}
    for sp,v in p['per_space'].items():
        fam=fam_map[d['spaces'][sp]['family']]
        for det,x in v['detection_per_detector'].items():
            ap=x['average_precision']['mean']
            if ap is None: continue
            if ap>best.get(fam,(-1,))[0]: best[fam]=(ap,sp,det,x['n_passes'])
        m=v.get('detection_average_precision_mean')
        if m is not None and m>bestmean.get(fam,(-1,))[0]: bestmean[fam]=(m,sp)
    w=max(best,key=lambda k:best[k][0]); wm=max(bestmean,key=lambda k:bestmean[k][0])
    print(f"{ds:18s} {cond:15s} L{lv} folds{d['folds_run']} models{d['models']} chance {ch:.4f} | max-setting: "+' '.join(f"{k}={best[k][0]:.4f}" for k in ['plain','impw','forest','hybrid'] if k in best)+f" WIN={w} ({best[w][1]} {best[w][2]} n={best[w][3]}) | mean-over-settings: "+' '.join(f"{k}={bestmean[k][0]:.4f}" for k in ['plain','impw','forest','hybrid'] if k in bestmean)+f" WIN={wm} ({bestmean[wm][1]})")
