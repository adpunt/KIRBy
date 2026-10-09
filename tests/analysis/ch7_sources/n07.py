import json,glob,os,re
D='/Volumes/seagate/kirby_results_2026-10-09/noise/n07_remediation'
ok=('qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins')
for f in sorted(glob.glob(D+'/*.json')):
    b=os.path.basename(f)[:-5]
    if re.search('tcga|annitia|nhanes|pam50',b): continue
    ds=b.split('__')[0]
    if ds not in ok: continue
    j=json.load(open(f))
    if 'by_level' not in j: print(b,'NO by_level',list(j)[:6]);continue
    print('##',b,'levels',j.get('levels'),'folds_run',j.get('folds_run'),'n_folds',j.get('n_folds'),'draws',j.get('draws_per_nonzero_level'))
    for lv,L in j['by_level'].items():
        if lv=='0': continue
        dd=L.get('detection_by_detector') or {}
        for dn,d in dd.items():
            if d: print('  lv',lv,'det_by_detector',dn,'AP=%.4f'%d['detection_average_precision'],'chance=%.4f'%(d['detection_n_corrupted']/d['detection_n_training']),'passes',d['detection_n_passes'])
        ds_=L.get('detection_by_similarity') or {}
        for dn,bs in ds_.items():
            for sn,bk in bs.items():
                for k,d in bk.items():
                    if d: print('  lv',lv,'sim',dn,sn,k,'AP=%.4f'%d['detection_average_precision'],'chance=%.4f'%(d['detection_n_corrupted']/d['detection_n_training']),'passes',d['detection_n_passes'])
        print('  lv',lv,'tally',L.get('detection_tally'))
