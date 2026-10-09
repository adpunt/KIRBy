import json,glob,os,re,statistics as st
D='/Volumes/seagate/kirby_results_2026-10-09/noise/n07_remediation'
ok=('qm9','caco2','herg_ki','logd','herg','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_titer','ginkgo_pr_cho','ginkgo_colloidal','ginkgo_acsins')
for f in sorted(glob.glob(D+'/*.json')):
    b=os.path.basename(f)[:-5]
    if re.search('tcga|annitia|nhanes|pam50',b): continue
    if b.split('__')[0] not in ok: continue
    j=json.load(open(f))
    if 'by_level' not in j: print(b,'SKIPPED file:',j.get('reason','')[:80]);continue
    for lv,L in j['by_level'].items():
        if lv=='0': continue
        cv=(L.get('detection_by_detector') or {}).get('cross_val')
        sims=[]
        for dn,bs in (L.get('detection_by_similarity') or {}).items():
            for sn,bk in bs.items():
                for k,d in bk.items():
                    if d: sims.append((d['detection_average_precision'],sn,k,d['detection_n_passes'],d['detection_n_corrupted']/d['detection_n_training']))
        if not cv and not sims: print(b,'lv',lv,'no grading; tally',L.get('detection_tally'));continue
        ch=cv['detection_n_corrupted']/cv['detection_n_training'] if cv else sims[0][4]
        s=sorted(sims)
        cvap=cv['detection_average_precision'] if cv else float('nan')
        nab=sum(1 for x in sims if x[0]>cvap)
        print(f"{b} lv{lv} folds{j['folds_run']} draws{j['draws_per_nonzero_level']} chance={ch:.4f} resid={cvap:.4f} passes={cv and cv['detection_n_passes']} | cliff n={len(s)} min={s[0][0]:.4f} med={st.median([x[0] for x in s]):.4f} max={s[-1][0]:.4f} [{s[-1][1]} k{s[-1][2]}] settings>resid={nab}")
