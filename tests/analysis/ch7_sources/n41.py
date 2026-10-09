import json,glob,sys,numpy as np
for f in sorted(glob.glob('*__*.json')):
    if any(x in f for x in ['herg__','gaussian','grouped']): continue
    d=json.load(open(f))
    if 'detection' not in d: print(f,'no detection', d.get('skipped')); continue
    print('==',f,'levels',d['noise_levels'],'folds_run',d.get('folds_run'),'draws',d.get('noise_draws'),'models',d.get('models'))
    for lvl,v in d['detection'].items():
        aps=[];aus=[]
        for sp,dd in v.items():
            r=dd.get('rescaled_residual')
            if r: aps.append((sp,round(r['detection_average_precision'],3),round(r['detection_auroc'],3))); ch=r['detection_n_corrupted']/r['detection_n_training']; npass=r['detection_n_passes']
        print('  L',lvl,'chance %.3f passes %d'%(ch,npass), 'resc_resid AP range %.3f-%.3f AUROC %.3f-%.3f'%(min(a[1] for a in aps),max(a[1] for a in aps),min(a[2] for a in aps),max(a[2] for a in aps)))
    for lvl,v in d['bh'].items():
        # aggregate across spaces
        agg={}
        for sp,byk in v.items():
            for k,e in byk.items():
                if not any(k.endswith(x) for x in ['q0.05','q0.1']): continue
                m=e['mean']; agg.setdefault(k,[]).append((m['molecules_flagged'],m['precision'],m['fewest_flags_bh_can_make']))
        for k,l in sorted(agg.items()):
            fl=[x[0] for x in l]; pr=[x[1] for x in l if x[1] is not None]
            print('   BH',lvl,k,'flagged %.0f-%.0f'%(min(fl),max(fl)),'precision',('%.2f-%.2f'%(min(pr),max(pr)) if pr else 'none'),'floor',l[0][2])
    s=d['summary']['per_level']
    for lvl,byrep in s.items():
        wins={}
        for rep,bym in byrep.items():
            for m,e in bym.items():
                if not isinstance(e,dict) or 'per_arm' not in e: continue
                for arm,a in e['per_arm'].items():
                    if '/seed' in arm or arm=='clean': 
                        if arm!='clean': continue
                    if not any(q in arm for q in ['q0.05','q0.1']) and arm!='clean': continue
                    if 'q0.2' in arm: continue
                    vn=a['vs_no_cleaning']
                    w=wins.setdefault(arm,[0,0,0,[]]); w[0]+=vn['passes_won'];w[1]+=vn['passes_lost'];w[2]+=vn['passes_tied'];w[3].append(vn['mean_gain'])
        for arm,w in sorted(wins.items()):
            print('   ARM',lvl,arm,'won/lost/tied %d/%d/%d'%tuple(w[:3]),'mean_gain range %.4f..%.4f'%(min(w[3]),max(w[3])), 'n cfg',len(w[3]))
