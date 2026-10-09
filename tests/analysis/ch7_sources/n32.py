import json,os,re,sys
import numpy as np
D='/Volumes/seagate/kirby_results_2026-10-09/noise/n32_iterative_and_dual_selection/'
names=['buchwald_hartwig','caco2','ddg_combined','flab_thermo_tresanco','ginkgo_colloidal','ginkgo_titer','herg_ki','logd','qm9']
conds=['censoring','outlier_p05','shift_only_p10','sparse_dprime_up_p10']
def mean(rows,k):
    v=[r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
    return np.mean(v) if v else np.nan
def load(ds,c):
    f=D+f'{ds}__{c}.json'
    if os.path.exists(f) and os.path.getsize(f)>2000:
        d=json.load(open(f)); 
        if 'detection' in d: return 'final',d['detection'],d.get('folds_run'),d.get('noise_draws'),d
    f=D+'_checkpoints/'+f'{ds}__{c}.json'
    if os.path.exists(f) and os.path.getsize(f)>200:
        d=json.load(open(f)); da=d['state']['det_acc']
        det={l:{sk:{dk:{'detection_average_precision':mean(rs,'detection_average_precision'),'detection_auroc':mean(rs,'detection_auroc'),
              'detection_n_corrupted':mean(rs,'detection_n_corrupted'),'detection_n_training':mean(rs,'detection_n_training'),'detection_n_passes':len(rs)} for dk,rs in b.items()} for sk,b in a.items()} for l,a in da.items()}
        return 'checkpoint',det,d['passes_done'],None,d
    return None,None,None,None,None
fam=lambda k:k.split('@')[0]
for ds in names:
    for c in conds:
        src,det,folds,draws,raw=load(ds,c)
        if not src: continue
        print(f'== {ds} {c} [{src}] folds/passes={folds} draws={draws}')
        if src=='final': print('   errors:', {k for k in (raw.get('summary') or {}) } if False else '', str(raw.get('models_unavailable'))[:150])
        for lvl,a in det.items():
            res=[];best={};ch=None;wins=0;n=0;wa=0
            for sk,b in a.items():
                for dk,v in b.items():
                    if ch is None: ch=v['detection_n_corrupted']/v['detection_n_training']; np_=v['detection_n_passes']
                    f=fam(dk)
                    best.setdefault(f,[]).append((v['detection_average_precision'],v.get('detection_auroc'),sk+'|'+dk))
                if 'oof_residual' in b:
                    r=b['oof_residual']['detection_average_precision']; ra=b['oof_residual'].get('detection_auroc')
                    others=[v['detection_average_precision'] for dk,v in b.items() if dk!='oof_residual']
                    othersa=[v.get('detection_auroc') for dk,v in b.items() if dk!='oof_residual']
                    n+=1; wins+= max(others)>r if others else 0; wa+= max(othersa)>ra if othersa else 0
            line=f'  lvl {lvl} chance {ch:.4f} passes {np_} | resid-config pairs beaten by another detector AP {wins}/{n} AUROC {wa}/{n}'
            print(line)
            for f,l in best.items():
                aps=[x[0] for x in l]; aus=[x[1] for x in l if x[1] is not None]
                mx=max(l,key=lambda x:x[0])
                print(f'     {f:26s} n={len(l):3d} AP mean {np.nanmean(aps):.4f} range {np.nanmin(aps):.4f}-{np.nanmax(aps):.4f} AUROC {np.nanmin(aus):.4f}-{np.nanmax(aus):.4f} best@{mx[2]}')
