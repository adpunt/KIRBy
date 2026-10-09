import json,glob,os,re
import numpy as np
D='/Volumes/seagate/kirby_results_2026-10-09/noise/n23_uncertainty_cleaning/'
for f in sorted(glob.glob(D+'*__*.json')):
    b=os.path.basename(f)
    if re.search('tcga|annitia|nhanes|pam50',b): continue
    d=json.load(open(f))
    det=d.get('detection') or {}
    if not det: print(b,'no detection'); continue
    print('==',b,'folds_run',d.get('folds_run'),'levels',d.get('noise_levels'),'status',d.get('result_status'))
    for lvl,a in det.items():
        bysrc={}
        n=None
        for sp,bb in a.items():
            for m,c in bb.items():
                for src,v in c.items():
                    ap=v.get('detection_average_precision')
                    if ap is None: continue
                    bysrc.setdefault(src,[]).append(ap)
                    n=(v['detection_n_corrupted'],v['detection_n_training'],v['detection_n_passes'])
        if not bysrc: continue
        ch=n[0]/n[1]
        print('  lvl',lvl,'chance %.4f'%ch,'passes',n[2],' '.join('%s=%.4f[%.4f-%.4f]'%(s,np.mean(x),min(x),max(x)) for s,x in bysrc.items()))
