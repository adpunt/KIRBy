import json,glob,os,re
D='/Volumes/seagate/kirby_results_2026-10-09/noise/n23_uncertainty_cleaning/'
main={'censoring':'0.25','outlier_p05':'1','shift_only_p10':'1','sparse_dprime_up_p10':'1'}
for f in sorted(glob.glob(D+'*__*.json')):
    b=os.path.basename(f)
    if 'fingerprint' in b: continue
    d=json.load(open(f)); det=d.get('detection') or {}
    if not det: continue
    ds,cond=b[:-5].split('__')
    defined=[s for s,v in d['sources_defined_under_this_condition'].items() if v and s!='activity_cliff_aware']
    lvl=main.get(cond, list(det)[-1])
    if lvl not in det: lvl=list(det)[-1]
    wins=0;n=0;best=[]
    for sp,bb in det[lvl].items():
        for m,c in bb.items():
            ca=c['activity_cliff_aware']['detection_average_precision']
            bm=max((c[s]['detection_average_precision'],s) for s in defined)
            n+=1; wins+= bm[0]>ca
    print(ds,cond,'lvl',lvl,'defined',defined,'wins %d/%d'%(wins,n), 'pass' if wins*2>=n else 'fail')
