import json,sys,statistics as st,collections
for f in sys.argv[1:]:
    units=[json.loads(l) for l in open(f)]
    units=[u for u in units if u['ckpt_key'][0]=='unit']
    keys=sorted(set((u['ckpt_key'][1],u['ckpt_key'][2],u['ckpt_key'][3]) for u in units))
    print('##',f.split('/')[-3],f.split('/')[-1],'units(fold,level,draw):',keys)
    bylv=collections.defaultdict(lambda:{'cv':[],'sim':collections.defaultdict(list)})
    for u in units:
        fo,lv,dr=u['ckpt_key'][1:4]
        if lv==0: continue
        v=u['value']
        for m,bs in (v.get('detection_by_detector') or {}).items():
            for sp,bd in bs.items():
                d=bd.get('cross_val')
                if d: bylv[(fo,lv)]['cv'].append(d)
        for dn,bs in (v.get('sim_detection') or {}).items():
            for sn,bk in bs.items():
                for k,rows in bk.items():
                    bylv[(fo,lv)]['sim'][(sn,k)].extend(rows)
    for key,B in sorted(bylv.items()):
        cv=B['cv']
        if not cv: print(' ',key,'no cross_val grading');continue
        ap=st.mean(d['detection_average_precision'] for d in cv)
        ch=st.mean(d['detection_n_corrupted']/d['detection_n_training'] for d in cv)
        sims=sorted((st.mean(r['detection_average_precision'] for r in rows),sk) for sk,rows in B['sim'].items() if rows)
        if sims:
            print(f"  fold{key[0]} lv{key[1]} chance={ch:.4f} resid={ap:.4f} passes={len(cv)} | cliff n={len(sims)} min={sims[0][0]:.4f} med={st.median([s[0] for s in sims]):.4f} max={sims[-1][0]:.4f} {sims[-1][1]} >resid={sum(s[0]>ap for s in sims)}")
        else: print(f"  fold{key[0]} lv{key[1]} chance={ch:.4f} resid={ap:.4f} passes={len(cv)} | no cliff grading")
