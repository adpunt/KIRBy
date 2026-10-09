import json,glob,os,re
D='/Volumes/seagate/kirby_results_2026-10-09/noise/n40_model_side_regularisation/'
bad=re.compile('tcga|annitia|nhanes|pam50',re.I)
for f in sorted(glob.glob(D+'*__*.json')):
    if bad.search(f): continue
    d=json.load(open(f)); lib=d['lower_is_better']
    if 'summary' not in d: print('==',os.path.basename(f),'NO SUMMARY',d.get('result_status'),str(d.get('reason') or d.get('skip_reason') or '')[:150]); continue
    print('==',os.path.basename(f),d.get('metric'),'folds',d.get('folds_run'),'levels',d.get('noise_levels'),'draws',d.get('noise_draws'),'models',d.get('models'), d.get('result_status'))
    for lvl,byspace in d['summary']['per_level'].items():
        for m in d['models']:
            out={}
            for grp,spaces in (('single',('base_1','base_2')),('hybrid',('hybrid','hybrid__general2'))):
                vals={}
                for sp in spaces:
                    r=byspace.get(sp,{}).get('per_model',{}).get(m)
                    if not r: continue
                    fx=r['fixed_mean']; vals.setdefault('fixed',[]).append((fx,sp))
                    for arm,e in r['per_arm'].items():
                        g=e['vs_fixed']['mean_gain']
                        if g is None: continue
                        am=fx-g if lib else fx+g
                        vals.setdefault(arm,[]).append((am,sp))
                    st=byspace[sp].get('stack',{}).get(m)
                    if st and st.get('mean_gain') is not None:
                        g=st['mean_gain']; vals.setdefault('stack',[]).append((fx-g if lib else fx+g,sp))
                best={a:(min(v) if lib else max(v)) for a,v in vals.items()}
                out[grp]=best
            if not out.get('single'): continue
            line=f'  L{lvl} {m}: '
            for grp in ('single','hybrid'):
                b=out.get(grp,{})
                if 'fixed' not in b: continue
                fx=b['fixed'][0]
                parts=[f"{fx:.4f}({b['fixed'][1][-3:]})"]
                for a in ('stack','seed_avg','retuned','tuned_clean_reused','clean'):
                    if a in b:
                        better = (b[a][0]<fx) if lib else (b[a][0]>fx)
                        parts.append(f"{ {'stack':'st','seed_avg':'sa','retuned':'rt','tuned_clean_reused':'tc','clean':'cl'}[a]} {b[a][0]:.4f}{'+' if better else '-'}")
                line+=f"[{grp}: "+', '.join(parts)+'] '
            print(line)
