import json,sys
base='/Volumes/seagate/kirby_results_2026-10-09/noise/n30_censored_likelihood/'
arms=['clean','clipped','constant_lift','truncated_normal_mean','censored_em','drop_clipped','per_molecule_lift','per_molecule_lift_shuffled/seed0','per_molecule_lift_shuffled/seed1','per_molecule_lift_shuffled/seed2']
short=['clean','clip','const','TNM','EM','drop','pml','shf0','shf1','shf2']
for ds in ['logd','herg_ki','caco2','qm9','buchwald_hartwig','flab_thermo_tresanco','ddg_combined','ginkgo_colloidal']:
    d=json.load(open(base+ds+'.json'))
    for L in sys.argv[1:]:
        if L not in d['per_level']: continue
        print(f'## {ds} level {L} metric {d["metric"]} lower={d["lower_is_better"]} folds_run={d["folds_run"]}')
        print('config model '+' '.join(short))
        for c,v in d['per_level'][L].items():
            for m in d['models']:
                vals=[]
                for a in arms:
                    x=v.get(a,{}).get(m,{})
                    vals.append(' '.join(f'{y:.4f}' for y in x.values()) if x else 'NA')
                keys=list(v.get('clean',{}).get(m,{}).keys())
                print(c,m,' '.join(vals),keys)
