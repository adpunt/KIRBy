import json,glob,sys
for f in sorted(glob.glob('*.json')):
    if any(p in f for p in ['tcga','annitia','nhanes','pam50']) or f.startswith('herg__'): continue
    d=json.load(open(f)); print('==',f,d['metric'],'folds',d['folds_run'],'levels',d['noise_levels'],'draws',d['noise_draws'])
    for lv,s in d['summary']['per_level'].items():
        if lv=='0': continue
        for sp in ['base_1','base_2','hybrid','hybrid__general2']:
            if sp not in s: continue
            for m,pm in s[sp]['per_model'].items():
                a=pm['per_arm']; htb=a.get('huber_teacher_blend',{})
                cl=a.get('clean',{})
                print(f" L{lv} {sp:17s} {m:4s} noisy {pm['noisy_mean']:.4f} clean {htb.get('clean_mean',float('nan')):.4f} HTB {htb.get('arm_mean',float('nan')):.4f} won {htb.get('vs_noisy',{}).get('passes_won')}/{htb.get('vs_noisy',{}).get('n_passes_paired')}")
            for sw,v in s[sp].get('model_swaps',{}).items():
                if sw.startswith('dnn'):
                    print(f"   {sw} noisy {v['noisy_mean']:.4f} arm {v['arm_mean']:.4f} clean {v['clean_mean']:.4f} won {v['vs_parent_on_noisy']['passes_won']}/{v['vs_parent_on_noisy']['n_passes_paired']}")
