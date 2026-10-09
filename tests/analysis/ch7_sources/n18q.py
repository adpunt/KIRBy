import json,sys
ds,model,cl=sys.argv[1:4]
f=f'/Volumes/seagate/kirby_results_2026-10-09/noise/n18_clean_the_real_labels/{ds}.json'
d=json.load(open(f))
print(f, d['metric'], 'folds_run',d['folds_run'])
for sp,v in d['per_space'].items():
  for det,w in v['per_detector'].items():
    for slot,x in w.items():
      m=x['per_model'].get(model)
      if not m or cl not in m: continue
      c=m[cl]
      sh=v.get('shuffled_control',{}).get(det,{}).get(slot,{}).get(model,{}).get(cl,{})
      shs=' '.join(f"s{k}={sh[k]['shuffled_mean']:.4f}" for k in sorted(sh)) if sh else '-'
      print(f"{sp:17s} {det:18s} slot={slot} setting={c['setting']} nc={m['no_cleaning_mean']:.4f} -> {c['cleaned_mean']:.4f} won/lost={c['folds_won']}/{c['folds_lost']} shuffled: {shs}")
